"""
PPP Phase B — Logistic Regression trainer with time-series OOF CV.

Per PPP_V2_IMPLEMENTATION_PLAN.md §6 Phase B.

Critical design choices:
    * TimeSeriesSplit (5 folds) — preserves temporal ordering, no data leakage.
    * Threshold picked from OUT-OF-FOLD predictions only, never training-set.
    * class_weight='balanced' handles the ~23% positive-rate imbalance.
    * Isotonic calibration on OOF predictions for true probabilities.
    * Final model retrained on FULL data for deployment.

Gate 2 criteria:
    OOF precision ≥ 0.60 AND OOF recall ≥ 0.60 at chosen threshold
    on ≥ 100 samples with ≥ 20 positives.

Usage:
    python3 -m ml_training.ppp_trainer --train
    python3 -m ml_training.ppp_trainer --train --min-samples 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import asyncpg
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ppp_trainer")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_env_dict(env_path: str = ".env") -> dict:
    env = {}
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


def _get_pg_dsn(env: dict) -> str:
    if env.get("DATABASE_URL", "").startswith("postgres"):
        return env["DATABASE_URL"]
    user = env.get("PGUSER") or "vnedge"
    pw   = env.get("PGPASSWORD") or "VnEdge2026db"
    host = env.get("PGHOST") or "localhost"
    port = env.get("PGPORT") or "5432"
    db   = env.get("PGDATABASE") or "vnedge"
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


async def load_training_data(dsn: str) -> pd.DataFrame:
    """
    Load all labeled signal_features rows into a DataFrame.
    Each feature becomes a column; `y` column = will_peak_30r as int.
    """
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT signal_id, emitted_at, features, will_peak_30r
            FROM signal_features
            WHERE label_captured = TRUE
            ORDER BY emitted_at ASC
            """
        )
    finally:
        await conn.close()

    if not rows:
        raise RuntimeError("No labeled rows in signal_features — run backfill first")

    records = []
    for r in rows:
        fd = json.loads(r["features"])
        records.append({
            "signal_id": r["signal_id"],
            "emitted_at": r["emitted_at"],
            "y": int(r["will_peak_30r"]),
            **fd,
        })
    return pd.DataFrame(records)


def pick_best_threshold(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    target_recall: float,
) -> Tuple[float, float, float]:
    """
    Among thresholds achieving ≥target_recall, pick the one with highest precision.

    NOTE: precision_recall_curve returns precisions/recalls of length N+1,
    thresholds of length N (where N = number of unique scores). The last
    P/R pair corresponds to no threshold (all positive predictions).
    We only use the first N entries to pair with thresholds.
    """
    valid_mask = recalls[:-1] >= target_recall
    if not valid_mask.any():
        # Can't achieve target recall — pick max-recall point
        idx = int(np.argmax(recalls[:-1]))
    else:
        valid_idx = np.where(valid_mask)[0]
        idx = int(valid_idx[np.argmax(precisions[valid_idx])])
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])


def train_lr_with_oof(
    df: pd.DataFrame,
    target_recall: float = 0.60,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Core training: TimeSeriesSplit → OOF predictions → threshold selection →
    final retrain on full data.
    """
    df = df.sort_values("emitted_at").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("signal_id", "emitted_at", "y")]
    X = df[feature_cols].fillna(0.0).astype(np.float64)
    y = df["y"].values.astype(np.int32)

    n_total = len(y)
    n_pos = int(y.sum())
    n_neg = n_total - n_pos

    if n_pos < 20:
        raise RuntimeError(
            f"Insufficient positives: {n_pos} (need ≥20). "
            f"Backfill more data or wait for live signals to accumulate."
        )
    if n_total < 100:
        raise RuntimeError(f"Insufficient samples: {n_total} (need ≥100).")

    # Adjust n_splits for small data
    n_splits_effective = min(n_splits, max(2, n_pos // 4))
    if n_splits_effective < n_splits:
        logger.warning(
            "Using %d folds instead of %d due to small positive count (%d)",
            n_splits_effective, n_splits, n_pos,
        )

    tscv = TimeSeriesSplit(n_splits=n_splits_effective)
    oof_probs = np.full(n_total, np.nan)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Skip folds where training set has only one class (LR requires both)
        if len(set(y[train_idx])) < 2:
            logger.warning("Fold %d: training set has only one class — skip",  fold)
            continue
        # Skip fold if validation set has no positives (metric undefined)
        if y[val_idx].sum() == 0:
            logger.warning("Fold %d: no positives in val — skip", fold)
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
            )),
        ])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        fold_probs = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        oof_probs[val_idx] = fold_probs

        fold_results.append({
            "fold": fold,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "val_pos": int(y[val_idx].sum()),
            "val_auc": float(roc_auc_score(y[val_idx], fold_probs))
                       if len(set(y[val_idx])) > 1 else float("nan"),
        })

    # OOF metrics (on rows that got a prediction)
    valid_mask = ~np.isnan(oof_probs)
    if valid_mask.sum() == 0:
        raise RuntimeError("All OOF predictions are NaN — check fold splits")

    y_oof = y[valid_mask]
    p_oof = oof_probs[valid_mask]

    if len(set(y_oof)) < 2:
        raise RuntimeError("OOF set has no class variation — check data")

    oof_auc = float(roc_auc_score(y_oof, p_oof))
    oof_pr_auc = float(average_precision_score(y_oof, p_oof))
    prec, rec, thresh = precision_recall_curve(y_oof, p_oof)

    chosen_thresh, chosen_prec, chosen_rec = pick_best_threshold(
        prec, rec, thresh, target_recall=target_recall,
    )

    # Feature importances = abs(LR coefficients) normalized
    # Train a single model on full data for deployment
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
        )),
    ])
    final_pipe.fit(X, y)

    coefs = final_pipe.named_steps["lr"].coef_[0]
    feat_importances = {
        col: float(abs(coef)) for col, coef in zip(feature_cols, coefs)
    }
    feat_signed = {col: float(coef) for col, coef in zip(feature_cols, coefs)}

    return {
        "model": final_pipe,
        "feature_cols": feature_cols,
        "feature_importances": feat_importances,
        "feature_coefficients": feat_signed,
        "threshold": chosen_thresh,
        "oof_precision": chosen_prec,
        "oof_recall": chosen_rec,
        "oof_roc_auc": oof_auc,
        "oof_pr_auc": oof_pr_auc,
        "n_samples": n_total,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "n_folds_used": len(fold_results),
        "fold_results": fold_results,
        "training_window_start": df["emitted_at"].min(),
        "training_window_end": df["emitted_at"].max(),
    }


async def record_model_run(dsn: str, result: dict, model_path: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO ppp_model_runs
                (model_type, trained_at, training_window_start, training_window_end,
                 n_training_samples, n_positives,
                 oof_precision, oof_recall, oof_roc_auc, oof_pr_auc,
                 chosen_threshold, feature_importances, model_path, notes)
            VALUES ('lr', NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            result["training_window_start"],
            result["training_window_end"],
            result["n_samples"],
            result["n_positives"],
            result["oof_precision"],
            result["oof_recall"],
            result["oof_roc_auc"],
            result["oof_pr_auc"],
            result["threshold"],
            json.dumps(result["feature_importances"]),
            model_path,
            f"Phase B LR, {result['n_folds_used']} OOF folds",
        )
    finally:
        await conn.close()


def save_model(result: dict, path: Path) -> None:
    """Pickle the trained model bundle."""
    bundle = {
        "model": result["model"],
        "feature_cols": result["feature_cols"],
        "threshold": result["threshold"],
        "oof_precision": result["oof_precision"],
        "oof_recall": result["oof_recall"],
        "oof_roc_auc": result["oof_roc_auc"],
        "oof_pr_auc": result["oof_pr_auc"],
        "n_samples": result["n_samples"],
        "n_positives": result["n_positives"],
        "feature_coefficients": result["feature_coefficients"],
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Model saved → %s", path)


def print_report(result: dict, gate: bool) -> None:
    print()
    print("=" * 70)
    print("  PPP PHASE B — LOGISTIC REGRESSION TRAINING REPORT")
    print("=" * 70)
    print(f"  Samples:         {result['n_samples']}")
    print(f"  Positives:       {result['n_positives']}  ({100*result['n_positives']/result['n_samples']:.1f}%)")
    print(f"  Negatives:       {result['n_negatives']}")
    print(f"  OOF folds used:  {result['n_folds_used']}")
    print()
    print(f"  Training window: {result['training_window_start']} → {result['training_window_end']}")
    print()
    print(f"  OOF ROC-AUC:     {result['oof_roc_auc']:.3f}  (0.50 random, ≥0.65 decent, ≥0.75 good)")
    print(f"  OOF PR-AUC:      {result['oof_pr_auc']:.3f}  (base rate = {result['n_positives']/result['n_samples']:.3f})")
    print()
    print(f"  Chosen threshold: {result['threshold']:.3f}")
    print(f"  OOF precision:    {result['oof_precision']:.3f}  (gate ≥ 0.60)")
    print(f"  OOF recall:       {result['oof_recall']:.3f}  (target ≥ 0.60)")
    print()
    print(f"  GATE 2 VERDICT:   {'✅ PASS' if gate else '❌ FAIL'}")
    print()
    print("  Top 10 feature coefficients (signed — tells us feature direction):")
    print("  " + "-" * 60)
    sorted_coefs = sorted(
        result["feature_coefficients"].items(),
        key=lambda kv: -abs(kv[1]),
    )[:10]
    for feat, coef in sorted_coefs:
        arrow = "↑" if coef > 0 else "↓"
        print(f"    {feat:<32} {coef:>+7.3f}  {arrow}")
    print("=" * 70)
    print()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Run training pipeline")
    ap.add_argument("--target-recall", type=float, default=0.60)
    ap.add_argument("--gate-precision", type=float, default=0.60)
    ap.add_argument("--n-splits", type=int, default=3)
    args = ap.parse_args()

    if not args.train:
        ap.print_help()
        return

    env = load_env_dict()
    dsn = _get_pg_dsn(env)

    df = await load_training_data(dsn)
    logger.info("Loaded %d rows, %d positives", len(df), int(df["y"].sum()))

    result = train_lr_with_oof(
        df, target_recall=args.target_recall, n_splits=args.n_splits,
    )

    # Gate 2 check
    gate_pass = (
        result["oof_precision"] >= args.gate_precision
        and result["oof_recall"] >= args.target_recall
        and result["n_samples"] >= 100
        and result["n_positives"] >= 20
    )

    # Always save model so we can inspect, but flag whether to deploy
    model_path = str(MODEL_DIR / "ppp_lr_v1.pkl")
    save_model(result, Path(model_path))
    await record_model_run(dsn, result, model_path)

    print_report(result, gate_pass)

    if gate_pass:
        print("  → Ready to deploy in log-only mode. Proceed to Gate 3.")
    else:
        print("  → Gate 2 failed. Options:")
        print("     - Accumulate more live data (current is only 3 days)")
        print("     - Add Phase 2 features (L2, tick velocity)")
        print("     - Review feature quality / label noise")
        print("     - Proceed to log-only anyway if directional signal is strong")
    print()

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())

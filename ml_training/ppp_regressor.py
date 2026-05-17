"""
Phase 5.15 — PPP Regressor: predicts peak_mfe_r as a continuous value.

Why regression instead of binary classification:
  - Binary loses information: a signal predicted to peak 0.45R vs 1.20R
    are both "winners" but very different value.
  - Regression output can be used for POSITION SIZING (Kelly-style),
    not just admit/reject.
  - Better calibration on small datasets — every trade contributes to
    learning, not just "did it cross threshold".
  - Threshold can be tuned post-hoc without retraining.

Model: gradient-boosted regression (LightGBM) with quantile loss for
robustness to outliers (e.g., one trade peaking at 3R shouldn't dominate).

Output: predicted_peak_r ∈ [0, 5] (clamped). Values <0 set to 0.

Decision use:
  - Admission gate: admit if predicted_peak_r >= peak_threshold (e.g. 0.30)
  - Position sizing: position_pct = clip(predicted_peak_r / 1.5, 0.3, 1.0)
    (small trades for low-confidence, full size for high-confidence)

Training pipeline:
  - Use same Phase 1 feature set as binary PPP (17 features)
  - Time-series split for OOS validation
  - Metric: Spearman rank correlation (we care about RANKING signals,
    not absolute peak prediction precision)
  - Bootstrap CI on rank correlation
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import asyncpg
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ppp_regressor")


def _load_env(path: str = ".env") -> None:
    try:
        with open(path) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v.strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _dsn() -> str:
    _load_env()
    if os.environ.get("DATABASE_URL", "").startswith("postgres"):
        return os.environ["DATABASE_URL"]
    return "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"


async def load_training_df() -> pd.DataFrame:
    """Pull labeled signals from signal_features."""
    conn = await asyncpg.connect(_dsn())
    try:
        rows = await conn.fetch("""
            SELECT signal_id, emitted_at, features, peak_mfe_r
            FROM signal_features
            WHERE label_captured = TRUE
              AND peak_mfe_r IS NOT NULL
            ORDER BY emitted_at ASC
        """)
    finally:
        await conn.close()

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        features = json.loads(r["features"])
        records.append({
            "signal_id": r["signal_id"],
            "emitted_at": r["emitted_at"],
            "y": float(r["peak_mfe_r"]),
            **features,
        })
    return pd.DataFrame(records)


def train_regressor(df: pd.DataFrame) -> Optional[Dict]:
    """
    Train LightGBM regressor on peak_mfe_r.
    Returns model bundle, or None if insufficient data.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed — pip install lightgbm")
        return None

    n = len(df)
    if n < 100:
        logger.warning("Need 100+ samples for regressor (have %d)", n)
        return None

    df = df.sort_values("emitted_at").reset_index(drop=True)
    feature_cols = [
        c for c in df.columns
        if c not in ("signal_id", "emitted_at", "y")
    ]
    # Clamp target — peaks above 3R are outliers and shouldn't dominate
    y = np.clip(df["y"].values.astype(float), 0.0, 3.0)
    X = df[feature_cols].fillna(0.0)

    # Time-series CV with OOF predictions
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.full(n, np.nan)

    params = dict(
        objective="regression",
        # Quantile loss is robust to outliers
        # objective="quantile", alpha=0.5, → median regression
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        min_data_in_leaf=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
    )

    feat_imps = []
    for fold, (tr, va) in enumerate(tscv.split(X)):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        oof_preds[va] = m.predict(X.iloc[va])
        feat_imps.append(dict(zip(feature_cols, m.feature_importances_)))

    valid = ~np.isnan(oof_preds)
    y_oof = y[valid]
    p_oof = oof_preds[valid]

    # Spearman rank correlation — what we actually care about
    rho, p_value = spearmanr(p_oof, y_oof)

    # Bootstrap CI on rank correlation
    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(500):
        idx = rng.integers(0, len(p_oof), size=len(p_oof))
        try:
            r, _ = spearmanr(p_oof[idx], y_oof[idx])
            if not np.isnan(r):
                boot_rhos.append(r)
        except Exception:
            pass
    if boot_rhos:
        boot = np.array(boot_rhos)
        rho_lo = float(np.quantile(boot, 0.025))
        rho_hi = float(np.quantile(boot, 0.975))
    else:
        rho_lo, rho_hi = float("nan"), float("nan")

    # Pearson correlation as well (sensitive to outliers)
    if len(p_oof) > 1:
        pearson = float(np.corrcoef(p_oof, y_oof)[0, 1])
    else:
        pearson = 0.0

    # MAE / RMSE
    mae = float(np.mean(np.abs(p_oof - y_oof)))
    rmse = float(np.sqrt(np.mean((p_oof - y_oof) ** 2)))

    # Decile lift: top decile predicted should peak higher than bottom decile
    if len(p_oof) >= 20:
        order = np.argsort(p_oof)
        top_decile_y = y_oof[order[-len(order) // 10:]]
        bot_decile_y = y_oof[order[:len(order) // 10]]
        decile_lift = float(top_decile_y.mean() - bot_decile_y.mean())
    else:
        decile_lift = 0.0

    # Final model on ALL data
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)

    avg_imp = {c: float(np.mean([fi[c] for fi in feat_imps])) for c in feature_cols}

    return {
        "model_type": "lgb_regressor",
        "model": final_model,
        "feature_cols": feature_cols,
        "n_samples": n,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        # Honest metrics with CI
        "oof_spearman": float(rho),
        "oof_spearman_ci_low": rho_lo,
        "oof_spearman_ci_high": rho_hi,
        "oof_spearman_p_value": float(p_value),
        "oof_pearson": pearson,
        "oof_mae_r": mae,
        "oof_rmse_r": rmse,
        "decile_lift_r": decile_lift,
        "feature_importances": avg_imp,
        # Default decision threshold (continuous output → binary admit)
        "default_admit_threshold_r": 0.20,
    }


def save_bundle(bundle: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Saved regressor → %s", path)


def format_report(bundle: Dict) -> str:
    if not bundle:
        return "(no bundle — training failed)"
    lines = [
        "",
        "=" * 60,
        "  PPP REGRESSOR — Phase 5.15",
        "=" * 60,
        f"  Training samples:        {bundle['n_samples']}",
        f"  Model:                   {bundle['model_type']}",
        f"  Trained at:              {bundle['trained_at']}",
        "",
        f"  Spearman rank corr:      {bundle['oof_spearman']:+.4f}  (p={bundle['oof_spearman_p_value']:.4f})",
        f"     95% CI:               [{bundle['oof_spearman_ci_low']:+.3f}, {bundle['oof_spearman_ci_high']:+.3f}]",
        f"  Pearson corr:            {bundle['oof_pearson']:+.4f}",
        f"  MAE:                     {bundle['oof_mae_r']:.4f} R",
        f"  RMSE:                    {bundle['oof_rmse_r']:.4f} R",
        f"  Decile lift:             {bundle['decile_lift_r']:+.4f} R (top10% peak avg − bot10% peak avg)",
        "",
        f"  Default admit threshold: {bundle['default_admit_threshold_r']:.2f} R predicted peak",
        "",
        "  Top features (by importance):",
    ]
    sorted_imps = sorted(
        bundle["feature_importances"].items(),
        key=lambda x: -x[1],
    )[:10]
    for name, imp in sorted_imps:
        lines.append(f"    {name:<35} {imp:>8.1f}")
    lines.append("=" * 60)
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "models" / "ppp_regressor_v1.pkl",
    )
    ap.add_argument("--print-only", action="store_true",
                    help="Train and report but don't save")
    args = ap.parse_args()

    df = await load_training_df()
    if df.empty:
        logger.error("No training data — exiting")
        return

    logger.info("Training regressor on %d labeled samples...", len(df))
    bundle = train_regressor(df)
    if bundle is None:
        return

    print(format_report(bundle))

    if not args.print_only:
        save_bundle(bundle, args.out)


if __name__ == "__main__":
    asyncio.run(main())

"""
PPP Phase A — hand-tuned heuristic scorer + Gate 1 evaluator.

Per PPP_V2_IMPLEMENTATION_PLAN.md §6 Phase A:
    Validate that Phase 1 features contain ANY discriminative signal before
    investing in ML training. If this scorer hits ≥55% precision at 60% recall
    on 300+ backfilled labels, proceed to Phase B (LR). Otherwise fix features
    first.

Usage (module):
    from ml_training.ppp_heuristic import score_heuristic, evaluate_gate1

Usage (CLI / Gate 1 check on backfilled data):
    python3 -m ml_training.ppp_heuristic --eval
    python3 -m ml_training.ppp_heuristic --eval --min-samples 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import asyncpg
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ppp_heuristic")


def score_heuristic(f: Dict[str, Any]) -> float:
    """
    Phase A hand-tuned scorer.

    Signal sources (weighted by trader intuition, validated empirically):
      + Momentum continuity (aligned bars, volume spike, thick body)
      + Favorable regime (breakout, trending)
      - Unfavorable regime (sideways = chop)
      + High grade + high ML setups
      - Over-extended from VWAP (mean-reversion risk)
      + Near VWAP (room to run)

    Returns probability-like score in [0.0, 1.0]. Neutral prior = 0.50.
    """
    score = 0.50  # neutral prior

    # -------- Momentum continuity --------
    last_3 = f.get("last_3bar_direction", 0)
    if last_3 >= 2:     score += 0.08   # 2+ of last 3 bars aligned
    elif last_3 <= -2:  score += 0.08   # for shorts, inverse works the same
    # (scorer is side-agnostic — features already encode direction)

    vol_ratio = f.get("volume_vs_20bar_avg", 1.0) or 1.0
    if vol_ratio > 1.8: score += 0.10   # strong volume spike
    elif vol_ratio > 1.3: score += 0.05

    body_ratio = f.get("bar_body_to_range_ratio", 0) or 0
    if body_ratio > 0.6: score += 0.08   # thick body = committed move
    elif body_ratio > 0.4: score += 0.03

    atr_norm = f.get("atr_normalized_range", 0) or 0
    if atr_norm > 1.5: score += 0.05     # expansion bar

    # -------- Positional / mean-reversion risk --------
    dvwap = abs(f.get("distance_from_vwap_bps", 0) or 0)
    if dvwap > 80:  score -= 0.10    # stretched >0.8% from VWAP = revert risk
    elif dvwap < 20: score += 0.05   # at VWAP = room to run

    dsma = abs(f.get("distance_from_20bar_avg_bps", 0) or 0)
    if dsma > 100:  score -= 0.05    # very stretched from 20-bar avg
    elif dsma < 30: score += 0.03

    # -------- Regime --------
    if f.get("is_breakout", 0) or f.get("is_trending_up", 0) or f.get("is_trending_down", 0):
        score += 0.06
    if f.get("is_sideways", 0):
        score -= 0.08
    if f.get("is_high_volatility", 0):
        score += 0.03  # double-edged but HV signals tend to peak deeper when they work

    # -------- Grade + ML --------
    g = f.get("grade_numeric", 0) or 0
    if g >= 4.0:  score += 0.05      # A+
    elif g >= 3.0: score += 0.02     # A

    ml = f.get("ml_probability", 0) or 0
    if ml > 0.90: score += 0.06
    elif ml > 0.80: score += 0.04
    elif ml > 0.70: score += 0.02

    conf = f.get("confidence", 0) or 0
    if conf >= 90: score += 0.02

    # -------- Scanner (some scanners historically better than others) --------
    # meme_burst and bb_squeeze tend to fire in developing-momentum conditions;
    # structure_bounce is more often at resistance → revert risk.
    if f.get("is_meme_burst", 0):    score += 0.03
    if f.get("is_bb_squeeze", 0):    score += 0.02
    # structure_bounce: neutral (it's the dominant scanner and mean-mixed)

    # -------- Clamp to [0, 1] --------
    return max(0.0, min(1.0, score))


def precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal PR curve implementation — avoids sklearn dependency for Phase A.
    Returns (precision, recall, thresholds) sorted by descending threshold.
    """
    n = len(y_true)
    # Sort by descending score
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    total_positives = y_true.sum()
    if total_positives == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])

    # Precision and recall at each possible threshold
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recalls = tp_cum / total_positives
    return precisions, recalls, s_sorted


def find_threshold_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float
) -> Tuple[float, float, float]:
    """
    Find threshold that achieves ≥target_recall with maximum precision.

    Returns (threshold, precision_at_threshold, recall_at_threshold).
    If no threshold achieves target_recall, returns best-achievable (warn).
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_score)

    # Among points where recall >= target, pick max precision
    valid = rec >= target_recall
    if not valid.any():
        # Fallback: pick the point with max recall
        idx = int(np.argmax(rec))
        logger.warning(
            "Target recall %.2f unreachable. Best recall = %.3f (precision=%.3f)",
            target_recall, rec[idx], prec[idx],
        )
        return float(thresh[idx]), float(prec[idx]), float(rec[idx])

    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(prec[valid_idx])]
    return float(thresh[best]), float(prec[best]), float(rec[best])


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


async def evaluate_gate1(
    min_samples: int = 300,
    target_recall: float = 0.60,
    gate_precision: float = 0.55,
) -> dict:
    """
    Run Phase A scorer on labeled signal_features rows and check Gate 1.

    Gate 1 passes if: ≥gate_precision precision at ≥target_recall recall
                      on ≥min_samples labeled examples.
    """
    env = load_env_dict()
    dsn = _get_pg_dsn(env)
    conn = await asyncpg.connect(dsn)

    try:
        rows = await conn.fetch(
            """
            SELECT signal_id, features, will_peak_30r
            FROM signal_features
            WHERE label_captured = TRUE
            ORDER BY emitted_at ASC
            """
        )

        n = len(rows)
        if n < min_samples:
            logger.warning(
                "Only %d labeled rows — need %d for Gate 1. Backfill more data.",
                n, min_samples,
            )
            return {
                "gate": "INSUFFICIENT_DATA",
                "n_samples": n,
                "min_required": min_samples,
            }

        # Build arrays
        y = np.array([int(r["will_peak_30r"]) for r in rows], dtype=np.int32)
        scores = np.array([
            score_heuristic(json.loads(r["features"])) for r in rows
        ], dtype=np.float64)

        n_pos = int(y.sum())
        n_neg = int(n - n_pos)

        # Overall metrics
        threshold, prec, rec = find_threshold_at_recall(y, scores, target_recall)

        # Also compute ROC-AUC-style diagnostic
        # (manual pair ranking for Phase A — no sklearn)
        pos_scores = scores[y == 1]
        neg_scores = scores[y == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            # Approximate ROC-AUC via Mann-Whitney
            ranks = np.concatenate([pos_scores, neg_scores]).argsort().argsort() + 1
            pos_ranks = ranks[:len(pos_scores)]
            roc_auc = (pos_ranks.sum() - len(pos_scores) * (len(pos_scores) + 1) / 2) \
                      / (len(pos_scores) * len(neg_scores))
        else:
            roc_auc = float("nan")

        # Verdict
        passed = (prec >= gate_precision) and (rec >= target_recall) and (n >= min_samples)

        result = {
            "gate": "PASS" if passed else "FAIL",
            "n_samples": n,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "positive_rate": n_pos / n,
            "threshold": threshold,
            "precision_at_recall": prec,
            "recall_achieved": rec,
            "roc_auc_approx": float(roc_auc),
            "gate_requirements": {
                "min_samples": min_samples,
                "target_recall": target_recall,
                "min_precision": gate_precision,
            },
        }

        # Log model run
        await conn.execute(
            """
            INSERT INTO ppp_model_runs
                (model_type, trained_at, n_training_samples, n_positives,
                 oof_precision, oof_recall, oof_roc_auc, chosen_threshold, notes)
            VALUES ('heuristic', NOW(), $1, $2, $3, $4, $5, $6, $7)
            """,
            n, n_pos, prec, rec, roc_auc, threshold,
            f"Phase A heuristic, Gate1={'PASS' if passed else 'FAIL'}",
        )

        return result
    finally:
        await conn.close()


def _print_result(r: dict) -> None:
    print("=" * 60)
    print(f"  PPP PHASE A — GATE 1 EVALUATION")
    print("=" * 60)
    if r["gate"] == "INSUFFICIENT_DATA":
        print(f"  ⏳ INSUFFICIENT DATA")
        print(f"  samples: {r['n_samples']} (need {r['min_required']})")
        return
    print(f"  Verdict:           {'✅ PASS' if r['gate']=='PASS' else '❌ FAIL'}")
    print()
    print(f"  Samples:           {r['n_samples']}")
    print(f"  Positives:         {r['n_positives']}  ({r['positive_rate']*100:.1f}%)")
    print(f"  Negatives:         {r['n_negatives']}")
    print()
    print(f"  At target recall = {r['gate_requirements']['target_recall']:.2f}:")
    print(f"    threshold        = {r['threshold']:.3f}")
    print(f"    precision        = {r['precision_at_recall']:.3f}  (gate ≥ {r['gate_requirements']['min_precision']:.2f})")
    print(f"    recall           = {r['recall_achieved']:.3f}")
    print()
    print(f"  ROC-AUC (approx):  {r['roc_auc_approx']:.3f}  (0.50 = random, ≥0.65 = decent)")
    print("=" * 60)
    if r["gate"] == "PASS":
        print("  → Phase A validated. Proceed to Phase B (LogisticRegression).")
    else:
        print("  → Gate failed. Options:")
        print("    - Add more labeled data (backfill more days or wait)")
        print("    - Add features (L2 data, tick velocity) — but see Phase 2 scope")
        print("    - Check label noise (peak_mfe_r captured correctly?)")
    print()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Run Gate 1 evaluation")
    ap.add_argument("--min-samples", type=int, default=300)
    ap.add_argument("--target-recall", type=float, default=0.60)
    ap.add_argument("--min-precision", type=float, default=0.55)
    args = ap.parse_args()

    if not args.eval:
        ap.print_help()
        return

    result = await evaluate_gate1(
        min_samples=args.min_samples,
        target_recall=args.target_recall,
        gate_precision=args.min_precision,
    )
    _print_result(result)
    # Exit code: 0 = pass, 1 = fail, 2 = insufficient data
    sys.exit({"PASS": 0, "FAIL": 1, "INSUFFICIENT_DATA": 2}.get(result["gate"], 1))


if __name__ == "__main__":
    asyncio.run(main())

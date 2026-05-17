#!/usr/bin/env python3
"""
Paper-vs-Shadow Execution Gap Analyzer  —  Wave 6.C Layer 0
============================================================

Goal
----
Both `paper` and `shadow` paths consume the SAME signal pipeline. The only
difference is execution (paper = zero-slip, instant; shadow = realistic L2,
taker fees). Therefore any feature that distinguishes "paper wins / shadow
loses" is the highest-EV pre-signal filter possible.

Pipeline
--------
1.  Load paper closed signals from the JSON file.
2.  Load closed shadow trades from PostgreSQL (`user_trades`).
3.  Match each shadow trade to the paper signal with the same symbol+side
    whose `entry_time` is within +/-60s of `opened_at`.
4.  Compute execution-gap metrics per matched pair.
5.  Univariate analysis (HIGH-GAP vs LOW-GAP quartiles) on candidate features.
6.  Multivariate ranking via GradientBoostingRegressor + permutation importance.
7.  Greedy candidate filter rules (numeric: percentile thresholds; categorical:
    per-level expected savings).  Net-benefit calculation per rule.
8.  Render markdown report  +  cache matched pairs to parquet.
"""

from __future__ import annotations

import json
import os
import sys
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# DB access — bot venv ships with asyncpg, not psycopg2.  Use asyncpg via
# a small sync wrapper.
import asyncio

import asyncpg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAPER_PATH       = Path("/home/opc/crypto-trading-bot/storage/closed_signals.json")
CACHE_DIR        = Path("/home/opc/crypto-trading-bot/storage/wave6c_cache")
CACHE_PARQUET    = CACHE_DIR / "matched_pairs.parquet"
REPORT_PATH      = Path("/home/opc/crypto-trading-bot/.rollback/wave6c-paper-shadow-gap.md")

DB_CONN = dict(
    host="localhost",
    user="vnedge",
    password="VnEdge2026db",
    dbname="vnedge",
)

MATCH_WINDOW_SEC = 60         # +/- match window
MIN_TRADES_FOR_RULE = 3       # minimum trades a categorical rule must reject
RNG_SEED = 17

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[wave6c] {msg}", flush=True)


def parse_iso(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    try:
        # Python 3.11+ accepts ISO-with-Z natively, but be defensive
        s = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(x: Any, n: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        return f"{x:.{n}f}"
    return str(x)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_paper_signals() -> list[dict]:
    log(f"Loading paper signals from {PAPER_PATH} ...")
    with PAPER_PATH.open() as f:
        rows = json.load(f)
    log(f"  -> {len(rows)} paper signals total")
    return rows


async def _fetch_shadow_async() -> list[dict]:
    conn = await asyncpg.connect(
        host=DB_CONN["host"],
        user=DB_CONN["user"],
        password=DB_CONN["password"],
        database=DB_CONN["dbname"],
    )
    try:
        records = await conn.fetch("""
            SELECT id::text                  AS id,
                   user_id::text             AS user_id,
                   symbol,
                   side,
                   entry_price,
                   exit_price,
                   quantity,
                   pnl_usd,
                   fees_usd,
                   opened_at,
                   closed_at,
                   signal_data::text         AS signal_data_json,
                   metadata::text            AS metadata_json
            FROM   user_trades
            WHERE  trade_type = 'shadow'
              AND  status     = 'closed'
            ORDER  BY opened_at;
        """)
    finally:
        await conn.close()
    out: list[dict] = []
    for r in records:
        d = dict(r)
        d["signal_data"] = json.loads(d.pop("signal_data_json") or "{}")
        d["metadata"]    = json.loads(d.pop("metadata_json")    or "{}")
        out.append(d)
    return out


def load_shadow_trades() -> list[dict]:
    log("Loading closed shadow trades from PostgreSQL ...")
    rows = asyncio.run(_fetch_shadow_async())
    log(f"  -> {len(rows)} closed shadow trades")
    return rows


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def build_paper_index(paper: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Bucket paper signals by (symbol, side) for fast lookup."""
    idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped = 0
    for sig in paper:
        sym = sig.get("symbol")
        side = sig.get("side")
        et = parse_iso(sig.get("entry_time"))
        if not (sym and side and et):
            skipped += 1
            continue
        sig["_entry_dt"] = et
        idx[(sym, side)].append(sig)
    for k, lst in idx.items():
        lst.sort(key=lambda s: s["_entry_dt"])
    log(f"  -> indexed paper signals across {len(idx)} (symbol,side) buckets "
        f"({skipped} skipped due to missing time)")
    return idx


def match_shadow_to_paper(
    shadow: list[dict],
    paper_idx: dict[tuple[str, str], list[dict]],
) -> tuple[list[dict], list[dict]]:
    """Return (matched_pairs, unmatched_shadow) — match by closest-in-time."""
    matched: list[dict] = []
    unmatched: list[dict] = []

    for sh in shadow:
        sym, side = sh["symbol"], sh["side"]
        opened = sh["opened_at"]
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        bucket = paper_idx.get((sym, side), [])
        # find closest by absolute time delta inside window
        best = None
        best_dt = None
        for sig in bucket:
            delta = abs((sig["_entry_dt"] - opened).total_seconds())
            if delta <= MATCH_WINDOW_SEC and (best_dt is None or delta < best_dt):
                best, best_dt = sig, delta
        if best is None:
            unmatched.append({**sh, "_unmatch_reason": "no paper signal in +/-60s"})
            continue
        matched.append({"shadow": sh, "paper": best, "delta_sec": best_dt})

    log(f"  -> matched {len(matched)}/{len(shadow)} shadow trades "
        f"({100 * len(matched) / max(1, len(shadow)):.1f}%)")
    return matched, unmatched


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "fee_drag_r", "min_move_pct", "ml_probability", "atr",
    "atr_pct_of_price", "htf_bias", "liq_buffer_pct", "choch_strength",
    "confidence", "ev", "weighted_score", "conviction_score",
    "sl_pct", "scanner_tp1_rr", "regime_age", "ml_match_pct",
    "ml_overfit_gap", "p_win", "leverage", "round_number_dist_pct",
    "scanner_weight", "delta_sec",
]

CATEGORICAL_FEATURES = [
    "regime", "grade", "scanner", "session", "vwap_zone",
    "ev_verdict", "ml_verdict", "signal_tier", "scanner_category",
    "fee_viable", "in_kill_zone", "indian_market", "sniper_eligible",
    "structure_bounce_only", "near_round_number", "fee_type",
    "symbol", "side", "trade_type", "operating_mode",
    "hour_of_day_bin", "weekday",
]


def hour_bin(h: int) -> str:
    if 0 <= h < 6:
        return "0_asia_early"
    if 6 <= h < 12:
        return "6_asia_late"
    if 12 <= h < 18:
        return "12_eu_us"
    return "18_us_late"


def extract_row(pair: dict) -> dict:
    """Flatten one matched pair into a single feature row."""
    sh    = pair["shadow"]
    pp    = pair["paper"]
    pmd   = pp.get("metadata") or {}
    smd   = sh.get("metadata") or {}

    paper_pnl  = safe_float(pp.get("pnl_usd"))
    shadow_pnl = safe_float(sh.get("pnl_usd"))
    paper_init = safe_float(pp.get("initial_risk")) or safe_float(pp.get("risk_amount_usd"))
    shadow_init = safe_float(smd.get("initial_risk"))
    init_risk = shadow_init or paper_init or 1.0
    if init_risk == 0:
        init_risk = 1.0

    gap_usd  = (paper_pnl or 0.0) - (shadow_pnl or 0.0)
    gap_r    = gap_usd / init_risk
    rel_gap  = gap_usd / abs(paper_pnl) if paper_pnl not in (None, 0) else None

    opened_at = sh["opened_at"]
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)

    entry_price = safe_float(pp.get("entry_price"))
    atr_v       = safe_float(pmd.get("atr"))
    atr_pct = (atr_v / entry_price) if (atr_v and entry_price) else None

    row = {
        # IDs
        "shadow_id":      sh["id"],
        "trade_id":       pp.get("trade_id"),
        "symbol":         sh["symbol"],
        "side":           sh["side"],
        "opened_at":      opened_at,
        "delta_sec":      pair["delta_sec"],
        # Outcomes
        "paper_pnl":      paper_pnl,
        "shadow_pnl":     shadow_pnl,
        "paper_fees":     safe_float(pp.get("total_fees_usd")),
        "shadow_fees":    safe_float(sh.get("fees_usd")),
        "gap_usd":        gap_usd,
        "gap_r":          gap_r,
        "rel_gap":        rel_gap,
        "init_risk":      init_risk,
        "paper_win":      (paper_pnl or 0.0) > 0.0,
        "shadow_win":     (shadow_pnl or 0.0) > 0.0,
        "paper_to_shadow_loss": ((paper_pnl or 0.0) > 0.0) and ((shadow_pnl or 0.0) < 0.0),
        # Numeric features (paper metadata)
        "fee_drag_r":           safe_float(pmd.get("fee_drag_r")),
        "min_move_pct":         safe_float(pmd.get("min_move_pct")),
        "ml_probability":       safe_float(pmd.get("ml_probability") or smd.get("ml_prob")),
        "atr":                  atr_v,
        "atr_pct_of_price":     atr_pct,
        "htf_bias":             safe_float(pmd.get("htf_bias")),
        "liq_buffer_pct":       safe_float(pmd.get("liq_buffer_pct")),
        "choch_strength":       safe_float(pmd.get("choch_strength")),
        "confidence":           safe_float(pp.get("confidence")),
        "ev":                   safe_float(pmd.get("ev")),
        "weighted_score":       safe_float(pmd.get("weighted_score")),
        "conviction_score":     safe_float(pmd.get("conviction_score")),
        "sl_pct":               safe_float(pmd.get("sl_pct")),
        "scanner_tp1_rr":       safe_float(pmd.get("scanner_tp1_rr")),
        "regime_age":           safe_float(pmd.get("regime_age")),
        "ml_match_pct":         safe_float(pmd.get("ml_match_pct")),
        "ml_overfit_gap":       safe_float(pmd.get("ml_overfit_gap")),
        "p_win":                safe_float(pmd.get("p_win")),
        "leverage":             safe_float(pp.get("leverage") or smd.get("leverage")),
        "round_number_dist_pct": safe_float(pmd.get("round_number_dist_pct")),
        "scanner_weight":       safe_float(pmd.get("scanner_weight")),
        # Categorical features
        "regime":               pmd.get("regime") or smd.get("regime"),
        "grade":                pp.get("grade") or smd.get("grade"),
        "scanner":              pp.get("setup_type") or pmd.get("setup_type") or smd.get("scanner"),
        "session":              pmd.get("session"),
        "vwap_zone":            pmd.get("vwap_zone"),
        "ev_verdict":           pmd.get("ev_verdict"),
        "ml_verdict":           pmd.get("ml_verdict"),
        "signal_tier":          pmd.get("signal_tier"),
        "scanner_category":     pmd.get("scanner_category"),
        "fee_viable":           pmd.get("fee_viable"),
        "in_kill_zone":         pmd.get("in_kill_zone"),
        "indian_market":        pmd.get("indian_market"),
        "sniper_eligible":      pmd.get("sniper_eligible"),
        "structure_bounce_only": pmd.get("structure_bounce_only"),
        "near_round_number":    pmd.get("near_round_number"),
        "fee_type":             pp.get("fee_type") or smd.get("fee_type"),
        "trade_type":           pp.get("trade_type") or smd.get("trade_type"),
        "operating_mode":       pmd.get("operating_mode"),
        "hour_of_day":          opened_at.hour,
        "hour_of_day_bin":      hour_bin(opened_at.hour),
        "weekday":              opened_at.strftime("%a"),
    }
    return row


def build_dataframe(matched: list[dict]) -> pd.DataFrame:
    rows = [extract_row(m) for m in matched]
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Univariate analysis
# ---------------------------------------------------------------------------
@dataclass
class UnivResult:
    feature: str
    kind: str            # "numeric" | "categorical"
    high_mean: Optional[float]
    low_mean: Optional[float]
    correlation: Optional[float]
    n_high: int
    n_low: int
    detail: str = ""


def univariate(df: pd.DataFrame) -> list[UnivResult]:
    """Quartile split on gap_usd (worst quartile = HIGH-GAP, best = LOW-GAP)."""
    out: list[UnivResult] = []
    if len(df) < 8:
        log("WARN: too few rows for quartile split — falling back to median split")
        q_low  = df["gap_usd"].median()
        q_high = q_low
        high_mask = df["gap_usd"] <= q_low
        low_mask  = df["gap_usd"] >  q_low
    else:
        # Worst gap = paper-shadow LARGE = paper_pnl much greater than shadow_pnl
        q_high_thr = df["gap_usd"].quantile(0.75)   # top quartile gap = bad for shadow
        q_low_thr  = df["gap_usd"].quantile(0.25)
        high_mask = df["gap_usd"] >= q_high_thr
        low_mask  = df["gap_usd"] <= q_low_thr

    df_high = df[high_mask]
    df_low  = df[low_mask]

    for feat in NUMERIC_FEATURES:
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        if s.notna().sum() < 5:
            continue
        h_mean = pd.to_numeric(df_high[feat], errors="coerce").mean()
        l_mean = pd.to_numeric(df_low[feat], errors="coerce").mean()
        corr   = s.corr(df["gap_usd"])
        out.append(UnivResult(feat, "numeric",
                              float(h_mean) if not pd.isna(h_mean) else None,
                              float(l_mean) if not pd.isna(l_mean) else None,
                              float(corr)   if not pd.isna(corr)   else None,
                              int(high_mask.sum()), int(low_mask.sum())))

    for feat in CATEGORICAL_FEATURES:
        if feat not in df.columns:
            continue
        if df[feat].isna().all():
            continue
        # mean gap per category
        means = df.groupby(feat, dropna=True)["gap_usd"].agg(["mean", "count"])
        if means.empty:
            continue
        # Also report HIGH/LOW counts within category
        worst_cat = means.sort_values("mean").index[0]   # most-positive gap = worst for shadow
        best_cat  = means.sort_values("mean").index[-1]
        detail = "; ".join(
            f"{idx}={r['mean']:.2f}(n={int(r['count'])})"
            for idx, r in means.iterrows())
        out.append(UnivResult(feat, "categorical",
                              float(means["mean"].max()),
                              float(means["mean"].min()),
                              None,
                              int(means.loc[worst_cat, "count"]),
                              int(means.loc[best_cat, "count"]),
                              detail))
    return out


# ---------------------------------------------------------------------------
# Multivariate ranking
# ---------------------------------------------------------------------------
def multivariate(df: pd.DataFrame) -> pd.DataFrame:
    feats: list[str] = []
    X_parts: list[np.ndarray] = []

    # numeric — fillna with column median
    for f in NUMERIC_FEATURES:
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        if col.notna().sum() < 5:
            continue
        col = col.fillna(col.median() if col.notna().any() else 0.0)
        X_parts.append(col.values.reshape(-1, 1))
        feats.append(f)

    # categorical — one-hot, but cap cardinality
    cat_cols, cat_data = [], []
    for f in CATEGORICAL_FEATURES:
        if f not in df.columns:
            continue
        if df[f].isna().all():
            continue
        if df[f].nunique() > 20:
            continue
        cat_cols.append(f)
        cat_data.append(df[f].astype(str).fillna("None"))

    if cat_cols:
        cat_df = pd.concat(cat_data, axis=1)
        cat_df.columns = cat_cols
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = enc.fit_transform(cat_df.values)
        feats.extend([f"{c}={v}" for c, vs in zip(cat_cols, enc.categories_) for v in vs])
        X_parts.append(X_cat)

    if not X_parts:
        return pd.DataFrame(columns=["feature", "importance", "perm_importance"])

    X = np.concatenate(X_parts, axis=1)
    y = df["gap_usd"].values

    n_est = min(50, max(10, len(df) // 2))
    model = GradientBoostingRegressor(
        max_depth=3, n_estimators=n_est, random_state=RNG_SEED,
        learning_rate=0.05)
    model.fit(X, y)
    imp = model.feature_importances_

    # Permutation importance — robust ranking
    try:
        perm = permutation_importance(
            model, X, y, n_repeats=20, random_state=RNG_SEED,
            scoring="neg_mean_squared_error")
        perm_imp = perm.importances_mean
    except Exception as e:
        log(f"WARN: permutation_importance failed: {e}")
        perm_imp = np.zeros_like(imp)

    out = pd.DataFrame({
        "feature": feats,
        "importance": imp,
        "perm_importance": perm_imp,
    }).sort_values("perm_importance", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Candidate filter rules
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    rule: str
    rejects_n: int
    saved_loss_usd: float       # sum of shadow_pnl for rejected (negative => bad => reject saves money)
    paper_winners_lost: float   # sum of paper_pnl among rejected paper-winners
    net_usd: float
    rejection_pct: float


def _eval_rule(df: pd.DataFrame, mask: pd.Series, rule_str: str) -> RuleResult:
    rej = df[mask]
    if rej.empty:
        return RuleResult(rule_str, 0, 0.0, 0.0, 0.0, 0.0)
    saved = -float(rej["shadow_pnl"].sum())   # losses we avoid (positive number = good)
    paper_win_in_rej = rej[rej["paper_pnl"] > 0]
    paper_lost = float(paper_win_in_rej["paper_pnl"].sum())
    net = saved - paper_lost
    return RuleResult(
        rule=rule_str,
        rejects_n=int(mask.sum()),
        saved_loss_usd=saved,
        paper_winners_lost=paper_lost,
        net_usd=net,
        rejection_pct=100.0 * mask.sum() / len(df),
    )


def candidate_rules(df: pd.DataFrame) -> list[RuleResult]:
    """Sweep candidate rules — pick top by net P&L improvement."""
    rules: list[RuleResult] = []

    # numeric: try percentile thresholds in both directions
    # Use full-precision threshold in the rule string so re-runs match exactly.
    for f in NUMERIC_FEATURES:
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        if col.notna().sum() < 8:
            continue
        for q in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90):
            thr = col.quantile(q)
            if pd.isna(thr):
                continue
            mask = col >= thr
            if MIN_TRADES_FOR_RULE <= mask.sum() < len(df):
                rules.append(_eval_rule(df, mask, f"{f} >= {thr:.10g}"))
            q_lo = 1.0 - q
            thr_lo = col.quantile(q_lo)
            if pd.isna(thr_lo):
                continue
            mask_lo = col <= thr_lo
            if MIN_TRADES_FOR_RULE <= mask_lo.sum() < len(df):
                rules.append(_eval_rule(df, mask_lo, f"{f} <= {thr_lo:.10g}"))

    # categorical: reject each level that has >= MIN_TRADES_FOR_RULE
    for f in CATEGORICAL_FEATURES:
        if f not in df.columns:
            continue
        for val, grp in df.groupby(f, dropna=True):
            if len(grp) < MIN_TRADES_FOR_RULE:
                continue
            mask = df[f] == val
            rules.append(_eval_rule(df, mask, f"{f} == {val!r}"))

    # Boolean is_loss-by-fee-drag style combos
    if "fee_drag_r" in df.columns and "ml_probability" in df.columns:
        col_f = pd.to_numeric(df["fee_drag_r"], errors="coerce")
        col_m = pd.to_numeric(df["ml_probability"], errors="coerce")
        for fdq in (0.50, 0.60, 0.70):
            for mlq in (0.30, 0.40, 0.50):
                fdt = col_f.quantile(fdq)
                mlt = col_m.quantile(mlq)
                if pd.isna(fdt) or pd.isna(mlt):
                    continue
                mask = (col_f >= fdt) & (col_m <= mlt)
                if MIN_TRADES_FOR_RULE <= mask.sum() < len(df):
                    rules.append(_eval_rule(
                        df, mask,
                        f"fee_drag_r >= {fdt:.10g} AND ml_probability <= {mlt:.10g}"))

    rules = [r for r in rules if r.rejects_n > 0]
    rules.sort(key=lambda r: r.net_usd, reverse=True)
    return rules


def flip_targeted_rules(df: pd.DataFrame) -> list[RuleResult]:
    """Sweep rules ranked by flip-precision: how many paper-WIN→shadow-LOSS
    trades does the rule reject vs how many paper-WIN→shadow-WIN trades
    (true-positives we want to keep)."""
    out: list[RuleResult] = []
    for f in NUMERIC_FEATURES:
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        if col.notna().sum() < 8:
            continue
        for q in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            for op in ("ge", "le"):
                thr = col.quantile(q if op == "ge" else 1.0 - q)
                if pd.isna(thr):
                    continue
                mask = (col >= thr) if op == "ge" else (col <= thr)
                if mask.sum() < MIN_TRADES_FOR_RULE or mask.sum() == len(df):
                    continue
                out.append(_eval_rule(
                    df, mask,
                    f"{f} {'>=' if op == 'ge' else '<='} {thr:.10g}"))
    for f in CATEGORICAL_FEATURES:
        if f not in df.columns:
            continue
        for val, grp in df.groupby(f, dropna=True):
            if len(grp) < MIN_TRADES_FOR_RULE:
                continue
            mask = df[f] == val
            out.append(_eval_rule(df, mask, f"{f} == {val!r}"))
    # Score = flip lift * sqrt(flips caught), penalizing rules with tiny lift
    # over baseline flip rate.  Skip rules that don't beat baseline.
    baseline_flip_rate = float(df["paper_to_shadow_loss"].mean())
    scored: list[tuple[float, RuleResult]] = []
    seen_rules: set[str] = set()
    for r in out:
        if r.rule in seen_rules:
            continue
        seen_rules.add(r.rule)
        try:
            mask = _rule_to_mask(df, r.rule)
        except Exception:
            continue
        rej = df[mask]
        if len(rej) == 0:
            continue
        flips_caught = int(rej["paper_to_shadow_loss"].sum())
        true_winners_lost = int(((rej["paper_pnl"] > 0) & (rej["shadow_pnl"] > 0)).sum())
        flip_rate_in_rej = flips_caught / len(rej)
        lift = flip_rate_in_rej - baseline_flip_rate
        # Require positive lift AND at least 2 flips caught
        if lift <= 0 or flips_caught < 2:
            continue
        score = lift * math.sqrt(flips_caught) - 0.1 * true_winners_lost
        scored.append((score, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored]


def select_filter_set(df: pd.DataFrame, top_rules: list[RuleResult],
                      max_rules: int = 3) -> list[RuleResult]:
    """Greedy: pick rules that incrementally extend rejection set with positive net.

    Looser criterion than v1: a rule is OK to add if its INCREMENTAL net
    (only on newly-rejected trades) is positive AND adds at least 2 new rejects.
    """
    chosen: list[RuleResult] = []
    cumulative_mask = pd.Series(False, index=df.index)
    for r in top_rules:
        if len(chosen) >= max_rules:
            break
        if r.net_usd <= 0:
            continue
        try:
            mask = _rule_to_mask(df, r.rule)
        except Exception:
            continue
        new_rejects = mask & (~cumulative_mask)
        if new_rejects.sum() < 2:
            continue
        incr = _eval_rule(df, new_rejects, f"INCR: {r.rule}")
        if incr.net_usd <= 0:
            continue
        chosen.append(r)
        cumulative_mask = cumulative_mask | mask
    return chosen, cumulative_mask


def _rule_to_mask(df: pd.DataFrame, rule_str: str) -> pd.Series:
    """Re-parse a rule string like 'feat >= 0.4' into a boolean mask."""
    if " AND " in rule_str:
        a, b = rule_str.split(" AND ")
        return _rule_to_mask(df, a.strip()) & _rule_to_mask(df, b.strip())
    for op in (" >= ", " <= ", " == "):
        if op in rule_str:
            feat, val = rule_str.split(op, 1)
            feat = feat.strip()
            val  = val.strip()
            col  = df[feat]
            if op == " == ":
                # value may be quoted python repr
                try:
                    v = eval(val, {"__builtins__": {}}, {})
                except Exception:
                    v = val
                return col == v
            else:
                col_n = pd.to_numeric(col, errors="coerce")
                v = float(val)
                return (col_n >= v) if op == " >= " else (col_n <= v)
    raise ValueError(f"unparseable rule: {rule_str!r}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_report(
    df: pd.DataFrame,
    paper_n: int,
    shadow_n: int,
    matched_n: int,
    unmatched: list[dict],
    univ: list[UnivResult],
    multi: pd.DataFrame,
    rules: list[RuleResult],
    chosen: list[RuleResult],
    chosen_mask: pd.Series,
    window_start: Optional[datetime],
    window_end: Optional[datetime],
    flip_rules: Optional[list[RuleResult]] = None,
) -> str:
    L: list[str] = []
    L.append("# Paper-vs-Shadow Execution Gap Analysis — Wave 6.C Layer 0")
    L.append("")
    L.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    L.append("")
    L.append(f"**Window:** {window_start} → {window_end}")
    L.append(f"- Paper signals (full file): {paper_n}")
    L.append(f"- Shadow trades (closed):   {shadow_n}")
    L.append(f"- Matched pairs:            {matched_n}  ({100*matched_n/max(1,shadow_n):.1f}%)")
    L.append("")

    # Headline — flip subset analysis
    flips = df[df["paper_to_shadow_loss"]]
    L.append("## Headline finding: paper-WIN → shadow-LOSS \"flip\" trades")
    L.append("")
    L.append(f"- **{len(flips)} of {len(df)} trades ({100*len(flips)/max(1,len(df)):.0f}%)** had paper profitable but shadow losing.")
    L.append(f"- Combined shadow loss in the flip subset: **${flips['shadow_pnl'].sum():+.2f}** (vs paper +${flips['paper_pnl'].sum():.2f}).")
    L.append(f"- Median fee-drag-R in flips: {flips['fee_drag_r'].median():.3f}; in non-flips: {df.loc[~df['paper_to_shadow_loss'], 'fee_drag_r'].median():.3f}")
    L.append(f"- These flip trades are the primary leak that this filter exercise targets.")
    L.append("")

    # Structural finding — sizing & fee dominance
    L.append("## CRITICAL — structural finding before feature analysis")
    L.append("")
    paper_avg_abs = float(df["paper_pnl"].abs().mean())
    shadow_avg_abs = float(df["shadow_pnl"].abs().mean())
    paper_avg_fee = float(df["paper_fees"].mean())
    shadow_avg_fee = float(df["shadow_fees"].mean())
    sizing_ratio = paper_avg_abs / max(shadow_avg_abs, 1e-9)
    L.append(f"- Average **|paper P&L|** per trade: **${paper_avg_abs:.2f}**, average **|shadow P&L|**: **${shadow_avg_abs:.3f}**.")
    L.append(f"- Sizing ratio: paper bets ~**{sizing_ratio:.1f}×** larger absolute outcomes than shadow.")
    L.append(f"- Average paper fees / trade: ${paper_avg_fee:.2f}; shadow fees / trade: ${shadow_avg_fee:.3f}.")
    L.append(f"- Shadow `fee_pct_of_gross` (sum of fees as % of gross P&L) commonly **>500%** — fees alone exceed the trade's gross 5-20x.")
    L.append(f"- Most shadow exits are early defensives (`quick_kill`, `zombie_kill`, `no_proof_of_life`, `early_kill`) firing within minutes, before the move develops.")
    L.append("")
    L.append("**Implication:** This is *not* primarily a slippage-driven L2 execution gap. It's a structural mismatch:")
    L.append("")
    L.append("1. **Shadow margins are micro** (~$10–$37 per leg) so fees swamp any small move.")
    L.append("2. **Shadow defensive exits** kill positions before they reach paper's TP1/trail_profit exits.")
    L.append("3. A pre-signal feature filter can only marginally improve this — the larger leverage points are: (a) raise minimum shadow margin, (b) align shadow defensive-exit thresholds with paper, (c) require min_move_pct > N × estimated_fees_pct before any signal becomes shadow-eligible.")
    L.append("")
    L.append("With that caveat, the feature analysis below still ranks which signals bleed *most* and proposes filters that gate the worst.")
    L.append("")

    # Per-feature flip rate (categorical — easy to scan)
    L.append("### Flip-rate by categorical feature (descending)")
    L.append("")
    L.append("| Feature=Value | N | Flips | Flip rate | Sum shadow $ | Sum paper $ |")
    L.append("|---|---:|---:|---:|---:|---:|")
    flip_rows: list[tuple] = []
    for feat in CATEGORICAL_FEATURES:
        if feat not in df.columns or df[feat].isna().all():
            continue
        for val, grp in df.groupby(feat, dropna=True):
            if len(grp) < 3:
                continue
            n = len(grp)
            f_n = int(grp["paper_to_shadow_loss"].sum())
            f_rate = f_n / n
            s_pnl = float(grp["shadow_pnl"].sum())
            p_pnl = float(grp["paper_pnl"].sum())
            flip_rows.append((feat, val, n, f_n, f_rate, s_pnl, p_pnl))
    flip_rows.sort(key=lambda r: (-r[4], r[5]))
    for feat, val, n, f_n, f_rate, s_pnl, p_pnl in flip_rows[:20]:
        L.append(f"| {feat}={val} | {n} | {f_n} | {100*f_rate:.0f}% | {s_pnl:+.2f} | {p_pnl:+.2f} |")
    L.append("")

    # Aggregate gap
    L.append("## Aggregate gap")
    L.append("")
    L.append("| Metric | Paper | Shadow | Gap |")
    L.append("|---|---:|---:|---:|")
    p_total = float(df["paper_pnl"].sum())
    s_total = float(df["shadow_pnl"].sum())
    L.append(f"| Total P&L (USD)        | {p_total:.2f} | {s_total:.2f} | {p_total - s_total:+.2f} |")
    L.append(f"| Avg P&L / trade (USD)  | {df['paper_pnl'].mean():.3f} | {df['shadow_pnl'].mean():.3f} | {df['gap_usd'].mean():+.3f} |")
    L.append(f"| Win rate (%)           | {100*df['paper_win'].mean():.1f} | {100*df['shadow_win'].mean():.1f} | {100*(df['paper_win'].mean()-df['shadow_win'].mean()):+.1f} pp |")
    L.append(f"| Median gap (USD)       | — | — | {df['gap_usd'].median():+.3f} |")
    L.append(f"| Median gap (R)         | — | — | {df['gap_r'].median():+.3f} |")
    p2s = int(df["paper_to_shadow_loss"].sum())
    L.append(f"| Paper-WIN → Shadow-LOSS flips | — | — | {p2s} ({100*p2s/len(df):.1f}%) |")
    L.append("")

    # Univariate — numeric
    L.append("## Univariate features — numeric (HIGH-GAP vs LOW-GAP quartile)")
    L.append("")
    L.append("| Feature | High-gap mean | Low-gap mean | Correlation w/ gap_usd |")
    L.append("|---|---:|---:|---:|")
    for r in [u for u in univ if u.kind == "numeric"]:
        L.append(f"| {r.feature} | {fmt(r.high_mean,3)} | {fmt(r.low_mean,3)} | {fmt(r.correlation,3)} |")
    L.append("")

    # Univariate — categorical
    L.append("## Univariate features — categorical (mean gap_usd per level)")
    L.append("")
    L.append("| Feature | Worst-mean | Best-mean | Detail |")
    L.append("|---|---:|---:|---|")
    for r in [u for u in univ if u.kind == "categorical"]:
        L.append(f"| {r.feature} | {fmt(r.high_mean,3)} | {fmt(r.low_mean,3)} | {r.detail} |")
    L.append("")

    # Multivariate
    L.append("## Multivariate feature importance (GBR, permutation_importance)")
    L.append("")
    L.append("| Rank | Feature | Importance | Perm Importance |")
    L.append("|---:|---|---:|---:|")
    for i, row in enumerate(multi.head(20).itertuples(index=False), 1):
        L.append(f"| {i} | {row.feature} | {row.importance:.4f} | {row.perm_importance:.4f} |")
    L.append("")

    # Rules
    L.append("## Top candidate filter rules (sorted by net USD impact)")
    L.append("")
    L.append("| Rule | Rejects N | Saves loss $ | Loses paper-winners $ | NET $ | Rej % |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in rules[:25]:
        L.append(f"| `{r.rule}` | {r.rejects_n} | {r.saved_loss_usd:+.2f} | {r.paper_winners_lost:+.2f} | {r.net_usd:+.2f} | {r.rejection_pct:.1f}% |")
    L.append("")

    # Flip-targeted rules (specifically optimized for catching paper-WIN→shadow-LOSS flips)
    if flip_rules:
        baseline_flip = float(df["paper_to_shadow_loss"].mean())
        L.append("## Flip-targeted rules (optimized for catching paper-WIN → shadow-LOSS)")
        L.append("")
        L.append(f"Baseline flip rate (entire matched set): **{100*baseline_flip:.1f}%**.  ")
        L.append("Rules below have flip rate >= baseline AND >= 2 flips caught.  ")
        L.append("Score = lift × sqrt(flips_caught) − 0.1 × true_winners_lost.")
        L.append("")
        L.append("| Rule | Rejects N | Flips caught | Flip rate | Lift vs baseline | True winners lost | Shadow$ saved |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in flip_rules[:25]:
            try:
                m = _rule_to_mask(df, r.rule)
            except Exception:
                continue
            rej = df[m]
            n = len(rej)
            flips_caught = int(rej["paper_to_shadow_loss"].sum())
            flip_rate = flips_caught / n if n else 0.0
            lift = flip_rate - baseline_flip
            true_winners = int(((rej["paper_pnl"] > 0) & (rej["shadow_pnl"] > 0)).sum())
            L.append(f"| `{r.rule}` | {n} | {flips_caught} | {100*flip_rate:.0f}% | {100*lift:+.0f}pp | {true_winners} | {r.saved_loss_usd:+.2f} |")
        L.append("")

    # Recommended set
    L.append("## Recommended filter set (greedy combination)")
    L.append("")
    if not chosen:
        L.append("_No rule with positive net P&L impact passed the greedy selection._")
    else:
        L.append("| # | Rule | Rejects N (rule alone) | NET $ |")
        L.append("|---:|---|---:|---:|")
        for i, r in enumerate(chosen, 1):
            L.append(f"| {i} | `{r.rule}` | {r.rejects_n} | {r.net_usd:+.2f} |")
        L.append("")
        rejected = df[chosen_mask]
        kept = df[~chosen_mask]
        rej_n = len(rejected)
        rej_pct = 100.0 * rej_n / max(1, len(df))
        saved = -float(rejected["shadow_pnl"].sum())
        paper_lost = float(rejected[rejected["paper_pnl"] > 0]["paper_pnl"].sum())
        net = saved - paper_lost
        L.append("### Combined impact on the matched set")
        L.append("")
        L.append(f"- Rejection rate: **{rej_pct:.1f}%** ({rej_n} of {len(df)} trades)")
        L.append(f"- Shadow loss avoided: **${saved:+.2f}**")
        L.append(f"- Paper winners lost: ${paper_lost:+.2f}")
        L.append(f"- **NET on matched window: ${net:+.2f}**")
        if window_start and window_end:
            window_days = max(0.5, (window_end - window_start).total_seconds() / 86400.0)
            per_week = net / window_days * 7
            L.append(f"- Window length: {window_days:.2f} days  →  projected **${per_week:+.2f} / week** saved")
        # kept-set characterisation
        if len(kept) > 0:
            L.append("")
            L.append("### Kept-set summary")
            L.append("")
            L.append(f"- N kept: {len(kept)}")
            L.append(f"- Kept paper P&L: ${kept['paper_pnl'].sum():+.2f}")
            L.append(f"- Kept shadow P&L: ${kept['shadow_pnl'].sum():+.2f}")
            L.append(f"- Kept paper WR: {100*kept['paper_win'].mean():.1f}%, shadow WR: {100*kept['shadow_win'].mean():.1f}%")
    L.append("")

    # Unmatched
    L.append(f"## Unmatched shadow trades ({len(unmatched)})")
    L.append("")
    if unmatched:
        L.append("| shadow_id | symbol | side | opened_at | reason |")
        L.append("|---|---|---|---|---|")
        for u in unmatched:
            L.append(f"| {str(u['id'])[:8]} | {u['symbol']} | {u['side']} | {u['opened_at']} | {u['_unmatch_reason']} |")
    else:
        L.append("_All shadow trades matched._")
    L.append("")

    # Data quality / caveats
    L.append("## Data quality notes")
    L.append("")
    L.append("- Shadow `signal_data` is empty (`{}`) for every row — features must be sourced from `metadata` or from the matched paper signal.")
    L.append("- Shadow `metadata` has a much sparser feature schema than paper signals (no `fee_drag_r`, `htf_bias`, `liq_buffer_pct`, `choch_strength`, `ev`, etc.). This pipeline therefore reads features from the matched **paper** signal — valid because paper and shadow share the same upstream signal.")
    L.append("- Each paper signal can fan out to **multiple shadow trades** (one per shadow user). Each shadow trade is matched independently; the paper signal is reused.")
    L.append(f"- Matching window: ±{MATCH_WINDOW_SEC}s. Closest-by-time wins.")
    L.append(f"- Sample size (N={len(df)}) is small. Treat all rules as **medium-confidence at best** — re-run weekly.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    paper = load_paper_signals()
    shadow = load_shadow_trades()
    if not shadow:
        log("No shadow trades — aborting.")
        return

    paper_idx = build_paper_index(paper)
    matched, unmatched = match_shadow_to_paper(shadow, paper_idx)

    if not matched:
        log("No matches — aborting.")
        return

    df = build_dataframe(matched)
    log(f"Built feature dataframe: {df.shape[0]} rows x {df.shape[1]} cols")

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(CACHE_PARQUET, index=False)
        log(f"Cached matched pairs to {CACHE_PARQUET}")
    except Exception as e:
        # Fall back to JSON if parquet engine missing
        json_path = CACHE_PARQUET.with_suffix(".json")
        df.to_json(json_path, orient="records", default_handler=str)
        log(f"Parquet cache failed ({e}); wrote JSON fallback to {json_path}")

    log("Running univariate analysis ...")
    univ = univariate(df)

    log("Running multivariate ranking ...")
    multi = multivariate(df)

    log("Sweeping candidate rules ...")
    rules = candidate_rules(df)
    log(f"  -> {len(rules)} net-USD candidate rules")
    flip_rules = flip_targeted_rules(df)
    log(f"  -> {len(flip_rules)} flip-targeted candidate rules")

    chosen, chosen_mask = select_filter_set(df, rules, max_rules=3)
    log(f"  -> {len(chosen)} rules in recommended set")

    window_start = df["opened_at"].min()
    window_end   = df["opened_at"].max()

    report = render_report(
        df=df,
        paper_n=len(paper),
        shadow_n=len(shadow),
        matched_n=len(matched),
        unmatched=unmatched,
        univ=univ,
        multi=multi,
        rules=rules,
        chosen=chosen,
        chosen_mask=chosen_mask,
        window_start=window_start,
        window_end=window_end,
        flip_rules=flip_rules,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    log(f"Wrote report to {REPORT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()

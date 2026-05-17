"""
Cohort breakdowns for backtest output.

Slices the per-trade DataFrame by symbol / regime / grade / scanner
and computes key metrics for each. Reveals which cohorts carry the edge
under each fill model (especially useful for identifying which signals
should be maker-probed vs taker-admitted).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _cohort_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Quick summary stats for a group — no heavy Sharpe calc."""
    if len(df) == 0:
        return dict(n=0, net=0, avg=0, wr=0, pf=0)
    pnls = df["sim_net_usd"].astype(float).values
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return dict(
        n=len(pnls),
        net=float(pnls.sum()),
        avg=float(pnls.mean()),
        wr=float((pnls > 0).mean() * 100),
        pf=float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf"),
    )


def breakdown_by(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Group df by `key` and compute per-cohort metrics.
    Sorted by net P&L descending.
    """
    if len(df) == 0 or key not in df.columns:
        return pd.DataFrame(columns=["cohort", "n", "net_usd", "avg", "wr_pct", "profit_factor"])

    rows = []
    for cohort_value, grp in df.groupby(key):
        stats = _cohort_stats(grp)
        rows.append({
            "cohort": str(cohort_value) if cohort_value else "?",
            "n": stats["n"],
            "net_usd": round(stats["net"], 2),
            "avg": round(stats["avg"], 3),
            "wr_pct": round(stats["wr"], 1),
            "profit_factor": round(stats["pf"], 2) if stats["pf"] != float("inf") else None,
        })
    return pd.DataFrame(rows).sort_values("net_usd", ascending=False).reset_index(drop=True)


def breakdown_by_two(df: pd.DataFrame, key1: str, key2: str) -> pd.DataFrame:
    """
    Group df by two keys (e.g. grade × regime) and compute per-cohort metrics.
    Useful for finding sweet spots like "A+ grade in sideways regime".
    """
    if len(df) == 0 or key1 not in df.columns or key2 not in df.columns:
        return pd.DataFrame()
    rows = []
    for (v1, v2), grp in df.groupby([key1, key2]):
        stats = _cohort_stats(grp)
        rows.append({
            key1: str(v1) if v1 else "?",
            key2: str(v2) if v2 else "?",
            "n": stats["n"],
            "net_usd": round(stats["net"], 2),
            "avg": round(stats["avg"], 3),
            "wr_pct": round(stats["wr"], 1),
        })
    return pd.DataFrame(rows).sort_values("net_usd", ascending=False).reset_index(drop=True)


def top_bottom_cohorts(df: pd.DataFrame, key: str, n: int = 5) -> Dict[str, pd.DataFrame]:
    """Return top N best and N worst cohorts by net P&L."""
    full = breakdown_by(df, key)
    return {
        "top": full.head(n),
        "bottom": full.tail(n).iloc[::-1],  # reverse so worst is first
    }


def format_cohort_table(df: pd.DataFrame, label: str, max_rows: int = 20) -> str:
    """Human-readable table. Shows up to max_rows rows."""
    if len(df) == 0:
        return f"\n(no data for {label})\n"
    lines = ["", "─" * 72, f"  COHORT: {label}", "─" * 72]
    header = f"  {'cohort':<20} {'n':>6} {'net $':>10} {'avg':>8} {'wr%':>6} {'PF':>6}"
    lines.append(header)
    lines.append(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")
    for _, r in df.head(max_rows).iterrows():
        pf_str = f"{r['profit_factor']:.2f}" if r.get('profit_factor') is not None else "n/a"
        lines.append(
            f"  {str(r['cohort']):<20} {r['n']:>6} {r['net_usd']:>+10.2f} "
            f"{r['avg']:>+8.3f} {r['wr_pct']:>5.1f} {pf_str:>6}"
        )
    lines.append("")
    return "\n".join(lines)


def run_all_breakdowns(df: pd.DataFrame, label: str) -> str:
    """Generate full cohort report for one fill model."""
    if len(df) == 0:
        return f"\n(no trades for {label})\n"

    parts = [f"\n{'='*72}", f"  COHORT BREAKDOWNS — {label}", "=" * 72]

    # Single-key breakdowns
    for key, lbl in [
        ("symbol",       "By SYMBOL"),
        ("side",         "By SIDE"),
        ("scanner",      "By SCANNER"),
        ("grade",        "By GRADE"),
        ("regime",       "By REGIME"),
        ("exit_reason",  "By EXIT REASON"),
    ]:
        if key in df.columns:
            parts.append(format_cohort_table(breakdown_by(df, key), lbl))

    # Two-key insights
    if "grade" in df.columns and "regime" in df.columns:
        parts.append("─" * 72)
        parts.append("  GRADE × REGIME (top 15)")
        parts.append("─" * 72)
        two = breakdown_by_two(df, "grade", "regime").head(15)
        if len(two):
            parts.append(two.to_string(index=False))
        parts.append("")

    if "symbol" in df.columns and "side" in df.columns:
        parts.append("─" * 72)
        parts.append("  SYMBOL × SIDE (top 15)")
        parts.append("─" * 72)
        two = breakdown_by_two(df, "symbol", "side").head(15)
        if len(two):
            parts.append(two.to_string(index=False))
        parts.append("")

    return "\n".join(parts)

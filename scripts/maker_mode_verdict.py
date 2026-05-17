#!/usr/bin/env python3
"""
Phase 5.19 — Maker Mode A/B/C verdict report.

Aggregates per-mode performance for the multimode test:
  standard | patient | l2_aware

Metrics per mode:
  - sample size (n trades)
  - maker fill rate (% of trades where fee_type='maker')
  - avg slippage bps
  - NET P&L (sum, avg)
  - bootstrap 95% CI on maker fill rate

Decision rule (verdict):
  - If a mode's maker fill rate CI dominates others (lower bound > others' upper) → WINNER
  - If all CIs overlap → MORE DATA NEEDED
  - If patient/l2_aware both <5% maker → maker approach failed; consider Pre-Signal Engine

Usage:
  python3 scripts/maker_mode_verdict.py            # last 24h
  python3 scripts/maker_mode_verdict.py --hours 6  # last 6h
  python3 scripts/maker_mode_verdict.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


def _load_env(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v.strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _dsn():
    _load_env()
    if os.environ.get("DATABASE_URL", "").startswith("postgres"):
        return os.environ["DATABASE_URL"]
    return "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"


def bootstrap_ci(values: list, n_boot: int = 2000, alpha: float = 0.05) -> tuple:
    """Bootstrap 95% CI on a fraction (e.g. fill rate)."""
    import random
    if not values:
        return 0.0, 0.0
    rng = random.Random(42)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(len(means) * alpha / 2)]
    hi = means[int(len(means) * (1 - alpha / 2))]
    return lo, hi


async def fetch_per_mode(conn, hours: int):
    """
    Pull per-mode trade outcomes from the last `hours`.
    Joins trade-level fee_type with the maker_mode_used metadata field.
    """
    rows = await conn.fetch(f"""
        SELECT
          COALESCE(NULLIF(metadata::jsonb->>'maker_mode_used', ''), 'unknown') AS mode,
          COALESCE(NULLIF(metadata::jsonb->>'fee_type', ''), 'unknown') AS fill_type,
          COALESCE(NULLIF(metadata::jsonb->>'entry_exec_mode', ''), 'unknown') AS entry_exec,
          pnl_usd,
          symbol,
          side,
          (metadata::jsonb->>'peak_mfe_r')::numeric AS peak_r
        FROM user_trades
        WHERE trade_type IN ('real', 'shadow')
          AND closed_at >= NOW() - INTERVAL '{int(hours)} hours'
          AND pnl_usd IS NOT NULL
    """)
    by_mode = {}
    for r in rows:
        m = r["mode"]
        by_mode.setdefault(m, []).append({
            "fill_type": r["fill_type"],
            "entry_exec": r["entry_exec"],
            "pnl": float(r["pnl_usd"] or 0),
            "symbol": r["symbol"],
            "side": r["side"],
            "peak_r": float(r["peak_r"] or 0),
        })
    return by_mode


def stats_for_mode(trades: list) -> dict:
    """Aggregate per-mode stats."""
    n = len(trades)
    if n == 0:
        return {
            "n": 0, "maker_fills": 0, "maker_fill_rate_pct": 0,
            "maker_fill_rate_ci_low": 0, "maker_fill_rate_ci_high": 0,
            "net_pnl_usd": 0, "avg_pnl_usd": 0,
            "wins": 0, "losses": 0, "wr_pct": 0,
            "avg_peak_r": 0,
        }
    is_maker = [1 if t["fill_type"] == "maker" else 0 for t in trades]
    pnls = [t["pnl"] for t in trades]
    peaks = [t["peak_r"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    maker_count = sum(is_maker)
    ci_lo, ci_hi = bootstrap_ci(is_maker, n_boot=2000)
    return {
        "n": n,
        "maker_fills": maker_count,
        "maker_fill_rate_pct": round(100 * maker_count / n, 1),
        "maker_fill_rate_ci_low": round(100 * ci_lo, 1),
        "maker_fill_rate_ci_high": round(100 * ci_hi, 1),
        "net_pnl_usd": round(sum(pnls), 2),
        "avg_pnl_usd": round(sum(pnls) / n, 3),
        "wins": wins,
        "losses": losses,
        "wr_pct": round(100 * wins / n, 1),
        "avg_peak_r": round(sum(peaks) / n, 3),
    }


def issue_verdict(stats_by_mode: dict) -> str:
    """Compare modes and emit a verdict."""
    if not stats_by_mode:
        return "🟡 NO DATA — no labeled trades in window"

    modes_with_data = {m: s for m, s in stats_by_mode.items()
                       if m in ("standard", "patient", "l2_aware") and s["n"] > 0}
    if not modes_with_data:
        return "🟡 NO MULTIMODE DATA — admin may not be in 'multimode' yet, or no signals fired"

    # Sort by maker fill rate descending
    by_rate = sorted(modes_with_data.items(),
                     key=lambda x: -x[1]["maker_fill_rate_pct"])
    best_name, best = by_rate[0]
    best_rate = best["maker_fill_rate_pct"]
    best_lo = best["maker_fill_rate_ci_low"]

    # Best mode CI dominance check
    dominant = True
    for other_name, other in by_rate[1:]:
        if other["maker_fill_rate_ci_high"] > best_lo:
            dominant = False
            break

    if best_rate >= 20 and best["n"] >= 10:
        if dominant:
            return (f"🟢 PATIENT MAKER WORKING — `{best_name}` best at {best_rate:.0f}% "
                    f"maker fills (CI [{best_lo:.0f}, {best['maker_fill_rate_ci_high']:.0f}]). "
                    f"Promote winning mode to default → recommend Wave 2: "
                    f"5.18 + 5.9-C + live readiness audit.")
        return (f"🟡 PROMISING — `{best_name}` leads at {best_rate:.0f}% maker fills "
                f"but CIs overlap others. Need more samples (have {best['n']}) — wait 12+h.")

    if best_rate >= 5 and best["n"] >= 10:
        return (f"🟡 PARTIAL — `{best_name}` at {best_rate:.0f}% maker fills. "
                f"Below the 20% target. Consider widening patience further OR "
                f"jumping to Pre-Signal Engine refactor.")

    if best_rate < 5 and best["n"] >= 15:
        return (f"🔴 MAKER APPROACH FAILED — best mode `{best_name}` at {best_rate:.0f}% "
                f"maker fills across {best['n']} trades. Resting limits aren't filling. "
                f"Pre-Signal Engine (5.8) refactor is now URGENT — only structural fix.")

    return f"🟡 INSUFFICIENT DATA — best mode `{best_name}` has {best['n']} trades, need 10+"


def format_report(stats_by_mode: dict, hours: int, fix1b_explore: dict | None = None) -> str:
    lines = [
        "",
        "=" * 72,
        f"  MAKER MODE A/B/C VERDICT — last {hours}h",
        "=" * 72,
        f"  Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"  {'Mode':<12} {'n':>5} {'Maker%':>8} {'CI low':>8} {'CI high':>9} {'NET $':>10} {'Avg $':>9} {'WR%':>6} {'Peak R':>8}",
        "  " + "-" * 76,
    ]
    for mode in ("standard", "patient", "l2_aware"):
        s = stats_by_mode.get(mode)
        if not s or s["n"] == 0:
            lines.append(f"  {mode:<12} {'0':>5}  (no data)")
            continue
        lines.append(
            f"  {mode:<12} {s['n']:>5} "
            f"{s['maker_fill_rate_pct']:>7.1f}% "
            f"{s['maker_fill_rate_ci_low']:>7.1f}% "
            f"{s['maker_fill_rate_ci_high']:>8.1f}% "
            f"{s['net_pnl_usd']:>+10.2f} "
            f"{s['avg_pnl_usd']:>+8.3f}  "
            f"{s['wr_pct']:>5.1f}% "
            f"{s['avg_peak_r']:>7.3f}"
        )
    # Show 'unknown' bucket if any (legacy trades without mode tag)
    s_unk = stats_by_mode.get("unknown")
    if s_unk and s_unk["n"] > 0:
        lines.append(
            f"  {'unknown':<12} {s_unk['n']:>5} "
            f"{s_unk['maker_fill_rate_pct']:>7.1f}% "
            f"{'':>16} "
            f"{s_unk['net_pnl_usd']:>+10.2f} "
            f"{s_unk['avg_pnl_usd']:>+8.3f}  "
            f"{s_unk['wr_pct']:>5.1f}% "
            f"{s_unk['avg_peak_r']:>7.3f}  (legacy / single-mode)"
        )

    if fix1b_explore is not None:
        lines.extend([
            "",
            f"  Fix 1b 5% explore samples: n={fix1b_explore.get('n', 0)} "
            f"net=${fix1b_explore.get('net', 0):+.2f} "
            f"wins={fix1b_explore.get('wins', 0)}",
        ])

    lines.extend([
        "",
        "  VERDICT:",
        f"  {issue_verdict(stats_by_mode)}",
        "",
        "=" * 72,
    ])
    return "\n".join(lines)


async def fetch_fix1b_explore(conn, hours: int) -> dict:
    row = await conn.fetchrow(f"""
        SELECT
          COUNT(*) AS n,
          COALESCE(SUM(pnl_usd), 0) AS net,
          SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins
        FROM user_trades
        WHERE metadata::jsonb->>'fee_wall_explore' = 'true'
          AND closed_at >= NOW() - INTERVAL '{int(hours)} hours'
    """)
    return {"n": row["n"] or 0, "net": float(row["net"] or 0), "wins": row["wins"] or 0}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="Also write report to file (markdown)")
    args = ap.parse_args()

    conn = await asyncpg.connect(_dsn())
    try:
        per_mode = await fetch_per_mode(conn, args.hours)
        fix1b_explore = await fetch_fix1b_explore(conn, args.hours)
    finally:
        await conn.close()

    stats_by_mode = {m: stats_for_mode(t) for m, t in per_mode.items()}
    # Ensure all 3 expected modes present in output (even if empty)
    for m in ("standard", "patient", "l2_aware"):
        stats_by_mode.setdefault(m, stats_for_mode([]))

    if args.json:
        print(json.dumps({
            "hours": args.hours,
            "stats_by_mode": stats_by_mode,
            "fix1b_explore": fix1b_explore,
            "verdict": issue_verdict(stats_by_mode),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        return

    report = format_report(stats_by_mode, args.hours, fix1b_explore)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())

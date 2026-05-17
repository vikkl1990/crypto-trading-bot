#!/usr/bin/env python3
"""Net-aware scanner weights recalibration (gap A).

Existing scanner_weights.update_from_live_calibration uses GROSS R expectancy
(realized MFE / avg_exit_r). At Delta India taker fees this is misleading:
structure_bounce shows +0.153R gross but actually nets -$1.20/trade after fees.

This script reads the DB directly, computes per-scanner NET expectancy in R
units, and calls update_weights with NET. Runs daily via cron.

Risk denominator: trade.metadata.initial_risk if present, else margin × 0.0065.
Fees: included in pnl_usd directly (DB column already net-of-fees).

Output: rewrites storage/scanner_weights.json with net-driven status.
       Logs to /home/opc/crypto-trading-bot/logs/scanner_net_recalib.log

Usage: python3 scripts/scanner_net_recalibrate.py [--days 14] [--min-n 20]
Cron:  0 4 * * * cd /home/opc/crypto-trading-bot && python3 scripts/scanner_net_recalibrate.py
"""
from __future__ import annotations
import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone

import asyncio
import asyncpg

# Ensure the bot package is on path so we reuse ScannerWeightManager
sys.path.insert(0, "/home/opc/crypto-trading-bot")

from strategies.scanner_weights import ScannerWeightManager  # noqa: E402

DB_DSN = "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"
SL_PCT_FALLBACK = 0.0065  # 0.65% if metadata.initial_risk missing

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("scanner_net_recalib")


async def fetch_trades(days: int):
    """Pull closed shadow trades with per-scanner P&L data."""
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(f"""
        SELECT
            COALESCE(metadata::jsonb->>'scanner','') as scanner,
            pnl_usd,
            COALESCE((metadata::jsonb->>'initial_risk')::numeric, 0) as initial_risk,
            COALESCE((metadata::jsonb->>'margin')::numeric, 0) as margin,
            entry_price,
            quantity,
            COALESCE((metadata::jsonb->>'gross_pnl_usd')::numeric, 0) as gross_pnl,
            COALESCE((metadata::jsonb->>'peak_mfe_r')::numeric, 0) as peak_mfe_r
        FROM user_trades
        WHERE trade_type='shadow'
          AND closed_at >= NOW() - INTERVAL '{days} days'
          AND COALESCE(metadata::jsonb->>'exit_config_id','')='primary'
          AND status='closed'
          AND COALESCE(metadata::jsonb->>'exit_reason','') NOT IN
              ('auto_responder_stuck_60m','restart_orphan_cleanup','reconcile_overaged_close')
    """)
    await conn.close()
    return rows


def compute_net_metrics(rows, min_n: int):
    """Aggregate per-scanner net R metrics."""
    agg = defaultdict(lambda: {"n": 0, "net_r_sum": 0.0, "wins": 0,
                               "gross_r_sum": 0.0, "mfe_sum": 0.0})
    for r in rows:
        sc = r["scanner"]
        if not sc:
            continue
        # Determine 1R denominator
        risk = float(r["initial_risk"] or 0)
        if risk == 0:
            margin = float(r["margin"] or 0)
            risk = margin * SL_PCT_FALLBACK
        if risk == 0:
            entry = float(r["entry_price"] or 0)
            qty = float(r["quantity"] or 0)
            risk = entry * qty * SL_PCT_FALLBACK
        if risk == 0:
            continue
        net_r = float(r["pnl_usd"] or 0) / risk
        gross_r = float(r["gross_pnl"] or 0) / risk
        a = agg[sc]
        a["n"] += 1
        a["net_r_sum"] += net_r
        a["gross_r_sum"] += gross_r
        a["mfe_sum"] += float(r["peak_mfe_r"] or 0)
        if net_r > 0:
            a["wins"] += 1

    by_setup = {}
    for sc, a in agg.items():
        if a["n"] < min_n:
            continue
        by_setup[sc] = {
            "total": a["n"],
            "wins": a["wins"],
            "win_rate": round(100.0 * a["wins"] / a["n"], 2),
            "avg_r": round(a["net_r_sum"] / a["n"], 4),
            "total_r": round(a["net_r_sum"], 4),
            "expectancy_r": round(a["net_r_sum"] / a["n"], 4),  # NET expectancy
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "avg_mae_r": 0.0,
            "avg_mfe_r": round(a["mfe_sum"] / a["n"], 4),
            "_gross_expectancy_r": round(a["gross_r_sum"] / a["n"], 4),  # for log
        }
    return by_setup


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args()

    log.info(f"Net-aware recalib — last {args.days}d, min_n={args.min_n}")
    rows = await fetch_trades(args.days)
    log.info(f"Loaded {len(rows)} closed shadow trades")
    if not rows:
        log.warning("No data — abort")
        return 1

    by_setup = compute_net_metrics(rows, args.min_n)
    if not by_setup:
        log.warning(f"No scanner met min_n={args.min_n} — abort")
        return 1

    log.info("Per-scanner net vs gross expectancy:")
    log.info(f"  {'scanner':25s} {'n':>4} {'wins':>5} {'WR%':>6} "
             f"{'NET_exp':>9} {'GROSS_exp':>10} {'gap':>8}")
    for sc, m in sorted(by_setup.items(), key=lambda x: -x[1]["expectancy_r"]):
        net = m["expectancy_r"]
        gross = m["_gross_expectancy_r"]
        gap = net - gross
        log.info(f"  {sc:25s} {m['total']:>4} {m['wins']:>5} {m['win_rate']:>5.1f}% "
                 f"{net:>+8.3f}R {gross:>+9.3f}R {gap:>+7.3f}")

    # Strip the diagnostic field before passing to update_weights
    for m in by_setup.values():
        m.pop("_gross_expectancy_r", None)

    mgr = ScannerWeightManager()
    log.info("Calling update_weights with NET expectancy …")
    mgr.update_weights(by_setup)

    # Print resulting status
    log.info("Resulting scanner states:")
    for st in mgr.get_dashboard_summary():
        log.info(f"  {st['scanner']:25s} {st['status']:12s} weight={st['weight']:.2f} "
                 f"net_exp={st['expectancy_r']:+.3f}R reason={st['reason']}")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)

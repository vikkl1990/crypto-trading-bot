#!/usr/bin/env python3
"""
Phase 5.20-C — End-of-day P&L reconciliation.

Compares bot's internal P&L ledger to Delta's reported balance/positions.
Flags any drift > $1 OR > 0.1% of balance.

Cron:
    # Run at 00:05 UTC daily
    5 0 * * * cd /home/opc/crypto-trading-bot && python3 scripts/eod_reconcile.py

Today's MVP behavior:
    - Sums bot's user_trades.pnl_usd for the prior 24h
    - Logs the bot-side number to eod_recon table
    - Delta-side reconciliation requires real account access (live mode);
      currently captured as NULL with a note
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta, date
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


async def reconcile_day(conn, recon_date: date):
    """Sum bot P&L for given calendar day (UTC)."""
    start = datetime.combine(recon_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    row = await conn.fetchrow("""
        SELECT
          COUNT(*) AS n,
          COALESCE(SUM(pnl_usd), 0) AS pnl,
          COALESCE(SUM(fees_usd), 0) AS fees,
          COALESCE(SUM((metadata::jsonb->>'funding_usd')::numeric), 0) AS funding
        FROM user_trades
        WHERE closed_at >= $1 AND closed_at < $2
          AND trade_type IN ('real', 'shadow')
          AND pnl_usd IS NOT NULL
    """, start, end)

    bot_pnl = float(row["pnl"] or 0)
    fees = float(row["fees"] or 0)
    funding = float(row["funding"] or 0)
    n = int(row["n"] or 0)

    # Live Delta reconciliation — TODO when bot_mode='live' is active
    # For now, mark as NULL with note
    delta_pnl = None
    diff = None
    diff_pct = None
    notes = "delta_pnl=NULL (testnet/shadow only; live recon pending)"

    if delta_pnl is not None:
        diff = bot_pnl - delta_pnl
        diff_pct = (abs(diff) / max(abs(bot_pnl), 1)) * 100

    await conn.execute("""
        INSERT INTO eod_recon
          (recon_date, bot_pnl_usd, delta_pnl_usd, diff_usd, diff_pct,
           n_trades, fees_usd, funding_usd, notes)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (recon_date) DO UPDATE SET
          bot_pnl_usd = EXCLUDED.bot_pnl_usd,
          delta_pnl_usd = EXCLUDED.delta_pnl_usd,
          diff_usd = EXCLUDED.diff_usd,
          diff_pct = EXCLUDED.diff_pct,
          n_trades = EXCLUDED.n_trades,
          fees_usd = EXCLUDED.fees_usd,
          funding_usd = EXCLUDED.funding_usd,
          reconciled_at = NOW(),
          notes = EXCLUDED.notes
    """, recon_date, bot_pnl, delta_pnl, diff, diff_pct, n, fees, funding, notes)

    return {
        "date": recon_date.isoformat(),
        "n_trades": n,
        "bot_pnl_usd": bot_pnl,
        "delta_pnl_usd": delta_pnl,
        "fees_usd": fees,
        "funding_usd": funding,
        "diff": diff,
        "alert": diff is not None and (abs(diff) > 1.0 or (diff_pct or 0) > 0.1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: yesterday UTC)")
    args = ap.parse_args()

    if args.date:
        recon_date = date.fromisoformat(args.date)
    else:
        recon_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    conn = await asyncpg.connect(_dsn())
    try:
        result = await reconcile_day(conn, recon_date)
    finally:
        await conn.close()

    print(f"\n=== EOD Reconciliation — {result['date']} ===")
    print(f"  Trades:      {result['n_trades']}")
    print(f"  Bot P&L:     ${result['bot_pnl_usd']:+.2f}")
    print(f"  Delta P&L:   {'$'+format(result['delta_pnl_usd'],'+.2f') if result['delta_pnl_usd'] is not None else 'N/A (testnet/shadow)'}")
    print(f"  Fees:        ${result['fees_usd']:.2f}")
    print(f"  Funding:     ${result['funding_usd']:+.2f}")
    if result["diff"] is not None:
        print(f"  Diff:        ${result['diff']:+.2f}")
        if result["alert"]:
            print("  ⚠️  RECONCILIATION DRIFT — investigate")
    print()


if __name__ == "__main__":
    asyncio.run(main())

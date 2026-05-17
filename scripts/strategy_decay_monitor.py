#!/usr/bin/env python3
"""
Phase 5.20-C — Strategy decay monitor.

For each scanner, compute rolling Sharpe and WR over 7d / 30d / 90d windows.
If 7d Sharpe drops below 50% of 90d baseline → flag as DEGRADED.

Persists to strategy_decay table for trend analysis + dashboard.

Cron usage:
    # Run hourly
    0 * * * * cd /home/opc/crypto-trading-bot && python3 scripts/strategy_decay_monitor.py
"""
from __future__ import annotations

import asyncio
import math
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


async def fetch_window(conn, days: int, scanner: str = None):
    where_extra = f"AND metadata::jsonb->>'scanner' = '{scanner}'" if scanner else ""
    rows = await conn.fetch(f"""
        SELECT pnl_usd, opened_at, closed_at,
               COALESCE(metadata::jsonb->>'scanner','?') AS scanner
        FROM user_trades
        WHERE closed_at >= NOW() - INTERVAL '{int(days)} days'
          AND pnl_usd IS NOT NULL
          AND trade_type IN ('real','shadow')
          {where_extra}
    """)
    return rows


def compute_sharpe(daily_returns):
    if len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(365)  # crypto = 24/7


def aggregate_daily(rows):
    by_day = {}
    for r in rows:
        day = r["closed_at"].date()
        by_day.setdefault(day, 0.0)
        by_day[day] += float(r["pnl_usd"] or 0)
    return list(by_day.values())


async def measure_scanner(conn, scanner: str):
    rows7  = await fetch_window(conn, 7,  scanner)
    rows30 = await fetch_window(conn, 30, scanner)
    rows90 = await fetch_window(conn, 90, scanner)

    s7  = compute_sharpe(aggregate_daily(rows7))
    s30 = compute_sharpe(aggregate_daily(rows30))
    s90 = compute_sharpe(aggregate_daily(rows90))

    n7 = len(rows7)
    wr7 = (sum(1 for r in rows7 if r["pnl_usd"] > 0) / n7 * 100) if n7 else 0
    avg7 = (sum(float(r["pnl_usd"] or 0) for r in rows7) / n7) if n7 else 0

    # Degradation: 7d Sharpe < 50% of 90d AND we have meaningful samples
    sharpe_vs_baseline = (s7 / s90 * 100) if s90 != 0 else 100
    is_degraded = (
        n7 >= 10 and
        s90 > 0.5 and  # baseline must itself be decent
        s7 < s90 * 0.5
    )

    await conn.execute("""
        INSERT INTO strategy_decay
          (scanner, window_days, n_trades, win_rate_pct, avg_pnl_usd,
           sharpe_ann, sharpe_vs_baseline_pct, is_degraded)
        VALUES ($1, 7, $2, $3, $4, $5, $6, $7)
    """, scanner, n7, wr7, avg7, s7, sharpe_vs_baseline, is_degraded)

    return {
        "scanner": scanner,
        "n7": n7, "wr7": wr7, "avg7": avg7,
        "sharpe_7d": s7, "sharpe_30d": s30, "sharpe_90d": s90,
        "vs_baseline_pct": sharpe_vs_baseline,
        "degraded": is_degraded,
    }


async def main():
    conn = await asyncpg.connect(_dsn())
    try:
        # Get list of scanners present in last 30d
        scanners_rows = await conn.fetch("""
            SELECT DISTINCT metadata::jsonb->>'scanner' AS scanner
            FROM user_trades
            WHERE closed_at >= NOW() - INTERVAL '30 days'
              AND metadata::jsonb->>'scanner' IS NOT NULL
        """)
        scanners = [r["scanner"] for r in scanners_rows if r["scanner"]]

        print(f"\n=== Strategy Decay Monitor — {datetime.now(timezone.utc).isoformat()} ===\n")
        print(f"{'Scanner':<22} {'n7':>4} {'WR7%':>6} {'Avg7$':>8} {'Sh7':>6} {'Sh30':>6} {'Sh90':>6} {'vs90%':>7}  Status")
        print("-" * 90)

        any_degraded = False
        for sc in scanners:
            result = await measure_scanner(conn, sc)
            status = "🔴 DEGRADED" if result["degraded"] else "🟢 OK"
            if result["degraded"]:
                any_degraded = True
            print(
                f"{sc:<22} {result['n7']:>4} {result['wr7']:>5.1f}% "
                f"{result['avg7']:>+8.3f} {result['sharpe_7d']:>+6.2f} "
                f"{result['sharpe_30d']:>+6.2f} {result['sharpe_90d']:>+6.2f} "
                f"{result['vs_baseline_pct']:>6.0f}%  {status}"
            )

        if any_degraded:
            print("\n⚠️  AT LEAST ONE SCANNER DEGRADED — investigate or pause via kill switch")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())

"""
Calibrate the maker-miss probability from real demo trade data.

Reads user_trades where trade_type='real' and has metadata.entry_exec_mode.
`entry_exec_mode` values:
    maker_aggr_1bp / maker_aggr_2bp  → maker fill succeeded
    market_taker                     → maker miss, fell through to taker
    unknown / ""                     → skip

Returns:
    per-symbol miss rate = market_taker_count / total_count

Usage:
    python3 -m backtest.execution_replay.calibrate_maker_miss
    python3 -m backtest.execution_replay.calibrate_maker_miss --days 14
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_maker_miss")


def load_env(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v.strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _get_dsn():
    load_env()
    if os.environ.get("DATABASE_URL", "").startswith("postgres"):
        return os.environ["DATABASE_URL"]
    return "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"


async def calibrate(days: int = 30) -> Dict:
    """Query real trades and compute per-symbol maker miss rate."""
    conn = await asyncpg.connect(_get_dsn())
    try:
        # Source of truth for fill type: metadata.fee_type (stored by
        # user_real_manager._record_trade_db for every closed trade).
        # Values observed in production:
        #   'taker' — market order fill (crossed spread)
        #   'maker' — post_only limit filled at our price
        #   ''     — legacy pre-5.0 rows without fee_type (treat as unknown)
        rows = await conn.fetch(f"""
            SELECT
                symbol,
                COALESCE(NULLIF(metadata::jsonb->>'fee_type', ''), 'unknown') AS fill_type,
                COUNT(*) AS n
            FROM user_trades
            WHERE trade_type = 'real'
              AND opened_at >= NOW() - INTERVAL '{days} days'
            GROUP BY symbol, fill_type
            ORDER BY symbol, fill_type
        """)
    finally:
        await conn.close()

    # Aggregate by symbol
    per_symbol: Dict[str, Dict[str, int]] = {}
    for r in rows:
        sym = r["symbol"]
        fill_type = (r["fill_type"] or "unknown").lower()
        per_symbol.setdefault(sym, {"maker": 0, "taker": 0, "other": 0, "total": 0})
        if fill_type == "maker":
            per_symbol[sym]["maker"] += r["n"]
        elif fill_type == "taker":
            per_symbol[sym]["taker"] += r["n"]
        else:
            per_symbol[sym]["other"] += r["n"]
        per_symbol[sym]["total"] += r["n"]

    # Overall stats
    total_maker = sum(s["maker"] for s in per_symbol.values())
    total_taker = sum(s["taker"] for s in per_symbol.values())
    total_other = sum(s["other"] for s in per_symbol.values())
    total_all   = sum(s["total"] for s in per_symbol.values())

    overall_miss = total_taker / max(total_maker + total_taker, 1)

    return {
        "window_days": days,
        "total_trades": total_all,
        "total_maker_fills": total_maker,
        "total_taker_fills": total_taker,
        "total_other": total_other,
        "overall_miss_rate": overall_miss,
        "per_symbol": {
            sym: {
                "n_maker": v["maker"],
                "n_taker": v["taker"],
                "n_other": v["other"],
                "miss_rate": v["taker"] / max(v["maker"] + v["taker"], 1),
            }
            for sym, v in per_symbol.items()
        },
    }


def format_report(result: dict) -> str:
    lines = [
        "",
        "=" * 60,
        f"  MAKER MISS RATE CALIBRATION — last {result['window_days']}d",
        "=" * 60,
        f"  Total real trades:         {result['total_trades']}",
        f"  Maker fills:               {result['total_maker_fills']}",
        f"  Taker fills (maker miss):  {result['total_taker_fills']}",
        f"  Other/unknown:             {result['total_other']}",
        "",
        f"  OVERALL MAKER MISS RATE:   {result['overall_miss_rate']*100:.1f}%",
        "",
        f"  Per-symbol breakdown:",
        f"  {'symbol':<15} {'maker':>7} {'taker':>7} {'miss%':>8}",
        "  " + "-" * 44,
    ]
    by_miss = sorted(
        result["per_symbol"].items(),
        key=lambda x: -x[1]["miss_rate"],
    )
    for sym, d in by_miss:
        lines.append(
            f"  {sym:<15} {d['n_maker']:>7} {d['n_taker']:>7} "
            f"{d['miss_rate']*100:>7.1f}%"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    args = ap.parse_args()
    result = await calibrate(days=args.days)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    asyncio.run(main())

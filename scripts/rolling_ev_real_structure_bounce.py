#!/usr/bin/env python3
"""Rolling-EV Auto-Disable — REAL DATA backtest on structure_bounce.

Pulls actual closed structure_bounce trades from user_trades (last 30 days),
groups by symbol, simulates the rolling_ev rule.

Key difference vs synthetic backtest:
  - Real fills, real fees, real slippage already baked in
  - Real time clustering of trades (the bot doesn't trade evenly)
  - Real regime sequencing

Scanner extraction (correct version — handles Bybit-mirror path):
  COALESCE(
    NULLIF(metadata::jsonb->>'scanner',''),
    NULLIF(metadata::jsonb->>'setup_type',''),
    NULLIF(metadata::jsonb->>'source_engine',''),
    NULLIF(signal_data::jsonb->>'scanner','')   ← Bybit-mirror path
  )

Cells:
  last_N      ∈ {15, 20, 30, 50, 75, 100}
  threshold_$ ∈ {0.0, -0.10, -0.20, -0.30}
  reenable    ∈ {auto_when_recovered}        (skip permanent — too crude)
  → 24 cells per symbol

Symbols: any with >= 30 closed structure_bounce trades in window.

Verdict gates:
  - n_total < 30 → PARKED
  - lift/trade > +$0.10 AND n_skipped > 0 → PASS
  - lift > +$0.05 → STRONG_HOLD
  - lift > 0 → HOLD
  - lift <= 0 → KILL

Output: storage/wf_studies/rolling_ev_real_structure_bounce/{report.md, walkforward.json}
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path("/home/opc/crypto-trading-bot")
sys.path.insert(0, str(ROOT))

import psycopg2

DB_URL = os.environ.get("DATABASE_URL") or "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"

OUT_DIR = ROOT / "storage" / "wf_studies" / "rolling_ev_real_structure_bounce"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAST_N_GRID = [15, 20, 30, 50, 75, 100]
THRESHOLD_GRID = [0.0, -0.10, -0.20, -0.30]
REENABLE = "auto_when_recovered"

N_TOTAL_MIN = 30
PASS_LIFT_FLOOR = 0.10
STRONG_HOLD_LIFT_FLOOR = 0.05

DAYS_BACK = 30


def fetch_trades() -> List[Dict[str, Any]]:
    sql = """
        SELECT
          symbol,
          opened_at,
          closed_at,
          pnl_usd,
          COALESCE(NULLIF(metadata::jsonb->>'scanner',''),
                    NULLIF(metadata::jsonb->>'setup_type',''),
                    NULLIF(metadata::jsonb->>'source_engine',''),
                    NULLIF(signal_data::jsonb->>'scanner','')) as scanner,
          side,
          trade_type
        FROM user_trades
        WHERE closed_at IS NOT NULL
          AND closed_at >= NOW() - INTERVAL '%s days'
          AND trade_type IN ('shadow', 'real')
          AND COALESCE(NULLIF(metadata::jsonb->>'scanner',''),
                        NULLIF(metadata::jsonb->>'setup_type',''),
                        NULLIF(metadata::jsonb->>'source_engine',''),
                        NULLIF(signal_data::jsonb->>'scanner','')) = 'structure_bounce'
        ORDER BY closed_at ASC
    """ % DAYS_BACK
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for sym, op, cl, pnl, sc, sd, tt in rows:
        if pnl is None:
            continue
        out.append({
            "symbol": sym, "scanner": sc, "side": sd, "trade_type": tt,
            "opened_at": op, "closed_at": cl, "pnl_usd": float(pnl),
        })
    return out


@dataclass
class CellResult:
    symbol: str
    last_n: int
    threshold: float
    reenable: str
    n_total: int = 0
    n_skipped: int = 0
    pnl_baseline: float = 0.0
    pnl_with_rule: float = 0.0
    lift_per_trade: float = 0.0
    lift_total: float = 0.0
    pct_skipped: float = 0.0
    max_disable_streak: int = 0
    verdict: str = "PARKED"
    n_decisions_disabled: int = 0


def simulate_cell(trades: List[Dict[str, Any]], symbol: str,
                  last_n: int, threshold: float) -> CellResult:
    cell = CellResult(symbol=symbol, last_n=last_n, threshold=threshold, reenable=REENABLE)
    cell.n_total = len(trades)
    if len(trades) < N_TOTAL_MIN:
        return cell

    pnl_baseline = 0.0
    pnl_with = 0.0
    n_skipped = 0
    cur_streak = 0
    max_streak = 0
    history: List[float] = []

    for t in trades:
        pnl_baseline += t["pnl_usd"]
        if len(history) >= last_n:
            roll = float(np.mean(history[-last_n:]))
        else:
            roll = None
        if roll is not None and roll < threshold:
            n_skipped += 1
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
            # skipped ⇒ $0 contribution
        else:
            pnl_with += t["pnl_usd"]
            cur_streak = 0
        history.append(t["pnl_usd"])

    cell.n_skipped = n_skipped
    cell.pnl_baseline = round(pnl_baseline, 4)
    cell.pnl_with_rule = round(pnl_with, 4)
    cell.lift_total = round(pnl_with - pnl_baseline, 4)
    cell.lift_per_trade = round((pnl_with - pnl_baseline) / len(trades), 4)
    cell.pct_skipped = round(100.0 * n_skipped / len(trades), 2)
    cell.max_disable_streak = max_streak

    if cell.lift_per_trade > PASS_LIFT_FLOOR and cell.n_skipped > 0:
        cell.verdict = "PASS"
    elif cell.lift_per_trade > STRONG_HOLD_LIFT_FLOOR and cell.n_skipped > 0:
        cell.verdict = "STRONG_HOLD"
    elif cell.lift_per_trade > 0 and cell.n_skipped > 0:
        cell.verdict = "HOLD"
    else:
        cell.verdict = "KILL"
    return cell


def main():
    print(f"=== rolling_ev_real_structure_bounce — {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Pulling last {DAYS_BACK} days of closed structure_bounce trades from user_trades...")
    trades = fetch_trades()
    print(f"  pulled {len(trades)} trades")
    if not trades:
        print("  no data; abort")
        sys.exit(0)

    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)

    print(f"  symbols with data: {len(by_sym)}")
    print(f"  sample: " + ", ".join(f"{s}={len(v)}" for s, v in
                                     sorted(by_sym.items(), key=lambda kv: -len(kv[1]))[:6]))

    rows: List[CellResult] = []
    for sym, sym_trades in by_sym.items():
        for last_n, thr in product(LAST_N_GRID, THRESHOLD_GRID):
            cell = simulate_cell(sym_trades, sym, last_n, thr)
            rows.append(cell)

    # Save JSON
    summary = {
        "study": "rolling_ev_real_structure_bounce",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "window_days": DAYS_BACK,
        "n_total_trades": len(trades),
        "n_symbols": len(by_sym),
        "grid": {
            "last_n": LAST_N_GRID,
            "threshold_$": THRESHOLD_GRID,
            "reenable": REENABLE,
        },
        "per_symbol_n": {s: len(v) for s, v in by_sym.items()},
        "rows": [asdict(r) for r in rows],
    }
    (OUT_DIR / "walkforward.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {OUT_DIR}/walkforward.json")

    # Markdown
    md = [
        f"# Rolling-EV Auto-Disable — REAL DATA on structure_bounce",
        "",
        f"Generated: {summary['started_at']}",
        f"Window: last {DAYS_BACK} days   Trades: {len(trades)}   Symbols: {len(by_sym)}",
        "",
        "## Setup",
        "- Source: user_trades (REAL live shadow/real closed trades)",
        "- Scanner extraction: metadata.scanner OR setup_type OR source_engine OR signal_data.scanner",
        "- Re-enable: stateless (auto_when_recovered)",
        "",
        "## Per-symbol trade volume + baseline EV",
        "",
        "| symbol | n | baseline_pnl | avg_pnl/trade |",
        "|---|---|---|---|",
    ]
    for sym, syt in sorted(by_sym.items(), key=lambda kv: -len(kv[1])):
        bp = sum(t["pnl_usd"] for t in syt)
        md.append(f"| {sym} | {len(syt)} | ${bp:+.2f} | ${bp/len(syt):+.4f} |")

    md += ["",
           "## Best cell per symbol (n >= 30)",
           "",
           "| symbol | last_N | thr_$ | n_total | n_skip | pct_skip | base_pnl | with_rule | lift/trade | lift_total | max_streak | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    by_sym_cells: Dict[str, List[CellResult]] = defaultdict(list)
    for r in rows:
        by_sym_cells[r.symbol].append(r)
    for sym in sorted(by_sym_cells.keys(), key=lambda s: -len(by_sym[s])):
        cells = by_sym_cells[sym]
        if not cells: continue
        if cells[0].n_total < N_TOTAL_MIN:
            md.append(f"| {sym} | — | — | {cells[0].n_total} | — | — | — | — | — | — | — | PARKED |")
            continue
        best = max(cells, key=lambda c: (c.lift_per_trade, c.lift_total))
        md.append(
            f"| {sym} | {best.last_n} | ${best.threshold:+.2f} | {best.n_total} | "
            f"{best.n_skipped} | {best.pct_skipped}% | "
            f"${best.pnl_baseline:+.2f} | ${best.pnl_with_rule:+.2f} | "
            f"${best.lift_per_trade:+.4f} | ${best.lift_total:+.2f} | "
            f"{best.max_disable_streak} | **{best.verdict}** |"
        )

    # Top 20 cells across all symbols (by lift_total)
    md += ["", "## Top 20 cells by lift_total (across all symbols)", "",
           "| symbol | last_N | thr_$ | n_skip | lift/trade | lift_total | verdict |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.lift_total, reverse=True)[:20]:
        md.append(
            f"| {r.symbol} | {r.last_n} | ${r.threshold:+.2f} | "
            f"{r.n_skipped} ({r.pct_skipped}%) | "
            f"${r.lift_per_trade:+.4f} | ${r.lift_total:+.2f} | {r.verdict} |"
        )

    md += ["", "## Verdict count", "", "| Verdict | Count |", "|---|---|"]
    vc: Dict[str, int] = {}
    for r in rows:
        vc[r.verdict] = vc.get(r.verdict, 0) + 1
    for v, n in sorted(vc.items(), key=lambda x: -x[1]):
        md.append(f"| {v} | {n} |")

    # Aggregate roll-up: total $ saved if best cell applied per symbol
    total_lift = 0.0
    total_baseline = 0.0
    md += ["", "## Aggregate (apply best cell per symbol with n>=30)", ""]
    md.append("| symbol | baseline_pnl | with_rule | lift |")
    md.append("|---|---|---|---|")
    for sym in sorted(by_sym_cells.keys(), key=lambda s: -len(by_sym[s])):
        cells = by_sym_cells[sym]
        if cells[0].n_total < N_TOTAL_MIN: continue
        best = max(cells, key=lambda c: (c.lift_per_trade, c.lift_total))
        md.append(f"| {sym} | ${best.pnl_baseline:+.2f} | ${best.pnl_with_rule:+.2f} | ${best.lift_total:+.2f} |")
        total_baseline += best.pnl_baseline
        total_lift += best.lift_total
    md.append(f"| **TOTAL** | **${total_baseline:+.2f}** | **${total_baseline+total_lift:+.2f}** | **${total_lift:+.2f}** |")

    (OUT_DIR / "report.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {OUT_DIR}/report.md")
    print(f"\n=== aggregate lift across all qualifying symbols: ${total_lift:+.2f} ===")
    print(f"=== baseline total: ${total_baseline:+.2f}  →  with rule: ${total_baseline+total_lift:+.2f} ===")


if __name__ == "__main__":
    main()

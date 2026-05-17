#!/usr/bin/env python3
"""Rolling-EV Auto-Disable — W/F backtest of architect's circuit-breaker spec.

Spec (architect, 2026-05-03):
  rolling_ev = mean(net_pnl of last_N trades for that scanner+symbol)
  if rolling_ev < threshold:
      auto_disable(scanner, symbol)

Backtest design:
  1. Re-generate trades for each (scanner, symbol) using the same detectors
     + baseline exit policies as the exit W/F runners shipped today.
  2. Walk trades chronologically. After each trade, recompute rolling_ev for
     that (scanner, symbol).
  3. For each cell — (last_N, threshold, reenable_method) — compute:
        - n_trades_total
        - n_trades_skipped (would have been blocked)
        - pnl_with_rule    (skipped trades = $0; kept trades = real net)
        - pnl_without_rule (baseline: take all trades)
        - lift_per_trade   = (pnl_with - pnl_without) / n_trades_total
        - max_disable_streak (longest stretch where scanner was off)
  4. PASS/HOLD/KILL verdict per (scanner, symbol, cell).

Re-enable methods (sensitivity):
  - "permanent"          : once disabled, stay disabled forever (worst-case)
  - "auto_when_recovered": stateless — re-check rolling_ev on every new
                            signal; trade only if rolling_ev >= threshold
  - "n_quiet_bars"       : after disable, wait N bars of "would-have-traded"
                            before re-enabling (TBD — defer for now)

We test "permanent" + "auto_when_recovered". The architect's spec maps to
"permanent" most literally — but the bot's natural behavior is closer to
"auto_when_recovered" (rolling window naturally recovers as bad trades age
out of the last_N window).

Cells:
  last_N      ∈ {15, 20, 30, 50}
  threshold_$ ∈ {0.0, -0.10, -0.20}
  reenable    ∈ {permanent, auto_when_recovered}
  → 24 cells per (scanner, symbol)

Scanners + symbols (using existing detector implementations):
  - absorption_bubble × ETH        (~116 trades; from exit W/F #1)
  - scalper_vwap_mr × {BTC, ETH}   (~1300 each; from exit W/F #2)
  - liq_grab+sweep_htf × SOL       (~1143 trades; from exit W/F #3)

Output: storage/wf_studies/rolling_ev_autodisable/{report.md, walkforward.json}

Verdict gates (per scanner-symbol-cell):
  - n_total < 30 → PARKED
  - lift_per_trade > +$0.05 AND n_skipped > 0 → PASS  (the rule actually saved money)
  - lift_per_trade > 0 AND n_skipped > 0 → HOLD       (positive but small)
  - lift_per_trade <= 0 → KILL                         (rule cost money or no effect)
"""
from __future__ import annotations

import json
import sys
import time as _time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exit_policy_harness import (  # noqa: E402
    Signal, ExitPolicy, ExitOutcome, evaluate_signal_through_policies,
    load_candles, fee_for_trade, compute_mfe_mae,
)
from exit_wf_absorption_fast import (  # noqa: E402
    generate_absorption_signals, AbsorptionBaseline,
)
from exit_wf_vwap_touch import (  # noqa: E402
    generate_vwap_mr_signals, VwapMrBaseline,
)
from exit_wf_tp1_be_trail import (  # noqa: E402
    generate_liq_grab_sweep_signals, LiqGrabBaseline,
)


OUT_DIR = ROOT / "storage" / "wf_studies" / "rolling_ev_autodisable"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAST_N_GRID = [15, 20, 30, 50]
THRESHOLD_GRID = [0.0, -0.10, -0.20]
REENABLE_GRID = ["permanent", "auto_when_recovered"]

NOTIONAL = 1000.0
EVAL_WINDOW_BARS = 30
TF_MIN = 5

PASS_LIFT_FLOOR = 0.05    # +$0.05/trade lift to call PASS
N_TOTAL_MIN = 30


# ─────────────────────────────────────────────────────────────────────
# Per-(scanner, symbol) trade generators
# ─────────────────────────────────────────────────────────────────────
GENERATORS: List[Dict[str, Any]] = [
    {
        "scanner": "absorption_bubble", "symbol": "ETH",
        "gen": generate_absorption_signals,
        "policy": AbsorptionBaseline(),
    },
    {
        "scanner": "scalper_vwap_mr", "symbol": "BTC",
        "gen": generate_vwap_mr_signals,
        "policy": VwapMrBaseline(),
    },
    {
        "scanner": "scalper_vwap_mr", "symbol": "ETH",
        "gen": generate_vwap_mr_signals,
        "policy": VwapMrBaseline(),
    },
    {
        "scanner": "liq_grab_ob_fvg+liq_sweep_htf", "symbol": "SOL",
        "gen": generate_liq_grab_sweep_signals,
        "policy": LiqGrabBaseline(),
    },
]


# ─────────────────────────────────────────────────────────────────────
# Build per-trade ledger for each (scanner, symbol)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LedgerTrade:
    scanner: str
    symbol: str
    entry_ts: pd.Timestamp
    side: str
    net_pnl: float

def build_ledger(scanner: str, symbol: str,
                 gen: Callable[[pd.DataFrame, str], List[Signal]],
                 policy: ExitPolicy) -> List[LedgerTrade]:
    df = load_candles(symbol, "5m")
    df.attrs["symbol"] = symbol
    sigs = gen(df, symbol)
    out: List[LedgerTrade] = []
    for s in sigs:
        # Use the harness helper to compute the trade outcome at baseline policy
        trade_results = evaluate_signal_through_policies(
            df, s, [policy], scanner, TF_MIN, EVAL_WINDOW_BARS, NOTIONAL,
        )
        if not trade_results:
            continue
        t = trade_results[0]
        out.append(LedgerTrade(
            scanner=scanner, symbol=symbol, entry_ts=s.entry_ts,
            side=s.side, net_pnl=float(t.net_usd),
        ))
    out.sort(key=lambda r: r.entry_ts)
    return out


# ─────────────────────────────────────────────────────────────────────
# Rolling-EV auto-disable simulator
# ─────────────────────────────────────────────────────────────────────
@dataclass
class CellResult:
    scanner: str
    symbol: str
    last_n: int
    threshold: float
    reenable: str
    n_total: int = 0
    n_skipped: int = 0
    pnl_with_rule: float = 0.0
    pnl_without_rule: float = 0.0
    lift_per_trade: float = 0.0
    pnl_lift_total: float = 0.0
    max_disable_streak: int = 0
    pct_skipped: float = 0.0
    verdict: str = "PARKED"


def simulate_cell(trades: List[LedgerTrade], last_n: int, threshold: float,
                  reenable: str) -> CellResult:
    """Walk trades in time order. Apply rule. Return cell stats."""
    cell = CellResult(scanner=trades[0].scanner if trades else "?",
                      symbol=trades[0].symbol if trades else "?",
                      last_n=last_n, threshold=threshold, reenable=reenable)
    if len(trades) < N_TOTAL_MIN:
        cell.n_total = len(trades)
        return cell

    pnl_baseline = 0.0
    pnl_with = 0.0
    n_skipped = 0
    disabled = False
    cur_streak = 0
    max_streak = 0
    history: List[float] = []
    for t in trades:
        pnl_baseline += t.net_pnl
        # Compute rolling EV from history (BEFORE this trade)
        if len(history) >= last_n:
            roll = float(np.mean(history[-last_n:]))
        else:
            roll = None
        # Decision: trade or skip?
        if reenable == "auto_when_recovered":
            # Stateless: skip if rolling_ev < threshold (only when we have enough history)
            if roll is not None and roll < threshold:
                skip = True
            else:
                skip = False
        else:  # permanent
            if not disabled and roll is not None and roll < threshold:
                disabled = True
            skip = disabled
        if skip:
            n_skipped += 1
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
            # Skipped trades: $0 contribution to pnl_with
        else:
            pnl_with += t.net_pnl
            cur_streak = 0
        history.append(t.net_pnl)

    cell.n_total = len(trades)
    cell.n_skipped = n_skipped
    cell.pnl_with_rule = round(pnl_with, 4)
    cell.pnl_without_rule = round(pnl_baseline, 4)
    cell.pnl_lift_total = round(pnl_with - pnl_baseline, 4)
    cell.lift_per_trade = round((pnl_with - pnl_baseline) / len(trades), 4)
    cell.max_disable_streak = max_streak
    cell.pct_skipped = round(100.0 * n_skipped / len(trades), 2)

    # Verdict
    if cell.n_total < N_TOTAL_MIN:
        cell.verdict = "PARKED"
    elif cell.lift_per_trade > PASS_LIFT_FLOOR and cell.n_skipped > 0:
        cell.verdict = "PASS"
    elif cell.lift_per_trade > 0 and cell.n_skipped > 0:
        cell.verdict = "HOLD"
    else:
        cell.verdict = "KILL"
    return cell


# ─────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=== rolling_ev_autodisable W/F ===")
    t0 = _time.time()
    rows: List[CellResult] = []

    for g in GENERATORS:
        scanner = g["scanner"]; symbol = g["symbol"]
        print(f"\n[{scanner}|{symbol}] building ledger...")
        ledger = build_ledger(scanner, symbol, g["gen"], g["policy"])
        print(f"[{scanner}|{symbol}] {len(ledger)} trades in ledger")

        for last_n, thr, reen in product(LAST_N_GRID, THRESHOLD_GRID, REENABLE_GRID):
            cell = simulate_cell(ledger, last_n, thr, reen)
            rows.append(cell)

    wall = round(_time.time() - t0, 1)
    print(f"\n=== DONE ({wall}s)  cells={len(rows)} ===")

    # Save
    summary = {
        "study": "rolling_ev_autodisable",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "wall_sec": wall,
        "scanners_tested": [{"scanner": g["scanner"], "symbol": g["symbol"]} for g in GENERATORS],
        "grid": {
            "last_n": LAST_N_GRID,
            "threshold_$": THRESHOLD_GRID,
            "reenable": REENABLE_GRID,
        },
        "rows": [asdict(r) for r in rows],
    }
    (OUT_DIR / "walkforward.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {OUT_DIR}/walkforward.json")

    # Markdown report
    md = [
        "# Rolling-EV Auto-Disable — Walk-Forward Report",
        "",
        f"Generated: {summary['started_at']}",
        f"Wall: {wall}s   Cells: {len(rows)}",
        "",
        "## Spec",
        "- `rolling_ev = mean(net_pnl) over last_N trades for (scanner, symbol)`",
        "- `if rolling_ev < threshold: skip next trade for that (scanner, symbol)`",
        "- Re-enable: `permanent` (architect spec) OR `auto_when_recovered` (stateless re-check)",
        "",
        "## Verdict gates",
        f"- n_total < {N_TOTAL_MIN} → PARKED",
        f"- lift_per_trade > +${PASS_LIFT_FLOOR} AND n_skipped > 0 → PASS",
        "- lift > 0 AND n_skipped > 0 → HOLD",
        "- otherwise → KILL",
        "",
        "## Best cell per (scanner × symbol)",
        "",
        "| scanner | symbol | last_N | thr_$ | reenable | n_total | n_skip | pnl_baseline | pnl_with | lift/trade | lift_total | max_streak | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    # Group by (scanner, symbol), pick best by lift/trade
    by_pair: Dict[tuple, List[CellResult]] = {}
    for r in rows:
        by_pair.setdefault((r.scanner, r.symbol), []).append(r)
    for (sc, sy), cells in by_pair.items():
        best = max(cells, key=lambda c: (c.lift_per_trade, c.pnl_lift_total))
        md.append(
            f"| {sc} | {sy} | {best.last_n} | ${best.threshold:+.2f} | {best.reenable} | "
            f"{best.n_total} | {best.n_skipped} ({best.pct_skipped}%) | "
            f"${best.pnl_without_rule:+.2f} | ${best.pnl_with_rule:+.2f} | "
            f"${best.lift_per_trade:+.4f} | ${best.pnl_lift_total:+.2f} | "
            f"{best.max_disable_streak} | **{best.verdict}** |"
        )

    md += ["", "## All cells (sorted by lift/trade desc)", "",
           "| scanner | symbol | last_N | thr_$ | reenable | n_skip | lift/trade | verdict |",
           "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.lift_per_trade, reverse=True):
        md.append(
            f"| {r.scanner} | {r.symbol} | {r.last_n} | ${r.threshold:+.2f} | "
            f"{r.reenable} | {r.n_skipped} ({r.pct_skipped}%) | "
            f"${r.lift_per_trade:+.4f} | {r.verdict} |"
        )

    md += ["", "## Verdict count", "", "| Verdict | Count |", "|---|---|"]
    vc: Dict[str, int] = {}
    for r in rows:
        vc[r.verdict] = vc.get(r.verdict, 0) + 1
    for v, n in sorted(vc.items(), key=lambda x: -x[1]):
        md.append(f"| {v} | {n} |")

    (OUT_DIR / "report.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-symbol Sweep Confluence — W/F study (candidate #4, 2026-05-03).

Hypothesis: a sweep on one major (BTC/ETH/SOL) has more edge if 1+ other
majors show same-side sweep within a short window (confluent macro flow).

Method:
  1. Pre-detect all sweep events on each of BTC, ETH, SOL (5m bars):
       LONG sweep  = bar low penetrates prior 20-bar low  by ≥ sweep_atr × ATR(14)
       SHORT sweep = bar high penetrates prior 20-bar high by ≥ sweep_atr × ATR(14)
  2. For each cell, define ADMITTED sweeps = those with at least
     `confluence_count_other` SAME-SIDE sweep events on OTHER majors within
     ±`confluence_window_min` minutes.
  3. Each admitted sweep → trade at the sweep candle's close. Exit:
       SL = sweep extreme ± 0.3 × ATR
       TP = RR=1.5
       Time stop = 30 min
  4. Score per cell × per fee variant.

Cells:
  sweep_atr ∈ {0.3, 0.5}
  confluence_window_min ∈ {10, 15, 30}
  confluence_count_other ∈ {1, 2}    # 1 = at least 1 other; 2 = both others
  → 12 cells × 3 syms × 3 fee variants

A BASELINE cell (no-confluence required) is also reported as `baseline_no_conf`.
This lets us see whether confluence ADDS edge vs. raw sweeps.

Output: storage/wf_studies/multi_symbol_sweep_confluence/{walkforward.json,report.md}
"""
from __future__ import annotations

import json
import math
import sys
import time as _time
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wf_harness import (  # noqa: E402
    Trade, WalkForwardEngine, FEE_VARIANTS, ROOT as HARNESS_ROOT,
    PASS_OOS_GAP_MAX, PASS_Q4_EV_MIN, PASS_IS_EV_MIN,
)

ATR_PERIOD = 14
SWEEP_LOOKBACK = 20
TIME_STOP_MIN = 30
TP_RR = 1.5
SL_BUFFER_ATR = 0.3
NOTIONAL = 1000.0

SYMBOLS = ["BTC", "ETH", "SOL"]
TF = "5m"
TF_MIN = 5


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h = df["high"].astype(float); l = df["low"].astype(float)
    c = df["close"].astype(float).shift(1)
    tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def detect_sweeps(df: pd.DataFrame, sweep_atr: float) -> List[Dict[str, Any]]:
    """Return list of {idx, ts, side, sweep_extreme} for each detected sweep."""
    out: List[Dict[str, Any]] = []
    if df.empty or len(df) < ATR_PERIOD + SWEEP_LOOKBACK + 5:
        return out
    atr = add_atr(df, ATR_PERIOD)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    for i in range(ATR_PERIOD + SWEEP_LOOKBACK + 1, n - 1):
        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        win_lo = float(np.min(lows[i - SWEEP_LOOKBACK: i]))
        win_hi = float(np.max(highs[i - SWEEP_LOOKBACK: i]))
        if (win_lo - lows[i]) >= sweep_atr * atr_i:
            out.append({"idx": i, "ts": df.index[i], "side": "long",
                        "sweep_extreme": float(lows[i]), "atr": atr_i})
        if (highs[i] - win_hi) >= sweep_atr * atr_i:
            out.append({"idx": i, "ts": df.index[i], "side": "short",
                        "sweep_extreme": float(highs[i]), "atr": atr_i})
    return out


def has_confluence(target: Dict[str, Any], other_sweeps: List[Dict[str, Any]],
                   window_min: int, count_required: int) -> bool:
    """Check if `count_required` other-symbol same-side sweeps exist within ±window_min."""
    matches = 0
    seen_syms = set()
    target_ts = target["ts"]
    target_side = target["side"]
    window = pd.Timedelta(minutes=window_min)
    for o in other_sweeps:
        if o["side"] != target_side:
            continue
        if abs(o["ts"] - target_ts) > window:
            continue
        sym = o["symbol"]
        if sym in seen_syms:
            continue
        seen_syms.add(sym)
        matches += 1
        if matches >= count_required:
            return True
    return matches >= count_required


def simulate_trade(
    df: pd.DataFrame, idx: int, side: str, sweep_extreme: float,
    atr_at_signal: float, symbol: str,
) -> Trade:
    entry_price = float(df["close"].iloc[idx])
    if side == "long":
        sl = sweep_extreme - SL_BUFFER_ATR * atr_at_signal
        risk = entry_price - sl
        tp = entry_price + TP_RR * max(risk, 1e-9)
    else:
        sl = sweep_extreme + SL_BUFFER_ATR * atr_at_signal
        risk = sl - entry_price
        tp = entry_price - TP_RR * max(risk, 1e-9)

    bars_max = max(1, int(math.ceil(TIME_STOP_MIN / TF_MIN)))
    end_idx = min(idx + bars_max, len(df) - 1)
    exit_reason = "time_stop"
    exit_price = float(df["close"].iloc[end_idx])
    exit_idx = end_idx
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    for j in range(idx + 1, end_idx + 1):
        bh = float(highs[j]); bl = float(lows[j])
        if side == "long":
            if bl <= sl:
                exit_reason = "sl"; exit_price = sl; exit_idx = j; break
            if bh >= tp:
                exit_reason = "tp"; exit_price = tp; exit_idx = j; break
        else:
            if bh >= sl:
                exit_reason = "sl"; exit_price = sl; exit_idx = j; break
            if bl <= tp:
                exit_reason = "tp"; exit_price = tp; exit_idx = j; break
    holding_sec = max(1, (exit_idx - idx) * TF_MIN * 60)
    return Trade(
        symbol=symbol, side=side,
        entry_price=entry_price, exit_price=exit_price,
        notional_usd=NOTIONAL, holding_sec=holding_sec,
        entry_ts=df.index[idx], exit_ts=df.index[exit_idx],
        exit_reason=exit_reason,
    )


def main():
    print("=== multi_symbol_sweep_confluence W/F ===")
    out_dir = HARNESS_ROOT / "storage" / "wf_studies" / "multi_symbol_sweep_confluence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # We'll instantiate WalkForwardEngine purely to use its aggregate()/apply_fees()/quarter_of()
    class _NullStrategy:
        name = "multi_symbol_sweep_confluence"
        def param_grid(self): return iter([])
        def simulate(self, df, p): return []
        def cell_id(self, p): return ""
    engine = WalkForwardEngine(
        study=_NullStrategy(),  # type: ignore[arg-type]
        symbols=SYMBOLS, timeframes=[TF], out_dir=out_dir, verbose=False,
    )

    # Load + pre-detect sweeps for each symbol × each sweep_atr value
    sweep_atr_grid = [0.3, 0.5]
    conf_window_grid = [10, 15, 30]
    conf_count_grid = [1, 2]   # 1 = ≥1 other; 2 = both others

    candles: Dict[str, pd.DataFrame] = {}
    sweeps_by_atr: Dict[float, Dict[str, List[Dict[str, Any]]]] = {}
    t0 = _time.time()

    for sym in SYMBOLS:
        df = engine.load_candles(sym, TF)
        candles[sym] = df
        print(f"  loaded {sym}: {len(df)} bars")

    for atr_v in sweep_atr_grid:
        sweeps_by_atr[atr_v] = {}
        total = 0
        for sym in SYMBOLS:
            sw = detect_sweeps(candles[sym], atr_v)
            for s in sw:
                s["symbol"] = sym
            sweeps_by_atr[atr_v][sym] = sw
            total += len(sw)
        print(f"  sweep_atr={atr_v}: total sweeps = {total} ({sum(len(s) for s in sweeps_by_atr[atr_v].values())})")

    # For each cell (sweep_atr × conf_window × conf_count) plus a baseline (no confluence)
    cells_results: Dict[str, Dict[str, Any]] = {}

    def cell_key(atr_v, win, cnt, label):
        return f"sweep_atr={atr_v}|conf_win={win}min|conf_cnt={cnt}|{label}"

    for atr_v in sweep_atr_grid:
        # Baseline (no confluence required)
        base_trades_by_sym = {}
        for sym in SYMBOLS:
            tlist = []
            for s in sweeps_by_atr[atr_v][sym]:
                tlist.append(simulate_trade(
                    candles[sym], s["idx"], s["side"],
                    s["sweep_extreme"], s["atr"], sym,
                ))
            base_trades_by_sym[sym] = tlist
        baseline_trades = [t for sym in SYMBOLS for t in base_trades_by_sym[sym]]
        for variant in FEE_VARIANTS.keys():
            res = engine.aggregate(baseline_trades, variant)
            cells_results[f"sweep_atr={atr_v}|baseline_no_conf|{variant}"] = asdict(res)

        # Confluence cells
        for win, cnt in product(conf_window_grid, conf_count_grid):
            cell_trades = []
            n_admitted_per_sym: Dict[str, int] = {s: 0 for s in SYMBOLS}
            for sym in SYMBOLS:
                others = [s for o in SYMBOLS if o != sym for s in sweeps_by_atr[atr_v][o]]
                for s in sweeps_by_atr[atr_v][sym]:
                    if not has_confluence(s, others, win, cnt):
                        continue
                    n_admitted_per_sym[sym] += 1
                    cell_trades.append(simulate_trade(
                        candles[sym], s["idx"], s["side"],
                        s["sweep_extreme"], s["atr"], sym,
                    ))
            for variant in FEE_VARIANTS.keys():
                res = engine.aggregate(cell_trades, variant)
                key = cell_key(atr_v, win, cnt, "conf") + f"|{variant}"
                d = asdict(res)
                d["admitted_per_sym"] = n_admitted_per_sym
                cells_results[key] = d

    wall = round(_time.time() - t0, 1)
    summary = {
        "study": "multi_symbol_sweep_confluence",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "symbols": SYMBOLS,
        "timeframes": [TF],
        "fee_variants": list(FEE_VARIANTS.keys()),
        "cells": cells_results,
        "wall_sec": wall,
        "cells_run": len(cells_results),
        "cells_pass": sum(1 for c in cells_results.values() if c.get("verdict") == "PASS"),
    }

    json_path = out_dir / "walkforward.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {json_path}")

    # Render simple report
    md = [
        f"# multi_symbol_sweep_confluence — Walk-Forward Report",
        "",
        f"Generated: {summary['started_at']}",
        f"Cells run: {summary['cells_run']}  PASS: {summary['cells_pass']}",
        f"Symbols: {', '.join(SYMBOLS)}",
        f"TF: {TF}",
        f"Fee variants: {list(FEE_VARIANTS.keys())}",
        "",
        "## Pass criteria",
        f"- IS EV per trade > ${PASS_IS_EV_MIN}",
        f"- Q4 EV per trade > ${PASS_Q4_EV_MIN}",
        f"- OOS gap ≤ {int(PASS_OOS_GAP_MAX*100)}%",
        f"- Q3 and Q4 both same sign as IS",
        "",
        "## Top 20 cells by Q4 EV (PASS only)",
        "",
        "| cell | IS_n | IS_EV | Q3_EV | Q4_EV | gap% | WR_oos |",
        "|---|---|---|---|---|---|---|",
    ]
    passed = sorted(
        [(k, v) for k, v in cells_results.items() if v.get("verdict") == "PASS"],
        key=lambda kv: kv[1].get("q4_ev", 0.0), reverse=True,
    )
    for k, v in passed[:20]:
        gap = f"{v['gap_pct']*100:.0f}" if v.get("gap_pct") is not None else "—"
        md.append(
            f"| `{k}` | {v['is_n']} | ${v['is_ev']:.3f} | "
            f"${v['q3_ev']:.3f} | ${v['q4_ev']:.3f} | {gap}% | {v['win_rate_oos']*100:.0f}% |"
        )
    if not passed:
        md.append("| (none) | | | | | | |")

    md += ["", "## Best per `(sweep_atr, baseline vs conf)` (variant C_maker_scalper)"]
    md.append("")
    md.append("| sweep_atr | confluence | n | IS_EV | Q3_EV | Q4_EV | verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for atr_v in sweep_atr_grid:
        # baseline
        k = f"sweep_atr={atr_v}|baseline_no_conf|C_maker_scalper"
        v = cells_results[k]
        md.append(
            f"| {atr_v} | baseline (no conf) | {v['is_n']} | "
            f"${v['is_ev']:.3f} | ${v['q3_ev']:.3f} | ${v['q4_ev']:.3f} | {v['verdict']} |"
        )
        for win, cnt in product(conf_window_grid, conf_count_grid):
            k = cell_key(atr_v, win, cnt, "conf") + "|C_maker_scalper"
            v = cells_results[k]
            md.append(
                f"| {atr_v} | conf_win={win}m, ≥{cnt} other | {v['is_n']} | "
                f"${v['is_ev']:.3f} | ${v['q3_ev']:.3f} | ${v['q4_ev']:.3f} | {v['verdict']} |"
            )

    # Verdict distribution
    verdicts: Dict[str, int] = {}
    for v in cells_results.values():
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    md += ["", "## Verdict distribution", "", "| Verdict | Count |", "|---|---|"]
    for vname, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        md.append(f"| {vname} | {n} |")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md) + "\n")
    print(f"  wrote {report_path}")
    print(f"=== DONE ({wall}s)  cells={summary['cells_run']}  PASS={summary['cells_pass']} ===")


if __name__ == "__main__":
    main()

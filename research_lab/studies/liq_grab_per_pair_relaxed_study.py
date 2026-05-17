"""W/F STUDY: Liquidity Grab + OB + FVG — RELAXED per-pair tuning.

Hypothesis: BTC/ETH/XRP failed the SOL-tuned study (regime_max=0.6, tp_atr=2.0)
because:
  - BTC/ETH spend more time in mid-vol regime (atr_pct 0.6-0.8) — gate too strict
  - XRP has lower per-bar movement — disp_atr=0.7 too strict; needs 0.4-0.5
  - All non-SOL might benefit from smaller TP (1.5R) for higher hit rate

This study expands the grid to test relaxed params per pair. If even ONE
cell per non-SOL symbol passes, we have validated edge there.

Grid expansion vs base study:
  - regime_max:   [0.6, 0.7, 0.8, 0.9]   (was [0.6, 0.8])
  - disp_atr:     [0.4, 0.5, 0.6, 0.7]   (was [0.5, 0.7])
  - fill_disp:    [0.3, 0.4, 0.5, 0.6]   (was [0.4, 0.6])
  - tp_atr:       [1.5, 2.0]              (NEW — added 1.5)
  - sweep_atr:    [0.1]                   (fixed — best from base)
  - reclaim_lb:   [5]                     (fixed — best from base)

Total: 4 × 4 × 4 × 2 × 1 × 1 = 128 cells × 4 syms × 3 variants = 1536 cell-variants
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine
# Reuse all helpers from base study
from research_lab.studies.liq_grab_ob_fvg_study import (
    BAR_SEC, add_atr, load_4h_atr_pct,
    find_proximal_ob_long, find_proximal_ob_short,
    detect_fvg_long, detect_fvg_short,
    fvg_filled_long, fvg_filled_short,
    _walk_forward_exit, _PatchedEngine,
)


class LiqGrabPerPairRelaxedStudy(Strategy):
    """Same setup as LiquidityGrabOBFVGStudy but with EXPANDED grid for
    per-pair tuning discovery.
    """
    name = "liq_grab_per_pair_relaxed"

    _atr_4h_cache: Dict[str, Optional[pd.Series]] = {}

    def param_grid(self):
        for disp_atr in [0.4, 0.5, 0.6, 0.7]:
            for fill_disp in [0.3, 0.4, 0.5, 0.6]:
                for regime_max in [0.6, 0.7, 0.8, 0.9]:
                    for tp_atr in [1.5, 2.0]:
                        yield {
                            "disp_atr": disp_atr,
                            "fill_disp": fill_disp,
                            "regime_max": regime_max,
                            "tp_atr": tp_atr,
                            "sweep_atr": 0.1,
                            "reclaim_lookback": 5,
                            "max_age": 600,
                            "sl_atr": 1.0,
                        }

    def simulate(self, df, params):
        if len(df) < 50: return []
        df = df.copy()
        df["atr"] = add_atr(df, 14)
        df["roll_high20"] = df["high"].rolling(20).max().shift(1)
        df["roll_low20"]  = df["low"].rolling(20).min().shift(1)
        symbol = df.attrs.get("symbol", "BTC")

        if symbol not in self._atr_4h_cache:
            self._atr_4h_cache[symbol] = load_4h_atr_pct(symbol)
        atr_4h = self._atr_4h_cache[symbol]
        if atr_4h is None: return []
        df["htf_atr_pct"] = atr_4h.reindex(df.index, method="ffill")

        disp_atr = float(params["disp_atr"])
        sweep_atr = float(params["sweep_atr"])
        reclaim_lb = int(params["reclaim_lookback"])
        fill_disp = float(params["fill_disp"])
        regime_max = float(params["regime_max"])
        max_age = int(params["max_age"])
        sl_atr = float(params["sl_atr"])
        tp_atr = float(params["tp_atr"])

        trades: List[Trade] = []
        open_until = -1

        for i in range(25, len(df) - (max_age // BAR_SEC + 10)):
            if i <= open_until: continue
            row = df.iloc[i]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0: continue
            htf_pct = row.get("htf_atr_pct", float("nan"))
            if pd.isna(htf_pct) or htf_pct > regime_max: continue

            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            roll_high = float(row["roll_high20"]); roll_low = float(row["roll_low20"])
            if pd.isna([roll_high, roll_low]).any(): continue
            body = abs(c - o)

            sweep_long = (l < roll_low - sweep_atr * atr) and (c < o) and (body >= disp_atr * atr)
            if sweep_long:
                ob = find_proximal_ob_long(df, i, lookback=5)
                if ob is None: continue
                ob_high, ob_low = ob
                reclaim_idx = None
                for j in range(i + 1, min(i + 1 + reclaim_lb, len(df))):
                    if float(df.iloc[j]["close"]) >= ob_high: reclaim_idx = j; break
                if reclaim_idx is None: continue
                fvg = detect_fvg_long(df, i, reclaim_idx + 2)
                if fvg is None: continue
                fvg_idx, gap_low, gap_high = fvg
                fill_idx = fvg_filled_long(df, fvg_idx, gap_low, gap_high,
                                            min(reclaim_idx + 5, len(df)), atr, fill_disp)
                if fill_idx is None: continue
                entry = float(df.iloc[fill_idx]["close"])
                sl = entry - sl_atr * atr; tp = entry + tp_atr * atr
                result = _walk_forward_exit(df, fill_idx, "long", entry, sl, tp, max_age)
                if result is None: continue
                ex_idx, ex_px, reason, peak_r = result
                trades.append(Trade(
                    symbol=symbol, side="long", entry_price=entry, exit_price=ex_px,
                    notional_usd=1000.0, holding_sec=(ex_idx - fill_idx) * BAR_SEC,
                    entry_ts=df.index[fill_idx], exit_ts=df.index[ex_idx],
                    exit_reason=reason, extra={"peak_r": round(peak_r, 3)},
                ))
                open_until = ex_idx
                continue

            sweep_short = (h > roll_high + sweep_atr * atr) and (c > o) and (body >= disp_atr * atr)
            if sweep_short:
                ob = find_proximal_ob_short(df, i, lookback=5)
                if ob is None: continue
                ob_high, ob_low = ob
                reclaim_idx = None
                for j in range(i + 1, min(i + 1 + reclaim_lb, len(df))):
                    if float(df.iloc[j]["close"]) <= ob_low: reclaim_idx = j; break
                if reclaim_idx is None: continue
                fvg = detect_fvg_short(df, i, reclaim_idx + 2)
                if fvg is None: continue
                fvg_idx, gap_low, gap_high = fvg
                fill_idx = fvg_filled_short(df, fvg_idx, gap_low, gap_high,
                                             min(reclaim_idx + 5, len(df)), atr, fill_disp)
                if fill_idx is None: continue
                entry = float(df.iloc[fill_idx]["close"])
                sl = entry + sl_atr * atr; tp = entry - tp_atr * atr
                result = _walk_forward_exit(df, fill_idx, "short", entry, sl, tp, max_age)
                if result is None: continue
                ex_idx, ex_px, reason, peak_r = result
                trades.append(Trade(
                    symbol=symbol, side="short", entry_price=entry, exit_price=ex_px,
                    notional_usd=1000.0, holding_sec=(ex_idx - fill_idx) * BAR_SEC,
                    entry_ts=df.index[fill_idx], exit_ts=df.index[ex_idx],
                    exit_reason=reason, extra={"peak_r": round(peak_r, 3)},
                ))
                open_until = ex_idx

        return trades


if __name__ == "__main__":
    print("=== LIQ_GRAB OB+FVG — PER-PAIR RELAXED W/F STUDY ===\n")
    print("Cell grid: 4 disp × 4 fill × 4 regime × 2 tp = 128")
    print("× 4 symbols × 3 fee variants = 1,536 cell-variants\n")

    engine = _PatchedEngine(
        study=LiqGrabPerPairRelaxedStudy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "liq_grab_per_pair_relaxed",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())
    verdicts = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print(f"\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    # Per-symbol breakdown
    from collections import defaultdict
    g = defaultdict(lambda: defaultdict(int))
    pass_cells_per_sym = defaultdict(list)
    for k, v in cells:
        sym, tf, cell, variant = k.split("|", 3)
        g[sym][v["verdict"]] += 1
        if v["verdict"] == "PASS":
            pass_cells_per_sym[sym].append((k, v))

    print(f"\n=== PER-SYMBOL VERDICT BREAKDOWN ===")
    print(f"{'SYM':4s} {'PASS':>5s} {'KILL_IS':>9s} {'KILL_Q4':>9s} {'HOLD':>5s} {'INSUFF':>7s}")
    print("-" * 55)
    for sym in ["BTC", "ETH", "SOL", "XRP"]:
        c = g[sym]
        print(f"{sym:4s} {c.get('PASS',0):>5d} {c.get('KILL_IS_NEGATIVE',0):>9d} "
              f"{c.get('KILL_Q4_BELOW_FLOOR',0):>9d} {c.get('HOLD_OOS_MIXED_SIGN',0):>5d} "
              f"{c.get('INSUFFICIENT_DATA',0):>7d}")

    # Top PASS per symbol
    for sym in ["BTC", "ETH", "SOL", "XRP"]:
        passes = pass_cells_per_sym[sym]
        if not passes:
            print(f"\n{sym}: NO PASS CELLS")
            continue
        passes.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        print(f"\n=== {sym} — TOP 5 PASS CELLS by Q4 EV ({len(passes)} total) ===")
        for k, v in passes[:5]:
            _, _, cell, variant = k.split("|", 3)
            short = {"A_full_taker":"A","B_scalper_taker":"B","C_maker_scalper":"C"}.get(variant,"?")
            print(f"  {short}  {cell}")
            print(f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  WR_oos={v['win_rate_oos']*100:.0f}%")

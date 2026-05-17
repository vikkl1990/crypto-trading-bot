"""W/F STUDY: BOS/CHoCH Scanner with FIXED MSS Retest-Confirmation Gate (Patch I).

Validates the bos_choch scanner after the NameError bug fix in
strategies/scalp_strategy.py::_scan_bos_choch (lines ~6014-6019).

Bug: an "MSS Confirmation Gate (Upgrade 3)" referenced bare `close`/`open_`/
`high`/`low` that don't exist in the function scope. NameError swallowed every
signal silently from Apr 30 onward.

Fix replicates the live entry logic with the gate parameters exposed:
  - 22-bar rolling structure window (skip last 4)
  - bar[-3] = BREAK candle, bar[-2] = CONFIRM candle, bar[-1] = ENTRY candle
  - displacement_atr_min: BREAK candle body >= X × ATR (sweep: 0.6, 0.8, 1.0)
  - Confirm candle: body/range >= 0.55 (fixed)
  - MSS GATE on ENTRY candle:
      * mss_body_atr_min: entry body >= X × ATR (sweep: 0.40, 0.55, 0.65, 0.75)
      * mss_body_pct_min: entry body / range >= X (sweep: 0.50, 0.55, 0.65)
      * direction: entry candle must close in trade direction

Exit: 2:1 RR (TP at 2× SL distance) + 30-min hard time stop.
Notional: $400 (matches existing studies).

If passes W/F, recommend the gate fix + the best-performing param triplet for
production.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

BAR_SEC = 300
MAX_AGE_SEC = 1800  # 30-min hard time stop
NOTIONAL_USD = 400.0


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_rel_vol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(lookback).median()


def _walk_forward_exit(df, i, side, entry, sl, tp, max_age_sec):
    """SL/TP/time-stop walker. Returns (exit_idx, exit_price, reason)."""
    initial_risk = abs(entry - sl)
    if initial_risk <= 0:
        return None
    max_bars = max_age_sec // BAR_SEC + 2
    for j in range(i + 1, min(i + 1 + max_bars, len(df))):
        bar = df.iloc[j]
        age_sec = (j - i) * BAR_SEC
        if side == "long":
            if float(bar["low"]) <= sl:
                return (j, sl, "sl_hit")
            if float(bar["high"]) >= tp:
                return (j, tp, "tp_hit")
        else:
            if float(bar["high"]) >= sl:
                return (j, sl, "sl_hit")
            if float(bar["low"]) <= tp:
                return (j, tp, "tp_hit")
        if age_sec >= max_age_sec:
            return (j, float(bar["close"]), "time_decay")
    last_idx = min(i + max_bars, len(df) - 1)
    return (last_idx, float(df.iloc[last_idx]["close"]), "forced_end")


class BosChochRetestStudy(Strategy):
    """Replicates _scan_bos_choch with the FIXED MSS retest gate.

    Param sweep:
      mss_body_atr_min: entry candle body >= X × ATR     (0.40, 0.55, 0.65, 0.75)
      mss_body_pct_min: entry candle body/range          (0.50, 0.55, 0.65)
      displacement_atr_min: break candle body >= X × ATR (0.6, 0.8, 1.0)
    Total: 4 × 3 × 3 = 36 cells per symbol.

    Other constants (matching live source):
      structure_window = bars [-22:-4]  (18 bars)
      structure_depth = 8 minimum
      pre_break = bars [-5:-3]
      confirm_body_pct = 0.55  (fixed)
      vol_floor = 1.3 (fixed; live also rejects below this)
    """
    name = "bos_choch_retest"

    # Diagnostics counters (reset per simulate call)
    _diag: Dict[str, int] = {}
    # Per-symbol cache to avoid recomputing ATR / rel_vol on every cell
    _df_cache: Dict[str, pd.DataFrame] = {}

    def param_grid(self):
        for disp in [0.6, 0.8, 1.0]:
            for body_atr in [0.40, 0.55, 0.65, 0.75]:
                for body_pct in [0.50, 0.55, 0.65]:
                    yield {
                        "displacement_atr_min": disp,
                        "mss_body_atr_min": body_atr,
                        "mss_body_pct_min": body_pct,
                    }

    def cell_id(self, params):
        return (
            f"disp={params['displacement_atr_min']}_"
            f"mssAtr={params['mss_body_atr_min']}_"
            f"mssPct={params['mss_body_pct_min']}"
        )

    def simulate(self, df, params):
        if len(df) < 30:
            return []
        symbol = df.attrs.get("symbol", "BTC")
        # Cache pre-computed indicators per symbol
        cache_key = f"{symbol}_{len(df)}"
        if cache_key not in self._df_cache:
            df2 = df.copy()
            df2["atr"] = add_atr(df2, 14)
            df2["rel_vol"] = add_rel_vol(df2, 20)
            # Pre-compute rolling structure once (skip last 4 means shift(4) on rolling 18)
            df2["roll_high18"] = df2["high"].rolling(18).max().shift(4)
            df2["roll_low18"]  = df2["low"].rolling(18).min().shift(4)
            self._df_cache[cache_key] = df2
        df = self._df_cache[cache_key]

        disp_min = float(params["displacement_atr_min"])
        body_atr_min = float(params["mss_body_atr_min"])
        body_pct_min = float(params["mss_body_pct_min"])

        # Vectorize column access via numpy arrays for fast indexed lookup
        opens  = df["open"].to_numpy(dtype=float)
        highs  = df["high"].to_numpy(dtype=float)
        lows   = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        atrs   = df["atr"].to_numpy(dtype=float)
        rvols  = df["rel_vol"].to_numpy(dtype=float)
        rolling_highs = df["roll_high18"].to_numpy(dtype=float)
        rolling_lows  = df["roll_low18"].to_numpy(dtype=float)
        index = df.index

        trades: List[Trade] = []
        open_until = -1

        diag = {"loop": 0, "atr_skip": 0, "structure_skip": 0,
                "no_break": 0, "disp_fail": 0, "confirm_fail": 0,
                "mss_fail": 0, "vol_fail": 0, "open": 0}

        max_bars_exit = MAX_AGE_SEC // BAR_SEC + 5
        n = len(df)

        for i in range(22, n - max_bars_exit):
            diag["loop"] += 1
            if i <= open_until:
                continue

            atr = atrs[i]
            if atr != atr or atr <= 0:  # NaN check
                diag["atr_skip"] += 1
                continue

            # Live: structure_window = df.iloc[-22:-4]  with -1 == i (current bar)
            # That's bars i-21 through i-5 inclusive (17 bars)
            # But code says bar[-3] is BREAK, bar[-2] is CONFIRM, bar[-1] is ENTRY.
            # So structure ends at i-5 (4 bars before break = exclude break/confirm/entry+1)
            # Match live exactly:  df.iloc[-22:-4]  → indices [n-22, n-4)  → for "now"=i, that's [i-21, i-3)
            # Since rolling_high18.shift(4) at index i covers windows ending at i-4 (i.e. [i-21, i-4]),
            # this matches structure_window.high.max() of df.iloc[-22:-4] (exclusive end).
            rolling_high = rolling_highs[i]
            rolling_low  = rolling_lows[i]
            if rolling_high != rolling_high or rolling_low != rolling_low:
                diag["structure_skip"] += 1
                continue

            # pre_break: bars [i-5, i-3) = i-5 and i-4
            pre_close_a = closes[i - 5]
            pre_close_b = closes[i - 4]

            break_close = closes[i - 2]
            break_open  = opens[i - 2]
            break_high  = highs[i - 2]
            break_low   = lows[i - 2]
            break_body  = abs(break_close - break_open)

            side = None
            if break_close > rolling_high and min(pre_close_a, pre_close_b) <= rolling_high:
                side = "long"
            elif break_close < rolling_low and max(pre_close_a, pre_close_b) >= rolling_low:
                side = "short"
            if side is None:
                diag["no_break"] += 1
                continue

            displacement = break_body / atr
            if displacement < disp_min:
                diag["disp_fail"] += 1
                continue

            # Confirm bar (bar[-2] in live = i-1)
            conf_close = closes[i - 1]
            conf_open  = opens[i - 1]
            conf_high  = highs[i - 1]
            conf_low   = lows[i - 1]
            conf_body  = abs(conf_close - conf_open)
            conf_range = conf_high - conf_low if conf_high > conf_low else atr * 0.01

            if side == "long":
                if conf_close <= conf_open:
                    diag["confirm_fail"] += 1; continue
                if conf_close < rolling_high:
                    diag["confirm_fail"] += 1; continue
            else:
                if conf_close >= conf_open:
                    diag["confirm_fail"] += 1; continue
                if conf_close > rolling_low:
                    diag["confirm_fail"] += 1; continue
            if (conf_body / conf_range) < 0.55:
                diag["confirm_fail"] += 1; continue

            # MSS ENTRY GATE
            entry_open  = opens[i]
            entry_close = closes[i]
            entry_high  = highs[i]
            entry_low   = lows[i]
            entry_body  = abs(entry_close - entry_open)
            entry_range = entry_high - entry_low if entry_high > entry_low else atr * 0.01

            if entry_body / entry_range < body_pct_min:
                diag["mss_fail"] += 1; continue
            if entry_body / atr < body_atr_min:
                diag["mss_fail"] += 1; continue
            if side == "long" and entry_close <= entry_open:
                diag["mss_fail"] += 1; continue
            if side == "short" and entry_close >= entry_open:
                diag["mss_fail"] += 1; continue

            # Volume floor
            break_vol = rvols[i - 2]
            if break_vol == break_vol and break_vol < 1.3:
                diag["vol_fail"] += 1; continue

            # SL + TP
            if side == "long":
                sl = min(entry_low, break_low, rolling_high) - atr * 0.3
                risk = entry_close - sl
                if risk <= 0: continue
                tp = entry_close + 2.0 * risk
            else:
                sl = max(entry_high, break_high, rolling_low) + atr * 0.3
                risk = sl - entry_close
                if risk <= 0: continue
                tp = entry_close - 2.0 * risk

            # Inline exit walker (fast)
            initial_risk = abs(entry_close - sl)
            exit_idx = -1
            exit_price = 0.0
            reason = "forced_end"
            max_bars = MAX_AGE_SEC // BAR_SEC + 2
            for j in range(i + 1, min(i + 1 + max_bars, n)):
                bj_low = lows[j]; bj_high = highs[j]; bj_close = closes[j]
                age_sec = (j - i) * BAR_SEC
                if side == "long":
                    if bj_low <= sl:
                        exit_idx = j; exit_price = sl; reason = "sl_hit"; break
                    if bj_high >= tp:
                        exit_idx = j; exit_price = tp; reason = "tp_hit"; break
                else:
                    if bj_high >= sl:
                        exit_idx = j; exit_price = sl; reason = "sl_hit"; break
                    if bj_low <= tp:
                        exit_idx = j; exit_price = tp; reason = "tp_hit"; break
                if age_sec >= MAX_AGE_SEC:
                    exit_idx = j; exit_price = bj_close; reason = "time_decay"; break
            if exit_idx < 0:
                exit_idx = min(i + max_bars, n - 1)
                exit_price = closes[exit_idx]
                reason = "forced_end"

            holding_sec = (exit_idx - i) * BAR_SEC
            trades.append(Trade(
                symbol=symbol, side=side,
                entry_price=entry_close, exit_price=exit_price,
                notional_usd=NOTIONAL_USD, holding_sec=holding_sec,
                entry_ts=index[i], exit_ts=index[exit_idx],
                exit_reason=reason,
                extra={"disp": round(displacement, 2),
                       "entry_body_atr": round(entry_body / atr, 2),
                       "entry_body_pct": round(entry_body / entry_range, 2)},
            ))
            open_until = exit_idx
            diag["open"] += 1

        BosChochRetestStudy._diag = diag
        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== BOS/CHoCH RETEST CONFIRMATION GATE — W/F STUDY (Patch I) ===\n")
    print("Cell grid: 3 disp × 4 mssAtr × 3 mssPct = 36 cells per symbol")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants = 432 cell-variants\n")

    engine = _PatchedEngine(
        study=BosChochRetestStudy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "bos_choch_retest",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    pass_cells = [c for c in cells if c[1]["verdict"] == "PASS"]
    if pass_cells:
        print(f"\n=== TOP 10 PASS CELLS by Q4 EV ===")
        pass_cells.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        for k, v in pass_cells[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B",
                     "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  {cell}")
            print(f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  "
                  f"Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  "
                  f"WR_oos={v['win_rate_oos']*100:.0f}%")
    else:
        print(f"\n=== NO PASS CELLS — Top 8 by IS_EV ===")
        cells_sorted = sorted(cells, key=lambda c: c[1]["is_ev"], reverse=True)
        for k, v in cells_sorted[:8]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B",
                     "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  {cell}")
            print(f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  "
                  f"Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  "
                  f"verdict={v['verdict']}")

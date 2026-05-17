"""W/F STUDY: Volume Absorption Bubble (smart-money / order-flow reversal).

Tests a NEW reversal pattern for the VN Edge bot. Today's three SMC RETEST
variants (Demand Zone, Break Block, bos_choch retest) all KILLED — they are
trend-continuation in a chop regime. Absorption Bubble is REVERSAL after a
liquidity sweep, so the mechanism is different and may survive chop.

Setup (LONG; mirror for SHORT):
  Step 1 — LIQUIDITY SWEEP: bar i sweeps below `lookback_n` 5m bars' low by
           >= sweep_atr × ATR.
  Step 2 — ABSORPTION CANDLE: bar i (or i-1) shows ABSORPTION:
             - High volume: rel_vol >= vol_mult (vol / 20-bar mean vol)
             - Small body / range:  body / range <= body_ratio_max
             - Long lower wick:     lower_wick / range >= wick_ratio_min
  Step 3 — RECLAIM CONFIRMATION: bar i+1 closes ABOVE the sweep low with
           displacement body >= displ_atr × ATR.
  Step 4 — ENTRY at confirmation candle close. STOP = absorption_low - 0.3 ATR.
  Step 5 — TARGET: fixed 1.5R or 2R (param).

Hard time stop 30 minutes (6 bars). Notional $400 fixed. Both LONG and SHORT.

Implementation note: per-cell execution is vectorised on numpy arrays — pandas
iloc was 30×+ slower per cell. Full 162-cell grid runs in ~3-5 minutes per pair.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

BAR_SEC = 300
TIME_STOP_SEC = 1800        # 30-minute hard time stop
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC   # 6 bars
SL_ATR_BUFFER = 0.3         # stop = absorption_low - 0.3 × ATR
NOTIONAL_USD = 400.0
VOL_LOOKBACK = 20           # rolling-mean window for rel_vol


def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """ATR via True Range, simple rolling mean. Pure numpy."""
    n = len(high)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan, dtype=np.float64)
    csum = np.cumsum(tr)
    for i in range(period - 1, n):
        if i == period - 1:
            atr[i] = csum[i] / period
        else:
            atr[i] = (csum[i] - csum[i - period]) / period
    return atr


def rolling_mean_np(x: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean (NaN until window-1)."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    csum = np.cumsum(x)
    for i in range(window - 1, n):
        if i == window - 1:
            out[i] = csum[i] / window
        else:
            out[i] = (csum[i] - csum[i - window]) / window
    return out


def rolling_min_shift1(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling min over `window` bars, shifted by 1 (excludes current bar)."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        out[i] = np.min(x[i - window:i])
    return out


def rolling_max_shift1(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        out[i] = np.max(x[i - window:i])
    return out


def _walk_forward_exit_np(open_a: np.ndarray, high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, entry_idx: int, side: str,
                          entry: float, sl: float, tp: float,
                          max_bars: int) -> Tuple[int, float, str]:
    """Bar-by-bar walk to first SL/TP/time-stop. Returns (exit_idx, exit_price, reason)."""
    n = len(close_a)
    end = min(entry_idx + 1 + max_bars, n)
    for j in range(entry_idx + 1, end):
        if side == "long":
            if low_a[j] <= sl:
                return (j, sl, "sl_hit")
            if high_a[j] >= tp:
                return (j, tp, "tp_hit")
        else:
            if high_a[j] >= sl:
                return (j, sl, "sl_hit")
            if low_a[j] <= tp:
                return (j, tp, "tp_hit")
        if (j - entry_idx) >= max_bars:
            return (j, close_a[j], "time_stop")
    last_idx = min(entry_idx + max_bars, n - 1)
    return (last_idx, close_a[last_idx], "forced_end")


class AbsorptionBubbleStrategy(Strategy):
    """Volume Absorption Bubble — 3-step pattern.

    Reduced grid (wick_ratio_min=0.50 and body_ratio_max=0.40 fixed):
      lookback_n:   [10, 20, 30]
      sweep_atr:    [0.1, 0.2, 0.3]
      vol_mult:     [1.5, 2.0, 2.5]
      displ_atr:    [0.4, 0.5, 0.65]
      tp_mode:      ["fixed_1.5r", "fixed_2r"]
    => 3 × 3 × 3 × 3 × 2 = 162 cells per pair.
    """
    name = "absorption_bubble"

    # Fixed knobs (the "default" ratios from the spec)
    BODY_RATIO_MAX = 0.40
    WICK_RATIO_MIN = 0.50

    def param_grid(self):
        for lookback_n in [10, 20, 30]:
            for sweep_atr in [0.1, 0.2, 0.3]:
                for vol_mult in [1.5, 2.0, 2.5]:
                    for displ_atr in [0.4, 0.5, 0.65]:
                        for tp_mode in ["fixed_1.5r", "fixed_2r"]:
                            yield {
                                "lookback_n": lookback_n,
                                "sweep_atr": sweep_atr,
                                "vol_mult": vol_mult,
                                "displ_atr": displ_atr,
                                "tp_mode": tp_mode,
                            }

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 60:
            return []
        symbol = df.attrs.get("symbol", "BTC")

        # Pre-compute numpy arrays once
        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        if "volume" not in df.columns:
            return []
        vol_a = df["volume"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, 14)
        ts_a = df.index.to_numpy()

        # rel_vol = volume / 20-bar rolling mean (shifted? — use rolling that includes
        # the current bar to mirror the spec; absorption candle volume is what counts).
        vol_mean = rolling_mean_np(vol_a, VOL_LOOKBACK)
        rel_vol = np.divide(
            vol_a, vol_mean,
            out=np.zeros_like(vol_a), where=(vol_mean > 0)
        )

        # Body / range / wick characterization
        rng = high_a - low_a
        body_abs = np.abs(close_a - open_a)
        # Lower wick: distance from low to bottom of body
        # bottom_of_body = min(open, close)
        bottom_of_body = np.minimum(open_a, close_a)
        top_of_body = np.maximum(open_a, close_a)
        lower_wick = bottom_of_body - low_a
        upper_wick = high_a - top_of_body

        # Avoid divide-by-zero: where rng == 0, ratios are 0 (no info)
        body_ratio = np.divide(body_abs, rng, out=np.zeros_like(rng), where=(rng > 0))
        lower_wick_ratio = np.divide(lower_wick, rng, out=np.zeros_like(rng), where=(rng > 0))
        upper_wick_ratio = np.divide(upper_wick, rng, out=np.zeros_like(rng), where=(rng > 0))

        # Absorption candle (LONG side): high vol AND small body AND long lower wick
        # We'll evaluate per-cell at the bar of interest. Absorption can be bar i OR i-1.
        # Sweep candle is bar i.

        lookback_n = int(params["lookback_n"])
        sweep_atr_k = float(params["sweep_atr"])
        vol_mult = float(params["vol_mult"])
        displ_atr_k = float(params["displ_atr"])
        tp_mode = str(params["tp_mode"])
        tp_r = 1.5 if tp_mode == "fixed_1.5r" else 2.0

        roll_low = rolling_min_shift1(low_a, lookback_n)
        roll_high = rolling_max_shift1(high_a, lookback_n)

        # Pre-mask absorption candidates per side (compactness; checked at i and i-1)
        # LONG absorption: rel_vol >= vol_mult, body_ratio <= body_ratio_max,
        #                  lower_wick_ratio >= wick_ratio_min
        absorb_long = (
            (rel_vol >= vol_mult) &
            (body_ratio <= self.BODY_RATIO_MAX) &
            (lower_wick_ratio >= self.WICK_RATIO_MIN)
        )
        # SHORT absorption: rel_vol >= vol_mult, body_ratio <= body_ratio_max,
        #                   upper_wick_ratio >= wick_ratio_min
        absorb_short = (
            (rel_vol >= vol_mult) &
            (body_ratio <= self.BODY_RATIO_MAX) &
            (upper_wick_ratio >= self.WICK_RATIO_MIN)
        )

        trades: List[Trade] = []
        open_until = -1
        max_bars_exit = TIME_STOP_BARS

        # Outer loop: confirmation candle is at index k (= sweep_idx + 1).
        # We need: sweep_idx = k - 1, absorption candidate at sweep_idx OR sweep_idx-1.
        # min start index = max(VOL_LOOKBACK, lookback_n) + 2.
        min_start = max(VOL_LOOKBACK, lookback_n, 14) + 2
        end_k = n - (TIME_STOP_BARS + 5)

        for k in range(min_start, end_k):
            if k <= open_until:
                continue
            atr = atr_a[k]
            if not np.isfinite(atr) or atr <= 0:
                continue

            sweep_idx = k - 1
            sweep_atr = atr_a[sweep_idx]
            if not np.isfinite(sweep_atr) or sweep_atr <= 0:
                continue

            sweep_low = roll_low[sweep_idx]
            sweep_high = roll_high[sweep_idx]
            if not np.isfinite(sweep_low) or not np.isfinite(sweep_high):
                continue

            # ─── LONG SETUP ─────────────────────────────────────────────
            # 1) bar sweep_idx swept below sweep_low by >= sweep_atr_k × atr
            swept_long = low_a[sweep_idx] < (sweep_low - sweep_atr_k * sweep_atr)
            # 2) bar sweep_idx OR sweep_idx-1 was an absorption candle (LONG style)
            absorb_idx_long = -1
            absorb_low_long = 0.0
            if swept_long:
                if absorb_long[sweep_idx]:
                    absorb_idx_long = sweep_idx
                    absorb_low_long = low_a[sweep_idx]
                elif absorb_long[sweep_idx - 1]:
                    absorb_idx_long = sweep_idx - 1
                    absorb_low_long = low_a[sweep_idx - 1]
                # 3) confirmation: bar k closes ABOVE the sweep_low with
                #    body >= displ_atr_k × atr AND bullish (close > open)
                if absorb_idx_long >= 0:
                    cl_k = close_a[k]
                    op_k = open_a[k]
                    body_k = abs(cl_k - op_k)
                    if (cl_k > sweep_low) and (cl_k > op_k) and (body_k >= displ_atr_k * atr):
                        # Entry at confirmation candle close
                        entry = cl_k
                        sl = absorb_low_long - SL_ATR_BUFFER * atr
                        if sl >= entry:
                            continue
                        risk = entry - sl
                        tp = entry + tp_r * risk
                        if tp <= entry:
                            continue
                        exit_idx, exit_price, reason = _walk_forward_exit_np(
                            open_a, high_a, low_a, close_a, k, "long",
                            entry, sl, tp, max_bars_exit
                        )
                        trades.append(Trade(
                            symbol=symbol, side="long",
                            entry_price=float(entry), exit_price=float(exit_price),
                            notional_usd=NOTIONAL_USD,
                            holding_sec=int((exit_idx - k) * BAR_SEC),
                            entry_ts=pd.Timestamp(ts_a[k]),
                            exit_ts=pd.Timestamp(ts_a[exit_idx]),
                            exit_reason=reason,
                            extra={"absorb_idx": int(absorb_idx_long),
                                   "sweep_low": float(sweep_low),
                                   "absorb_low": float(absorb_low_long),
                                   "tp_mode": tp_mode},
                        ))
                        open_until = exit_idx
                        continue

            # ─── SHORT SETUP ────────────────────────────────────────────
            swept_short = high_a[sweep_idx] > (sweep_high + sweep_atr_k * sweep_atr)
            absorb_idx_short = -1
            absorb_high_short = 0.0
            if swept_short:
                if absorb_short[sweep_idx]:
                    absorb_idx_short = sweep_idx
                    absorb_high_short = high_a[sweep_idx]
                elif absorb_short[sweep_idx - 1]:
                    absorb_idx_short = sweep_idx - 1
                    absorb_high_short = high_a[sweep_idx - 1]
                if absorb_idx_short >= 0:
                    cl_k = close_a[k]
                    op_k = open_a[k]
                    body_k = abs(cl_k - op_k)
                    if (cl_k < sweep_high) and (cl_k < op_k) and (body_k >= displ_atr_k * atr):
                        entry = cl_k
                        sl = absorb_high_short + SL_ATR_BUFFER * atr
                        if sl <= entry:
                            continue
                        risk = sl - entry
                        tp = entry - tp_r * risk
                        if tp >= entry:
                            continue
                        exit_idx, exit_price, reason = _walk_forward_exit_np(
                            open_a, high_a, low_a, close_a, k, "short",
                            entry, sl, tp, max_bars_exit
                        )
                        trades.append(Trade(
                            symbol=symbol, side="short",
                            entry_price=float(entry), exit_price=float(exit_price),
                            notional_usd=NOTIONAL_USD,
                            holding_sec=int((exit_idx - k) * BAR_SEC),
                            entry_ts=pd.Timestamp(ts_a[k]),
                            exit_ts=pd.Timestamp(ts_a[exit_idx]),
                            exit_reason=reason,
                            extra={"absorb_idx": int(absorb_idx_short),
                                   "sweep_high": float(sweep_high),
                                   "absorb_high": float(absorb_high_short),
                                   "tp_mode": tp_mode},
                        ))
                        open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== VOLUME ABSORPTION BUBBLE — W/F STUDY ===\n")
    print("Cell grid: 3 lookback × 3 sweep × 3 vol × 3 displ × 2 tp = 162 cells per pair")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants = 1944 cell-variants total\n")

    engine = _PatchedEngine(
        study=AbsorptionBubbleStrategy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "absorption_bubble",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1

    pair_trade_n: Dict[str, List[int]] = {}
    for k, v in cells:
        sym, tf, cell, variant = k.split("|", 3)
        if variant != "A_full_taker":
            continue
        pair_trade_n.setdefault(sym, []).append(v["n_total"])

    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")
    print("\nTrade counts per pair (variant A, across cells):")
    for sym, counts in pair_trade_n.items():
        if counts:
            print(f"  {sym}: max={max(counts)}  med={sorted(counts)[len(counts)//2]}  "
                  f"avg={sum(counts)/len(counts):.0f}  n_cells={len(counts)}")

    pass_cells = [c for c in cells if c[1]["verdict"] == "PASS"]
    if pass_cells:
        print(f"\n=== TOP 15 PASS CELLS by Q4 EV ===")
        pass_cells.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        for k, v in pass_cells[:15]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B",
                     "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  {cell}")
            print(f"    n_total={v['n_total']} IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  "
                  f"Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  WR_oos={v['win_rate_oos']*100:.0f}%")
    else:
        cells_sorted = sorted(cells, key=lambda c: c[1]["is_ev"], reverse=True)
        print(f"\n=== Top 15 by IS_EV (no PASS cells) ===")
        for k, v in cells_sorted[:15]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B",
                     "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  IS_n={v['is_n']} n_total={v['n_total']} "
                  f"IS=${v['is_ev']:+.3f}  Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  "
                  f"verdict={v['verdict']}")

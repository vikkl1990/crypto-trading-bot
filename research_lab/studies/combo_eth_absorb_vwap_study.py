"""W/F STUDY: ETH Ensemble of absorption_bubble + scalper_vwap_mr.

Both engines are W/F-validated on ETH individually. Test if ensemble agreement
(both fire same-side within `window_bars` of each other) yields per-trade EV
uplift over either alone.

Modes:
  - "absorb_only" : trade only when absorption_bubble fires
  - "vwap_only"   : trade only when scalper_vwap_mr fires
  - "ensemble_AND": trade only when BOTH fire same-side within last
                    `window_bars` of each other (use most-recent confirmation
                    for entry/SL/TP)
  - "ensemble_OR" : trade when either fires (sanity check)

Locked param values (per spec):
  Absorption:  lookback_n=30, sweep_atr=0.3, vol_mult=1.5, displ_atr=0.65,
               body_ratio_max=0.40, wick_ratio_min=0.50
  VWAP MR  :   dist_atr=1.0, body_atr=0.3 (Bollinger ±k×stdev VWAP-style; we
               use VWAP distance + bullish/bearish reversal candle, mirroring
               chop_vwap_mr_study)

Sweep:
  mode        : ["absorb_only","vwap_only","ensemble_AND","ensemble_OR"]
  window_bars : [1, 3, 6]   (5/15/30 min agreement window)
  tp_rr       : [1.5, 2.0]
  -> 4 × 3 × 2 = 24 cells × ETH-only × 3 fee variants = 72 cell-variants

Notes:
  - Ensemble entry uses the entry/SL/TP of whichever fired LAST in the window
    (most recent confirmation), matching real-time trade logic.
  - Same hard 30-min time stop, $400 notional, 5m bars.
  - One open trade at a time (open_until guard).
  - For modes that allow OR / single-engine, all signals are eligible.

Vectorised numpy, well under 5 min wall time.
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

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
BAR_SEC = 300
TIME_STOP_SEC = 1800
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6
NOTIONAL_USD = 400.0
ATR_PERIOD = 14
VOL_LOOKBACK = 20
SWING_LOOKBACK = 10
SL_ATR_BUFFER_ABS = 0.3      # absorption: stop = absorb_low - 0.3 × ATR
SL_ATR_BUFFER_VWAP = 0.3     # vwap-mr: stop = swing - 0.3 × ATR

# Locked absorption params (per spec)
ABS_LOOKBACK_N = 30
ABS_SWEEP_ATR = 0.3
ABS_VOL_MULT = 1.5
ABS_DISPL_ATR = 0.65
ABS_BODY_RATIO_MAX = 0.40
ABS_WICK_RATIO_MIN = 0.50

# Locked vwap-mr params (per spec — close-style mean revert with VWAP distance)
VWAP_DIST_ATR = 1.0
VWAP_BODY_ATR = 0.3


# ─────────────────────────────────────────────────────────────────────
# Numpy helpers
# ─────────────────────────────────────────────────────────────────────
def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
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


def compute_vwap_daily_reset(ts_a: np.ndarray, high: np.ndarray, low: np.ndarray,
                             close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(ts_a)
    vwap = np.full(n, np.nan, dtype=np.float64)
    typical = (high + low + close) / 3.0
    pv = typical * volume
    days = pd.DatetimeIndex(ts_a).floor("D").asi8
    cum_pv = 0.0
    cum_v = 0.0
    cur_day = days[0]
    for i in range(n):
        if days[i] != cur_day:
            cur_day = days[i]
            cum_pv = 0.0
            cum_v = 0.0
        cum_pv += pv[i]
        cum_v += volume[i]
        if cum_v > 0:
            vwap[i] = cum_pv / cum_v
    return vwap


def _walk_forward_exit_np(open_a, high_a, low_a, close_a,
                          entry_idx, side, entry, sl, tp, max_bars):
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


# ─────────────────────────────────────────────────────────────────────
# Signal precomputation
# ─────────────────────────────────────────────────────────────────────
def compute_absorption_signals(df: pd.DataFrame
                               ) -> Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
    """Return (long_mask, short_mask, entry_long, sl_long, entry_short, sl_short, atr).

    Signal at index k means absorption confirmation candle is at k.
    """
    n = len(df)
    open_a = df["open"].to_numpy(dtype=np.float64)
    high_a = df["high"].to_numpy(dtype=np.float64)
    low_a = df["low"].to_numpy(dtype=np.float64)
    close_a = df["close"].to_numpy(dtype=np.float64)
    vol_a = df["volume"].to_numpy(dtype=np.float64)
    atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)

    vol_mean = rolling_mean_np(vol_a, VOL_LOOKBACK)
    rel_vol = np.divide(vol_a, vol_mean, out=np.zeros_like(vol_a),
                        where=(vol_mean > 0))

    rng = high_a - low_a
    body_abs = np.abs(close_a - open_a)
    bottom_of_body = np.minimum(open_a, close_a)
    top_of_body = np.maximum(open_a, close_a)
    lower_wick = bottom_of_body - low_a
    upper_wick = high_a - top_of_body
    body_ratio = np.divide(body_abs, rng, out=np.zeros_like(rng), where=(rng > 0))
    lower_wick_ratio = np.divide(lower_wick, rng, out=np.zeros_like(rng), where=(rng > 0))
    upper_wick_ratio = np.divide(upper_wick, rng, out=np.zeros_like(rng), where=(rng > 0))

    absorb_long = (
        (rel_vol >= ABS_VOL_MULT) &
        (body_ratio <= ABS_BODY_RATIO_MAX) &
        (lower_wick_ratio >= ABS_WICK_RATIO_MIN)
    )
    absorb_short = (
        (rel_vol >= ABS_VOL_MULT) &
        (body_ratio <= ABS_BODY_RATIO_MAX) &
        (upper_wick_ratio >= ABS_WICK_RATIO_MIN)
    )

    roll_low = rolling_min_shift1(low_a, ABS_LOOKBACK_N)
    roll_high = rolling_max_shift1(high_a, ABS_LOOKBACK_N)

    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)
    entry_long = np.full(n, np.nan, dtype=np.float64)
    sl_long = np.full(n, np.nan, dtype=np.float64)
    entry_short = np.full(n, np.nan, dtype=np.float64)
    sl_short = np.full(n, np.nan, dtype=np.float64)

    min_start = max(VOL_LOOKBACK, ABS_LOOKBACK_N, ATR_PERIOD) + 2
    end_k = n - (TIME_STOP_BARS + 5)

    for k in range(min_start, end_k):
        atr = atr_a[k]
        if not np.isfinite(atr) or atr <= 0:
            continue
        sweep_idx = k - 1
        sweep_atr_v = atr_a[sweep_idx]
        if not np.isfinite(sweep_atr_v) or sweep_atr_v <= 0:
            continue
        sweep_low = roll_low[sweep_idx]
        sweep_high = roll_high[sweep_idx]
        if not np.isfinite(sweep_low) or not np.isfinite(sweep_high):
            continue

        # LONG
        swept_long = low_a[sweep_idx] < (sweep_low - ABS_SWEEP_ATR * sweep_atr_v)
        if swept_long:
            absorb_idx = -1
            absorb_low = 0.0
            if absorb_long[sweep_idx]:
                absorb_idx = sweep_idx
                absorb_low = low_a[sweep_idx]
            elif absorb_long[sweep_idx - 1]:
                absorb_idx = sweep_idx - 1
                absorb_low = low_a[sweep_idx - 1]
            if absorb_idx >= 0:
                cl_k = close_a[k]
                op_k = open_a[k]
                body_k = abs(cl_k - op_k)
                if (cl_k > sweep_low) and (cl_k > op_k) and (body_k >= ABS_DISPL_ATR * atr):
                    entry = cl_k
                    sl = absorb_low - SL_ATR_BUFFER_ABS * atr
                    if sl < entry:
                        long_mask[k] = True
                        entry_long[k] = entry
                        sl_long[k] = sl

        # SHORT
        swept_short = high_a[sweep_idx] > (sweep_high + ABS_SWEEP_ATR * sweep_atr_v)
        if swept_short:
            absorb_idx = -1
            absorb_high = 0.0
            if absorb_short[sweep_idx]:
                absorb_idx = sweep_idx
                absorb_high = high_a[sweep_idx]
            elif absorb_short[sweep_idx - 1]:
                absorb_idx = sweep_idx - 1
                absorb_high = high_a[sweep_idx - 1]
            if absorb_idx >= 0:
                cl_k = close_a[k]
                op_k = open_a[k]
                body_k = abs(cl_k - op_k)
                if (cl_k < sweep_high) and (cl_k < op_k) and (body_k >= ABS_DISPL_ATR * atr):
                    entry = cl_k
                    sl = absorb_high + SL_ATR_BUFFER_ABS * atr
                    if sl > entry:
                        short_mask[k] = True
                        entry_short[k] = entry
                        sl_short[k] = sl

    return (long_mask, short_mask, entry_long, sl_long, entry_short, sl_short, atr_a)


def compute_vwap_mr_signals(df: pd.DataFrame
                            ) -> Tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Return (long_mask, short_mask, entry_long, sl_long, entry_short, sl_short).

    Signal at index i means VWAP-MR confirmation candle is at i.
    Logic: price ≤ VWAP - dist_atr × ATR (or above for short),
           reversal candle (close > open for long, close in upper half),
           body ≥ body_atr × ATR.
    Stop: swing low/high last 10 bars - 0.3 × ATR (long) / + (short).
    """
    n = len(df)
    open_a = df["open"].to_numpy(dtype=np.float64)
    high_a = df["high"].to_numpy(dtype=np.float64)
    low_a = df["low"].to_numpy(dtype=np.float64)
    close_a = df["close"].to_numpy(dtype=np.float64)
    vol_a = df["volume"].to_numpy(dtype=np.float64)
    ts_a = df.index.to_numpy()
    atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)
    vwap_a = compute_vwap_daily_reset(ts_a, high_a, low_a, close_a, vol_a)

    bullish = close_a > open_a
    bearish = close_a < open_a
    body = np.abs(close_a - open_a)
    bar_range = high_a - low_a
    upper_half_close = (close_a - low_a) > (bar_range / 2.0)
    lower_half_close = (high_a - close_a) > (bar_range / 2.0)

    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)
    entry_long = np.full(n, np.nan, dtype=np.float64)
    sl_long = np.full(n, np.nan, dtype=np.float64)
    entry_short = np.full(n, np.nan, dtype=np.float64)
    sl_short = np.full(n, np.nan, dtype=np.float64)

    min_start = max(SWING_LOOKBACK, ATR_PERIOD) + 2
    end_i = n - (TIME_STOP_BARS + 5)

    for i in range(min_start, end_i):
        atr = atr_a[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        cur_vwap = vwap_a[i]
        if not np.isfinite(cur_vwap):
            continue
        cl = close_a[i]
        body_i = body[i]

        # LONG: price ≤ vwap - dist_atr × atr, bullish reversal in upper half
        if cl <= (cur_vwap - VWAP_DIST_ATR * atr):
            if bullish[i] and (body_i >= VWAP_BODY_ATR * atr) and upper_half_close[i]:
                swing_lo = low_a[max(0, i - SWING_LOOKBACK + 1):i + 1].min()
                sl = swing_lo - SL_ATR_BUFFER_VWAP * atr
                if sl < cl:
                    long_mask[i] = True
                    entry_long[i] = cl
                    sl_long[i] = sl

        # SHORT: price ≥ vwap + dist_atr × atr, bearish reversal in lower half
        if cl >= (cur_vwap + VWAP_DIST_ATR * atr):
            if bearish[i] and (body_i >= VWAP_BODY_ATR * atr) and lower_half_close[i]:
                swing_hi = high_a[max(0, i - SWING_LOOKBACK + 1):i + 1].max()
                sl = swing_hi + SL_ATR_BUFFER_VWAP * atr
                if sl > cl:
                    short_mask[i] = True
                    entry_short[i] = cl
                    sl_short[i] = sl

    return (long_mask, short_mask, entry_long, sl_long, entry_short, sl_short)


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────
class ComboETHAbsorbVwapStrategy(Strategy):
    """Ensemble of absorption_bubble + scalper_vwap_mr on ETH 5m."""
    name = "combo_eth_absorb_vwap"

    # Cache per (symbol, n_bars) of precomputed signal arrays
    _SIG_CACHE: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    def param_grid(self):
        for mode in ["absorb_only", "vwap_only", "ensemble_AND", "ensemble_OR"]:
            for window_bars in [1, 3, 6]:
                for tp_rr in [1.5, 2.0]:
                    yield {"mode": mode, "window_bars": window_bars, "tp_rr": tp_rr}

    def _ensure_signals(self, df: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
        key = (symbol, len(df))
        if key in self._SIG_CACHE:
            return self._SIG_CACHE[key]
        (a_long, a_short, a_entry_l, a_sl_l, a_entry_s, a_sl_s, atr_a
         ) = compute_absorption_signals(df)
        (v_long, v_short, v_entry_l, v_sl_l, v_entry_s, v_sl_s
         ) = compute_vwap_mr_signals(df)
        sigs = {
            "absorb_long": a_long, "absorb_short": a_short,
            "absorb_entry_long": a_entry_l, "absorb_sl_long": a_sl_l,
            "absorb_entry_short": a_entry_s, "absorb_sl_short": a_sl_s,
            "vwap_long": v_long, "vwap_short": v_short,
            "vwap_entry_long": v_entry_l, "vwap_sl_long": v_sl_l,
            "vwap_entry_short": v_entry_s, "vwap_sl_short": v_sl_s,
            "atr": atr_a,
        }
        self._SIG_CACHE[key] = sigs
        return sigs

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 100:
            return []
        symbol = df.attrs.get("symbol", "ETH")

        sigs = self._ensure_signals(df, symbol)
        a_long = sigs["absorb_long"]; a_short = sigs["absorb_short"]
        a_entry_l = sigs["absorb_entry_long"]; a_sl_l = sigs["absorb_sl_long"]
        a_entry_s = sigs["absorb_entry_short"]; a_sl_s = sigs["absorb_sl_short"]
        v_long = sigs["vwap_long"]; v_short = sigs["vwap_short"]
        v_entry_l = sigs["vwap_entry_long"]; v_sl_l = sigs["vwap_sl_long"]
        v_entry_s = sigs["vwap_entry_short"]; v_sl_s = sigs["vwap_sl_short"]

        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        ts_a = df.index.to_numpy()

        mode = str(params["mode"])
        window_bars = int(params["window_bars"])
        tp_rr = float(params["tp_rr"])

        trades: List[Trade] = []
        open_until = -1
        max_bars_exit = TIME_STOP_BARS

        # For each bar i we determine if a trade should be emitted under this mode.
        end_i = n - (TIME_STOP_BARS + 5)
        for i in range(window_bars + 2, end_i):
            if i <= open_until:
                continue

            # Look for any signal at bar i (most recent confirmation when ensembling)
            # Determine per-mode entry payload:
            entry = sl = np.nan
            side = None
            ensemble_flag = False

            # Helper: most recent absorption signal in [i-window_bars+1, i] per side
            def _last_in_window(mask, lo, hi):
                # return index in [lo, hi] inclusive of last True, else -1
                # vectorised: argmax of reversed slice if any True
                seg = mask[lo:hi + 1]
                if not seg.any():
                    return -1
                # last True index in seg
                idx_in_seg = len(seg) - 1 - np.argmax(seg[::-1])
                return lo + int(idx_in_seg)

            lo = max(0, i - window_bars + 1)
            hi = i

            if mode == "absorb_only":
                if a_long[i]:
                    side = "long"; entry = a_entry_l[i]; sl = a_sl_l[i]
                elif a_short[i]:
                    side = "short"; entry = a_entry_s[i]; sl = a_sl_s[i]

            elif mode == "vwap_only":
                if v_long[i]:
                    side = "long"; entry = v_entry_l[i]; sl = v_sl_l[i]
                elif v_short[i]:
                    side = "short"; entry = v_entry_s[i]; sl = v_sl_s[i]

            elif mode == "ensemble_AND":
                # Require BOTH same-side within window
                a_l_idx = _last_in_window(a_long, lo, hi)
                v_l_idx = _last_in_window(v_long, lo, hi)
                a_s_idx = _last_in_window(a_short, lo, hi)
                v_s_idx = _last_in_window(v_short, lo, hi)
                # Only emit when current bar i carries the most recent confirmation
                # (otherwise we'd retrigger every bar for the duration of the window).
                long_ok = (a_l_idx >= 0 and v_l_idx >= 0 and max(a_l_idx, v_l_idx) == i)
                short_ok = (a_s_idx >= 0 and v_s_idx >= 0 and max(a_s_idx, v_s_idx) == i)
                if long_ok and not short_ok:
                    last_idx = max(a_l_idx, v_l_idx)
                    if last_idx == a_l_idx:
                        side = "long"; entry = a_entry_l[a_l_idx]; sl = a_sl_l[a_l_idx]
                    else:
                        side = "long"; entry = v_entry_l[v_l_idx]; sl = v_sl_l[v_l_idx]
                    ensemble_flag = True
                elif short_ok and not long_ok:
                    last_idx = max(a_s_idx, v_s_idx)
                    if last_idx == a_s_idx:
                        side = "short"; entry = a_entry_s[a_s_idx]; sl = a_sl_s[a_s_idx]
                    else:
                        side = "short"; entry = v_entry_s[v_s_idx]; sl = v_sl_s[v_s_idx]
                    ensemble_flag = True
                # If both sides simultaneously: skip (conflict)

            elif mode == "ensemble_OR":
                # Trade if either side fires at bar i (de-dup within window: only act
                # when current bar i is the trigger, i.e., signal exists at i).
                long_any = a_long[i] or v_long[i]
                short_any = a_short[i] or v_short[i]
                if long_any and not short_any:
                    side = "long"
                    # Prefer the engine that fired AT bar i (could be both)
                    if a_long[i]:
                        entry = a_entry_l[i]; sl = a_sl_l[i]
                    else:
                        entry = v_entry_l[i]; sl = v_sl_l[i]
                elif short_any and not long_any:
                    side = "short"
                    if a_short[i]:
                        entry = a_entry_s[i]; sl = a_sl_s[i]
                    else:
                        entry = v_entry_s[i]; sl = v_sl_s[i]

            if side is None or not np.isfinite(entry) or not np.isfinite(sl):
                continue

            # Compute TP from R-multiple
            if side == "long":
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + tp_rr * risk
            else:
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - tp_rr * risk

            exit_idx, exit_price, reason = _walk_forward_exit_np(
                open_a, high_a, low_a, close_a, i, side,
                entry, sl, tp, max_bars_exit
            )
            trades.append(Trade(
                symbol=symbol, side=side,
                entry_price=float(entry), exit_price=float(exit_price),
                notional_usd=NOTIONAL_USD,
                holding_sec=int((exit_idx - i) * BAR_SEC),
                entry_ts=pd.Timestamp(ts_a[i]),
                exit_ts=pd.Timestamp(ts_a[exit_idx]),
                exit_reason=reason,
                extra={"mode": mode, "window_bars": window_bars,
                       "tp_rr": tp_rr, "ensemble": bool(ensemble_flag)},
            ))
            open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


# ─────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────
def render_combo_report(results: Dict[str, Any], out_path: Path) -> str:
    """Build the W/F study summary tailored to the A/B ensemble question."""
    cells = list(results["cells"].items())

    # Per-mode stats: aggregate across windows × tp_rr × variants
    # We focus on per-cell IS_EV/n_total/PASS counts grouped by mode.
    by_mode: Dict[str, List[Dict[str, Any]]] = {
        "absorb_only": [], "vwap_only": [],
        "ensemble_AND": [], "ensemble_OR": [],
    }
    for k, v in cells:
        # cell key form: ETH|5m|mode=...|variant
        # cell_id is "mode=...window_bars=...tp_rr=..."
        sym, tf, cell_id, variant = k.split("|", 3)
        mode = "?"
        for m in by_mode.keys():
            if f"mode={m}_" in cell_id or cell_id.startswith(f"mode={m}"):
                mode = m
                break
        if mode == "?":
            continue
        by_mode[mode].append({
            "cell_id": cell_id, "variant": variant,
            "n_total": v["n_total"], "is_n": v["is_n"], "is_ev": v["is_ev"],
            "q3_n": v["q3_n"], "q3_ev": v["q3_ev"],
            "q4_n": v["q4_n"], "q4_ev": v["q4_ev"],
            "oos_n": v["oos_n"], "oos_ev": v["oos_ev"],
            "verdict": v["verdict"], "wr_oos": v["win_rate_oos"],
            "gap_pct": v["gap_pct"],
        })

    def mode_stats(rows):
        if not rows:
            return None
        # Per-trade EV (weighted by trades) and per-cell EV (mean)
        n_cells = len(rows)
        n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
        avg_is_ev_cells = sum(r["is_ev"] for r in rows) / n_cells
        avg_q4_ev_cells = sum(r["q4_ev"] for r in rows) / n_cells
        # Trade-weighted EV (sum ev*n / sum n)
        is_n_sum = sum(r["is_n"] for r in rows)
        is_ev_w = (sum(r["is_ev"] * r["is_n"] for r in rows) / is_n_sum
                   if is_n_sum > 0 else 0.0)
        oos_n_sum = sum(r["oos_n"] for r in rows)
        oos_ev_w = (sum(r["oos_ev"] * r["oos_n"] for r in rows) / oos_n_sum
                    if oos_n_sum > 0 else 0.0)
        n_total_sum = sum(r["n_total"] for r in rows)
        avg_n_total = n_total_sum / n_cells
        return {
            "n_cells": n_cells, "n_pass": n_pass,
            "pass_rate": n_pass / n_cells if n_cells else 0.0,
            "avg_is_ev_per_cell": avg_is_ev_cells,
            "avg_q4_ev_per_cell": avg_q4_ev_cells,
            "trade_w_is_ev": is_ev_w,
            "trade_w_oos_ev": oos_ev_w,
            "is_n_sum": is_n_sum,
            "oos_n_sum": oos_n_sum,
            "n_total_sum": n_total_sum,
            "avg_trades_per_cell": avg_n_total,
        }

    stats = {m: mode_stats(rows) for m, rows in by_mode.items()}

    # ──── Summary tables ────
    lines = [
        "# Combo ETH (Absorption + VWAP-MR) — Walk-Forward Report",
        "",
        f"Generated: {results.get('finished_at','')}",
        f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}  Wall: {results.get('wall_sec','?')}s",
        "",
        "## Hypothesis",
        "When BOTH absorption_bubble AND scalper_vwap_mr fire same-side on ETH within "
        "a ~15-min window, the ensemble has higher per-trade EV than either signal alone.",
        "",
        "## Headline A/B by mode (across windows × tp_rr × variants)",
        "",
        "| mode | n_cells | PASS | pass_rate | avg IS_EV/cell | avg Q4_EV/cell | trade-weighted IS_EV | trade-weighted OOS_EV | total trades | avg trades/cell |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ["absorb_only", "vwap_only", "ensemble_AND", "ensemble_OR"]:
        s = stats[m]
        if s is None:
            lines.append(f"| {m} | 0 | 0 | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {m} | {s['n_cells']} | {s['n_pass']} | {s['pass_rate']*100:.0f}% | "
            f"${s['avg_is_ev_per_cell']:+.3f} | ${s['avg_q4_ev_per_cell']:+.3f} | "
            f"${s['trade_w_is_ev']:+.3f} | ${s['trade_w_oos_ev']:+.3f} | "
            f"{s['n_total_sum']} | {s['avg_trades_per_cell']:.0f} |"
        )

    # Best cell per mode
    lines += ["", "## Best cell per mode by Q4 EV", "",
              "| mode | cell | variant | n_total | IS_n | IS_EV | Q3_EV | Q4_EV | OOS_n | gap% | WR_oos | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ["absorb_only", "vwap_only", "ensemble_AND", "ensemble_OR"]:
        rows = by_mode[m]
        if not rows:
            lines.append(f"| {m} | – | – | – | – | – | – | – | – | – | – | – |")
            continue
        rows_sorted = sorted(rows, key=lambda r: r["q4_ev"], reverse=True)
        b = rows_sorted[0]
        gap = f"{b['gap_pct']*100:.0f}%" if b["gap_pct"] is not None else "–"
        lines.append(
            f"| {m} | {b['cell_id']} | {b['variant']} | {b['n_total']} | {b['is_n']} | "
            f"${b['is_ev']:+.3f} | ${b['q3_ev']:+.3f} | ${b['q4_ev']:+.3f} | "
            f"{b['oos_n']} | {gap} | {b['wr_oos']*100:.0f}% | {b['verdict']} |"
        )

    # Per-trade EV uplift of ensemble_AND vs better single
    lines += ["", "## Per-trade EV uplift: ensemble_AND vs best single", ""]
    if stats["ensemble_AND"] and stats["absorb_only"] and stats["vwap_only"]:
        absorb_w = stats["absorb_only"]["trade_w_oos_ev"]
        vwap_w = stats["vwap_only"]["trade_w_oos_ev"]
        ens_w = stats["ensemble_AND"]["trade_w_oos_ev"]
        better_single = max(absorb_w, vwap_w)
        better_name = "absorb_only" if absorb_w >= vwap_w else "vwap_only"
        uplift = ens_w - better_single
        lines += [
            f"- absorb_only trade-weighted OOS_EV  = ${absorb_w:+.3f}",
            f"- vwap_only   trade-weighted OOS_EV  = ${vwap_w:+.3f}",
            f"- ensemble_AND trade-weighted OOS_EV = ${ens_w:+.3f}",
            f"- Better single = **{better_name}** (${better_single:+.3f})",
            f"- Uplift ensemble_AND − better_single = **${uplift:+.3f}/trade**",
            "",
        ]
        # Also IS comparison
        absorb_is = stats["absorb_only"]["trade_w_is_ev"]
        vwap_is = stats["vwap_only"]["trade_w_is_ev"]
        ens_is = stats["ensemble_AND"]["trade_w_is_ev"]
        better_is = max(absorb_is, vwap_is)
        lines += [
            f"- (IS reference: absorb=${absorb_is:+.3f}, vwap=${vwap_is:+.3f}, "
            f"ensemble=${ens_is:+.3f}, uplift IS = ${ens_is - better_is:+.3f}/trade)",
            "",
        ]

    # Verdict distribution
    verdicts: Dict[str, int] = {}
    for v in results["cells"].values():
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    lines += ["## Verdict distribution", "",
              "| Verdict | Count |", "|---|---|"]
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        lines.append(f"| {v} | {n} |")

    # Top PASS cells
    lines += ["", "## Top PASS cells by Q4 EV (all modes)", "",
              "| mode | cell | variant | n_total | IS_n | IS_EV | Q3_EV | Q4_EV | gap% | WR_oos |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    pass_rows = []
    for m, rows in by_mode.items():
        for r in rows:
            if r["verdict"] == "PASS":
                pass_rows.append((m, r))
    pass_rows.sort(key=lambda x: x[1]["q4_ev"], reverse=True)
    if not pass_rows:
        lines.append("| (none) | | | | | | | | | |")
    else:
        for m, r in pass_rows[:25]:
            gap = f"{r['gap_pct']*100:.0f}%" if r["gap_pct"] is not None else "–"
            lines.append(
                f"| {m} | {r['cell_id']} | {r['variant']} | {r['n_total']} | {r['is_n']} | "
                f"${r['is_ev']:+.3f} | ${r['q3_ev']:+.3f} | ${r['q4_ev']:+.3f} | "
                f"{gap} | {r['wr_oos']*100:.0f}% |"
            )

    # Recommendation
    lines += ["", "## Recommendation", ""]
    if stats["ensemble_AND"] and stats["absorb_only"] and stats["vwap_only"]:
        ens = stats["ensemble_AND"]
        absorb = stats["absorb_only"]; vwap = stats["vwap_only"]
        better_single = max(absorb["trade_w_oos_ev"], vwap["trade_w_oos_ev"])
        ens_oos = ens["trade_w_oos_ev"]
        ens_pass = ens["pass_rate"]
        if ens["n_total_sum"] < 30:
            rec = ("STICK WITH SINGLES — ensemble too rare "
                   f"(only {ens['n_total_sum']} trades across all cells; "
                   "no statistical confidence in EV uplift).")
        elif ens_oos > better_single + 0.10:
            rec = ("SHIP ENSEMBLE AS A+ BOOSTER — ensemble_AND OOS_EV exceeds "
                   "best single by >+$0.10/trade with much fewer trades. Use as "
                   "high-conviction grade for ETH.")
        elif ens_oos > better_single:
            rec = ("MARGINAL UPLIFT — ensemble_AND OOS_EV is slightly better than "
                   "the best single but not enough to justify a separate engine. "
                   "Could be added as a TAG (boost grade) rather than a standalone signal.")
        elif abs(ens_oos - better_single) < 0.05:
            rec = ("NO UPLIFT — ensemble_AND OOS_EV is statistically the same as "
                   "the best single. Stick with individual scanners.")
        else:
            rec = ("ENSEMBLE LOGIC IS BROKEN — ensemble_AND has LOWER OOS_EV per "
                   "trade than the better of the two singles. Investigate before "
                   "shipping; may indicate that agreement happens in the worst-EV "
                   "regime (e.g. high-vol whipsaw).")
        lines.append(rec)

    txt = "\n".join(lines) + "\n"
    out_path.write_text(txt)
    return txt


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time
    print("=== ETH ABSORPTION × VWAP-MR ENSEMBLE — W/F STUDY ===\n")
    print("Cell grid: 4 modes × 3 windows × 2 tp_rr = 24 cells per pair")
    print("Symbols: ETH × 5m × 3 fee variants = 72 cell-variants total\n")

    t0 = _time.time()
    engine = _PatchedEngine(
        study=ComboETHAbsorbVwapStrategy(),
        symbols=["ETH"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "combo_eth_absorb_vwap",
        verbose=False,
    )
    results = engine.run()
    wall = _time.time() - t0
    print(f"\n=== W/F engine done ({wall:.1f}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")

    # Custom combo report
    out_md = ROOT / "storage" / "wf_studies" / "combo_eth_absorb_vwap" / "combo_report.md"
    txt = render_combo_report(results, out_md)
    print(f"Wrote: {out_md}")
    # Also write to /tmp for the agent
    tmp_path = Path("/tmp/agent_combo_eth_absorb_vwap.md")
    tmp_path.write_text(txt)
    print(f"Wrote: {tmp_path}")

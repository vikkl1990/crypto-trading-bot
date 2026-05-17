"""W/F STUDY: Chop-Optimized Range-bounded VWAP Mean Revert.

Tests a NEW chop-specific scanner candidate for the VN Edge bot. The current
structure_bounce strategy collapses in chop regimes (53% WR vs 88% on trending
days). This pattern targets sideways/low-vol markets with mean-revert entries.

Setup (LONG; mirror for SHORT):
  Step 1 — CHOP REGIME: 4h ATR percentile rank ≤ atr_rank_max (default 0.30) on
           the same symbol. Skip if ATR rank > threshold (= not chop).
  Step 2 — RANGE: in last 60 5m bars, (highest_high − lowest_low) ≤
           range_atr_max × current 14-period 5m ATR. Price has been bracketed.
  Step 3 — VWAP DISTANCE: current price ≤ VWAP − vwap_dist_atr × ATR.
           Price overstretched below VWAP within bracket.
  Step 4 — CONFIRMATION: most recent closed bar = bullish reversal candle.
           Body ≥ body_atr_min × ATR, close > open, close in upper half of range.
  Step 5 — ENTRY at confirmation candle close.
           STOP at swing_low (last 10 bars) − 0.3 × ATR OR fixed 0.6 × ATR
           (whichever is closer / tighter).
           TARGET: VWAP itself, fixed_1r, or fixed_2r.

Hard time stop 30 minutes (6 bars). Notional $400 fixed.

Implementation: vectorized numpy, 4h ATR percentile pre-computed once per
symbol & cached. Total grid 81 cells per pair.
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
TIME_STOP_SEC = 1800       # 30 minutes hard time stop
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
SL_ATR_BUFFER = 0.3        # stop = swing - 0.3 × ATR
SL_FIXED_ATR = 0.6         # fallback fixed stop at 0.6 × ATR
SWING_LOOKBACK = 10        # bars for swing low/high
RANGE_LOOKBACK = 60        # 60 5m bars = 5h window
ATR4H_LOOKBACK = 100       # 4h percentile window
NOTIONAL_USD = 400.0


def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """ATR via True Range, simple rolling mean. Fully numpy."""
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


def compute_vwap_daily_reset(ts_a: np.ndarray, high: np.ndarray, low: np.ndarray,
                             close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Rolling VWAP that resets at each UTC midnight."""
    n = len(ts_a)
    vwap = np.full(n, np.nan, dtype=np.float64)
    typical = (high + low + close) / 3.0
    pv = typical * volume
    # Identify day boundaries: same UTC day = same group
    days = pd.DatetimeIndex(ts_a).floor("D").asi8  # epoch ns at day start
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


def compute_4h_atr_percentile_for_5m(
    ts_5m_a: np.ndarray,
    df_4h: pd.DataFrame,
    atr_period: int = 14,
    rank_window: int = 100,
) -> np.ndarray:
    """For each 5m bar timestamp, return the 4h ATR percentile rank (0..1).

    Algorithm:
      1. Compute 4h ATR(14) on the 4h dataframe.
      2. For each 4h bar, compute percentile rank within the trailing
         rank_window 4h bars (excludes current bar from its own ranking).
      3. Map each 5m timestamp to the most-recent CLOSED 4h bar's percentile.
    """
    high_4h = df_4h["high"].to_numpy(dtype=np.float64)
    low_4h = df_4h["low"].to_numpy(dtype=np.float64)
    close_4h = df_4h["close"].to_numpy(dtype=np.float64)
    atr_4h = add_atr_np(high_4h, low_4h, close_4h, atr_period)
    n4 = len(atr_4h)

    pct_4h = np.full(n4, np.nan, dtype=np.float64)
    for i in range(rank_window, n4):
        if not np.isfinite(atr_4h[i]):
            continue
        window = atr_4h[i - rank_window:i]  # exclude current
        valid = window[np.isfinite(window)]
        if len(valid) < 10:
            continue
        # percentile of current ATR vs window
        cur = atr_4h[i]
        pct_4h[i] = (valid < cur).sum() / len(valid)

    # Map 5m timestamps to most-recent closed 4h bar percentile
    ts_4h_a = df_4h.index.to_numpy()
    # Build a sorted array of 4h bar END times (each 4h bar represents
    # data from ts to ts+4h, so the bar is "closed" at ts+4h).
    # We want: for 5m bar at time T, use the percentile of the 4h bar
    # whose END time is the latest <= T.
    # ts_4h_a is the bar OPEN time, so end = ts + 4h.
    ts_4h_end_ns = (ts_4h_a.astype("datetime64[ns]").astype(np.int64) +
                    int(4 * 3600 * 1e9))
    ts_5m_ns = ts_5m_a.astype("datetime64[ns]").astype(np.int64)

    # searchsorted to find for each 5m the index of latest 4h with end<=T
    idx = np.searchsorted(ts_4h_end_ns, ts_5m_ns, side="right") - 1
    pct_5m = np.full(len(ts_5m_ns), np.nan, dtype=np.float64)
    valid_mask = idx >= 0
    valid_idx = idx[valid_mask]
    pct_5m[valid_mask] = pct_4h[valid_idx]
    return pct_5m


def _walk_forward_exit_np(open_a: np.ndarray, high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, vwap_a: np.ndarray,
                          entry_idx: int, side: str,
                          entry: float, sl: float, tp: float, tp_mode: str,
                          max_bars: int) -> Tuple[int, float, str]:
    """Bar-by-bar walk to first SL/TP/time-stop. Numpy version.

    For tp_mode == "vwap", TP is dynamic — check VWAP at each bar.
    Returns (exit_idx, exit_price, reason).
    """
    n = len(close_a)
    end = min(entry_idx + 1 + max_bars, n)
    for j in range(entry_idx + 1, end):
        if side == "long":
            if low_a[j] <= sl:
                return (j, sl, "sl_hit")
            # Dynamic VWAP target
            if tp_mode == "vwap":
                cur_vwap = vwap_a[j]
                if np.isfinite(cur_vwap) and high_a[j] >= cur_vwap:
                    return (j, cur_vwap, "tp_hit")
            else:
                if high_a[j] >= tp:
                    return (j, tp, "tp_hit")
        else:
            if high_a[j] >= sl:
                return (j, sl, "sl_hit")
            if tp_mode == "vwap":
                cur_vwap = vwap_a[j]
                if np.isfinite(cur_vwap) and low_a[j] <= cur_vwap:
                    return (j, cur_vwap, "tp_hit")
            else:
                if low_a[j] <= tp:
                    return (j, tp, "tp_hit")
        if (j - entry_idx) >= max_bars:
            return (j, close_a[j], "time_stop")
    last_idx = min(entry_idx + max_bars, n - 1)
    return (last_idx, close_a[last_idx], "forced_end")


# Cache for 4h ATR percentile per symbol — populated once, reused across cells
_PCT_CACHE: Dict[str, np.ndarray] = {}


class ChopVwapMrStrategy(Strategy):
    """Chop-Optimized Range-bounded VWAP Mean Revert — 5-step pattern.

    Param sweep (3 × 3 × 3 × 3 = 81 cells, range_atr_max & body_atr_min fixed):
      atr_rank_max: chop regime threshold on 4h ATR percentile (0.20, 0.30, 0.40)
      vwap_dist_atr: VWAP distance trigger in ATR (0.4, 0.6, 0.8)
      sl_mode: "swing" or "fixed" stop
      tp_mode: "vwap" (dynamic target = current VWAP), "fixed_1r", "fixed_2r"
    """
    name = "chop_vwap_mr"

    def param_grid(self):
        # range_atr_max calibrated empirically: chop bars have 60-bar range
        # of 5-13 × current 14-bar ATR (median ~6.4), so 1.5x ATR was unreachable.
        # Use 6.0 / 8.0 / 10.0 to bracket realistic chop ranges.
        for atr_rank_max in [0.20, 0.30, 0.40]:
            for range_atr_max in [6.0, 8.0, 10.0]:
                for vwap_dist_atr in [0.4, 0.6, 0.8]:
                    for sl_mode in ["swing", "fixed"]:
                        for tp_mode in ["vwap", "fixed_1r", "fixed_2r"]:
                            yield {
                                "atr_rank_max": atr_rank_max,
                                "range_atr_max": range_atr_max,
                                "vwap_dist_atr": vwap_dist_atr,
                                "body_atr_min": 0.4,
                                "sl_mode": sl_mode,
                                "tp_mode": tp_mode,
                            }

    def _ensure_pct_cache(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Load + cache 4h ATR percentile aligned to the 5m index."""
        if symbol in _PCT_CACHE:
            cached = _PCT_CACHE[symbol]
            if len(cached) == len(df):
                return cached
        path_4h = ROOT / "storage" / "candle_cache" / f"{symbol}_USDT_4h.parquet"
        if not path_4h.exists():
            raise FileNotFoundError(f"No 4h cache for {symbol}: {path_4h}")
        df_4h = pd.read_parquet(path_4h)
        if "datetime" in df_4h.columns:
            df_4h = df_4h.set_index("datetime")
        if df_4h.index.tz is None:
            df_4h.index = df_4h.index.tz_localize("UTC")
        df_4h = df_4h.sort_index()
        ts_5m_a = df.index.to_numpy()
        pct_5m = compute_4h_atr_percentile_for_5m(
            ts_5m_a, df_4h, atr_period=14, rank_window=ATR4H_LOOKBACK
        )
        _PCT_CACHE[symbol] = pct_5m
        return pct_5m

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 200:
            return []
        symbol = df.attrs.get("symbol", "BTC")

        # Pre-compute numpy arrays once per cell (atr/vwap independent of params)
        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        volume_a = df["volume"].to_numpy(dtype=np.float64)
        ts_a = df.index.to_numpy()

        atr_a = add_atr_np(high_a, low_a, close_a, 14)
        vwap_a = compute_vwap_daily_reset(ts_a, high_a, low_a, close_a, volume_a)

        try:
            pct_a = self._ensure_pct_cache(df, symbol)
        except FileNotFoundError:
            return []

        bullish = close_a > open_a
        bearish = close_a < open_a
        body = np.abs(close_a - open_a)
        bar_range = high_a - low_a
        upper_half_close = (close_a - low_a) > (bar_range / 2.0)
        lower_half_close = (high_a - close_a) > (bar_range / 2.0)

        atr_rank_max = float(params["atr_rank_max"])
        range_atr_max = float(params["range_atr_max"])
        vwap_dist_atr = float(params["vwap_dist_atr"])
        body_atr_min = float(params["body_atr_min"])
        sl_mode = str(params["sl_mode"])
        tp_mode = str(params["tp_mode"])

        trades: List[Trade] = []
        open_until = -1
        min_lookback = max(RANGE_LOOKBACK, SWING_LOOKBACK + 5)
        max_bars_exit = TIME_STOP_BARS
        end_i = n - (TIME_STOP_BARS + 5)

        for i in range(min_lookback, end_i):
            if i <= open_until:
                continue
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            cur_pct = pct_a[i]
            if not np.isfinite(cur_pct) or cur_pct > atr_rank_max:
                continue  # not chop
            cur_vwap = vwap_a[i]
            if not np.isfinite(cur_vwap):
                continue

            # Range check (last RANGE_LOOKBACK bars including current)
            r_start = i - RANGE_LOOKBACK + 1
            r_high = high_a[r_start:i + 1].max()
            r_low = low_a[r_start:i + 1].min()
            if (r_high - r_low) > range_atr_max * atr:
                continue  # not bracketed

            cur_close = close_a[i]
            cur_body = body[i]

            # ─── LONG SETUP — price below VWAP, bullish reversal ────
            if cur_close <= (cur_vwap - vwap_dist_atr * atr):
                if not bullish[i]:
                    continue
                if cur_body < body_atr_min * atr:
                    continue
                if not upper_half_close[i]:
                    continue
                # Stop selection
                swing_lo = low_a[max(0, i - SWING_LOOKBACK + 1):i + 1].min()
                sl_swing = swing_lo - SL_ATR_BUFFER * atr
                sl_fixed = cur_close - SL_FIXED_ATR * atr
                if sl_mode == "swing":
                    sl = max(sl_swing, sl_fixed)  # tighter (closer to entry)
                else:
                    sl = sl_fixed
                if sl >= cur_close:
                    continue
                risk = cur_close - sl
                if tp_mode == "fixed_1r":
                    tp = cur_close + risk
                elif tp_mode == "fixed_2r":
                    tp = cur_close + 2.0 * risk
                else:  # vwap
                    tp = cur_vwap  # initial, but walk uses dynamic
                    if tp <= cur_close:
                        continue
                if tp <= cur_close:
                    continue

                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    open_a, high_a, low_a, close_a, vwap_a, i, "long",
                    cur_close, sl, tp, tp_mode, max_bars_exit
                )
                trades.append(Trade(
                    symbol=symbol, side="long",
                    entry_price=float(cur_close), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - i) * BAR_SEC),
                    entry_ts=pd.Timestamp(ts_a[i]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"atr_pct": float(cur_pct), "vwap": float(cur_vwap),
                           "atr": float(atr), "tp_mode": tp_mode,
                           "sl_mode": sl_mode},
                ))
                open_until = exit_idx
                continue

            # ─── SHORT SETUP — price above VWAP, bearish reversal ────
            if cur_close >= (cur_vwap + vwap_dist_atr * atr):
                if not bearish[i]:
                    continue
                if cur_body < body_atr_min * atr:
                    continue
                if not lower_half_close[i]:
                    continue
                swing_hi = high_a[max(0, i - SWING_LOOKBACK + 1):i + 1].max()
                sl_swing = swing_hi + SL_ATR_BUFFER * atr
                sl_fixed = cur_close + SL_FIXED_ATR * atr
                if sl_mode == "swing":
                    sl = min(sl_swing, sl_fixed)  # tighter
                else:
                    sl = sl_fixed
                if sl <= cur_close:
                    continue
                risk = sl - cur_close
                if tp_mode == "fixed_1r":
                    tp = cur_close - risk
                elif tp_mode == "fixed_2r":
                    tp = cur_close - 2.0 * risk
                else:  # vwap
                    tp = cur_vwap
                    if tp >= cur_close:
                        continue
                if tp >= cur_close:
                    continue

                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    open_a, high_a, low_a, close_a, vwap_a, i, "short",
                    cur_close, sl, tp, tp_mode, max_bars_exit
                )
                trades.append(Trade(
                    symbol=symbol, side="short",
                    entry_price=float(cur_close), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - i) * BAR_SEC),
                    entry_ts=pd.Timestamp(ts_a[i]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"atr_pct": float(cur_pct), "vwap": float(cur_vwap),
                           "atr": float(atr), "tp_mode": tp_mode,
                           "sl_mode": sl_mode},
                ))
                open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== CHOP-OPTIMIZED VWAP MEAN REVERT — W/F STUDY ===\n")
    print("Cell grid: 3 (atr_rank) × 3 (range) × 3 (vwap_dist) × 2 (sl) × 3 (tp) = 162 cells per pair")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants = 1944 cell-variants total")
    print("4h ATR percentile cached once per symbol (computed at first cell)\n")

    engine = _PatchedEngine(
        study=ChopVwapMrStrategy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "chop_vwap_mr",
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

"""W/F STUDY: BTC SHORT-only Break Block Reversal + HTF mature uptrend filter.

Asymmetric edge resurrection test. Earlier `break_block_reversal_study` surfaced
strong asymmetric edge:
  - BTC SHORT-after-bullish-break: n=22, +$0.52/trade, 77% WR (strong edge)
  - BTC LONG-after-bearish-break:  n=19, -$0.02/trade, 53% WR (no edge)

Hypothesis: in mature uptrends, bullish breaks are reliably faded (smart-money
distribution into retail breakout buyers). Adding HTF up-trend filter (4h EMA21
RISING) ensures we only fire SHORT-after-bullish-break in genuine uptrends, not
chop with bullish bars.

A/B test: htf_filter ∈ {True, False} side-by-side over the same setup grid.

Setup (SHORT-ONLY):
  Step 1 — BULLISH BREAK: bar i closes ABOVE rolling_high(20) with
           displacement ≥ break_atr × ATR
  Step 2 — REVERSAL: within reversal_lookback bars, price re-enters back
           below rolling_high with body ≥ reversal_body_atr × ATR
  Step 3 — RETEST: within retest_window bars, price retests rolling_high level
  Step 4 — CONFIRMATION: bearish candle at retest, body ≥ confirm_body_atr × ATR
  Step 5 — ENTRY: at confirmation candle close (SHORT)
  Step 6 — STOP: above retest extreme + 0.3 × ATR
  Step 7 — TARGET: 1.5R or 2.0R

HTF up-trend filter (when htf_filter=True):
  - 4h EMA21 must be RISING for last `htf_window` bars
  - "Rising" = current EMA21 > EMA21 N bars ago by ≥ `htf_slope_atr` × 4h ATR

Hard time stop: 30 minutes (6 bars). Notional $400.

If passes W/F, ship as PAPER-ONLY engine (asymmetric edges need long observation
before live trading).

Param grid (reduced): break_atr=0.8 fixed, tp_rr=2.0 fixed → 18 cells.
Full A/B: 2 (htf_filter) × 3 (htf_window) × 3 (htf_slope_atr) × ... = 18 base.
BTC ONLY × 3 fee variants = 54 cell-variants.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

BAR_SEC = 300
TIME_STOP_SEC = 1800  # 30 min
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
NOTIONAL_USD = 400.0
RETEST_TOL_ATR = 0.3
SL_PAD_ATR = 0.3
ROLLING_WINDOW = 20
H4_SEC = 14400
H4_EMA_PERIOD = 21
H4_ATR_PERIOD = 14
ATR_PERIOD = 14
REVERSAL_LOOKBACK = 5
RETEST_WINDOW = 8
REVERSAL_BODY_ATR = 0.55
CONFIRM_BODY_ATR = 0.65


def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """Numpy ATR via True Range, simple rolling mean."""
    n = len(high)
    if n == 0:
        return np.array([], dtype=np.float64)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        v = a
        if b > v:
            v = b
        if c > v:
            v = c
        tr[i] = v
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr
    csum = np.cumsum(tr)
    for i in range(period - 1, n):
        if i == period - 1:
            atr[i] = csum[i] / period
        else:
            atr[i] = (csum[i] - csum[i - period]) / period
    return atr


def add_ema_np(values: np.ndarray, period: int) -> np.ndarray:
    """Numpy EMA. Simple SMA seed for first `period` values, then standard EMA."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = values[:period].mean()
    out[period - 1] = seed
    for i in range(period, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _walk_forward_exit_np(high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, entry_idx: int, side: str,
                          entry: float, sl: float, tp: float,
                          max_bars: int) -> Tuple[int, float, str]:
    """Bar-by-bar TP/SL detection (vectorized iteration)."""
    n = len(close_a)
    end = min(entry_idx + 1 + max_bars, n)
    for j in range(entry_idx + 1, end):
        if side == "long":
            if low_a[j] <= sl:
                return (j, sl, "sl")
            if high_a[j] >= tp:
                return (j, tp, "tp")
        else:
            if high_a[j] >= sl:
                return (j, sl, "sl")
            if low_a[j] <= tp:
                return (j, tp, "tp")
    last_idx = min(entry_idx + max_bars, n - 1)
    if last_idx <= entry_idx:
        last_idx = min(entry_idx + 1, n - 1)
    return (last_idx, close_a[last_idx], "time_stop")


def _load_4h(symbol: str) -> pd.DataFrame:
    p = ROOT / "storage" / "candle_cache" / f"{symbol}_USDT_4h.parquet"
    df4 = pd.read_parquet(p)
    if "datetime" in df4.columns:
        df4 = df4.set_index("datetime")
    if df4.index.tz is None:
        df4.index = df4.index.tz_localize("UTC")
    df4 = df4.sort_index()
    for c in ("open", "high", "low", "close"):
        df4[c] = df4[c].astype(float)
    return df4


def precompute_h4_uptrend(
    df5_index: pd.DatetimeIndex,
    df4: pd.DataFrame,
    htf_window: int,
    htf_slope_atr: float,
) -> np.ndarray:
    """Return is_uptrend[i] (bool) for each 5m bar i.

    Algorithm:
      1. Compute 4h EMA21 + 4h ATR(14) using closed bars.
      2. For each 5m bar, find the containing 4h bar k.
      3. Use the PRIOR closed 4h bar (k-1) for both EMA and ATR (no look-ahead).
      4. is_uptrend = (ema_now - ema_N_ago) >= htf_slope_atr × atr_4h
         where N = htf_window, both indexed by k-1 and k-1-htf_window.
    """
    n = len(df5_index)
    out = np.zeros(n, dtype=bool)
    if df4.empty or len(df4) < (htf_window + H4_EMA_PERIOD + 5):
        return out

    h4 = df4["high"].to_numpy(dtype=np.float64)
    l4 = df4["low"].to_numpy(dtype=np.float64)
    c4 = df4["close"].to_numpy(dtype=np.float64)
    ema4 = add_ema_np(c4, H4_EMA_PERIOD)
    atr4 = add_atr_np(h4, l4, c4, H4_ATR_PERIOD)

    # Convert both indexes to int microseconds (cache may be ms or us, normalize)
    ts5_us = pd.DatetimeIndex(df5_index).as_unit("us").asi8
    h4_open_us = pd.DatetimeIndex(df4.index).as_unit("us").asi8

    # For each 5m bar, find which 4h bar it belongs to.
    # idx_in_h4 = position such that h4_open_us[idx] <= ts < h4_open_us[idx+1].
    idx_in_h4 = np.searchsorted(h4_open_us, ts5_us, side="right") - 1
    n_h4 = len(h4_open_us)

    for i in range(n):
        k = idx_in_h4[i]
        # Use PRIOR closed 4h bar = k-1 (no look-ahead inside current 4h bar)
        k_prev = k - 1
        k_back = k_prev - htf_window
        if k_back < 0 or k_prev < 0 or k_prev >= n_h4:
            continue
        ema_now = ema4[k_prev]
        ema_back = ema4[k_back]
        ref_atr = atr4[k_prev]
        if not (np.isfinite(ema_now) and np.isfinite(ema_back) and np.isfinite(ref_atr)):
            continue
        if ref_atr <= 0:
            continue
        slope = ema_now - ema_back
        if slope >= htf_slope_atr * ref_atr:
            out[i] = True
    return out


class BTCBreakBlockShortHTFStrategy(Strategy):
    """BTC SHORT-only Break Block Reversal with HTF up-trend filter."""

    name = "combo_btc_break_block_short_htf"

    # Cache 4h data + base 5m arrays per symbol so we don't recompute per cell.
    _h4_cache: Dict[str, pd.DataFrame] = {}
    _base_cache: Dict[str, Dict[str, Any]] = {}
    # Cache htf_uptrend per (symbol, htf_window, htf_slope_atr)
    _htf_cache: Dict[Tuple[str, int, float], np.ndarray] = {}

    def param_grid(self) -> Iterator[Dict[str, Any]]:
        # Reduced grid per task: break_atr=0.8 fixed, tp_rr=2.0 fixed.
        # 2 × 3 × 3 = 18 cells (× 3 fee variants = 54 cell-variants for BTC).
        for htf_filter in [True, False]:
            for htf_window in [3, 5, 10]:
                for htf_slope_atr in [0.1, 0.2, 0.3]:
                    yield {
                        "htf_filter": htf_filter,
                        "htf_window": htf_window,
                        "htf_slope_atr": htf_slope_atr,
                        "break_atr": 0.8,
                        "tp_rr": 2.0,
                    }

    def _build_base(self, df: pd.DataFrame) -> Dict[str, Any]:
        symbol = df.attrs.get("symbol", "BTC")
        if symbol in self._base_cache:
            return self._base_cache[symbol]

        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)

        # Rolling high over PRIOR 20 bars (shifted to avoid look-ahead)
        h_ser = pd.Series(high_a)
        roll_high = h_ser.rolling(ROLLING_WINDOW).max().shift(1).to_numpy()

        if symbol not in self._h4_cache:
            self._h4_cache[symbol] = _load_4h(symbol)

        base = {
            "open": open_a, "high": high_a, "low": low_a, "close": close_a,
            "atr": atr_a, "roll_high": roll_high,
            "ts": df.index.to_numpy(),
            "symbol": symbol,
            "df_index": df.index,
        }
        self._base_cache[symbol] = base
        return base

    def _get_htf_uptrend(self, symbol: str, df_index: pd.DatetimeIndex,
                        htf_window: int, htf_slope_atr: float) -> np.ndarray:
        key = (symbol, htf_window, htf_slope_atr)
        if key in self._htf_cache:
            return self._htf_cache[key]
        df4 = self._h4_cache[symbol]
        arr = precompute_h4_uptrend(df_index, df4, htf_window, htf_slope_atr)
        self._htf_cache[key] = arr
        return arr

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 100:
            return []
        base = self._build_base(df)

        open_a = base["open"]; high_a = base["high"]; low_a = base["low"]
        close_a = base["close"]; atr_a = base["atr"]; roll_high = base["roll_high"]
        ts_a = base["ts"]; symbol = base["symbol"]; df_index = base["df_index"]

        htf_filter = bool(params["htf_filter"])
        htf_window = int(params["htf_window"])
        htf_slope_atr = float(params["htf_slope_atr"])
        break_atr_min = float(params["break_atr"])
        tp_rr = float(params["tp_rr"])

        # HTF up-trend mask (computed once per (symbol, htf_window, htf_slope_atr))
        htf_uptrend = self._get_htf_uptrend(
            symbol, df_index, htf_window, htf_slope_atr
        )

        trades: List[Trade] = []
        open_until = -1
        max_age_bars = TIME_STOP_BARS + 2
        max_lookahead = REVERSAL_LOOKBACK + RETEST_WINDOW + max_age_bars + 5
        start_i = max(ROLLING_WINDOW + ATR_PERIOD, 30)
        end_i = n - max_lookahead

        for i in range(start_i, end_i):
            if i <= open_until:
                continue
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            rh = roll_high[i]
            if not np.isfinite(rh):
                continue

            # HTF gate (A/B test)
            if htf_filter and not htf_uptrend[i]:
                continue

            o = open_a[i]; h = high_a[i]; l = low_a[i]; c = close_a[i]
            body = abs(c - o)

            # ─── SHORT SETUP — bullish break, then reversal back DOWN ───
            # Step 1: bullish break — close above roll_high with displacement
            displacement_up = c - rh
            bullish_break = (
                c > rh
                and displacement_up >= break_atr_min * atr
                and c > o  # bullish bar
            )
            if not bullish_break:
                continue

            broken_level = rh
            tol = RETEST_TOL_ATR * atr

            # Step 2: scan reversal_lookback bars for re-entry below
            reversal_idx: Optional[int] = None
            for j in range(i + 1, min(i + 1 + REVERSAL_LOOKBACK, n)):
                rb_c = close_a[j]; rb_o = open_a[j]
                rb_body = abs(rb_c - rb_o)
                if (
                    rb_c < broken_level
                    and rb_c < rb_o  # bearish bar
                    and rb_body >= REVERSAL_BODY_ATR * atr
                ):
                    reversal_idx = j
                    break
            if reversal_idx is None:
                continue

            # Step 3: scan retest_window bars after reversal for retest
            retest_idx: Optional[int] = None
            retest_high: Optional[float] = None
            for j in range(reversal_idx + 1, min(reversal_idx + 1 + RETEST_WINDOW, n)):
                bar_high = high_a[j]
                if bar_high >= broken_level - tol and bar_high <= broken_level + tol:
                    bar_c = close_a[j]; bar_o = open_a[j]
                    bar_body = abs(bar_c - bar_o)
                    if (
                        bar_c < broken_level
                        and bar_c < bar_o  # bearish confirmation
                        and bar_body >= CONFIRM_BODY_ATR * atr
                    ):
                        retest_idx = j
                        retest_high = bar_high
                        break
            if retest_idx is None or retest_high is None:
                continue

            # Entry at confirmation candle close
            entry_idx = retest_idx
            entry = close_a[entry_idx]
            sl = retest_high + SL_PAD_ATR * atr
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - tp_rr * risk

            exit_idx, exit_price, reason = _walk_forward_exit_np(
                high_a, low_a, close_a, entry_idx, "short", entry, sl, tp,
                TIME_STOP_BARS
            )
            hold_sec = max((exit_idx - entry_idx) * BAR_SEC, BAR_SEC)
            trades.append(Trade(
                symbol=symbol,
                side="short",
                entry_price=float(entry),
                exit_price=float(exit_price),
                notional_usd=NOTIONAL_USD,
                holding_sec=int(hold_sec),
                entry_ts=pd.Timestamp(ts_a[entry_idx]),
                exit_ts=pd.Timestamp(ts_a[exit_idx]),
                exit_reason=reason,
                extra={
                    "break_idx": int(i),
                    "reversal_lag": int(reversal_idx - i),
                    "retest_lag": int(retest_idx - reversal_idx),
                    "broken_level": float(broken_level),
                    "atr": float(atr),
                    "htf_filter": htf_filter,
                    "htf_window": htf_window,
                    "htf_slope_atr": htf_slope_atr,
                },
            ))
            open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    """Wrap base engine to attach symbol attribute on the loaded df."""

    def load_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


def _parse_cell_kv(cell: str) -> Dict[str, str]:
    """Parse cell_id like 'break_atr=0.8_htf_filter=True_htf_slope_atr=0.1_...'

    Param names contain underscores, so we can't naively split on '_'. Use
    known param names and split on '=' to extract values.
    """
    known_params = [
        "break_atr", "htf_filter", "htf_slope_atr", "htf_window", "tp_rr",
    ]
    out: Dict[str, str] = {}
    for p in known_params:
        # Match 'p=<value>' followed by either end-of-string or '_<next_param>=...'
        marker = f"{p}="
        idx = cell.find(marker)
        if idx < 0:
            continue
        start = idx + len(marker)
        # Find the next marker among known_params
        end = len(cell)
        for q in known_params:
            if q == p:
                continue
            qmarker = f"_{q}="
            qidx = cell.find(qmarker, start)
            if qidx >= 0 and qidx < end:
                end = qidx
        out[p] = cell[start:end]
    return out


if __name__ == "__main__":
    print("=== COMBO BTC BREAK BLOCK SHORT + HTF UPTREND — W/F STUDY ===\n")
    grid_size = sum(1 for _ in BTCBreakBlockShortHTFStrategy().param_grid())
    print(f"Grid size: {grid_size} cells per symbol")
    print(f"Symbols: BTC × 5m × 3 fee variants = {grid_size * 1 * 3} cell-variants\n")

    engine = _PatchedEngine(
        study=BTCBreakBlockShortHTFStrategy(),
        symbols=["BTC"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "combo_btc_break_block_short_htf",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())

    # Verdict distribution
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1

    # A/B comparison: htf_filter True vs False (variant A_full_taker only for headline)
    ab_true_pass = 0; ab_true_total = 0
    ab_false_pass = 0; ab_false_total = 0
    ab_true_is_evs: List[float] = []
    ab_false_is_evs: List[float] = []
    ab_true_q4_evs: List[float] = []
    ab_false_q4_evs: List[float] = []
    ab_true_n_total: List[int] = []
    ab_false_n_total: List[int] = []
    for k, v in cells:
        sym, tf, cell, variant = k.split("|", 3)
        if variant != "A_full_taker":
            continue
        kv = _parse_cell_kv(cell)
        is_htf = (kv.get("htf_filter") == "True")
        if is_htf:
            ab_true_total += 1
            if v["verdict"] == "PASS":
                ab_true_pass += 1
            ab_true_is_evs.append(v["is_ev"])
            ab_true_q4_evs.append(v["q4_ev"])
            ab_true_n_total.append(v["n_total"])
        else:
            ab_false_total += 1
            if v["verdict"] == "PASS":
                ab_false_pass += 1
            ab_false_is_evs.append(v["is_ev"])
            ab_false_q4_evs.append(v["q4_ev"])
            ab_false_n_total.append(v["n_total"])

    # Aggregate across all variants for PASS-rate-by-htf
    all_pass_by_htf = {True: 0, False: 0}
    all_total_by_htf = {True: 0, False: 0}
    for k, v in cells:
        sym, tf, cell, variant = k.split("|", 3)
        kv = _parse_cell_kv(cell)
        is_htf = (kv.get("htf_filter") == "True")
        all_total_by_htf[is_htf] += 1
        if v["verdict"] == "PASS":
            all_pass_by_htf[is_htf] += 1

    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    print(f"\n=== A/B HTF_FILTER COMPARISON (variant A_full_taker) ===")
    if ab_true_total and ab_false_total:
        print(f"htf_filter=True  PASS rate: {ab_true_pass}/{ab_true_total} ({100*ab_true_pass/ab_true_total:.1f}%)")
        print(f"htf_filter=False PASS rate: {ab_false_pass}/{ab_false_total} ({100*ab_false_pass/ab_false_total:.1f}%)")
        if ab_true_is_evs and ab_false_is_evs:
            avg_t_is = sum(ab_true_is_evs)/len(ab_true_is_evs)
            avg_f_is = sum(ab_false_is_evs)/len(ab_false_is_evs)
            avg_t_q4 = sum(ab_true_q4_evs)/len(ab_true_q4_evs)
            avg_f_q4 = sum(ab_false_q4_evs)/len(ab_false_q4_evs)
            print(f"htf_filter=True  avg IS_EV: ${avg_t_is:+.4f}   avg Q4_EV: ${avg_t_q4:+.4f}")
            print(f"htf_filter=False avg IS_EV: ${avg_f_is:+.4f}   avg Q4_EV: ${avg_f_q4:+.4f}")
            print(f"  IS_EV uplift from HTF: ${avg_t_is - avg_f_is:+.4f}")
            print(f"  Q4_EV uplift from HTF: ${avg_t_q4 - avg_f_q4:+.4f}")
        avg_n_t = sum(ab_true_n_total)/len(ab_true_n_total) if ab_true_n_total else 0
        avg_n_f = sum(ab_false_n_total)/len(ab_false_n_total) if ab_false_n_total else 0
        if avg_n_f > 0:
            reduction = 100 * (avg_n_f - avg_n_t) / avg_n_f
            print(f"Avg n_total per cell:  True={avg_n_t:.1f}  False={avg_n_f:.1f}  reduction={reduction:+.1f}%")

    print(f"\n=== A/B PASS RATE (across all variants) ===")
    print(f"htf_filter=True  PASS: {all_pass_by_htf[True]}/{all_total_by_htf[True]}")
    print(f"htf_filter=False PASS: {all_pass_by_htf[False]}/{all_total_by_htf[False]}")

    pass_cells = [c for c in cells if c[1]["verdict"] == "PASS"]
    if pass_cells:
        print(f"\n=== TOP 10 PASS CELLS by Q4 EV ===")
        pass_cells.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        for k, v in pass_cells[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B",
                     "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  {cell}")
            print(f"    n_total={v['n_total']} IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  "
                  f"Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  WR_oos={v['win_rate_oos']*100:.0f}%")
    else:
        cells_sorted = sorted(cells, key=lambda c: c[1]["is_ev"], reverse=True)
        print(f"\n=== Top 10 by IS_EV (no PASS cells) ===")
        for k, v in cells_sorted[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B",
                     "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  IS_n={v['is_n']} n_total={v['n_total']} "
                  f"IS=${v['is_ev']:+.3f}  Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  "
                  f"verdict={v['verdict']}")

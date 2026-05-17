"""W/F STUDY: 4H Candle Phase Filter applied to structure_bounce.

Hypothesis (from VN Edge weakspot map): structure_bounce signals fired during
the 4h bar's CONTRACTION phase = guaranteed peak_floor_stall losers (9% WR
in the 12h shadow sample). structure_bounce signals fired during EXPANSION =
exhaustion_wick winners (the only winning bucket in shadow).

If true: gating structure_bounce by 4h phase == EXPANSION should INCREASE
W/F PASS rate (rather than decrease, like single-pattern studies have done).

A/B test: phase_check ∈ {True, False} side-by-side over the same setup grid.

4h phase definition for the 4h bar containing the 5m signal time:
  progress = (signal_ts - bar_open_ts) / 4h_seconds   ∈ (0, 1]
  current_range = bar.high_so_far - bar.low_so_far    (computed from 5m within
                  this 4h bar — NOT the closed bar high/low)
  ref_atr = ATR(14) of prior CLOSED 4h bars
  expected_range_at_progress = progress * ref_atr
  expansion_ratio = current_range / expected_range_at_progress

  EXPANSION if ratio > 1.3
  CONTRACTION if ratio < 0.6
  TRANSITION otherwise

Gate: only allow structure_bounce when expansion_ratio >= expansion_min_ratio.

Setup (re-implemented minimally — does NOT import bot code):
  - 5m bar shows S/R rejection wick (lower wick for long, upper wick for short)
    wick >= wick_atr_min × ATR
  - Inside structure zone: price within 1 ATR of recent 20-bar swing high/low
  - Volume confirmation: rel_vol = volume / volume.rolling(20).mean() >= vol_min
  - LONG: rejection at swing low.  SHORT: rejection at swing high.
  - Entry at signal bar close.  TP per tp_mode.  SL = swing extreme ± 0.3 × ATR.
  - Hard time stop 30 min (6 bars).  Notional $400.

Param grid: 5 × 3 × 2 × 2 × 2 = 120 cells (× 4 pairs × 3 fee variants = 1440).
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
TIME_STOP_SEC = 1800
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
NOTIONAL_USD = 400.0
SL_ATR_BUFFER = 0.3
SWING_WINDOW = 20
ATR_PERIOD = 14
H4_SEC = 14400
H4_ATR_PERIOD = 14


def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """Numpy ATR via True Range, simple rolling mean."""
    n = len(high)
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
    csum = np.cumsum(tr)
    for i in range(period - 1, n):
        if i == period - 1:
            atr[i] = csum[i] / period
        else:
            atr[i] = (csum[i] - csum[i - period]) / period
    return atr


def _walk_forward_exit_np(high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, entry_idx: int, side: str,
                          entry: float, sl: float, tp: float,
                          max_bars: int) -> Tuple[int, float, str]:
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
# 4h phase precompute — for each 5m bar i, we need expansion_ratio of the
# containing 4h bar at that signal time. We build it once per simulate(...)
# call so the inner loop is O(1).
# ─────────────────────────────────────────────────────────────────────
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


def precompute_h4_expansion_ratio(
    df5_index: pd.DatetimeIndex,
    high5: np.ndarray, low5: np.ndarray,
    df4: pd.DataFrame,
) -> np.ndarray:
    """Return expansion_ratio[i] for each 5m bar i.

    Algorithm:
      1. Compute 4h ATR using PRIOR CLOSED bars (atr_4h_prev[k] = ATR ending at
         4h bar k-1).
      2. For each 5m bar i, find the 4h bar k that contains i (4h bars open at
         00:00, 04:00, 08:00, ... UTC).
      3. Track per-4h-bar running high/low using only 5m bars within that bar.
      4. progress = (i_5m_ts - bar_open_ts).total_seconds() / 14400
      5. current_range = running_high - running_low
      6. expansion_ratio = current_range / (progress * atr_4h_prev[k])
    """
    n = len(df5_index)
    out = np.full(n, np.nan, dtype=np.float64)

    # 4h ATR using prior closed 4h bars.
    h4 = df4["high"].to_numpy(dtype=np.float64)
    l4 = df4["low"].to_numpy(dtype=np.float64)
    c4 = df4["close"].to_numpy(dtype=np.float64)
    atr4 = add_atr_np(h4, l4, c4, H4_ATR_PERIOD)
    # atr4[k] is ATR ending at bar k. For the "current" bar k, we want ATR of
    # PRIOR closed bars i.e. atr4[k-1].
    atr4_prev = np.full(len(atr4), np.nan, dtype=np.float64)
    atr4_prev[1:] = atr4[:-1]

    # Convert both indexes to int microseconds via .as_unit("us") then asi8
    # to be unit-agnostic (cache is datetime64[ms, UTC]).
    ts5_us = pd.DatetimeIndex(df5_index).as_unit("us").asi8
    h4_open_us = pd.DatetimeIndex(df4.index).as_unit("us").asi8
    H4_US = H4_SEC * 1_000_000  # microseconds per 4h

    # For each 5m bar, find which 4h bar it belongs to via searchsorted.
    # 4h bar k covers [h4_open_us[k], h4_open_us[k] + H4_US).
    # idx_in_h4 = position such that h4_open_us[idx] <= ts < h4_open_us[idx+1].
    idx_in_h4 = np.searchsorted(h4_open_us, ts5_us, side="right") - 1

    # Now walk through 5m bars, tracking running high/low per 4h bin.
    cur_bin = -1
    run_hi = -np.inf
    run_lo = np.inf
    for i in range(n):
        k = idx_in_h4[i]
        if k < 0 or k >= len(h4_open_us):
            continue
        if k != cur_bin:
            cur_bin = k
            run_hi = high5[i]
            run_lo = low5[i]
        else:
            if high5[i] > run_hi:
                run_hi = high5[i]
            if low5[i] < run_lo:
                run_lo = low5[i]
        ref_atr = atr4_prev[k]
        if not np.isfinite(ref_atr) or ref_atr <= 0:
            continue
        # progress = elapsed_within_bar / 4h. The 5m bar OPEN ts is the start
        # of the bar; we want progress AT signal time which equals the OPEN ts
        # of the next 5m bar. Use OPEN+1 bar offset (5min) so progress=0 is
        # impossible and a 5m bar at the exact 4h open contributes some range.
        elapsed_us = ts5_us[i] - h4_open_us[k] + 5 * 60 * 1_000_000
        progress = elapsed_us / H4_US
        if progress <= 0 or progress > 1.0001:
            continue
        expected = progress * ref_atr
        if expected <= 0:
            continue
        cur_range = run_hi - run_lo
        out[i] = cur_range / expected
    return out


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────
class SBPhaseFilterStrategy(Strategy):
    name = "sb_phase_filter"

    # cache 4h frames + per-symbol expansion arrays — keyed by symbol
    _h4_cache: Dict[str, pd.DataFrame] = {}
    # cache base setup arrays per symbol so we don't recompute per cell
    _base_cache: Dict[str, Dict[str, Any]] = {}

    def param_grid(self):
        for expansion_min_ratio in [1.0, 1.2, 1.3, 1.5, 1.8]:
            for wick_atr_min in [0.4, 0.55, 0.7]:
                for vol_min in [1.0, 1.2]:
                    for tp_mode in ["fixed_2r", "atr_2.0"]:
                        for phase_check in [True, False]:
                            yield {
                                "expansion_min_ratio": expansion_min_ratio,
                                "wick_atr_min": wick_atr_min,
                                "vol_min": vol_min,
                                "tp_mode": tp_mode,
                                "phase_check": phase_check,
                            }

    def _build_base(self, df: pd.DataFrame) -> Dict[str, Any]:
        symbol = df.attrs.get("symbol", "BTC")
        key = symbol
        if key in self._base_cache:
            return self._base_cache[key]

        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        vol_a = df["volume"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)

        # rel_vol = volume / rolling20 mean (shifted by 0 — current bar can use
        # current vol; consistent with bot-side scanners that compare to the
        # last N bars including the trigger bar).
        v_ser = pd.Series(vol_a)
        rel_vol = (v_ser / v_ser.rolling(20, min_periods=10).mean()).to_numpy()

        # Swing high/low over PRIOR 20 bars (shifted)
        h_ser = pd.Series(high_a)
        l_ser = pd.Series(low_a)
        swing_high = h_ser.rolling(SWING_WINDOW).max().shift(1).to_numpy()
        swing_low = l_ser.rolling(SWING_WINDOW).min().shift(1).to_numpy()

        # Wicks
        upper_wick = high_a - np.maximum(open_a, close_a)
        lower_wick = np.minimum(open_a, close_a) - low_a

        # 4h expansion ratio per 5m bar
        if symbol not in self._h4_cache:
            self._h4_cache[symbol] = _load_4h(symbol)
        df4 = self._h4_cache[symbol]
        expansion_ratio = precompute_h4_expansion_ratio(
            df.index, high_a, low_a, df4
        )

        base = {
            "open": open_a, "high": high_a, "low": low_a, "close": close_a,
            "atr": atr_a, "rel_vol": rel_vol,
            "swing_high": swing_high, "swing_low": swing_low,
            "upper_wick": upper_wick, "lower_wick": lower_wick,
            "expansion_ratio": expansion_ratio,
            "ts": df.index.to_numpy(),
            "symbol": symbol,
        }
        self._base_cache[key] = base
        return base

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 100:
            return []
        base = self._build_base(df)

        open_a = base["open"]; high_a = base["high"]; low_a = base["low"]
        close_a = base["close"]; atr_a = base["atr"]; rel_vol = base["rel_vol"]
        swing_high = base["swing_high"]; swing_low = base["swing_low"]
        upper_wick = base["upper_wick"]; lower_wick = base["lower_wick"]
        expansion_ratio = base["expansion_ratio"]
        ts_a = base["ts"]; symbol = base["symbol"]

        wick_atr_min = float(params["wick_atr_min"])
        vol_min = float(params["vol_min"])
        expansion_min = float(params["expansion_min_ratio"])
        tp_mode = str(params["tp_mode"])
        phase_check = bool(params["phase_check"])

        max_bars = TIME_STOP_BARS

        trades: List[Trade] = []
        open_until = -1
        end_i = n - (TIME_STOP_BARS + 5)
        start_i = max(SWING_WINDOW + ATR_PERIOD + 5, 25)

        for i in range(start_i, end_i):
            if i <= open_until:
                continue
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            rv = rel_vol[i]
            if not np.isfinite(rv) or rv < vol_min:
                continue

            # Phase gate
            if phase_check:
                er = expansion_ratio[i]
                if not np.isfinite(er) or er < expansion_min:
                    continue

            sh = swing_high[i]; sl_ref = swing_low[i]
            if not (np.isfinite(sh) and np.isfinite(sl_ref)):
                continue

            o = open_a[i]; h = high_a[i]; l = low_a[i]; c = close_a[i]
            uw = upper_wick[i]; lw = lower_wick[i]

            # ─── LONG: rejection at swing low ─────────────────────────────
            # Long candidate: lower_wick large; price within 1 ATR of swing low
            if lw >= wick_atr_min * atr and l <= sl_ref + atr and l >= sl_ref - atr:
                # Body confirmation — close above bar mid (rejection candle)
                mid = (h + l) / 2.0
                if c > mid:
                    entry = c
                    sl = l - SL_ATR_BUFFER * atr
                    if sl >= entry:
                        continue
                    if tp_mode == "fixed_2r":
                        tp = entry + 2.0 * (entry - sl)
                    else:  # atr_2.0
                        tp = entry + 2.0 * atr
                    if tp <= entry:
                        continue
                    exit_idx, exit_price, reason = _walk_forward_exit_np(
                        high_a, low_a, close_a, i, "long", entry, sl, tp, max_bars
                    )
                    trades.append(Trade(
                        symbol=symbol, side="long",
                        entry_price=float(entry), exit_price=float(exit_price),
                        notional_usd=NOTIONAL_USD,
                        holding_sec=int((exit_idx - i) * BAR_SEC),
                        entry_ts=pd.Timestamp(ts_a[i]),
                        exit_ts=pd.Timestamp(ts_a[exit_idx]),
                        exit_reason=reason,
                        extra={
                            "phase_check": phase_check,
                            "expansion_ratio": float(expansion_ratio[i])
                                                 if np.isfinite(expansion_ratio[i]) else -1.0,
                            "tp_mode": tp_mode,
                        },
                    ))
                    open_until = exit_idx
                    continue

            # ─── SHORT: rejection at swing high ───────────────────────────
            if uw >= wick_atr_min * atr and h >= sh - atr and h <= sh + atr:
                mid = (h + l) / 2.0
                if c < mid:
                    entry = c
                    sl_p = h + SL_ATR_BUFFER * atr
                    if sl_p <= entry:
                        continue
                    if tp_mode == "fixed_2r":
                        tp = entry - 2.0 * (sl_p - entry)
                    else:
                        tp = entry - 2.0 * atr
                    if tp >= entry:
                        continue
                    exit_idx, exit_price, reason = _walk_forward_exit_np(
                        high_a, low_a, close_a, i, "short", entry, sl_p, tp, max_bars
                    )
                    trades.append(Trade(
                        symbol=symbol, side="short",
                        entry_price=float(entry), exit_price=float(exit_price),
                        notional_usd=NOTIONAL_USD,
                        holding_sec=int((exit_idx - i) * BAR_SEC),
                        entry_ts=pd.Timestamp(ts_a[i]),
                        exit_ts=pd.Timestamp(ts_a[exit_idx]),
                        exit_reason=reason,
                        extra={
                            "phase_check": phase_check,
                            "expansion_ratio": float(expansion_ratio[i])
                                                 if np.isfinite(expansion_ratio[i]) else -1.0,
                            "tp_mode": tp_mode,
                        },
                    ))
                    open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== SB PHASE FILTER — W/F STUDY ===\n")
    print("Cell grid: 5 × 3 × 2 × 2 × 2 = 120 cells per pair")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants = 1440 cell-variants\n")

    engine = _PatchedEngine(
        study=SBPhaseFilterStrategy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "sb_phase_filter",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())

    # Verdict distribution
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1

    # Build A/B comparison: phase_check=True vs phase_check=False
    def parse_cell_key(k: str):
        sym, tf, cell, variant = k.split("|", 3)
        kv = dict(p.split("=", 1) for p in cell.split("_") if "=" in p)
        return sym, tf, cell, variant, kv

    ab_true_pass = 0; ab_true_total = 0
    ab_false_pass = 0; ab_false_total = 0
    ab_true_is_evs = []
    ab_false_is_evs = []
    ab_true_n_total = []
    ab_false_n_total = []
    for k, v in cells:
        sym, tf, cell, variant, kv = parse_cell_key(k)
        if variant != "A_full_taker":
            continue
        is_phase = (kv.get("phase_check") == "True")
        if is_phase:
            ab_true_total += 1
            if v["verdict"] == "PASS":
                ab_true_pass += 1
            ab_true_is_evs.append(v["is_ev"])
            ab_true_n_total.append(v["n_total"])
        else:
            ab_false_total += 1
            if v["verdict"] == "PASS":
                ab_false_pass += 1
            ab_false_is_evs.append(v["is_ev"])
            ab_false_n_total.append(v["n_total"])

    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    print(f"\n=== A/B PHASE_CHECK COMPARISON (variant A_full_taker) ===")
    if ab_true_total and ab_false_total:
        print(f"phase_check=True  PASS rate: {ab_true_pass}/{ab_true_total} ({100*ab_true_pass/ab_true_total:.1f}%)")
        print(f"phase_check=False PASS rate: {ab_false_pass}/{ab_false_total} ({100*ab_false_pass/ab_false_total:.1f}%)")
        if ab_true_is_evs and ab_false_is_evs:
            print(f"phase_check=True  avg IS_EV: ${sum(ab_true_is_evs)/len(ab_true_is_evs):+.4f}")
            print(f"phase_check=False avg IS_EV: ${sum(ab_false_is_evs)/len(ab_false_is_evs):+.4f}")
        avg_n_t = sum(ab_true_n_total)/len(ab_true_n_total) if ab_true_n_total else 0
        avg_n_f = sum(ab_false_n_total)/len(ab_false_n_total) if ab_false_n_total else 0
        if avg_n_f > 0:
            reduction = 100 * (avg_n_f - avg_n_t) / avg_n_f
            print(f"Avg n_total per cell:  True={avg_n_t:.1f}  False={avg_n_f:.1f}  reduction={reduction:+.1f}%")

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

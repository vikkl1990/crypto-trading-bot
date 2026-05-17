"""W/F STUDY: Combo VWAP MR + Structure Bounce ensemble.

Hypothesis: scalper_vwap_mr fires when price stretched from VWAP (mean-revert).
structure_bounce fires on S/R rejection. When BOTH fire same-side within window,
the mean-revert has additional structural confirmation.

4 modes tested A/B:
  - vwap_only: baseline VWAP MR
  - sb_only: baseline structure_bounce
  - ensemble_AND: both must fire within window_bars same-side
  - vwap_with_sb_booster: VWAP fires; SB in last window_bars boosts conviction
                          (treated same as ensemble_AND for entry filtering since
                          we test the lift on EV/trade, not size scaling).

Pairs: BTC + ETH only (per instructions).

Param grid: 4 (mode) × 3 (window_bars) × 2 (tp_rr) = 24 cells × 2 pairs × 3 fee
variants = 144 cell-variants total.

Vectorized inner: signal arrays computed once per pair, modes share masks.
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
TIME_STOP_SEC = 1800        # 30 min hard stop
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
NOTIONAL_USD = 400.0
ATR_PERIOD = 14
SWING_WINDOW = 20
SL_ATR_BUFFER = 0.3
SL_FIXED_ATR = 0.6

# VWAP MR fixed params (calibrated from chop_vwap_mr_study top cells)
VWAP_DIST_ATR = 0.6           # price stretched ≥ 0.6 ATR from VWAP
VWAP_BODY_ATR_MIN = 0.4       # body ≥ 0.4 ATR
VWAP_RANGE_ATR_MAX = 8.0      # range over last 60 bars ≤ 8 ATR (chop bracket)
VWAP_ATR_RANK_MAX = 0.40      # 4h ATR rank ≤ 0.40 (chop regime)
RANGE_LOOKBACK = 60

# SB fixed params (calibrated from sb_phase_filter_study)
SB_WICK_ATR_MIN = 0.55        # wick ≥ 0.55 ATR
SB_VOL_MIN = 1.0              # rel_vol ≥ 1.0
SB_ZONE_TOL_ATR = 1.0         # within 1 ATR of swing extreme

H4_SEC = 14400
H4_ATR_PERIOD = 14


# ─────────────────────────────────────────────────────────────────────
# Numpy helpers
# ─────────────────────────────────────────────────────────────────────
def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
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


def compute_4h_atr_percentile_for_5m(
    ts_5m_a: np.ndarray, df_4h: pd.DataFrame,
    atr_period: int = 14, rank_window: int = 100,
) -> np.ndarray:
    high_4h = df_4h["high"].to_numpy(dtype=np.float64)
    low_4h = df_4h["low"].to_numpy(dtype=np.float64)
    close_4h = df_4h["close"].to_numpy(dtype=np.float64)
    atr_4h = add_atr_np(high_4h, low_4h, close_4h, atr_period)
    n4 = len(atr_4h)
    pct_4h = np.full(n4, np.nan, dtype=np.float64)
    for i in range(rank_window, n4):
        if not np.isfinite(atr_4h[i]):
            continue
        window = atr_4h[i - rank_window:i]
        valid = window[np.isfinite(window)]
        if len(valid) < 10:
            continue
        cur = atr_4h[i]
        pct_4h[i] = (valid < cur).sum() / len(valid)
    ts_4h_a = df_4h.index.to_numpy()
    ts_4h_end_ns = (ts_4h_a.astype("datetime64[ns]").astype(np.int64) +
                    int(4 * 3600 * 1e9))
    ts_5m_ns = ts_5m_a.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(ts_4h_end_ns, ts_5m_ns, side="right") - 1
    pct_5m = np.full(len(ts_5m_ns), np.nan, dtype=np.float64)
    valid_mask = idx >= 0
    valid_idx = idx[valid_mask]
    pct_5m[valid_mask] = pct_4h[valid_idx]
    return pct_5m


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


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────
class ComboVwapSBStrategy(Strategy):
    """Combo: VWAP MR + Structure Bounce ensemble.

    Modes:
      - vwap_only: VWAP MR signal only
      - sb_only: structure_bounce signal only
      - ensemble_AND: BOTH within window_bars same-side
      - vwap_with_sb_booster: VWAP fires AND SB fired within last window_bars
                              (structurally same as AND for entry, but we keep
                              it to evaluate VWAP-led variant separately)
    """
    name = "combo_vwap_sb"

    _base_cache: Dict[str, Dict[str, Any]] = {}

    def param_grid(self):
        for mode in ["vwap_only", "sb_only", "ensemble_AND", "vwap_with_sb_booster"]:
            for window_bars in [1, 3, 6]:
                for tp_rr in [1.5, 2.0]:
                    yield {
                        "mode": mode,
                        "window_bars": window_bars,
                        "tp_rr": tp_rr,
                    }

    def _build_base(self, df: pd.DataFrame) -> Dict[str, Any]:
        symbol = df.attrs.get("symbol", "BTC")
        if symbol in self._base_cache:
            cached = self._base_cache[symbol]
            if len(cached["close"]) == len(df):
                return cached

        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        vol_a = df["volume"].to_numpy(dtype=np.float64)
        ts_a = df.index.to_numpy()
        n = len(close_a)

        atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)
        vwap_a = compute_vwap_daily_reset(ts_a, high_a, low_a, close_a, vol_a)

        try:
            df4 = _load_4h(symbol)
            atr_pct_a = compute_4h_atr_percentile_for_5m(ts_a, df4)
        except FileNotFoundError:
            atr_pct_a = np.full(n, np.nan, dtype=np.float64)

        # Body / wicks
        bullish = close_a > open_a
        bearish = close_a < open_a
        body = np.abs(close_a - open_a)
        bar_range = high_a - low_a
        upper_half_close = (close_a - low_a) > (bar_range / 2.0)
        lower_half_close = (high_a - close_a) > (bar_range / 2.0)
        upper_wick = high_a - np.maximum(open_a, close_a)
        lower_wick = np.minimum(open_a, close_a) - low_a

        # Vol
        v_ser = pd.Series(vol_a)
        rel_vol = (v_ser / v_ser.rolling(20, min_periods=10).mean()).to_numpy()

        # Swing high/low over PRIOR 20 bars (shifted)
        h_ser = pd.Series(high_a)
        l_ser = pd.Series(low_a)
        swing_high = h_ser.rolling(SWING_WINDOW).max().shift(1).to_numpy()
        swing_low = l_ser.rolling(SWING_WINDOW).min().shift(1).to_numpy()

        # Range over last 60 bars (vectorized)
        range_high = h_ser.rolling(RANGE_LOOKBACK, min_periods=20).max().to_numpy()
        range_low = l_ser.rolling(RANGE_LOOKBACK, min_periods=20).min().to_numpy()

        # ─── VWAP MR signals (vectorized) ──────────────────────────
        # LONG: close <= vwap - VWAP_DIST_ATR*atr, bullish, body>=min, upper-half close
        # Chop: atr_pct ≤ 0.40; range bracket: (range_high-range_low) ≤ 8*atr
        chop_ok = np.where(np.isfinite(atr_pct_a), atr_pct_a <= VWAP_ATR_RANK_MAX, False)
        range_ok = np.where(
            np.isfinite(range_high) & np.isfinite(range_low) & np.isfinite(atr_a),
            (range_high - range_low) <= VWAP_RANGE_ATR_MAX * atr_a,
            False,
        )
        atr_ok = np.isfinite(atr_a) & (atr_a > 0)
        vwap_ok = np.isfinite(vwap_a)

        body_ok = np.where(np.isfinite(atr_a), body >= VWAP_BODY_ATR_MIN * atr_a, False)

        vwap_long_sig = (
            chop_ok & range_ok & atr_ok & vwap_ok
            & (close_a <= (vwap_a - VWAP_DIST_ATR * atr_a))
            & bullish & body_ok & upper_half_close
        )
        vwap_short_sig = (
            chop_ok & range_ok & atr_ok & vwap_ok
            & (close_a >= (vwap_a + VWAP_DIST_ATR * atr_a))
            & bearish & body_ok & lower_half_close
        )

        # ─── SB signals (vectorized) ───────────────────────────────
        # LONG: lower_wick >= SB_WICK_ATR_MIN*atr; price within zone of swing low
        # close > mid (rejection candle); rel_vol >= SB_VOL_MIN
        mid = (high_a + low_a) / 2.0
        rel_vol_ok = np.where(np.isfinite(rel_vol), rel_vol >= SB_VOL_MIN, False)

        sb_long_zone = np.where(
            np.isfinite(swing_low) & np.isfinite(atr_a),
            (low_a <= swing_low + SB_ZONE_TOL_ATR * atr_a) & (low_a >= swing_low - SB_ZONE_TOL_ATR * atr_a),
            False,
        )
        sb_long_sig = (
            atr_ok & rel_vol_ok & sb_long_zone
            & (lower_wick >= SB_WICK_ATR_MIN * atr_a)
            & (close_a > mid)
        )

        sb_short_zone = np.where(
            np.isfinite(swing_high) & np.isfinite(atr_a),
            (high_a >= swing_high - SB_ZONE_TOL_ATR * atr_a) & (high_a <= swing_high + SB_ZONE_TOL_ATR * atr_a),
            False,
        )
        sb_short_sig = (
            atr_ok & rel_vol_ok & sb_short_zone
            & (upper_wick >= SB_WICK_ATR_MIN * atr_a)
            & (close_a < mid)
        )

        base = {
            "open": open_a, "high": high_a, "low": low_a, "close": close_a,
            "atr": atr_a, "vwap": vwap_a,
            "swing_high": swing_high, "swing_low": swing_low,
            "vwap_long": vwap_long_sig, "vwap_short": vwap_short_sig,
            "sb_long": sb_long_sig, "sb_short": sb_short_sig,
            "ts": ts_a, "symbol": symbol,
        }
        self._base_cache[symbol] = base
        return base

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 200:
            return []
        base = self._build_base(df)
        open_a = base["open"]; high_a = base["high"]; low_a = base["low"]
        close_a = base["close"]; atr_a = base["atr"]; vwap_a = base["vwap"]
        swing_high = base["swing_high"]; swing_low = base["swing_low"]
        vwap_long = base["vwap_long"]; vwap_short = base["vwap_short"]
        sb_long = base["sb_long"]; sb_short = base["sb_short"]
        ts_a = base["ts"]; symbol = base["symbol"]

        mode = str(params["mode"])
        window_bars = int(params["window_bars"])
        tp_rr = float(params["tp_rr"])

        # Construct entry signal array per side based on mode.
        # For modes that require "SB fired within last window_bars", build a
        # rolling OR mask of SB on prior bars (excluding current via shift-1).
        def rolling_or(arr: np.ndarray, w: int) -> np.ndarray:
            """True if any of the last w bars (INCLUDING current) is True."""
            s = pd.Series(arr.astype(np.int8))
            r = s.rolling(w, min_periods=1).max()
            return r.to_numpy(dtype=bool)

        if mode == "vwap_only":
            long_entry = vwap_long.copy()
            short_entry = vwap_short.copy()
        elif mode == "sb_only":
            long_entry = sb_long.copy()
            short_entry = sb_short.copy()
        elif mode == "ensemble_AND":
            sb_long_recent = rolling_or(sb_long, window_bars)
            sb_short_recent = rolling_or(sb_short, window_bars)
            vwap_long_recent = rolling_or(vwap_long, window_bars)
            vwap_short_recent = rolling_or(vwap_short, window_bars)
            # Both must have fired within window. Trigger on the LATER of the two.
            # Simplest: at bar i, fire if (vwap fires now AND sb fired in last w)
            # OR (sb fires now AND vwap fired in last w). To avoid double-counting
            # we use OR, which is fine because open_until cooldown prevents dupes.
            long_entry = (vwap_long & sb_long_recent) | (sb_long & vwap_long_recent)
            short_entry = (vwap_short & sb_short_recent) | (sb_short & vwap_short_recent)
        elif mode == "vwap_with_sb_booster":
            # VWAP fires now AND SB fired within last window_bars.
            sb_long_recent = rolling_or(sb_long, window_bars)
            sb_short_recent = rolling_or(sb_short, window_bars)
            long_entry = vwap_long & sb_long_recent
            short_entry = vwap_short & sb_short_recent
        else:
            return []

        trades: List[Trade] = []
        open_until = -1
        max_bars = TIME_STOP_BARS
        end_i = n - (TIME_STOP_BARS + 5)
        start_i = max(SWING_WINDOW + ATR_PERIOD + 5, RANGE_LOOKBACK + 5, 25)

        for i in range(start_i, end_i):
            if i <= open_until:
                continue
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue

            if long_entry[i]:
                entry = close_a[i]
                # Stop: tighter of swing-based vs fixed
                lo = low_a[i]
                # For VWAP-mode use swing_low_intra (last 10 bars); for SB use bar low
                # Use bar low (current low) as proximate swing for simplicity.
                sl_swing = lo - SL_ATR_BUFFER * atr
                sl_fixed = entry - SL_FIXED_ATR * atr
                sl = max(sl_swing, sl_fixed)  # tighter (closer to entry)
                if sl >= entry:
                    continue
                risk = entry - sl
                tp = entry + tp_rr * risk
                if tp <= entry:
                    continue
                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    high_a, low_a, close_a, i, "long", entry, sl, tp, max_bars,
                )
                trades.append(Trade(
                    symbol=symbol, side="long",
                    entry_price=float(entry), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - i) * BAR_SEC),
                    entry_ts=pd.Timestamp(ts_a[i]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"mode": mode, "window_bars": window_bars, "tp_rr": tp_rr},
                ))
                open_until = exit_idx
                continue

            if short_entry[i]:
                entry = close_a[i]
                hi = high_a[i]
                sl_swing = hi + SL_ATR_BUFFER * atr
                sl_fixed = entry + SL_FIXED_ATR * atr
                sl = min(sl_swing, sl_fixed)  # tighter (closer to entry)
                if sl <= entry:
                    continue
                risk = sl - entry
                tp = entry - tp_rr * risk
                if tp >= entry:
                    continue
                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    high_a, low_a, close_a, i, "short", entry, sl, tp, max_bars,
                )
                trades.append(Trade(
                    symbol=symbol, side="short",
                    entry_price=float(entry), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - i) * BAR_SEC),
                    entry_ts=pd.Timestamp(ts_a[i]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"mode": mode, "window_bars": window_bars, "tp_rr": tp_rr},
                ))
                open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== COMBO VWAP MR + STRUCTURE BOUNCE — W/F STUDY ===\n")
    print("Cell grid: 4 (mode) × 3 (window) × 2 (tp_rr) = 24 cells per pair")
    print("Symbols: BTC, ETH × 5m × 3 fee variants = 144 cell-variants\n")

    engine = _PatchedEngine(
        study=ComboVwapSBStrategy(),
        symbols=["BTC", "ETH"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "combo_vwap_sb",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1

    def parse_cell_key(k: str):
        sym, tf, cell, variant = k.split("|", 3)
        kv = dict(p.split("=", 1) for p in cell.split("_") if "=" in p)
        return sym, tf, cell, variant, kv

    # Per-mode A/B aggregates by pair (variant A_full_taker for headline)
    from collections import defaultdict
    pair_mode_pass = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # (pass, total)
    pair_mode_isev = defaultdict(lambda: defaultdict(list))
    pair_mode_n = defaultdict(lambda: defaultdict(list))
    pair_mode_q4ev = defaultdict(lambda: defaultdict(list))

    for k, v in cells:
        sym, tf, cell, variant, kv = parse_cell_key(k)
        if variant != "A_full_taker":
            continue
        # mode is the first chunk in cell_id; cell_id from sorted keys:
        # mode=X_tp_rr=Y_window_bars=Z (sorted alphabetically: mode, tp_rr, window_bars)
        mode = kv.get("mode", "?")
        # When parsing "mode=ensemble_AND", split by "=" gives "ensemble", and "AND_tp"
        # would attach. Handle separately:
        # cell looks like: mode=ensemble_AND_tp_rr=1.5_window_bars=1
        # Split by "_" then re-key. Let's recompute.
        # The cell_id is: "mode=...,tp_rr=...,window_bars=..." joined by "_"
        # but values like "ensemble_AND" contain underscores. Use regex parse:
        import re
        kv2 = {}
        # Match: key=value where key is alphanum/underscore and value is everything until next "key="
        # Or simpler: use known keys
        for pat in ["mode", "tp_rr", "window_bars"]:
            m = re.search(rf"{pat}=([^_]+(?:_[^=]+?(?=_[a-z]+=))?)(?=_[a-z]+=|$)", cell)
            if m:
                kv2[pat] = m.group(1)
        # For mode specifically use known set
        for cand in ["vwap_only", "sb_only", "ensemble_AND", "vwap_with_sb_booster"]:
            if f"mode={cand}" in cell:
                kv2["mode"] = cand
                break
        mode = kv2.get("mode", mode)
        is_pass = (v["verdict"] == "PASS")
        pair_mode_pass[sym][mode][1] += 1
        if is_pass:
            pair_mode_pass[sym][mode][0] += 1
        if v.get("is_n", 0) > 0:
            pair_mode_isev[sym][mode].append(v["is_ev"])
        pair_mode_n[sym][mode].append(v["n_total"])
        if v.get("q4_n", 0) > 0:
            pair_mode_q4ev[sym][mode].append(v["q4_ev"])

    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    print("\n=== PER-PAIR PER-MODE A/B (variant A_full_taker) ===")
    for sym in sorted(pair_mode_pass.keys()):
        print(f"\n{sym}:")
        for mode in ["vwap_only", "sb_only", "ensemble_AND", "vwap_with_sb_booster"]:
            p, t = pair_mode_pass[sym][mode]
            isevs = pair_mode_isev[sym][mode]
            ns = pair_mode_n[sym][mode]
            q4evs = pair_mode_q4ev[sym][mode]
            avg_isev = sum(isevs)/len(isevs) if isevs else float('nan')
            avg_n = sum(ns)/len(ns) if ns else 0
            avg_q4 = sum(q4evs)/len(q4evs) if q4evs else float('nan')
            print(f"  {mode:28s}  PASS {p}/{t}  avg_IS_EV ${avg_isev:+.4f}  "
                  f"avg_Q4_EV ${avg_q4:+.4f}  avg_n {avg_n:.0f}")

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

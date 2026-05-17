"""W/F STUDY: FVG Multi-Timeframe Cascade (1h FVG + 5m FVG fill).

Tests a NEW multi-timeframe SMC pattern after 3 single-TF retest patterns
(Demand Zone, Break Block, bos_choch) all KILLED in W/F today with regime-
collapse signature. The MTF gate (1h FVG existence) is structurally tighter:
1h FVGs only form on impulse moves, so the pattern is implicitly trend-gated.

Setup (LONG; mirror for SHORT as bearish 1h FVG / bearish 5m FVG fill):
  Step 1 — 1H FVG IDENTIFICATION: scan last `htf_fvg_max_age_h` hours of 1h
           candles for an unfilled bullish FVG (3-candle: bar j+1.low > bar
           j-1.high). FVG width must be >= htf_fvg_min_atr × 1h ATR(14).
           Track zone (fvg_low, fvg_high).
  Step 2 — 5M ENTRY INTO ZONE: current 5m bar's price enters the 1h FVG
           zone (low <= fvg_high AND high >= fvg_low).
  Step 3 — 5M NEW FVG FORMS: within fvg_lookback_5m bars after zone entry,
           a new bullish 5m FVG forms (3-candle pattern).
  Step 4 — 5M FVG FILL CANDLE: a 5m bar closes back through the 5m FVG with
           displacement body >= displ_atr × 5m ATR(14). Default 0.65.
  Step 5 — ENTRY: at fill candle close.
  Step 6 — STOP: below 1h FVG low - 0.3 × 1h ATR.
  Step 7 — TARGET: 1h FVG midpoint (fvg_mid) OR fixed 1:2 RR (fixed_2r).

Optional `htf_align`: require HTF EMA21 slope to match side (LONG only if
1h close >= 1h EMA21 and EMA21 slope >0 over last 6 bars; mirror SHORT).

Hard time stop 30 min (6 × 5m bars). Notional $400 fixed.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

BAR_SEC_5M = 300
TIME_STOP_SEC = 1800        # 30 min hard time stop
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC_5M  # 6 bars
SL_ATR_BUFFER = 0.3         # stop = fvg_low - 0.3 × 1h ATR
NOTIONAL_USD = 400.0


def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """ATR via True Range, simple rolling mean. Numpy version."""
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


def add_ema_np(close: np.ndarray, period: int = 21) -> np.ndarray:
    """EMA via standard recursive formula."""
    n = len(close)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return ema
    alpha = 2.0 / (period + 1)
    ema[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
    return ema


def _walk_forward_exit_np(open_a: np.ndarray, high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, entry_idx: int, side: str,
                          entry: float, sl: float, tp: float,
                          max_bars: int) -> Tuple[int, float, str]:
    """Bar-by-bar walk to first SL/TP/time-stop. Numpy."""
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


def _identify_1h_fvgs(high_a: np.ndarray, low_a: np.ndarray, close_a: np.ndarray,
                      open_a: np.ndarray, atr_a: np.ndarray,
                      ts_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 1h FVG identification.

    For each bar j, check if a 3-candle FVG at (j-1, j, j+1) exists:
      bullish FVG: low[j+1] > high[j-1]   → zone = (high[j-1], low[j+1])
      bearish FVG: high[j+1] < low[j-1]   → zone = (high[j+1], low[j-1])

    We index FVGs by their FORMATION bar (j+1), so an FVG is "available" from
    that timestamp onwards until either filled or aged out.

    Returns parallel arrays: form_idx, side ('long'/'short' as 1/-1),
                             fvg_low, fvg_high, formation_ts_ns
    """
    n = len(high_a)
    forms = []
    sides = []
    lows = []
    highs = []
    tss = []
    for j in range(1, n - 1):
        prev_high = high_a[j - 1]
        prev_low = low_a[j - 1]
        next_high = high_a[j + 1]
        next_low = low_a[j + 1]
        atr = atr_a[j]
        if not np.isfinite(atr) or atr <= 0:
            continue
        # Bullish FVG
        if next_low > prev_high:
            forms.append(j + 1)
            sides.append(1)
            lows.append(prev_high)
            highs.append(next_low)
            tss.append(ts_ns[j + 1])
        # Bearish FVG
        elif next_high < prev_low:
            forms.append(j + 1)
            sides.append(-1)
            lows.append(next_high)
            highs.append(prev_low)
            tss.append(ts_ns[j + 1])
    return (np.array(forms, dtype=np.int64),
            np.array(sides, dtype=np.int32),
            np.array(lows, dtype=np.float64),
            np.array(highs, dtype=np.float64),
            np.array(tss, dtype=np.int64))


class FVGMTFCascadeStrategy(Strategy):
    """FVG MTF Cascade — 1h FVG + 5m FVG fill cascade.

    Param sweep (108 cells):
      htf_fvg_max_age_h: [12, 24, 48]    (hours of staleness allowed)
      htf_fvg_min_atr:   [0.5, 1.0, 1.5] (1h FVG width as multiple of 1h ATR)
      fvg_lookback_5m:   [5, 10, 15]      (5m bars after entry to wait for FVG)
      tp_mode:           ["fvg_mid", "fixed_2r"]
      htf_align:         [True, False]
    `displ_atr` fixed at 0.65 to keep grid manageable.
    """
    name = "fvg_mtf_cascade"
    _htf_cache: Dict[str, Dict[str, Any]] = {}

    def param_grid(self):
        for max_age in [12, 24, 48]:
            for min_atr in [0.5, 1.0, 1.5]:
                for lookback in [5, 10, 15]:
                    for tp_mode in ["fvg_mid", "fixed_2r"]:
                        for htf_align in [True, False]:
                            yield {
                                "htf_fvg_max_age_h": max_age,
                                "htf_fvg_min_atr": min_atr,
                                "fvg_lookback_5m": lookback,
                                "displ_atr": 0.65,
                                "tp_mode": tp_mode,
                                "htf_align": htf_align,
                            }

    def _load_htf(self, symbol: str) -> Dict[str, Any]:
        """Load 1h candles + identify FVGs once per symbol (cached)."""
        if symbol in self._htf_cache:
            return self._htf_cache[symbol]
        path = ROOT / "storage" / "candle_cache" / f"{symbol}_USDT_1h.parquet"
        df_1h = pd.read_parquet(path)
        if "datetime" in df_1h.columns:
            df_1h = df_1h.set_index("datetime")
        if df_1h.index.tz is None:
            df_1h.index = df_1h.index.tz_localize("UTC")
        df_1h = df_1h.sort_index()
        for c in ("open", "high", "low", "close"):
            df_1h[c] = df_1h[c].astype(float)

        h_high = df_1h["high"].to_numpy(dtype=np.float64)
        h_low = df_1h["low"].to_numpy(dtype=np.float64)
        h_close = df_1h["close"].to_numpy(dtype=np.float64)
        h_open = df_1h["open"].to_numpy(dtype=np.float64)
        h_atr = add_atr_np(h_high, h_low, h_close, 14)
        h_ema21 = add_ema_np(h_close, 21)
        h_ts_ns = df_1h.index.values.astype("datetime64[ns]").astype(np.int64)

        forms, sides, lows, highs, tss = _identify_1h_fvgs(
            h_high, h_low, h_close, h_open, h_atr, h_ts_ns
        )

        bundle = {
            "ts_ns": h_ts_ns,
            "high": h_high,
            "low": h_low,
            "close": h_close,
            "atr": h_atr,
            "ema21": h_ema21,
            "fvg_form_idx": forms,
            "fvg_side": sides,
            "fvg_low": lows,
            "fvg_high": highs,
            "fvg_form_ts_ns": tss,
        }
        self._htf_cache[symbol] = bundle
        return bundle

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 50:
            return []
        symbol = df.attrs.get("symbol", "BTC")

        try:
            htf = self._load_htf(symbol)
        except FileNotFoundError:
            return []

        # 5m arrays
        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, 14)
        ts_a = df.index.to_numpy()
        ts_ns_5m = df.index.values.astype("datetime64[ns]").astype(np.int64)

        bullish = close_a > open_a
        bearish = close_a < open_a
        body = np.abs(close_a - open_a)

        max_age_h = int(params["htf_fvg_max_age_h"])
        min_atr = float(params["htf_fvg_min_atr"])
        lookback_5m = int(params["fvg_lookback_5m"])
        displ_atr = float(params["displ_atr"])
        tp_mode = str(params["tp_mode"])
        htf_align = bool(params["htf_align"])

        # ── Pre-filter 1h FVGs by min_atr width
        fvg_form_idx = htf["fvg_form_idx"]
        fvg_side = htf["fvg_side"]
        fvg_low = htf["fvg_low"]
        fvg_high = htf["fvg_high"]
        fvg_form_ts_ns = htf["fvg_form_ts_ns"]
        h_atr = htf["atr"]
        h_close = htf["close"]
        h_ema21 = htf["ema21"]
        h_ts_ns = htf["ts_ns"]

        fvg_widths = fvg_high - fvg_low
        fvg_atr_at_form = h_atr[fvg_form_idx]  # 1h ATR at formation
        valid_w = fvg_widths >= (min_atr * fvg_atr_at_form)
        # Also drop FVGs without valid ATR
        valid_w &= np.isfinite(fvg_atr_at_form) & (fvg_atr_at_form > 0)

        f_form_idx = fvg_form_idx[valid_w]
        f_side = fvg_side[valid_w]
        f_low = fvg_low[valid_w]
        f_high = fvg_high[valid_w]
        f_form_ts_ns = fvg_form_ts_ns[valid_w]
        f_atr = fvg_atr_at_form[valid_w]
        n_fvgs = len(f_form_idx)
        if n_fvgs == 0:
            return []

        max_age_ns = max_age_h * 3600 * 1_000_000_000

        # Map 5m timestamps to last-known 1h index for fast HTF align lookup
        # h_ts_ns is sorted; for each 5m ts, find searchsorted on h_ts_ns
        h_ts_for_5m = np.searchsorted(h_ts_ns, ts_ns_5m, side="right") - 1
        h_ts_for_5m = np.clip(h_ts_for_5m, 0, len(h_ts_ns) - 1)

        # For HTF alignment: EMA21 slope over last 6 1h bars
        # Pre-compute slope sign
        if htf_align:
            ema_slope_sign = np.zeros(len(h_ema21), dtype=np.int8)
            for k in range(6, len(h_ema21)):
                if np.isfinite(h_ema21[k]) and np.isfinite(h_ema21[k - 6]):
                    diff = h_ema21[k] - h_ema21[k - 6]
                    if diff > 0:
                        ema_slope_sign[k] = 1
                    elif diff < 0:
                        ema_slope_sign[k] = -1

        trades: List[Trade] = []
        open_until = -1
        max_bars_exit = TIME_STOP_BARS
        end_i = n - (TIME_STOP_BARS + 5)
        # Need ATR; lookback_5m for FVG; +5 for fill
        start_i = 30

        # For efficiency: filter active FVGs per 5m bar by ts
        # Active = form_ts <= ts_5m AND ts_5m - form_ts <= max_age_ns
        # Use searchsorted to skip FVGs not yet formed.
        # We iterate 5m bars and for each bar do a window query into f_form_ts_ns.
        f_form_ts_sorted_idx = np.argsort(f_form_ts_ns)
        f_form_ts_sorted = f_form_ts_ns[f_form_ts_sorted_idx]

        for i in range(start_i, end_i):
            if i <= open_until:
                continue
            atr5 = atr_a[i]
            if not np.isfinite(atr5) or atr5 <= 0:
                continue

            ts_now = ts_ns_5m[i]
            # Find FVGs formed within last max_age_ns
            ts_min = ts_now - max_age_ns
            # Rightmost formed strictly before ts_now (we allow same-time touch)
            r = np.searchsorted(f_form_ts_sorted, ts_now, side="right")
            l = np.searchsorted(f_form_ts_sorted, ts_min, side="left")
            if r <= l:
                continue
            # Active sorted indices in [l, r)
            sorted_active = f_form_ts_sorted_idx[l:r]

            # Current 5m bar properties
            o = open_a[i]; h = high_a[i]; lo = low_a[i]; c = close_a[i]

            # Iterate active FVGs to find one where 5m bar enters the zone
            chosen_fvg_idx = -1
            chosen_side = 0
            chosen_low = 0.0
            chosen_high = 0.0
            chosen_atr1h = 0.0
            for k in sorted_active:
                z_low = f_low[k]
                z_high = f_high[k]
                # Zone entry: 5m bar overlaps zone
                if lo <= z_high and h >= z_low:
                    side_k = f_side[k]
                    # HTF align check
                    if htf_align:
                        h_k = h_ts_for_5m[i]
                        slope = ema_slope_sign[h_k]
                        # LONG side: slope must be >0 AND close >= ema
                        if side_k == 1:
                            if slope <= 0:
                                continue
                            if not (np.isfinite(h_ema21[h_k]) and h_close[h_k] >= h_ema21[h_k]):
                                continue
                        else:  # SHORT
                            if slope >= 0:
                                continue
                            if not (np.isfinite(h_ema21[h_k]) and h_close[h_k] <= h_ema21[h_k]):
                                continue
                    chosen_fvg_idx = k
                    chosen_side = side_k
                    chosen_low = z_low
                    chosen_high = z_high
                    chosen_atr1h = f_atr[k]
                    break  # take first valid (most recently formed in window)

            if chosen_fvg_idx < 0:
                continue

            # ── Step 3: Look forward up to lookback_5m bars for new 5m FVG
            # ── Step 4: Then look further for fill candle
            # We do this from bar i (zone-entry bar) onwards.
            j_end = min(i + lookback_5m + 1, n - 1)
            new_fvg_idx = -1
            new_fvg_low = 0.0
            new_fvg_high = 0.0
            for j in range(i, j_end - 1):
                # 3-candle FVG centred on j: requires j-1, j, j+1
                if j < 1:
                    continue
                if chosen_side == 1:
                    # bullish: low[j+1] > high[j-1]
                    if low_a[j + 1] > high_a[j - 1]:
                        new_fvg_idx = j + 1  # formation bar (newest)
                        new_fvg_low = high_a[j - 1]
                        new_fvg_high = low_a[j + 1]
                        break
                else:
                    # bearish: high[j+1] < low[j-1]
                    if high_a[j + 1] < low_a[j - 1]:
                        new_fvg_idx = j + 1
                        new_fvg_low = high_a[j + 1]
                        new_fvg_high = low_a[j - 1]
                        break
            if new_fvg_idx < 0:
                continue

            # Step 4: fill candle — bar after new_fvg_idx that closes back through
            # the 5m FVG with displacement body >= displ_atr × 5m ATR
            fill_idx = -1
            fill_end = min(new_fvg_idx + 8, n - 1)  # allow up to 8 bars to fill
            for j in range(new_fvg_idx + 1, fill_end):
                atr5j = atr_a[j]
                if not np.isfinite(atr5j) or atr5j <= 0:
                    continue
                bj = body[j]
                if chosen_side == 1:
                    # Long: bullish bar, fills bottom of 5m FVG, close above gap_high
                    if (low_a[j] <= new_fvg_high and bullish[j]
                        and bj >= displ_atr * atr5j and close_a[j] > new_fvg_high):
                        fill_idx = j
                        break
                else:
                    if (high_a[j] >= new_fvg_low and bearish[j]
                        and bj >= displ_atr * atr5j and close_a[j] < new_fvg_low):
                        fill_idx = j
                        break
            if fill_idx < 0:
                continue

            # Sanity: fill_idx still within trade horizon
            if fill_idx >= end_i:
                continue

            # Step 5: Entry at fill close
            entry = close_a[fill_idx]
            atr1h = chosen_atr1h

            if chosen_side == 1:
                sl = chosen_low - SL_ATR_BUFFER * atr1h
                if sl >= entry:
                    continue
                if tp_mode == "fixed_2r":
                    tp = entry + 2.0 * (entry - sl)
                else:  # fvg_mid
                    tp = (chosen_low + chosen_high) / 2.0
                if tp <= entry:
                    continue
                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    open_a, high_a, low_a, close_a, fill_idx, "long",
                    entry, sl, tp, max_bars_exit
                )
                trades.append(Trade(
                    symbol=symbol, side="long",
                    entry_price=float(entry), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - fill_idx) * BAR_SEC_5M),
                    entry_ts=pd.Timestamp(ts_a[fill_idx]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"fvg_low": float(chosen_low), "fvg_high": float(chosen_high),
                           "tp_mode": tp_mode, "htf_align": htf_align},
                ))
            else:
                sl = chosen_high + SL_ATR_BUFFER * atr1h
                if sl <= entry:
                    continue
                if tp_mode == "fixed_2r":
                    tp = entry - 2.0 * (sl - entry)
                else:
                    tp = (chosen_low + chosen_high) / 2.0
                if tp >= entry:
                    continue
                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    open_a, high_a, low_a, close_a, fill_idx, "short",
                    entry, sl, tp, max_bars_exit
                )
                trades.append(Trade(
                    symbol=symbol, side="short",
                    entry_price=float(entry), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - fill_idx) * BAR_SEC_5M),
                    entry_ts=pd.Timestamp(ts_a[fill_idx]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"fvg_low": float(chosen_low), "fvg_high": float(chosen_high),
                           "tp_mode": tp_mode, "htf_align": htf_align},
                ))
            open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== FVG MTF CASCADE — W/F STUDY ===\n")
    print("Cell grid: 3×3×3×2×2 = 108 cells per pair (displ_atr fixed at 0.65)")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants\n")

    engine = _PatchedEngine(
        study=FVGMTFCascadeStrategy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "fvg_mtf_cascade",
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

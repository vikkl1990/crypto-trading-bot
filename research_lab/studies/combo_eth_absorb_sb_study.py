"""W/F STUDY: ETH absorption_bubble + structure_bounce reversal confirmation combo.

Hypothesis (user request):
  absorption_bubble fires on liquidity sweep + heavy volume rejection.
  structure_bounce fires on S/R rejection wick.
  When BOTH fire same-side on ETH within a small window, conviction is much
  higher (different mechanisms confirming same setup).

Modes:
  - "absorb_only"          : absorption only
  - "sb_only"              : structure_bounce only
  - "ensemble_AND"         : both fire same-side within window_bars
  - "ensemble_AS_BOOSTER"  : SB fires; treat as A+ grade if absorption ALSO
                             fired same-side in the last window_bars (booster
                             pattern, not gate). All SB trades pass through;
                             we tag boosted/not for split EV reporting.

Param grid:
  mode         : ["absorb_only", "sb_only", "ensemble_AND", "ensemble_AS_BOOSTER"]
  window_bars  : [1, 3, 6]
  tp_rr        : [1.5, 2.0]
  => 4 × 3 × 2 = 24 cells per pair × ETH only × 3 fee variants = 72 cell-variants.

Notional $400, hard time-stop 30 min (6 bars), SL = absorb_low/swing_low ± 0.3 ATR.

Vectorised; full ETH grid expected to run in ≤5 min.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import (
    Strategy, Trade, WalkForwardEngine,
    QUARTERS, IS_QUARTERS, OOS_QUARTERS, FEE_VARIANTS,
)

BAR_SEC = 300
TIME_STOP_BARS = 6
NOTIONAL_USD = 400.0
SL_ATR_BUFFER = 0.3
VOL_LOOKBACK = 20
ATR_PERIOD = 14
SWING_WINDOW = 20

# absorption-bubble fixed knobs (matched to original study)
ABSORB_BODY_RATIO_MAX = 0.40
ABSORB_WICK_RATIO_MIN = 0.50
ABSORB_LOOKBACK_N = 20      # mid of [10,20,30] grid in original
ABSORB_SWEEP_ATR = 0.2      # mid of [0.1,0.2,0.3]
ABSORB_VOL_MULT = 2.0       # mid of [1.5,2.0,2.5]
ABSORB_DISPL_ATR = 0.5      # mid of [0.4,0.5,0.65]

# structure-bounce fixed knobs (matched to sb_phase_filter_study defaults)
SB_WICK_ATR_MIN = 0.55      # mid of [0.4,0.55,0.7]
SB_VOL_MIN = 1.2            # mid of [1.0,1.2]


# ─────────────────────────────────────────────────────────────────────
# Numpy helpers
# ─────────────────────────────────────────────────────────────────────
def add_atr_np(high, low, close, period=14):
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


def rolling_mean_np(x, window):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    csum = np.cumsum(x)
    for i in range(window - 1, n):
        if i == window - 1:
            out[i] = csum[i] / window
        else:
            out[i] = (csum[i] - csum[i - window]) / window
    return out


def rolling_min_shift1(x, window):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        out[i] = np.min(x[i - window:i])
    return out


def rolling_max_shift1(x, window):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        out[i] = np.max(x[i - window:i])
    return out


def _walk_forward_exit_np(high_a, low_a, close_a, entry_idx, side,
                          entry, sl, tp, max_bars):
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
# Strategy
# ─────────────────────────────────────────────────────────────────────
class ComboETHAbsorbSBStrategy(Strategy):
    """Combo ETH absorption + structure_bounce reversal."""
    name = "combo_eth_absorb_sb"

    # Per-symbol cache of base arrays + signal arrays
    _base_cache: Dict[str, Dict[str, Any]] = {}

    def param_grid(self):
        for mode in ["absorb_only", "sb_only", "ensemble_AND", "ensemble_AS_BOOSTER"]:
            for window_bars in [1, 3, 6]:
                for tp_rr in [1.5, 2.0]:
                    yield {
                        "mode": mode,
                        "window_bars": window_bars,
                        "tp_rr": tp_rr,
                    }

    # ──── Build base arrays + per-bar signal flags (one-time per symbol) ───
    def _build_base(self, df: pd.DataFrame) -> Dict[str, Any]:
        symbol = df.attrs.get("symbol", "ETH")
        if symbol in self._base_cache:
            return self._base_cache[symbol]

        n = len(df)
        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        vol_a = df["volume"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)
        ts_a = df.index.to_numpy()

        # rel_vol (current bar / 20-bar mean including current)
        vmean = rolling_mean_np(vol_a, VOL_LOOKBACK)
        rel_vol = np.divide(
            vol_a, vmean,
            out=np.zeros_like(vol_a), where=(vmean > 0)
        )

        # candle structure
        rng = high_a - low_a
        body_abs = np.abs(close_a - open_a)
        bottom_of_body = np.minimum(open_a, close_a)
        top_of_body = np.maximum(open_a, close_a)
        lower_wick = bottom_of_body - low_a
        upper_wick = high_a - top_of_body
        body_ratio = np.divide(body_abs, rng, out=np.zeros_like(rng), where=(rng > 0))
        lower_wick_ratio = np.divide(lower_wick, rng, out=np.zeros_like(rng), where=(rng > 0))
        upper_wick_ratio = np.divide(upper_wick, rng, out=np.zeros_like(rng), where=(rng > 0))

        # rolling extremes for both sweep (absorption lookback_n) and structure (swing window)
        roll_low_absorb = rolling_min_shift1(low_a, ABSORB_LOOKBACK_N)
        roll_high_absorb = rolling_max_shift1(high_a, ABSORB_LOOKBACK_N)
        swing_low_sb = rolling_min_shift1(low_a, SWING_WINDOW)
        swing_high_sb = rolling_max_shift1(high_a, SWING_WINDOW)

        # absorption candle flags (high vol AND small body AND long wick on relevant side)
        absorb_long_cand = (
            (rel_vol >= ABSORB_VOL_MULT) &
            (body_ratio <= ABSORB_BODY_RATIO_MAX) &
            (lower_wick_ratio >= ABSORB_WICK_RATIO_MIN)
        )
        absorb_short_cand = (
            (rel_vol >= ABSORB_VOL_MULT) &
            (body_ratio <= ABSORB_BODY_RATIO_MAX) &
            (upper_wick_ratio >= ABSORB_WICK_RATIO_MIN)
        )

        # ──── Pre-compute ABSORPTION SIGNAL per bar (the confirmation candle) ───
        # absorption signal at bar k means: bar k = confirmation candle.
        # sweep_idx = k-1, absorption candidate at sweep_idx OR sweep_idx-1.
        # We store: absorb_long_sig[k], absorb_short_sig[k], plus the absorb_low/absorb_high
        # so we can build trade entry/SL when this signal fires.
        absorb_long_sig = np.zeros(n, dtype=np.bool_)
        absorb_short_sig = np.zeros(n, dtype=np.bool_)
        absorb_long_low = np.full(n, np.nan, dtype=np.float64)
        absorb_short_high = np.full(n, np.nan, dtype=np.float64)

        min_start_a = max(VOL_LOOKBACK, ABSORB_LOOKBACK_N, ATR_PERIOD) + 2
        for k in range(min_start_a, n):
            atr_k = atr_a[k]
            sweep_idx = k - 1
            atr_s = atr_a[sweep_idx]
            if not (np.isfinite(atr_k) and atr_k > 0
                    and np.isfinite(atr_s) and atr_s > 0):
                continue

            # LONG absorption signal
            sl_low = roll_low_absorb[sweep_idx]
            if np.isfinite(sl_low):
                swept_long = low_a[sweep_idx] < (sl_low - ABSORB_SWEEP_ATR * atr_s)
                if swept_long:
                    absorb_idx = -1
                    absorb_low_v = 0.0
                    if absorb_long_cand[sweep_idx]:
                        absorb_idx = sweep_idx
                        absorb_low_v = low_a[sweep_idx]
                    elif absorb_long_cand[sweep_idx - 1]:
                        absorb_idx = sweep_idx - 1
                        absorb_low_v = low_a[sweep_idx - 1]
                    if absorb_idx >= 0:
                        cl_k = close_a[k]
                        op_k = open_a[k]
                        body_k = abs(cl_k - op_k)
                        if (cl_k > sl_low) and (cl_k > op_k) and (body_k >= ABSORB_DISPL_ATR * atr_k):
                            absorb_long_sig[k] = True
                            absorb_long_low[k] = absorb_low_v

            # SHORT absorption signal
            sh_high = roll_high_absorb[sweep_idx]
            if np.isfinite(sh_high):
                swept_short = high_a[sweep_idx] > (sh_high + ABSORB_SWEEP_ATR * atr_s)
                if swept_short:
                    absorb_idx = -1
                    absorb_high_v = 0.0
                    if absorb_short_cand[sweep_idx]:
                        absorb_idx = sweep_idx
                        absorb_high_v = high_a[sweep_idx]
                    elif absorb_short_cand[sweep_idx - 1]:
                        absorb_idx = sweep_idx - 1
                        absorb_high_v = high_a[sweep_idx - 1]
                    if absorb_idx >= 0:
                        cl_k = close_a[k]
                        op_k = open_a[k]
                        body_k = abs(cl_k - op_k)
                        if (cl_k < sh_high) and (cl_k < op_k) and (body_k >= ABSORB_DISPL_ATR * atr_k):
                            absorb_short_sig[k] = True
                            absorb_short_high[k] = absorb_high_v

        # ──── Pre-compute STRUCTURE_BOUNCE SIGNAL per bar ────
        # rejection wick at S/R + body confirmation + vol_min.
        # LONG: lower_wick >= sb_wick_atr_min × ATR AND price within 1 ATR of swing_low
        #       AND close > bar_mid AND rel_vol >= vol_min
        # We also store SL reference (= bar low, used as the swing extreme).
        sb_long_sig = np.zeros(n, dtype=np.bool_)
        sb_short_sig = np.zeros(n, dtype=np.bool_)
        sb_long_low = np.full(n, np.nan, dtype=np.float64)
        sb_short_high = np.full(n, np.nan, dtype=np.float64)

        bar_mid = (high_a + low_a) / 2.0
        min_start_sb = max(SWING_WINDOW + ATR_PERIOD + 5, 25)
        for i in range(min_start_sb, n):
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            rv = rel_vol[i]
            if not np.isfinite(rv) or rv < SB_VOL_MIN:
                continue
            sh = swing_high_sb[i]; sl_ref = swing_low_sb[i]
            if not (np.isfinite(sh) and np.isfinite(sl_ref)):
                continue
            o = open_a[i]; h = high_a[i]; l = low_a[i]; c = close_a[i]
            uw = upper_wick[i]; lw = lower_wick[i]
            mid = bar_mid[i]
            # LONG SB
            if lw >= SB_WICK_ATR_MIN * atr and l <= sl_ref + atr and l >= sl_ref - atr:
                if c > mid:
                    sb_long_sig[i] = True
                    sb_long_low[i] = l
            # SHORT SB
            if uw >= SB_WICK_ATR_MIN * atr and h >= sh - atr and h <= sh + atr:
                if c < mid:
                    sb_short_sig[i] = True
                    sb_short_high[i] = h

        base = {
            "open": open_a, "high": high_a, "low": low_a, "close": close_a,
            "atr": atr_a, "ts": ts_a, "symbol": symbol,
            "absorb_long_sig": absorb_long_sig,
            "absorb_short_sig": absorb_short_sig,
            "absorb_long_low": absorb_long_low,
            "absorb_short_high": absorb_short_high,
            "sb_long_sig": sb_long_sig,
            "sb_short_sig": sb_short_sig,
            "sb_long_low": sb_long_low,
            "sb_short_high": sb_short_high,
        }
        self._base_cache[symbol] = base
        return base

    # ──── Per-cell simulate ────────────────────────────────────────────
    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 100:
            return []
        base = self._build_base(df)
        open_a = base["open"]; high_a = base["high"]; low_a = base["low"]
        close_a = base["close"]; atr_a = base["atr"]; ts_a = base["ts"]
        symbol = base["symbol"]
        absorb_long_sig = base["absorb_long_sig"]
        absorb_short_sig = base["absorb_short_sig"]
        absorb_long_low = base["absorb_long_low"]
        absorb_short_high = base["absorb_short_high"]
        sb_long_sig = base["sb_long_sig"]
        sb_short_sig = base["sb_short_sig"]
        sb_long_low = base["sb_long_low"]
        sb_short_high = base["sb_short_high"]

        mode = str(params["mode"])
        window_bars = int(params["window_bars"])
        tp_rr = float(params["tp_rr"])

        trades: List[Trade] = []
        open_until = -1
        end_i = n - (TIME_STOP_BARS + 5)
        start_i = max(VOL_LOOKBACK + ABSORB_LOOKBACK_N + ATR_PERIOD + 5, 50)

        # Helper: was absorption fired in [i - window_bars, i] same-side?
        def absorb_fired_recent(i, side):
            lo = max(0, i - window_bars)
            hi = i + 1
            if side == "long":
                return bool(np.any(absorb_long_sig[lo:hi]))
            else:
                return bool(np.any(absorb_short_sig[lo:hi]))

        # Helper: was SB fired in [i - window_bars, i] same-side?
        def sb_fired_recent(i, side):
            lo = max(0, i - window_bars)
            hi = i + 1
            if side == "long":
                return bool(np.any(sb_long_sig[lo:hi]))
            else:
                return bool(np.any(sb_short_sig[lo:hi]))

        for i in range(start_i, end_i):
            if i <= open_until:
                continue
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue

            # Determine candidates (side, entry, sl, boost_flag) per mode
            #   We may have at most one LONG and one SHORT candidate per bar;
            #   prefer LONG if both fire (rare; just pick one deterministically).
            cand = None  # (side, entry, sl_price, boosted)

            # ──── absorb_only: trade absorption signals ─────────────────
            if mode == "absorb_only":
                if absorb_long_sig[i]:
                    entry = close_a[i]
                    sl = absorb_long_low[i] - SL_ATR_BUFFER * atr
                    if sl < entry:
                        cand = ("long", entry, sl, False)
                elif absorb_short_sig[i]:
                    entry = close_a[i]
                    sl = absorb_short_high[i] + SL_ATR_BUFFER * atr
                    if sl > entry:
                        cand = ("short", entry, sl, False)

            # ──── sb_only: trade SB signals only ────────────────────────
            elif mode == "sb_only":
                if sb_long_sig[i]:
                    entry = close_a[i]
                    sl = sb_long_low[i] - SL_ATR_BUFFER * atr
                    if sl < entry:
                        cand = ("long", entry, sl, False)
                elif sb_short_sig[i]:
                    entry = close_a[i]
                    sl = sb_short_high[i] + SL_ATR_BUFFER * atr
                    if sl > entry:
                        cand = ("short", entry, sl, False)

            # ──── ensemble_AND: BOTH same-side within window_bars ───────
            #   Trigger bar is whichever fires; require the other within window.
            elif mode == "ensemble_AND":
                # LONG ensemble — SB or absorb fires at i AND the other in window
                fired_sb_long = sb_long_sig[i]
                fired_ab_long = absorb_long_sig[i]
                if (fired_sb_long and absorb_fired_recent(i, "long")) or \
                   (fired_ab_long and sb_fired_recent(i, "long")):
                    # Use whichever fired AT i for SL; prefer SB if both
                    if fired_sb_long:
                        entry = close_a[i]
                        sl = sb_long_low[i] - SL_ATR_BUFFER * atr
                    else:
                        entry = close_a[i]
                        sl = absorb_long_low[i] - SL_ATR_BUFFER * atr
                    if sl < entry:
                        cand = ("long", entry, sl, True)
                else:
                    fired_sb_short = sb_short_sig[i]
                    fired_ab_short = absorb_short_sig[i]
                    if (fired_sb_short and absorb_fired_recent(i, "short")) or \
                       (fired_ab_short and sb_fired_recent(i, "short")):
                        if fired_sb_short:
                            entry = close_a[i]
                            sl = sb_short_high[i] + SL_ATR_BUFFER * atr
                        else:
                            entry = close_a[i]
                            sl = absorb_short_high[i] + SL_ATR_BUFFER * atr
                        if sl > entry:
                            cand = ("short", entry, sl, True)

            # ──── ensemble_AS_BOOSTER: SB fires; flag boosted if absorb in window ─
            elif mode == "ensemble_AS_BOOSTER":
                if sb_long_sig[i]:
                    entry = close_a[i]
                    sl = sb_long_low[i] - SL_ATR_BUFFER * atr
                    if sl < entry:
                        boosted = absorb_fired_recent(i, "long")
                        cand = ("long", entry, sl, boosted)
                elif sb_short_sig[i]:
                    entry = close_a[i]
                    sl = sb_short_high[i] + SL_ATR_BUFFER * atr
                    if sl > entry:
                        boosted = absorb_fired_recent(i, "short")
                        cand = ("short", entry, sl, boosted)

            if cand is None:
                continue

            side, entry, sl, boosted = cand
            if side == "long":
                risk = entry - sl
                tp = entry + tp_rr * risk
                if tp <= entry:
                    continue
            else:
                risk = sl - entry
                tp = entry - tp_rr * risk
                if tp >= entry:
                    continue

            exit_idx, exit_price, reason = _walk_forward_exit_np(
                high_a, low_a, close_a, i, side,
                entry, sl, tp, TIME_STOP_BARS,
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
                       "tp_rr": tp_rr, "boosted": boosted},
            ))
            open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


# ─────────────────────────────────────────────────────────────────────
# Booster-split helper (post-run, scoped to ensemble_AS_BOOSTER cells)
# ─────────────────────────────────────────────────────────────────────
def compute_booster_split(study: ComboETHAbsorbSBStrategy,
                          engine: _PatchedEngine,
                          params: Dict[str, Any],
                          symbol: str = "ETH",
                          tf: str = "5m") -> Dict[str, Any]:
    """Re-run booster cell, split trades into boosted vs non-boosted; compute
    EV/trade per group across all quarters under variant A_full_taker."""
    df = engine.load_candles(symbol, tf)
    trades = study.simulate(df, params)
    boosted = [t for t in trades if t.extra.get("boosted")]
    plain = [t for t in trades if not t.extra.get("boosted")]

    def evs(buf: List[Trade]) -> Tuple[int, float, float]:
        if not buf:
            return (0, 0.0, 0.0)
        nets = [engine.apply_fees(t, "A_full_taker") for t in buf]
        ev = sum(nets) / len(nets)
        wr = sum(1 for x in nets if x > 0) / len(nets)
        return (len(nets), ev, wr)

    n_b, ev_b, wr_b = evs(boosted)
    n_p, ev_p, wr_p = evs(plain)
    return {
        "params": params,
        "n_boosted": n_b, "ev_boosted": ev_b, "wr_boosted": wr_b,
        "n_plain": n_p, "ev_plain": ev_p, "wr_plain": wr_p,
        "lift_ev": ev_b - ev_p,
    }


if __name__ == "__main__":
    print("=== COMBO ETH ABSORPTION + STRUCTURE_BOUNCE — W/F STUDY ===\n")
    print("Cell grid: 4 modes × 3 window_bars × 2 tp_rr = 24 cells per pair")
    print("Symbols: ETH × 5m × 3 fee variants = 72 cell-variants total\n")

    study = ComboETHAbsorbSBStrategy()
    engine = _PatchedEngine(
        study=study,
        symbols=["ETH"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "combo_eth_absorb_sb",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())

    # Verdict distribution
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1

    # Per-mode aggregation under variant A_full_taker
    def parse_cell_key(k: str):
        sym, tf, cell, variant = k.split("|", 3)
        kv = dict(p.split("=", 1) for p in cell.split("_") if "=" in p)
        return sym, tf, cell, variant, kv

    mode_pass = defaultdict(lambda: [0, 0])  # [pass, total]
    mode_is_evs = defaultdict(list)
    mode_q4_evs = defaultdict(list)
    mode_n_total = defaultdict(list)
    for k, v in cells:
        sym, tf, cell, variant, kv = parse_cell_key(k)
        if variant != "A_full_taker":
            continue
        m = kv.get("mode", "?")
        mode_pass[m][1] += 1
        if v["verdict"] == "PASS":
            mode_pass[m][0] += 1
        mode_is_evs[m].append(v["is_ev"])
        mode_q4_evs[m].append(v["q4_ev"])
        mode_n_total[m].append(v["n_total"])

    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}\n")

    print("Verdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    print(f"\n=== PER-MODE A/B (variant A_full_taker) ===")
    for m in ["absorb_only", "sb_only", "ensemble_AND", "ensemble_AS_BOOSTER"]:
        if m not in mode_pass:
            continue
        p, t = mode_pass[m]
        avg_is = sum(mode_is_evs[m]) / len(mode_is_evs[m]) if mode_is_evs[m] else 0
        avg_q4 = sum(mode_q4_evs[m]) / len(mode_q4_evs[m]) if mode_q4_evs[m] else 0
        avg_n = sum(mode_n_total[m]) / len(mode_n_total[m]) if mode_n_total[m] else 0
        print(f"  {m:25s} PASS {p}/{t}  avg IS_EV=${avg_is:+.4f}  "
              f"avg Q4_EV=${avg_q4:+.4f}  avg n_total={avg_n:.1f}")

    # ─── Booster split (the headline question) ────────────────────────
    print("\n=== BOOSTER SPLIT — ensemble_AS_BOOSTER cells (variant A_full_taker) ===")
    print("For each ensemble_AS_BOOSTER cell, compare SB-with-absorb-confirm vs SB-without:")
    booster_rows = []
    for tp_rr in [1.5, 2.0]:
        for wb in [1, 3, 6]:
            params = {"mode": "ensemble_AS_BOOSTER", "window_bars": wb, "tp_rr": tp_rr}
            split = compute_booster_split(study, engine, params)
            booster_rows.append(split)
            print(f"  window={wb} tp_rr={tp_rr}: "
                  f"BOOSTED n={split['n_boosted']:3d} EV=${split['ev_boosted']:+.4f} WR={split['wr_boosted']*100:.0f}%   "
                  f"PLAIN n={split['n_plain']:3d} EV=${split['ev_plain']:+.4f} WR={split['wr_plain']*100:.0f}%   "
                  f"lift=${split['lift_ev']:+.4f}")

    # Top PASS / top IS cells
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

    # Stash booster split into JSON for later
    import json
    extras_path = ROOT / "storage" / "wf_studies" / "combo_eth_absorb_sb" / "booster_split.json"
    extras_path.write_text(json.dumps(booster_rows, indent=2, default=str))
    print(f"\nBooster split JSON: {extras_path}")

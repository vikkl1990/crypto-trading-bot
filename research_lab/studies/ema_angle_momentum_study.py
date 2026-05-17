"""W/F STUDY: EMA Momentum + Angle/Slope Filter (5m).

Tests a refined version of the (silent) `ema_momentum` scanner. The KEY new
variable is an EMA SLOPE/ANGLE filter — the hypothesis is that EMA crossover
ALONE is noise (today's ema_momentum_relax study killed all 120 cells), but
EMA crossover + steep slope is a momentum trade with edge.

Setup (LONG; mirror for SHORT):
  Direction bias    : EMA(fast) > EMA(slow) at bar i
  Angle/slope filter: slope_atr = (ema_fast[i] - ema_fast[i-N]) / atr / N
                      require abs(slope_atr) >= angle_min_atr
                      slope sign must match bias direction
  Entry trigger     : "fresh_cross"   = bias true at i, false at i-1
                      "stack_aligned" = bias persistent, slope just stepped above thr
  Confirmation     : body >= body_atr_min × ATR
  Entry            : confirmation candle close
  Stop             : entry - 0.5 × ATR (long) / + 0.5 × ATR (short)
  Target           : tp_rr × R
  Hard time stop   : 30 min (6 bars)
  Notional         : $400 fixed

CRITICAL A/B: `slope_check` ∈ {True, False} — does the angle filter add value?

Param grid (reduced per architect spec):
  ema_fast=7, ema_slow=17, body_atr_min=0.3 fixed (matches user spec)
  slope_window_n  ∈ [3, 5, 10]
  angle_min_atr   ∈ [0.05, 0.10, 0.15, 0.20]
  tp_rr           ∈ [1.5, 2.0]
  entry_mode      ∈ ["fresh_cross", "stack_aligned"]
  slope_check     ∈ [True, False]
  → 3 × 4 × 2 × 2 × 2 = 96 cells per pair
  × 4 pairs × 3 fee variants = 1152 cell-variants total
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
TIME_STOP_SEC = 1800       # 30 min hard time stop
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
SL_ATR = 0.5               # tight scalp stop
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


def add_ema_np(values: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA — alpha = 2/(period+1). Vectorised seed + iterative tail."""
    n = len(values)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return ema
    # Seed with SMA of first `period` values
    ema[period - 1] = np.mean(values[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _walk_forward_exit_np(open_a: np.ndarray, high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, entry_idx: int, side: str,
                          entry: float, sl: float, tp: float,
                          max_bars: int) -> Tuple[int, float, str]:
    """Bar-by-bar walk to first SL/TP/time-stop. Numpy version."""
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


class EMAAngleMomentumStrategy(Strategy):
    """EMA Momentum + Angle/Slope Filter — 5m scalp.

    See module docstring for full spec.
    """
    name = "ema_angle_momentum"

    # Fixed per architect spec (matches user's clean idea — 7/17 with body 0.3)
    EMA_FAST = 7
    EMA_SLOW = 17
    BODY_ATR_MIN = 0.3

    def param_grid(self):
        for slope_window_n in [3, 5, 10]:
            for angle_min_atr in [0.05, 0.10, 0.15, 0.20]:
                for tp_rr in [1.5, 2.0]:
                    for entry_mode in ["fresh_cross", "stack_aligned"]:
                        for slope_check in [True, False]:
                            yield {
                                "slope_window_n": slope_window_n,
                                "angle_min_atr": angle_min_atr,
                                "tp_rr": tp_rr,
                                "entry_mode": entry_mode,
                                "slope_check": slope_check,
                            }

    def cell_id(self, params: Dict[str, Any]) -> str:
        # Compact, readable cell id
        sw = params["slope_window_n"]
        am = params["angle_min_atr"]
        rr = params["tp_rr"]
        em = params["entry_mode"]
        sc = "sON" if params["slope_check"] else "sOFF"
        return f"sw{sw}_am{am}_rr{rr}_{em}_{sc}"

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 100:
            return []
        symbol = df.attrs.get("symbol", "BTC")

        # ── Pre-compute arrays (cell-level — recomputed per call but cheap) ──
        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, 14)
        ema_f = add_ema_np(close_a, self.EMA_FAST)
        ema_s = add_ema_np(close_a, self.EMA_SLOW)
        ts_a = df.index.to_numpy()

        body = np.abs(close_a - open_a)
        bullish = close_a > open_a
        bearish = close_a < open_a

        # Direction bias
        long_bias = ema_f > ema_s
        short_bias = ema_f < ema_s

        # Slope of EMA(fast) over N bars (ATR-normalized per bar)
        N = int(params["slope_window_n"])
        slope = np.full(n, np.nan, dtype=np.float64)
        for i in range(N, n):
            if not (np.isfinite(ema_f[i]) and np.isfinite(ema_f[i - N])
                    and np.isfinite(atr_a[i]) and atr_a[i] > 0):
                continue
            slope[i] = (ema_f[i] - ema_f[i - N]) / atr_a[i] / N

        angle_min = float(params["angle_min_atr"])
        slope_check = bool(params["slope_check"])
        tp_rr = float(params["tp_rr"])
        entry_mode = str(params["entry_mode"])
        body_atr_min = self.BODY_ATR_MIN

        # Per-bar slope gate (vectorised)
        slope_long_ok = np.zeros(n, dtype=bool)
        slope_short_ok = np.zeros(n, dtype=bool)
        finite_slope = np.isfinite(slope)
        if slope_check:
            slope_long_ok = finite_slope & (slope >= angle_min)
            slope_short_ok = finite_slope & (slope <= -angle_min)
        else:
            slope_long_ok = np.ones(n, dtype=bool)
            slope_short_ok = np.ones(n, dtype=bool)

        # Entry mode masks
        if entry_mode == "fresh_cross":
            # Cross detection: bias true at i, false at i-1
            long_cross = np.zeros(n, dtype=bool)
            short_cross = np.zeros(n, dtype=bool)
            long_cross[1:] = long_bias[1:] & ~long_bias[:-1]
            short_cross[1:] = short_bias[1:] & ~short_bias[:-1]
            long_trigger = long_cross
            short_trigger = short_cross
        else:  # stack_aligned: bias persistent + slope just crossed threshold
            if slope_check:
                # Slope just stepped above threshold (this bar yes, prior bar no)
                slope_step_long = np.zeros(n, dtype=bool)
                slope_step_short = np.zeros(n, dtype=bool)
                slope_step_long[1:] = slope_long_ok[1:] & ~slope_long_ok[:-1]
                slope_step_short[1:] = slope_short_ok[1:] & ~slope_short_ok[:-1]
                long_trigger = long_bias & slope_step_long
                short_trigger = short_bias & slope_step_short
            else:
                # If slope_check=False, "stack_aligned" reduces to "any persistent
                # bar where bias was already true the previous bar" — sample one
                # entry per persistent stack to avoid firing every bar. Use the
                # bar AFTER the cross (stack now confirmed for ≥1 bar).
                long_trigger = np.zeros(n, dtype=bool)
                short_trigger = np.zeros(n, dtype=bool)
                long_trigger[2:] = long_bias[2:] & long_bias[1:-1] & ~long_bias[:-2]
                short_trigger[2:] = short_bias[2:] & short_bias[1:-1] & ~short_bias[:-2]

        # Body confirmation (always required)
        body_ok = np.zeros(n, dtype=bool)
        finite_atr = np.isfinite(atr_a) & (atr_a > 0)
        body_ok[finite_atr] = body[finite_atr] >= body_atr_min * atr_a[finite_atr]

        # Final candidate masks
        long_candidates = (long_bias & long_trigger & slope_long_ok
                           & body_ok & bullish & finite_atr)
        short_candidates = (short_bias & short_trigger & slope_short_ok
                            & body_ok & bearish & finite_atr)

        trades: List[Trade] = []
        open_until = -1
        max_bars_exit = TIME_STOP_BARS
        # Don't enter near the end (need lookahead bars to resolve)
        end_i = n - (TIME_STOP_BARS + 5)
        # Need enough lookback for EMA/ATR/slope to be defined
        start_i = max(self.EMA_SLOW + 1, N + 1, 14 + 1)

        for i in range(start_i, end_i):
            if i <= open_until:
                continue
            if not finite_atr[i]:
                continue
            atr = atr_a[i]

            if long_candidates[i]:
                entry = close_a[i]
                sl = entry - SL_ATR * atr
                tp = entry + tp_rr * (entry - sl)
                if sl >= entry or tp <= entry:
                    continue
                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    open_a, high_a, low_a, close_a, i, "long",
                    entry, sl, tp, max_bars_exit
                )
                trades.append(Trade(
                    symbol=symbol, side="long",
                    entry_price=float(entry), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - i) * BAR_SEC),
                    entry_ts=pd.Timestamp(ts_a[i]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"slope": float(slope[i]) if np.isfinite(slope[i]) else None,
                           "ema_f": float(ema_f[i]), "ema_s": float(ema_s[i]),
                           "atr": float(atr)},
                ))
                open_until = exit_idx
                continue

            if short_candidates[i]:
                entry = close_a[i]
                sl = entry + SL_ATR * atr
                tp = entry - tp_rr * (sl - entry)
                if sl <= entry or tp >= entry:
                    continue
                exit_idx, exit_price, reason = _walk_forward_exit_np(
                    open_a, high_a, low_a, close_a, i, "short",
                    entry, sl, tp, max_bars_exit
                )
                trades.append(Trade(
                    symbol=symbol, side="short",
                    entry_price=float(entry), exit_price=float(exit_price),
                    notional_usd=NOTIONAL_USD,
                    holding_sec=int((exit_idx - i) * BAR_SEC),
                    entry_ts=pd.Timestamp(ts_a[i]),
                    exit_ts=pd.Timestamp(ts_a[exit_idx]),
                    exit_reason=reason,
                    extra={"slope": float(slope[i]) if np.isfinite(slope[i]) else None,
                           "ema_f": float(ema_f[i]), "ema_s": float(ema_s[i]),
                           "atr": float(atr)},
                ))
                open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== EMA ANGLE MOMENTUM — W/F STUDY ===\n")
    print("Cell grid: 3 × 4 × 2 × 2 × 2 = 96 cells per pair")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants = 1152 cell-variants total")
    print("Critical A/B: slope_check=True vs slope_check=False\n")

    engine = _PatchedEngine(
        study=EMAAngleMomentumStrategy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "ema_angle_momentum",
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

    # ── A/B headline: slope_check=True vs False ───────────────────────────
    son_cells = []
    soff_cells = []
    for k, v in cells:
        if "_sON_" in k or k.endswith("_sON"):
            son_cells.append(v)
        elif "_sOFF_" in k or k.endswith("_sOFF"):
            soff_cells.append(v)

    def cell_stats(buf):
        if not buf:
            return None
        n_total = len(buf)
        passed = sum(1 for v in buf if v["verdict"] == "PASS")
        held = sum(1 for v in buf if v["verdict"].startswith("HOLD"))
        avg_is = sum(v["is_ev"] for v in buf if v["is_n"] >= 10) / max(
            1, sum(1 for v in buf if v["is_n"] >= 10))
        avg_n = sum(v["n_total"] for v in buf) / n_total
        return n_total, passed, held, avg_is, avg_n

    print("\n=== A/B HEADLINE: slope_check ===")
    son = cell_stats(son_cells)
    soff = cell_stats(soff_cells)
    if son:
        n, p, h, ev, nt = son
        print(f"  slope_check=ON : {n} cells | PASS={p} | HOLD={h} | "
              f"avg_IS_EV=${ev:+.3f} | avg_trades_per_cell={nt:.0f}")
    if soff:
        n, p, h, ev, nt = soff
        print(f"  slope_check=OFF: {n} cells | PASS={p} | HOLD={h} | "
              f"avg_IS_EV=${ev:+.3f} | avg_trades_per_cell={nt:.0f}")

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

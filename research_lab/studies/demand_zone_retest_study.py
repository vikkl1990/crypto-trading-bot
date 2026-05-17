"""W/F STUDY: Demand Zone Retest (smart money concept).

Tests a NEW scanner candidate for the VN Edge bot — Phase B BTC/ETH
diversification. The current bot is 97.8% structure_bounce monoculture; this
study validates whether a Demand Zone Retest pattern can be a 2nd validated
strategy with maker-friendly entries at visible zone levels.

Setup (LONG; mirror for SHORT as Supply Zone Retest):
  Step 1 — IMPULSE LEG: 3-N consecutive bullish bars whose cumulative range
           (last_high - first_low) >= impulse_atr_min × ATR.
  Step 2 — DEMAND ZONE: the LAST bullish-then-bearish-rejection candle BEFORE
           the impulse — operationalized as the most-recent BEARISH bar before
           impulse_start. Zone = (zone_high, zone_low) of that bar.
  Step 3 — PULLBACK: within retest_lookback bars after impulse peak, price
           retraces back into the zone (low <= zone_high AND high >= zone_low).
  Step 4 — CONTINUATION CANDLE: a bar that closes above zone_high with body
           >= body_atr_min × ATR (retest confirmation).
  Step 5 — ENTRY at continuation close. STOP at zone_low - 0.3 × ATR.
           TARGET: 1:2 RR (fixed_2r) OR measured-move from impulse height.

Hard time stop 30 minutes (6 bars). Notional $400 fixed.

Implementation note: per-cell execution is vectorised on numpy arrays — pandas
iloc was 30×+ slower per cell. Full grid (162 × 2) runs in ~3-5 minutes.
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
SL_ATR_BUFFER = 0.3         # stop = zone_low - 0.3 × ATR
NOTIONAL_USD = 400.0
ZONE_LOOKBACK = 8           # max bars before impulse_start to look for zone


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
    # Simple rolling mean
    atr = np.full(n, np.nan, dtype=np.float64)
    csum = np.cumsum(tr)
    for i in range(period - 1, n):
        if i == period - 1:
            atr[i] = csum[i] / period
        else:
            atr[i] = (csum[i] - csum[i - period]) / period
    return atr


def _walk_forward_exit_np(open_a: np.ndarray, high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, entry_idx: int, side: str,
                          entry: float, sl: float, tp: float,
                          max_bars: int) -> Tuple[int, float, str]:
    """Bar-by-bar walk to first SL/TP/time-stop. Numpy version.

    Returns (exit_idx, exit_price, reason). Reason ∈ {sl_hit, tp_hit, time_stop}.
    Always returns; falls back to forced_end if data runs out.
    """
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


class DemandZoneRetestStrategy(Strategy):
    """Demand Zone Retest — 5-step pattern.

    Param sweep:
      impulse_bars: max consecutive bars to look back for impulse leg (3, 4, 5)
      impulse_atr_min: min impulse cumulative range in ATRs (1.0, 1.5, 2.0)
      retest_lookback: bars after impulse peak to wait for pullback (5, 10, 15)
      body_atr_min: min continuation candle body in ATRs (0.4, 0.55, 0.65)
      tp_mode: "fixed_2r" or "measured" (impulse-height projection)
    """
    name = "demand_zone_retest"

    def param_grid(self):
        for impulse_bars in [3, 4, 5]:
            for impulse_atr_min in [1.0, 1.5, 2.0]:
                for retest_lookback in [5, 10, 15]:
                    for body_atr_min in [0.4, 0.55, 0.65]:
                        for tp_mode in ["fixed_2r", "measured"]:
                            yield {
                                "impulse_bars": impulse_bars,
                                "impulse_atr_min": impulse_atr_min,
                                "retest_lookback": retest_lookback,
                                "body_atr_min": body_atr_min,
                                "tp_mode": tp_mode,
                            }

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 50:
            return []
        symbol = df.attrs.get("symbol", "BTC")

        # Pre-compute numpy arrays once
        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, 14)
        ts_a = df.index.to_numpy()  # for entry_ts assignment

        bullish = close_a > open_a
        bearish = close_a < open_a
        body = np.abs(close_a - open_a)

        impulse_bars_max = int(params["impulse_bars"])
        impulse_atr_min = float(params["impulse_atr_min"])
        retest_lookback = int(params["retest_lookback"])
        body_atr_min = float(params["body_atr_min"])
        tp_mode = str(params["tp_mode"])

        impulse_bars_min = 3
        if impulse_bars_max < impulse_bars_min:
            return []

        # Pre-compute per-bar: bullish streak length ENDING at this bar
        # bull_streak[i] = number of consecutive bullish bars ending at i (incl i)
        bull_streak = np.zeros(n, dtype=np.int32)
        bear_streak = np.zeros(n, dtype=np.int32)
        bull_streak[0] = 1 if bullish[0] else 0
        bear_streak[0] = 1 if bearish[0] else 0
        for i in range(1, n):
            bull_streak[i] = bull_streak[i - 1] + 1 if bullish[i] else 0
            bear_streak[i] = bear_streak[i - 1] + 1 if bearish[i] else 0

        trades: List[Trade] = []
        open_until = -1
        min_lookback = impulse_bars_max + ZONE_LOOKBACK + retest_lookback + 5
        max_bars_exit = TIME_STOP_BARS

        end_i = n - (TIME_STOP_BARS + 5)

        for i in range(min_lookback, end_i):
            if i <= open_until:
                continue
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue

            cont_body = body[i]

            # ─── LONG SETUP ────────────────────────────────────────────────
            cont_long_ok = bullish[i] and (cont_body >= body_atr_min * atr)
            if cont_long_ok:
                # Look back through possible impulse-end indices in retest window
                start_search = max(min_lookback, i - retest_lookback - 1)
                trade_emitted = False
                for idx_p in range(i - 1, start_search - 1, -1):
                    n_streak = bull_streak[idx_p]
                    if n_streak < impulse_bars_min:
                        continue
                    # Use up to impulse_bars_max from the streak
                    used = min(n_streak, impulse_bars_max)
                    imp_end = idx_p
                    imp_start = idx_p - used + 1
                    imp_high = high_a[imp_end]
                    imp_low = low_a[imp_start]
                    if (imp_high - imp_low) < impulse_atr_min * atr:
                        continue

                    # Demand zone: most-recent BEARISH bar before imp_start
                    zone_idx = -1
                    zone_min = max(0, imp_start - ZONE_LOOKBACK)
                    for j in range(imp_start - 1, zone_min - 1, -1):
                        if bearish[j]:
                            zone_idx = j
                            break
                    if zone_idx < 0:
                        continue
                    zone_high = high_a[zone_idx]
                    zone_low = low_a[zone_idx]

                    # Continuation must close above zone_high
                    if close_a[i] <= zone_high:
                        continue

                    # Pullback: any bar in (imp_end, i) touched zone
                    pulled_back = False
                    for j in range(imp_end + 1, i):
                        if low_a[j] <= zone_high and high_a[j] >= zone_low:
                            pulled_back = True
                            break
                    if not pulled_back:
                        continue

                    # Emit LONG
                    entry = close_a[i]
                    sl = zone_low - SL_ATR_BUFFER * atr
                    if tp_mode == "fixed_2r":
                        tp = entry + 2.0 * (entry - sl)
                    else:
                        tp = entry + (imp_high - imp_low)
                    if tp <= entry or sl >= entry:
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
                        extra={"zone_low": float(zone_low), "zone_high": float(zone_high),
                               "imp_high": float(imp_high), "imp_low": float(imp_low),
                               "tp_mode": tp_mode},
                    ))
                    open_until = exit_idx
                    trade_emitted = True
                    break
                if trade_emitted:
                    continue

            # ─── SHORT SETUP ───────────────────────────────────────────────
            cont_short_ok = bearish[i] and (cont_body >= body_atr_min * atr)
            if cont_short_ok:
                start_search = max(min_lookback, i - retest_lookback - 1)
                for idx_p in range(i - 1, start_search - 1, -1):
                    n_streak = bear_streak[idx_p]
                    if n_streak < impulse_bars_min:
                        continue
                    used = min(n_streak, impulse_bars_max)
                    imp_end = idx_p
                    imp_start = idx_p - used + 1
                    imp_low = low_a[imp_end]
                    imp_high = high_a[imp_start]
                    if (imp_high - imp_low) < impulse_atr_min * atr:
                        continue

                    # Supply zone: most-recent BULLISH bar before imp_start
                    zone_idx = -1
                    zone_min = max(0, imp_start - ZONE_LOOKBACK)
                    for j in range(imp_start - 1, zone_min - 1, -1):
                        if bullish[j]:
                            zone_idx = j
                            break
                    if zone_idx < 0:
                        continue
                    zone_high = high_a[zone_idx]
                    zone_low = low_a[zone_idx]

                    if close_a[i] >= zone_low:
                        continue

                    pulled_back = False
                    for j in range(imp_end + 1, i):
                        if low_a[j] <= zone_high and high_a[j] >= zone_low:
                            pulled_back = True
                            break
                    if not pulled_back:
                        continue

                    entry = close_a[i]
                    sl = zone_high + SL_ATR_BUFFER * atr
                    if tp_mode == "fixed_2r":
                        tp = entry - 2.0 * (sl - entry)
                    else:
                        tp = entry - (imp_high - imp_low)
                    if tp >= entry or sl <= entry:
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
                        extra={"zone_low": float(zone_low), "zone_high": float(zone_high),
                               "imp_high": float(imp_high), "imp_low": float(imp_low),
                               "tp_mode": tp_mode},
                    ))
                    open_until = exit_idx
                    break

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== DEMAND ZONE RETEST — W/F STUDY ===\n")
    print("Cell grid: 3 × 3 × 3 × 3 × 2 = 162 cells per pair")
    print("Symbols: BTC, ETH × 5m × 3 fee variants = 972 cell-variants total\n")

    engine = _PatchedEngine(
        study=DemandZoneRetestStrategy(),
        symbols=["BTC", "ETH"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "demand_zone_retest",
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

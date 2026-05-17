#!/usr/bin/env python3
"""Exhaustion 5-bar Reversal — W/F study (candidate #2, 2026-05-03).

Pattern (LONG mirror; spec'd as SHORT after up-exhaustion):
  Step 1 — STREAK: last `streak_n` closed bars all directional same way
           (5 consecutive higher closes for SHORT setup; 5 lower for LONG).
  Step 2 — ATR EXPANSION: ATR(14) at last bar >= ATR(14) average over
           last `expansion_lb` bars × `expansion_mult`.
  Step 3 — SWEEP: streak's terminal bar high (or low) makes a new local extreme,
           penetrating prior `sweep_lookback`-bar extreme by >= sweep_atr × ATR.
  Step 4 — RECLAIM: NEXT bar closes back through the streak's terminal bar's
           midpoint AND opposes the streak direction (close < open for SHORT).
  Entry:  reclaim candle close (maker — variant C).
  SL:     beyond streak terminal extreme + 0.3 × ATR.
  TP:     RR=tp_rr from entry.
  Time:   45 min hard stop.

Symbols: BTC, ETH, SOL.
TF: 5m.

Cells:
  streak_n ∈ {5, 7}
  expansion_mult ∈ {1.5, 2.0}
  sweep_atr ∈ {0.3, 0.5}
  tp_rr ∈ {1.5, 2.0}
  → 16 cells × 3 syms × 3 fee variants

Output: storage/wf_studies/exhaustion_5bar_reversal/{walkforward.json,report.md}
"""
from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wf_harness import (  # noqa: E402
    Strategy, Trade, WalkForwardEngine, ROOT as HARNESS_ROOT,
)

ATR_PERIOD = 14
EXPANSION_LB = 30
TIME_STOP_MIN = 45
SL_BUFFER_ATR = 0.3
NOTIONAL = 1000.0


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h = df["high"].astype(float); l = df["low"].astype(float)
    c = df["close"].astype(float).shift(1)
    tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


class Exhaustion5barReversal(Strategy):
    name = "exhaustion_5bar_reversal"

    def param_grid(self) -> Iterator[Dict[str, Any]]:
        for streak_n, expansion_mult, sweep_atr, tp_rr in product(
            [5, 7],
            [1.5, 2.0],
            [0.3, 0.5],
            [1.5, 2.0],
        ):
            yield {
                "streak_n": streak_n,
                "expansion_mult": expansion_mult,
                "sweep_atr": sweep_atr,
                "tp_rr": tp_rr,
            }

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        if df.empty or len(df) < ATR_PERIOD + EXPANSION_LB + 10:
            return []
        symbol = df.attrs.get("symbol", "?")
        streak_n = int(params["streak_n"])
        expansion_mult = float(params["expansion_mult"])
        sweep_atr = float(params["sweep_atr"])
        tp_rr = float(params["tp_rr"])
        tf_min = 5

        atr = add_atr(df, ATR_PERIOD)
        atr_mean = atr.rolling(EXPANSION_LB, min_periods=EXPANSION_LB).mean()

        closes = df["close"].astype(float).values
        opens = df["open"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values

        trades: List[Trade] = []
        n = len(df)
        i = ATR_PERIOD + EXPANSION_LB + 5
        while i < n - 1:
            atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
            atr_mu_i = float(atr_mean.iloc[i]) if not pd.isna(atr_mean.iloc[i]) else 0.0
            if atr_i <= 0 or atr_mu_i <= 0:
                i += 1; continue
            if atr_i < expansion_mult * atr_mu_i:
                i += 1; continue   # no expansion — skip

            # Check streak ending at bar i (i.e. bars i-streak_n+1 .. i)
            if i - streak_n + 1 < 0:
                i += 1; continue
            seg_closes = closes[i - streak_n + 1: i + 1]
            up_streak = all(seg_closes[k] > seg_closes[k - 1] for k in range(1, streak_n))
            down_streak = all(seg_closes[k] < seg_closes[k - 1] for k in range(1, streak_n))
            if not (up_streak or down_streak):
                i += 1; continue

            # Sweep: terminal bar i extreme penetrates prior sweep_lookback-bar extreme
            sweep_lb = 20
            if i - sweep_lb < 0:
                i += 1; continue
            prior_high = float(np.max(highs[i - sweep_lb: i]))
            prior_low = float(np.min(lows[i - sweep_lb: i]))

            term_hi = float(highs[i]); term_lo = float(lows[i])

            short_setup = up_streak and (term_hi - prior_high >= sweep_atr * atr_i)
            long_setup = down_streak and (prior_low - term_lo >= sweep_atr * atr_i)
            if not (short_setup or long_setup):
                i += 1; continue

            # Reclaim: next bar (i+1) opposes streak + closes through midpoint of bar i
            j = i + 1
            if j >= n:
                break
            mid_i = (highs[i] + lows[i]) / 2.0
            close_j = float(closes[j]); open_j = float(opens[j])

            if short_setup:
                if close_j >= open_j:    # not bearish reclaim
                    i += 1; continue
                if close_j > mid_i:      # didn't reclaim through midpoint
                    i += 1; continue
                side = "short"
                entry_price = close_j
                sl = term_hi + SL_BUFFER_ATR * atr_i
                risk = sl - entry_price
                if risk <= 0: i += 1; continue
                tp = entry_price - tp_rr * risk
            else:
                if close_j <= open_j:    # not bullish reclaim
                    i += 1; continue
                if close_j < mid_i:
                    i += 1; continue
                side = "long"
                entry_price = close_j
                sl = term_lo - SL_BUFFER_ATR * atr_i
                risk = entry_price - sl
                if risk <= 0: i += 1; continue
                tp = entry_price + tp_rr * risk

            entry_idx = j
            bars_max = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
            end_idx = min(entry_idx + bars_max, n - 1)
            exit_reason = "time_stop"
            exit_price = float(closes[end_idx])
            exit_idx = end_idx
            for k in range(entry_idx + 1, end_idx + 1):
                bh = float(highs[k]); bl = float(lows[k])
                if side == "short":
                    if bh >= sl:
                        exit_reason = "sl"; exit_price = sl; exit_idx = k; break
                    if bl <= tp:
                        exit_reason = "tp"; exit_price = tp; exit_idx = k; break
                else:
                    if bl <= sl:
                        exit_reason = "sl"; exit_price = sl; exit_idx = k; break
                    if bh >= tp:
                        exit_reason = "tp"; exit_price = tp; exit_idx = k; break

            holding_sec = max(1, (exit_idx - entry_idx) * tf_min * 60)
            trades.append(Trade(
                symbol=symbol, side=side,
                entry_price=entry_price, exit_price=exit_price,
                notional_usd=NOTIONAL, holding_sec=holding_sec,
                entry_ts=df.index[entry_idx], exit_ts=df.index[exit_idx],
                exit_reason=exit_reason,
            ))
            i = exit_idx + 1
        return trades


def main():
    print("=== exhaustion_5bar_reversal W/F ===")
    out_dir = HARNESS_ROOT / "storage" / "wf_studies" / "exhaustion_5bar_reversal"

    class _Engine(WalkForwardEngine):
        def load_candles(self, symbol, tf):
            df = super().load_candles(symbol, tf)
            df.attrs["symbol"] = symbol
            return df

    engine = _Engine(
        study=Exhaustion5barReversal(),
        symbols=["BTC", "ETH", "SOL"],
        timeframes=["5m"],
        out_dir=out_dir,
    )
    engine.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Liquidity Sweep + CVD Divergence — W/F study (candidate #3, 2026-05-03).

CVD PROXY (we don't have trade-tape data; use candle-derived proxy):
  bar_cvd  = volume × sign(close - open)
  cum_cvd  = rolling sum over `cvd_window` bars (e.g., 10–20)

This is the standard candle-proxy used widely when tick CVD isn't available.
It captures the IMBALANCE between up-volume and down-volume but loses some
intra-bar nuance.

Pattern (LONG):
  Step 1 — SWEEP: bar i's low penetrates `sweep_lookback`-bar low by
           >= sweep_atr × ATR(14).
  Step 2 — CVD DIVERGENCE: cum_cvd at bar i is HIGHER than cum_cvd at the
           bar where the prior `sweep_lookback`-bar low was set.
           (Price made lower low but CVD did not — bullish divergence.)
  Step 3 — RECLAIM: bar i+1 closes back above the sweep level
           (= prior_n_bar_low) AND with body >= reclaim_atr × ATR.
  Entry:   reclaim candle close (maker — variant C).
  SL:      sweep low − 0.3 × ATR.
  TP:      RR=1.5 from entry.
  Time:    30 min hard stop.

Pattern (SHORT): mirror around prior high + bearish divergence.

Symbols: BTC, ETH, SOL.
TF: 5m.

Cells:
  sweep_lookback ∈ {20, 30}
  cvd_window ∈ {10, 20}
  sweep_atr ∈ {0.3, 0.5}
  reclaim_atr ∈ {0.3, 0.5}
  → 16 cells × 3 syms × 3 fee variants

Output: storage/wf_studies/liq_sweep_cvd_divergence/{walkforward.json,report.md}
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
TIME_STOP_MIN = 30
TP_RR = 1.5
SL_BUFFER_ATR = 0.3
NOTIONAL = 1000.0


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h = df["high"].astype(float); l = df["low"].astype(float)
    c = df["close"].astype(float).shift(1)
    tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def cvd_proxy(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling CVD proxy = sum(vol × sign(close-open)) over `window` bars."""
    sign = np.sign(df["close"].astype(float) - df["open"].astype(float))
    bar_cvd = df["volume"].astype(float) * sign
    return bar_cvd.rolling(window, min_periods=window).sum()


class LiqSweepCvdDivergence(Strategy):
    name = "liq_sweep_cvd_divergence"

    def param_grid(self) -> Iterator[Dict[str, Any]]:
        for sweep_lb, cvd_w, sweep_atr, reclaim_atr in product(
            [20, 30],
            [10, 20],
            [0.3, 0.5],
            [0.3, 0.5],
        ):
            yield {
                "sweep_lb": sweep_lb,
                "cvd_w": cvd_w,
                "sweep_atr": sweep_atr,
                "reclaim_atr": reclaim_atr,
            }

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        if df.empty or len(df) < ATR_PERIOD + 50:
            return []
        symbol = df.attrs.get("symbol", "?")
        sweep_lb = int(params["sweep_lb"])
        cvd_w = int(params["cvd_w"])
        sweep_atr = float(params["sweep_atr"])
        reclaim_atr = float(params["reclaim_atr"])
        tf_min = 5

        atr = add_atr(df, ATR_PERIOD)
        cvd = cvd_proxy(df, cvd_w)

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        opens = df["open"].astype(float).values
        closes = df["close"].astype(float).values

        trades: List[Trade] = []
        n = len(df)
        i = max(ATR_PERIOD, sweep_lb, cvd_w) + 5
        while i < n - 2:
            atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
            cvd_i = float(cvd.iloc[i]) if not pd.isna(cvd.iloc[i]) else None
            if atr_i <= 0 or cvd_i is None:
                i += 1; continue

            # Prior sweep_lb-bar extremes (excl current)
            window_lo = float(np.min(lows[i - sweep_lb: i]))
            window_hi = float(np.max(highs[i - sweep_lb: i]))
            # Index of those extremes (within window)
            argmin_in_window = int(np.argmin(lows[i - sweep_lb: i])) + (i - sweep_lb)
            argmax_in_window = int(np.argmax(highs[i - sweep_lb: i])) + (i - sweep_lb)

            bar_lo = float(lows[i]); bar_hi = float(highs[i])

            # LONG sweep: bar_lo < window_lo by >= sweep_atr × ATR
            long_sweep = (window_lo - bar_lo >= sweep_atr * atr_i)
            short_sweep = (bar_hi - window_hi >= sweep_atr * atr_i)
            if not (long_sweep or short_sweep):
                i += 1; continue

            # CVD divergence check
            ref_idx = argmin_in_window if long_sweep else argmax_in_window
            cvd_ref = float(cvd.iloc[ref_idx]) if not pd.isna(cvd.iloc[ref_idx]) else None
            if cvd_ref is None:
                i += 1; continue

            if long_sweep:
                divergent = cvd_i > cvd_ref     # price LL but CVD higher → bullish div
                side = "long"
            else:
                divergent = cvd_i < cvd_ref     # price HH but CVD lower → bearish div
                side = "short"
            if not divergent:
                i += 1; continue

            # Reclaim check on bar i+1
            j = i + 1
            close_j = float(closes[j]); open_j = float(opens[j])
            body_j = abs(close_j - open_j)
            sweep_level = window_lo if long_sweep else window_hi

            if side == "long":
                if close_j <= sweep_level:
                    i += 1; continue
                if close_j <= open_j:    # need bullish candle
                    i += 1; continue
                if body_j < reclaim_atr * atr_i:
                    i += 1; continue
                entry_price = close_j
                sl = bar_lo - SL_BUFFER_ATR * atr_i
                risk = entry_price - sl
                if risk <= 0: i += 1; continue
                tp = entry_price + TP_RR * risk
            else:
                if close_j >= sweep_level:
                    i += 1; continue
                if close_j >= open_j:
                    i += 1; continue
                if body_j < reclaim_atr * atr_i:
                    i += 1; continue
                entry_price = close_j
                sl = bar_hi + SL_BUFFER_ATR * atr_i
                risk = sl - entry_price
                if risk <= 0: i += 1; continue
                tp = entry_price - TP_RR * risk

            entry_idx = j
            bars_max = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
            end_idx = min(entry_idx + bars_max, n - 1)
            exit_reason = "time_stop"
            exit_price = float(closes[end_idx])
            exit_idx = end_idx
            for k in range(entry_idx + 1, end_idx + 1):
                bh = float(highs[k]); bl = float(lows[k])
                if side == "long":
                    if bl <= sl:
                        exit_reason = "sl"; exit_price = sl; exit_idx = k; break
                    if bh >= tp:
                        exit_reason = "tp"; exit_price = tp; exit_idx = k; break
                else:
                    if bh >= sl:
                        exit_reason = "sl"; exit_price = sl; exit_idx = k; break
                    if bl <= tp:
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
    print("=== liq_sweep_cvd_divergence W/F ===")
    out_dir = HARNESS_ROOT / "storage" / "wf_studies" / "liq_sweep_cvd_divergence"

    class _Engine(WalkForwardEngine):
        def load_candles(self, symbol, tf):
            df = super().load_candles(symbol, tf)
            df.attrs["symbol"] = symbol
            return df

    engine = _Engine(
        study=LiqSweepCvdDivergence(),
        symbols=["BTC", "ETH", "SOL"],
        timeframes=["5m"],
        out_dir=out_dir,
    )
    engine.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Session Sweep + VWAP Reclaim — W/F study (candidate #1, 2026-05-03).

Pattern (LONG; mirror SHORT):
  Step 1 — SESSION SWEEP: in current session window (Asia 00–07, EU 07–15,
           US 15–22 UTC), bar i's low penetrates PRIOR session's low by
           >= sweep_atr × ATR(14).
  Step 2 — VWAP RECLAIM: within `reclaim_window` bars of the sweep, price
           closes back above the session-anchored VWAP.
  Step 3 — CONFIRM: reclaim candle is bullish (close > open) and closes
           above the VWAP by >= reclaim_dist_atr × ATR(14).
  Entry:   close of confirm candle (maker only — variant C).
  SL:      sweep low − 0.3 × ATR(14).
  TP:      RR=1.5 from entry.
  Time:    30 min hard stop.

Symbols: BTC, ETH, SOL.
TF: 5m.

Cells:
  sweep_atr ∈ {0.2, 0.3, 0.5}
  reclaim_window ∈ {3, 6}
  reclaim_dist_atr ∈ {0.0, 0.1, 0.2}
  → 18 cells × 3 syms × 3 fee variants

Output: storage/wf_studies/session_sweep_vwap_reclaim/{walkforward.json,report.md}
"""
from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make scripts/ importable so we can import wf_harness
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
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float).shift(1)
    tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def session_label(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 15:
        return "eu"
    if 15 <= h < 22:
        return "us"
    return "offhrs"


def session_anchored_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at session boundaries (00, 07, 15 UTC)."""
    typ = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float).clip(lower=1e-9)
    pv = typ * vol
    sess = df.index.to_series().apply(session_label)
    sess_change = (sess != sess.shift()).cumsum()
    cum_pv = pv.groupby(sess_change).cumsum()
    cum_v = vol.groupby(sess_change).cumsum()
    return (cum_pv / cum_v).rename("vwap_session")


def prior_session_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """For each bar, return the prior-session high and low (ffilled until next session
    flip). NaN until at least one full session has elapsed."""
    sess = df.index.to_series().apply(session_label)
    sess_change = (sess != sess.shift()).cumsum()
    grouped = df.groupby(sess_change)
    sess_hi = grouped["high"].transform("max")
    sess_lo = grouped["low"].transform("min")
    # Prior session = previous group; we want for current group the extremes of the previous
    # Build a mapping group_id -> (hi, lo)
    g_hi = grouped["high"].max()
    g_lo = grouped["low"].min()
    prior_hi = sess_change.map(lambda gid: g_hi.shift(1).get(gid, np.nan)).astype(float)
    prior_lo = sess_change.map(lambda gid: g_lo.shift(1).get(gid, np.nan)).astype(float)
    out = pd.DataFrame({"prior_session_high": prior_hi.values,
                        "prior_session_low": prior_lo.values}, index=df.index)
    return out


class SessionSweepVwapReclaim(Strategy):
    name = "session_sweep_vwap_reclaim"

    def param_grid(self) -> Iterator[Dict[str, Any]]:
        for sweep_atr, reclaim_window, reclaim_dist_atr in product(
            [0.2, 0.3, 0.5],
            [3, 6],
            [0.0, 0.1, 0.2],
        ):
            yield {
                "sweep_atr": sweep_atr,
                "reclaim_window": reclaim_window,
                "reclaim_dist_atr": reclaim_dist_atr,
            }

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        if df.empty or len(df) < ATR_PERIOD + 30:
            return []

        symbol = df.attrs.get("symbol", "?")
        sweep_atr = float(params["sweep_atr"])
        reclaim_window = int(params["reclaim_window"])
        reclaim_dist_atr = float(params["reclaim_dist_atr"])
        tf_min = 5  # 5m TF

        atr = add_atr(df, ATR_PERIOD)
        vwap = session_anchored_vwap(df)
        prior = prior_session_extremes(df)
        prior_hi = prior["prior_session_high"]
        prior_lo = prior["prior_session_low"]

        trades: List[Trade] = []
        n = len(df)
        # Need warmup
        i = ATR_PERIOD + 5
        while i < n - 1:
            atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
            if atr_i <= 0:
                i += 1; continue
            phi = float(prior_hi.iloc[i]) if not pd.isna(prior_hi.iloc[i]) else None
            plo = float(prior_lo.iloc[i]) if not pd.isna(prior_lo.iloc[i]) else None
            bar_lo = float(df["low"].iloc[i])
            bar_hi = float(df["high"].iloc[i])

            # LONG sweep: bar i's low penetrates prior_session_low by >= sweep_atr × ATR
            long_sweep = (plo is not None) and (plo - bar_lo >= sweep_atr * atr_i)
            short_sweep = (phi is not None) and (bar_hi - phi >= sweep_atr * atr_i)
            if not (long_sweep or short_sweep):
                i += 1; continue

            # Look forward up to reclaim_window bars for VWAP reclaim + bullish/bearish confirm
            sweep_low = bar_lo
            sweep_high = bar_hi
            entry_idx = None
            side = "long" if long_sweep else "short"
            for k in range(1, reclaim_window + 1):
                if i + k >= n: break
                bar = df.iloc[i + k]
                vwap_k = float(vwap.iloc[i + k]) if not pd.isna(vwap.iloc[i + k]) else None
                if vwap_k is None: continue
                close_k = float(bar["close"])
                open_k = float(bar["open"])
                if side == "long":
                    if close_k > vwap_k and close_k > open_k:
                        if (close_k - vwap_k) >= reclaim_dist_atr * atr_i:
                            entry_idx = i + k
                            break
                else:
                    if close_k < vwap_k and close_k < open_k:
                        if (vwap_k - close_k) >= reclaim_dist_atr * atr_i:
                            entry_idx = i + k
                            break
            if entry_idx is None:
                # No reclaim within window — advance past sweep
                i += 1; continue

            entry_price = float(df["close"].iloc[entry_idx])
            if side == "long":
                sl = sweep_low - SL_BUFFER_ATR * atr_i
                risk = entry_price - sl
                if risk <= 0: i = entry_idx + 1; continue
                tp = entry_price + TP_RR * risk
            else:
                sl = sweep_high + SL_BUFFER_ATR * atr_i
                risk = sl - entry_price
                if risk <= 0: i = entry_idx + 1; continue
                tp = entry_price - TP_RR * risk

            # Walk forward for outcome
            bars_max = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
            end_idx = min(entry_idx + bars_max, n - 1)
            exit_reason = "time_stop"
            exit_price = float(df["close"].iloc[end_idx])
            exit_idx = end_idx
            for j in range(entry_idx + 1, end_idx + 1):
                bh = float(df["high"].iloc[j]); bl = float(df["low"].iloc[j])
                if side == "long":
                    if bl <= sl:
                        exit_reason = "sl"; exit_price = sl; exit_idx = j; break
                    if bh >= tp:
                        exit_reason = "tp"; exit_price = tp; exit_idx = j; break
                else:
                    if bh >= sl:
                        exit_reason = "sl"; exit_price = sl; exit_idx = j; break
                    if bl <= tp:
                        exit_reason = "tp"; exit_price = tp; exit_idx = j; break

            holding_sec = max(1, (exit_idx - entry_idx) * tf_min * 60)
            trades.append(Trade(
                symbol=symbol, side=side,
                entry_price=entry_price, exit_price=exit_price,
                notional_usd=NOTIONAL, holding_sec=holding_sec,
                entry_ts=df.index[entry_idx], exit_ts=df.index[exit_idx],
                exit_reason=exit_reason,
            ))
            # Skip past this trade window
            i = exit_idx + 1
        return trades


def main():
    print("=== session_sweep_vwap_reclaim W/F ===")
    out_dir = HARNESS_ROOT / "storage" / "wf_studies" / "session_sweep_vwap_reclaim"

    # Inject symbol into df.attrs via a wrapper engine since base harness doesn't
    class _Engine(WalkForwardEngine):
        def load_candles(self, symbol, tf):
            df = super().load_candles(symbol, tf)
            df.attrs["symbol"] = symbol
            return df

    engine = _Engine(
        study=SessionSweepVwapReclaim(),
        symbols=["BTC", "ETH", "SOL"],
        timeframes=["5m"],
        out_dir=out_dir,
    )
    engine.run()


if __name__ == "__main__":
    main()

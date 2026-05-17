#!/usr/bin/env python3
"""Exit W/F #2 — vwap_touch_exit vs baseline (scalper_vwap_mr scanner).

Same-entry comparison: re-detects scalper_vwap_mr signals, evaluates each
under each exit policy.

Policies:
  - vwap_mr_baseline           — current live (TP=VWAP touch, SL=1×ATR, time=28min)
                                 Note: baseline ALREADY uses VWAP touch — so the candidate
                                 here tests TIGHTER abandonment + opposite-structure exit.
  - vwap_touch_8bar_no_trail   — exit at VWAP touch OR ≤ 8 bars OR opposite structure break
                                 (defined as: bar closes BACK through entry on the wrong side)
                                 NO TRAIL.
  - vwap_touch_5bar            — same but max 5 bars (sensitivity)
  - vwap_touch_oppstruct_only  — exit at VWAP touch OR opposite structure break,
                                 NO time stop (lets winners run if structure intact)

Symbols: BTC, ETH (scalper-offer eligible).
TF: 5m.
Variant: C_maker_scalper.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exit_policy_harness import (  # noqa: E402
    Signal, ExitOutcome, ExitPolicy, run_exit_wf, add_atr,
)

# ── VWAP MR detector params (use a representative cell from prior W/F) ───────
TF_MIN = 5
LOOKBACK = 50            # stdev/vwap rolling
K = 1.5                  # band width
ATR_PCT_LOOKBACK = 100
ATR_PCT_THR = 0.50       # only trade when ATR_pct rank <= 0.50 (range regime)
ATR_PERIOD = 14
SL_ATR_MULT = 1.0

EVAL_WINDOW_BARS = 60    # 5h — covers any policy max_hold

NOTIONAL = 1000.0


def add_vwap_bands(df: pd.DataFrame, lookback: int):
    typ = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float).clip(lower=1e-9)
    pv = typ * vol
    vwap = pv.rolling(lookback, min_periods=lookback).sum() / vol.rolling(lookback, min_periods=lookback).sum()
    stdev = df["close"].astype(float).rolling(lookback, min_periods=lookback).std(ddof=0)
    return vwap, stdev


def generate_vwap_mr_signals(df: pd.DataFrame, symbol: str) -> List[Signal]:
    """Mean-reversion entry: close BEYOND VWAP±k*stdev + reversal candle, in range regime."""
    if df.empty or len(df) < max(LOOKBACK, ATR_PCT_LOOKBACK) + ATR_PERIOD + 5:
        return []
    vwap, stdev = add_vwap_bands(df, LOOKBACK)
    atr = add_atr(df, ATR_PERIOD)
    atr_pct = atr.rolling(ATR_PCT_LOOKBACK, min_periods=ATR_PCT_LOOKBACK).rank(pct=True)

    closes = df["close"].astype(float).values
    opens = df["open"].astype(float).values
    sigs: List[Signal] = []
    n = len(df)
    for i in range(max(LOOKBACK, ATR_PCT_LOOKBACK) + 1, n - 1):
        atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
        atr_p = float(atr_pct.iloc[i]) if not pd.isna(atr_pct.iloc[i]) else 1.0
        v = float(vwap.iloc[i]) if not pd.isna(vwap.iloc[i]) else None
        s = float(stdev.iloc[i]) if not pd.isna(stdev.iloc[i]) else None
        if atr_i <= 0 or v is None or s is None:
            continue
        if atr_p > ATR_PCT_THR:    # too volatile / trending
            continue
        upper = v + K * s; lower = v - K * s
        c = float(closes[i]); o = float(opens[i])
        c_prev = float(closes[i - 1])

        # LONG: close < lower band AND bullish reversal (close > open AND close > prev close)
        if c < lower and c > o and c > c_prev:
            sigs.append(Signal(
                symbol=symbol, entry_idx=i, entry_ts=df.index[i],
                side="long", entry_price=c,
                sweep_extreme=c - SL_ATR_MULT * atr_i,    # SL anchor = entry - 1*ATR
                atr_at_entry=atr_i,
                extra={"vwap_at_signal": v, "stdev_at_signal": s,
                       "atr_pct_at_signal": atr_p, "k": K, "lookback": LOOKBACK},
            ))
        # SHORT: close > upper band AND bearish reversal
        elif c > upper and c < o and c < c_prev:
            sigs.append(Signal(
                symbol=symbol, entry_idx=i, entry_ts=df.index[i],
                side="short", entry_price=c,
                sweep_extreme=c + SL_ATR_MULT * atr_i,
                atr_at_entry=atr_i,
                extra={"vwap_at_signal": v, "stdev_at_signal": s,
                       "atr_pct_at_signal": atr_p, "k": K, "lookback": LOOKBACK},
            ))
    return sigs


def _outcome(df, sig, entry_idx, exit_idx, exit_price, exit_reason, tf_min):
    holding_bars = exit_idx - entry_idx
    holding_sec = max(1, holding_bars * tf_min * 60)
    if sig.side == "long":
        realized = (exit_price - sig.entry_price) / sig.entry_price
    else:
        realized = (sig.entry_price - exit_price) / sig.entry_price
    return ExitOutcome(
        exit_idx=exit_idx, exit_ts=df.index[exit_idx],
        exit_price=exit_price, exit_reason=exit_reason,
        holding_bars=holding_bars, holding_sec=holding_sec,
        mfe_pct=0.0, mae_pct=0.0, realized_pct=realized,
    )


def _walk_vwap_or_sl(df, sig, max_bars: int, allow_oppstruct: bool):
    """Walk forward; exit on VWAP touch, SL hit, or time stop (max_bars).
    If allow_oppstruct: also exit on opposite-structure break (close back
    through entry on wrong side)."""
    sl = sig.sweep_extreme
    vwap, _ = add_vwap_bands(df, LOOKBACK)    # df-wide (cached by caller via attrs)
    end_idx = min(sig.entry_idx + max_bars, len(df) - 1)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    for j in range(sig.entry_idx + 1, end_idx + 1):
        bh = float(highs[j]); bl = float(lows[j]); bc = float(closes[j])
        v = float(vwap.iloc[j]) if not pd.isna(vwap.iloc[j]) else None
        if sig.side == "long":
            if bl <= sl: return j, sl, "sl"
            if v is not None and bh >= v: return j, v, "vwap_touch"
            if allow_oppstruct and bc < sig.entry_price:
                # Wait — for LONG, "opposite structure break" means a CLOSE
                # back below the entry price after we've already moved up.
                # We require at least 2 bars elapsed.
                if j - sig.entry_idx >= 2:
                    return j, bc, "opp_struct"
        else:
            if bh >= sl: return j, sl, "sl"
            if v is not None and bl <= v: return j, v, "vwap_touch"
            if allow_oppstruct and bc > sig.entry_price:
                if j - sig.entry_idx >= 2:
                    return j, bc, "opp_struct"
    return end_idx, float(closes[end_idx]), "time_stop"


class VwapMrBaseline(ExitPolicy):
    """Live: TP=VWAP, SL=1×ATR, time=28min (~6 bars). NO opp-struct."""
    name = "vwap_mr_baseline"
    is_baseline = True

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        max_bars = max(1, int(math.ceil(28 / tf_min)))
        idx, px, reason = _walk_vwap_or_sl(df, sig, max_bars, allow_oppstruct=False)
        return _outcome(df, sig, sig.entry_idx, idx, px, reason, tf_min)


class VwapTouch8barNoTrail(ExitPolicy):
    name = "vwap_touch_8bar_oppstruct"
    is_baseline = False
    MAX_BARS = 8

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        idx, px, reason = _walk_vwap_or_sl(df, sig, self.MAX_BARS, allow_oppstruct=True)
        return _outcome(df, sig, sig.entry_idx, idx, px, reason, tf_min)


class VwapTouch5bar(ExitPolicy):
    name = "vwap_touch_5bar"
    is_baseline = False
    MAX_BARS = 5

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        idx, px, reason = _walk_vwap_or_sl(df, sig, self.MAX_BARS, allow_oppstruct=False)
        return _outcome(df, sig, sig.entry_idx, idx, px, reason, tf_min)


class VwapTouchOppStructOnly(ExitPolicy):
    """Lets trades run as long as structure holds (no time stop, ≤eval_window cap)."""
    name = "vwap_touch_oppstruct_no_timecap"
    is_baseline = False

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        idx, px, reason = _walk_vwap_or_sl(df, sig, eval_window_bars, allow_oppstruct=True)
        return _outcome(df, sig, sig.entry_idx, idx, px, reason, tf_min)


def main():
    out_dir = ROOT / "storage" / "wf_studies" / "exit_vwap_touch"
    run_exit_wf(
        scanner="scalper_vwap_mr",
        symbols=["BTC", "ETH"],
        tf="5m", tf_min=TF_MIN,
        signal_generator=generate_vwap_mr_signals,
        policies=[
            VwapMrBaseline(),
            VwapTouch8barNoTrail(),
            VwapTouch5bar(),
            VwapTouchOppStructOnly(),
        ],
        out_dir=out_dir,
        eval_window_bars=EVAL_WINDOW_BARS,
        notional=NOTIONAL,
        extra_notes=[
            "- Baseline = current scalper_vwap_mr (TP=VWAP touch, SL=1×ATR, time=28min, no opp-struct).",
            "- Candidate policies vary the time cap and whether opposite-structure break is allowed as exit.",
            "- BTC + ETH only (scalper-offer fee waiver eligibility).",
            "- Detector cell fixed at K=1.5 / lookback=50 / atr_pct_thr=0.50 (representative).",
        ],
    )


if __name__ == "__main__":
    main()

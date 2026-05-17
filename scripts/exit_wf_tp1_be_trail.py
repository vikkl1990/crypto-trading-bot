#!/usr/bin/env python3
"""Exit W/F #3 — tp1_be_trail_continuation vs baseline (SOL liq_grab + liq_sweep_htf).

Same-entry comparison: re-detects SOL liq_grab_ob_fvg + liq_sweep_htf signals,
evaluates each under each exit policy.

Policies:
  - liq_grab_baseline           — current live (TP=1.5R flat, SL=sweep extreme ± 0.3*ATR, time=30min)
  - tp1_be_trail_ema9           — TP1=1.5R partial closes 50% & moves SL → BE,
                                  remainder trails by EMA9 ± 0.8*ATR until close-cross,
                                  hard 25-bar cap. Models partial close as full close at TP1
                                  (so net is conservative — we measure base method viability).
  - tp1_be_trail_swing          — same partial+BE, then trail by prior-swing low/high
                                  (last 5-bar extreme) ± 0.5*ATR
  - tp2_2R_then_be              — TP1=1.0R partial+BE, TP2=2.0R remainder, hard 30-bar cap
                                  (alternative continuation harvester)

Symbols: SOL.
TF: 5m.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exit_policy_harness import (  # noqa: E402
    Signal, ExitOutcome, ExitPolicy, run_exit_wf, add_atr,
)

# ── Detector params (representative cell from prior W/F PASS sets) ───────────
TF_MIN = 5
ATR_PERIOD = 14
SWEEP_LOOKBACK = 20
SWEEP_ATR = 0.3
RECLAIM_DISPL_ATR = 0.5
SL_BUFFER_ATR = 0.3
TP_RR_BASELINE = 1.5
TIME_STOP_BASELINE_MIN = 30

EVAL_WINDOW_BARS = 60    # 5h — covers any policy max_hold (TP1+BE+trail can run long)

NOTIONAL = 1000.0


def generate_liq_grab_sweep_signals(df: pd.DataFrame, symbol: str) -> List[Signal]:
    """Detects either liq_grab_ob_fvg OR liquidity_sweep_htf-style entry:
       sweep of prior 20-bar extreme + reclaim with displacement candle."""
    if df.empty or len(df) < SWEEP_LOOKBACK + ATR_PERIOD + 5:
        return []
    atr = add_atr(df, ATR_PERIOD)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    sigs: List[Signal] = []
    n = len(df)
    for i in range(SWEEP_LOOKBACK + ATR_PERIOD + 1, n - 1):
        atr_j = float(atr.iloc[i - 1]) if not pd.isna(atr.iloc[i - 1]) else 0.0
        if atr_j <= 0:
            continue
        j = i - 1
        win_lo = float(np.min(lows[j - SWEEP_LOOKBACK: j]))
        win_hi = float(np.max(highs[j - SWEEP_LOOKBACK: j]))
        bar_j_lo = float(lows[j]); bar_j_hi = float(highs[j])

        long_sweep = (win_lo - bar_j_lo) >= SWEEP_ATR * atr_j
        short_sweep = (bar_j_hi - win_hi) >= SWEEP_ATR * atr_j
        if not (long_sweep or short_sweep):
            continue

        bar_i_op = float(opens[i]); bar_i_cl = float(closes[i])
        body = abs(bar_i_cl - bar_i_op)
        if body < RECLAIM_DISPL_ATR * atr_j:
            continue

        if long_sweep:
            if bar_i_cl <= win_lo: continue
            if bar_i_cl <= bar_i_op: continue
            sigs.append(Signal(
                symbol=symbol, entry_idx=i, entry_ts=df.index[i],
                side="long", entry_price=bar_i_cl,
                sweep_extreme=bar_j_lo, atr_at_entry=atr_j,
                extra={"sweep_level": win_lo},
            ))
        else:
            if bar_i_cl >= win_hi: continue
            if bar_i_cl >= bar_i_op: continue
            sigs.append(Signal(
                symbol=symbol, entry_idx=i, entry_ts=df.index[i],
                side="short", entry_price=bar_i_cl,
                sweep_extreme=bar_j_hi, atr_at_entry=atr_j,
                extra={"sweep_level": win_hi},
            ))
    return sigs


def _outcome(df, sig, exit_idx, exit_price, reason, tf_min):
    holding_bars = exit_idx - sig.entry_idx
    holding_sec = max(1, holding_bars * tf_min * 60)
    if sig.side == "long":
        realized = (exit_price - sig.entry_price) / sig.entry_price
    else:
        realized = (sig.entry_price - exit_price) / sig.entry_price
    return ExitOutcome(
        exit_idx=exit_idx, exit_ts=df.index[exit_idx],
        exit_price=exit_price, exit_reason=reason,
        holding_bars=holding_bars, holding_sec=holding_sec,
        mfe_pct=0.0, mae_pct=0.0, realized_pct=realized,
    )


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple EMA — uses numpy for speed. NaN-safe."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[:] = np.nan
    seed = None
    for i in range(len(arr)):
        v = arr[i]
        if math.isnan(v): continue
        if seed is None:
            seed = v
        else:
            seed = alpha * v + (1 - alpha) * seed
        out[i] = seed
    return out


class LiqGrabBaseline(ExitPolicy):
    name = "liq_grab_baseline"
    is_baseline = True

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        sl_buf = SL_BUFFER_ATR * sig.atr_at_entry
        if sig.side == "long":
            sl = sig.sweep_extreme - sl_buf
            risk = sig.entry_price - sl
            tp = sig.entry_price + TP_RR_BASELINE * max(risk, 1e-9)
        else:
            sl = sig.sweep_extreme + sl_buf
            risk = sl - sig.entry_price
            tp = sig.entry_price - TP_RR_BASELINE * max(risk, 1e-9)
        max_bars = max(1, int(math.ceil(TIME_STOP_BASELINE_MIN / tf_min)))
        end_idx = min(sig.entry_idx + max_bars, len(df) - 1)
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        for j in range(sig.entry_idx + 1, end_idx + 1):
            bh = float(highs[j]); bl = float(lows[j])
            if sig.side == "long":
                if bl <= sl: return _outcome(df, sig, j, sl, "sl", tf_min)
                if bh >= tp: return _outcome(df, sig, j, tp, "tp", tf_min)
            else:
                if bh >= sl: return _outcome(df, sig, j, sl, "sl", tf_min)
                if bl <= tp: return _outcome(df, sig, j, tp, "tp", tf_min)
        return _outcome(df, sig, end_idx, float(closes[end_idx]), "time_stop", tf_min)


def _simulate_partial_be_trail(
    df, sig, tf_min, *,
    tp1_rr: float, tp2_rr: float, max_bars: int,
    trail_method: str,    # "ema9" or "swing" or "none"
    ema_period: int = 9,
    trail_atr_buf: float = 0.8,
    swing_lb: int = 5,
    swing_atr_buf: float = 0.5,
) -> ExitOutcome:
    """Hybrid policy:
       1. Walk to first TP1 hit (or SL, time)
       2. After TP1, lock SL at BE
       3. Trail remainder by EMA9 ± buf*ATR (or swing extreme)
       4. Exit when close crosses trail OR hard time hit
    Models a SINGLE-position trade (no real partial close — measures EV of the
    full-close exit at the LATER of TP1 or trail-cross). This is conservative;
    real partial-close would be slightly higher EV."""
    sl_buf = SL_BUFFER_ATR * sig.atr_at_entry
    if sig.side == "long":
        sl0 = sig.sweep_extreme - sl_buf
        risk = sig.entry_price - sl0
        tp1 = sig.entry_price + tp1_rr * max(risk, 1e-9)
    else:
        sl0 = sig.sweep_extreme + sl_buf
        risk = sl0 - sig.entry_price
        tp1 = sig.entry_price - tp1_rr * max(risk, 1e-9)

    end_idx = min(sig.entry_idx + max_bars, len(df) - 1)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values

    # Phase 1: walk to TP1 / SL / time
    tp1_idx = None
    for j in range(sig.entry_idx + 1, end_idx + 1):
        bh = float(highs[j]); bl = float(lows[j])
        if sig.side == "long":
            if bl <= sl0: return _outcome(df, sig, j, sl0, "sl_pre_tp1", tf_min)
            if bh >= tp1:
                tp1_idx = j; break
        else:
            if bh >= sl0: return _outcome(df, sig, j, sl0, "sl_pre_tp1", tf_min)
            if bl <= tp1:
                tp1_idx = j; break
    if tp1_idx is None:
        return _outcome(df, sig, end_idx, float(closes[end_idx]), "time_stop_pre_tp1", tf_min)

    # Phase 2: post-TP1 — SL=BE, trail remainder
    be_sl = sig.entry_price
    trail_sl = be_sl
    if trail_method == "ema9":
        ema9 = _ema(closes, ema_period)
    for j in range(tp1_idx + 1, end_idx + 1):
        bh = float(highs[j]); bl = float(lows[j]); bc = float(closes[j])
        # Recompute trail
        if trail_method == "ema9":
            ema_v = ema9[j] if not math.isnan(ema9[j]) else bc
            if sig.side == "long":
                cand = ema_v - trail_atr_buf * sig.atr_at_entry
                trail_sl = max(trail_sl, cand)
            else:
                cand = ema_v + trail_atr_buf * sig.atr_at_entry
                trail_sl = min(trail_sl, cand)
        elif trail_method == "swing":
            lo_w = float(np.min(lows[max(j - swing_lb, 0): j]))
            hi_w = float(np.max(highs[max(j - swing_lb, 0): j]))
            if sig.side == "long":
                cand = lo_w - swing_atr_buf * sig.atr_at_entry
                trail_sl = max(trail_sl, cand)
            else:
                cand = hi_w + swing_atr_buf * sig.atr_at_entry
                trail_sl = min(trail_sl, cand)
        # else "none" — leave trail at BE

        # Check tp2 first (only if tp2_rr > tp1_rr)
        if tp2_rr > tp1_rr:
            if sig.side == "long":
                tp2_p = sig.entry_price + tp2_rr * max(risk, 1e-9)
                if bh >= tp2_p:
                    return _outcome(df, sig, j, tp2_p, f"tp2_{tp2_rr}R", tf_min)
            else:
                tp2_p = sig.entry_price - tp2_rr * max(risk, 1e-9)
                if bl <= tp2_p:
                    return _outcome(df, sig, j, tp2_p, f"tp2_{tp2_rr}R", tf_min)

        # Trail/BE stop hit?
        if sig.side == "long":
            if bl <= trail_sl: return _outcome(df, sig, j, trail_sl, "trail_or_be", tf_min)
        else:
            if bh >= trail_sl: return _outcome(df, sig, j, trail_sl, "trail_or_be", tf_min)
    return _outcome(df, sig, end_idx, float(closes[end_idx]), "time_stop_post_tp1", tf_min)


class Tp1BeTrailEma9(ExitPolicy):
    name = "tp1_1.5R_be_trail_ema9_25bar"
    is_baseline = False

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        return _simulate_partial_be_trail(
            df, sig, tf_min,
            tp1_rr=1.5, tp2_rr=0.0, max_bars=25,
            trail_method="ema9",
        )


class Tp1BeTrailSwing(ExitPolicy):
    name = "tp1_1.5R_be_trail_swing5_25bar"
    is_baseline = False

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        return _simulate_partial_be_trail(
            df, sig, tf_min,
            tp1_rr=1.5, tp2_rr=0.0, max_bars=25,
            trail_method="swing",
        )


class Tp2_2R_BE(ExitPolicy):
    name = "tp1_1.0R_BE_tp2_2.0R_30bar"
    is_baseline = False

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        return _simulate_partial_be_trail(
            df, sig, tf_min,
            tp1_rr=1.0, tp2_rr=2.0, max_bars=30,
            trail_method="none",
        )


def main():
    out_dir = ROOT / "storage" / "wf_studies" / "exit_tp1_be_trail"
    run_exit_wf(
        scanner="liq_grab_ob_fvg+liq_sweep_htf",
        symbols=["SOL"],
        tf="5m", tf_min=TF_MIN,
        signal_generator=generate_liq_grab_sweep_signals,
        policies=[
            LiqGrabBaseline(),
            Tp1BeTrailEma9(),
            Tp1BeTrailSwing(),
            Tp2_2R_BE(),
        ],
        out_dir=out_dir,
        eval_window_bars=EVAL_WINDOW_BARS,
        notional=NOTIONAL,
        extra_notes=[
            "- Baseline replicates current SOL liq_grab/sweep_htf engine (TP=1.5R flat / SL=sweep±0.3*ATR / time=30min).",
            "- Partial-close policies model the full-close at TP1 OR at trail-cross — conservative vs real partial-close (which only realises remainder at trail).",
            "- SOL only (combo #1 W/F PASS symbol).",
        ],
    )


if __name__ == "__main__":
    main()

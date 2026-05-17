#!/usr/bin/env python3
"""Exit W/F #1 — absorption_fast_exit vs baseline (absorption_bubble scanner).

Same-entry comparison: re-detects absorption_bubble signals on cached candles,
then evaluates each signal under each exit policy.

Policies:
  - absorption_baseline           — current live params (TP=1.5R, SL=sweep±0.3*ATR, time=30min)
  - absorption_fast_exit          — TP1=1.0R partial, exit on volume fade, max 6 bars
                                    (we model "vol fade" as: volume on bar drops below
                                     0.7 × 20-bar mean — flow stopped confirming)
  - absorption_fast_exit_8bar     — same as above but max=8 bars (sensitivity)

Symbols: ETH (W/F-validated; only PASSing absorption symbol).
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

# ── Locked entry detector params (match live absorption_bubble engine) ───────
LOOKBACK_N = 30
SWEEP_ATR = 0.3
VOL_MULT = 1.5
DISPL_ATR = 0.65
BODY_RATIO_MAX = 0.40
WICK_RATIO_MIN = 0.50
SL_BUFFER_ATR = 0.3
ATR_PERIOD = 14
VOL_PERIOD = 20

TF_MIN = 5
EVAL_WINDOW_BARS = 24    # 24 × 5m = 2 hours — covers any policy max_hold

NOTIONAL = 1000.0


# ─────────────────────────────────────────────────────────────────────
# Signal generator — re-uses live absorption_bubble detection logic
# (re-implemented self-contained to avoid coupling to the paper engine file)
# ─────────────────────────────────────────────────────────────────────
def generate_absorption_signals(df: pd.DataFrame, symbol: str) -> List[Signal]:
    if df.empty or len(df) < LOOKBACK_N + ATR_PERIOD + 5:
        return []
    atr = add_atr(df, ATR_PERIOD)
    vol_mean = df["volume"].astype(float).rolling(VOL_PERIOD, min_periods=VOL_PERIOD).mean()
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    vols = df["volume"].astype(float).values
    sigs: List[Signal] = []
    n = len(df)
    for i in range(LOOKBACK_N + ATR_PERIOD + 1, n - 1):
        atr_j = float(atr.iloc[i - 1]) if not pd.isna(atr.iloc[i - 1]) else 0.0
        if atr_j <= 0:
            continue
        # Step 1: bar j (i-1) sweeps prior 30-bar low/high by >= 0.3 * ATR
        j = i - 1
        if j < LOOKBACK_N:
            continue
        roll_lo = float(np.min(lows[j - LOOKBACK_N: j]))
        roll_hi = float(np.max(highs[j - LOOKBACK_N: j]))
        bar_j_lo = float(lows[j]); bar_j_hi = float(highs[j])
        bar_j_op = float(opens[j]); bar_j_cl = float(closes[j])

        long_sweep = (roll_lo - bar_j_lo) >= SWEEP_ATR * atr_j
        short_sweep = (bar_j_hi - roll_hi) >= SWEEP_ATR * atr_j
        if not (long_sweep or short_sweep):
            continue

        # Step 2: absorption properties
        bar_j_range = bar_j_hi - bar_j_lo
        if bar_j_range <= 0:
            continue
        body = abs(bar_j_cl - bar_j_op)
        body_ratio = body / bar_j_range
        if body_ratio > BODY_RATIO_MAX:
            continue
        if long_sweep:
            wick = min(bar_j_op, bar_j_cl) - bar_j_lo
            wick_ratio = wick / bar_j_range
            ref_extreme = bar_j_lo
            sweep_level = roll_lo
            side = "long"
        else:
            wick = bar_j_hi - max(bar_j_op, bar_j_cl)
            wick_ratio = wick / bar_j_range
            ref_extreme = bar_j_hi
            sweep_level = roll_hi
            side = "short"
        if wick_ratio < WICK_RATIO_MIN:
            continue
        vm_j = float(vol_mean.iloc[j]) if not pd.isna(vol_mean.iloc[j]) else 0.0
        if vm_j <= 0:
            continue
        rel_vol = float(vols[j]) / vm_j
        if rel_vol < VOL_MULT:
            continue

        # Step 3: bar i reclaims with displacement
        bar_i_op = float(opens[i]); bar_i_cl = float(closes[i])
        bar_i_body = abs(bar_i_cl - bar_i_op)
        if bar_i_body < DISPL_ATR * atr_j:
            continue
        if side == "long":
            if bar_i_cl <= sweep_level: continue
            if bar_i_cl <= bar_i_op: continue
        else:
            if bar_i_cl >= sweep_level: continue
            if bar_i_cl >= bar_i_op: continue

        sigs.append(Signal(
            symbol=symbol, entry_idx=i, entry_ts=df.index[i],
            side=side, entry_price=bar_i_cl,
            sweep_extreme=ref_extreme, atr_at_entry=atr_j,
            extra={"sweep_level": sweep_level, "rel_vol_at_signal": rel_vol},
        ))
    return sigs


# ─────────────────────────────────────────────────────────────────────
# Exit policies
# ─────────────────────────────────────────────────────────────────────
def _walk_to_first_hit(df, entry_idx, side, sl, tp, max_bars):
    """Standard SL/TP/time walker. Returns (exit_idx, exit_price, exit_reason)."""
    end_idx = min(entry_idx + max_bars, len(df) - 1)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    for j in range(entry_idx + 1, end_idx + 1):
        bh = float(highs[j]); bl = float(lows[j])
        if side == "long":
            if bl <= sl: return j, sl, "sl"
            if bh >= tp: return j, tp, "tp"
        else:
            if bh >= sl: return j, sl, "sl"
            if bl <= tp: return j, tp, "tp"
    return end_idx, float(closes[end_idx]), "time_stop"


def _outcome(df, sig, entry_idx, exit_idx, exit_price, exit_reason, tf_min):
    """Build ExitOutcome from raw exit info."""
    holding_bars = exit_idx - entry_idx
    holding_sec = max(1, holding_bars * tf_min * 60)
    if sig.side == "long":
        realized_pct = (exit_price - sig.entry_price) / sig.entry_price
    else:
        realized_pct = (sig.entry_price - exit_price) / sig.entry_price
    return ExitOutcome(
        exit_idx=exit_idx, exit_ts=df.index[exit_idx],
        exit_price=exit_price, exit_reason=exit_reason,
        holding_bars=holding_bars, holding_sec=holding_sec,
        mfe_pct=0.0, mae_pct=0.0,    # filled by harness via compute_mfe_mae
        realized_pct=realized_pct,
    )


class AbsorptionBaseline(ExitPolicy):
    name = "absorption_baseline"
    is_baseline = True
    # current live: TP=1.5R, SL=sweep±0.3*ATR, time=30min (= 6 bars)

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        sl_buf = SL_BUFFER_ATR * sig.atr_at_entry
        if sig.side == "long":
            sl = sig.sweep_extreme - sl_buf
            risk = sig.entry_price - sl
            tp = sig.entry_price + 1.5 * max(risk, 1e-9)
        else:
            sl = sig.sweep_extreme + sl_buf
            risk = sl - sig.entry_price
            tp = sig.entry_price - 1.5 * max(risk, 1e-9)
        max_bars = max(1, int(math.ceil(30 / tf_min)))
        exit_idx, exit_price, reason = _walk_to_first_hit(df, sig.entry_idx, sig.side, sl, tp, max_bars)
        return _outcome(df, sig, sig.entry_idx, exit_idx, exit_price, reason, tf_min)


class AbsorptionFastExit(ExitPolicy):
    """TP1=1.0R, vol-fade exit, max 6 bars. Tighter target + earlier abandonment."""
    name = "absorption_fast_exit_6bar"
    is_baseline = False
    MAX_BARS = 6
    TP_RR = 1.0
    VOL_FADE_RATIO = 0.7    # exit if bar volume < 0.7 × 20-bar mean

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        sl_buf = SL_BUFFER_ATR * sig.atr_at_entry
        if sig.side == "long":
            sl = sig.sweep_extreme - sl_buf
            risk = sig.entry_price - sl
            tp = sig.entry_price + self.TP_RR * max(risk, 1e-9)
        else:
            sl = sig.sweep_extreme + sl_buf
            risk = sl - sig.entry_price
            tp = sig.entry_price - self.TP_RR * max(risk, 1e-9)
        end_idx = min(sig.entry_idx + self.MAX_BARS, len(df) - 1)
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        vols = df["volume"].astype(float).values
        vol_mean = df["volume"].astype(float).rolling(VOL_PERIOD, min_periods=VOL_PERIOD).mean().values

        for j in range(sig.entry_idx + 1, end_idx + 1):
            bh = float(highs[j]); bl = float(lows[j])
            # Hard SL / TP1 first
            if sig.side == "long":
                if bl <= sl:
                    return _outcome(df, sig, sig.entry_idx, j, sl, "sl", tf_min)
                if bh >= tp:
                    return _outcome(df, sig, sig.entry_idx, j, tp, "tp1_1.0R", tf_min)
            else:
                if bh >= sl:
                    return _outcome(df, sig, sig.entry_idx, j, sl, "sl", tf_min)
                if bl <= tp:
                    return _outcome(df, sig, sig.entry_idx, j, tp, "tp1_1.0R", tf_min)
            # Vol fade: bar's volume below threshold AND we're past bar+1
            if j > sig.entry_idx + 1:
                vmj = vol_mean[j] if not math.isnan(vol_mean[j]) else 0.0
                vj = vols[j]
                if vmj > 0 and vj < self.VOL_FADE_RATIO * vmj:
                    return _outcome(df, sig, sig.entry_idx, j, float(closes[j]), "vol_fade", tf_min)
        # Max bars hit
        return _outcome(df, sig, sig.entry_idx, end_idx, float(closes[end_idx]),
                        f"max_{self.MAX_BARS}bar", tf_min)


class AbsorptionFastExit8bar(AbsorptionFastExit):
    name = "absorption_fast_exit_8bar"
    MAX_BARS = 8


class AbsorptionTp15FastExit(AbsorptionFastExit):
    """Same vol-fade logic but TP1=1.5R (preserves baseline TP magnitude,
    only adds vol-fade abandonment + tighter time stop)."""
    name = "absorption_tp1.5_volfade_6bar"
    TP_RR = 1.5


def main():
    out_dir = ROOT / "storage" / "wf_studies" / "exit_absorption_fast"
    run_exit_wf(
        scanner="absorption_bubble",
        symbols=["ETH"],
        tf="5m", tf_min=TF_MIN,
        signal_generator=generate_absorption_signals,
        policies=[
            AbsorptionBaseline(),
            AbsorptionFastExit(),
            AbsorptionFastExit8bar(),
            AbsorptionTp15FastExit(),
        ],
        out_dir=out_dir,
        eval_window_bars=EVAL_WINDOW_BARS,
        notional=NOTIONAL,
        extra_notes=[
            "- Baseline replicates live `absorption_bubble_paper_engine.py` exit (TP=1.5R/SL=sweep±0.3*ATR/time=30min).",
            "- Fast-exit policies model 'vol fade' as bar volume < 0.7 × 20-bar mean.",
            "- ETH only (only W/F-PASSing symbol for absorption_bubble per 2026-05-02 batch).",
        ],
    )


if __name__ == "__main__":
    main()

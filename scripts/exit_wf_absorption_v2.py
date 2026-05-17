#!/usr/bin/env python3
"""Exit W/F #1.v2 — absorption_bubble exit policies, expanded with 3 NEW variants.

Same-entry comparison. Same baseline as v1 (absorption_baseline live engine).
We test the existing v1 winner + 3 new ideas (architect-mandated 2026-05-03):

  v1 (already shipped to live):
    - absorption_baseline             [BASELINE]
    - absorption_fast_exit_6bar       [HOLD, +$0.146/trade lift, mfe_cap 41%→46%]

  v2 NEW (this study):
    - absorption_tp_scaling_3stage    — 33% out at TP=1R, 33% at 1.5R, 34% at 2R.
                                         Modeled as weighted-avg exit price across
                                         hits within max_bars; un-hit stages exit
                                         at trail/time stop with the rest.
    - absorption_partial_50_trail05   — 50% out at TP=1R, 50% trails by
                                         max_high(LONG)/min_low(SHORT) − 0.5×ATR.
                                         Models the partial as full-close at TP1
                                         OR at trail-cross.
    - absorption_dynamic_mfe_trail    — Track running MFE peak. Exit when current
                                         price retraces from MFE peak by 33%.
                                         No fixed TP — pure MFE-driven trail.

Same SL = sweep extreme ± 0.3×ATR. Same time cap = 30min (=6 bars on 5m).
ETH only. C_maker_scalper fees.
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
    Signal, ExitOutcome, ExitPolicy, run_exit_wf,
)
from exit_wf_absorption_fast import (  # noqa: E402
    generate_absorption_signals, AbsorptionBaseline, AbsorptionFastExit,
    SL_BUFFER_ATR, VOL_PERIOD,
)

TF_MIN = 5
EVAL_WINDOW_BARS = 24    # 2h — covers any policy max_hold
NOTIONAL = 1000.0


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
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


def _build_sl_anchor(sig):
    sl_buf = SL_BUFFER_ATR * sig.atr_at_entry
    if sig.side == "long":
        return sig.sweep_extreme - sl_buf
    return sig.sweep_extreme + sl_buf


def _R(sig, sl):
    return abs(sig.entry_price - sl)


def _tp_at(sig, sl, rr):
    R = _R(sig, sl)
    if sig.side == "long":
        return sig.entry_price + rr * R
    return sig.entry_price - rr * R


# ─────────────────────────────────────────────────────────────────────
# v2.1 — TP Scaling (3-stage)
# ─────────────────────────────────────────────────────────────────────
class AbsorptionTpScaling3Stage(ExitPolicy):
    """33% out at 1R, 33% at 1.5R, 34% at 2R. Un-hit stages exit at time/trail."""
    name = "absorption_tp_scaling_3stage"
    is_baseline = False
    MAX_BARS = 6

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        sl = _build_sl_anchor(sig)
        tps = [(0.33, _tp_at(sig, sl, 1.0)),
               (0.33, _tp_at(sig, sl, 1.5)),
               (0.34, _tp_at(sig, sl, 2.0))]
        end_idx = min(sig.entry_idx + self.MAX_BARS, len(df) - 1)
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values

        # Track which stages have been hit + at what price
        stage_hit = [False, False, False]
        stage_exit = [None, None, None]
        last_exit_idx = sig.entry_idx
        sl_hit = False

        for j in range(sig.entry_idx + 1, end_idx + 1):
            bh = float(highs[j]); bl = float(lows[j])
            # SL first (worst-case assumption)
            if sig.side == "long":
                if bl <= sl:
                    sl_hit = True; last_exit_idx = j
                    # All un-hit stages exit at SL price
                    for k in range(3):
                        if not stage_hit[k]:
                            stage_exit[k] = sl
                    break
                # TP stages
                for k, (_, tp_p) in enumerate(tps):
                    if not stage_hit[k] and bh >= tp_p:
                        stage_hit[k] = True
                        stage_exit[k] = tp_p
                        last_exit_idx = j
            else:
                if bh >= sl:
                    sl_hit = True; last_exit_idx = j
                    for k in range(3):
                        if not stage_hit[k]:
                            stage_exit[k] = sl
                    break
                for k, (_, tp_p) in enumerate(tps):
                    if not stage_hit[k] and bl <= tp_p:
                        stage_hit[k] = True
                        stage_exit[k] = tp_p
                        last_exit_idx = j
            if all(stage_hit):
                break
        else:
            # Loop completed without break — fill un-hit stages at end_idx close
            for k in range(3):
                if not stage_hit[k]:
                    stage_exit[k] = float(closes[end_idx])
                    last_exit_idx = end_idx

        if sl_hit and not all(stage_hit):
            # At least one stage hit SL
            pass

        # Weighted avg exit price (treat as single full-close at the weighted price)
        weights = [w for w, _ in tps]
        prices = stage_exit
        if any(p is None for p in prices):
            # Defensive: should not happen — fill any None with end-of-window close
            prices = [p if p is not None else float(closes[end_idx]) for p in prices]
        weighted_price = sum(w * p for w, p in zip(weights, prices))
        n_tp_hit = sum(stage_hit)
        reason = f"tp_scaled_{n_tp_hit}of3" if n_tp_hit > 0 else "time_or_sl"
        return _outcome(df, sig, last_exit_idx, weighted_price, reason, tf_min)


# ─────────────────────────────────────────────────────────────────────
# v2.2 — Partial 50% TP1 + 50% trail by max-extreme − 0.5 ATR
# ─────────────────────────────────────────────────────────────────────
class AbsorptionPartial50Trail05(ExitPolicy):
    name = "absorption_partial_50_trail0.5atr"
    is_baseline = False
    MAX_BARS = 8     # slightly longer window since trail can run
    TRAIL_ATR_MULT = 0.5
    TP1_RR = 1.0

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        sl0 = _build_sl_anchor(sig)
        tp1 = _tp_at(sig, sl0, self.TP1_RR)
        end_idx = min(sig.entry_idx + self.MAX_BARS, len(df) - 1)
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        atr_buf = self.TRAIL_ATR_MULT * sig.atr_at_entry

        # Phase 1: walk to TP1 / SL / time
        tp1_idx = None
        for j in range(sig.entry_idx + 1, end_idx + 1):
            bh = float(highs[j]); bl = float(lows[j])
            if sig.side == "long":
                if bl <= sl0:
                    return _outcome(df, sig, j, sl0, "sl_pre_tp1", tf_min)
                if bh >= tp1:
                    tp1_idx = j; break
            else:
                if bh >= sl0:
                    return _outcome(df, sig, j, sl0, "sl_pre_tp1", tf_min)
                if bl <= tp1:
                    tp1_idx = j; break
        if tp1_idx is None:
            return _outcome(df, sig, end_idx, float(closes[end_idx]), "time_pre_tp1", tf_min)

        # Phase 2: half closed at TP1 + half trails by max-extreme − atr_buf
        # Trail seed = entry price (BE) at the moment of TP1
        trail = sig.entry_price
        peak = sig.entry_price
        for j in range(tp1_idx + 1, end_idx + 1):
            bh = float(highs[j]); bl = float(lows[j])
            if sig.side == "long":
                peak = max(peak, bh)
                cand = peak - atr_buf
                trail = max(trail, cand)
                if bl <= trail:
                    # Weighted exit: 50% at TP1, 50% at trail
                    weighted = 0.5 * tp1 + 0.5 * trail
                    return _outcome(df, sig, j, weighted, "partial_tp1_trail", tf_min)
            else:
                peak = min(peak, bl)
                cand = peak + atr_buf
                trail = min(trail, cand)
                if bh >= trail:
                    weighted = 0.5 * tp1 + 0.5 * trail
                    return _outcome(df, sig, j, weighted, "partial_tp1_trail", tf_min)

        # Time stop on remainder — exit at end_idx close, weighted with TP1
        end_close = float(closes[end_idx])
        weighted = 0.5 * tp1 + 0.5 * end_close
        return _outcome(df, sig, end_idx, weighted, "partial_tp1_time_remainder", tf_min)


# ─────────────────────────────────────────────────────────────────────
# v2.3 — Dynamic MFE-trail (no fixed TP, pure MFE peak retrace exit)
# ─────────────────────────────────────────────────────────────────────
class AbsorptionDynamicMfeTrail(ExitPolicy):
    name = "absorption_dynamic_mfe_trail_33pct"
    is_baseline = False
    MAX_BARS = 8
    GIVEBACK_PCT = 0.33    # exit when price retraces 33% from MFE peak

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        sl0 = _build_sl_anchor(sig)
        end_idx = min(sig.entry_idx + self.MAX_BARS, len(df) - 1)
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values

        peak_fav = sig.entry_price   # most-favorable price seen so far
        for j in range(sig.entry_idx + 1, end_idx + 1):
            bh = float(highs[j]); bl = float(lows[j])
            # SL check first
            if sig.side == "long":
                if bl <= sl0:
                    return _outcome(df, sig, j, sl0, "sl", tf_min)
                peak_fav = max(peak_fav, bh)
                # Compute giveback threshold
                fav_move = peak_fav - sig.entry_price
                if fav_move > 0:
                    giveback_price = peak_fav - self.GIVEBACK_PCT * fav_move
                    # Exit if low touches giveback_price
                    if bl <= giveback_price and giveback_price > sig.entry_price:
                        return _outcome(df, sig, j, giveback_price, "mfe_giveback_33%", tf_min)
            else:
                if bh >= sl0:
                    return _outcome(df, sig, j, sl0, "sl", tf_min)
                peak_fav = min(peak_fav, bl)
                fav_move = sig.entry_price - peak_fav
                if fav_move > 0:
                    giveback_price = peak_fav + self.GIVEBACK_PCT * fav_move
                    if bh >= giveback_price and giveback_price < sig.entry_price:
                        return _outcome(df, sig, j, giveback_price, "mfe_giveback_33%", tf_min)

        # Time stop
        return _outcome(df, sig, end_idx, float(closes[end_idx]), "time_stop", tf_min)


def main():
    out_dir = ROOT / "storage" / "wf_studies" / "exit_absorption_v2"
    run_exit_wf(
        scanner="absorption_bubble",
        symbols=["ETH"],
        tf="5m", tf_min=TF_MIN,
        signal_generator=generate_absorption_signals,
        policies=[
            AbsorptionBaseline(),                         # baseline (live OLD)
            AbsorptionFastExit(),                         # v1 winner — already shipped
            AbsorptionTpScaling3Stage(),                  # v2.1 — TP scaling
            AbsorptionPartial50Trail05(),                 # v2.2 — partial + trail
            AbsorptionDynamicMfeTrail(),                  # v2.3 — dynamic MFE trail
        ],
        out_dir=out_dir,
        eval_window_bars=EVAL_WINDOW_BARS,
        notional=NOTIONAL,
        extra_notes=[
            "- Baseline = original live (TP=1.5R / SL=sweep±0.3*ATR / time=30min, no vol-fade).",
            "- v1 winner (absorption_fast_exit_6bar) ALREADY shipped to live engine 2026-05-03 — TP=1.0R + vol-fade + max 6 bars.",
            "- v2 variants test architect-requested ideas: TP scaling, partial exits, dynamic MFE trail.",
            "- All policies share IDENTICAL entries (same sweep-absorption-reclaim signals).",
            "- ETH only (only W/F-PASSing symbol for absorption_bubble).",
        ],
    )


if __name__ == "__main__":
    main()

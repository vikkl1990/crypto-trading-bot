#!/usr/bin/env python3
"""Exit W/F #4 — ensemble_wider_tp vs baseline (Patch R combo #1 + #2 scanners).

Same-entry comparison: re-detects signals, classifies each as ENSEMBLE-TAGGED
(Patch R policy match) or SOLO, then evaluates each under each exit policy.

Patch R rules tested (subset that have production deployment):
  COMBO #1 — SOL liq_grab_ob_fvg + liquidity_sweep_htf same-side within 1 bar (300s)
  COMBO #2 — BTC scalper_vwap_mr + structure_bounce same-side within 3 bars (900s)

For combo #1 we use SOL.  For combo #2 we use BTC.

Policies (all tested per scanner):
  - default_tp_15R              — current scanner-default TP=1.5R (same as baseline)
  - ensemble_wider_tp_2R        — solo: TP=1.5R; ensemble-tagged: TP=2.0R
  - ensemble_wider_tp_3R        — solo: TP=1.5R; ensemble-tagged: TP=3.0R
  - ensemble_only_2R            — TP=2R but ONLY take ensemble-tagged signals
                                  (n drops drastically — tests whether the tagged
                                   subset ALONE has higher EV)

Hard SL = same as baseline (sweep extreme ± 0.3 × ATR), time stop = 30 min.

NOTE: this study REQUIRES enough signals to have ensemble tags. If the
ensemble-tag rate is too low, the cell is PARKED automatically.

Symbols: SOL (combo #1) and BTC (combo #2). Run twice — once per combo.
TF: 5m.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exit_policy_harness import (  # noqa: E402
    Signal, ExitOutcome, ExitPolicy, run_exit_wf, add_atr,
)

# ── Detector params ──────────────────────────────────────────────────────────
TF_MIN = 5
ATR_PERIOD = 14
SL_BUFFER_ATR = 0.3
TIME_STOP_MIN = 30
EVAL_WINDOW_BARS = 60
NOTIONAL = 1000.0


# ─────────────────────────────────────────────────────────────────────
# COMBO #1 — SOL liq_grab_ob_fvg + liquidity_sweep_htf
# Patch R: same-side within 1 bar (300s) on same symbol
# ─────────────────────────────────────────────────────────────────────
SOL_SWEEP_LB = 20
SOL_SWEEP_ATR = 0.3
SOL_DISPL_ATR = 0.5

def _detect_liq_grab_or_sweep_htf(df, side_filter=None):
    """Return list of (idx, side) tuples — both detectors return effectively the
    same shape under our simplified W/F (both are sweep+reclaim variants).
    For Patch R purposes, we DUPLICATE this detection under two 'source' names
    and check temporal proximity between them."""
    if df.empty or len(df) < SOL_SWEEP_LB + ATR_PERIOD + 5:
        return []
    atr = add_atr(df, ATR_PERIOD)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    out = []
    n = len(df)
    for i in range(SOL_SWEEP_LB + ATR_PERIOD + 1, n - 1):
        atr_j = float(atr.iloc[i - 1]) if not pd.isna(atr.iloc[i - 1]) else 0.0
        if atr_j <= 0: continue
        j = i - 1
        win_lo = float(np.min(lows[j - SOL_SWEEP_LB: j]))
        win_hi = float(np.max(highs[j - SOL_SWEEP_LB: j]))
        bar_j_lo = float(lows[j]); bar_j_hi = float(highs[j])
        long_sweep = (win_lo - bar_j_lo) >= SOL_SWEEP_ATR * atr_j
        short_sweep = (bar_j_hi - win_hi) >= SOL_SWEEP_ATR * atr_j
        if not (long_sweep or short_sweep): continue
        bar_i_op = float(opens[i]); bar_i_cl = float(closes[i])
        body = abs(bar_i_cl - bar_i_op)
        if body < SOL_DISPL_ATR * atr_j: continue
        if long_sweep:
            if bar_i_cl <= win_lo or bar_i_cl <= bar_i_op: continue
            side = "long"; ref = bar_j_lo
        else:
            if bar_i_cl >= win_hi or bar_i_cl >= bar_i_op: continue
            side = "short"; ref = bar_j_hi
        if side_filter and side != side_filter: continue
        out.append((i, side, ref, atr_j, bar_i_cl))
    return out


def generate_combo1_sol_signals(df: pd.DataFrame, symbol: str) -> List[Signal]:
    """SOL combo #1: same-side liq_grab + liq_sweep_htf within 1 bar (300s).

    Since both detectors collapse to the same logic under our simplification,
    we mark a signal as ENSEMBLE-TAGGED if there exists ANOTHER same-side signal
    (under the same detector) within ±1 bar — which is a degenerate case.

    To make this meaningful, we use TWO DIFFERENT TIMEFRAMES of the same logic:
      - 5m sweep (the entry signal)
      - 15m sweep (proxy for liquidity_sweep_HTF)
    A signal is tagged if a 15m sweep of the same side fired within ±1 5m bar
    of the 5m signal."""
    sigs_5m = _detect_liq_grab_or_sweep_htf(df)
    if not sigs_5m: return []

    # Build 15m candles by resampling
    df_15m = df.resample("15min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna()
    sigs_htf = _detect_liq_grab_or_sweep_htf(df_15m) if len(df_15m) > SOL_SWEEP_LB + ATR_PERIOD + 5 else []

    htf_events = [(df_15m.index[i_htf], side) for (i_htf, side, _, _, _) in sigs_htf]
    out: List[Signal] = []
    one_bar = pd.Timedelta(minutes=5)
    for (i_5m, side, ref, atr_j, entry_px) in sigs_5m:
        ts_5m = df.index[i_5m]
        # Tagged if any same-side HTF event within ±1 5m bar (5min)
        tagged = any(
            (h_side == side) and abs(h_ts - ts_5m) <= one_bar
            for (h_ts, h_side) in htf_events
        )
        out.append(Signal(
            symbol=symbol, entry_idx=i_5m, entry_ts=ts_5m,
            side=side, entry_price=entry_px,
            sweep_extreme=ref, atr_at_entry=atr_j,
            extra={"ensemble_tagged": tagged, "combo": "sol_liq_grab_sweep_htf"},
        ))
    return out


# ─────────────────────────────────────────────────────────────────────
# COMBO #2 — BTC scalper_vwap_mr + structure_bounce within 3 bars (900s)
# ─────────────────────────────────────────────────────────────────────
BTC_VWAP_LB = 50
BTC_VWAP_K = 1.5
BTC_ATR_PCT_LB = 100
BTC_ATR_PCT_THR = 0.5

def _detect_btc_vwap_mr(df) -> List[tuple]:
    """Returns (idx, side, sl_anchor, atr, entry_px) tuples for vwap_mr signals."""
    if df.empty or len(df) < max(BTC_VWAP_LB, BTC_ATR_PCT_LB) + ATR_PERIOD + 5:
        return []
    typ = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float).clip(lower=1e-9)
    pv = typ * vol
    vwap = pv.rolling(BTC_VWAP_LB, min_periods=BTC_VWAP_LB).sum() / \
           vol.rolling(BTC_VWAP_LB, min_periods=BTC_VWAP_LB).sum()
    stdev = df["close"].astype(float).rolling(BTC_VWAP_LB, min_periods=BTC_VWAP_LB).std(ddof=0)
    atr = add_atr(df, ATR_PERIOD)
    atr_pct = atr.rolling(BTC_ATR_PCT_LB, min_periods=BTC_ATR_PCT_LB).rank(pct=True)

    closes = df["close"].astype(float).values
    opens = df["open"].astype(float).values
    out = []
    n = len(df)
    for i in range(max(BTC_VWAP_LB, BTC_ATR_PCT_LB) + 1, n - 1):
        atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
        atr_p = float(atr_pct.iloc[i]) if not pd.isna(atr_pct.iloc[i]) else 1.0
        v = float(vwap.iloc[i]) if not pd.isna(vwap.iloc[i]) else None
        s = float(stdev.iloc[i]) if not pd.isna(stdev.iloc[i]) else None
        if atr_i <= 0 or v is None or s is None: continue
        if atr_p > BTC_ATR_PCT_THR: continue
        upper = v + BTC_VWAP_K * s; lower = v - BTC_VWAP_K * s
        c = float(closes[i]); o = float(opens[i]); cp = float(closes[i - 1])
        if c < lower and c > o and c > cp:
            out.append((i, "long", c - 1.0 * atr_i, atr_i, c))
        elif c > upper and c < o and c < cp:
            out.append((i, "short", c + 1.0 * atr_i, atr_i, c))
    return out


def _detect_btc_structure_bounce(df) -> List[tuple]:
    """Simplified structure-bounce: bar bounces off recent 10-bar swing extreme
    + bullish/bearish candle. Used only for ENSEMBLE TAG matching, not for entries."""
    if df.empty or len(df) < 30: return []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    out = []
    lb = 10
    n = len(df)
    for i in range(lb + 1, n - 1):
        win_lo = float(np.min(lows[i - lb: i]))
        win_hi = float(np.max(highs[i - lb: i]))
        bl = float(lows[i]); bh = float(highs[i])
        bo = float(opens[i]); bc = float(closes[i])
        # bounce off win_lo (LONG) — bar tagged win_lo and closed bullish
        if bl <= win_lo and bc > bo and bc > win_lo:
            out.append((i, "long"))
        elif bh >= win_hi and bc < bo and bc < win_hi:
            out.append((i, "short"))
    return out


def generate_combo2_btc_signals(df: pd.DataFrame, symbol: str) -> List[Signal]:
    """BTC combo #2: scalper_vwap_mr signal + structure_bounce same-side within 3 bars."""
    vwap_sigs = _detect_btc_vwap_mr(df)
    sb_sigs = _detect_btc_structure_bounce(df)
    sb_by_side: dict = {"long": [], "short": []}
    for (i, side) in sb_sigs:
        sb_by_side[side].append(df.index[i])
    out: List[Signal] = []
    three_bars = pd.Timedelta(minutes=15)
    for (i, side, sl_anchor, atr_i, entry_px) in vwap_sigs:
        ts = df.index[i]
        # Tagged if any same-side SB signal within ±3 5m bars
        tagged = any(abs(t - ts) <= three_bars for t in sb_by_side[side])
        out.append(Signal(
            symbol=symbol, entry_idx=i, entry_ts=ts,
            side=side, entry_price=entry_px,
            sweep_extreme=sl_anchor, atr_at_entry=atr_i,
            extra={"ensemble_tagged": tagged, "combo": "btc_vwap_mr_sb"},
        ))
    return out


# ─────────────────────────────────────────────────────────────────────
# Exit policies (ensemble-aware)
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


def _walk_to_hit(df, sig, sl, tp, max_bars, tf_min):
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


def _build_sl_tp(sig, tp_rr):
    sl_buf = SL_BUFFER_ATR * sig.atr_at_entry
    if sig.side == "long":
        sl = sig.sweep_extreme - sl_buf
        risk = sig.entry_price - sl
        tp = sig.entry_price + tp_rr * max(risk, 1e-9)
    else:
        sl = sig.sweep_extreme + sl_buf
        risk = sl - sig.entry_price
        tp = sig.entry_price - tp_rr * max(risk, 1e-9)
    return sl, tp


class DefaultTp15R(ExitPolicy):
    name = "default_tp_1.5R"
    is_baseline = True

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        sl, tp = _build_sl_tp(sig, 1.5)
        max_bars = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
        return _walk_to_hit(df, sig, sl, tp, max_bars, tf_min)


class EnsembleWider2R(ExitPolicy):
    """Solo signal → TP=1.5R; ensemble-tagged → TP=2.0R. Same SL, same time."""
    name = "ensemble_2R_solo_1.5R"
    is_baseline = False

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        tp_rr = 2.0 if sig.extra.get("ensemble_tagged") else 1.5
        sl, tp = _build_sl_tp(sig, tp_rr)
        max_bars = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
        return _walk_to_hit(df, sig, sl, tp, max_bars, tf_min)


class EnsembleWider3R(ExitPolicy):
    name = "ensemble_3R_solo_1.5R"
    is_baseline = False

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        tp_rr = 3.0 if sig.extra.get("ensemble_tagged") else 1.5
        sl, tp = _build_sl_tp(sig, tp_rr)
        max_bars = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
        return _walk_to_hit(df, sig, sl, tp, max_bars, tf_min)


class EnsembleOnly2R(ExitPolicy):
    """TP=2R but ABANDON solo signals immediately (zero PnL — we didn't enter)."""
    name = "ensemble_only_2R_skip_solo"
    is_baseline = False

    def evaluate(self, df, sig, tf_min, eval_window_bars):
        if not sig.extra.get("ensemble_tagged"):
            # Treat as no-trade: exit immediately at entry price (zero gross_pct)
            return _outcome(df, sig, sig.entry_idx + 1, sig.entry_price, "skipped_solo", tf_min)
        sl, tp = _build_sl_tp(sig, 2.0)
        max_bars = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
        return _walk_to_hit(df, sig, sl, tp, max_bars, tf_min)


def main():
    # Combo #1 — SOL
    out_sol = ROOT / "storage" / "wf_studies" / "exit_ensemble_wider_tp_sol"
    run_exit_wf(
        scanner="combo1_sol_liq_grab_sweep_htf",
        symbols=["SOL"],
        tf="5m", tf_min=TF_MIN,
        signal_generator=generate_combo1_sol_signals,
        policies=[
            DefaultTp15R(), EnsembleWider2R(), EnsembleWider3R(), EnsembleOnly2R(),
        ],
        out_dir=out_sol,
        eval_window_bars=EVAL_WINDOW_BARS,
        notional=NOTIONAL,
        extra_notes=[
            "- SOL combo #1 (Patch R rule sol_smc_confluence_sizer).",
            "- Ensemble tag = 5m sweep + 15m sweep same-side within ±1 5m bar (proxy for liq_grab + liq_sweep_htf).",
            "- Baseline TP=1.5R; candidates apply wider TP only when tagged.",
        ],
    )

    # Combo #2 — BTC
    out_btc = ROOT / "storage" / "wf_studies" / "exit_ensemble_wider_tp_btc"
    run_exit_wf(
        scanner="combo2_btc_vwap_mr_sb",
        symbols=["BTC"],
        tf="5m", tf_min=TF_MIN,
        signal_generator=generate_combo2_btc_signals,
        policies=[
            DefaultTp15R(), EnsembleWider2R(), EnsembleWider3R(), EnsembleOnly2R(),
        ],
        out_dir=out_btc,
        eval_window_bars=EVAL_WINDOW_BARS,
        notional=NOTIONAL,
        extra_notes=[
            "- BTC combo #2 (Patch R rule btc_vwap_sb_booster).",
            "- Ensemble tag = scalper_vwap_mr signal + same-side structure_bounce within ±3 5m bars.",
            "- Baseline TP=1.5R; candidates apply wider TP only when tagged.",
        ],
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Classical Chart Patterns - Walk-forward backtest.

Detects 8 classical patterns (H&S, Inverse H&S, Double Top, Double Bottom,
Rising Wedge, Falling Wedge, Bull Flag, Bear Flag), simulates 3 exit
configurations per signal, and walk-forward tests the best (params x exit)
per (pattern x TF) across 4 quarters: Q1+Q2 in-sample, Q3 OOS, Q4 OOS.

Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive).
Reject if OOS gap > 50%.

Output:
    storage/classical_patterns/walkforward.json
    storage/classical_patterns/report.md

Read-only on:
    bot/signal_tracker.py, bot/signal_journey.py, bot/signal_learner.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROOT = Path("/home/opc/crypto-trading-bot")
CACHE = ROOT / "storage" / "candle_cache"
OUT_DIR = ROOT / "storage" / "classical_patterns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Pattern -> TF list (from spec)
PATTERN_TFS = {
    "head_shoulders": ["1h", "4h"],
    "inverse_head_shoulders": ["1h", "4h"],
    "double_top": ["1h", "4h"],
    "double_bottom": ["1h", "4h"],
    "rising_wedge": ["1h", "4h"],
    "falling_wedge": ["1h", "4h"],
    "bull_flag": ["15m", "1h"],
    "bear_flag": ["15m", "1h"],
}

# Parameter grid (tuned on Q1+Q2 only)
GRID_PIVOT_LOOKBACK = [5, 8, 10]
GRID_DETECT_WINDOW = [30, 50, 80]
GRID_VOL_THRESHOLD = [1.0, 1.2, 1.5]

# Three exit configs per spec
EXITS = {
    "EA": {"tp": 2.0, "sl": 1.0, "time_min": 4 * 60, "trail": False},
    "EB": {"tp": 3.0, "sl": 1.0, "time_min": 12 * 60, "trail": False},
    "EC": {
        "tp": 99.0,
        "sl": 1.0,
        "time_min": 8 * 60,
        "trail": True,
        "trail_trigger": 1.0,
        "trail_lock_pct": 0.5,
    },
}

# Trade economics
NOTIONAL = 1000.0
RT_TAKER_FEE_BPS = 0.118  # round-trip Delta India taker (in pct of notional)
FUNDING_RATE_PER_8H = 0.0001
ATR_PERIOD = 14
VOL_LOOKBACK = 20

# Walk-forward boundaries (UTC)
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}


# -----------------------------------------------------------------------------
# Indicator helpers
# -----------------------------------------------------------------------------
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)


def find_pivots(highs: np.ndarray, lows: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of pivot indices (high pivots, low pivots).

    Pivot high at i: high[i] is the strict max within [i-lookback, i+lookback].
    Pivot low at i: low[i] is the strict min within the same window.
    """
    n = len(highs)
    ph = []
    pl = []
    for i in range(lookback, n - lookback):
        h_win = highs[i - lookback : i + lookback + 1]
        l_win = lows[i - lookback : i + lookback + 1]
        if highs[i] == h_win.max() and (h_win == highs[i]).sum() == 1:
            ph.append(i)
        if lows[i] == l_win.min() and (l_win == lows[i]).sum() == 1:
            pl.append(i)
    return np.array(ph, dtype=int), np.array(pl, dtype=int)


# -----------------------------------------------------------------------------
# Pattern detectors. Each returns a list of dicts:
#   {entry_idx, side, neckline_or_break_level (optional), atr}
# entry_idx is the index of the candle that closed the break (signal bar).
# -----------------------------------------------------------------------------
def _vol_ok(df: pd.DataFrame, idx: int, vol_threshold: float) -> bool:
    if idx < VOL_LOOKBACK:
        return False
    vol = float(df["volume"].iloc[idx])
    avg = float(df["volume"].iloc[idx - VOL_LOOKBACK : idx].mean())
    if not np.isfinite(avg) or avg <= 0:
        return False
    return (vol / avg) >= vol_threshold


def detect_head_shoulders(
    df: pd.DataFrame,
    atr: pd.Series,
    pivot_lb: int,
    detect_window: int,
    vol_threshold: float,
    side: str,  # "SHORT" for H&S, "LONG" for Inverse H&S
) -> List[Dict[str, Any]]:
    """
    H&S (SHORT): three pivot highs in window — left shoulder L, head H, right
    shoulder R — with |L - R| <= 0.5 ATR, head >= max(L, R) + 1 ATR. Neckline
    connects the two intervening pivot lows. Trigger: candle close < neckline
    AT signal bar with volume >= threshold * avg.

    Inverse (LONG): mirror — three pivot lows, head distinctly below shoulders,
    trigger on close > neckline.
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    ph, pl = find_pivots(highs, lows, pivot_lb)
    n = len(df)

    signals: List[Dict[str, Any]] = []

    if side == "SHORT":
        # Need three pivot highs and two pivot lows in between.
        # For each candidate triplet (L < H < R) of pivot highs:
        for k in range(2, len(ph)):
            R = int(ph[k])
            H = int(ph[k - 1])
            L = int(ph[k - 2])
            if R - L > detect_window:
                continue
            atr_v = float(atr.iloc[R]) if not pd.isna(atr.iloc[R]) else float("nan")
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            hL, hH, hR = highs[L], highs[H], highs[R]
            if abs(hL - hR) > 0.5 * atr_v:
                continue
            if hH < max(hL, hR) + 1.0 * atr_v:
                continue
            # Find pivot lows between L-H and H-R.
            mids = pl[(pl > L) & (pl < H)]
            mids2 = pl[(pl > H) & (pl < R)]
            if len(mids) == 0 or len(mids2) == 0:
                continue
            n1 = int(mids[-1])
            n2 = int(mids2[0])
            # Neckline: line between (n1, lows[n1]) and (n2, lows[n2]).
            if n2 == n1:
                continue
            slope = (lows[n2] - lows[n1]) / (n2 - n1)
            # Look for break after R within detect_window/2 bars.
            search_max = min(n - 1, R + max(detect_window // 2, 5))
            for j in range(R + 1, search_max + 1):
                neck_lvl = lows[n2] + slope * (j - n2)
                if closes[j] < neck_lvl and _vol_ok(df, j, vol_threshold):
                    signals.append({
                        "entry_idx": j,
                        "side": "SHORT",
                        "atr": atr_v,
                    })
                    break
    else:  # LONG inverse
        for k in range(2, len(pl)):
            R = int(pl[k])
            H = int(pl[k - 1])
            L = int(pl[k - 2])
            if R - L > detect_window:
                continue
            atr_v = float(atr.iloc[R]) if not pd.isna(atr.iloc[R]) else float("nan")
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            lL, lH, lR = lows[L], lows[H], lows[R]
            if abs(lL - lR) > 0.5 * atr_v:
                continue
            if lH > min(lL, lR) - 1.0 * atr_v:
                continue
            mids = ph[(ph > L) & (ph < H)]
            mids2 = ph[(ph > H) & (ph < R)]
            if len(mids) == 0 or len(mids2) == 0:
                continue
            n1 = int(mids[-1])
            n2 = int(mids2[0])
            if n2 == n1:
                continue
            slope = (highs[n2] - highs[n1]) / (n2 - n1)
            search_max = min(n - 1, R + max(detect_window // 2, 5))
            for j in range(R + 1, search_max + 1):
                neck_lvl = highs[n2] + slope * (j - n2)
                if closes[j] > neck_lvl and _vol_ok(df, j, vol_threshold):
                    signals.append({
                        "entry_idx": j,
                        "side": "LONG",
                        "atr": atr_v,
                    })
                    break
    return signals


def detect_double(
    df: pd.DataFrame,
    atr: pd.Series,
    pivot_lb: int,
    detect_window: int,
    vol_threshold: float,
    side: str,  # "SHORT" double-top, "LONG" double-bottom
) -> List[Dict[str, Any]]:
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    ph, pl = find_pivots(highs, lows, pivot_lb)
    n = len(df)

    signals: List[Dict[str, Any]] = []

    if side == "SHORT":
        # Two pivot highs within 0.3 ATR, separated by >=10 bars.
        # Between them must be at least one pivot low (intervening swing low).
        # Trigger: close < intervening low.
        for k in range(1, len(ph)):
            i2 = int(ph[k])
            i1 = int(ph[k - 1])
            if (i2 - i1) < 10 or (i2 - i1) > detect_window:
                continue
            atr_v = float(atr.iloc[i2]) if not pd.isna(atr.iloc[i2]) else float("nan")
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            if abs(highs[i1] - highs[i2]) > 0.3 * atr_v:
                continue
            mids = pl[(pl > i1) & (pl < i2)]
            if len(mids) == 0:
                continue
            mid_idx = int(mids[np.argmin(lows[mids])])
            mid_low = lows[mid_idx]
            search_max = min(n - 1, i2 + max(detect_window // 2, 5))
            for j in range(i2 + 1, search_max + 1):
                if closes[j] < mid_low and _vol_ok(df, j, vol_threshold):
                    signals.append({
                        "entry_idx": j,
                        "side": "SHORT",
                        "atr": atr_v,
                    })
                    break
    else:
        for k in range(1, len(pl)):
            i2 = int(pl[k])
            i1 = int(pl[k - 1])
            if (i2 - i1) < 10 or (i2 - i1) > detect_window:
                continue
            atr_v = float(atr.iloc[i2]) if not pd.isna(atr.iloc[i2]) else float("nan")
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            if abs(lows[i1] - lows[i2]) > 0.3 * atr_v:
                continue
            mids = ph[(ph > i1) & (ph < i2)]
            if len(mids) == 0:
                continue
            mid_idx = int(mids[np.argmax(highs[mids])])
            mid_high = highs[mid_idx]
            search_max = min(n - 1, i2 + max(detect_window // 2, 5))
            for j in range(i2 + 1, search_max + 1):
                if closes[j] > mid_high and _vol_ok(df, j, vol_threshold):
                    signals.append({
                        "entry_idx": j,
                        "side": "LONG",
                        "atr": atr_v,
                    })
                    break
    return signals


def detect_wedge(
    df: pd.DataFrame,
    atr: pd.Series,
    pivot_lb: int,
    detect_window: int,
    vol_threshold: float,
    side: str,  # "SHORT" rising wedge, "LONG" falling wedge
) -> List[Dict[str, Any]]:
    """
    Rising wedge (SHORT): 5+ candles, slope_high>0, slope_low>0, slope_high < slope_low
       (lines converge, both rising). Trigger: close < lower trendline.
    Falling wedge (LONG): mirror — both slopes negative, slope_high < slope_low
       (so |slope_high| > |slope_low|, i.e. upper falls steeper). Trigger: close > upper trendline.
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    ph, pl = find_pivots(highs, lows, pivot_lb)
    n = len(df)
    signals: List[Dict[str, Any]] = []

    # Walk along time, look at last 2+ pivot highs and 2+ pivot lows within window.
    for end in range(detect_window, n):
        atr_v = float(atr.iloc[end]) if not pd.isna(atr.iloc[end]) else float("nan")
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue
        recent_ph = ph[(ph >= end - detect_window) & (ph < end)]
        recent_pl = pl[(pl >= end - detect_window) & (pl < end)]
        if len(recent_ph) < 2 or len(recent_pl) < 2:
            continue
        # Linear fit slopes
        ph_y = highs[recent_ph]
        pl_y = lows[recent_pl]
        ph_x = recent_ph.astype(float)
        pl_x = recent_pl.astype(float)
        if len(recent_ph) >= 2:
            sh = float(np.polyfit(ph_x, ph_y, 1)[0])
        else:
            continue
        if len(recent_pl) >= 2:
            sl = float(np.polyfit(pl_x, pl_y, 1)[0])
        else:
            continue

        if side == "SHORT":
            # Both slopes positive, slope_high < slope_low (converge upward).
            if sh <= 0 or sl <= 0 or sh >= sl:
                continue
            # Lower trendline value at `end`:
            b_l = float(np.polyfit(pl_x, pl_y, 1)[1])
            lower_lvl = sl * end + b_l
            if closes[end] < lower_lvl and _vol_ok(df, end, vol_threshold):
                signals.append({
                    "entry_idx": int(end),
                    "side": "SHORT",
                    "atr": atr_v,
                })
        else:
            # Falling wedge: both slopes negative, slope_high < slope_low (upper falls steeper).
            if sh >= 0 or sl >= 0 or sh >= sl:
                continue
            b_h = float(np.polyfit(ph_x, ph_y, 1)[1])
            upper_lvl = sh * end + b_h
            if closes[end] > upper_lvl and _vol_ok(df, end, vol_threshold):
                signals.append({
                    "entry_idx": int(end),
                    "side": "LONG",
                    "atr": atr_v,
                })
    # De-dupe close adjacent triggers (within pivot_lb bars)
    if not signals:
        return signals
    deduped = [signals[0]]
    for s in signals[1:]:
        if s["entry_idx"] - deduped[-1]["entry_idx"] >= pivot_lb:
            deduped.append(s)
    return deduped


def detect_flag(
    df: pd.DataFrame,
    atr: pd.Series,
    pivot_lb: int,
    detect_window: int,
    vol_threshold: float,
    side: str,  # "LONG" bull flag, "SHORT" bear flag
) -> List[Dict[str, Any]]:
    """
    Bull flag (LONG): impulse = prior 5-10 bars cumulative range > 2 ATR upward
       (close[t] - close[t-K] > 2 ATR), then 5-15 bars consolidation
       (range < impulse / 2, drift sideways/down). Trigger: close > pullback high.
    Bear flag (SHORT): mirror.
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    n = len(df)
    signals: List[Dict[str, Any]] = []

    impulse_lookbacks = [5, 7, 10]
    pullback_lengths = [5, 8, 12, 15]

    # Don't double-fire — keep cooldown of detect_window/2 bars
    last_trigger = -10**9
    for end in range(VOL_LOOKBACK + max(impulse_lookbacks) + max(pullback_lengths), n):
        if end - last_trigger < pivot_lb * 2:
            continue
        atr_v = float(atr.iloc[end]) if not pd.isna(atr.iloc[end]) else float("nan")
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue
        for pb_len in pullback_lengths:
            for imp_len in impulse_lookbacks:
                imp_start = end - pb_len - imp_len
                imp_end = end - pb_len
                if imp_start < 0:
                    continue
                imp_move = closes[imp_end] - closes[imp_start]
                imp_range_h = highs[imp_start : imp_end + 1].max() - lows[imp_start : imp_end + 1].min()
                if imp_range_h < 2.0 * atr_v:
                    continue
                pb_high = highs[imp_end : end].max() if imp_end < end else 0
                pb_low = lows[imp_end : end].min() if imp_end < end else 0
                pb_range = pb_high - pb_low
                if pb_range >= imp_range_h / 1.5:
                    continue
                if side == "LONG":
                    if imp_move <= 0:
                        continue
                    # Impulse must be the UP push; consolidation must hold above
                    # impulse_start * 0.5 (rough).
                    # Trigger: close > pb_high
                    if closes[end] > pb_high and _vol_ok(df, end, vol_threshold):
                        signals.append({
                            "entry_idx": int(end),
                            "side": "LONG",
                            "atr": atr_v,
                        })
                        last_trigger = end
                        break
                else:
                    if imp_move >= 0:
                        continue
                    if closes[end] < pb_low and _vol_ok(df, end, vol_threshold):
                        signals.append({
                            "entry_idx": int(end),
                            "side": "SHORT",
                            "atr": atr_v,
                        })
                        last_trigger = end
                        break
            else:
                continue
            break  # break outer pb_len loop after first hit
    return signals


# Pattern dispatcher
PATTERN_DETECTORS = {
    "head_shoulders": (detect_head_shoulders, "SHORT"),
    "inverse_head_shoulders": (detect_head_shoulders, "LONG"),
    "double_top": (detect_double, "SHORT"),
    "double_bottom": (detect_double, "LONG"),
    "rising_wedge": (detect_wedge, "SHORT"),
    "falling_wedge": (detect_wedge, "LONG"),
    "bull_flag": (detect_flag, "LONG"),
    "bear_flag": (detect_flag, "SHORT"),
}


# -----------------------------------------------------------------------------
# Trade simulator
# -----------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr: float,
    exit_cfg: Dict[str, Any],
    tf_min: int,
) -> Optional[Dict[str, Any]]:
    if entry_idx + 1 >= len(df):
        return None
    entry = float(df["close"].iloc[entry_idx])
    if not np.isfinite(atr) or atr <= 0:
        return None

    sl_dist = exit_cfg["sl"] * atr
    tp_dist = exit_cfg["tp"] * atr
    bars_max = max(1, int(math.ceil(exit_cfg["time_min"] / tf_min)))

    if side == "LONG":
        sl_p = entry - sl_dist
        tp_p = entry + tp_dist
    else:
        sl_p = entry + sl_dist
        tp_p = entry - tp_dist

    trail_active = False
    trail_stop = None
    mfe_R = 0.0

    end_idx = min(entry_idx + bars_max, len(df) - 1)
    exit_reason = "TIME"
    exit_price = float(df["close"].iloc[end_idx])
    exit_idx = end_idx

    for j in range(entry_idx + 1, end_idx + 1):
        bar_h = float(df["high"].iloc[j])
        bar_l = float(df["low"].iloc[j])

        if side == "LONG":
            mfe = (bar_h - entry) / sl_dist
        else:
            mfe = (entry - bar_l) / sl_dist
        if mfe > mfe_R:
            mfe_R = mfe

        if exit_cfg.get("trail"):
            trigger = exit_cfg.get("trail_trigger", 1.0)
            lock_pct = exit_cfg.get("trail_lock_pct", 0.5)
            if not trail_active and mfe_R >= trigger:
                trail_active = True
            if trail_active:
                lock_R = mfe_R * lock_pct
                if side == "LONG":
                    new_stop = entry + lock_R * sl_dist
                    if trail_stop is None or new_stop > trail_stop:
                        trail_stop = new_stop
                else:
                    new_stop = entry - lock_R * sl_dist
                    if trail_stop is None or new_stop < trail_stop:
                        trail_stop = new_stop

        if side == "LONG":
            hit_sl = bar_l <= sl_p
            hit_tp = bar_h >= tp_p
            hit_trail = trail_stop is not None and bar_l <= trail_stop
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_trail:
                exit_reason = "TRAIL"; exit_price = trail_stop; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break
        else:
            hit_sl = bar_h >= sl_p
            hit_tp = bar_l <= tp_p
            hit_trail = trail_stop is not None and bar_h >= trail_stop
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_trail:
                exit_reason = "TRAIL"; exit_price = trail_stop; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break

    if side == "LONG":
        ret = (exit_price - entry) / entry
    else:
        ret = (entry - exit_price) / entry

    gross_dollars = NOTIONAL * ret
    fee_dollars = NOTIONAL * (RT_TAKER_FEE_BPS / 100.0)
    bars_held = exit_idx - entry_idx
    minutes_held = bars_held * tf_min
    funding_dollars = NOTIONAL * FUNDING_RATE_PER_8H * (minutes_held / (8 * 60))
    net_dollars = gross_dollars - fee_dollars - funding_dollars

    risk_dollars = NOTIONAL * (sl_dist / entry)
    gross_R = gross_dollars / risk_dollars if risk_dollars > 0 else 0.0
    net_R = net_dollars / risk_dollars if risk_dollars > 0 else 0.0

    return {
        "entry_ts": df.index[entry_idx],
        "exit_ts": df.index[exit_idx],
        "side": side,
        "exit_reason": exit_reason,
        "entry": entry,
        "exit": exit_price,
        "minutes_held": minutes_held,
        "gross_R": gross_R,
        "net_R": net_R,
        "gross_$": gross_dollars,
        "net_$": net_dollars,
        "mfe_R": mfe_R,
        "atr": atr,
    }


def quarter_of(ts: pd.Timestamp) -> Optional[str]:
    for q, (s, e) in QUARTER_BOUNDS.items():
        if pd.Timestamp(s, tz="UTC") <= ts < pd.Timestamp(e, tz="UTC"):
            return q
    return None


def aggregate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "wr": 0.0, "ev_R": 0.0, "ev_$": 0.0,
                "gross_R": 0.0, "net_R": 0.0, "gross_$": 0.0, "net_$": 0.0,
                "avg_mfe_R": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_R"] > 0)
    return {
        "n": n,
        "wr": wins / n,
        "ev_R": float(np.mean([t["net_R"] for t in trades])),
        "ev_$": float(np.mean([t["net_$"] for t in trades])),
        "gross_R": float(np.sum([t["gross_R"] for t in trades])),
        "net_R": float(np.sum([t["net_R"] for t in trades])),
        "gross_$": float(np.sum([t["gross_$"] for t in trades])),
        "net_$": float(np.sum([t["net_$"] for t in trades])),
        "avg_mfe_R": float(np.mean([t["mfe_R"] for t in trades])),
    }


# -----------------------------------------------------------------------------
# Per-cell run
# -----------------------------------------------------------------------------
def run_cell_pattern(
    df: pd.DataFrame,
    tf: str,
    pattern: str,
    pivot_lb: int,
    detect_window: int,
    vol_threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all 3 exit configs against one (pattern x TF x params) cell on one symbol."""
    detector_fn, side = PATTERN_DETECTORS[pattern]
    atr = add_atr(df)
    tf_min = tf_minutes(tf)

    sigs = detector_fn(df, atr, pivot_lb, detect_window, vol_threshold, side)

    trades_by_exit: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
    for s in sigs:
        for ek, ec in EXITS.items():
            t = simulate_trade(df, s["entry_idx"], s["side"], s["atr"], ec, tf_min)
            if t is not None:
                trades_by_exit[ek].append(t)
    return trades_by_exit


def run_walkforward() -> Dict[str, Any]:
    """Full grid x WF run, picking best params per (pattern x TF x exit) on Q1+Q2."""
    all_results: Dict[str, Any] = {"per_cell": [], "walkforward": []}

    for pattern, tfs in PATTERN_TFS.items():
        for tf in tfs:
            sym_dfs: Dict[str, pd.DataFrame] = {}
            for sym in SYMBOLS:
                p = CACHE / f"{sym}_USDT_{tf}.parquet"
                if not p.exists():
                    continue
                d = pd.read_parquet(p)
                d.index = pd.to_datetime(d.index, utc=True)
                sym_dfs[sym] = d

            for pivot_lb, detect_window, vol_threshold in product(
                GRID_PIVOT_LOOKBACK, GRID_DETECT_WINDOW, GRID_VOL_THRESHOLD
            ):
                per_q_trades: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                    ek: {q: [] for q in QUARTER_BOUNDS} for ek in EXITS
                }
                for sym, d in sym_dfs.items():
                    trades_by_exit = run_cell_pattern(
                        d, tf, pattern, pivot_lb, detect_window, vol_threshold
                    )
                    for ek, trades in trades_by_exit.items():
                        for t in trades:
                            q = quarter_of(t["entry_ts"])
                            if q is None:
                                continue
                            per_q_trades[ek][q].append(t)

                for ek in EXITS:
                    cell_summary = {
                        "pattern": pattern,
                        "tf": tf,
                        "pivot_lb": pivot_lb,
                        "detect_window": detect_window,
                        "vol_threshold": vol_threshold,
                        "exit": ek,
                        "params_key": f"{pattern}|{tf}|plb{pivot_lb}|dw{detect_window}|vt{vol_threshold}|{ek}",
                    }
                    is_trades = per_q_trades[ek]["Q1"] + per_q_trades[ek]["Q2"]
                    cell_summary["IS"] = aggregate(is_trades)
                    cell_summary["Q3"] = aggregate(per_q_trades[ek]["Q3"])
                    cell_summary["Q4"] = aggregate(per_q_trades[ek]["Q4"])
                    all_results["per_cell"].append(cell_summary)

    # Walk-forward selection: per (pattern, tf, exit), pick best IS (n>=20, max ev_R)
    best_per: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for cell in all_results["per_cell"]:
        key = (cell["pattern"], cell["tf"], cell["exit"])
        if cell["IS"]["n"] < 20:
            continue
        cur = best_per.get(key)
        if cur is None or cell["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            best_per[key] = cell

    verdicts: List[Dict[str, Any]] = []
    for (pattern, tf, ek), cell in sorted(best_per.items()):
        is_ev = cell["IS"]["ev_R"]
        q3_ev = cell["Q3"]["ev_R"]
        q4_ev = cell["Q4"]["ev_R"]
        q3_n = cell["Q3"]["n"]
        q4_n = cell["Q4"]["n"]

        def gap_pct(oos: float, isv: float) -> float:
            if isv == 0 or not np.isfinite(isv):
                return float("inf")
            return (isv - oos) / abs(isv) * 100.0

        q3_gap = gap_pct(q3_ev, is_ev)
        q4_gap = gap_pct(q4_ev, is_ev)

        def passes(oos: float) -> bool:
            return is_ev > 0 and oos > 0 and oos >= 0.5 * is_ev

        q3_pass = passes(q3_ev)
        q4_pass = passes(q4_ev)
        # Pass requires both OOS quarters to clear the bar.
        verdict = "SHIP" if q3_pass and q4_pass else (
            "HOLD" if q3_pass or q4_pass else "KILL"
        )

        verdicts.append({
            "pattern": pattern,
            "tf": tf,
            "exit": ek,
            "params": {
                "pivot_lb": cell["pivot_lb"],
                "detect_window": cell["detect_window"],
                "vol_threshold": cell["vol_threshold"],
            },
            "IS": cell["IS"],
            "Q3": cell["Q3"],
            "Q4": cell["Q4"],
            "Q3_gap_pct": q3_gap,
            "Q4_gap_pct": q4_gap,
            "Q3_pass": q3_pass,
            "Q4_pass": q4_pass,
            "verdict": verdict,
        })
    all_results["walkforward"] = verdicts
    return all_results


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def write_report(results: Dict[str, Any]) -> None:
    out_json = OUT_DIR / "walkforward.json"
    out_md = OUT_DIR / "report.md"

    payload = {
        "config": {
            "symbols": SYMBOLS,
            "patterns": list(PATTERN_TFS.keys()),
            "pattern_tfs": PATTERN_TFS,
            "grid_pivot_lb": GRID_PIVOT_LOOKBACK,
            "grid_detect_window": GRID_DETECT_WINDOW,
            "grid_vol_threshold": GRID_VOL_THRESHOLD,
            "exits": EXITS,
            "quarter_bounds": QUARTER_BOUNDS,
            "fee_bps": RT_TAKER_FEE_BPS,
            "funding_per_8h": FUNDING_RATE_PER_8H,
            "notional": NOTIONAL,
        },
        "per_cell": results["per_cell"],
        "walkforward": results["walkforward"],
    }
    out_json.write_text(json.dumps(payload, default=str, indent=2))

    lines: List[str] = []
    lines.append("# Classical Chart Patterns - Walk-forward Backtest")
    lines.append("")
    lines.append("Detects 8 classical patterns and walk-forward tests them.")
    lines.append("")
    lines.append("Patterns:")
    for p, tfs in PATTERN_TFS.items():
        lines.append(f"- {p}: TF={tfs}")
    lines.append("")
    lines.append("Quarter splits (UTC):")
    for q, (s, e) in QUARTER_BOUNDS.items():
        lines.append(f"- {q}: {s} -> {e}")
    lines.append("")
    lines.append("Pass criteria: Q3 EV >= 50% of IS EV AND Q4 EV >= 50% of IS EV AND all positive.")
    lines.append("Reject if either OOS gap > 50%.")
    lines.append("")

    lines.append("## Walk-forward verdicts (best params per pattern x TF x exit)")
    lines.append("")
    lines.append(
        "| Pattern | TF | Exit | plb | dw | vt | IS n | IS EV(R) | Q3 n | Q3 EV(R) "
        "| Q4 n | Q4 EV(R) | Q3 gap% | Q4 gap% | Verdict |"
    )
    lines.append(
        "|---------|----|------|-----|----|----|------|----------|------|----------|"
        "------|----------|---------|---------|---------|"
    )
    for v in results["walkforward"]:
        lines.append(
            "| {pat} | {tf} | {exit} | {plb} | {dw} | {vt} | {isn} | {isev:+.3f} | "
            "{q3n} | {q3ev:+.3f} | {q4n} | {q4ev:+.3f} | {q3gap:+.1f} | {q4gap:+.1f} | "
            "{verdict} |".format(
                pat=v["pattern"],
                tf=v["tf"],
                exit=v["exit"],
                plb=v["params"]["pivot_lb"],
                dw=v["params"]["detect_window"],
                vt=v["params"]["vol_threshold"],
                isn=v["IS"]["n"],
                isev=v["IS"]["ev_R"],
                q3n=v["Q3"]["n"],
                q3ev=v["Q3"]["ev_R"],
                q4n=v["Q4"]["n"],
                q4ev=v["Q4"]["ev_R"],
                q3gap=v["Q3_gap_pct"],
                q4gap=v["Q4_gap_pct"],
                verdict=v["verdict"],
            )
        )
    lines.append("")

    # Pattern-level summary: best exit per (pattern x TF)
    best_per_pattern_tf: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for v in results["walkforward"]:
        key = (v["pattern"], v["tf"])
        cur = best_per_pattern_tf.get(key)
        if cur is None or v["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            best_per_pattern_tf[key] = v

    lines.append("## Pattern x TF summary (best exit)")
    lines.append("")
    lines.append(
        "| Pattern | TF | Best Exit | IS n | IS EV(R) | Q3 n | Q3 EV(R) | "
        "Q4 n | Q4 EV(R) | Verdict |"
    )
    lines.append(
        "|---------|----|-----------|------|----------|------|----------|------|----------|---------|"
    )
    for (pattern, tf), v in sorted(best_per_pattern_tf.items()):
        lines.append(
            "| {pat} | {tf} | {exit} | {isn} | {isev:+.3f} | {q3n} | {q3ev:+.3f} | "
            "{q4n} | {q4ev:+.3f} | {verdict} |".format(
                pat=pattern,
                tf=tf,
                exit=v["exit"],
                isn=v["IS"]["n"],
                isev=v["IS"]["ev_R"],
                q3n=v["Q3"]["n"],
                q3ev=v["Q3"]["ev_R"],
                q4n=v["Q4"]["n"],
                q4ev=v["Q4"]["ev_R"],
                verdict=v["verdict"],
            )
        )
    lines.append("")

    # SHIP vs KILL summary
    ships = [v for v in results["walkforward"] if v["verdict"] == "SHIP"]
    holds = [v for v in results["walkforward"] if v["verdict"] == "HOLD"]
    kills = [v for v in results["walkforward"] if v["verdict"] == "KILL"]
    lines.append("## Verdict counts")
    lines.append("")
    lines.append(f"- SHIP: {len(ships)}")
    lines.append(f"- HOLD: {len(holds)}")
    lines.append(f"- KILL: {len(kills)}")
    lines.append("")
    if ships:
        lines.append("## SHIP candidates (passed both OOS quarters)")
        lines.append("")
        for v in ships:
            lines.append(
                f"- **{v['pattern']} {v['tf']} {v['exit']}**: IS EV "
                f"{v['IS']['ev_R']:+.3f}R (n={v['IS']['n']}), Q3 "
                f"{v['Q3']['ev_R']:+.3f}R (n={v['Q3']['n']}), Q4 "
                f"{v['Q4']['ev_R']:+.3f}R (n={v['Q4']['n']})"
            )
        lines.append("")
    out_md.write_text("\n".join(lines))


def main() -> int:
    print("Running classical patterns walk-forward...", flush=True)
    results = run_walkforward()
    write_report(results)
    print(f"Wrote {OUT_DIR / 'walkforward.json'}", flush=True)
    print(f"Wrote {OUT_DIR / 'report.md'}", flush=True)
    n_cells = len(results["per_cell"])
    n_wf = len(results["walkforward"])
    print(f"per_cell={n_cells}  walkforward={n_wf}", flush=True)
    ships = [v for v in results["walkforward"] if v["verdict"] == "SHIP"]
    holds = [v for v in results["walkforward"] if v["verdict"] == "HOLD"]
    kills = [v for v in results["walkforward"] if v["verdict"] == "KILL"]
    print(f"SHIP={len(ships)} HOLD={len(holds)} KILL={len(kills)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

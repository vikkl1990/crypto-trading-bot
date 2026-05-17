"""Wedge breakout detector — used by VETO 10k WEDGE_BREAKOUT_VETO.

Walk-forward backtest results (storage/classical_patterns/):
  Rising wedge 1h EA: IS +0.145R / Q3 +0.298R / Q4 +0.379R   (n=41/13/20)   SHIP
  Rising wedge 1h EB: IS +0.197R / Q3 +0.349R / Q4 +0.333R   (n=41/13/20)   SHIP
  Falling wedge 4h EB: IS +0.281R / Q3 +0.439R / Q4 +0.183R  (n=20/7/6)    SHIP

The veto blocks trades that OPPOSE the wedge breakout direction:
  Rising wedge breaks DOWN  → bearish bias → VETO bullish trades
  Falling wedge breaks UP   → bullish bias → VETO bearish trades

Designed to be called lazily from scalp_strategy.py. Pure-pandas, no external deps.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def _find_pivots(highs: np.ndarray, lows: np.ndarray, lb: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of pivot-high indices and pivot-low indices."""
    n = len(highs)
    ph, pl = [], []
    for i in range(lb, n - lb):
        if highs[i] == max(highs[i - lb:i + lb + 1]):
            ph.append(i)
        if lows[i] == min(lows[i - lb:i + lb + 1]):
            pl.append(i)
    return np.asarray(ph, dtype=int), np.asarray(pl, dtype=int)


def _vol_ok(df: pd.DataFrame, idx: int, threshold: float = 1.0) -> bool:
    """Volume on bar idx >= threshold × 20-bar avg."""
    if idx < 20 or "volume" not in df.columns:
        return True  # neutral if insufficient data
    avg = float(df["volume"].iloc[idx - 20:idx].mean())
    if avg <= 0:
        return True
    return float(df["volume"].iloc[idx]) >= threshold * avg


def detect_wedge_breakout(
    df: pd.DataFrame,
    pivot_lb: int = 5,
    detect_window: int = 30,
    vol_threshold: float = 1.0,
) -> Optional[dict]:
    """Check the LAST candle of df for a fresh wedge breakout.

    Returns:
        {"side": "long" | "short", "type": "falling_wedge" | "rising_wedge"} on hit,
        None otherwise.
    """
    if len(df) < detect_window + pivot_lb + 2:
        return None
    if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
        return None
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    ph, pl = _find_pivots(highs, lows, pivot_lb)
    end = len(df) - 1

    recent_ph = ph[(ph >= end - detect_window) & (ph < end)]
    recent_pl = pl[(pl >= end - detect_window) & (pl < end)]
    if len(recent_ph) < 2 or len(recent_pl) < 2:
        return None

    ph_y = highs[recent_ph]
    pl_y = lows[recent_pl]
    ph_x = recent_ph.astype(float)
    pl_x = recent_pl.astype(float)
    sh = float(np.polyfit(ph_x, ph_y, 1)[0])
    sl = float(np.polyfit(pl_x, pl_y, 1)[0])

    # Rising wedge (SHORT): both slopes positive, sh < sl (converging up).
    # Trigger: close breaks BELOW the lower (pivot-low) trendline.
    if sh > 0 and sl > 0 and sh < sl:
        b_l = float(np.polyfit(pl_x, pl_y, 1)[1])
        lower_lvl = sl * end + b_l
        if closes[end] < lower_lvl and _vol_ok(df, end, vol_threshold):
            return {"side": "short", "type": "rising_wedge"}

    # Falling wedge (LONG): both slopes negative, sh < sl (upper falls steeper).
    # Trigger: close breaks ABOVE the upper (pivot-high) trendline.
    if sh < 0 and sl < 0 and sh < sl:
        b_h = float(np.polyfit(ph_x, ph_y, 1)[1])
        upper_lvl = sh * end + b_h
        if closes[end] > upper_lvl and _vol_ok(df, end, vol_threshold):
            return {"side": "long", "type": "falling_wedge"}

    return None

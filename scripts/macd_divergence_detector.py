"""MACD divergence detector.

Lightweight, dependency-free (numpy + pandas) module for spotting four
classic MACD-vs-price divergence patterns near the *last bar* of a candle
DataFrame.

Public API
----------

    detect_divergence(df: pd.DataFrame, lookback: int = 20, pivot_n: int = 2,
                      fast: int = 12, slow: int = 26, signal: int = 9) -> dict

Returned dict keys:

    div_type        "regular_bull" | "regular_bear" | "hidden_bull" |
                    "hidden_bear" | "none"
    div_strength    0.0 - 1.0  (higher = clearer divergence)
    price_pivot_idx integer bar index of the prior matching pivot
    macd_pivot_idx  same idx (we use price-pivot-aligned MACD reading)
    direction       +1 (bullish/long-favoring) | -1 (bearish/short-favoring) | 0
    candidate_idx   bar index of the *recent* pivot used in the comparison
    pivot_age       bars between the recent and prior pivot

How "the last candle" is treated
--------------------------------
A confirmed pivot needs `pivot_n` bars on EACH side. The last bar of `df`
therefore can never be a confirmed pivot — we don't yet have right-side
bars. We instead look at the **most recent confirmed pivot** (i.e. the
youngest bar `idx` for which `idx + pivot_n <= last_idx`) and compare it
to the prior pivot of the same kind. This is the standard ZigZag-pattern
treatment.

The *recent pivot* must lie within the last `pivot_n + 3` bars to count
as "happening at signal time" — older recent pivots return `none`. That
keeps the signal time-aligned with the trade open.

Divergence rules
----------------
At a swing-LOW pivot:
  Regular bull   = price LL & MACD HL  (reversal up)
  Hidden bull    = price HL & MACD LL  (continuation up)
At a swing-HIGH pivot:
  Regular bear   = price HH & MACD LH  (reversal down)
  Hidden bear    = price LH & MACD HH  (continuation down)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# core helpers                                                                 #
# --------------------------------------------------------------------------- #


def compute_macd_columns(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """Append `macd`, `macd_signal`, `macd_hist` columns; do not mutate input."""
    out = df.copy()
    ema_fast = out[column].ewm(span=fast, adjust=False).mean()
    ema_slow = out[column].ewm(span=slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def _is_pivot_high(highs: np.ndarray, idx: int, n: int) -> bool:
    if idx - n < 0 or idx + n >= len(highs):
        return False
    h = highs[idx]
    for k in range(1, n + 1):
        if highs[idx - k] >= h or highs[idx + k] >= h:
            return False
    return True


def _is_pivot_low(lows: np.ndarray, idx: int, n: int) -> bool:
    if idx - n < 0 or idx + n >= len(lows):
        return False
    lo = lows[idx]
    for k in range(1, n + 1):
        if lows[idx - k] <= lo or lows[idx + k] <= lo:
            return False
    return True


def _find_pivots(arr: np.ndarray, kind: str, start: int, stop: int, n: int) -> List[int]:
    """Find indices of pivots of `kind` ('high'|'low') in [start, stop)."""
    out: List[int] = []
    if kind == "high":
        for i in range(start, stop):
            if _is_pivot_high(arr, i, n):
                out.append(i)
    else:
        for i in range(start, stop):
            if _is_pivot_low(arr, i, n):
                out.append(i)
    return out


def _most_recent_pivot(
    arr: np.ndarray, kind: str, n: int, last_idx: int
) -> Optional[int]:
    """Return the *most recent* confirmed pivot index, scanning backward."""
    # Pivot needs `n` bars to right, so candidate idx <= last_idx - n
    for i in range(last_idx - n, n - 1, -1):
        if kind == "high" and _is_pivot_high(arr, i, n):
            return i
        if kind == "low" and _is_pivot_low(arr, i, n):
            return i
    return None


def _prior_pivot(
    arr: np.ndarray, kind: str, n: int, before_idx: int, lookback: int
) -> Optional[int]:
    """Return the most recent pivot before `before_idx`, within `lookback` bars."""
    start = max(n, before_idx - lookback)
    for i in range(before_idx - n, start - 1, -1):  # walk backward
        if kind == "high" and _is_pivot_high(arr, i, n):
            return i
        if kind == "low" and _is_pivot_low(arr, i, n):
            return i
    return None


# --------------------------------------------------------------------------- #
# main entry point                                                             #
# --------------------------------------------------------------------------- #

DEFAULT_RECENT_WINDOW = 6  # recent pivot must be within last (pivot_n + this) bars


def detect_divergence(
    df: pd.DataFrame,
    lookback: int = 20,
    pivot_n: int = 2,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> Dict[str, object]:
    """Detect MACD divergence near the last candle of `df`.

    Returns dict with: div_type, div_strength, price_pivot_idx,
    macd_pivot_idx, direction, candidate_idx, pivot_age.
    """
    out_default: Dict[str, object] = {
        "div_type": "none",
        "div_strength": 0.0,
        "price_pivot_idx": -1,
        "macd_pivot_idx": -1,
        "direction": 0,
        "candidate_idx": -1,
        "pivot_age": -1,
    }

    if df is None or len(df) < max(slow, lookback) + pivot_n + 3:
        return out_default

    work = compute_macd_columns(df, fast=fast, slow=slow, signal=signal)
    highs = work["high"].to_numpy()
    lows = work["low"].to_numpy()
    macd = work["macd"].to_numpy()
    n = len(work)
    last_idx = n - 1

    # find candidate (most recent confirmed) pivot of EACH kind
    cand_high = _most_recent_pivot(highs, "high", pivot_n, last_idx)
    cand_low = _most_recent_pivot(lows, "low", pivot_n, last_idx)

    # only count it if it's recent (within pivot_n + recent_window of last_idx)
    age_cap = pivot_n + recent_window
    if cand_high is not None and (last_idx - cand_high) > age_cap:
        cand_high = None
    if cand_low is not None and (last_idx - cand_low) > age_cap:
        cand_low = None

    # if both kinds exist, prefer whichever pivot is YOUNGEST (closest to last bar)
    candidate_idx = None
    cand_kind = None
    if cand_high is not None and cand_low is not None:
        if last_idx - cand_high <= last_idx - cand_low:
            candidate_idx, cand_kind = cand_high, "high"
        else:
            candidate_idx, cand_kind = cand_low, "low"
    elif cand_high is not None:
        candidate_idx, cand_kind = cand_high, "high"
    elif cand_low is not None:
        candidate_idx, cand_kind = cand_low, "low"
    else:
        return out_default

    # find prior pivot of the same kind
    prior_idx = _prior_pivot(
        highs if cand_kind == "high" else lows,
        cand_kind, pivot_n, candidate_idx, lookback,
    )
    if prior_idx is None:
        return out_default

    # extract values
    if cand_kind == "high":
        price_now = highs[candidate_idx]
        price_then = highs[prior_idx]
    else:
        price_now = lows[candidate_idx]
        price_then = lows[prior_idx]
    macd_now = macd[candidate_idx]
    macd_then = macd[prior_idx]

    # reference window for normalization (between prior and candidate, padded)
    ref_start = max(0, prior_idx - 5)
    ref_stop = min(n, candidate_idx + 5)
    p_ref = (highs if cand_kind == "high" else lows)[ref_start:ref_stop]
    m_ref = macd[ref_start:ref_stop]

    if cand_kind == "high":
        return _classify_high_pivot(
            candidate_idx, prior_idx, price_now, price_then, macd_now, macd_then,
            p_ref, m_ref, lookback,
        )
    else:
        return _classify_low_pivot(
            candidate_idx, prior_idx, price_now, price_then, macd_now, macd_then,
            p_ref, m_ref, lookback,
        )


def _classify_high_pivot(
    candidate_idx: int, prior_idx: int,
    price_now: float, price_then: float,
    macd_now: float, macd_then: float,
    price_ref: np.ndarray, macd_ref: np.ndarray, lookback: int,
) -> Dict[str, object]:
    """Classify divergence at a swing-high pivot."""
    price_higher = price_now > price_then
    macd_higher = macd_now > macd_then
    if price_higher and not macd_higher:
        div_type = "regular_bear"
        direction = -1
    elif (not price_higher) and macd_higher:
        div_type = "hidden_bear"
        direction = -1
    else:
        return {
            "div_type": "none",
            "div_strength": 0.0,
            "price_pivot_idx": int(prior_idx),
            "macd_pivot_idx": int(prior_idx),
            "direction": 0,
            "candidate_idx": int(candidate_idx),
            "pivot_age": int(candidate_idx - prior_idx),
        }
    strength = _strength_score(
        price_now, price_then, macd_now, macd_then,
        price_ref, macd_ref, candidate_idx, prior_idx, lookback,
    )
    return {
        "div_type": div_type,
        "div_strength": strength,
        "price_pivot_idx": int(prior_idx),
        "macd_pivot_idx": int(prior_idx),
        "direction": direction,
        "candidate_idx": int(candidate_idx),
        "pivot_age": int(candidate_idx - prior_idx),
    }


def _classify_low_pivot(
    candidate_idx: int, prior_idx: int,
    price_now: float, price_then: float,
    macd_now: float, macd_then: float,
    price_ref: np.ndarray, macd_ref: np.ndarray, lookback: int,
) -> Dict[str, object]:
    """Classify divergence at a swing-low pivot."""
    price_lower = price_now < price_then
    macd_lower = macd_now < macd_then
    if price_lower and not macd_lower:
        div_type = "regular_bull"
        direction = +1
    elif (not price_lower) and macd_lower:
        div_type = "hidden_bull"
        direction = +1
    else:
        return {
            "div_type": "none",
            "div_strength": 0.0,
            "price_pivot_idx": int(prior_idx),
            "macd_pivot_idx": int(prior_idx),
            "direction": 0,
            "candidate_idx": int(candidate_idx),
            "pivot_age": int(candidate_idx - prior_idx),
        }
    strength = _strength_score(
        price_now, price_then, macd_now, macd_then,
        price_ref, macd_ref, candidate_idx, prior_idx, lookback,
    )
    return {
        "div_type": div_type,
        "div_strength": strength,
        "price_pivot_idx": int(prior_idx),
        "macd_pivot_idx": int(prior_idx),
        "direction": direction,
        "candidate_idx": int(candidate_idx),
        "pivot_age": int(candidate_idx - prior_idx),
    }


def _strength_score(
    price_now: float, price_then: float,
    macd_now: float, macd_then: float,
    price_ref: np.ndarray, macd_ref: np.ndarray,
    candidate_idx: int, prior_idx: int, lookback: int,
) -> float:
    """Heuristic 0..1 strength score."""
    p_range = float(np.nanmax(price_ref) - np.nanmin(price_ref))
    if p_range <= 0:
        return 0.0
    p_dist = abs(price_now - price_then) / p_range  # 0..1

    m_range = float(np.nanmax(macd_ref) - np.nanmin(macd_ref))
    if m_range <= 0:
        return 0.0
    m_dist = abs(macd_now - macd_then) / m_range  # 0..1

    age = candidate_idx - prior_idx
    age_factor = max(0.5, 1.0 - 0.5 * max(0, age - lookback) / max(1, lookback))

    raw = 0.5 * p_dist + 0.5 * m_dist
    return float(max(0.0, min(1.0, raw * age_factor)))


# --------------------------------------------------------------------------- #
# self-test                                                                    #
# --------------------------------------------------------------------------- #


def _build_synthetic_lower_low_higher_macd_low(n: int = 80) -> pd.DataFrame:
    """Synthetic series: price LL with MACD HL (regular_bull)."""
    close = np.zeros(n)
    for i in range(n):
        if i < 25:
            close[i] = 100 - i * 1.0           # decline to bottom 1
        elif i < 45:
            close[i] = 75 + (i - 25) * 0.6     # bounce
        elif i < 70:
            close[i] = 87 - (i - 45) * 0.7     # decline to deeper bottom 2
        else:
            close[i] = 69 + (i - 70) * 0.5     # bounce
    df = pd.DataFrame({
        "open": close,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.ones(n),
    })
    return df


def _build_synthetic_higher_high_lower_macd(n: int = 80) -> pd.DataFrame:
    """Synthetic: price HH with MACD LH (regular_bear)."""
    close = np.zeros(n)
    for i in range(n):
        if i < 20:
            close[i] = 100 + i * 1.5           # rise to peak 1
        elif i < 35:
            close[i] = 130 - (i - 20) * 1.3    # pullback
        elif i < 60:
            close[i] = 110 + (i - 35) * 1.4    # rise to higher peak 2
        else:
            close[i] = 145 - (i - 60) * 0.5    # pullback
    df = pd.DataFrame({
        "open": close,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.ones(n),
    })
    return df


if __name__ == "__main__":
    print("== regular_bull synthetic ==")
    df = _build_synthetic_lower_low_higher_macd_low()
    res = detect_divergence(df, lookback=60, pivot_n=2)
    print(res)

    print("\n== regular_bear synthetic ==")
    df2 = _build_synthetic_higher_high_lower_macd()
    res2 = detect_divergence(df2, lookback=60, pivot_n=2)
    print(res2)

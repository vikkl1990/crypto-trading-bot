"""
PPP Phase 1 feature builder — 17 cheap, reconstructable features.

Per PPP_V2_IMPLEMENTATION_PLAN.md §4:
    All features computable from 25 bars of OHLCV + signal metadata.
    No L2 book, no tick velocity, no structure detector dependencies.

Reuses ml_training.feature_builder.compute_indicators() for VWAP/ATR/body_ratio
that are already battle-tested.

Usage:
    from ml_training.ppp_features import build_phase1_features

    features = build_phase1_features(
        candles_df=df_with_25_bars,
        symbol="BTC/USDT",
        side="long",
        emitted_at=datetime.utcnow(),
        grade="A+",
        ml_prob=0.85,
        confidence=90,
        regime="sideways",
        scanner_type="structure_bounce",
    )
    # Returns dict: {feature_name: float_value}
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from ml_training.feature_builder import compute_indicators

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SCANNER_TYPES = ("structure_bounce", "bb_squeeze", "meme_burst", "latency_arb")
REGIME_TYPES  = ("sideways", "breakout", "trending_up", "trending_down", "high_volatility")

# Grade → numeric mapping (higher = better)
GRADE_MAP = {"A+": 4.0, "A": 3.0, "B": 2.0, "C": 1.0}


def _onehot(value: str, options: tuple) -> Dict[str, float]:
    """Produce one-hot dict: {option: 1.0 if match else 0.0}."""
    v = (value or "").lower()
    return {f"is_{opt}": float(1.0 if v == opt else 0.0) for opt in options}


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def build_phase1_features(
    candles_df: pd.DataFrame,
    symbol: str,
    side: str,
    emitted_at: datetime,
    grade: Optional[str] = None,
    ml_prob: Optional[float] = None,
    confidence: Optional[int] = None,
    regime: Optional[str] = None,
    scanner_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Build PPP Phase 1 feature vector from candle history + signal metadata.

    Args:
        candles_df: pandas DataFrame with ≥20 rows of OHLCV.
                    Columns must include: open, high, low, close, volume.
                    Rows sorted ascending by time. Last row = current/signal bar.
        symbol: e.g. "BTC/USDT"
        side: "long" or "short"
        emitted_at: datetime of signal emission (used for time features)
        grade: "A+"/"A"/"B"/"C" or None
        ml_prob: existing ML score [0, 1] or None
        confidence: scanner confidence 0-100 or None
        regime: "sideways"/"breakout"/"trending_up"/"trending_down"/"high_volatility"
        scanner_type: one of SCANNER_TYPES

    Returns:
        dict of 17 features (all floats). Missing inputs → feature = 0.0.
        Safe to call with incomplete metadata; never raises.
    """
    if candles_df is None or len(candles_df) < 20:
        # Insufficient data — return neutral feature vector.
        # Caller should check and decide whether to persist.
        return _empty_feature_vector()

    # Run the shared indicator pipeline (adds VWAP, ATR, body_ratio, etc.)
    try:
        df = compute_indicators(candles_df.copy())
    except Exception:
        return _empty_feature_vector()

    if len(df) < 20:
        return _empty_feature_vector()

    last = df.iloc[-1]
    prev_3 = df.iloc[-4:-1]  # 3 bars before signal bar
    last_20 = df.iloc[-20:]

    # -------------------------------------------------------------------
    # Scalar extractors (safe)
    # -------------------------------------------------------------------
    open_ = _safe_float(last.get("open"))
    high  = _safe_float(last.get("high"))
    low   = _safe_float(last.get("low"))
    close = _safe_float(last.get("close"))
    volume = _safe_float(last.get("volume"))

    bar_range = max(high - low, 1e-9)  # avoid div0

    vwap  = _safe_float(last.get("vwap"))
    atr14 = _safe_float(last.get("atr_14"))

    # Body / wick structure (recompute cleanly from raw OHLC — don't trust mixed schema)
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    bar_body_to_range_ratio = body / bar_range
    bar_upper_wick_ratio   = upper_wick / bar_range
    bar_lower_wick_ratio   = lower_wick / bar_range

    # Volume vs 20-bar average
    vol_sma_20 = _safe_float(last_20["volume"].mean())
    volume_vs_20bar_avg = volume / vol_sma_20 if vol_sma_20 > 0 else 1.0

    # VWAP distance (bps)
    distance_from_vwap_bps = ((close - vwap) / vwap * 10000.0) if vwap > 0 else 0.0

    # 20-bar SMA distance (bps)
    sma_20 = _safe_float(last_20["close"].mean())
    distance_from_20bar_avg_bps = (
        (close - sma_20) / sma_20 * 10000.0 if sma_20 > 0 else 0.0
    )

    # ATR-normalized range (is this bar big relative to recent vol?)
    atr_normalized_range = bar_range / atr14 if atr14 > 0 else 0.0

    # Last 3-bar directional bias: sum of sign(close - open) for last 3 bars
    # Max value 3 (all green), min -3 (all red)
    last_3bar_direction = sum(
        (1 if r["close"] > r["open"] else (-1 if r["close"] < r["open"] else 0))
        for _, r in prev_3.iterrows()
    )

    # -------------------------------------------------------------------
    # Time features (cyclical encoding)
    # -------------------------------------------------------------------
    # Ensure emitted_at is timezone-aware
    if emitted_at.tzinfo is None:
        emitted_at = emitted_at.replace(tzinfo=timezone.utc)

    hour = emitted_at.hour + emitted_at.minute / 60.0
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    day_of_week = emitted_at.weekday()  # 0=Monday, 6=Sunday

    # -------------------------------------------------------------------
    # Metadata features
    # -------------------------------------------------------------------
    grade_numeric = GRADE_MAP.get((grade or "").upper(), 0.0)
    ml_probability = _safe_float(ml_prob)
    conf = float(confidence or 0)
    side_is_long = 1.0 if (side or "").lower() == "long" else 0.0

    # One-hot scanner + regime
    scanner_oh = _onehot(scanner_type or "", SCANNER_TYPES)
    regime_oh  = _onehot(regime or "", REGIME_TYPES)

    # -------------------------------------------------------------------
    # Assemble feature dict (order matches docs for readability)
    # -------------------------------------------------------------------
    features: Dict[str, float] = {
        # Metadata (6)
        "grade_numeric":        grade_numeric,
        "ml_probability":       ml_probability,
        "confidence":           conf,
        "side_is_long":         side_is_long,

        # Time (3)
        "hour_of_day_sin":      hour_sin,
        "hour_of_day_cos":      hour_cos,
        "day_of_week":          float(day_of_week),

        # Bar structure (6)
        "bar_body_to_range_ratio": bar_body_to_range_ratio,
        "bar_upper_wick_ratio":    bar_upper_wick_ratio,
        "bar_lower_wick_ratio":    bar_lower_wick_ratio,
        "volume_vs_20bar_avg":     volume_vs_20bar_avg,
        "atr_normalized_range":    atr_normalized_range,
        "last_3bar_direction":     float(last_3bar_direction),

        # Positional (2)
        "distance_from_vwap_bps":      distance_from_vwap_bps,
        "distance_from_20bar_avg_bps": distance_from_20bar_avg_bps,
    }

    # One-hot expansions (scanner: 4, regime: 5 → 9 total)
    features.update(scanner_oh)
    features.update(regime_oh)

    return features


def _empty_feature_vector() -> Dict[str, float]:
    """Return zero-filled vector matching the full schema. Used when inputs are insufficient."""
    features: Dict[str, float] = {
        "grade_numeric": 0.0,
        "ml_probability": 0.0,
        "confidence": 0.0,
        "side_is_long": 0.0,
        "hour_of_day_sin": 0.0,
        "hour_of_day_cos": 0.0,
        "day_of_week": 0.0,
        "bar_body_to_range_ratio": 0.0,
        "bar_upper_wick_ratio": 0.0,
        "bar_lower_wick_ratio": 0.0,
        "volume_vs_20bar_avg": 0.0,
        "atr_normalized_range": 0.0,
        "last_3bar_direction": 0.0,
        "distance_from_vwap_bps": 0.0,
        "distance_from_20bar_avg_bps": 0.0,
    }
    features.update({f"is_{s}": 0.0 for s in SCANNER_TYPES})
    features.update({f"is_{r}": 0.0 for r in REGIME_TYPES})
    return features


def feature_names() -> list:
    """Canonical ordered list of all Phase 1 feature names. Used by training/inference."""
    return list(_empty_feature_vector().keys())

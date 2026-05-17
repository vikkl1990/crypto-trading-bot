"""
Feature Builder — Polars Optimized Version
=============================================
Drop-in replacement for feature_builder.py using Polars for 5-10x speedup.
All functions accept pandas DataFrames, convert internally to Polars,
and return pandas DataFrames to maintain downstream compatibility.
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import Optional


def _pd_to_pl(df: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars, preserving datetime index as column."""
    pdf = df.copy()
    if not isinstance(pdf.index, pd.RangeIndex):
        pdf = pdf.reset_index()
    return pl.from_pandas(pdf)


def _safe_div(num: pl.Expr, denom: pl.Expr) -> pl.Expr:
    """Safe division replacing 0 denominators with null."""
    return num / pl.when(denom == 0).then(None).otherwise(denom)


def compute_indicators_polars(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base indicators needed for features and backtesting scanners.

    Accepts pandas DataFrame, returns pandas DataFrame.
    Internally uses Polars for speed.
    """
    # Detect index column name
    idx_col = df.index.name or "index"
    has_dt_index = hasattr(df.index, 'hour')

    pldf = _pd_to_pl(df)

    c = pl.col("close").cast(pl.Float64)
    h = pl.col("high").cast(pl.Float64)
    l = pl.col("low").cast(pl.Float64)
    o = pl.col("open").cast(pl.Float64)
    v = pl.col("volume").cast(pl.Float64)

    # ---- EMAs ----
    ema_exprs = []
    for period in [8, 13, 21, 50, 100, 200]:
        ema_exprs.append(
            c.ewm_mean(span=period, adjust=False).alias(f"ema_{period}")
        )
    pldf = pldf.with_columns(ema_exprs)

    # ---- True Range & ATR ----
    prev_close = c.shift(1)
    tr = pl.max_horizontal(
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    )
    pldf = pldf.with_columns([
        tr.rolling_mean(14).alias("atr_14"),
        tr.rolling_mean(7).alias("atr_7"),
    ])

    # ---- RSI ----
    delta = c.diff()
    gain = delta.clip(lower_bound=0).rolling_mean(14)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(14)
    rs = _safe_div(gain, loss)
    pldf = pldf.with_columns([
        (pl.lit(100) - (pl.lit(100) / (pl.lit(1) + rs))).alias("rsi_14"),
    ])

    # ---- Bollinger Bands ----
    sma20 = c.rolling_mean(20)
    std20 = c.rolling_std(20)
    pldf = pldf.with_columns([
        (sma20 + pl.lit(2) * std20).alias("bb_upper"),
        (sma20 - pl.lit(2) * std20).alias("bb_lower"),
    ])
    pldf = pldf.with_columns([
        _safe_div(pl.col("bb_upper") - pl.col("bb_lower"), sma20).alias("bb_width"),
    ])

    # ---- VWAP ----
    cum_vol = v.cum_sum()
    cum_vp = (c * v).cum_sum()
    pldf = pldf.with_columns([
        _safe_div(cum_vp, cum_vol).alias("vwap"),
    ])

    # ---- Volume rolling stats ----
    pldf = pldf.with_columns([
        v.rolling_mean(20).alias("vol_sma_20"),
        v.rolling_std(20).alias("vol_std_20"),
    ])
    pldf = pldf.with_columns([
        _safe_div(v, pl.col("vol_sma_20")).alias("rel_vol"),
    ])

    # ---- MACD ----
    ema12 = c.ewm_mean(span=12, adjust=False)
    ema26 = c.ewm_mean(span=26, adjust=False)
    pldf = pldf.with_columns([
        (ema12 - ema26).alias("macd"),
    ])
    pldf = pldf.with_columns([
        pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal"),
    ])
    pldf = pldf.with_columns([
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
    ])

    # ---- Ichimoku ----
    pldf = pldf.with_columns([
        ((h.rolling_max(9) + l.rolling_min(9)) / 2).alias("ichimoku_tenkan"),
        ((h.rolling_max(26) + l.rolling_min(26)) / 2).alias("ichimoku_kijun"),
    ])
    pldf = pldf.with_columns([
        ((pl.col("ichimoku_tenkan") + pl.col("ichimoku_kijun")) / 2).shift(26).alias("ichimoku_span_a"),
        ((h.rolling_max(52) + l.rolling_min(52)) / 2).shift(26).alias("ichimoku_span_b"),
    ])
    pldf = pldf.with_columns([
        pl.max_horizontal("ichimoku_span_a", "ichimoku_span_b").alias("ichimoku_cloud_top"),
        pl.min_horizontal("ichimoku_span_a", "ichimoku_span_b").alias("ichimoku_cloud_bottom"),
    ])

    # ---- Candle components ----
    pldf = pldf.with_columns([
        (c - o).abs().alias("body"),
        pl.when(h - l == 0).then(None).otherwise(h - l).alias("range"),
    ])
    pldf = pldf.with_columns([
        _safe_div(pl.col("body"), pl.col("range")).alias("body_ratio"),
        (h - pl.max_horizontal(c, o)).alias("upper_wick"),
        (pl.min_horizontal(c, o) - l).alias("lower_wick"),
        pl.when(c > o).then(1).otherwise(0).alias("is_bullish"),
    ])

    result = _pl_to_pd(pldf, idx_col)
    return result


def _pl_to_pd(pldf: pl.DataFrame, idx_col: str) -> pd.DataFrame:
    """Convert Polars DataFrame back to pandas, restoring index if possible."""
    pdf = pldf.to_pandas()
    if idx_col in pdf.columns:
        pdf = pdf.set_index(idx_col)
    elif "timestamp" in pdf.columns:
        pdf = pdf.set_index("timestamp")
    return pdf


def build_features_polars(df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build ML features from pure candle math — Polars optimized.

    Returns pandas DataFrame with ~130+ features for downstream compatibility.
    Instead of inserting columns one-by-one into a pandas DataFrame (causing
    fragmentation warnings), we build a dict of numpy arrays and create the
    DataFrame in a single shot at the end.
    """
    df = compute_indicators_polars(df)

    # Detect index info
    has_dt_index = hasattr(df.index, 'hour')

    # We build features as a dict of numpy arrays, then create a single
    # DataFrame at the end — avoids fragmentation entirely.
    c = df["close"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    o = df["open"].astype(float).values
    v = df["volume"].astype(float).values
    atr = df["atr_14"].values.astype(float)
    atr7 = df["atr_7"].values.astype(float)

    n = len(df)
    feats = {}

    # ---- Helpers (numpy + Polars rolling for speed) ----
    def sdiv(a, b):
        """Safe division: a / b with 0 denominator -> nan."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(b == 0, np.nan, a / b)
        return result.astype(np.float64)

    def pct_change(arr, periods=1):
        shifted = np.empty_like(arr, dtype=np.float64)
        shifted[:periods] = np.nan
        shifted[periods:] = arr[:-periods]
        return sdiv(arr - shifted, np.where(shifted == 0, np.nan, shifted))

    def shift(arr, periods=1, fill=np.nan):
        result = np.empty(len(arr), dtype=np.float64)
        if periods > 0:
            result[:periods] = fill
            result[periods:] = arr[:-periods]
        elif periods < 0:
            result[periods:] = fill
            result[:periods] = arr[-periods:]
        else:
            result[:] = arr
        return result

    def rolling_mean(arr, window):
        return pl.Series(arr).rolling_mean(window).to_numpy()

    def rolling_std(arr, window):
        return pl.Series(arr).rolling_std(window).to_numpy()

    def rolling_max(arr, window):
        return pl.Series(arr).rolling_max(window).to_numpy()

    def rolling_min(arr, window):
        return pl.Series(arr).rolling_min(window).to_numpy()

    def rolling_sum(arr, window):
        return pl.Series(arr).rolling_sum(window).to_numpy()

    def ewm_mean(arr, span):
        return pl.Series(arr).ewm_mean(span=span, adjust=False).to_numpy()

    def ffill(arr, limit=None):
        s = pl.Series(arr)
        if limit is not None:
            s = s.fill_null(strategy="forward", limit=limit)
        else:
            s = s.fill_null(strategy="forward")
        return s.to_numpy()

    # ================================================================
    # Precompute commonly used series
    # ================================================================
    ema8 = df["ema_8"].values.astype(float)
    ema21 = df["ema_21"].values.astype(float)
    ema50 = df["ema_50"].values.astype(float)
    ema200 = df["ema_200"].values.astype(float)
    body = df["body"].values.astype(float)
    rng = df["range"].values.astype(float)
    body_ratio = df["body_ratio"].values.astype(float)
    upper_wick = df["upper_wick"].values.astype(float)
    lower_wick = df["lower_wick"].values.astype(float)
    is_bullish = df["is_bullish"].values.astype(float)
    vwap = df["vwap"].values.astype(float)
    vol_sma_20 = df["vol_sma_20"].values.astype(float)
    vol_std_20 = df["vol_std_20"].values.astype(float)

    atr_safe = np.where(atr == 0, np.nan, atr)
    rng_safe = np.where(np.isnan(rng) | (rng == 0), np.nan, rng)

    # ================================================================
    # 1. MOMENTUM
    # ================================================================
    feats["return_1"] = pct_change(c, 1)
    feats["return_3"] = pct_change(c, 3)
    feats["return_5"] = pct_change(c, 5)
    feats["return_10"] = pct_change(c, 10)
    feats["return_20"] = pct_change(c, 20)
    feats["momentum_accel"] = feats["return_1"] - feats["return_5"]

    # ================================================================
    # 2. VOLATILITY
    # ================================================================
    feats["atr_ratio"] = sdiv(atr, c)
    feats["range_vs_atr"] = sdiv(rng, atr_safe)
    feats["atr_expansion"] = sdiv(atr7, atr_safe)
    atr_rm100 = rolling_mean(atr, 100)
    atr_rm100_safe = np.where((atr_rm100 == 0) | np.isnan(atr_rm100), np.nan, atr_rm100)
    feats["vol_regime"] = sdiv(atr, atr_rm100_safe)

    # ================================================================
    # 3. CANDLE STRUCTURE
    # ================================================================
    feats["body_ratio"] = body_ratio
    feats["upper_wick_ratio"] = sdiv(upper_wick, rng_safe)
    feats["lower_wick_ratio"] = sdiv(lower_wick, rng_safe)
    feats["body_displacement"] = sdiv(body, atr_safe)
    feats["close_position"] = sdiv(c - l, rng_safe)

    # ================================================================
    # 4. TREND STRENGTH
    # ================================================================
    feats["trend_strength"] = sdiv(ema8 - ema21, atr_safe)
    feats["trend_strength_long"] = sdiv(ema21 - ema50, atr_safe)
    feats["ema_slope_8"] = pct_change(ema8, 3)
    feats["ema_slope_21"] = pct_change(ema21, 5)
    feats["dist_from_ema21"] = sdiv(c - ema21, atr_safe)

    # ================================================================
    # 5. VOLUME INTELLIGENCE
    # ================================================================
    vol_std_safe = np.where((vol_std_20 == 0) | np.isnan(vol_std_20), np.nan, vol_std_20)
    vol_sma_safe = np.where((vol_sma_20 == 0) | np.isnan(vol_sma_20), np.nan, vol_sma_20)
    feats["volume_zscore"] = sdiv(v - vol_sma_20, vol_std_safe)
    feats["volume_spike"] = sdiv(v, vol_sma_safe)

    # ================================================================
    # 6. MARKET CONTEXT
    # ================================================================
    feats["dist_from_vwap"] = sdiv(c - vwap, atr_safe)
    rh20 = rolling_max(h, 20)
    rl20 = rolling_min(l, 20)
    rr20 = rh20 - rl20
    rr20_safe = np.where((rr20 == 0) | np.isnan(rr20), np.nan, rr20)
    feats["dist_from_high"] = sdiv(c - rh20, atr_safe)
    feats["dist_from_low"] = sdiv(c - rl20, atr_safe)
    feats["range_position"] = sdiv(c - rl20, rr20_safe)

    # ================================================================
    # 7. COMPRESSION / EXPANSION
    # ================================================================
    range_5 = rolling_mean(rng, 5)
    range_20 = rolling_mean(rng, 20)
    range_20_safe = np.where((range_20 == 0) | np.isnan(range_20), np.nan, range_20)
    feats["vol_compression"] = sdiv(range_5, range_20_safe)

    # ================================================================
    # 8. TIME FEATURES
    # ================================================================
    if has_dt_index:
        hour = df.index.hour
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feats["session"] = np.where(hour < 8, 0, np.where(hour < 16, 1, 2)).astype(float)
        feats["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    else:
        feats["hour_sin"] = np.zeros(n)
        feats["hour_cos"] = np.zeros(n)
        feats["session"] = np.ones(n)
        feats["dow_sin"] = np.zeros(n)

    # ================================================================
    # 9. STATE TRANSITION / DELTA FEATURES
    # ================================================================
    atr_s5 = shift(atr, 5)
    atr_s5_safe = np.where((atr_s5 == 0) | np.isnan(atr_s5), np.nan, atr_s5)
    atr_s3 = shift(atr, 3)
    atr_s3_safe = np.where((atr_s3 == 0) | np.isnan(atr_s3), np.nan, atr_s3)

    feats["trend_change"] = feats["trend_strength"] - sdiv(shift(ema8, 5) - shift(ema21, 5), atr_s5_safe)
    feats["trend_change_3"] = feats["trend_strength"] - sdiv(shift(ema8, 3) - shift(ema21, 3), atr_s3_safe)
    feats["trend_long_change"] = feats["trend_strength_long"] - sdiv(shift(ema21, 5) - shift(ema50, 5), atr_s5_safe)
    feats["vol_change"] = sdiv(atr, atr_s5_safe) - 1

    atr_ratio_prev = sdiv(atr_s3, np.where(rolling_mean(atr_s3, 100) == 0, np.nan, rolling_mean(atr_s3, 100)))
    feats["atr_ratio_change"] = feats["vol_regime"] - atr_ratio_prev

    vwap_s3 = shift(vwap, 3)
    c_s3 = shift(c, 3)
    vwap_dist_prev = sdiv(c_s3 - vwap_s3, atr_s3_safe)
    feats["vwap_dist_change"] = feats["dist_from_vwap"] - vwap_dist_prev
    feats["vwap_reversion_speed"] = feats["vwap_dist_change"].copy()

    v_rm5 = rolling_mean(v, 5)
    v_rm5_safe = np.where((v_rm5 == 0) | np.isnan(v_rm5), np.nan, v_rm5)
    feats["volume_change"] = sdiv(v, v_rm5_safe) - 1

    ret1_abs = np.abs(feats["return_1"])
    ret5_abs = np.abs(feats["return_5"])
    ret5_abs_safe = np.where((ret5_abs == 0) | np.isnan(ret5_abs), np.nan, ret5_abs)
    feats["impulse_decay"] = np.clip(sdiv(ret1_abs, ret5_abs_safe), 0, 5)

    range_3m = rolling_mean(rng, 3)
    range_10m = rolling_mean(rng, 10)
    range_10m_safe = np.where((range_10m == 0) | np.isnan(range_10m), np.nan, range_10m)
    feats["range_change"] = sdiv(range_3m, range_10m_safe)

    ema8_pct3 = pct_change(ema8, 3)
    feats["ema_slope_change"] = ema8_pct3 - shift(ema8_pct3, 3)

    # ================================================================
    # 9c. TRANSITION / DELTA FEATURES
    # ================================================================
    vwap_dist_now = sdiv(c - vwap, atr_safe)
    vwap_dist_3ago = sdiv(c_s3 - vwap_s3, atr_s3_safe)
    c_s5 = shift(c, 5)
    vwap_s5 = shift(vwap, 5)
    vwap_dist_5ago = sdiv(c_s5 - vwap_s5, atr_s5_safe)
    feats["delta_vwap_dist_3"] = vwap_dist_now - vwap_dist_3ago
    feats["delta_vwap_dist_5"] = vwap_dist_now - vwap_dist_5ago

    trend_now = sdiv(ema8 - ema21, atr_safe)
    trend_3ago = sdiv(shift(ema8, 3) - shift(ema21, 3), atr_s3_safe)
    trend_5ago = sdiv(shift(ema8, 5) - shift(ema21, 5), atr_s5_safe)
    feats["delta_trend_strength_3"] = trend_now - trend_3ago
    feats["delta_trend_strength_5"] = trend_now - trend_5ago

    atr_ratio_now = sdiv(atr, atr_rm100_safe)
    atr_s3_rm100 = rolling_mean(atr_s3, 100)
    atr_s3_rm100_safe = np.where((atr_s3_rm100 == 0) | np.isnan(atr_s3_rm100), np.nan, atr_s3_rm100)
    atr_s5_arr = shift(atr, 5)
    atr_s5_rm100 = rolling_mean(atr_s5_arr, 100)
    atr_s5_rm100_safe = np.where((atr_s5_rm100 == 0) | np.isnan(atr_s5_rm100), np.nan, atr_s5_rm100)
    feats["delta_atr_ratio_3"] = atr_ratio_now - sdiv(atr_s3, atr_s3_rm100_safe)
    feats["delta_atr_ratio_5"] = atr_ratio_now - sdiv(atr_s5_arr, atr_s5_rm100_safe)

    # impulse_decay (bars since last impulse candle)
    is_impulse_candle = (body > 0.8 * atr).astype(float)
    bars_since = np.full(n, 50.0)
    last_imp = -999
    for i in range(n):
        if is_impulse_candle[i] > 0:
            last_imp = i
        bars_since[i] = float(i - last_imp) if last_imp >= 0 else 50.0
    feats["transition_impulse_decay"] = np.clip(bars_since, 0, 50) / 50.0

    feats["transition_vwap_reversion_speed"] = (np.abs(vwap_dist_3ago) - np.abs(vwap_dist_now)) / 3.0

    return_5 = pct_change(c, 5)
    feats["delta_momentum_acceleration"] = return_5 - shift(return_5, 5)

    vol_zscore_now = sdiv(v - vol_sma_20, vol_std_safe)
    v_s5 = shift(v, 5)
    vsma_s5 = shift(vol_sma_20, 5)
    vstd_s5 = shift(vol_std_20, 5)
    vstd_s5_safe = np.where((vstd_s5 == 0) | np.isnan(vstd_s5), np.nan, vstd_s5)
    vol_zscore_5ago = sdiv(v_s5 - vsma_s5, vstd_s5_safe)
    feats["delta_vol_regime_change"] = vol_zscore_now - vol_zscore_5ago

    feats["delta_slope_change_ema8"] = ema8_pct3 - shift(ema8_pct3, 3)

    # ================================================================
    # 9b. RECENT BEHAVIOR MEMORY
    # ================================================================
    feats["last_3_return"] = pct_change(c, 3)
    feats["last_5_volatility"] = sdiv(rolling_mean(rng, 5), atr_safe)
    bullish_count = rolling_sum(is_bullish, 5)
    feats["trend_persistence"] = (bullish_count - 2.5) / 2.5

    # ================================================================
    # 10. DERIVED MULTI-TIMEFRAME
    # ================================================================
    feats["return_5bar"] = pct_change(c, 5)
    feats["atr_5bar"] = sdiv(rolling_mean(rng, 5), c)
    feats["return_15bar"] = pct_change(c, 15)
    feats["atr_15bar"] = sdiv(rolling_mean(rng, 15), c)
    feats["tf_agreement"] = (np.sign(ema8 - ema21) + np.sign(ema21 - ema50)) / 2.0

    # ================================================================
    # 11. FVG
    # ================================================================
    h_s2 = shift(h, 2)
    l_s2 = shift(l, 2)
    fvg_bull = (l > h_s2).astype(float)
    fvg_bull_size = np.clip(l - h_s2, 0, None) / atr_safe
    fvg_bear = (h < l_s2).astype(float)
    fvg_bear_size = np.clip(l_s2 - h, 0, None) / atr_safe

    feats["fvg_bullish"] = fvg_bull
    feats["fvg_bull_size"] = fvg_bull_size
    feats["fvg_bearish"] = fvg_bear
    feats["fvg_bear_size"] = fvg_bear_size
    feats["fvg_present"] = ((fvg_bull + fvg_bear) > 0).astype(float)
    feats["fvg_bull_recent_5"] = np.nan_to_num(rolling_max(fvg_bull, 5), 0)
    feats["fvg_bear_recent_5"] = np.nan_to_num(rolling_max(fvg_bear, 5), 0)
    feats["fvg_bull_recent_10"] = np.nan_to_num(rolling_max(fvg_bull, 10), 0)
    feats["fvg_bear_recent_10"] = np.nan_to_num(rolling_max(fvg_bear, 10), 0)
    feats["fvg_max_bull_size_10"] = np.nan_to_num(rolling_max(fvg_bull_size, 10), 0)
    feats["fvg_max_bear_size_10"] = np.nan_to_num(rolling_max(fvg_bear_size, 10), 0)
    feats["fvg_trend_aligned"] = (
        fvg_bull * (feats["trend_strength"] > 0).astype(float) +
        fvg_bear * (feats["trend_strength"] < 0).astype(float)
    )

    # ================================================================
    # 12. VOLUME INTELLIGENCE
    # ================================================================
    feats["buy_sell_imbalance"] = (feats["close_position"] - 0.5) * 2.0 * sdiv(v, vol_sma_safe)
    cvd_raw = (sdiv(c - l, rng_safe) - 0.5) * v
    vol_sma_10x = vol_sma_20 * 10
    vol_sma_10x_safe = np.where((vol_sma_10x == 0) | np.isnan(vol_sma_10x), np.nan, vol_sma_10x)
    feats["cvd_proxy_10"] = np.clip(sdiv(rolling_sum(cvd_raw, 10), vol_sma_10x_safe), -5, 5)
    v_rm3 = rolling_mean(v, 3)
    v_rm3_safe = np.where((v_rm3 == 0) | np.isnan(v_rm3), np.nan, v_rm3)
    feats["vol_spike_ratio_3"] = sdiv(v, v_rm3_safe)

    # ================================================================
    # 13. VWAP BANDS
    # ================================================================
    vwap_dev = rolling_std(c - vwap, 20)
    vwap_dev_safe = np.where((vwap_dev == 0) | np.isnan(vwap_dev), np.nan, vwap_dev)
    feats["vwap_band_distance"] = sdiv(c - vwap, vwap_dev_safe)
    feats["vwap_upper_band"] = sdiv(vwap + 2 * vwap_dev - c, atr_safe)
    feats["vwap_lower_band"] = sdiv(c - vwap + 2 * vwap_dev, atr_safe)

    # ================================================================
    # 14. ATR EXPANSION RATIO
    # ================================================================
    atr_s10 = shift(atr, 10)
    atr_s10_safe = np.where((atr_s10 == 0) | np.isnan(atr_s10), np.nan, atr_s10)
    feats["atr_expansion_10"] = sdiv(atr, atr_s10_safe)
    feats["atr_quiet"] = (feats["vol_regime"] < 0.5).astype(float)
    feats["atr_expanding"] = (feats["atr_expansion_10"] > 1.2).astype(float)
    body_avg_5 = rolling_mean(body_ratio, 5)
    feats["atr_chaotic"] = ((feats["vol_regime"] > 1.5) & (body_avg_5 < 0.4)).astype(float)

    # ================================================================
    # 15. EMA SLOPE ACCELERATION
    # ================================================================
    feats["ema_slope_accel"] = feats["ema_slope_8"] - shift(ema8_pct3, 3)

    # ================================================================
    # 16. MTF ALIGNMENT
    # ================================================================
    bull_ema = (
        (c > ema8).astype(int) +
        (ema8 > ema21).astype(int) +
        (ema21 > ema50).astype(int) +
        (ema50 > ema200).astype(int)
    )
    feats["ema_alignment"] = (bull_ema - 2) / 2.0
    feats["dist_from_ema200"] = sdiv(c - ema200, atr_safe)
    feats["htf_bias"] = np.zeros(n)
    feats["htf_trend_strength"] = np.zeros(n)

    # ================================================================
    # 17. FVG ENHANCED
    # ================================================================
    feats["fvg_size_atr"] = np.maximum(
        np.nan_to_num(fvg_bull_size, 0),
        np.nan_to_num(fvg_bear_size, 0)
    )
    feats["fvg_distance"] = np.full(n, 10.0)
    feats["fvg_alignment_score"] = np.zeros(n)

    # ================================================================
    # 18. ORDER BLOCK PROXY
    # ================================================================
    is_impulse = (body > 1.5 * atr).astype(float)
    impulse_bars_ago = np.zeros(n, dtype=np.float64)
    last_impulse = -999
    for i in range(n):
        if is_impulse[i] > 0:
            last_impulse = i
        impulse_bars_ago[i] = (i - last_impulse) / 10.0 if last_impulse >= 0 else 10.0
    feats["ob_distance"] = np.clip(impulse_bars_ago, 0, 10)
    feats["ob_impulse_strength"] = np.clip(sdiv(body, atr_safe), 0, 5)
    feats["ob_consolidation_size"] = np.clip(
        sdiv(rolling_max(h, 5) - rolling_min(l, 5), atr_safe), 0, 5
    )

    # ================================================================
    # 19. REGIME FEATURES
    # ================================================================
    feats["regime_trend_score"] = feats["ema_alignment"]
    feats["regime_vol_score"] = feats["vol_regime"]
    feats["regime_range_score"] = feats["range_position"]

    # ================================================================
    # 20. REGIME INTERACTION
    # ================================================================
    feats["trend_x_return5"] = feats["trend_strength"] * feats["return_5"]
    feats["trend_x_ema_slope"] = feats["trend_strength"] * feats["ema_slope_8"]
    feats["vol_x_volume"] = feats["atr_expansion"] * feats["volume_zscore"]
    feats["vwap_x_trend"] = feats["dist_from_vwap"] * feats["trend_strength"]
    feats["range_x_regime_vol"] = feats["range_position"] * feats["vol_regime"]

    # ================================================================
    # 21. ICHIMOKU CONTEXT
    # ================================================================
    cloud_top = df["ichimoku_cloud_top"].values.astype(float)
    cloud_bottom = df["ichimoku_cloud_bottom"].values.astype(float)
    tenkan = df["ichimoku_tenkan"].values.astype(float)
    kijun = df["ichimoku_kijun"].values.astype(float)

    feats["ichi_price_vs_cloud"] = np.where(
        c > cloud_top, 1.0, np.where(c < cloud_bottom, -1.0, 0.0)
    )
    feats["ichi_tk_alignment"] = np.sign(tenkan - kijun)
    cloud_thickness = np.abs(cloud_top - cloud_bottom)
    feats["ichi_cloud_thickness"] = sdiv(cloud_thickness, atr_safe)
    cloud_top_d3 = cloud_top - shift(cloud_top, 3)
    cloud_bot_d3 = cloud_bottom - shift(cloud_bottom, 3)
    feats["ichi_cloud_slope"] = sdiv(cloud_top_d3 + cloud_bot_d3, 2 * atr_safe)
    dist_above = sdiv(c - cloud_top, atr_safe)
    dist_below = sdiv(cloud_bottom - c, atr_safe)
    feats["ichi_dist_from_cloud"] = np.where(
        c > cloud_top, dist_above,
        np.where(c < cloud_bottom, -dist_below, 0.0)
    )
    feats["ichi_chikou_clearance"] = sdiv(c - shift(c, 26), atr_safe)

    # ================================================================
    # 22. FVG FEATURES (extended)
    # ================================================================
    bull_fvg_gap = l - h_s2
    bear_fvg_gap = l_s2 - h
    bull_fvg_exists = (bull_fvg_gap > 0).astype(float)
    bear_fvg_exists = (bear_fvg_gap > 0).astype(float)

    feats["fvg_bull_count_20"] = rolling_sum(bull_fvg_exists, 20)
    feats["fvg_bear_count_20"] = rolling_sum(bear_fvg_exists, 20)
    total_fvg = np.nan_to_num(feats["fvg_bull_count_20"], 0) + np.nan_to_num(feats["fvg_bear_count_20"], 0)
    feats["fvg_imbalance"] = np.where(
        total_fvg > 0,
        sdiv(
            np.nan_to_num(feats["fvg_bull_count_20"], 0) - np.nan_to_num(feats["fvg_bear_count_20"], 0),
            total_fvg
        ),
        0.0
    )

    bull_fvg_mid = np.where(bull_fvg_gap > 0, (h_s2 + l) / 2, np.nan)
    bull_fvg_mid = ffill(bull_fvg_mid, limit=10)
    bear_fvg_mid = np.where(bear_fvg_gap > 0, (l_s2 + h) / 2, np.nan)
    bear_fvg_mid = ffill(bear_fvg_mid, limit=10)

    feats["dist_from_bull_fvg"] = sdiv(c - bull_fvg_mid, atr_safe)
    feats["dist_from_bear_fvg"] = sdiv(bear_fvg_mid - c, atr_safe)

    # FVG recency (bars since last bull FVG)
    bull_fvg_bars_ago = np.zeros(n, dtype=float)
    last_fvg = -999
    for i in range(n):
        if bull_fvg_exists[i] > 0:
            last_fvg = i
            bull_fvg_bars_ago[i] = 0
        else:
            bull_fvg_bars_ago[i] = float(i - last_fvg) if last_fvg >= 0 else float(n)
    has_any_fvg = (np.nan_to_num(feats["fvg_bull_count_20"], 0) + np.nan_to_num(feats["fvg_bear_count_20"], 0)) > 0
    feats["fvg_recency"] = np.where(
        has_any_fvg,
        sdiv(np.ones(n), np.clip(bull_fvg_bars_ago, 1, None)),
        0.0
    )

    # ================================================================
    # 23. ORDER BLOCK PROXIMITY
    # ================================================================
    body_size = np.abs(c - o)
    body_atr_ratio_arr = sdiv(body_size, atr_safe)
    is_large_body = np.nan_to_num(body_atr_ratio_arr, 0) > 1.0
    is_bearish = c < o
    is_bullish_candle = c > o

    bull_ob_full = np.zeros(n, dtype=bool)
    bear_ob_full = np.zeros(n, dtype=bool)
    for i in range(1, n):
        bull_ob_full[i] = is_large_body[i-1] and is_bearish[i-1] and is_bullish_candle[i]
        bear_ob_full[i] = is_large_body[i-1] and is_bullish_candle[i-1] and is_bearish[i]

    bull_ob_level = np.where(bull_ob_full, (h + l) / 2, np.nan)
    bull_ob_level = ffill(bull_ob_level, limit=30)
    bear_ob_level = np.where(bear_ob_full, (h + l) / 2, np.nan)
    bear_ob_level = ffill(bear_ob_level, limit=30)

    feats["dist_from_bull_ob"] = sdiv(c - bull_ob_level, atr_safe)
    feats["dist_from_bear_ob"] = sdiv(bear_ob_level - c, atr_safe)
    feats["at_bull_ob"] = (np.abs(np.nan_to_num(feats["dist_from_bull_ob"], 999)) < 0.3).astype(float)
    feats["at_bear_ob"] = (np.abs(np.nan_to_num(feats["dist_from_bear_ob"], 999)) < 0.3).astype(float)
    feats["ob_count_20"] = (
        rolling_sum(bull_ob_full.astype(float), 20) +
        rolling_sum(bear_ob_full.astype(float), 20)
    )

    # ================================================================
    # 24. EMA 200 + VWAP CONTEXT
    # ================================================================
    feats["price_vs_ema200"] = sdiv(c - ema200, atr_safe)
    feats["price_vs_ema200_sign"] = np.sign(c - ema200)
    feats["ema200_slope"] = sdiv(ema200 - shift(ema200, 5), atr_safe)
    above_vwap = (c > vwap).astype(float)
    above_ema200 = (c > ema200).astype(float)
    feats["vwap_ema200_agreement"] = above_vwap + above_ema200 - 1
    feats["structural_alignment"] = np.where(
        (c > vwap) & (vwap > ema200), 1.0,
        np.where((c < vwap) & (vwap < ema200), -1.0, 0.0)
    )

    # ================================================================
    # 25. MULTI-TIMEFRAME ALIGNMENT
    # ================================================================
    if htf_df is not None and len(htf_df) > 0:
        htf_df = compute_indicators_polars(htf_df)
        htf_c = htf_df["close"].astype(float)
        htf_atr = htf_df["atr_14"]
        htf_ema8 = htf_df["ema_8"]
        htf_ema21 = htf_df["ema_21"]
        htf_ema50 = htf_df["ema_50"]
        htf_trend = np.where(
            (htf_ema8 > htf_ema21) & (htf_ema21 > htf_ema50), 1.0,
            np.where((htf_ema8 < htf_ema21) & (htf_ema21 < htf_ema50), -1.0, 0.0)
        )
        htf_trend_series = pd.Series(htf_trend, index=htf_df.index)
        htf_return5 = htf_c.pct_change(5)
        htf_atr_ratio = htf_df["atr_7"] / htf_atr.replace(0, np.nan)
        # Resample HTF features to LTF index (shift(1) = only completed bars)
        htf_trend_resampled = htf_trend_series.reindex(df.index, method="ffill").shift(1)
        htf_return_resampled = htf_return5.reindex(df.index, method="ffill").shift(1)
        htf_atr_resampled = htf_atr_ratio.reindex(df.index, method="ffill").shift(1)
        feats["htf_trend_bias"] = htf_trend_resampled.fillna(0.0).values
        feats["htf_momentum"] = htf_return_resampled.fillna(0.0).values
        feats["htf_vol_ratio"] = htf_atr_resampled.fillna(1.0).values
        feats["tf_alignment"] = feats["htf_trend_bias"] * feats["trend_strength"]
    else:
        feats["htf_trend_bias"] = np.zeros(n)
        feats["htf_momentum"] = np.zeros(n)
        feats["htf_vol_ratio"] = np.ones(n)
        feats["tf_alignment"] = np.zeros(n)

    # ================================================================
    # 26. LIQUIDITY SWEEP / GRAB DETECTION
    # ================================================================
    recent_highs_max = rolling_max(h, 20)
    recent_lows_min = rolling_min(l, 20)
    near_high = (sdiv(np.abs(np.nan_to_num(recent_highs_max, 0) - h), atr_safe) < 0.15)
    near_low = (sdiv(np.abs(l - np.nan_to_num(recent_lows_min, 0)), atr_safe) < 0.15)
    feats["equal_highs_count"] = rolling_sum(near_high.astype(float), 20)
    feats["equal_lows_count"] = rolling_sum(near_low.astype(float), 20)

    rhm_s1 = shift(recent_highs_max, 1)
    rlm_s1 = shift(recent_lows_min, 1)
    swept_high = (h > rhm_s1) & (c < rhm_s1)
    swept_low = (l < rlm_s1) & (c > rlm_s1)
    feats["sweep_high_size"] = np.where(swept_high, sdiv(h - rhm_s1, atr_safe), 0.0)
    feats["sweep_low_size"] = np.where(swept_low, sdiv(rlm_s1 - l, atr_safe), 0.0)
    feats["sweep_high_reversal"] = (swept_high & (c < o)).astype(float)
    feats["sweep_low_reversal"] = (swept_low & (c > o)).astype(float)
    feats["recent_sweep_bull"] = np.nan_to_num(rolling_max(feats["sweep_low_reversal"], 5), 0)
    feats["recent_sweep_bear"] = np.nan_to_num(rolling_max(feats["sweep_high_reversal"], 5), 0)

    sweep_into_fvg = np.where(
        (feats["sweep_low_reversal"] > 0) & (np.abs(np.nan_to_num(feats["dist_from_bull_fvg"], 999)) < 0.5),
        1.0,
        np.where(
            (feats["sweep_high_reversal"] > 0) & (np.abs(np.nan_to_num(feats["dist_from_bear_fvg"], 999)) < 0.5),
            1.0, 0.0
        )
    )
    feats["sweep_into_fvg"] = sweep_into_fvg

    body_vals = np.abs(c - o)
    post_sweep_disp = np.where(
        (feats["sweep_low_reversal"] > 0) | (feats["sweep_high_reversal"] > 0),
        sdiv(body_vals, atr_safe), 0.0
    )
    feats["post_sweep_displacement_atr"] = post_sweep_disp

    candle_range = np.where(h - l == 0, np.nan, h - l)
    reclaim_bull = sdiv(c - l, candle_range)
    reclaim_bear = sdiv(h - c, candle_range)
    feats["reclaim_strength"] = np.where(
        feats["sweep_low_reversal"] > 0, reclaim_bull,
        np.where(feats["sweep_high_reversal"] > 0, reclaim_bear, 0.0)
    )

    sweep_into_ob = np.where(
        (feats["sweep_low_reversal"] > 0) & (np.abs(np.nan_to_num(feats["dist_from_bull_ob"], 999)) < 0.5),
        1.0,
        np.where(
            (feats["sweep_high_reversal"] > 0) & (np.abs(np.nan_to_num(feats["dist_from_bear_ob"], 999)) < 0.5),
            1.0, 0.0
        )
    )
    feats["sweep_into_ob"] = sweep_into_ob

    # ================================================================
    # 27. BOS / CHOCH + DISPLACEMENT
    # ================================================================
    swing_high = rolling_max(h, 5)
    swing_low = rolling_min(l, 5)
    is_swing_high = (h == swing_high)
    is_swing_low = (l == swing_low)

    recent_swing_high = ffill(np.where(is_swing_high, h, np.nan))
    recent_swing_high = shift(recent_swing_high, 1)
    recent_swing_low = ffill(np.where(is_swing_low, l, np.nan))
    recent_swing_low = shift(recent_swing_low, 1)

    rsh_s1 = shift(recent_swing_high, 1)
    rsl_s1 = shift(recent_swing_low, 1)
    c_s1 = shift(c, 1)

    bullish_bos = (c > rsh_s1) & (c_s1 <= rsh_s1)
    bearish_bos = (c < rsl_s1) & (c_s1 >= rsl_s1)

    feats["bullish_bos"] = bullish_bos.astype(float)
    feats["bearish_bos"] = bearish_bos.astype(float)

    body_atr_bos = sdiv(np.abs(c - o), atr_safe)
    feats["bos_displacement"] = np.where(bullish_bos | bearish_bos, body_atr_bos, 0.0)
    feats["strong_displacement"] = (feats["bos_displacement"] > 1.5).astype(float)

    ema8_slope_10 = sdiv(ema8 - shift(ema8, 10), atr_safe)
    prior_trend_bull = ema8_slope_10 > 0.3
    prior_trend_bear = ema8_slope_10 < -0.3
    feats["choch_bull"] = (bullish_bos & prior_trend_bear).astype(float)
    feats["choch_bear"] = (bearish_bos & prior_trend_bull).astype(float)

    feats["recent_bos_bull"] = np.nan_to_num(rolling_max(feats["bullish_bos"], 3), 0)
    feats["recent_bos_bear"] = np.nan_to_num(rolling_max(feats["bearish_bos"], 3), 0)

    _any_bos = bullish_bos | bearish_bos
    _any_fvg = (bull_fvg_exists > 0) | (bear_fvg_exists > 0)
    feats["bos_with_fvg"] = np.where(_any_bos & _any_fvg, 1.0, 0.0)

    break_dist_bull = np.where(bullish_bos, sdiv(c - rsh_s1, atr_safe), 0.0)
    break_dist_bear = np.where(bearish_bos, sdiv(rsl_s1 - c, atr_safe), 0.0)
    feats["break_distance_atr"] = np.maximum(
        np.nan_to_num(break_dist_bull, 0) + np.nan_to_num(break_dist_bear, 0), 0
    )

    # Retest flag
    retest_bull = np.zeros(n, dtype=bool)
    retest_bear = np.zeros(n, dtype=bool)
    for lb in range(1, 4):
        bos_bull_shifted = shift(bullish_bos.astype(float), lb)
        bos_bear_shifted = shift(bearish_bos.astype(float), lb)
        rsh_shifted = shift(recent_swing_high, lb)
        rsl_shifted = shift(recent_swing_low, lb)
        retest_bull = retest_bull | (
            (bos_bull_shifted > 0) & (np.abs(l - rsh_shifted) < atr * 0.3)
        )
        retest_bear = retest_bear | (
            (bos_bear_shifted > 0) & (np.abs(h - rsl_shifted) < atr * 0.3)
        )
    feats["retest_flag"] = np.clip(
        retest_bull.astype(float) + retest_bear.astype(float), 0, 1.0
    )

    # BOS impulse decay
    bos_disp = feats["bos_displacement"]
    current_body_atr = sdiv(body_vals, atr_safe)
    bos_disp_safe = np.where((bos_disp == 0) | np.isnan(bos_disp), np.nan, bos_disp)
    impulse_decay_raw = np.where(
        bos_disp > 0.5,
        1.0 - sdiv(current_body_atr, bos_disp_safe),
        0.0
    )
    feats["bos_impulse_decay"] = np.clip(np.nan_to_num(impulse_decay_raw, 0), 0, 1.0)

    # HTF alignment with BOS
    htf_bias = feats["htf_trend_bias"]
    htf_bos_align = np.where(
        bullish_bos & (htf_bias > 0), 1.0,
        np.where(
            bearish_bos & (htf_bias < 0), 1.0,
            np.where(
                (feats["choch_bull"] > 0) | (feats["choch_bear"] > 0), -0.5, 0.0
            )
        )
    )
    feats["htf_bos_alignment"] = htf_bos_align

    # ================================================================
    # 28. KILLZONE / SESSION BOOST
    # ================================================================
    if has_dt_index:
        hour = df.index.hour
        minute = df.index.minute if hasattr(df.index, 'minute') else np.zeros(n)
        hour_frac = hour + minute / 60.0
        feats["kz_london"] = ((hour_frac >= 7.0) & (hour_frac < 10.0)).astype(float)
        feats["kz_newyork"] = ((hour_frac >= 13.0) & (hour_frac < 16.0)).astype(float)
        feats["kz_silver_bullet"] = (
            ((hour_frac >= 14.0) & (hour_frac < 15.0)) |
            ((hour_frac >= 19.0) & (hour_frac < 20.0))
        ).astype(float)
        feats["kz_dead_zone"] = ((hour_frac >= 0.0) & (hour_frac < 6.0)).astype(float)
        feats["kz_active"] = np.clip(
            feats["kz_london"] + feats["kz_newyork"] + feats["kz_silver_bullet"], 0, 1.0
        )
    else:
        feats["kz_london"] = np.zeros(n)
        feats["kz_newyork"] = np.zeros(n)
        feats["kz_silver_bullet"] = np.zeros(n)
        feats["kz_dead_zone"] = np.zeros(n)
        feats["kz_active"] = np.zeros(n)

    # ================================================================
    # 29. CONFLUENCE SCORE
    # ================================================================
    bull_signals = (
        feats["recent_sweep_bull"] +
        feats["recent_bos_bull"] +
        feats["at_bull_ob"] +
        (np.nan_to_num(feats["fvg_imbalance"], 0) > 0.3).astype(float) +
        (feats["structural_alignment"] > 0).astype(float) +
        feats["kz_active"]
    )
    bear_signals = (
        feats["recent_sweep_bear"] +
        feats["recent_bos_bear"] +
        feats["at_bear_ob"] +
        (np.nan_to_num(feats["fvg_imbalance"], 0) < -0.3).astype(float) +
        (feats["structural_alignment"] < 0).astype(float) +
        feats["kz_active"]
    )
    feats["confluence_bull"] = bull_signals
    feats["confluence_bear"] = bear_signals
    feats["confluence_max"] = np.maximum(bull_signals, bear_signals)

    # ================================================================
    # 30. Funding Rate Proxy (fixed bug: original used undefined 'closes')
    # ================================================================
    vwap_safe = np.where((vwap == 0) | np.isnan(vwap), np.nan, vwap)
    funding_proxy = sdiv(c - vwap, vwap_safe) * 100
    feats["funding_proxy"] = funding_proxy
    feats["funding_proxy_ma5"] = rolling_mean(funding_proxy, 5)
    feats["funding_positive"] = (np.nan_to_num(funding_proxy, 0) > 0).astype(float)

    # ================================================================
    # 31. Cross-Pair Correlation
    # ================================================================
    feats["momentum_ema_divergence"] = feats["return_5"] - feats["ema_slope_8"] * 10
    feats["trend_acceleration"] = feats["return_5"] - feats["return_10"] * 0.5

    feats["vol_regime_high"] = (np.nan_to_num(feats["atr_ratio"], 0) > 1.5).astype(float)
    feats["vol_regime_low"] = (np.nan_to_num(feats["atr_ratio"], 0) < 0.7).astype(float)

    # ================================================================
    # 32. Price Level Context
    # ================================================================
    last_close = c[-1] if n > 0 else 0
    if last_close > 100:  # BTC/ETH
        round_level = round(last_close / 1000) * 1000
        feats["dist_from_round_number"] = np.abs(c - round_level) / np.where(c == 0, np.nan, c) * 100
    elif last_close > 1:  # SOL/XRP
        round_level = round(last_close / 10) * 10
        feats["dist_from_round_number"] = np.abs(c - round_level) / np.where(c == 0, np.nan, c) * 100
    else:
        feats["dist_from_round_number"] = np.zeros(n)

    # ================================================================
    # CLEANUP: Build final DataFrame in one shot (no fragmentation!)
    # ================================================================
    # Replace inf with nan
    for k in feats:
        arr = feats[k]
        if isinstance(arr, np.ndarray):
            feats[k] = np.where(np.isinf(arr), np.nan, arr)

    # Build Polars DataFrame from dict (single allocation — no fragmentation)
    feat_series = {}
    for k, v_arr in feats.items():
        if isinstance(v_arr, np.ndarray):
            feat_series[k] = v_arr.astype(np.float64)
        else:
            feat_series[k] = np.full(n, float(v_arr), dtype=np.float64)

    feat_pl = pl.DataFrame(feat_series)

    # Add warmup flag
    warmup = np.zeros(n, dtype=np.float64)
    warmup[:200] = 1.0
    feat_pl = feat_pl.with_columns(pl.Series("is_warmup", warmup))

    # Fill nulls and NaNs with 0.0
    feat_pl = feat_pl.fill_null(0.0)
    feat_pl = feat_pl.fill_nan(0.0)

    # Convert to pandas with original index restored
    result = feat_pl.to_pandas()
    result.index = df.index

    return result


# ================================================================
# Convenience aliases for drop-in replacement
# ================================================================
compute_indicators = compute_indicators_polars
build_features = build_features_polars

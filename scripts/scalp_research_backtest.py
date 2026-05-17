#!/usr/bin/env python3
"""
Scalp Research Backtester (secondary engine candidates)
=======================================================

Goal
----
Exhaustively backtest a curated set of SCALP-class strategies against Delta India
specifics (taker fees, tick sizes, server-side stops, funding) on cached 5m/15m/1h/4h
candles for BTC/ETH/SOL/XRP. Output a scoreboard of strategy x exit x symbol cells
plus a top-K shortlist suitable for a parallel paper engine alongside S5.

This script is READ-ONLY against the live bot code. It only reads the cached parquet
candles and writes its outputs to ``storage/scalp_research/``.

Cost model (Delta India)
------------------------
- Taker fee per leg: 0.05% * 1.18 GST = 0.059%   -> 0.118% RT taker
- Maker fee per leg: 0.02% * 1.18 GST = 0.0236%  (only used for strategies that
  explicitly specify maker entry; the script then assumes maker entry + taker exit
  for safety -> 0.0826% RT)
- Funding: 0.01% per 8h period, applied if a trade is held across 00:00/08:00/16:00 UTC
- Slippage: 1 tick on entry (taker), 0 on stop-out (server-side stop fills at trigger)

Trade mechanics
---------------
- Position sizing: $1000 notional per trade, SL distance is strategy-specific in
  terms of ATR or tick units. R is computed from the actual SL distance:
      R_$ = (SL_distance_in_price / entry_price) * notional
  All P&L is reported in $ and as multiples of R.
- One trade per (symbol, strategy, exit) at a time.
- Same-bar SL/TP touch: SL wins (pessimistic).
- Time stop: closes at the bar that crosses the time limit, exit price = bar close.
- Trailing stops: peak MFE is tracked using bar high (long) / bar low (short);
  once MFE crosses the activation threshold, a hard stop is moved to lock the
  configured fraction of peak MFE. From there onward, SL behaves as a normal stop.

Strategies covered
------------------
14 candidates spanning microstructure, time-of-day, volatility, SMC, mean reversion,
trend continuation, and cross-asset (see README in report.md). MS1 (L2 depth
imbalance) is documented and KILL'd up front because we have no L2 in cache.

Author: Claude (VN Edge research)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDLE_DIR = os.path.join(REPO_ROOT, "storage", "candle_cache")
OUT_DIR = os.path.join(REPO_ROOT, "storage", "scalp_research")

SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT"]
TICK_SIZE = {
    "BTC_USDT": 0.5,
    "ETH_USDT": 0.01,
    "SOL_USDT": 0.001,
    "XRP_USDT": 0.0001,
}
NOTIONAL = 1000.0  # $ per trade

TAKER_FEE = 0.00059   # one leg, taker, GST inclusive
MAKER_FEE = 0.000236  # one leg, maker, GST inclusive
FUNDING_RATE = 0.0001  # per 8h period, midpoint estimate
FUNDING_HOURS_UTC = (0, 8, 16)


# --------------------------------------------------------------------------------------
# Data loading + indicator pre-compute
# --------------------------------------------------------------------------------------

def load_candles(symbol: str, tf: str) -> pd.DataFrame:
    path = os.path.join(CANDLE_DIR, f"{symbol}_{tf}.parquet")
    df = pd.read_parquet(path)
    df = df.copy()
    if "datetime" not in df.index.names and not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_index()
    # ensure tz-aware UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    return df


def add_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    # EMAs
    df["ema8"] = c.ewm(span=8, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    # ATR14
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["range"] = h - l
    # RSI14
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    # BB20
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    df["sma20"] = sma20
    df["std20"] = std20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    # Volume MA
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    # session-day VWAP (resets at 00:00 UTC)
    typical = (h + l + c) / 3.0
    pv = typical * v
    day = df.index.floor("D")
    grp = df.groupby(day, group_keys=False)
    df["vwap"] = grp.apply(lambda x: (pv.loc[x.index].cumsum() / v.loc[x.index].cumsum()))
    # 20-bar high/low (used for sweeps)
    df["hh20"] = h.rolling(20).max()
    df["ll20"] = l.rolling(20).min()
    df["hh20_prev"] = df["hh20"].shift(1)
    df["ll20_prev"] = df["ll20"].shift(1)
    # prev-bar 5m ATR (TOD1 reference)
    df["atr14_prev"] = df["atr14"].shift(1)
    return df


def add_indicators_15m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_width_pct100"] = df["bb_width"].rolling(100).rank(pct=True)
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["sma20"] = sma20
    return df


def add_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    # 1h trend label: ema21 vs ema50 + slope
    df["trend_up"] = (df["ema21"] > df["ema50"]).astype(int)
    df["trend_dn"] = (df["ema21"] < df["ema50"]).astype(int)
    # ranging proxy: |close - sma50| < 0.5 ATR over last 12 bars
    sma50 = c.rolling(50).mean()
    df["sma50"] = sma50
    df["ranging"] = ((c - sma50).abs() / df["atr14"] < 0.6).rolling(12).mean()
    df["ranging_flag"] = (df["ranging"] > 0.7).astype(int)
    # last 1h high/low for TOD2 momentum reference
    df["prev_1h_high"] = h.shift(1)
    df["prev_1h_low"] = l.shift(1)
    return df


def add_indicators_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    return df


def asof_join(df_low: pd.DataFrame, df_high: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Join high-TF context onto low-TF using merge_asof (no look-ahead)."""
    high = df_high.copy()
    high.columns = [f"{prefix}_{c}" for c in high.columns]
    high = high.reset_index().rename(columns={"datetime": "key"})
    low = df_low.reset_index().rename(columns={"datetime": "key"})
    out = pd.merge_asof(low, high, on="key", direction="backward")
    out = out.set_index("key").rename_axis("datetime")
    return out


def load_symbol_frame(symbol: str) -> pd.DataFrame:
    """Return a 5m-indexed DataFrame with HTF context joined."""
    d5 = add_indicators_5m(load_candles(symbol, "5m"))
    d15 = add_indicators_15m(load_candles(symbol, "15m"))
    d1h = add_indicators_1h(load_candles(symbol, "1h"))
    d4h = add_indicators_4h(load_candles(symbol, "4h"))
    out = asof_join(d5, d15, "h15")
    out = asof_join(out, d1h, "h1")
    out = asof_join(out, d4h, "h4")
    return out


# --------------------------------------------------------------------------------------
# Strategy signal definitions
#   Each strategy returns a DataFrame with columns: side (+1/-1/0), sl_dist, tp_dist
#   sl_dist and tp_dist are PRICE distances from entry (bar close + tick).
#   tp_dist is optional structural target (used by some strategies) but exit configs
#   may override; if a strategy has no structural TP we set NaN and rely on R-based
#   exit configs.
#   sl_dist must be > 0; entries with 0 / NaN SL are dropped.
# --------------------------------------------------------------------------------------

def _atr_sl(df: pd.DataFrame, mult: float) -> pd.Series:
    return df["atr14"] * mult


def _cross_edge(side_arr: np.ndarray) -> np.ndarray:
    """Convert a level-triggered side array (+1/-1/0) into a cross-edge one:
    only mark the first bar that enters a long-active or short-active state.
    This kills repeated firing while a condition is still True.
    """
    side = pd.Series(side_arr).fillna(0).astype(int).values
    prev = np.roll(side, 1)
    prev[0] = 0
    out = np.where((side != 0) & (side != prev), side, 0)
    return out.astype(int)


def sig_MS2(df: pd.DataFrame) -> pd.DataFrame:
    """Liquidity sweep + reclaim on 5m.

    Long: low < prior 20-bar low AND close > prior 20-bar low AND vol_ratio > 1.5.
    Short: high > prior 20-bar high AND close < prior 20-bar high AND vol_ratio > 1.5.
    SL = swept extreme + 0.2 ATR, TP = NaN (let exit cfg drive).
    """
    out = pd.DataFrame(index=df.index)
    long_cond = (df["low"] < df["ll20_prev"]) & (df["close"] > df["ll20_prev"]) & (df["vol_ratio"] > 1.5)
    short_cond = (df["high"] > df["hh20_prev"]) & (df["close"] < df["hh20_prev"]) & (df["vol_ratio"] > 1.5)
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    sl_dist = np.where(
        long_cond,
        (df["close"] - df["low"]) + 0.2 * df["atr14"],
        np.where(short_cond, (df["high"] - df["close"]) + 0.2 * df["atr14"], np.nan),
    )
    out["sl_dist"] = sl_dist
    out["tp_dist"] = np.nan
    return out


def sig_TOD1(df: pd.DataFrame) -> pd.DataFrame:
    """Asia open impulse fade (00:30-01:30 UTC).

    On the candle CLOSING at 00:35 UTC (i.e., the 5m bar 00:30->00:35), if range >
    1.5 * prev 5m ATR, fade with target = 50% retrace of that candle.
    Implemented: signal on bars where time == 00:35 and range > 1.5 * atr14_prev.
    """
    out = pd.DataFrame(index=df.index)
    is_window = (df.index.hour == 0) & (df.index.minute == 35)
    big = df["range"] > 1.5 * df["atr14_prev"]
    cond = is_window & big
    bull = df["close"] > df["open"]
    long_cond = cond & ~bull   # bull impulse -> fade short? we fade direction:
    # impulse bull = price up = fade short; impulse bear = fade long
    long_cond = cond & (~bull)
    short_cond = cond & bull
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    # SL = beyond impulse extreme + 0.2 ATR
    sl_dist = np.where(
        long_cond,
        (df["close"] - df["low"]) + 0.2 * df["atr14"],
        np.where(short_cond, (df["high"] - df["close"]) + 0.2 * df["atr14"], np.nan),
    )
    out["sl_dist"] = sl_dist
    out["tp_dist"] = np.nan
    return out


def sig_TOD2(df: pd.DataFrame) -> pd.DataFrame:
    """NYC open momentum 13:30-14:30 UTC.

    On 5m bars within window: close > prev_1h_high AND vol_ratio > 1.5 -> long.
    Symmetric short with prev_1h_low.
    SL = 1 ATR.
    """
    out = pd.DataFrame(index=df.index)
    # 13:30 -> 14:30 UTC inclusive
    hours = df.index.hour
    minutes = df.index.minute
    in_window = ((hours == 13) & (minutes >= 30)) | ((hours == 14) & (minutes <= 30))
    long_cond = in_window & (df["close"] > df["h1_prev_1h_high"]) & (df["vol_ratio"] > 1.5)
    short_cond = in_window & (df["close"] < df["h1_prev_1h_low"]) & (df["vol_ratio"] > 1.5)
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 1.0)
    out["tp_dist"] = np.nan
    return out


def sig_TOD3(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-funding squeeze.

    30 minutes before funding (00:30, 08:30, 16:30 UTC): if last 30min range
    contracted (current_range < 0.7 * prev 5m ATR) AND price > VWAP, take long;
    < VWAP take short. Direction = side of VWAP. SL = 0.8 ATR.
    """
    out = pd.DataFrame(index=df.index)
    pre_fund = ((df.index.hour.isin([0, 8, 16])) & (df.index.minute == 30))
    contracted = df["range"] < 0.7 * df["atr14_prev"]
    cond = pre_fund & contracted
    long_cond = cond & (df["close"] > df["vwap"])
    short_cond = cond & (df["close"] < df["vwap"])
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 0.8)
    out["tp_dist"] = np.nan
    return out


def sig_VOL1(df: pd.DataFrame) -> pd.DataFrame:
    """15m BB squeeze breakout, signal carried on 5m.

    15m bb_width_pct100 < 0.10 (bottom decile of last 100 bars) -> on the next 5m
    bar where price closes outside the 15m BB and vol_ratio > 1.5, take direction.
    """
    out = pd.DataFrame(index=df.index)
    squeeze = df["h15_bb_width_pct100"] < 0.10
    long_cond = squeeze & (df["close"] > df["h15_bb_upper"]) & (df["vol_ratio"] > 1.5)
    short_cond = squeeze & (df["close"] < df["h15_bb_lower"]) & (df["vol_ratio"] > 1.5)
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 1.0)
    out["tp_dist"] = np.nan
    return out


def sig_VOL2(df: pd.DataFrame) -> pd.DataFrame:
    """5m volatility expansion fade.

    If candle range > 2 * ATR14, fade direction with target 50% retrace; SL beyond
    the impulse extreme + 0.2 ATR.
    """
    out = pd.DataFrame(index=df.index)
    big = df["range"] > 2.0 * df["atr14"]
    bull = df["close"] > df["open"]
    long_cond = big & (~bull)
    short_cond = big & bull
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    sl_dist = np.where(
        long_cond,
        (df["close"] - df["low"]) + 0.2 * df["atr14"],
        np.where(short_cond, (df["high"] - df["close"]) + 0.2 * df["atr14"], np.nan),
    )
    out["sl_dist"] = sl_dist
    out["tp_dist"] = np.nan
    return out


def sig_SMC1(df: pd.DataFrame) -> pd.DataFrame:
    """1h Order Block + FVG retest -- approximated.

    Use 1h displacement: a 1h candle whose range > 1.5 * 1h ATR and that closes
    in the top/bottom 25% of its range. The opposite candle BEFORE displacement is
    the OB. We trigger when the 5m price retraces back into the OB zone in the
    direction of the displacement, with a reversal candle (bull/bear engulfing-ish).

    Approximation: we tag bars where on the most recent 1h displacement we are at
    50%-100% retrace of displacement and the current 5m candle is a reversal in
    the trend direction.
    """
    out = pd.DataFrame(index=df.index)
    # Without full bar-by-bar OB tracking we approximate: 1h close > prev 1h close
    # by > 1.5 ATR (displacement up); on subsequent bars retracing into the prior
    # 1h candle range, take long on a 5m bullish reversal (close > open AND high
    # > prev high).
    h1_disp_up = (df["h1_close"] - df["h1_close"].shift()) > 1.5 * df["h1_atr14"]
    h1_disp_dn = (df["h1_close"].shift() - df["h1_close"]) > 1.5 * df["h1_atr14"]
    # Retrace: 5m close <= 1h prior open (long case) but still > 1h prior low
    # (we approximate by: close < ema21 of 5m and close > ema50 of 5m for longs)
    long_retr = h1_disp_up & (df["close"] < df["ema21"]) & (df["close"] > df["ema50"])
    short_retr = h1_disp_dn & (df["close"] > df["ema21"]) & (df["close"] < df["ema50"])
    bull_rev = (df["close"] > df["open"]) & (df["high"] > df["high"].shift())
    bear_rev = (df["close"] < df["open"]) & (df["low"] < df["low"].shift())
    long_cond = long_retr & bull_rev
    short_cond = short_retr & bear_rev
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 1.0)
    out["tp_dist"] = np.nan
    return out


def sig_SMC2(df: pd.DataFrame) -> pd.DataFrame:
    """5m double bottom/top with vol confirmation.

    Two touches of a swing low within 0.3 ATR over the last 24 bars, vol on 2nd >
    1.5x. We approximate: current low is within 0.3 ATR of the rolling-min over
    last 24 bars (excluding current), and a prior bar within last 24 was also at
    that level, and current vol_ratio > 1.5, and current close > open.
    """
    out = pd.DataFrame(index=df.index)
    win = 24
    rmin = df["low"].rolling(win).min().shift(1)
    rmax = df["high"].rolling(win).max().shift(1)
    near_low = (df["low"] - rmin).abs() < 0.3 * df["atr14"]
    near_high = (df["high"] - rmax).abs() < 0.3 * df["atr14"]
    # prior touch existed: rolling min equals a prior low within tolerance
    long_cond = near_low & (df["vol_ratio"] > 1.5) & (df["close"] > df["open"])
    short_cond = near_high & (df["vol_ratio"] > 1.5) & (df["close"] < df["open"])
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    sl_dist = np.where(
        long_cond,
        (df["close"] - df["low"]) + 0.2 * df["atr14"],
        np.where(short_cond, (df["high"] - df["close"]) + 0.2 * df["atr14"], np.nan),
    )
    out["sl_dist"] = sl_dist
    out["tp_dist"] = np.nan
    return out


def sig_MR1(df: pd.DataFrame) -> pd.DataFrame:
    """5m Z-score reversion: (close - sma20) / std20 outside +/- 2.

    Long when z < -2, short when z > +2. SL = 1.2 ATR, TP = sma20 (mean).
    """
    out = pd.DataFrame(index=df.index)
    z = (df["close"] - df["sma20"]) / df["std20"].replace(0, np.nan)
    long_cond = z < -2.0
    short_cond = z > 2.0
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 1.2)
    # structural TP -> distance to sma20
    tp_dist = np.where(
        long_cond,
        (df["sma20"] - df["close"]).clip(lower=0),
        np.where(short_cond, (df["close"] - df["sma20"]).clip(lower=0), np.nan),
    )
    out["tp_dist"] = tp_dist
    return out


def sig_MR2(df: pd.DataFrame) -> pd.DataFrame:
    """5m VWAP mean reversion in 1h-ranging regime.

    1h ranging_flag == 1, |close - vwap| > 1 * atr14, AND reversal candle in
    direction of vwap.
    """
    out = pd.DataFrame(index=df.index)
    far = (df["close"] - df["vwap"]).abs() > df["atr14"]
    above = df["close"] > df["vwap"]
    below = df["close"] < df["vwap"]
    bear = df["close"] < df["open"]
    bull = df["close"] > df["open"]
    ranging = df["h1_ranging_flag"] == 1
    long_cond = ranging & far & below & bull
    short_cond = ranging & far & above & bear
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 1.0)
    tp_dist = np.where(
        long_cond,
        (df["vwap"] - df["close"]).clip(lower=0),
        np.where(short_cond, (df["close"] - df["vwap"]).clip(lower=0), np.nan),
    )
    out["tp_dist"] = tp_dist
    return out


def sig_TC1(df: pd.DataFrame) -> pd.DataFrame:
    """5m EMA21 pullback in 1h trend.

    1h trend_up=1 + 5m low touches ema21 (low <= ema21 <= high) + reversal candle.
    Symmetric short. SL = 1 ATR.
    """
    out = pd.DataFrame(index=df.index)
    touches = (df["low"] <= df["ema21"]) & (df["ema21"] <= df["high"])
    bull_rev = df["close"] > df["open"]
    bear_rev = df["close"] < df["open"]
    long_cond = (df["h1_trend_up"] == 1) & touches & bull_rev
    short_cond = (df["h1_trend_dn"] == 1) & touches & bear_rev
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 1.0)
    out["tp_dist"] = np.nan
    return out


def sig_TC2(df: pd.DataFrame) -> pd.DataFrame:
    """5m HH/HL structure ride.

    Approximate: in 1h uptrend, current 5m high > rolling-max(20) (new HH)
    AND current 5m low > rolling-min(5).shift(5) (HL holds). Symmetric short.
    """
    out = pd.DataFrame(index=df.index)
    new_hh = df["high"] > df["high"].rolling(20).max().shift(1)
    new_ll = df["low"] < df["low"].rolling(20).min().shift(1)
    hl_holds = df["low"] > df["low"].rolling(5).min().shift(5)
    lh_holds = df["high"] < df["high"].rolling(5).max().shift(5)
    long_cond = (df["h1_trend_up"] == 1) & new_hh & hl_holds
    short_cond = (df["h1_trend_dn"] == 1) & new_ll & lh_holds
    side = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    out["side"] = side
    out["sl_dist"] = _atr_sl(df, 1.0)
    out["tp_dist"] = np.nan
    return out


# CA1 needs joint pricing for BTC and ETH; handled separately at runtime.

STRATEGIES_SINGLE: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "MS2_sweep_reclaim": sig_MS2,
    "TOD1_asia_fade": sig_TOD1,
    "TOD2_nyc_momo": sig_TOD2,
    "TOD3_pre_funding": sig_TOD3,
    "VOL1_bb_squeeze_brk": sig_VOL1,
    "VOL2_expansion_fade": sig_VOL2,
    "SMC1_OB_FVG_retest": sig_SMC1,
    "SMC2_double_top_bot": sig_SMC2,
    "MR1_zscore_revert": sig_MR1,
    "MR2_vwap_revert": sig_MR2,
    "TC1_ema21_pullback": sig_TC1,
    "TC2_struct_ride": sig_TC2,
}

# MS1 (L2 depth imbalance) is excluded -- no L2 in cache.
DOCUMENTED_KILL = {
    "MS1_depth_imbalance": "No L2 / order-book data in cache; can't backtest reliably.",
}


# --------------------------------------------------------------------------------------
# Exit configurations
# --------------------------------------------------------------------------------------

@dataclass
class ExitCfg:
    name: str
    tp_R: Optional[float] = None
    sl_R: float = 1.0
    time_stop_min: int = 60
    trail_activate_R: Optional[float] = None
    trail_lock_frac: float = 0.0  # fraction of peak MFE locked when active
    use_struct_tp: bool = False  # use strategy-defined TP if available
    maker_entry: bool = False    # for MS1 if reactivated


EXIT_CONFIGS: Dict[str, ExitCfg] = {
    "EA_15R_1R_30m":   ExitCfg("EA", tp_R=1.5, sl_R=1.0, time_stop_min=30),
    "EB_2R_1R_60m":    ExitCfg("EB", tp_R=2.0, sl_R=1.0, time_stop_min=60),
    "EC_trail_05R_50pct_60m": ExitCfg("EC", tp_R=None, sl_R=1.0,
                                      time_stop_min=60, trail_activate_R=0.5,
                                      trail_lock_frac=0.5),
    "ED_trail_1R_33pct_4h":   ExitCfg("ED", tp_R=None, sl_R=1.0,
                                      time_stop_min=240, trail_activate_R=1.0,
                                      trail_lock_frac=0.33),
}


# --------------------------------------------------------------------------------------
# Trade simulator
# --------------------------------------------------------------------------------------

@dataclass
class Trade:
    sym: str
    strat: str
    exit: str
    side: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: Optional[float]
    R_dist: float    # price distance for 1R
    R_dollar: float  # $ value of 1R
    notional: float
    fee_in: float
    fee_out: float
    exit_idx: int = -1
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    raw_pnl_pct: float = 0.0
    fees_pct: float = 0.0
    funding_pct: float = 0.0
    hold_bars: int = 0
    hold_min: int = 0
    mfe_R: float = 0.0
    mae_R: float = 0.0


def _funding_cost_pct(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> float:
    """Sum of 0.01% applied at each 00/08/16 UTC mark crossed in (entry, exit]."""
    cur = entry_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    while cur <= exit_time and cur.hour not in FUNDING_HOURS_UTC:
        cur += pd.Timedelta(hours=1)
    n = 0
    while cur <= exit_time:
        if cur.hour in FUNDING_HOURS_UTC:
            n += 1
        cur += pd.Timedelta(hours=1)
    return n * FUNDING_RATE


def simulate(
    df: pd.DataFrame,
    sig: pd.DataFrame,
    sym: str,
    strat: str,
    exit_cfg: ExitCfg,
) -> List[Trade]:
    """Walk-forward simulation: at each signal, find next bar entry, then trade out."""
    trades: List[Trade] = []
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    times = df.index.values  # numpy datetime64
    side_arr = _cross_edge(sig["side"].values)
    sl_dist_arr = sig["sl_dist"].values
    tp_dist_arr = sig["tp_dist"].values
    tick = TICK_SIZE[sym]

    fee_per_leg_in = MAKER_FEE if exit_cfg.maker_entry else TAKER_FEE
    fee_per_leg_out = TAKER_FEE  # exit always taker (TP/SL/time-stop)

    i = 1  # start at 1 so we can reference i-1
    cooldown = 0
    while i < n - 1:
        if cooldown > 0:
            cooldown -= 1
            i += 1
            continue
        side = side_arr[i]
        if side == 0 or sl_dist_arr[i] <= 0 or not np.isfinite(sl_dist_arr[i]):
            i += 1
            continue
        # entry on next bar's OPEN approximated by current bar CLOSE + 1 tick slip
        entry_price = close[i] + tick * (1 if side == 1 else -1)
        sl_price = entry_price - side * sl_dist_arr[i]
        R_dist = abs(entry_price - sl_price)
        if R_dist <= 0:
            i += 1
            continue
        R_dollar = (R_dist / entry_price) * NOTIONAL
        # TP price
        tp_price = None
        struct_tp = None
        if np.isfinite(tp_dist_arr[i]) and tp_dist_arr[i] > 0:
            struct_tp = entry_price + side * tp_dist_arr[i]
        if exit_cfg.tp_R is not None:
            tp_price = entry_price + side * exit_cfg.tp_R * R_dist
        elif struct_tp is not None and exit_cfg.tp_R is None and exit_cfg.trail_activate_R is None:
            tp_price = struct_tp

        # walk forward
        j = i + 1
        peak_mfe_R = 0.0
        worst_mae_R = 0.0
        trail_active = False
        cur_sl = sl_price
        time_stop_idx = j
        # compute time-stop horizon in bars (5m bars)
        max_bars = max(1, int(round(exit_cfg.time_stop_min / 5)))
        end_idx = min(n - 1, j + max_bars)

        exit_idx = -1
        exit_price = 0.0
        exit_reason = ""

        while j <= end_idx:
            bh = high[j]
            bl = low[j]
            # update MFE/MAE
            if side == 1:
                up_R = (bh - entry_price) / R_dist
                dn_R = (entry_price - bl) / R_dist
            else:
                up_R = (entry_price - bl) / R_dist
                dn_R = (bh - entry_price) / R_dist
            peak_mfe_R = max(peak_mfe_R, up_R)
            worst_mae_R = max(worst_mae_R, dn_R)

            # activate trailing stop?
            if exit_cfg.trail_activate_R is not None and not trail_active and peak_mfe_R >= exit_cfg.trail_activate_R:
                trail_active = True
            if trail_active and peak_mfe_R > 0:
                lock_R = exit_cfg.trail_lock_frac * peak_mfe_R
                # trailing stop is at entry + side * lock_R * R_dist
                trail_price = entry_price + side * lock_R * R_dist
                # only ratchet in favor (long: stop only goes up; short: stop only goes down)
                if side == 1:
                    cur_sl = max(cur_sl, trail_price)
                else:
                    cur_sl = min(cur_sl, trail_price)

            # check stop and TP within this bar
            sl_hit = (side == 1 and bl <= cur_sl) or (side == -1 and bh >= cur_sl)
            tp_hit = False
            if tp_price is not None:
                tp_hit = (side == 1 and bh >= tp_price) or (side == -1 and bl <= tp_price)

            if sl_hit and tp_hit:
                # pessimistic: SL wins
                exit_idx = j
                exit_price = cur_sl
                exit_reason = "sl_with_tp_same_bar"
                break
            elif sl_hit:
                exit_idx = j
                exit_price = cur_sl
                exit_reason = "sl" if not trail_active else "trail"
                break
            elif tp_hit:
                exit_idx = j
                exit_price = tp_price
                exit_reason = "tp"
                break
            j += 1

        if exit_idx == -1:
            # time stop
            exit_idx = end_idx
            exit_price = close[end_idx]
            exit_reason = "time"

        # P&L
        raw_pnl_pct = side * (exit_price - entry_price) / entry_price
        fees_pct = fee_per_leg_in + fee_per_leg_out
        funding_pct = 0.0
        et = pd.Timestamp(times[i])
        xt = pd.Timestamp(times[exit_idx])
        if (xt - et) >= pd.Timedelta(hours=8):
            funding_pct = _funding_cost_pct(et, xt)

        trade = Trade(
            sym=sym,
            strat=strat,
            exit=exit_cfg.name,
            side=int(side),
            entry_idx=i,
            entry_time=et,
            entry_price=float(entry_price),
            sl=float(sl_price),
            tp=float(tp_price) if tp_price is not None else None,
            R_dist=float(R_dist),
            R_dollar=float(R_dollar),
            notional=NOTIONAL,
            fee_in=fee_per_leg_in,
            fee_out=fee_per_leg_out,
            exit_idx=int(exit_idx),
            exit_time=xt,
            exit_price=float(exit_price),
            exit_reason=exit_reason,
            raw_pnl_pct=float(raw_pnl_pct),
            fees_pct=float(fees_pct),
            funding_pct=float(funding_pct),
            hold_bars=int(exit_idx - i),
            hold_min=int((exit_idx - i) * 5),
            mfe_R=float(peak_mfe_R),
            mae_R=float(worst_mae_R),
        )
        trades.append(trade)

        # 1-bar cooldown after each exit (avoid back-to-back signals on same setup)
        cooldown = 1
        i = exit_idx + 1

    return trades


# --------------------------------------------------------------------------------------
# Cross-asset CA1: BTC/ETH ratio reversion
# --------------------------------------------------------------------------------------

def run_CA1(symbol_data: Dict[str, pd.DataFrame], exit_cfgs: Dict[str, ExitCfg]) -> List[Trade]:
    """When BTC/ETH ratio is 2 sigma from 20-bar mean, take convergence on the
    leg that overshot. We simulate two trades per signal (one per leg) and tag
    them under symbol-specific rows so the scoreboard handles them naturally."""
    if "BTC_USDT" not in symbol_data or "ETH_USDT" not in symbol_data:
        return []
    btc = symbol_data["BTC_USDT"][["close", "high", "low", "open", "atr14"]].rename(columns=lambda c: f"btc_{c}")
    eth = symbol_data["ETH_USDT"][["close", "high", "low", "open", "atr14"]].rename(columns=lambda c: f"eth_{c}")
    df = btc.join(eth, how="inner").dropna()
    df["ratio"] = df["btc_close"] / df["eth_close"]
    df["ratio_mean"] = df["ratio"].rolling(20).mean()
    df["ratio_std"] = df["ratio"].rolling(20).std(ddof=0)
    df["z"] = (df["ratio"] - df["ratio_mean"]) / df["ratio_std"].replace(0, np.nan)
    long_btc = df["z"] < -2
    short_btc = df["z"] > 2
    # When BTC is cheap (z<-2): long BTC, short ETH. When BTC rich (z>2): short BTC, long ETH.
    out = []
    for sym, leg_long_when in [("BTC_USDT", long_btc), ("ETH_USDT", short_btc)]:
        sig = pd.DataFrame(index=df.index)
        side = np.where(long_btc & (sym == "BTC_USDT"), 1,
              np.where(short_btc & (sym == "BTC_USDT"), -1,
              np.where(long_btc & (sym == "ETH_USDT"), -1,
              np.where(short_btc & (sym == "ETH_USDT"), 1, 0))))
        sig["side"] = side
        sig["sl_dist"] = df[f"{sym.split('_')[0].lower()}_atr14"] * 1.2
        sig["tp_dist"] = np.nan
        # align with the symbol's 5m frame to use the simulator
        full_df = symbol_data[sym]
        sig_full = sig.reindex(full_df.index)
        for ex_name, ex_cfg in exit_cfgs.items():
            trades = simulate(full_df, sig_full, sym, "CA1_btc_eth_revert", ex_cfg)
            out.extend(trades)
    return out


# --------------------------------------------------------------------------------------
# Top-level run
# --------------------------------------------------------------------------------------

def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        net_pct = t.raw_pnl_pct - t.fees_pct - t.funding_pct
        gross_R = t.raw_pnl_pct / (t.R_dist / t.entry_price)
        net_R = net_pct / (t.R_dist / t.entry_price)
        gross_dollar = t.raw_pnl_pct * t.notional
        net_dollar = net_pct * t.notional
        rows.append({
            "sym": t.sym,
            "strat": t.strat,
            "exit": t.exit,
            "side": t.side,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "sl": t.sl,
            "tp": t.tp,
            "exit_reason": t.exit_reason,
            "hold_min": t.hold_min,
            "R_dist": t.R_dist,
            "R_dollar": t.R_dollar,
            "raw_pnl_pct": t.raw_pnl_pct,
            "fees_pct": t.fees_pct,
            "funding_pct": t.funding_pct,
            "net_pct": net_pct,
            "gross_R": gross_R,
            "net_R": net_R,
            "gross_dollar": gross_dollar,
            "net_dollar": net_dollar,
            "mfe_R": t.mfe_R,
            "mae_R": t.mae_R,
        })
    return pd.DataFrame(rows)


def aggregate_scoreboard(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    g = trades_df.groupby(["strat", "exit", "sym"])
    agg = g.agg(
        n=("net_dollar", "size"),
        wr=("net_dollar", lambda s: float((s > 0).mean())),
        avg_mfe_R=("mfe_R", "mean"),
        avg_mae_R=("mae_R", "mean"),
        avg_hold_min=("hold_min", "mean"),
        gross_R=("gross_R", "sum"),
        net_R=("net_R", "sum"),
        gross_dollar=("gross_dollar", "sum"),
        net_dollar=("net_dollar", "sum"),
        ev_per_trade_dollar=("net_dollar", "mean"),
        net_pct_mean=("net_pct", "mean"),
    ).reset_index()
    # max DD: cumulative net_dollar within group (as ordered by entry_time)
    dd_rows = []
    for (strat, ex, sym), grp in trades_df.sort_values("entry_time").groupby(["strat", "exit", "sym"]):
        cum = grp["net_dollar"].cumsum()
        if len(cum) > 0:
            running_max = cum.cummax()
            dd = (cum - running_max).min()
        else:
            dd = 0.0
        dd_rows.append({"strat": strat, "exit": ex, "sym": sym, "max_dd_dollar": float(dd)})
    dd_df = pd.DataFrame(dd_rows)
    out = agg.merge(dd_df, on=["strat", "exit", "sym"], how="left")
    return out.sort_values("net_dollar", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Subset of strategy keys to run (default: all)")
    parser.add_argument("--exits", nargs="+", default=None,
                        help="Subset of exit keys to run (default: all)")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: tiny window")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load all symbol frames once
    print(f"[{datetime.utcnow():%H:%M:%S}] Loading + indicating {len(args.symbols)} symbols...", flush=True)
    sym_data: Dict[str, pd.DataFrame] = {}
    for s in args.symbols:
        sym_data[s] = load_symbol_frame(s)
        print(f"  {s}: rows={len(sym_data[s])}, span={sym_data[s].index.min()} -> {sym_data[s].index.max()}", flush=True)

    if args.smoke:
        for s in sym_data:
            sym_data[s] = sym_data[s].iloc[:5000]

    # Pick strategies
    strat_keys = list(STRATEGIES_SINGLE.keys())
    if args.strategies:
        strat_keys = [k for k in strat_keys if k in args.strategies]
    exit_keys = list(EXIT_CONFIGS.keys())
    if args.exits:
        exit_keys = [k for k in exit_keys if k in args.exits]

    all_trades: List[Trade] = []
    for sym, df in sym_data.items():
        for sk in strat_keys:
            print(f"[{datetime.utcnow():%H:%M:%S}] {sym} {sk} ...", flush=True)
            sig_df = STRATEGIES_SINGLE[sk](df)
            n_sig = int((sig_df["side"] != 0).sum())
            for ek in exit_keys:
                ex_cfg = EXIT_CONFIGS[ek]
                trades = simulate(df, sig_df, sym, sk, ex_cfg)
                all_trades.extend(trades)
            print(f"   raw_signals={n_sig}, total trades after exits={sum(1 for t in all_trades if t.sym==sym and t.strat==sk)}", flush=True)

    # CA1 cross-asset (only if both BTC and ETH selected)
    if "BTC_USDT" in sym_data and "ETH_USDT" in sym_data and (
        not args.strategies or "CA1_btc_eth_revert" in args.strategies):
        print(f"[{datetime.utcnow():%H:%M:%S}] CA1 BTC/ETH ratio reversion ...", flush=True)
        ca1_trades = run_CA1(sym_data, {ek: EXIT_CONFIGS[ek] for ek in exit_keys})
        all_trades.extend(ca1_trades)

    print(f"[{datetime.utcnow():%H:%M:%S}] Total trades: {len(all_trades)}", flush=True)
    tdf = trades_to_df(all_trades)
    if not tdf.empty:
        tdf.to_csv(os.path.join(args.out, "trades.csv"), index=False)
        scoreboard = aggregate_scoreboard(tdf)
        scoreboard.to_csv(os.path.join(args.out, "scoreboard.csv"), index=False)
        print(f"[{datetime.utcnow():%H:%M:%S}] Wrote scoreboard.csv ({len(scoreboard)} rows)", flush=True)
        # Top 20
        top20 = scoreboard.sort_values("net_dollar", ascending=False).head(20)
        top20.to_csv(os.path.join(args.out, "top20.csv"), index=False)
        # Per-strategy aggregate (sum across symbols/exits then we can rank)
        strat_agg = (
            scoreboard.groupby("strat")
            .agg(n=("n", "sum"),
                 net_dollar=("net_dollar", "sum"),
                 gross_dollar=("gross_dollar", "sum"),
                 wr_w=("wr", "mean"),
                 ev_avg=("ev_per_trade_dollar", "mean"),
                 max_dd_dollar=("max_dd_dollar", "min"))
            .reset_index()
            .sort_values("net_dollar", ascending=False)
        )
        strat_agg.to_csv(os.path.join(args.out, "by_strategy.csv"), index=False)
        # Per-strategy x exit rollup (across symbols)
        sx = (
            scoreboard.groupby(["strat", "exit"])
            .agg(n=("n", "sum"),
                 net_dollar=("net_dollar", "sum"),
                 wr_w=("wr", "mean"),
                 ev_avg=("ev_per_trade_dollar", "mean"))
            .reset_index()
            .sort_values("net_dollar", ascending=False)
        )
        sx.to_csv(os.path.join(args.out, "by_strategy_exit.csv"), index=False)
    print(f"[{datetime.utcnow():%H:%M:%S}] Done.")


if __name__ == "__main__":
    main()

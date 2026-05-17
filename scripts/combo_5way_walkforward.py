#!/usr/bin/env python3
"""
5-way combo strategy walk-forward backtest.

Tests the architect's hypothesis that confluence stacks of HOLD-class signals
can produce real edge. Five candidate combos:

  COMBO 1 — Exhaustion Confluence Stack (MACD div + RSI div + Volume Climax 1h)
  COMBO 2 — Wedge x Regime Filtered (wedge breakout + HA bias 4h alignment)
  COMBO 3 — Double Pattern x HTF Confluence (double top/bottom + EMA21/50 1h)
  COMBO 4 — Regime Double-Confirm (HA bias 4h + EMA200_d as veto on engulfings)
  COMBO 5 — Fragile-Edge Portfolio (equal-weight aggregate of 5 component classes)

Walk-forward boundaries:
  Q1 (Oct-Nov 2025) and Q2 (Dec 2025 - Jan 2026)  = IS
  Q3 (Feb 2026)                                    = OOS Q3
  Q4 (Mar-Apr 2026)                                = OOS Q4
Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive); n>=30 IS, n>=20 OOS.

Outputs:
    storage/combo_5way/walkforward.json
    storage/combo_5way/report.md

Read-only on bot/signal_tracker.py, bot/signal_journey.py, bot/signal_learner.py,
and on all paper-trading engines (S5, SMC1, SMC1.5, SMC1.5v2). Does not modify
scalp_strategy.py.
"""
from __future__ import annotations

import json
import math
import sys
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
OUT_DIR = ROOT / "storage" / "combo_5way"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make `scripts.*` importable so we can re-use shipped detectors
sys.path.insert(0, str(ROOT))

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Trade economics
NOTIONAL = 1000.0
RT_TAKER_FEE_BPS = 0.118   # round-trip Delta India taker (pct)
FUNDING_RATE_PER_8H = 0.0001
ATR_PERIOD = 14

# Quarter boundaries
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}

# Exit configs (same as volume_climax_walkforward / classical_patterns_walkforward)
EXITS = {
    "EA": {"tp": 1.5, "sl": 1.0, "time_min": 30,  "trail": False},
    "EB": {"tp": 2.0, "sl": 1.0, "time_min": 60,  "trail": False},
    "EC": {"tp": 99.0, "sl": 1.0, "time_min": 240, "trail": True,
           "trail_trigger": 1.0, "trail_lock_pct": 0.5},
    "ED": {"tp": 2.5, "sl": 1.0, "time_min": 240, "trail": False},
}


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    raise ValueError(tf)


def quarter_of(ts: Any) -> Optional[str]:
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    for q, (s, e) in QUARTER_BOUNDS.items():
        if pd.Timestamp(s, tz="UTC") <= ts < pd.Timestamp(e, tz="UTC"):
            return q
    return None


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi_v = 100 - 100 / (1 + rs)
    return rsi_v.fillna(50.0)


_DF_CACHE: Dict[str, pd.DataFrame] = {}


def load_tf(sym: str, tf: str) -> Optional[pd.DataFrame]:
    key = f"{sym}_{tf}"
    if key in _DF_CACHE:
        return _DF_CACHE[key]
    p = CACHE / f"{sym}_USDT_{tf}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d.index = pd.to_datetime(d.index, utc=True)
    d = d.sort_index()
    _DF_CACHE[key] = d
    return d


def cache_key_for(sym: str, tf: str) -> str:
    return f"{sym}_{tf}"


# -----------------------------------------------------------------------------
# Heiken Ashi (re-implemented inline to avoid touching live code paths)
# -----------------------------------------------------------------------------
def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    n = len(df)
    if n == 0:
        return df.copy()
    ha_open = np.zeros(n)
    ha_close = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)
    ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    ha_open[0] = (o[0] + c[0]) / 2.0
    ha_high[0] = max(h[0], ha_open[0], ha_close[0])
    ha_low[0] = min(l[0], ha_open[0], ha_close[0])
    for i in range(1, n):
        ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close[i])
    eps = 1e-9
    bull = (ha_close > ha_open) & (ha_low >= ha_open - eps)
    bear = (ha_close < ha_open) & (ha_high <= ha_open + eps)
    out = df.copy()
    out["ha_open"] = ha_open
    out["ha_close"] = ha_close
    out["ha_high"] = ha_high
    out["ha_low"] = ha_low
    out["ha_dir"] = np.where(bull, 1, np.where(bear, -1, 0))
    return out


def ha_bias_series(df_ha: pd.DataFrame, n: int) -> pd.Series:
    """+1/-1/0 regime per bar based on >=n consecutive same-direction HA bars."""
    d = df_ha["ha_dir"].astype(int).values
    out = np.zeros(len(d), dtype=int)
    run, last_dir = 0, 0
    for i in range(len(d)):
        if d[i] != 0 and d[i] == last_dir:
            run += 1
        elif d[i] != 0:
            run = 1
            last_dir = d[i]
        else:
            run = 0
            last_dir = 0
        if run >= n and last_dir != 0:
            out[i] = last_dir
    return pd.Series(out, index=df_ha.index, dtype=int)


def ema200_d_bias_series(df_4h: pd.DataFrame) -> pd.Series:
    """Daily EMA200 macro bias derived from 4h cache (resampled to daily)."""
    day = df_4h["close"].astype(float).resample("1D").last().dropna()
    min_warm = 50  # cache is only ~6mo => never warm a strict 200; use min_warm=50
    if len(day) < min_warm:
        return pd.Series(0, index=df_4h.index, dtype=int)
    e200 = day.ewm(span=200, adjust=False, min_periods=min_warm).mean()
    chop_band = 0.005
    diff = (day - e200) / e200
    bias_d = pd.Series(0, index=day.index, dtype=int)
    bias_d[diff > chop_band] = 1
    bias_d[diff < -chop_band] = -1
    bias_d = bias_d.shift(1).dropna()
    bias_d.index = bias_d.index + pd.Timedelta(days=1)
    return bias_d.reindex(df_4h.index, method="ffill").fillna(0).astype(int)


def ema21_50_bias_series(df: pd.DataFrame) -> pd.Series:
    e21 = ema(df["close"], 21)
    e50 = ema(df["close"], 50)
    c = df["close"].astype(float)
    out = pd.Series(0, index=df.index, dtype=int)
    out[(c > e21) & (e21 > e50)] = 1
    out[(c < e21) & (e21 < e50)] = -1
    return out


# -----------------------------------------------------------------------------
# MACD / RSI divergence helpers (vectorised, simpler than scripts/macd_div_detector
# and operate on the entire df rather than only the last bar — needed for
# historical scan).
# -----------------------------------------------------------------------------
def _find_pivots(arr: np.ndarray, lb: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pivot_high_idx, pivot_low_idx). Conservative: needs lb bars on each side."""
    n = len(arr)
    ph, pl = [], []
    for i in range(lb, n - lb):
        v = arr[i]
        if v == max(arr[i - lb: i + lb + 1]):
            ph.append(i)
        if v == min(arr[i - lb: i + lb + 1]):
            pl.append(i)
    return np.asarray(ph, dtype=int), np.asarray(pl, dtype=int)


def detect_divergence_series(
    df: pd.DataFrame,
    indicator_col: str,
    lookback: int = 20,
    pivot_lb: int = 2,
) -> pd.DataFrame:
    """Mark each bar as 'regular_bull', 'regular_bear', or '' divergence.

    The divergence is detected at the moment the *right side* of a pivot is
    confirmed — i.e. at index (pivot_idx + pivot_lb). The signal ts is therefore
    `pivot_idx + pivot_lb` so trades open on the next bar (no leak).
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    ind = df[indicator_col].astype(float).values
    n = len(df)
    out = np.full(n, "", dtype=object)

    # _find_pivots(arr, lb) returns (max_pivot_indices, min_pivot_indices) of arr.
    # For price-based pivots we want max pivots of highs and min pivots of lows.
    ph_idx, _ = _find_pivots(highs, pivot_lb)
    _, pl_idx = _find_pivots(lows, pivot_lb)

    # bearish regular divergence: price HH (higher high) & indicator LH (lower high)
    for k in range(1, len(ph_idx)):
        i2 = int(ph_idx[k])
        i1 = int(ph_idx[k - 1])
        if (i2 - i1) > lookback or (i2 - i1) < 2:
            continue
        if highs[i2] <= highs[i1]:
            continue
        if ind[i2] >= ind[i1]:
            continue
        # signal confirms at i2 + pivot_lb (right-side bars now in)
        sig_idx = i2 + pivot_lb
        if sig_idx >= n:
            continue
        out[sig_idx] = "regular_bear"

    # bullish regular divergence: price LL & indicator HL
    for k in range(1, len(pl_idx)):
        i2 = int(pl_idx[k])
        i1 = int(pl_idx[k - 1])
        if (i2 - i1) > lookback or (i2 - i1) < 2:
            continue
        if lows[i2] >= lows[i1]:
            continue
        if ind[i2] <= ind[i1]:
            continue
        sig_idx = i2 + pivot_lb
        if sig_idx >= n:
            continue
        out[sig_idx] = "regular_bull"

    return pd.Series(out, index=df.index, dtype=object)


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=sig, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


# -----------------------------------------------------------------------------
# Volume climax 1h (re-uses logic from scripts/volume_climax_walkforward)
# -----------------------------------------------------------------------------
def detect_climax_series(
    df: pd.DataFrame,
    min_rel_vol: float = 2.5,
    lookback: int = 20,
    body_max: float = 0.4,
) -> pd.Series:
    """Return string side per bar: 'LONG'/'SHORT'/'' for volume-climax candles."""
    vol = df["volume"].astype(float)
    vol_avg = vol.rolling(20, min_periods=20).mean()
    rel_vol = vol / vol_avg.replace(0, np.nan)

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    rng = (h - l).clip(lower=1e-12)
    body = (c - o).abs()
    body_ratio = body / rng
    lo_wick = np.minimum(o, c) - l
    up_wick = h - np.maximum(o, c)
    roll_low = l.shift(1).rolling(lookback, min_periods=lookback).min()
    roll_high = h.shift(1).rolling(lookback, min_periods=lookback).max()
    new_low = l < roll_low
    new_high = h > roll_high

    long_mask = (
        (rel_vol >= min_rel_vol) & new_low & (body_ratio < body_max)
        & (lo_wick > body) & (c > o)
    )
    short_mask = (
        (rel_vol >= min_rel_vol) & new_high & (body_ratio < body_max)
        & (up_wick > body) & (c < o)
    )
    out = pd.Series("", index=df.index, dtype=object)
    out[long_mask.fillna(False)] = "LONG"
    out[short_mask.fillna(False)] = "SHORT"
    return out


# -----------------------------------------------------------------------------
# Wedge breakout (lazy-import from scripts.wedge_detector to scan historical
# data — but the shipped detector is "last-bar only", so we wrap it to run
# bar-by-bar over a sliding window).
# -----------------------------------------------------------------------------
_WEDGE_CACHE: Dict[str, pd.Series] = {}


def detect_wedge_series(
    df: pd.DataFrame,
    pivot_lb: int = 5,
    detect_window: int = 30,
    vol_threshold: float = 1.0,
    cache_key: Optional[str] = None,
) -> pd.Series:
    """Run shipped detector at every bar (returns 'LONG'/'SHORT'/'').

    Cached by `cache_key` (e.g. "BTC-1h-5-30-1.0") to avoid recomputing
    across cells.
    """
    ck = f"{cache_key}:{pivot_lb}:{detect_window}:{vol_threshold}"
    if cache_key and ck in _WEDGE_CACHE:
        return _WEDGE_CACHE[ck]
    from scripts.wedge_detector import detect_wedge_breakout
    n = len(df)
    out = pd.Series("", index=df.index, dtype=object)
    min_len = detect_window + pivot_lb + 2
    # Pre-compute the running pivot-detection in vectorised form: scan once
    # and detect pivots within rolling window of detect_window+pivot_lb. Then
    # only invoke the shipped detector at end indices where the trendline math
    # could fire — i.e., we always invoke it at every bar (correctness-first).
    for end in range(min_len, n):
        sub = df.iloc[max(0, end - detect_window - pivot_lb - 5): end + 1]
        try:
            res = detect_wedge_breakout(sub, pivot_lb=pivot_lb,
                                        detect_window=detect_window,
                                        vol_threshold=vol_threshold)
        except Exception:
            res = None
        if res is not None:
            out.iloc[end] = "LONG" if res["side"] == "long" else "SHORT"
    if cache_key:
        _WEDGE_CACHE[ck] = out
    return out


# -----------------------------------------------------------------------------
# Engulfing reversal — synthetic test cohort for COMBO 4
# -----------------------------------------------------------------------------
def detect_engulfing_series(df: pd.DataFrame) -> pd.Series:
    o = df["open"].astype(float)
    c = df["close"].astype(float)
    po = o.shift(1)
    pc = c.shift(1)
    bull_eng = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear_eng = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    out = pd.Series("", index=df.index, dtype=object)
    out[bull_eng.fillna(False)] = "LONG"
    out[bear_eng.fillna(False)] = "SHORT"
    return out


# -----------------------------------------------------------------------------
# Double top/bottom (re-implementation of scripts/classical_patterns logic)
# Returns *all* signals as list-of-dicts for the dataframe.
# -----------------------------------------------------------------------------
def detect_double_signals(
    df: pd.DataFrame,
    atr: pd.Series,
    pivot_lb: int = 5,
    detect_window: int = 50,
    side: str = "SHORT",
) -> List[Dict[str, Any]]:
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    n = len(df)
    # _find_pivots(arr) treats arr as a single series — but for double top/bottom
    # we want pivot-highs of `highs` and pivot-lows of `lows` separately.
    # Find pivot-highs in highs:
    nlen = len(highs)
    ph_list, pl_list = [], []
    for ii in range(pivot_lb, nlen - pivot_lb):
        if highs[ii] == max(highs[ii - pivot_lb: ii + pivot_lb + 1]):
            ph_list.append(ii)
        if lows[ii] == min(lows[ii - pivot_lb: ii + pivot_lb + 1]):
            pl_list.append(ii)
    ph = np.asarray(ph_list, dtype=int)
    pl = np.asarray(pl_list, dtype=int)
    signals: List[Dict[str, Any]] = []

    if side == "SHORT":
        # double top: 2 pivot highs within 0.3 ATR, separated by >=10 bars,
        # trigger when close < intervening swing low
        for k in range(1, len(ph)):
            i2 = int(ph[k]); i1 = int(ph[k - 1])
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
                if closes[j] < mid_low:
                    signals.append({"entry_idx": j, "side": "SHORT", "atr": atr_v})
                    break
    else:
        for k in range(1, len(pl)):
            i2 = int(pl[k]); i1 = int(pl[k - 1])
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
                if closes[j] > mid_high:
                    signals.append({"entry_idx": j, "side": "LONG", "atr": atr_v})
                    break
    return signals


# -----------------------------------------------------------------------------
# Trade simulator (4 exit configs)
# -----------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr_v: float,
    exit_cfg: Dict[str, Any],
    tf_min: int,
) -> Optional[Dict[str, Any]]:
    if entry_idx + 1 >= len(df) or not np.isfinite(atr_v) or atr_v <= 0:
        return None
    entry = float(df["close"].iloc[entry_idx])
    sl_dist = exit_cfg["sl"] * atr_v
    tp_dist = exit_cfg["tp"] * atr_v
    bars_max = max(1, int(math.ceil(exit_cfg["time_min"] / tf_min)))

    if side == "LONG":
        sl_p = entry - sl_dist; tp_p = entry + tp_dist
    else:
        sl_p = entry + sl_dist; tp_p = entry - tp_dist

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
            if hit_sl: exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_trail: exit_reason = "TRAIL"; exit_price = trail_stop; exit_idx = j; break
            if hit_tp: exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break
        else:
            hit_sl = bar_h >= sl_p
            hit_tp = bar_l <= tp_p
            hit_trail = trail_stop is not None and bar_h >= trail_stop
            if hit_sl: exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_trail: exit_reason = "TRAIL"; exit_price = trail_stop; exit_idx = j; break
            if hit_tp: exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break

    if side == "LONG":
        ret = (exit_price - entry) / entry
    else:
        ret = (entry - exit_price) / entry
    gross_d = NOTIONAL * ret
    fee_d = NOTIONAL * (RT_TAKER_FEE_BPS / 100.0)
    bars_held = exit_idx - entry_idx
    minutes_held = bars_held * tf_min
    funding_d = NOTIONAL * FUNDING_RATE_PER_8H * (minutes_held / (8 * 60))
    net_d = gross_d - fee_d - funding_d
    risk_d = NOTIONAL * (sl_dist / entry)
    gross_R = gross_d / risk_d if risk_d > 0 else 0.0
    net_R = net_d / risk_d if risk_d > 0 else 0.0
    return {
        "entry_ts": df.index[entry_idx],
        "exit_ts": df.index[exit_idx],
        "side": side,
        "exit_reason": exit_reason,
        "minutes_held": minutes_held,
        "gross_R": gross_R,
        "net_R": net_R,
        "gross_$": gross_d,
        "net_$": net_d,
        "mfe_R": mfe_R,
    }


# -----------------------------------------------------------------------------
# Aggregation utilities
# -----------------------------------------------------------------------------
def aggregate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "wr": 0.0, "ev_R": 0.0, "ev_$": 0.0,
                "net_R": 0.0, "net_$": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_R"] > 0)
    return {
        "n": n,
        "wr": wins / n,
        "ev_R": float(np.mean([t["net_R"] for t in trades])),
        "ev_$": float(np.mean([t["net_$"] for t in trades])),
        "net_R": float(np.sum([t["net_R"] for t in trades])),
        "net_$": float(np.sum([t["net_$"] for t in trades])),
    }


def split_quarters(trades: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {q: [] for q in QUARTER_BOUNDS}
    for t in trades:
        q = quarter_of(t["entry_ts"])
        if q is not None:
            out[q].append(t)
    return out


def walk_forward_verdict(is_ev: float, q3_ev: float, q4_ev: float,
                         is_n: int, q3_n: int, q4_n: int) -> Dict[str, Any]:
    """Compute SHIP / HOLD / KILL verdict using user-spec criteria.

    Pass: OOS EV >= 50% of IS EV AND same sign (positive) AND n>=30 IS, n>=20 OOS each.
    """
    def gap_pct(is_v: float, oos_v: float) -> float:
        if abs(is_v) < 1e-9:
            return 0.0
        return (is_v - oos_v) / abs(is_v) * 100.0

    q3_gap = gap_pct(is_ev, q3_ev)
    q4_gap = gap_pct(is_ev, q4_ev)

    is_pos = is_ev > 0
    q3_pos = q3_ev > 0
    q4_pos = q4_ev > 0
    q3_keep_half = q3_ev >= 0.5 * is_ev if is_pos else False
    q4_keep_half = q4_ev >= 0.5 * is_ev if is_pos else False

    n_ok = is_n >= 30 and q3_n >= 20 and q4_n >= 20

    verdict = "KILL"
    if is_pos and q3_pos and q4_pos and q3_keep_half and q4_keep_half and n_ok:
        verdict = "SHIP"
    elif is_pos and ((q3_pos and q3_keep_half) or (q4_pos and q4_keep_half)):
        verdict = "HOLD"
    return {
        "verdict": verdict,
        "is_ev": is_ev, "q3_ev": q3_ev, "q4_ev": q4_ev,
        "is_n": is_n, "q3_n": q3_n, "q4_n": q4_n,
        "q3_gap_pct": q3_gap, "q4_gap_pct": q4_gap,
        "n_ok": n_ok,
    }


# =============================================================================
# COMBO 1 — Exhaustion Confluence Stack
# =============================================================================
def combo1_signals(
    df_test: pd.DataFrame,
    df_climax: pd.DataFrame,
    div_lookback: int,
    climax_min_rel: float,
    same_tf: bool,
    confluence_window_bars: int = 12,
) -> List[Dict[str, Any]]:
    """Generate combo-1 signals on the test TF.

    Rule: at the bar where MACD div fires, require RSI div within
    +/- confluence_window_bars AND volume climax (ffilled) within
    +/- confluence_window_bars. All 3 same direction (LONG or SHORT).
    Trigger ts is the LATEST of the 3 component fires (so at trigger time all
    3 are in evidence — no leak).
    """
    work = add_macd(df_test)
    work["rsi"] = rsi(work["close"], 14)
    macd_div = detect_divergence_series(work, "macd", lookback=div_lookback, pivot_lb=2)
    rsi_div = detect_divergence_series(work, "rsi", lookback=div_lookback, pivot_lb=2)

    # Volume climax on its native TF
    climax = detect_climax_series(df_climax, min_rel_vol=climax_min_rel,
                                  lookback=20, body_max=0.4)
    # Map climax timestamps back into test-TF nearest indices
    climax_hits = [(idx, side) for idx, side in climax.items() if side]

    # Build per-bar (idx, dir) lists for MACD and RSI
    macd_hits = [(i, "LONG" if v == "regular_bull" else "SHORT")
                 for i, v in enumerate(macd_div.values) if v]
    rsi_hits = [(i, "LONG" if v == "regular_bull" else "SHORT")
                for i, v in enumerate(rsi_div.values) if v]

    rsi_idx_by_dir: Dict[str, List[int]] = {"LONG": [], "SHORT": []}
    for i, d in rsi_hits:
        rsi_idx_by_dir[d].append(i)

    # Build climax-time map: list of (test_tf_idx, side)
    climax_to_test: List[Tuple[int, str]] = []
    test_index = df_test.index
    for ts, side in climax_hits:
        # find test_tf bar at or after ts
        loc = test_index.searchsorted(ts, side="left")
        if loc >= len(test_index):
            continue
        climax_to_test.append((int(loc), side))
    climax_idx_by_dir: Dict[str, List[int]] = {"LONG": [], "SHORT": []}
    for i, d in climax_to_test:
        climax_idx_by_dir[d].append(i)

    signals: List[Dict[str, Any]] = []
    used_ts: set = set()
    for m_idx, m_dir in macd_hits:
        # Find RSI hit of same direction within window
        rsi_candidates = [i for i in rsi_idx_by_dir[m_dir]
                          if abs(i - m_idx) <= confluence_window_bars]
        if not rsi_candidates:
            continue
        rsi_idx = max(rsi_candidates) if any(i <= m_idx for i in rsi_candidates) else min(rsi_candidates)
        # Find climax hit of same direction within window
        cl_candidates = [i for i in climax_idx_by_dir[m_dir]
                         if abs(i - m_idx) <= confluence_window_bars]
        if not cl_candidates:
            continue
        cl_idx = max([c for c in cl_candidates if c <= m_idx], default=None)
        if cl_idx is None:
            cl_idx = min(cl_candidates)

        # Trigger at the LATEST of the 3 (no leak)
        trigger = max(m_idx, rsi_idx, cl_idx)
        if trigger >= len(df_test):
            continue
        if trigger in used_ts:
            continue
        signals.append({"entry_idx": trigger, "side": m_dir})
        used_ts.add(trigger)
    return signals


# =============================================================================
# COMBO 2 — Wedge x HA Bias 4h alignment
# =============================================================================
def combo2_signals(
    df_test: pd.DataFrame,
    df_wedge: pd.DataFrame,
    df_ha_4h: pd.DataFrame,
    ha_n: int,
    ha_lookback_4h_bars: int = 3,
    wedge_cache_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Wedge breakout on wedge TF, aligned with HA 4h bias.

    The wedge breakout direction must agree with the HA-4h regime (>=N consec)
    on at least ONE of the prior `ha_lookback_4h_bars` 4h bars.
    Trade direction = wedge breakout direction.
    """
    wedge_series = detect_wedge_series(df_wedge,
                                        pivot_lb=5, detect_window=30,
                                        vol_threshold=1.0,
                                        cache_key=wedge_cache_key)
    ha_4h = compute_ha(df_ha_4h)
    ha_bias_4h_raw = ha_bias_series(ha_4h, ha_n)
    # Shift by 1 to use last CLOSED 4h bar's regime
    ha_bias_4h = ha_bias_4h_raw.shift(1).fillna(0).astype(int)

    # Re-index both onto df_test (wedge fires at native TF)
    wedge_on_test = wedge_series.reindex(df_test.index, method="ffill").fillna("")
    # For each test bar, also build "max_in_lookback" of HA bias's recent values
    ha_on_test = ha_bias_4h.reindex(df_test.index, method="ffill").fillna(0).astype(int)

    out: List[Dict[str, Any]] = []
    last_emitted_idx = -10
    # Build a "prior N 4h bars HA bias" check: convert ha_lookback to test-TF bars
    test_tf_min = (df_test.index[1] - df_test.index[0]).total_seconds() / 60.0 if len(df_test) > 1 else 60
    lookback_test_bars = max(1, int(ha_lookback_4h_bars * 240 / test_tf_min))
    for i in range(len(df_test)):
        side = wedge_on_test.iloc[i]
        if not side:
            continue
        # Detect transitions to avoid ffill duplicates
        if i > 0 and wedge_on_test.iloc[i - 1] == side:
            continue
        side_int = 1 if side == "LONG" else -1
        # HA must have been aligned at least once in the prior N 4h bars
        start_lb = max(0, i - lookback_test_bars)
        recent_ha = ha_on_test.iloc[start_lb: i + 1].values
        if not (recent_ha == side_int).any():
            continue
        if i - last_emitted_idx < 5:
            continue
        out.append({"entry_idx": i, "side": side})
        last_emitted_idx = i
    return out


# =============================================================================
# COMBO 3 — Double Top/Bottom x EMA21/50 1h confluence
# =============================================================================
def combo3_signals(
    df_test: pd.DataFrame,
    df_htf_1h: pd.DataFrame,
    pivot_lb: int = 5,
    detect_window: int = 50,
    htf_lookback: int = 20,
) -> List[Dict[str, Any]]:
    """Double top -> SHORT only when HTF was bullish prior 20 bars (overshoot reversal).
    Double bottom -> LONG only when HTF was bearish prior 20 bars.
    """
    atr_test = add_atr(df_test, ATR_PERIOD)
    short_sigs = detect_double_signals(df_test, atr_test, pivot_lb=pivot_lb,
                                        detect_window=detect_window, side="SHORT")
    long_sigs = detect_double_signals(df_test, atr_test, pivot_lb=pivot_lb,
                                       detect_window=detect_window, side="LONG")

    htf_bias = ema21_50_bias_series(df_htf_1h)  # +1 bull, -1 bear, 0 neutral
    htf_bias_shift = htf_bias.shift(1).fillna(0).astype(int)
    htf_on_test = htf_bias_shift.reindex(df_test.index, method="ffill").fillna(0).astype(int)

    out: List[Dict[str, Any]] = []
    for sig in short_sigs:
        i = int(sig["entry_idx"])
        # HTF must have been bullish for prior_lookback bars
        prior_idxs = range(max(0, i - htf_lookback), i)
        prior_vals = [int(htf_on_test.iloc[k]) for k in prior_idxs]
        bull_count = sum(1 for v in prior_vals if v == 1)
        if bull_count >= int(0.5 * htf_lookback):
            out.append({"entry_idx": i, "side": "SHORT"})
    for sig in long_sigs:
        i = int(sig["entry_idx"])
        prior_idxs = range(max(0, i - htf_lookback), i)
        prior_vals = [int(htf_on_test.iloc[k]) for k in prior_idxs]
        bear_count = sum(1 for v in prior_vals if v == -1)
        if bear_count >= int(0.5 * htf_lookback):
            out.append({"entry_idx": i, "side": "LONG"})
    return out


# =============================================================================
# COMBO 4 — Regime Double-Confirm (HA bias 4h AND EMA200_d) on engulfings
# =============================================================================
def combo4_tagged(
    df_test: pd.DataFrame,
    df_4h: pd.DataFrame,
    ha_n: int = 3,
) -> List[Dict[str, Any]]:
    """Tag every engulfing reversal with:
       - ha_align: ALIGNED/OPPOSED/NEUTRAL
       - e200d_align: ALIGNED/OPPOSED/NEUTRAL
       - both_aligned (bool)
       - both_opposed (bool)
    """
    eng = detect_engulfing_series(df_test)
    ha_4h = compute_ha(df_4h)
    ha_bias_4h = ha_bias_series(ha_4h, ha_n).shift(1).fillna(0).astype(int)
    ha_on_test = ha_bias_4h.reindex(df_test.index, method="ffill").fillna(0).astype(int)
    e200d = ema200_d_bias_series(df_4h)
    e200d_on_test = e200d.reindex(df_test.index, method="ffill").fillna(0).astype(int)

    sig_idxs = [i for i in range(len(df_test)) if eng.iloc[i]]
    out: List[Dict[str, Any]] = []
    for i in sig_idxs:
        side = eng.iloc[i]
        side_int = 1 if side == "LONG" else -1
        ha_b = int(ha_on_test.iloc[i])
        e2_b = int(e200d_on_test.iloc[i])
        ha_align = "NEUTRAL" if ha_b == 0 else ("ALIGNED" if ha_b == side_int else "OPPOSED")
        e2_align = "NEUTRAL" if e2_b == 0 else ("ALIGNED" if e2_b == side_int else "OPPOSED")
        out.append({
            "entry_idx": i, "side": side,
            "ha_align": ha_align, "e200d_align": e2_align,
            "both_aligned": ha_align == "ALIGNED" and e2_align == "ALIGNED",
            "both_opposed": ha_align == "OPPOSED" and e2_align == "OPPOSED",
        })
    return out


# =============================================================================
# Per-symbol runners
# =============================================================================
def run_combo1_symbol(sym: str,
                      div_lookback: int, climax_min_rel: float,
                      test_tf: str, climax_tf: str) -> Dict[str, List[Dict[str, Any]]]:
    """Returns {exit_label: [trades]}"""
    df_test = load_tf(sym, test_tf)
    df_climax = load_tf(sym, climax_tf)
    if df_test is None or df_climax is None or len(df_test) < 200 or len(df_climax) < 200:
        return {k: [] for k in EXITS}

    sigs = combo1_signals(df_test, df_climax, div_lookback, climax_min_rel,
                           same_tf=(test_tf == climax_tf))
    atr_test = add_atr(df_test, ATR_PERIOD)
    tf_min = tf_minutes(test_tf)
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
    for sig in sigs:
        i = int(sig["entry_idx"])
        atr_v = float(atr_test.iloc[i])
        for ex_lbl, ex_cfg in EXITS.items():
            t = simulate_trade(df_test, i, sig["side"], atr_v, ex_cfg, tf_min)
            if t:
                t["symbol"] = sym
                t["test_tf"] = test_tf
                t["climax_tf"] = climax_tf
                out[ex_lbl].append(t)
    return out


def run_combo2_symbol(sym: str, ha_n: int,
                      test_tf: str, wedge_tf: str,
                      ha_lookback_4h_bars: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    df_test = load_tf(sym, test_tf)
    df_wedge = load_tf(sym, wedge_tf)
    df_4h = load_tf(sym, "4h")
    if df_test is None or df_wedge is None or df_4h is None or len(df_test) < 200:
        return {k: [] for k in EXITS}
    sigs = combo2_signals(df_test, df_wedge, df_4h, ha_n,
                           ha_lookback_4h_bars=ha_lookback_4h_bars,
                           wedge_cache_key=cache_key_for(sym, wedge_tf))
    atr_test = add_atr(df_test, ATR_PERIOD)
    tf_min = tf_minutes(test_tf)
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
    for sig in sigs:
        i = int(sig["entry_idx"])
        atr_v = float(atr_test.iloc[i])
        for ex_lbl, ex_cfg in EXITS.items():
            t = simulate_trade(df_test, i, sig["side"], atr_v, ex_cfg, tf_min)
            if t:
                t["symbol"] = sym; t["ha_n"] = ha_n; t["test_tf"] = test_tf
                out[ex_lbl].append(t)
    return out


def run_combo3_symbol(sym: str, test_tf: str) -> Dict[str, List[Dict[str, Any]]]:
    df_test = load_tf(sym, test_tf)
    df_htf_1h = load_tf(sym, "1h")
    if df_test is None or df_htf_1h is None or len(df_test) < 200:
        return {k: [] for k in EXITS}
    sigs = combo3_signals(df_test, df_htf_1h)
    atr_test = add_atr(df_test, ATR_PERIOD)
    tf_min = tf_minutes(test_tf)
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
    for sig in sigs:
        i = int(sig["entry_idx"])
        atr_v = float(atr_test.iloc[i])
        for ex_lbl, ex_cfg in EXITS.items():
            t = simulate_trade(df_test, i, sig["side"], atr_v, ex_cfg, tf_min)
            if t:
                t["symbol"] = sym; t["test_tf"] = test_tf
                out[ex_lbl].append(t)
    return out


def run_combo4_symbol(sym: str, test_tf: str, ha_n: int = 3) -> List[Dict[str, Any]]:
    df_test = load_tf(sym, test_tf)
    df_4h = load_tf(sym, "4h")
    if df_test is None or df_4h is None or len(df_test) < 200 or len(df_4h) < 200:
        return []
    sigs = combo4_tagged(df_test, df_4h, ha_n=ha_n)
    atr_test = add_atr(df_test, ATR_PERIOD)
    tf_min = tf_minutes(test_tf)
    out: List[Dict[str, Any]] = []
    for sig in sigs:
        i = int(sig["entry_idx"])
        atr_v = float(atr_test.iloc[i])
        # Use a single fixed exit (TP=1.5R/SL=1R/60m) for the regime-veto test
        ex_cfg = {"tp": 1.5, "sl": 1.0, "time_min": 60, "trail": False}
        t = simulate_trade(df_test, i, sig["side"], atr_v, ex_cfg, tf_min)
        if t is None:
            continue
        t.update({
            "symbol": sym, "test_tf": test_tf,
            "ha_align": sig["ha_align"], "e200d_align": sig["e200d_align"],
            "both_aligned": sig["both_aligned"], "both_opposed": sig["both_opposed"],
        })
        out.append(t)
    return out


# =============================================================================
# COMBO 1 — Walk-forward driver
# =============================================================================
def combo1_walkforward() -> Dict[str, Any]:
    """Tune (div_lookback, climax_min_rel, TF combo) on Q1+Q2 IS, evaluate on Q3+Q4."""
    grid_div_lb = [15, 20, 30]
    grid_min_rel = [2.0, 2.5, 3.0]
    tf_combos = [
        ("1h", "1h"),     # all on 1h
        ("15m", "1h"),    # divs on 15m, climax ffilled from 1h
        ("5m", "1h"),     # divs on 5m, climax ffilled from 1h
    ]
    cell_results: List[Dict[str, Any]] = []

    for (test_tf, climax_tf), div_lb, min_rel in product(tf_combos, grid_div_lb, grid_min_rel):
        # Aggregate trades across symbols
        trades_by_exit: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
        for sym in SYMBOLS:
            res = run_combo1_symbol(sym, div_lb, min_rel, test_tf, climax_tf)
            for k, v in res.items():
                trades_by_exit[k].extend(v)
        for ex_lbl, trades in trades_by_exit.items():
            qsplit = split_quarters(trades)
            is_trades = qsplit["Q1"] + qsplit["Q2"]
            cell_results.append({
                "test_tf": test_tf, "climax_tf": climax_tf,
                "div_lookback": div_lb, "climax_min_rel": min_rel,
                "exit": ex_lbl,
                "IS": aggregate(is_trades),
                "Q3": aggregate(qsplit["Q3"]),
                "Q4": aggregate(qsplit["Q4"]),
                "n_total": len(trades),
            })

    # Pick best (TF, params) cell per exit by IS EV (with n>=15 IS — slight relax
    # because confluence drives sample down; we still apply the n>=30 IS pass
    # criterion in verdict)
    best_per_exit: Dict[str, Dict[str, Any]] = {}
    for ex_lbl in EXITS:
        cands = [r for r in cell_results if r["exit"] == ex_lbl and r["IS"]["n"] >= 15]
        if not cands:
            cands = [r for r in cell_results if r["exit"] == ex_lbl]
        cands.sort(key=lambda r: r["IS"]["ev_R"], reverse=True)
        best = cands[0] if cands else None
        if best is None:
            continue
        verdict = walk_forward_verdict(
            best["IS"]["ev_R"], best["Q3"]["ev_R"], best["Q4"]["ev_R"],
            best["IS"]["n"], best["Q3"]["n"], best["Q4"]["n"],
        )
        best["verdict"] = verdict
        best_per_exit[ex_lbl] = best

    return {
        "all_cells": cell_results,
        "best_per_exit": best_per_exit,
    }


# =============================================================================
# COMBO 2 — Walk-forward driver
# =============================================================================
def combo2_walkforward() -> Dict[str, Any]:
    """Wedge x HA bias confluence — tune ha_n on Q1+Q2 IS.

    Wedge is detected on 1h only (validated SHIP) — trade TF is either 1h
    (full bar wait) or 15m (entry on the bar nearest to the 1h wedge fire).
    HA bias is on 4h (per architect's spec).
    """
    grid_ha_n = [3, 4, 5]
    grid_ha_lookback = [3, 5, 8]  # how many recent 4h bars to require alignment
    tf_combos = [
        ("1h", "1h"),
        ("15m", "1h"),
    ]
    cell_results: List[Dict[str, Any]] = []
    for (test_tf, wedge_tf), ha_n, ha_lb in product(tf_combos, grid_ha_n, grid_ha_lookback):
        trades_by_exit: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
        for sym in SYMBOLS:
            res = run_combo2_symbol(sym, ha_n, test_tf, wedge_tf, ha_lookback_4h_bars=ha_lb)
            for k, v in res.items():
                trades_by_exit[k].extend(v)
        for ex_lbl, trades in trades_by_exit.items():
            qsplit = split_quarters(trades)
            is_trades = qsplit["Q1"] + qsplit["Q2"]
            cell_results.append({
                "test_tf": test_tf, "wedge_tf": wedge_tf,
                "ha_n": ha_n, "ha_lookback_4h": ha_lb, "exit": ex_lbl,
                "IS": aggregate(is_trades),
                "Q3": aggregate(qsplit["Q3"]),
                "Q4": aggregate(qsplit["Q4"]),
                "n_total": len(trades),
            })

    best_per_exit: Dict[str, Dict[str, Any]] = {}
    for ex_lbl in EXITS:
        cands = [r for r in cell_results if r["exit"] == ex_lbl and r["IS"]["n"] >= 5]
        cands.sort(key=lambda r: r["IS"]["ev_R"], reverse=True)
        best = cands[0] if cands else None
        if best is None:
            continue
        verdict = walk_forward_verdict(
            best["IS"]["ev_R"], best["Q3"]["ev_R"], best["Q4"]["ev_R"],
            best["IS"]["n"], best["Q3"]["n"], best["Q4"]["n"],
        )
        best["verdict"] = verdict
        best_per_exit[ex_lbl] = best
    return {"all_cells": cell_results, "best_per_exit": best_per_exit}


# =============================================================================
# COMBO 3 — Walk-forward driver
# =============================================================================
def combo3_walkforward() -> Dict[str, Any]:
    """Double pattern x HTF EMA21/50 1h confluence — no tuning, fixed params."""
    cell_results: List[Dict[str, Any]] = []
    for test_tf in ["15m", "1h", "4h"]:
        trades_by_exit: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
        for sym in SYMBOLS:
            res = run_combo3_symbol(sym, test_tf)
            for k, v in res.items():
                trades_by_exit[k].extend(v)
        for ex_lbl, trades in trades_by_exit.items():
            qsplit = split_quarters(trades)
            is_trades = qsplit["Q1"] + qsplit["Q2"]
            cell_results.append({
                "test_tf": test_tf, "exit": ex_lbl,
                "IS": aggregate(is_trades),
                "Q3": aggregate(qsplit["Q3"]),
                "Q4": aggregate(qsplit["Q4"]),
                "n_total": len(trades),
            })
    best_per_exit: Dict[str, Dict[str, Any]] = {}
    for ex_lbl in EXITS:
        cands = [r for r in cell_results if r["exit"] == ex_lbl]
        cands.sort(key=lambda r: r["IS"]["ev_R"], reverse=True)
        best = cands[0] if cands else None
        if best is None:
            continue
        verdict = walk_forward_verdict(
            best["IS"]["ev_R"], best["Q3"]["ev_R"], best["Q4"]["ev_R"],
            best["IS"]["n"], best["Q3"]["n"], best["Q4"]["n"],
        )
        best["verdict"] = verdict
        best_per_exit[ex_lbl] = best
    return {"all_cells": cell_results, "best_per_exit": best_per_exit}


# =============================================================================
# COMBO 4 — Regime double-confirm counterfactual
# =============================================================================
def combo4_walkforward() -> Dict[str, Any]:
    """No tuning. Test as regime-double-confirm veto on engulfing cohort."""
    out: Dict[str, Any] = {}
    for test_tf in ["5m", "15m"]:
        all_trades: List[Dict[str, Any]] = []
        for sym in SYMBOLS:
            all_trades.extend(run_combo4_symbol(sym, test_tf, ha_n=3))

        # Bucket by quarter and by alignment-pair tag
        def bucket_key(t):
            return (t["ha_align"], t["e200d_align"])

        per_q: Dict[str, Dict[Tuple[str, str], List[Dict[str, Any]]]] = {q: {} for q in QUARTER_BOUNDS}
        for t in all_trades:
            q = quarter_of(t["entry_ts"])
            if q is None:
                continue
            k = bucket_key(t)
            per_q[q].setdefault(k, []).append(t)

        # Compute IS (Q1+Q2) and OOS (Q3, Q4) aggregates per bucket
        all_keys = set()
        for q in per_q:
            all_keys.update(per_q[q].keys())

        bucket_stats = {}
        for k in all_keys:
            is_trades = per_q["Q1"].get(k, []) + per_q["Q2"].get(k, [])
            q3_trades = per_q["Q3"].get(k, [])
            q4_trades = per_q["Q4"].get(k, [])
            bucket_stats[f"{k[0]}/{k[1]}"] = {
                "IS": aggregate(is_trades),
                "Q3": aggregate(q3_trades),
                "Q4": aggregate(q4_trades),
            }
        # Headline: ALL trades (no veto) vs only allowed (veto OPPOSED/OPPOSED), vs both_aligned only
        is_all = [t for t in all_trades if quarter_of(t["entry_ts"]) in ("Q1", "Q2")]
        q3_all = [t for t in all_trades if quarter_of(t["entry_ts"]) == "Q3"]
        q4_all = [t for t in all_trades if quarter_of(t["entry_ts"]) == "Q4"]

        # Veto policy: BLOCK trades where both filters OPPOSE (= bucket OPPOSED/OPPOSED)
        def veto_blocked(t):
            return t["both_opposed"]

        is_veto = [t for t in is_all if not veto_blocked(t)]
        q3_veto = [t for t in q3_all if not veto_blocked(t)]
        q4_veto = [t for t in q4_all if not veto_blocked(t)]

        is_aligned_only = [t for t in is_all if t["both_aligned"]]
        q3_aligned_only = [t for t in q3_all if t["both_aligned"]]
        q4_aligned_only = [t for t in q4_all if t["both_aligned"]]

        out[test_tf] = {
            "buckets": bucket_stats,
            "ALL_trades": {
                "IS": aggregate(is_all), "Q3": aggregate(q3_all), "Q4": aggregate(q4_all),
            },
            "VETO_double_opposed_blocked": {
                "IS": aggregate(is_veto), "Q3": aggregate(q3_veto), "Q4": aggregate(q4_veto),
            },
            "ALLOW_double_aligned_only": {
                "IS": aggregate(is_aligned_only),
                "Q3": aggregate(q3_aligned_only), "Q4": aggregate(q4_aligned_only),
            },
        }

        # Verdict on the "double-aligned only" allow policy
        out[test_tf]["aligned_verdict"] = walk_forward_verdict(
            out[test_tf]["ALLOW_double_aligned_only"]["IS"]["ev_R"],
            out[test_tf]["ALLOW_double_aligned_only"]["Q3"]["ev_R"],
            out[test_tf]["ALLOW_double_aligned_only"]["Q4"]["ev_R"],
            out[test_tf]["ALLOW_double_aligned_only"]["IS"]["n"],
            out[test_tf]["ALLOW_double_aligned_only"]["Q3"]["n"],
            out[test_tf]["ALLOW_double_aligned_only"]["Q4"]["n"],
        )
        # Verdict on the "veto-double-opposed" policy (allows ALL except both_opposed)
        out[test_tf]["veto_verdict"] = walk_forward_verdict(
            out[test_tf]["VETO_double_opposed_blocked"]["IS"]["ev_R"],
            out[test_tf]["VETO_double_opposed_blocked"]["Q3"]["ev_R"],
            out[test_tf]["VETO_double_opposed_blocked"]["Q4"]["ev_R"],
            out[test_tf]["VETO_double_opposed_blocked"]["IS"]["n"],
            out[test_tf]["VETO_double_opposed_blocked"]["Q3"]["n"],
            out[test_tf]["VETO_double_opposed_blocked"]["Q4"]["n"],
        )
    return out


# =============================================================================
# COMBO 5 — Fragile-Edge Portfolio (variance-reduction play)
# =============================================================================
def combo5_walkforward() -> Dict[str, Any]:
    """Equal-weight portfolio of 5 component classes — each contributes 1/5
    of NOTIONAL. We simulate ALL signals from each component's *default* config
    on 1h (or each component's previously-validated TF) and aggregate.

    Components (default-validated configs):
      A. HA bias 4h — engulfing reversals where HA aligned (combo4 'allow_aligned')
         on test_tf=15m
      B. EMA200_d — engulfings allowed where e200d aligned (test_tf=15m)
      C. MACD div on 15m
      D. Volume Climax 1h
      E. Double Top/Bottom on 1h
    """
    portfolio: Dict[str, Dict[str, List[Dict[str, Any]]]] = {q: {} for q in QUARTER_BOUNDS}
    portfolio_all: List[Dict[str, Any]] = []
    component_stats: Dict[str, Dict[str, Any]] = {}

    # COMPONENT A — HA-aligned engulfings on 15m
    a_trades: List[Dict[str, Any]] = []
    for sym in SYMBOLS:
        sigs = run_combo4_symbol(sym, "15m", ha_n=3)
        # Component A keeps only HA-aligned (single regime, not double)
        a_trades.extend([t for t in sigs if t["ha_align"] == "ALIGNED"])
    component_stats["A_HA_aligned_engulfing"] = {
        "IS": aggregate([t for t in a_trades if quarter_of(t["entry_ts"]) in ("Q1", "Q2")]),
        "Q3": aggregate([t for t in a_trades if quarter_of(t["entry_ts"]) == "Q3"]),
        "Q4": aggregate([t for t in a_trades if quarter_of(t["entry_ts"]) == "Q4"]),
    }

    # COMPONENT B — EMA200d-aligned engulfings on 15m
    b_trades: List[Dict[str, Any]] = []
    for sym in SYMBOLS:
        sigs = run_combo4_symbol(sym, "15m", ha_n=3)
        b_trades.extend([t for t in sigs if t["e200d_align"] == "ALIGNED"])
    component_stats["B_EMA200d_aligned_engulfing"] = {
        "IS": aggregate([t for t in b_trades if quarter_of(t["entry_ts"]) in ("Q1", "Q2")]),
        "Q3": aggregate([t for t in b_trades if quarter_of(t["entry_ts"]) == "Q3"]),
        "Q4": aggregate([t for t in b_trades if quarter_of(t["entry_ts"]) == "Q4"]),
    }

    # COMPONENT C — MACD div on 15m using EB exit
    c_trades: List[Dict[str, Any]] = []
    for sym in SYMBOLS:
        df_test = load_tf(sym, "15m")
        if df_test is None:
            continue
        work = add_macd(df_test)
        macd_div = detect_divergence_series(work, "macd", lookback=20, pivot_lb=2)
        atr_test = add_atr(df_test, ATR_PERIOD)
        tf_min = tf_minutes("15m")
        for i in range(len(df_test)):
            v = macd_div.iloc[i]
            if not v:
                continue
            side = "LONG" if v == "regular_bull" else "SHORT"
            atr_v = float(atr_test.iloc[i])
            t = simulate_trade(df_test, i, side, atr_v, EXITS["EB"], tf_min)
            if t:
                t["symbol"] = sym; t["component"] = "C_MACD_div_15m"
                c_trades.append(t)
    component_stats["C_MACD_div_15m"] = {
        "IS": aggregate([t for t in c_trades if quarter_of(t["entry_ts"]) in ("Q1", "Q2")]),
        "Q3": aggregate([t for t in c_trades if quarter_of(t["entry_ts"]) == "Q3"]),
        "Q4": aggregate([t for t in c_trades if quarter_of(t["entry_ts"]) == "Q4"]),
    }

    # COMPONENT D — Volume Climax 1h, EB exit
    d_trades: List[Dict[str, Any]] = []
    for sym in SYMBOLS:
        df_climax = load_tf(sym, "1h")
        if df_climax is None:
            continue
        climax = detect_climax_series(df_climax, min_rel_vol=2.5, lookback=20, body_max=0.4)
        atr_test = add_atr(df_climax, ATR_PERIOD)
        tf_min = tf_minutes("1h")
        for i in range(len(df_climax)):
            v = climax.iloc[i]
            if not v:
                continue
            atr_v = float(atr_test.iloc[i])
            t = simulate_trade(df_climax, i, v, atr_v, EXITS["EB"], tf_min)
            if t:
                t["symbol"] = sym; t["component"] = "D_VolClimax_1h"
                d_trades.append(t)
    component_stats["D_VolClimax_1h"] = {
        "IS": aggregate([t for t in d_trades if quarter_of(t["entry_ts"]) in ("Q1", "Q2")]),
        "Q3": aggregate([t for t in d_trades if quarter_of(t["entry_ts"]) == "Q3"]),
        "Q4": aggregate([t for t in d_trades if quarter_of(t["entry_ts"]) == "Q4"]),
    }

    # COMPONENT E — Double Top/Bottom on 1h, EB exit
    e_trades: List[Dict[str, Any]] = []
    for sym in SYMBOLS:
        df_test = load_tf(sym, "1h")
        if df_test is None:
            continue
        atr_test = add_atr(df_test, ATR_PERIOD)
        tf_min = tf_minutes("1h")
        for side in ("SHORT", "LONG"):
            sigs = detect_double_signals(df_test, atr_test, pivot_lb=5,
                                         detect_window=50, side=side)
            for sig in sigs:
                i = int(sig["entry_idx"])
                atr_v = float(atr_test.iloc[i])
                t = simulate_trade(df_test, i, side, atr_v, EXITS["EB"], tf_min)
                if t:
                    t["symbol"] = sym; t["component"] = "E_Double_1h"
                    e_trades.append(t)
    component_stats["E_Double_1h"] = {
        "IS": aggregate([t for t in e_trades if quarter_of(t["entry_ts"]) in ("Q1", "Q2")]),
        "Q3": aggregate([t for t in e_trades if quarter_of(t["entry_ts"]) == "Q3"]),
        "Q4": aggregate([t for t in e_trades if quarter_of(t["entry_ts"]) == "Q4"]),
    }

    # Build portfolio: 1/5 NOTIONAL per component (so divide each $ amount by 5)
    portfolio_all = []
    for trades, comp_label in [(a_trades, "A"), (b_trades, "B"),
                                (c_trades, "C"), (d_trades, "D"),
                                (e_trades, "E")]:
        for t in trades:
            tt = dict(t)
            tt["net_$_portfolio"] = tt["net_$"] / 5.0
            tt["gross_$_portfolio"] = tt["gross_$"] / 5.0
            tt["net_R_portfolio"] = tt["net_R"]   # R is invariant to NOTIONAL
            tt["component_label"] = comp_label
            portfolio_all.append(tt)

    is_portfolio = [t for t in portfolio_all if quarter_of(t["entry_ts"]) in ("Q1", "Q2")]
    q3_portfolio = [t for t in portfolio_all if quarter_of(t["entry_ts"]) == "Q3"]
    q4_portfolio = [t for t in portfolio_all if quarter_of(t["entry_ts"]) == "Q4"]

    def agg_pf(trades):
        # average the per-trade R since each is 1/5 of notional;
        # report total $ as sum-of-net_$_portfolio
        if not trades:
            return {"n": 0, "wr": 0.0, "ev_R": 0.0,
                    "net_R": 0.0, "net_$_portfolio": 0.0, "gross_$_portfolio": 0.0}
        n = len(trades)
        wins = sum(1 for t in trades if t["net_R_portfolio"] > 0)
        return {
            "n": n, "wr": wins / n,
            "ev_R": float(np.mean([t["net_R_portfolio"] for t in trades])),
            "net_R": float(np.sum([t["net_R_portfolio"] for t in trades])),
            "net_$_portfolio": float(np.sum([t["net_$_portfolio"] for t in trades])),
            "gross_$_portfolio": float(np.sum([t["gross_$_portfolio"] for t in trades])),
        }

    pf_is = agg_pf(is_portfolio)
    pf_q3 = agg_pf(q3_portfolio)
    pf_q4 = agg_pf(q4_portfolio)

    # Compute drawdown for portfolio (rough — running sum of net_$_portfolio
    # in entry-ts order)
    sorted_all = sorted(portfolio_all, key=lambda t: t["entry_ts"])
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for t in sorted_all:
        cum += t["net_$_portfolio"]
        peak = max(peak, cum)
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "components": component_stats,
        "portfolio": {
            "IS": pf_is, "Q3": pf_q3, "Q4": pf_q4,
            "max_drawdown_$": max_dd,
            "verdict": walk_forward_verdict(
                pf_is["ev_R"], pf_q3["ev_R"], pf_q4["ev_R"],
                pf_is["n"], pf_q3["n"], pf_q4["n"],
            ),
        },
    }


# =============================================================================
# Reporting
# =============================================================================
def fmt_cell(cell: Dict[str, Any]) -> str:
    if not cell:
        return "(no data)"
    is_n = cell["IS"]["n"]; is_ev = cell["IS"]["ev_R"]
    q3_n = cell["Q3"]["n"]; q3_ev = cell["Q3"]["ev_R"]
    q4_n = cell["Q4"]["n"]; q4_ev = cell["Q4"]["ev_R"]
    v = cell.get("verdict", {})
    return (f"IS n={is_n} ev={is_ev:+.4f}R | "
            f"Q3 n={q3_n} ev={q3_ev:+.4f}R gap={v.get('q3_gap_pct', 0):.0f}% | "
            f"Q4 n={q4_n} ev={q4_ev:+.4f}R gap={v.get('q4_gap_pct', 0):.0f}% "
            f"-> {v.get('verdict', '?')}")


def write_report(results: Dict[str, Any]) -> None:
    lines = ["# 5-Way Combo Walk-Forward Report",
             "",
             "Hypothesis: confluence stacks of HOLD-class signals can produce real edge.",
             "Walk-forward: Q1+Q2 IS, Q3+Q4 OOS. Pass: OOS EV >= 50% of IS EV, same sign,",
             "n>=30 IS, n>=20 OOS each.",
             ""]

    # COMBO 1
    lines.append("## COMBO 1 — Exhaustion Confluence Stack (MACD div + RSI div + Vol Climax 1h)")
    c1 = results["combo1"]["best_per_exit"]
    for ex in ("EA", "EB", "EC", "ED"):
        c = c1.get(ex)
        if c is None:
            lines.append(f"- {ex}: (no cells)"); continue
        lines.append(
            f"- {ex} best: test_tf={c['test_tf']} climax_tf={c['climax_tf']} "
            f"div_lb={c['div_lookback']} climax_min_rel={c['climax_min_rel']}: {fmt_cell(c)}"
        )
    lines.append("")

    # COMBO 2
    lines.append("## COMBO 2 — Wedge x HA Bias 4h alignment")
    c2 = results["combo2"]["best_per_exit"]
    for ex in ("EA", "EB", "EC", "ED"):
        c = c2.get(ex)
        if c is None:
            lines.append(f"- {ex}: (no cells)"); continue
        ha_lb = c.get('ha_lookback_4h', '?')
        lines.append(
            f"- {ex} best: test_tf={c['test_tf']} wedge_tf={c['wedge_tf']} "
            f"ha_n={c['ha_n']} ha_lookback_4h={ha_lb}: {fmt_cell(c)}"
        )
    lines.append("")

    # COMBO 3
    lines.append("## COMBO 3 — Double Pattern x HTF EMA21/50 1h confluence")
    c3 = results["combo3"]["best_per_exit"]
    for ex in ("EA", "EB", "EC", "ED"):
        c = c3.get(ex)
        if c is None:
            lines.append(f"- {ex}: (no cells)"); continue
        lines.append(f"- {ex} best: test_tf={c['test_tf']}: {fmt_cell(c)}")
    lines.append("")

    # COMBO 4
    lines.append("## COMBO 4 — Regime Double-Confirm (HA 4h + EMA200_d) on engulfing cohort")
    for tf in ("5m", "15m"):
        d = results["combo4"].get(tf, {})
        lines.append(f"### Engulfing entry TF: {tf}")
        for k in ("ALL_trades", "VETO_double_opposed_blocked", "ALLOW_double_aligned_only"):
            block = d.get(k)
            if block is None:
                continue
            is_a = block["IS"]; q3 = block["Q3"]; q4 = block["Q4"]
            lines.append(
                f"- {k}: IS n={is_a['n']} ev={is_a['ev_R']:+.4f}R | "
                f"Q3 n={q3['n']} ev={q3['ev_R']:+.4f}R | "
                f"Q4 n={q4['n']} ev={q4['ev_R']:+.4f}R"
            )
        if "veto_verdict" in d:
            v = d["veto_verdict"]
            lines.append(f"- VETO verdict: {v['verdict']} "
                         f"(IS {v['is_ev']:+.4f}R / Q3 {v['q3_ev']:+.4f}R "
                         f"gap {v['q3_gap_pct']:.0f}% / Q4 {v['q4_ev']:+.4f}R "
                         f"gap {v['q4_gap_pct']:.0f}%)")
        if "aligned_verdict" in d:
            v = d["aligned_verdict"]
            lines.append(f"- ALIGNED-ONLY verdict: {v['verdict']} "
                         f"(IS {v['is_ev']:+.4f}R / Q3 {v['q3_ev']:+.4f}R / Q4 {v['q4_ev']:+.4f}R)")
    lines.append("")

    # COMBO 5
    lines.append("## COMBO 5 — Fragile-Edge Portfolio (1/5 notional per component)")
    pf = results["combo5"]["portfolio"]
    is_p = pf["IS"]; q3_p = pf["Q3"]; q4_p = pf["Q4"]; v = pf["verdict"]
    lines.append(
        f"- Portfolio: IS n={is_p['n']} ev={is_p['ev_R']:+.4f}R | "
        f"Q3 n={q3_p['n']} ev={q3_p['ev_R']:+.4f}R | "
        f"Q4 n={q4_p['n']} ev={q4_p['ev_R']:+.4f}R -> {v['verdict']}"
    )
    lines.append(f"- Max drawdown: ${pf['max_drawdown_$']:.2f}")
    for comp, stats in results["combo5"]["components"].items():
        is_a = stats["IS"]; q3 = stats["Q3"]; q4 = stats["Q4"]
        lines.append(f"  - {comp}: IS n={is_a['n']} ev={is_a['ev_R']:+.4f}R | "
                     f"Q3 n={q3['n']} ev={q3['ev_R']:+.4f}R | "
                     f"Q4 n={q4['n']} ev={q4['ev_R']:+.4f}R")
    lines.append("")

    (OUT_DIR / "report.md").write_text("\n".join(lines))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("=" * 60)
    print("5-Way Combo Walk-Forward Backtest")
    print("=" * 60)

    print("[1/5] COMBO 1 — Exhaustion Confluence Stack...")
    c1 = combo1_walkforward()
    print(f"      cells={len(c1['all_cells'])} best_exits={list(c1['best_per_exit'].keys())}")

    print("[2/5] COMBO 2 — Wedge x HA bias...")
    c2 = combo2_walkforward()
    print(f"      cells={len(c2['all_cells'])}")

    print("[3/5] COMBO 3 — Double Pattern x HTF...")
    c3 = combo3_walkforward()
    print(f"      cells={len(c3['all_cells'])}")

    print("[4/5] COMBO 4 — Regime double-confirm on engulfings...")
    c4 = combo4_walkforward()

    print("[5/5] COMBO 5 — Fragile-edge portfolio...")
    c5 = combo5_walkforward()

    # Convert pd.Timestamps to ISO strings for JSON serialization
    def _coerce(o):
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, dict):
            return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_coerce(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    results = {"combo1": c1, "combo2": c2, "combo3": c3, "combo4": c4, "combo5": c5}
    results = _coerce(results)

    out_json = OUT_DIR / "walkforward.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote: {out_json}")

    write_report(results)
    print(f"Wrote: {OUT_DIR / 'report.md'}")

    # Verdict summary
    print("\n=" * 60)
    print("VERDICT SUMMARY")
    print("=" * 60)
    for combo_key in ("combo1", "combo2", "combo3"):
        best = results[combo_key].get("best_per_exit", {})
        for ex_lbl, c in best.items():
            v = c.get("verdict", {})
            print(f"  {combo_key} {ex_lbl} test_tf={c.get('test_tf','?')}: {v.get('verdict','?')}"
                  f" IS {v.get('is_ev',0):+.3f}R Q3 {v.get('q3_ev',0):+.3f}R "
                  f"Q4 {v.get('q4_ev',0):+.3f}R")
    for tf, blk in results["combo4"].items():
        if isinstance(blk, dict):
            v = blk.get("veto_verdict", {})
            v2 = blk.get("aligned_verdict", {})
            print(f"  combo4 tf={tf} VETO: {v.get('verdict','?')} | ALIGNED-ONLY: {v2.get('verdict','?')}")
    pf = results["combo5"]["portfolio"]
    print(f"  combo5 portfolio verdict: {pf['verdict']['verdict']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Williams Fractals × 100MA Deep-Cross × Stochastic Crossover - Walk-forward backtest.

Tests three candidate signals/filters on cached 6-month data with the standard
walk-forward protocol:
  Q1+Q2 (in-sample tuning) → Q3 (OOS) → Q4 (OOS)
  Pass = OOS EV ≥ 50% of IS EV AND same sign (positive). KILL if either OOS gap > 50%.

Candidates:
  A. Williams Fractals: boost layer on engulfing-reversal cohort. Tests whether
     a confirmed prior-bar fractal in signal direction lifts cohort EV.
  B. 100MA Deep-Cross veto: blocks engulfing reversals that crossed the EMA100
     against signal direction within last N bars.
  C. Stochastic Crossover: standalone reversal scanner using %K/%D crosses
     from oversold/overbought, with optional EMA50 trend alignment.

Outputs:
  storage/fractals_100ma_stoch/walkforward.json
  storage/fractals_100ma_stoch/report.md

Read-only on:
  bot/signal_tracker.py, bot/signal_journey.py, bot/signal_learner.py.
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
OUT_DIR = ROOT / "storage" / "fractals_100ma_stoch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# 4 exit configs from spec
EXITS = {
    "EA": {"tp": 1.5, "sl": 1.0, "time_min": 30, "trail": False},
    "EB": {"tp": 2.0, "sl": 1.0, "time_min": 60, "trail": False},
    "EC": {
        "tp": 99.0,
        "sl": 1.0,
        "time_min": 4 * 60,
        "trail": True,
        "trail_trigger": 1.0,
        "trail_lock_pct": 0.5,
    },
    "ED": {"tp": 2.5, "sl": 1.0, "time_min": 4 * 60, "trail": False},
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
# Helpers
# -----------------------------------------------------------------------------
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)


def quarter_of(ts: pd.Timestamp) -> Optional[str]:
    for q, (s, e) in QUARTER_BOUNDS.items():
        if pd.Timestamp(s, tz="UTC") <= ts < pd.Timestamp(e, tz="UTC"):
            return q
    return None


def aggregate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "wr": 0.0, "ev_R": 0.0, "ev_$": 0.0,
                "gross_R": 0.0, "net_R": 0.0,
                "gross_$": 0.0, "net_$": 0.0, "avg_mfe_R": 0.0}
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
# Trade simulator (re-used across all 3 candidates)
# -----------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr_v: float,
    exit_cfg: Dict[str, Any],
    tf_min: int,
) -> Optional[Dict[str, Any]]:
    if entry_idx + 1 >= len(df):
        return None
    entry = float(df["close"].iloc[entry_idx])
    if not np.isfinite(atr_v) or atr_v <= 0:
        return None

    sl_dist = exit_cfg["sl"] * atr_v
    tp_dist = exit_cfg["tp"] * atr_v
    bars_max = max(1, int(math.ceil(exit_cfg["time_min"] / tf_min)))

    if side == "LONG":
        sl_p = entry - sl_dist
        tp_p = entry + tp_dist
    else:
        sl_p = entry + sl_dist
        tp_p = entry - tp_dist

    trail_active = False
    trail_stop: Optional[float] = None
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
        "atr": atr_v,
    }


# -----------------------------------------------------------------------------
# Cohort detector: engulfing reversal candles (re-used by Fractals + 100MA tests)
# -----------------------------------------------------------------------------
def detect_engulfing_reversals(
    df: pd.DataFrame, atr: pd.Series
) -> List[Dict[str, Any]]:
    """Bullish engulfing → LONG; bearish engulfing → SHORT.

    Bullish engulfing: prior candle red, current green, current body engulfs prior body.
    Bearish engulfing: prior candle green, current red, current body engulfs prior body.

    Entry at close of the engulfing bar (signal bar). ATR from the same bar.
    """
    o = df["open"].astype(float).values
    c = df["close"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)
    sigs: List[Dict[str, Any]] = []
    for i in range(VOL_LOOKBACK + 1, n - 1):
        atr_v = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else float("nan")
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue
        # Body sizes
        prior_body = abs(c[i - 1] - o[i - 1])
        cur_body = abs(c[i] - o[i])
        if cur_body < 0.5 * atr_v:
            continue  # require meaningful body
        # Bullish engulf: prior red (c[i-1]<o[i-1]), cur green (c[i]>o[i])
        # current body engulfs prior body
        prior_red = c[i - 1] < o[i - 1]
        prior_green = c[i - 1] > o[i - 1]
        cur_green = c[i] > o[i]
        cur_red = c[i] < o[i]
        if prior_red and cur_green and o[i] <= c[i - 1] and c[i] >= o[i - 1] and cur_body >= prior_body:
            sigs.append({"entry_idx": i, "side": "LONG", "atr": atr_v})
        elif prior_green and cur_red and o[i] >= c[i - 1] and c[i] <= o[i - 1] and cur_body >= prior_body:
            sigs.append({"entry_idx": i, "side": "SHORT", "atr": atr_v})
    return sigs


# -----------------------------------------------------------------------------
# A. Williams Fractals
# -----------------------------------------------------------------------------
def find_fractals(df: pd.DataFrame, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (top_fractal_indices, bottom_fractal_indices).

    A bar i is a TOP fractal (period=N) if highs[i] is strictly the max of
    [i-N..i+N]. (For Williams period=2: 5-bar window, i±2.)
    BOTTOM fractal: lows[i] strictly the min over the same window.

    Note: a fractal at index i is only confirmed once we've seen `period` more
    bars. We return the index where the fractal occurred, not where confirmed.
    """
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)
    tops: List[int] = []
    bots: List[int] = []
    for i in range(period, n - period):
        h_win = h[i - period : i + period + 1]
        l_win = l[i - period : i + period + 1]
        if h[i] == h_win.max() and (h_win == h[i]).sum() == 1:
            tops.append(i)
        if l[i] == l_win.min() and (l_win == l[i]).sum() == 1:
            bots.append(i)
    return np.array(tops, dtype=int), np.array(bots, dtype=int)


def fractal_confirms_signal(
    df_tf: pd.DataFrame,
    signal_ts: pd.Timestamp,
    side: str,
    period: int,
    recency: int,
) -> bool:
    """Check if there is a CONFIRMED fractal in `signal_ts` direction in the
    last `recency` bars of `df_tf` PRIOR to the signal_ts.

    A fractal at index k is confirmed at index k+period (window must be
    fully formed). Recency means: confirmation index must be in the last
    `recency` bars before/at signal_ts.

    LONG signal: bottom fractal (price made a low and reversed up).
    SHORT signal: top fractal.
    """
    # Find the last bar in df_tf with timestamp <= signal_ts
    idx = df_tf.index.searchsorted(signal_ts, side="right") - 1
    if idx < 0 or idx >= len(df_tf):
        return False

    if not hasattr(df_tf, "_fractals_cache"):
        return False  # safety

    tops, bots = df_tf._fractals_cache.get(period, (np.array([]), np.array([])))
    targets = bots if side == "LONG" else tops
    if len(targets) == 0:
        return False
    # confirmation_idx = fractal_idx + period; we want confirmation_idx in
    # [idx - recency + 1, idx], AND fractal_idx < idx (prior bar restriction).
    conf_idxs = targets + period
    mask = (conf_idxs >= idx - recency + 1) & (conf_idxs <= idx) & (targets < idx)
    return bool(mask.any())


def attach_fractals_cache(df: pd.DataFrame, periods: List[int]) -> None:
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for p in periods:
        cache[p] = find_fractals(df, p)
    # Hack: attach onto the df object as a private attr.
    df._fractals_cache = cache  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# B. 100MA deep-cross detector
# -----------------------------------------------------------------------------
def deep_cross_against(
    df_tf: pd.DataFrame,
    ema100: pd.Series,
    signal_ts: pd.Timestamp,
    side: str,
    n_bars: int,
) -> bool:
    """Did price recently CROSS the EMA100 against the signal direction?

    SHORT signal: was there a recent close ABOVE EMA100? (price already crossed
      up against our short -> entering late on a counter-trend rally).
    LONG signal: was there a recent close BELOW EMA100?

    'Recent' = within last n_bars of df_tf prior to signal_ts.

    Returns True if cross detected (so caller should VETO).
    """
    idx = df_tf.index.searchsorted(signal_ts, side="right") - 1
    if idx < 0 or idx >= len(df_tf):
        return False
    start = max(0, idx - n_bars + 1)
    closes = df_tf["close"].astype(float).iloc[start : idx + 1].values
    ema_vals = ema100.iloc[start : idx + 1].values
    if len(closes) == 0 or np.isnan(ema_vals).any():
        return False
    if side == "SHORT":
        # Recent close above EMA100 = price crossed up against short
        return bool(np.any(closes > ema_vals))
    else:
        return bool(np.any(closes < ema_vals))


# -----------------------------------------------------------------------------
# C. Stochastic crossover
# -----------------------------------------------------------------------------
def stochastic(df: pd.DataFrame, k_period: int = 14, k_smooth: int = 3, d_period: int = 3
               ) -> Tuple[pd.Series, pd.Series]:
    """%K(14,3) %D(3). Standard formula: K_raw = 100*(close - lowest_n)/(highest_n - lowest_n),
    smoothed by SMA(k_smooth)."""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    ll = l.rolling(k_period, min_periods=k_period).min()
    hh = h.rolling(k_period, min_periods=k_period).max()
    rng = (hh - ll).replace(0, np.nan)
    k_raw = 100.0 * (c - ll) / rng
    k = k_raw.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def detect_stoch_crosses(
    df: pd.DataFrame,
    atr: pd.Series,
    oversold: float,
    overbought: float,
    require_trend: bool,
) -> List[Dict[str, Any]]:
    """LONG: %K crosses up through %D AND prior %K < oversold AND (close > EMA50 if require_trend).
    SHORT: %K crosses down through %D AND prior %K > overbought AND (close < EMA50 if require_trend).
    """
    k, d = stochastic(df)
    e50 = ema(df["close"].astype(float), 50)
    o = df["open"].astype(float).values
    c = df["close"].astype(float).values
    n = len(df)
    sigs: List[Dict[str, Any]] = []
    for i in range(60, n - 1):
        atr_v = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else float("nan")
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue
        k0, k1 = float(k.iloc[i - 1]), float(k.iloc[i])
        d0, d1 = float(d.iloc[i - 1]), float(d.iloc[i])
        if not (np.isfinite(k0) and np.isfinite(k1) and np.isfinite(d0) and np.isfinite(d1)):
            continue
        # Bull cross: k crosses up through d, prior bar k <= d, current bar k > d
        bull_cross = k0 <= d0 and k1 > d1
        bear_cross = k0 >= d0 and k1 < d1
        e50v = float(e50.iloc[i])
        if bull_cross and k0 < oversold:
            if require_trend and not (c[i] > e50v):
                continue
            sigs.append({"entry_idx": i, "side": "LONG", "atr": atr_v})
        elif bear_cross and k0 > overbought:
            if require_trend and not (c[i] < e50v):
                continue
            sigs.append({"entry_idx": i, "side": "SHORT", "atr": atr_v})
    return sigs


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_dfs(tf: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        p = CACHE / f"{sym}_USDT_{tf}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d.index = pd.to_datetime(d.index, utc=True)
        out[sym] = d
    return out


# -----------------------------------------------------------------------------
# A. Williams Fractals walk-forward run
# -----------------------------------------------------------------------------
def run_fractals(per_cell: List[Dict[str, Any]]) -> None:
    """Test fractal confirmation as a boost layer on engulfing reversals.

    Cohort: engulfing reversal candles on entry_tf in {5m, 15m}.
    For each, check whether fractal confirmed on confirm_tf in {5m, 15m, 1h}
    in the prior `recency` bars with `period`-bar fractal definition.
    Compare confirmed cohort vs non-confirmed cohort EV across exits and quarters.
    """
    PERIODS = [2, 3]
    RECENCIES = [5]
    ENTRY_TFS = ["5m", "15m"]
    CONFIRM_TFS = ["5m", "15m", "1h"]

    print("[Fractals] Loading data...", flush=True)
    dfs_by_tf: Dict[str, Dict[str, pd.DataFrame]] = {}
    for tf in set(ENTRY_TFS + CONFIRM_TFS):
        dfs_by_tf[tf] = load_dfs(tf)

    # Pre-attach fractal caches for all confirm TFs at all periods
    for tf in CONFIRM_TFS:
        for sym, d in dfs_by_tf[tf].items():
            attach_fractals_cache(d, PERIODS)

    for entry_tf in ENTRY_TFS:
        tf_min = tf_minutes(entry_tf)
        for sym, df in dfs_by_tf[entry_tf].items():
            atr = add_atr(df)
            engulfs = detect_engulfing_reversals(df, atr)
            for confirm_tf, period, recency in product(CONFIRM_TFS, PERIODS, RECENCIES):
                df_conf = dfs_by_tf[confirm_tf].get(sym)
                if df_conf is None:
                    continue
                # Bucket trades by confirmed/not
                for split in ("confirmed", "not_confirmed"):
                    per_q: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                        ek: {q: [] for q in QUARTER_BOUNDS} for ek in EXITS
                    }
                    for sig in engulfs:
                        signal_ts = df.index[sig["entry_idx"]]
                        confirmed = fractal_confirms_signal(
                            df_conf, signal_ts, sig["side"], period, recency
                        )
                        if (split == "confirmed") != confirmed:
                            continue
                        for ek, ec in EXITS.items():
                            t = simulate_trade(
                                df, sig["entry_idx"], sig["side"], sig["atr"], ec, tf_min
                            )
                            if t is None:
                                continue
                            q = quarter_of(t["entry_ts"])
                            if q is None:
                                continue
                            per_q[ek][q].append(t)
                    for ek in EXITS:
                        cell = {
                            "candidate": "fractals",
                            "split": split,
                            "entry_tf": entry_tf,
                            "confirm_tf": confirm_tf,
                            "period": period,
                            "recency": recency,
                            "symbol": sym,
                            "exit": ek,
                            "params_key": (
                                f"fractals|{entry_tf}|conf{confirm_tf}|p{period}|r{recency}"
                                f"|{ek}|{split}|{sym}"
                            ),
                        }
                        is_trades = per_q[ek]["Q1"] + per_q[ek]["Q2"]
                        cell["IS"] = aggregate(is_trades)
                        cell["Q3"] = aggregate(per_q[ek]["Q3"])
                        cell["Q4"] = aggregate(per_q[ek]["Q4"])
                        per_cell.append(cell)


# -----------------------------------------------------------------------------
# B. 100MA deep-cross veto walk-forward run
# -----------------------------------------------------------------------------
def run_100ma_veto(per_cell: List[Dict[str, Any]]) -> None:
    """Test 100MA deep-cross as a veto on engulfing reversals.

    For each engulfing reversal on entry_tf in {5m, 15m}:
      Check 1h or 4h EMA100. If price closed on the OPPOSITE side of EMA100
      within last N bars (against signal direction), veto.

    Compare 'kept' vs 'blocked' cohort across quarters.
    """
    N_BARS = [3, 5, 10]
    HTF_LIST = ["1h", "4h"]
    ENTRY_TFS = ["5m", "15m"]

    print("[100MA] Loading data...", flush=True)
    dfs_entry: Dict[str, Dict[str, pd.DataFrame]] = {tf: load_dfs(tf) for tf in ENTRY_TFS}
    dfs_htf: Dict[str, Dict[str, pd.DataFrame]] = {tf: load_dfs(tf) for tf in HTF_LIST}
    # Pre-compute EMA100 per HTF per symbol
    ema100_by: Dict[Tuple[str, str], pd.Series] = {}
    for htf, d_by_sym in dfs_htf.items():
        for sym, d in d_by_sym.items():
            ema100_by[(htf, sym)] = ema(d["close"].astype(float), 100)

    for entry_tf in ENTRY_TFS:
        tf_min = tf_minutes(entry_tf)
        for sym, df in dfs_entry[entry_tf].items():
            atr = add_atr(df)
            engulfs = detect_engulfing_reversals(df, atr)
            for htf, n_bars in product(HTF_LIST, N_BARS):
                df_htf = dfs_htf[htf].get(sym)
                e100 = ema100_by.get((htf, sym))
                if df_htf is None or e100 is None:
                    continue
                for split in ("kept", "blocked"):
                    per_q: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                        ek: {q: [] for q in QUARTER_BOUNDS} for ek in EXITS
                    }
                    for sig in engulfs:
                        signal_ts = df.index[sig["entry_idx"]]
                        crossed = deep_cross_against(
                            df_htf, e100, signal_ts, sig["side"], n_bars
                        )
                        # 'blocked' cohort = crossed True (would be vetoed)
                        # 'kept'    cohort = crossed False (would pass)
                        if (split == "blocked") != crossed:
                            continue
                        for ek, ec in EXITS.items():
                            t = simulate_trade(
                                df, sig["entry_idx"], sig["side"], sig["atr"], ec, tf_min
                            )
                            if t is None:
                                continue
                            q = quarter_of(t["entry_ts"])
                            if q is None:
                                continue
                            per_q[ek][q].append(t)
                    for ek in EXITS:
                        cell = {
                            "candidate": "100ma_veto",
                            "split": split,
                            "entry_tf": entry_tf,
                            "htf": htf,
                            "n_bars": n_bars,
                            "symbol": sym,
                            "exit": ek,
                            "params_key": (
                                f"100ma|{entry_tf}|h{htf}|n{n_bars}|{ek}|{split}|{sym}"
                            ),
                        }
                        is_trades = per_q[ek]["Q1"] + per_q[ek]["Q2"]
                        cell["IS"] = aggregate(is_trades)
                        cell["Q3"] = aggregate(per_q[ek]["Q3"])
                        cell["Q4"] = aggregate(per_q[ek]["Q4"])
                        per_cell.append(cell)


# -----------------------------------------------------------------------------
# C. Stochastic crossover walk-forward run
# -----------------------------------------------------------------------------
def run_stoch(per_cell: List[Dict[str, Any]]) -> None:
    OS_LIST = [15, 20, 25]
    OB_LIST = [85, 80, 75]  # symmetric pair: (15,85), (20,80), (25,75)
    TFS = ["5m", "15m", "1h"]
    TREND_LIST = [True, False]

    # Symmetric pairs only (3 thresholds, not 9)
    THRESHOLD_PAIRS = list(zip(OS_LIST, OB_LIST))

    print("[Stoch] Loading data...", flush=True)
    for tf in TFS:
        tf_min = tf_minutes(tf)
        dfs = load_dfs(tf)
        for sym, df in dfs.items():
            atr = add_atr(df)
            for (os_v, ob_v), require_trend in product(THRESHOLD_PAIRS, TREND_LIST):
                sigs = detect_stoch_crosses(df, atr, os_v, ob_v, require_trend)
                per_q: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                    ek: {q: [] for q in QUARTER_BOUNDS} for ek in EXITS
                }
                for sig in sigs:
                    for ek, ec in EXITS.items():
                        t = simulate_trade(
                            df, sig["entry_idx"], sig["side"], sig["atr"], ec, tf_min
                        )
                        if t is None:
                            continue
                        q = quarter_of(t["entry_ts"])
                        if q is None:
                            continue
                        per_q[ek][q].append(t)
                for ek in EXITS:
                    cell = {
                        "candidate": "stoch",
                        "tf": tf,
                        "oversold": os_v,
                        "overbought": ob_v,
                        "require_trend": bool(require_trend),
                        "symbol": sym,
                        "exit": ek,
                        "params_key": (
                            f"stoch|{tf}|os{os_v}|ob{ob_v}|trend{int(require_trend)}|{ek}|{sym}"
                        ),
                    }
                    is_trades = per_q[ek]["Q1"] + per_q[ek]["Q2"]
                    cell["IS"] = aggregate(is_trades)
                    cell["Q3"] = aggregate(per_q[ek]["Q3"])
                    cell["Q4"] = aggregate(per_q[ek]["Q4"])
                    per_cell.append(cell)


# -----------------------------------------------------------------------------
# Walk-forward selection / verdicts
# -----------------------------------------------------------------------------
def gap_pct(oos: float, isv: float) -> float:
    if isv == 0 or not np.isfinite(isv):
        return float("inf")
    return (isv - oos) / abs(isv) * 100.0


def passes_oos(oos_ev: float, is_ev: float) -> bool:
    return is_ev > 0 and oos_ev > 0 and oos_ev >= 0.5 * is_ev


def aggregate_per_cell_across_symbols(per_cell: List[Dict[str, Any]],
                                      candidate: str,
                                      group_keys: List[str]) -> List[Dict[str, Any]]:
    """Aggregate per-symbol trades back into a per-(params,exit) bucket so we
    have the canonical IS/Q3/Q4 buckets summed across symbols.

    Returns a fresh list of cell dicts keyed by group_keys (no per-symbol
    breakdown), with re-aggregated IS/Q3/Q4 fields based on the SUMS of the
    per-symbol summaries.
    """
    # Group key: tuple of values for given keys
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for cell in per_cell:
        if cell.get("candidate") != candidate:
            continue
        gk = tuple(cell.get(k) for k in group_keys)
        if gk not in grouped:
            base = {k: cell.get(k) for k in group_keys}
            base["candidate"] = candidate
            base["IS"] = {"n": 0, "wr_num": 0, "ev_R_sum": 0.0, "ev_$_sum": 0.0,
                          "mfe_R_sum": 0.0}
            base["Q3"] = {"n": 0, "wr_num": 0, "ev_R_sum": 0.0, "ev_$_sum": 0.0,
                          "mfe_R_sum": 0.0}
            base["Q4"] = {"n": 0, "wr_num": 0, "ev_R_sum": 0.0, "ev_$_sum": 0.0,
                          "mfe_R_sum": 0.0}
            grouped[gk] = base
        merged = grouped[gk]
        for q in ("IS", "Q3", "Q4"):
            n = cell[q]["n"]
            merged[q]["n"] += n
            merged[q]["wr_num"] += int(round(cell[q]["wr"] * n))
            merged[q]["ev_R_sum"] += cell[q]["ev_R"] * n
            merged[q]["ev_$_sum"] += cell[q]["ev_$"] * n
            merged[q]["mfe_R_sum"] += cell[q]["avg_mfe_R"] * n
    # Finalize: convert sums → means
    finalized: List[Dict[str, Any]] = []
    for gk, base in grouped.items():
        for q in ("IS", "Q3", "Q4"):
            n = base[q]["n"]
            base[q] = {
                "n": n,
                "wr": base[q]["wr_num"] / n if n > 0 else 0.0,
                "ev_R": base[q]["ev_R_sum"] / n if n > 0 else 0.0,
                "ev_$": base[q]["ev_$_sum"] / n if n > 0 else 0.0,
                "avg_mfe_R": base[q]["mfe_R_sum"] / n if n > 0 else 0.0,
            }
        finalized.append(base)
    return finalized


def select_walkforward_fractals(per_cell: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For Fractals: best 'confirmed' params per (entry_tf x confirm_tf x period x exit)
    on Q1+Q2 IS EV. Then verify 'confirmed' beats 'not_confirmed' and walk-forward holds.
    """
    aggr = aggregate_per_cell_across_symbols(
        per_cell, "fractals",
        group_keys=["entry_tf", "confirm_tf", "period", "recency", "exit", "split"],
    )
    # Pivot: index = (entry_tf, confirm_tf, period, recency, exit), columns = split
    by_key: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}
    for c in aggr:
        gk = (c["entry_tf"], c["confirm_tf"], c["period"], c["recency"], c["exit"])
        by_key.setdefault(gk, {})[c["split"]] = c

    verdicts: List[Dict[str, Any]] = []
    for gk, splits in by_key.items():
        conf = splits.get("confirmed")
        nconf = splits.get("not_confirmed")
        if conf is None or nconf is None:
            continue
        if conf["IS"]["n"] < 20:
            continue  # tiny sample
        # Lift = confirmed EV - not_confirmed EV
        lift_IS = conf["IS"]["ev_R"] - nconf["IS"]["ev_R"]
        lift_Q3 = conf["Q3"]["ev_R"] - nconf["Q3"]["ev_R"]
        lift_Q4 = conf["Q4"]["ev_R"] - nconf["Q4"]["ev_R"]
        # Pass criteria: confirmed lift >= +0.05R IS AND OOS lifts hold same sign,
        # AND confirmed cohort EV positive in IS and OOS.
        is_ev = conf["IS"]["ev_R"]
        q3_ev = conf["Q3"]["ev_R"]
        q4_ev = conf["Q4"]["ev_R"]
        q3_gap = gap_pct(q3_ev, is_ev)
        q4_gap = gap_pct(q4_ev, is_ev)
        q3_pass = passes_oos(q3_ev, is_ev) and lift_Q3 > 0
        q4_pass = passes_oos(q4_ev, is_ev) and lift_Q4 > 0
        verdict = "SHIP" if (lift_IS >= 0.05 and q3_pass and q4_pass) else (
            "HOLD" if (q3_pass or q4_pass) and lift_IS >= 0.05 else "KILL"
        )
        verdicts.append({
            "candidate": "fractals",
            "entry_tf": gk[0], "confirm_tf": gk[1], "period": gk[2],
            "recency": gk[3], "exit": gk[4],
            "confirmed": conf, "not_confirmed": nconf,
            "lift_IS": lift_IS, "lift_Q3": lift_Q3, "lift_Q4": lift_Q4,
            "Q3_gap_pct": q3_gap, "Q4_gap_pct": q4_gap,
            "Q3_pass": q3_pass, "Q4_pass": q4_pass,
            "verdict": verdict,
        })
    return verdicts


def select_walkforward_100ma(per_cell: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For 100MA: 'blocked' cohort EV must be <= 'kept' cohort EV across quarters,
    AND blocked EV negative AND kept EV positive across all 3 IS+Q3+Q4 buckets.
    """
    aggr = aggregate_per_cell_across_symbols(
        per_cell, "100ma_veto",
        group_keys=["entry_tf", "htf", "n_bars", "exit", "split"],
    )
    by_key: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}
    for c in aggr:
        gk = (c["entry_tf"], c["htf"], c["n_bars"], c["exit"])
        by_key.setdefault(gk, {})[c["split"]] = c

    verdicts: List[Dict[str, Any]] = []
    for gk, splits in by_key.items():
        kept = splits.get("kept")
        blocked = splits.get("blocked")
        if kept is None or blocked is None:
            continue
        if kept["IS"]["n"] < 20 or blocked["IS"]["n"] < 5:
            continue
        # Pass: blocked EV NEGATIVE AND kept EV POSITIVE across all 3 buckets.
        kept_pos_all = all(kept[q]["ev_R"] > 0 for q in ("IS", "Q3", "Q4"))
        blocked_neg_all = all(blocked[q]["ev_R"] < 0 for q in ("IS", "Q3", "Q4"))
        # Lift (kept - blocked) for tracking
        lift_IS = kept["IS"]["ev_R"] - blocked["IS"]["ev_R"]
        lift_Q3 = kept["Q3"]["ev_R"] - blocked["Q3"]["ev_R"]
        lift_Q4 = kept["Q4"]["ev_R"] - blocked["Q4"]["ev_R"]
        # Soft pass: kept beats blocked by >= +0.10R in all 3 buckets, even if
        # signs aren't strictly negative (looser bar for documentation).
        soft_pass = all(
            kept[q]["ev_R"] > blocked[q]["ev_R"] + 0.10
            for q in ("IS", "Q3", "Q4")
        )
        if kept_pos_all and blocked_neg_all and soft_pass:
            verdict = "SHIP"
        elif soft_pass and kept["IS"]["ev_R"] > 0:
            verdict = "HOLD"
        else:
            verdict = "KILL"
        verdicts.append({
            "candidate": "100ma_veto",
            "entry_tf": gk[0], "htf": gk[1], "n_bars": gk[2], "exit": gk[3],
            "kept": kept, "blocked": blocked,
            "lift_IS": lift_IS, "lift_Q3": lift_Q3, "lift_Q4": lift_Q4,
            "kept_pos_all": kept_pos_all, "blocked_neg_all": blocked_neg_all,
            "soft_pass": soft_pass,
            "verdict": verdict,
        })
    return verdicts


def select_walkforward_stoch(per_cell: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For Stoch: pick best (tf x os/ob x trend x exit) on IS EV (n>=20),
    OOS pass = Q3 EV >= 0.5*IS EV AND Q4 EV >= 0.5*IS EV, all positive.
    """
    aggr = aggregate_per_cell_across_symbols(
        per_cell, "stoch",
        group_keys=["tf", "oversold", "overbought", "require_trend", "exit"],
    )
    verdicts: List[Dict[str, Any]] = []
    # Best per (tf x exit)
    best_per: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in aggr:
        if c["IS"]["n"] < 20:
            continue
        key = (c["tf"], c["exit"])
        cur = best_per.get(key)
        if cur is None or c["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            best_per[key] = c
    for (tf, ek), c in sorted(best_per.items()):
        is_ev = c["IS"]["ev_R"]
        q3_ev = c["Q3"]["ev_R"]
        q4_ev = c["Q4"]["ev_R"]
        q3_gap = gap_pct(q3_ev, is_ev)
        q4_gap = gap_pct(q4_ev, is_ev)
        q3_pass = passes_oos(q3_ev, is_ev)
        q4_pass = passes_oos(q4_ev, is_ev)
        verdict = "SHIP" if q3_pass and q4_pass else (
            "HOLD" if q3_pass or q4_pass else "KILL"
        )
        verdicts.append({
            "candidate": "stoch",
            "tf": tf,
            "oversold": c["oversold"], "overbought": c["overbought"],
            "require_trend": c["require_trend"], "exit": ek,
            "IS": c["IS"], "Q3": c["Q3"], "Q4": c["Q4"],
            "Q3_gap_pct": q3_gap, "Q4_gap_pct": q4_gap,
            "Q3_pass": q3_pass, "Q4_pass": q4_pass,
            "verdict": verdict,
        })
    return verdicts


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def write_report(payload: Dict[str, Any]) -> None:
    out_json = OUT_DIR / "walkforward.json"
    out_md = OUT_DIR / "report.md"
    out_json.write_text(json.dumps(payload, default=str, indent=2))

    lines: List[str] = []
    lines.append("# Williams Fractals × 100MA Deep-Cross × Stochastic — Walk-forward Backtest")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Symbols: " + ", ".join(SYMBOLS))
    lines.append("- Quarters (UTC):")
    for q, (s, e) in QUARTER_BOUNDS.items():
        lines.append(f"  - {q}: {s} → {e}")
    lines.append(f"- Notional ${NOTIONAL}, fee {RT_TAKER_FEE_BPS}% RT, funding {FUNDING_RATE_PER_8H}/8h, ATR{ATR_PERIOD}")
    lines.append("- Pass: OOS EV ≥ 50% of IS EV AND same sign (positive). Reject if either OOS gap > 50%.")
    lines.append("")

    # Counts
    fr_v = payload["walkforward"]["fractals"]
    ma_v = payload["walkforward"]["100ma_veto"]
    st_v = payload["walkforward"]["stoch"]

    def vc(arr: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            "SHIP": sum(1 for v in arr if v["verdict"] == "SHIP"),
            "HOLD": sum(1 for v in arr if v["verdict"] == "HOLD"),
            "KILL": sum(1 for v in arr if v["verdict"] == "KILL"),
        }

    lines.append("## Verdict counts")
    lines.append("")
    lines.append("| Candidate | Total | SHIP | HOLD | KILL |")
    lines.append("|-----------|-------|------|------|------|")
    for name, arr in (("Fractals (lift)", fr_v), ("100MA-veto", ma_v), ("Stoch", st_v)):
        c = vc(arr)
        lines.append(f"| {name} | {len(arr)} | {c['SHIP']} | {c['HOLD']} | {c['KILL']} |")
    lines.append("")

    # Fractals
    lines.append("## A. Williams Fractals (boost on engulfing reversal cohort)")
    lines.append("")
    lines.append("Lift = confirmed EV − not_confirmed EV (R units).")
    lines.append("")
    lines.append("| entry_tf | conf_tf | p | r | exit | C n IS | C EV IS | NC EV IS | Lift IS | C n Q3 | C EV Q3 | Lift Q3 | C n Q4 | C EV Q4 | Lift Q4 | Verdict |")
    lines.append("|----------|---------|---|---|------|--------|---------|----------|---------|--------|---------|---------|--------|---------|---------|---------|")
    for v in sorted(fr_v, key=lambda x: -x["confirmed"]["IS"]["ev_R"]):
        c = v["confirmed"]; nc = v["not_confirmed"]
        lines.append(
            f"| {v['entry_tf']} | {v['confirm_tf']} | {v['period']} | {v['recency']} | {v['exit']} "
            f"| {c['IS']['n']} | {c['IS']['ev_R']:+.3f} | {nc['IS']['ev_R']:+.3f} | {v['lift_IS']:+.3f} "
            f"| {c['Q3']['n']} | {c['Q3']['ev_R']:+.3f} | {v['lift_Q3']:+.3f} "
            f"| {c['Q4']['n']} | {c['Q4']['ev_R']:+.3f} | {v['lift_Q4']:+.3f} "
            f"| {v['verdict']} |"
        )
    lines.append("")

    # 100MA
    lines.append("## B. 100MA Deep-Cross Veto (engulfing reversal cohort)")
    lines.append("")
    lines.append("Pass = Kept EV>0 AND Blocked EV<0 across IS+Q3+Q4, AND Kept beats Blocked by ≥+0.10R in each.")
    lines.append("")
    lines.append("| entry_tf | htf | N | exit | K n IS | K EV IS | B n IS | B EV IS | K Q3 | B Q3 | K Q4 | B Q4 | Verdict |")
    lines.append("|----------|-----|---|------|--------|---------|--------|---------|------|------|------|------|---------|")
    for v in sorted(ma_v, key=lambda x: -x["kept"]["IS"]["ev_R"]):
        k = v["kept"]; b = v["blocked"]
        lines.append(
            f"| {v['entry_tf']} | {v['htf']} | {v['n_bars']} | {v['exit']} "
            f"| {k['IS']['n']} | {k['IS']['ev_R']:+.3f} "
            f"| {b['IS']['n']} | {b['IS']['ev_R']:+.3f} "
            f"| {k['Q3']['ev_R']:+.3f} | {b['Q3']['ev_R']:+.3f} "
            f"| {k['Q4']['ev_R']:+.3f} | {b['Q4']['ev_R']:+.3f} "
            f"| {v['verdict']} |"
        )
    lines.append("")

    # Stoch
    lines.append("## C. Stochastic Crossover (standalone reversal scanner)")
    lines.append("")
    lines.append("Best (params x exit) per TF; classic walk-forward.")
    lines.append("")
    lines.append("| TF | OS | OB | trend | exit | IS n | IS EV | Q3 n | Q3 EV | Q3 gap% | Q4 n | Q4 EV | Q4 gap% | Verdict |")
    lines.append("|----|----|----|-------|------|------|-------|------|-------|---------|------|-------|---------|---------|")
    for v in sorted(st_v, key=lambda x: -x["IS"]["ev_R"]):
        lines.append(
            f"| {v['tf']} | {v['oversold']} | {v['overbought']} | {v['require_trend']} | {v['exit']} "
            f"| {v['IS']['n']} | {v['IS']['ev_R']:+.3f} "
            f"| {v['Q3']['n']} | {v['Q3']['ev_R']:+.3f} | {v['Q3_gap_pct']:+.1f} "
            f"| {v['Q4']['n']} | {v['Q4']['ev_R']:+.3f} | {v['Q4_gap_pct']:+.1f} "
            f"| {v['verdict']} |"
        )
    lines.append("")

    # SHIP highlights
    ships_all = [
        *((f"FRACTALS {v['entry_tf']}/{v['confirm_tf']}/p{v['period']}/{v['exit']}", v) for v in fr_v if v["verdict"] == "SHIP"),
        *((f"100MA {v['entry_tf']}/{v['htf']}/N{v['n_bars']}/{v['exit']}", v) for v in ma_v if v["verdict"] == "SHIP"),
        *((f"STOCH {v['tf']}/OS{v['oversold']}/OB{v['overbought']}/trend{v['require_trend']}/{v['exit']}", v) for v in st_v if v["verdict"] == "SHIP"),
    ]
    lines.append(f"## SHIP highlights — {len(ships_all)} passes")
    lines.append("")
    if not ships_all:
        lines.append("_No candidates passed walk-forward._ Today's pattern continues — most candidates KILL.")
    else:
        for label, v in ships_all:
            lines.append(f"- **{label}**: verdict={v['verdict']}")
    lines.append("")

    out_md.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    print("Running fractals × 100MA × stoch walk-forward...", flush=True)
    per_cell: List[Dict[str, Any]] = []

    run_fractals(per_cell)
    print(f"  fractals: {sum(1 for c in per_cell if c.get('candidate')=='fractals')} per-cell rows", flush=True)
    run_100ma_veto(per_cell)
    print(f"  100ma:    {sum(1 for c in per_cell if c.get('candidate')=='100ma_veto')} per-cell rows", flush=True)
    run_stoch(per_cell)
    print(f"  stoch:    {sum(1 for c in per_cell if c.get('candidate')=='stoch')} per-cell rows", flush=True)

    print("Selecting walk-forward verdicts...", flush=True)
    fr_v = select_walkforward_fractals(per_cell)
    ma_v = select_walkforward_100ma(per_cell)
    st_v = select_walkforward_stoch(per_cell)

    payload = {
        "config": {
            "symbols": SYMBOLS,
            "exits": EXITS,
            "quarter_bounds": QUARTER_BOUNDS,
            "fee_bps": RT_TAKER_FEE_BPS,
            "funding_per_8h": FUNDING_RATE_PER_8H,
            "notional": NOTIONAL,
        },
        "per_cell": per_cell,
        "walkforward": {"fractals": fr_v, "100ma_veto": ma_v, "stoch": st_v},
    }
    write_report(payload)
    print(f"Wrote {OUT_DIR / 'walkforward.json'}", flush=True)
    print(f"Wrote {OUT_DIR / 'report.md'}", flush=True)
    n_total = len(fr_v) + len(ma_v) + len(st_v)
    n_ship = sum(1 for v in fr_v + ma_v + st_v if v["verdict"] == "SHIP")
    n_hold = sum(1 for v in fr_v + ma_v + st_v if v["verdict"] == "HOLD")
    n_kill = sum(1 for v in fr_v + ma_v + st_v if v["verdict"] == "KILL")
    print(f"verdicts: total={n_total} SHIP={n_ship} HOLD={n_hold} KILL={n_kill}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

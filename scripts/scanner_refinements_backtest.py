#!/usr/bin/env python3
"""Walk-forward backtest of two scanner refinements.

Refinement 1: wick-to-body ratio gate for liquidity_sweep + structure_bounce
   Thresholds tested: 0.6, 1.0, 1.5

Refinement 2: HTF (1h) HARD VETO for A+/A grade signals
   Synthetic A-grade proxy: engulfing reversal candle in chop regime AGAINST 4h trend.

Walk-forward split (cached candle range 2025-11-05 → 2026-04-17):
   Q1: Nov 2025
   Q2: Dec 2025 - Jan 2026
   Q3: Feb 2026
   Q4: Mar - Apr 2026

Usage:
   python3 scripts/scanner_refinements_backtest.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Paths ──
ROOT = Path("/home/opc/crypto-trading-bot")
CACHE = ROOT / "storage" / "candle_cache"
OUT = ROOT / "storage" / "scanner_refinements"
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
TIMEFRAMES_NEEDED = ["5m", "1h", "4h"]

# ── Walk-forward window split (chronological) ──
QUARTERS = {
    "Q1": (datetime(2025, 11, 5, tzinfo=timezone.utc),
           datetime(2025, 11, 30, 23, 59, tzinfo=timezone.utc)),
    "Q2": (datetime(2025, 12, 1, tzinfo=timezone.utc),
           datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)),
    "Q3": (datetime(2026, 2, 1, tzinfo=timezone.utc),
           datetime(2026, 2, 28, 23, 59, tzinfo=timezone.utc)),
    "Q4": (datetime(2026, 3, 1, tzinfo=timezone.utc),
           datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)),
}

# Trade simulation parameters
TP_R = 1.5
SL_R = 1.0
HOLD_BARS_5M = 12  # 60 minutes / 5m
ATR_LOOKBACK = 14
SWING_LOOKBACK = 20


# ──────────────────────────────────────────────────────────────────────────────
# Candle loading + indicators
# ──────────────────────────────────────────────────────────────────────────────

def load_candles(symbol: str, tf: str) -> pd.DataFrame:
    p = CACHE / f"{symbol}_USDT_{tf}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    # Parquet already has DatetimeIndex named 'datetime'; the 'timestamp' column
    # is an int64 unix-ms duplicate. Don't overwrite the index.
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, unit="ms")
            df = df.set_index(ts)
    # Ensure UTC tz
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Drop any 'timestamp' column to avoid confusion downstream
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    return df


def add_atr(df: pd.DataFrame, lookback: int = ATR_LOOKBACK) -> pd.DataFrame:
    df = df.copy()
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    df["atr"] = pd.Series(tr, index=df.index).rolling(lookback, min_periods=1).mean()
    return df


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    return df


def add_rel_vol(df: pd.DataFrame, win: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["rel_vol"] = df["volume"] / df["volume"].rolling(win, min_periods=1).mean()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_liquidity_sweep_candidate(df: pd.DataFrame, i: int) -> Optional[Dict]:
    """Return dict with side/wick/body/atr or None.

    Mirrors core liquidity_sweep semantics: price wicks beyond prior swing extreme,
    closes back inside (reclaim), with body ratio >= 0.55.
    """
    if i < SWING_LOOKBACK + 5:
        return None
    bar = df.iloc[i]
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    atr = bar["atr"]
    if not np.isfinite(atr) or atr <= 0:
        return None
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    body_ratio = body / rng
    if body_ratio < 0.55:  # baseline gate (existing)
        return None

    win_lo = i - SWING_LOOKBACK
    win = df.iloc[win_lo:i]
    swing_high = float(win["high"].max())
    swing_low = float(win["low"].min())

    # LONG: low pierces swing low AND closes back above
    if l < swing_low and c > swing_low:
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if lower_wick <= 0:
            return None
        return {
            "side": "long",
            "open": float(o), "high": float(h), "low": float(l), "close": float(c),
            "atr": float(atr),
            "body": float(body), "rng": float(rng),
            "upper_wick": float(upper_wick), "lower_wick": float(lower_wick),
            "rejection_wick": float(lower_wick),  # for LONG, wick below body
            "wick_body_ratio": float(lower_wick / body) if body > 0 else 0.0,
            "sweep_size_atr": float((swing_low - l) / atr),
        }

    # SHORT: high pierces swing high AND closes back below
    if h > swing_high and c < swing_high:
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if upper_wick <= 0:
            return None
        return {
            "side": "short",
            "open": float(o), "high": float(h), "low": float(l), "close": float(c),
            "atr": float(atr),
            "body": float(body), "rng": float(rng),
            "upper_wick": float(upper_wick), "lower_wick": float(lower_wick),
            "rejection_wick": float(upper_wick),
            "wick_body_ratio": float(upper_wick / body) if body > 0 else 0.0,
            "sweep_size_atr": float((h - swing_high) / atr),
        }

    return None


def detect_structure_bounce_candidate(df: pd.DataFrame, i: int) -> Optional[Dict]:
    """Lite structure_bounce: rejection candle at 20-bar swing extreme, closes back inside.

    Simulates the "rejection at S/R" pattern. Differs from liquidity_sweep in that
    this requires bouncing off a level WITHOUT necessarily piercing it deeply.
    Uses the prior 20-bar swing high/low as the S/R proxy.
    """
    if i < SWING_LOOKBACK + 5:
        return None
    bar = df.iloc[i]
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    atr = bar["atr"]
    if not np.isfinite(atr) or atr <= 0:
        return None
    rng = h - l
    if rng <= 0 or atr <= 0:
        return None
    body = abs(c - o)

    win_lo = i - SWING_LOOKBACK
    win = df.iloc[win_lo:i]
    swing_high = float(win["high"].max())
    swing_low = float(win["low"].min())

    # Approach proximity: within 0.5 ATR of swing low/high
    near_low = (l - swing_low) <= atr * 0.5 and l >= swing_low * 0.9985
    near_high = (swing_high - h) <= atr * 0.5 and h <= swing_high * 1.0015

    # SR-bounce LONG: bar approaches swing low, has lower wick, closes bullish in upper half
    if near_low:
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        # rejection criteria from current scanner: wick > 30% of range, close > 45% of range from low
        rejection_ok = lower_wick > rng * 0.30 and c > (l + rng * 0.45)
        if rejection_ok and c > o and lower_wick > 0:
            return {
                "side": "long",
                "open": float(o), "high": float(h), "low": float(l), "close": float(c),
                "atr": float(atr),
                "body": float(body), "rng": float(rng),
                "upper_wick": float(upper_wick), "lower_wick": float(lower_wick),
                "rejection_wick": float(lower_wick),
                "wick_body_ratio": float(lower_wick / body) if body > 0 else 0.0,
            }

    # SR-bounce SHORT
    if near_high:
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        rejection_ok = upper_wick > rng * 0.30 and c < (l + rng * 0.55)
        if rejection_ok and c < o and upper_wick > 0:
            return {
                "side": "short",
                "open": float(o), "high": float(h), "low": float(l), "close": float(c),
                "atr": float(atr),
                "body": float(body), "rng": float(rng),
                "upper_wick": float(upper_wick), "lower_wick": float(lower_wick),
                "rejection_wick": float(upper_wick),
                "wick_body_ratio": float(upper_wick / body) if body > 0 else 0.0,
            }

    return None


def simulate_outcome(df_5m: pd.DataFrame, i: int, side: str,
                     entry: float, sl_r: float = SL_R, tp_r: float = TP_R,
                     hold_bars: int = HOLD_BARS_5M) -> float:
    """Simulate 1.5R TP / 1R SL / hold-time exit. Returns R-multiple PnL."""
    bar = df_5m.iloc[i]
    atr = float(bar["atr"])
    if atr <= 0 or not np.isfinite(atr):
        return 0.0
    risk = atr  # 1 ATR risk for normalization
    if side == "long":
        sl = entry - risk * sl_r
        tp = entry + risk * tp_r
    else:
        sl = entry + risk * sl_r
        tp = entry - risk * tp_r

    end = min(i + 1 + hold_bars, len(df_5m))
    for j in range(i + 1, end):
        b = df_5m.iloc[j]
        bh, bl = float(b["high"]), float(b["low"])
        if side == "long":
            # SL first (conservative)
            if bl <= sl:
                return -sl_r
            if bh >= tp:
                return tp_r
        else:
            if bh >= sl:
                return -sl_r
            if bl <= tp:
                return tp_r
    # time exit at last close
    last_c = float(df_5m.iloc[end - 1]["close"])
    if side == "long":
        return (last_c - entry) / risk if risk > 0 else 0.0
    return (entry - last_c) / risk if risk > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# HTF bias (1h) via EMA21+EMA50 stack on 1h
# ──────────────────────────────────────────────────────────────────────────────

def htf_bias_1h(df_1h: pd.DataFrame, ts: pd.Timestamp) -> int:
    """+1 bull (close>ema21>ema50), -1 bear (close<ema21<ema50), 0 neutral."""
    sub = df_1h.loc[df_1h.index <= ts]
    if len(sub) < 51:
        return 0
    bar = sub.iloc[-1]
    c = float(bar["close"])
    e21 = float(bar["ema21"])
    e50 = float(bar["ema50"])
    if not (np.isfinite(c) and np.isfinite(e21) and np.isfinite(e50)):
        return 0
    if c > e21 > e50:
        return 1
    if c < e21 < e50:
        return -1
    return 0


def htf_bias_4h(df_4h: pd.DataFrame, ts: pd.Timestamp) -> int:
    sub = df_4h.loc[df_4h.index <= ts]
    if len(sub) < 51:
        return 0
    bar = sub.iloc[-1]
    c = float(bar["close"])
    e21 = float(bar["ema21"])
    e50 = float(bar["ema50"])
    if not (np.isfinite(c) and np.isfinite(e21) and np.isfinite(e50)):
        return 0
    if c > e21 > e50:
        return 1
    if c < e21 < e50:
        return -1
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Refinement 1: wick/body ratio walk-forward
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRec:
    ts: str
    symbol: str
    scanner: str
    side: str
    wick_body_ratio: float
    pnl_R: float


def collect_refinement1_trades() -> List[TradeRec]:
    """Scan all 5m bars on cached symbols, detect liq_sweep + structure_bounce
    candidates with their wick_body_ratio, simulate outcome with 1.5R TP/1R SL.
    """
    trades: List[TradeRec] = []
    for sym in SYMBOLS:
        df5 = load_candles(sym, "5m")
        df5 = add_atr(df5)
        df5 = add_rel_vol(df5)

        for i in range(SWING_LOOKBACK + 5, len(df5) - HOLD_BARS_5M - 1):
            ts = df5.index[i]
            bar_close = float(df5.iloc[i]["close"])

            # liq_sweep
            ls = detect_liquidity_sweep_candidate(df5, i)
            if ls is not None:
                pnl = simulate_outcome(df5, i, ls["side"], bar_close)
                trades.append(TradeRec(
                    ts=ts.isoformat(), symbol=sym, scanner="liq_sweep",
                    side=ls["side"],
                    wick_body_ratio=ls["wick_body_ratio"],
                    pnl_R=float(pnl),
                ))
                continue  # don't double-count same bar with sb if sweep already

            # structure_bounce (only if no sweep)
            sb = detect_structure_bounce_candidate(df5, i)
            if sb is not None:
                pnl = simulate_outcome(df5, i, sb["side"], bar_close)
                trades.append(TradeRec(
                    ts=ts.isoformat(), symbol=sym, scanner="structure_bounce",
                    side=sb["side"],
                    wick_body_ratio=sb["wick_body_ratio"],
                    pnl_R=float(pnl),
                ))

    return trades


def evaluate_threshold(trades: List[TradeRec], thr: float) -> Dict:
    """Filter to wick/body >= thr, compute n / WR / EV (mean R) / total R."""
    passing = [t for t in trades if t.wick_body_ratio >= thr]
    n = len(passing)
    if n == 0:
        return {"threshold": thr, "n": 0, "win_rate": 0.0, "ev_R": 0.0, "total_R": 0.0}
    wins = sum(1 for t in passing if t.pnl_R > 0)
    pnl_sum = sum(t.pnl_R for t in passing)
    return {
        "threshold": thr,
        "n": n,
        "win_rate": round(wins / n, 4),
        "ev_R": round(pnl_sum / n, 4),
        "total_R": round(pnl_sum, 3),
    }


def in_quarter(ts_iso: str, q: str) -> bool:
    ts = pd.Timestamp(ts_iso)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    lo, hi = QUARTERS[q]
    return lo <= ts <= hi


def filter_quarter(trades: List[TradeRec], q: str) -> List[TradeRec]:
    return [t for t in trades if in_quarter(t.ts, q)]


def walk_forward_refinement1(trades: List[TradeRec]) -> Dict:
    thresholds = [0.6, 1.0, 1.5]
    is_trades = [t for t in trades if in_quarter(t.ts, "Q1") or in_quarter(t.ts, "Q2")]
    q3_trades = filter_quarter(trades, "Q3")
    q4_trades = filter_quarter(trades, "Q4")

    # Baseline (no wick/body filter — all trades pass)
    baseline_is = evaluate_threshold(is_trades, 0.0)
    baseline_q3 = evaluate_threshold(q3_trades, 0.0)
    baseline_q4 = evaluate_threshold(q4_trades, 0.0)

    per_threshold = []
    for thr in thresholds:
        is_metrics = evaluate_threshold(is_trades, thr)
        q3_metrics = evaluate_threshold(q3_trades, thr)
        q4_metrics = evaluate_threshold(q4_trades, thr)
        # Walk-forward gap: OOS EV / IS EV ratio
        is_ev = is_metrics["ev_R"]
        q3_gap_pct = ((q3_metrics["ev_R"] - is_ev) / abs(is_ev) * 100) if is_ev not in (0, None) else None
        q4_gap_pct = ((q4_metrics["ev_R"] - is_ev) / abs(is_ev) * 100) if is_ev not in (0, None) else None
        # Pass criteria: OOS EV >= 0.5 * IS EV AND same sign (positive), for BOTH Q3 and Q4
        def passes(oos_ev: float) -> bool:
            return (is_ev > 0 and oos_ev >= 0.5 * is_ev)
        per_threshold.append({
            "threshold": thr,
            "is_q1q2": is_metrics,
            "oos_q3": q3_metrics,
            "oos_q4": q4_metrics,
            "q3_gap_vs_is_pct": round(q3_gap_pct, 2) if q3_gap_pct is not None else None,
            "q4_gap_vs_is_pct": round(q4_gap_pct, 2) if q4_gap_pct is not None else None,
            "q3_passes": passes(q3_metrics["ev_R"]),
            "q4_passes": passes(q4_metrics["ev_R"]),
        })

    return {
        "baseline_is_q1q2": baseline_is,
        "baseline_oos_q3": baseline_q3,
        "baseline_oos_q4": baseline_q4,
        "per_threshold": per_threshold,
    }


def walk_forward_refinement1_by_scanner(trades: List[TradeRec]) -> Dict:
    """Repeat walk-forward separately for liq_sweep and structure_bounce."""
    out = {}
    for scanner in ("liq_sweep", "structure_bounce"):
        sub = [t for t in trades if t.scanner == scanner]
        out[scanner] = walk_forward_refinement1(sub)
        out[scanner]["total_n_all_quarters"] = len(sub)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Refinement 2: HTF A+/A hard veto walk-forward
# ──────────────────────────────────────────────────────────────────────────────

def detect_engulfing(df: pd.DataFrame, i: int) -> Optional[str]:
    """Bullish/bearish engulfing pattern. Returns 'long'/'short' if detected."""
    if i < 1:
        return None
    cur = df.iloc[i]
    prev = df.iloc[i - 1]
    co, cc = float(cur["open"]), float(cur["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    if not (po and pc and co and cc):
        return None
    # Bullish engulfing: prev bearish, cur bullish, cur body engulfs prev body
    if pc < po and cc > co and cc >= po and co <= pc:
        return "long"
    # Bearish engulfing: prev bullish, cur bearish, cur engulfs prev
    if pc > po and cc < co and cc <= po and co >= pc:
        return "short"
    return None


def is_chop_regime_5m(df: pd.DataFrame, i: int) -> bool:
    """Crude chop proxy: ATR ratio (5-bar/20-bar) within 0.7-1.3 AND no clear ema21 trend."""
    if i < 50:
        return False
    short_atr = df.iloc[i - 5:i]["atr"].mean()
    long_atr = df.iloc[i - 20:i]["atr"].mean()
    if long_atr <= 0:
        return False
    ratio = short_atr / long_atr
    bar = df.iloc[i]
    c = float(bar["close"])
    e21 = float(bar.get("ema21", c))
    e50 = float(bar.get("ema50", c))
    # weak trend if close within 0.5 ATR of ema21 AND ema21 slope flat
    if i < 10:
        return False
    e21_then = float(df.iloc[i - 10].get("ema21", c))
    slope = abs(e21 - e21_then) / max(bar["atr"], 1e-9)
    is_flat = slope < 0.5
    near_ema = abs(c - e21) < bar["atr"] * 0.7
    return (0.7 <= ratio <= 1.3) and is_flat and near_ema


@dataclass
class CTradeRec:
    ts: str
    symbol: str
    side: str
    htf_1h_bias: int
    htf_4h_bias: int
    grade_proxy: str  # "A" if chop-engulfing-counter-HTF
    pnl_R: float


def collect_refinement2_trades() -> List[CTradeRec]:
    """Synthetic A-grade signals: engulfing reversal in chop regime AGAINST 4h trend.

    Captures the 'A-grade counter-trend reversal in chop' cohort.
    Returns ALL such candidates (regardless of HTF agreement) so we can
    counterfactual the veto.
    """
    out: List[CTradeRec] = []
    for sym in SYMBOLS:
        df5 = add_emas(add_atr(load_candles(sym, "5m")))
        df5 = add_rel_vol(df5)
        df1h = add_emas(load_candles(sym, "1h"))
        df4h = add_emas(load_candles(sym, "4h"))

        for i in range(60, len(df5) - HOLD_BARS_5M - 1):
            if not is_chop_regime_5m(df5, i):
                continue
            sig = detect_engulfing(df5, i)
            if sig is None:
                continue

            ts = df5.index[i]
            entry = float(df5.iloc[i]["close"])
            h1_bias = htf_bias_1h(df1h, ts)
            h4_bias = htf_bias_4h(df4h, ts)

            # Define A-grade proxy: engulfing reversal in chop AGAINST 4h trend
            # i.e., 4h is bull but engulfing says short (or 4h bear but engulf says long)
            against_4h = (h4_bias > 0 and sig == "short") or (h4_bias < 0 and sig == "long")
            grade_proxy = "A" if against_4h else "OTHER"

            pnl = simulate_outcome(df5, i, sig, entry)

            out.append(CTradeRec(
                ts=ts.isoformat(), symbol=sym, side=sig,
                htf_1h_bias=int(h1_bias), htf_4h_bias=int(h4_bias),
                grade_proxy=grade_proxy, pnl_R=float(pnl),
            ))
    return out


def walk_forward_refinement2(trades: List[CTradeRec]) -> Dict:
    """HARD VETO scenario: block A-grade signals where 1h HTF opposes side.

    For walk-forward, split chronologically: IS=Q1+Q2, OOS1=Q3, OOS2=Q4.

    For each split, compare:
       BASELINE (all signals taken) vs FILTERED (block A-grade where 1h HTF opposes).
    """
    def opposes_1h(t: CTradeRec) -> bool:
        if t.htf_1h_bias == 0:
            return False
        return (t.htf_1h_bias > 0 and t.side == "short") or \
               (t.htf_1h_bias < 0 and t.side == "long")

    def cohort_metrics(sub: List[CTradeRec]) -> Dict:
        n = len(sub)
        if n == 0:
            return {"n": 0, "win_rate": 0.0, "ev_R": 0.0, "total_R": 0.0}
        wins = sum(1 for t in sub if t.pnl_R > 0)
        pnl_sum = sum(t.pnl_R for t in sub)
        return {"n": n,
                "win_rate": round(wins / n, 4),
                "ev_R": round(pnl_sum / n, 4),
                "total_R": round(pnl_sum, 3)}

    def split_metrics(sub: List[CTradeRec]) -> Dict:
        agrade = [t for t in sub if t.grade_proxy == "A"]
        agrade_blocked = [t for t in agrade if opposes_1h(t)]
        agrade_kept = [t for t in agrade if not opposes_1h(t)]
        return {
            "n_total": len(sub),
            "all_signals": cohort_metrics(sub),
            "agrade_n": len(agrade),
            "agrade_blocked_by_1h_veto": cohort_metrics(agrade_blocked),
            "agrade_kept_after_veto": cohort_metrics(agrade_kept),
        }

    is_trades = [t for t in trades if in_quarter(t.ts, "Q1") or in_quarter(t.ts, "Q2")]
    q3_trades = [t for t in trades if in_quarter(t.ts, "Q3")]
    q4_trades = [t for t in trades if in_quarter(t.ts, "Q4")]

    is_m = split_metrics(is_trades)
    q3_m = split_metrics(q3_trades)
    q4_m = split_metrics(q4_trades)

    # The "edge" of the veto = how negative the BLOCKED cohort is.
    # If blocked cohort EV < 0 in IS and remains < 0 in Q3 + Q4, ship.
    is_blocked_ev = is_m["agrade_blocked_by_1h_veto"]["ev_R"]
    q3_blocked_ev = q3_m["agrade_blocked_by_1h_veto"]["ev_R"]
    q4_blocked_ev = q4_m["agrade_blocked_by_1h_veto"]["ev_R"]

    # Pass: blocked cohort negative EV in IS AND OOS Q3 AND OOS Q4
    # (and we want savings — blocking these helps).
    veto_passes = (is_blocked_ev < 0
                   and q3_blocked_ev < 0
                   and q4_blocked_ev < 0)
    # Walk-forward gap on blocked cohort EV
    def gap(a: float, b: float) -> Optional[float]:
        if a == 0:
            return None
        return round((b - a) / abs(a) * 100, 2)

    return {
        "is_q1q2": is_m,
        "oos_q3": q3_m,
        "oos_q4": q4_m,
        "blocked_ev_walk": {
            "is": is_blocked_ev, "q3": q3_blocked_ev, "q4": q4_blocked_ev,
            "q3_gap_pct": gap(is_blocked_ev, q3_blocked_ev),
            "q4_gap_pct": gap(is_blocked_ev, q4_blocked_ev),
        },
        "veto_walk_forward_pass": veto_passes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def fmt_metrics(m: Dict) -> str:
    return f"n={m['n']:>4} WR={m['win_rate']*100:5.1f}% EV={m['ev_R']:+.3f}R total={m['total_R']:+.2f}R"


def write_reports(r1: Dict, r1_by_scanner: Dict, r2: Dict) -> None:
    # JSON
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "refinement_1_combined": r1,
        "refinement_1_per_scanner": r1_by_scanner,
        "refinement_2_htf_agrade_veto": r2,
        "split_definition": {q: [str(lo), str(hi)] for q, (lo, hi) in QUARTERS.items()},
        "trade_simulation": {
            "tp_R": TP_R, "sl_R": SL_R, "hold_bars_5m": HOLD_BARS_5M,
            "atr_lookback": ATR_LOOKBACK, "swing_lookback": SWING_LOOKBACK,
        },
    }
    with open(OUT / "walkforward.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Markdown
    lines = []
    lines.append("# Scanner Refinements — Walk-Forward Backtest")
    lines.append("")
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append("## Splits")
    for q, (lo, hi) in QUARTERS.items():
        lines.append(f"- **{q}**: {lo.date()} → {hi.date()}")
    lines.append("")
    lines.append("Trade simulation: TP=1.5R, SL=1R, hold=60min (12 × 5m bars), risk=1×ATR.")
    lines.append("Pass rule: OOS EV ≥ 0.5×IS EV AND same sign (positive). Reject if OOS gap > 50%.")
    lines.append("")

    # Refinement 1 — combined
    lines.append("## Refinement 1 — Wick-to-body ratio (combined: liq_sweep + structure_bounce)")
    lines.append("")
    lines.append("Baseline (no filter):")
    lines.append(f"- IS Q1+Q2: {fmt_metrics(r1['baseline_is_q1q2'])}")
    lines.append(f"- OOS Q3:   {fmt_metrics(r1['baseline_oos_q3'])}")
    lines.append(f"- OOS Q4:   {fmt_metrics(r1['baseline_oos_q4'])}")
    lines.append("")
    lines.append("Per threshold:")
    lines.append("")
    lines.append("| thr | IS n | IS WR | IS EV | Q3 n | Q3 EV | Q3 gap% | Q4 n | Q4 EV | Q4 gap% | pass |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in r1["per_threshold"]:
        ism = row["is_q1q2"]; q3 = row["oos_q3"]; q4 = row["oos_q4"]
        passes = row["q3_passes"] and row["q4_passes"]
        lines.append(
            f"| {row['threshold']} | {ism['n']} | {ism['win_rate']*100:.1f}% | {ism['ev_R']:+.3f} "
            f"| {q3['n']} | {q3['ev_R']:+.3f} | {row['q3_gap_vs_is_pct']} "
            f"| {q4['n']} | {q4['ev_R']:+.3f} | {row['q4_gap_vs_is_pct']} | {'YES' if passes else 'NO'} |"
        )
    lines.append("")

    # Refinement 1 — per scanner
    lines.append("### Refinement 1 — Per-scanner walk-forward")
    lines.append("")
    for scanner in ("liq_sweep", "structure_bounce"):
        sub = r1_by_scanner[scanner]
        lines.append(f"#### {scanner} (total trades all quarters: {sub['total_n_all_quarters']})")
        lines.append("")
        lines.append(f"Baseline IS: {fmt_metrics(sub['baseline_is_q1q2'])}")
        lines.append(f"Baseline Q3: {fmt_metrics(sub['baseline_oos_q3'])}")
        lines.append(f"Baseline Q4: {fmt_metrics(sub['baseline_oos_q4'])}")
        lines.append("")
        lines.append("| thr | IS n | IS EV | Q3 n | Q3 EV | Q3 gap% | Q4 n | Q4 EV | Q4 gap% | pass |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in sub["per_threshold"]:
            ism = row["is_q1q2"]; q3 = row["oos_q3"]; q4 = row["oos_q4"]
            passes = row["q3_passes"] and row["q4_passes"]
            lines.append(
                f"| {row['threshold']} | {ism['n']} | {ism['ev_R']:+.3f} "
                f"| {q3['n']} | {q3['ev_R']:+.3f} | {row['q3_gap_vs_is_pct']} "
                f"| {q4['n']} | {q4['ev_R']:+.3f} | {row['q4_gap_vs_is_pct']} | {'YES' if passes else 'NO'} |"
            )
        lines.append("")

    # Refinement 2
    lines.append("## Refinement 2 — HTF (1h) HARD VETO for synthetic A-grade")
    lines.append("")
    lines.append("Synthetic A-grade proxy = engulfing reversal candle in chop regime AGAINST 4h trend.")
    lines.append("Veto fires when grade=='A' AND 1h HTF opposes side.")
    lines.append("")
    lines.append("| split | total n | A-grade n | A-grade BLOCKED n | blocked EV | blocked WR | A-grade KEPT EV |")
    lines.append("|---|---|---|---|---|---|---|")
    for split in (("IS Q1+Q2", r2["is_q1q2"]), ("OOS Q3", r2["oos_q3"]), ("OOS Q4", r2["oos_q4"])):
        name, m = split
        b = m["agrade_blocked_by_1h_veto"]
        k = m["agrade_kept_after_veto"]
        lines.append(
            f"| {name} | {m['n_total']} | {m['agrade_n']} | {b['n']} "
            f"| {b['ev_R']:+.3f} | {b['win_rate']*100:.1f}% | {k['ev_R']:+.3f} |"
        )
    lines.append("")
    bw = r2["blocked_ev_walk"]
    lines.append(f"Blocked-cohort EV walk: IS={bw['is']:+.3f}R → Q3={bw['q3']:+.3f}R "
                 f"({bw['q3_gap_pct']}%) → Q4={bw['q4']:+.3f}R ({bw['q4_gap_pct']}%)")
    lines.append(f"Veto walk-forward pass: **{'YES' if r2['veto_walk_forward_pass'] else 'NO'}**")
    lines.append("")

    # Verdicts
    lines.append("## Verdicts")
    lines.append("")
    # Refinement 1 verdict: BOTH OOS quarters must pass (per spec)
    r1_verdict = "KILL"
    r1_thr = None
    for row in r1["per_threshold"]:
        if row["q3_passes"] and row["q4_passes"]:
            r1_thr = row["threshold"]
            r1_verdict = "SHIP"
            break
    # If exactly one OOS quarter passes for some threshold AND the other is
    # only marginally negative (within 25% of zero) → HOLD (not KILL).
    if r1_verdict == "KILL":
        for row in r1["per_threshold"]:
            q3_ev = row["oos_q3"]["ev_R"]
            q4_ev = row["oos_q4"]["ev_R"]
            if (row["q3_passes"] or row["q4_passes"]) \
                    and min(q3_ev, q4_ev) > -0.05:
                r1_verdict = "HOLD"
                r1_thr = row["threshold"]
                break

    lines.append(f"**Refinement 1 (wick/body gate):** {r1_verdict}"
                 + (f" at threshold {r1_thr}" if r1_thr is not None else ""))
    if r1_verdict == "SHIP":
        lines.append(f"  - Recommend WICK_BODY_RATIO_MIN={r1_thr} (default OFF, ship dark)")

    r2_verdict = "SHIP" if r2["veto_walk_forward_pass"] else (
        "HOLD" if (r2["blocked_ev_walk"]["q3"] < 0 or r2["blocked_ev_walk"]["q4"] < 0) else "KILL"
    )
    lines.append(f"**Refinement 2 (HTF A-grade veto):** {r2_verdict}")
    lines.append("")

    with open(OUT / "report.md", "w") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[1/3] Collecting Refinement 1 trades (wick/body)...")
    r1_trades = collect_refinement1_trades()
    print(f"      n_trades={len(r1_trades)}")

    print("[2/3] Walk-forward Refinement 1...")
    r1_combined = walk_forward_refinement1(r1_trades)
    r1_by_scanner = walk_forward_refinement1_by_scanner(r1_trades)

    print("[3/3] Walk-forward Refinement 2 (HTF A-grade veto)...")
    r2_trades = collect_refinement2_trades()
    print(f"      n_synthetic_signals={len(r2_trades)}")
    r2_result = walk_forward_refinement2(r2_trades)

    print("[+] Writing reports...")
    write_reports(r1_combined, r1_by_scanner, r2_result)
    print(f"[OK] wrote {OUT/'report.md'} and {OUT/'walkforward.json'}")


if __name__ == "__main__":
    main()

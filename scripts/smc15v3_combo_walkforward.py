#!/usr/bin/env python3
"""SMC1.5v3 combo walk-forward backtest.

Tests SMC1.5v2 (WICK-entry) baseline vs three filter variants:
  - HTF-only:  + HTF A-grade veto (1h EMA21/50 stack, A-grade = full SMC stack)
  - Wedge-only: + Wedge breakout veto (1h pivots → wedge alignment)
  - Combo:    + BOTH filters

Pattern detection (verbatim from smc15_fullstack_backtest.py):
  Sweep + BOS opp + CHoCH same-as-sweep-target + zone (OB/BRK/RB)

Entry (verbatim from smc15_entry_variants_backtest.py WICK variant):
  LIMIT at OB nearest extreme. Long → zone_high, short → zone_low.
  Maker fee 0.024% on entry leg. 6h limit expiry.

Exit (EB config):
  TP 3R fixed, SL = sweep_wick + 0.05× ATR, 8h time stop.
  Taker 0.059% on exit leg.

Walk-forward:
  Q1+Q2 = IS (Oct-Nov 2025 + Dec 2025-Jan 2026)
  Q3   = OOS1 (Feb 2026)
  Q4   = OOS2 (Mar-Apr 2026)
  Pass: OOS gap <= 50%, same sign, n>=30 IS, n>=10 each OOS.

Outputs:
  storage/smc15v3_combo/walkforward.json
  storage/smc15v3_combo/report.md
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
CACHE_DIR = ROOT / "storage" / "candle_cache"
OUT_DIR = ROOT / "storage" / "smc15v3_combo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Cost model — Delta India
TAKER_FEE = 0.00059
MAKER_FEE = 0.00024
FUNDING_PER_8H = 0.0001
NOTIONAL_USD = 1000.0

# Strategy params (mirror SMC1.5v2 exactly)
SWING_LOOKBACK = 5
SWEEP_LOOKBACK = 20
BOS_LOOKBACK = 30
CHOCH_WINDOW = 15
PATTERN_WINDOW = 30
ATR_LEN = 14
DISPLACEMENT_ATR_MULT = 1.0
RB_WICK_FRAC = 0.55
ENTRY_TOL_ATR = 0.10
LIMIT_FILL_WINDOW_BARS = 6   # 6h on 1h
SL_BUFFER_ATR = 0.05

# EB exit config
TP_R = 3.0
MAX_BARS = 8

# HTF veto
HTF_EMA_FAST = 21
HTF_EMA_SLOW = 50

# Wedge detector
WEDGE_PIVOT_LB = 5
WEDGE_DETECT_WINDOW = 30
WEDGE_VOL_THRESHOLD = 1.0

QUARTERS = {
    "Q1": ("2025-10-07T00:00:00+00:00", "2025-11-30T23:59:59+00:00"),
    "Q2": ("2025-12-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00"),
    "Q3": ("2026-02-01T00:00:00+00:00", "2026-02-28T23:59:59+00:00"),
    "Q4": ("2026-03-01T00:00:00+00:00", "2026-04-17T23:59:59+00:00"),
}

VARIANTS = ["baseline", "htf_only", "wedge_only", "combo"]


# ─────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────

def load_candles(sym: str, tf: str = "1h") -> pd.DataFrame:
    p = CACHE_DIR / f"{sym}_USDT_{tf}.parquet"
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["range"] = df.high - df.low
    df["body"] = (df.close - df.open).abs()
    df["upper_wick"] = df.high - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df.low
    return df


def add_atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.DataFrame:
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df = df.copy()
    df["atr"] = tr.rolling(n).mean()
    return df


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema21"] = df["close"].ewm(span=HTF_EMA_FAST, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=HTF_EMA_SLOW, adjust=False).mean()
    return df


# ─────────────────────────────────────────────────────────
# SMC primitives
# ─────────────────────────────────────────────────────────

@dataclass
class Pivot:
    idx: int
    price: float
    kind: str
    time: pd.Timestamp


@dataclass
class Sweep:
    idx: int
    swept_pivot_idx: int
    direction: str  # bsl or ssl
    sweep_extreme: float
    swept_price: float
    time: pd.Timestamp


@dataclass
class BOS:
    idx: int
    broken_pivot_idx: int
    direction: str  # up or down
    broken_price: float
    time: pd.Timestamp


@dataclass
class CHoCH:
    idx: int
    direction: str
    pivot_idx: int
    time: pd.Timestamp


@dataclass
class OrderBlock:
    formed_idx: int
    direction: str  # bull or bear
    high: float
    low: float
    midpoint: float
    body_high: float
    body_low: float
    displacement_atr: float
    mitigated_idx: Optional[int] = None
    time: pd.Timestamp = None


@dataclass
class Breaker:
    parent_high: float
    parent_low: float
    flip_idx: int
    direction: str
    high: float
    low: float
    midpoint: float
    body_high: float
    body_low: float
    time: pd.Timestamp


@dataclass
class RejectionBlock:
    idx: int
    direction: str
    high: float
    low: float
    midpoint: float
    body_high: float
    body_low: float
    near_pivot_idx: int
    time: pd.Timestamp


def find_swing_pivots(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> list[Pivot]:
    pivots: list[Pivot] = []
    n = len(df)
    highs = df.high.values
    lows = df.low.values
    times = df.index
    for i in range(lookback, n - lookback):
        win_h = highs[i - lookback:i + lookback + 1]
        win_l = lows[i - lookback:i + lookback + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            pivots.append(Pivot(i, float(highs[i]), "high", times[i]))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            pivots.append(Pivot(i, float(lows[i]), "low", times[i]))
    return pivots


def detect_liquidity_sweeps(df: pd.DataFrame, pivots: list[Pivot]) -> list[Sweep]:
    sweeps: list[Sweep] = []
    pivot_highs = [p for p in pivots if p.kind == "high"]
    pivot_lows = [p for p in pivots if p.kind == "low"]
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        bar = df.iloc[i]
        recent_highs = [p for p in pivot_highs if (i - SWEEP_LOOKBACK) <= p.idx < i]
        if recent_highs:
            highest = max(recent_highs, key=lambda p: p.price)
            if bar.high > highest.price and bar.close < highest.price:
                sweeps.append(Sweep(i, highest.idx, "bsl",
                                    float(bar.high), highest.price, df.index[i]))
        recent_lows = [p for p in pivot_lows if (i - SWEEP_LOOKBACK) <= p.idx < i]
        if recent_lows:
            lowest = min(recent_lows, key=lambda p: p.price)
            if bar.low < lowest.price and bar.close > lowest.price:
                sweeps.append(Sweep(i, lowest.idx, "ssl",
                                    float(bar.low), lowest.price, df.index[i]))
    return sweeps


def detect_bos(df: pd.DataFrame, pivots: list[Pivot]) -> list[BOS]:
    bos_events: list[BOS] = []
    pivot_highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.idx)
    pivot_lows = sorted([p for p in pivots if p.kind == "low"], key=lambda p: p.idx)
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        c = float(df.iloc[i].close)
        recent_highs = [p for p in pivot_highs if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_highs):
            if c > p.price:
                bos_events.append(BOS(i, p.idx, "up", p.price, df.index[i]))
                break
        recent_lows = [p for p in pivot_lows if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_lows):
            if c < p.price:
                bos_events.append(BOS(i, p.idx, "down", p.price, df.index[i]))
                break
    return bos_events


def detect_choch_after(df: pd.DataFrame, pivots: list[Pivot],
                       bos: BOS, window: int = CHOCH_WINDOW) -> Optional[CHoCH]:
    n = len(df)
    end = min(n, bos.idx + window)
    if bos.direction == "down":
        pre = [p for p in pivots if p.kind == "high" and p.idx <= bos.idx]
        if not pre:
            return None
        last_high = max(pre, key=lambda p: p.idx)
        post = [p for p in pivots if p.kind == "high"
                and bos.idx < p.idx <= end]
        for p in post:
            if p.price < last_high.price:
                return CHoCH(p.idx, "down", p.idx, p.time)
    else:
        pre = [p for p in pivots if p.kind == "low" and p.idx <= bos.idx]
        if not pre:
            return None
        last_low = max(pre, key=lambda p: p.idx)
        post = [p for p in pivots if p.kind == "low"
                and bos.idx < p.idx <= end]
        for p in post:
            if p.price > last_low.price:
                return CHoCH(p.idx, "up", p.idx, p.time)
    return None


def detect_order_blocks(df: pd.DataFrame) -> list[OrderBlock]:
    obs: list[OrderBlock] = []
    n = len(df)
    if "atr" not in df.columns:
        return obs
    for i in range(2, n):
        bar = df.iloc[i]
        if pd.isna(bar.atr) or bar.atr <= 0 or bar.range <= 0:
            continue
        if (bar.range < DISPLACEMENT_ATR_MULT * bar.atr or
                bar.body / bar.range < 0.55):
            continue
        is_bull = bar.close > bar.open
        for j in range(i - 1, max(i - 6, 0), -1):
            prev = df.iloc[j]
            body_high = max(prev.open, prev.close)
            body_low = min(prev.open, prev.close)
            if is_bull and prev.close < prev.open:
                obs.append(OrderBlock(
                    formed_idx=j, direction="bull",
                    high=float(prev.high), low=float(prev.low),
                    midpoint=float((prev.high + prev.low) / 2),
                    body_high=float(body_high), body_low=float(body_low),
                    displacement_atr=float(bar.range / bar.atr),
                    time=df.index[j]))
                break
            if (not is_bull) and prev.close > prev.open:
                obs.append(OrderBlock(
                    formed_idx=j, direction="bear",
                    high=float(prev.high), low=float(prev.low),
                    midpoint=float((prev.high + prev.low) / 2),
                    body_high=float(body_high), body_low=float(body_low),
                    displacement_atr=float(bar.range / bar.atr),
                    time=df.index[j]))
                break
    closes = df.close.values
    for ob in obs:
        for k in range(ob.formed_idx + 2, n):
            if ob.direction == "bull":
                if closes[k] < ob.low:
                    ob.mitigated_idx = k
                    break
            else:
                if closes[k] > ob.high:
                    ob.mitigated_idx = k
                    break
    return obs


def detect_breakers(obs: list[OrderBlock]) -> list[Breaker]:
    breakers: list[Breaker] = []
    for ob in obs:
        if ob.mitigated_idx is None:
            continue
        flip_dir = "bear" if ob.direction == "bull" else "bull"
        breakers.append(Breaker(
            parent_high=ob.high, parent_low=ob.low,
            flip_idx=ob.mitigated_idx, direction=flip_dir,
            high=ob.high, low=ob.low, midpoint=ob.midpoint,
            body_high=ob.body_high, body_low=ob.body_low,
            time=ob.time))
    return breakers


def detect_rejection_blocks(df: pd.DataFrame, pivots: list[Pivot]) -> list[RejectionBlock]:
    rbs: list[RejectionBlock] = []
    pivot_highs = [p for p in pivots if p.kind == "high"]
    pivot_lows = [p for p in pivots if p.kind == "low"]
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        bar = df.iloc[i]
        if bar.range <= 0:
            continue
        upper_frac = bar.upper_wick / bar.range
        lower_frac = bar.lower_wick / bar.range
        body_high = max(bar.open, bar.close)
        body_low = min(bar.open, bar.close)
        if upper_frac >= RB_WICK_FRAC:
            recent = [p for p in pivot_highs if (i - SWEEP_LOOKBACK) <= p.idx < i]
            if recent:
                top = max(recent, key=lambda p: p.price)
                if bar.high >= top.price * 0.998:
                    rbs.append(RejectionBlock(
                        idx=i, direction="bear",
                        high=float(bar.high), low=float(bar.low),
                        midpoint=float((bar.high + bar.low) / 2),
                        body_high=float(body_high), body_low=float(body_low),
                        near_pivot_idx=top.idx, time=df.index[i]))
        if lower_frac >= RB_WICK_FRAC:
            recent = [p for p in pivot_lows if (i - SWEEP_LOOKBACK) <= p.idx < i]
            if recent:
                bot = min(recent, key=lambda p: p.price)
                if bar.low <= bot.price * 1.002:
                    rbs.append(RejectionBlock(
                        idx=i, direction="bull",
                        high=float(bar.high), low=float(bar.low),
                        midpoint=float((bar.high + bar.low) / 2),
                        body_high=float(body_high), body_low=float(body_low),
                        near_pivot_idx=bot.idx, time=df.index[i]))
    return rbs


@dataclass
class Setup:
    sweep: Sweep
    bos: BOS
    choch: CHoCH
    zone_kind: str
    zone_high: float
    zone_low: float
    zone_midpoint: float
    side: str
    sl_anchor: float
    setup_idx: int


def assemble_setups(df, pivots, sweeps, bos_events, obs, brks, rbs) -> list[Setup]:
    setups: list[Setup] = []
    for sweep in sweeps:
        target_dir = "down" if sweep.direction == "bsl" else "up"
        side = "short" if sweep.direction == "bsl" else "long"
        candidate = [b for b in bos_events
                     if b.direction == target_dir
                     and sweep.idx < b.idx <= sweep.idx + PATTERN_WINDOW]
        if not candidate:
            continue
        first_bos = candidate[0]
        choch = detect_choch_after(df, pivots, first_bos)
        if choch is None or choch.idx > sweep.idx + PATTERN_WINDOW:
            continue
        zones = []
        for ob in obs:
            if ob.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= ob.formed_idx <= choch.idx:
                    zones.append(("OB", ob.high, ob.low, ob.midpoint))
        for brk in brks:
            if brk.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= brk.flip_idx <= choch.idx + 5:
                    zones.append(("BRK", brk.high, brk.low, brk.midpoint))
        for rb in rbs:
            if rb.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx - 1 <= rb.idx <= sweep.idx + 1:
                    zones.append(("RB", rb.high, rb.low, rb.midpoint))
        if not zones:
            continue
        if side == "short":
            best = min(zones, key=lambda z: abs(z[2] - sweep.swept_price))
        else:
            best = min(zones, key=lambda z: abs(z[1] - sweep.swept_price))
        kind, zh, zl, zm = best
        setups.append(Setup(sweep, first_bos, choch, kind, zh, zl, zm,
                            side, sweep.sweep_extreme, choch.idx))
    return setups


# ─────────────────────────────────────────────────────────
# Wedge detector (mirror scripts/wedge_detector.py)
# ─────────────────────────────────────────────────────────

def _find_pivots_simple(highs: np.ndarray, lows: np.ndarray, lb: int = 5):
    n = len(highs)
    ph, pl = [], []
    for i in range(lb, n - lb):
        if highs[i] == max(highs[i - lb:i + lb + 1]):
            ph.append(i)
        if lows[i] == min(lows[i - lb:i + lb + 1]):
            pl.append(i)
    return np.asarray(ph, dtype=int), np.asarray(pl, dtype=int)


def detect_wedge_at(df: pd.DataFrame, end_idx: int,
                    pivot_lb: int = WEDGE_PIVOT_LB,
                    detect_window: int = WEDGE_DETECT_WINDOW,
                    vol_threshold: float = WEDGE_VOL_THRESHOLD) -> Optional[dict]:
    """Wedge detection at bar end_idx (using only data <= end_idx).

    Returns:
        {"side": "long" | "short", "type": "falling_wedge" | "rising_wedge"} on hit,
        None otherwise.
    """
    if end_idx < detect_window + pivot_lb + 2:
        return None
    sub = df.iloc[:end_idx + 1]
    if "high" not in sub.columns or "low" not in sub.columns or "close" not in sub.columns:
        return None
    highs = sub["high"].astype(float).values
    lows = sub["low"].astype(float).values
    closes = sub["close"].astype(float).values
    ph, pl = _find_pivots_simple(highs, lows, pivot_lb)
    end = len(sub) - 1

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

    # Volume gate
    vol_ok = True
    if "volume" in sub.columns and end >= 20:
        avg = float(sub["volume"].iloc[end - 20:end].mean())
        if avg > 0:
            vol_ok = float(sub["volume"].iloc[end]) >= vol_threshold * avg

    # Rising wedge → SHORT bias
    if sh > 0 and sl > 0 and sh < sl:
        b_l = float(np.polyfit(pl_x, pl_y, 1)[1])
        lower_lvl = sl * end + b_l
        if closes[end] < lower_lvl and vol_ok:
            return {"side": "short", "type": "rising_wedge"}

    # Falling wedge → LONG bias
    if sh < 0 and sl < 0 and sh < sl:
        b_h = float(np.polyfit(ph_x, ph_y, 1)[1])
        upper_lvl = sh * end + b_h
        if closes[end] > upper_lvl and vol_ok:
            return {"side": "long", "type": "falling_wedge"}
    return None


def find_wedge_in_window(df: pd.DataFrame, center_idx: int, window_back: int = 8) -> Optional[dict]:
    """Look back up to window_back bars for any wedge detection.
    A wedge is in effect if it broke out within the prior `window_back` bars
    of `center_idx` (CHoCH bar). Returns the most recent.
    """
    for k in range(center_idx, max(0, center_idx - window_back), -1):
        w = detect_wedge_at(df, k)
        if w is not None:
            return w
    return None


# ─────────────────────────────────────────────────────────
# HTF veto (mirror VETO 10i: 1h EMA21/50 stack at trigger)
# ─────────────────────────────────────────────────────────

def htf_bias_at(df_1h_with_ema: pd.DataFrame, idx: int) -> int:
    """Return +1/-1/0 = bullish / bearish / neutral 1h EMA21/50 stack."""
    if idx < HTF_EMA_SLOW:
        return 0
    bar = df_1h_with_ema.iloc[idx]
    c = float(bar.close)
    e21 = float(bar.ema21)
    e50 = float(bar.ema50)
    if pd.isna(e21) or pd.isna(e50):
        return 0
    if c > e21 > e50:
        return 1
    if c < e21 < e50:
        return -1
    return 0


def htf_a_grade_veto(df_1h_with_ema: pd.DataFrame, setup: Setup) -> bool:
    """SMC1.5 setups are full-stack (Sweep+BOS+CHoCH+OB) → ALL qualify as A-grade.
    Veto fires when trade direction OPPOSES 1h EMA bias at trigger candle.
    """
    bias = htf_bias_at(df_1h_with_ema, setup.choch.idx)
    if bias == 0:
        return False  # neutral → no veto
    if setup.side == "long" and bias < 0:
        return True
    if setup.side == "short" and bias > 0:
        return True
    return False


# ─────────────────────────────────────────────────────────
# Wedge veto (mirror VETO 10k)
# ─────────────────────────────────────────────────────────

def wedge_veto(df_1h: pd.DataFrame, setup: Setup) -> bool:
    """If a fresh wedge breakout fired on 1h within recent window of CHoCH
    bar AND its bias OPPOSES trade.side, veto.
    """
    w = find_wedge_in_window(df_1h, setup.choch.idx, window_back=8)
    if w is None:
        return False
    if w["side"] != setup.side:
        return True
    return False


# ─────────────────────────────────────────────────────────
# Entry simulation (WICK = limit at OB extreme)
# ─────────────────────────────────────────────────────────

def simulate_wick_entry(df: pd.DataFrame, setup: Setup) -> Optional[dict]:
    """Limit at OB extreme nearest entry side.
    Long → zone_high, short → zone_low. 6h fill window.
    """
    n = len(df)
    n_max = min(n, setup.setup_idx + 1 + LIMIT_FILL_WINDOW_BARS)
    atr_at_choch = float(df.iloc[setup.setup_idx].atr) if not pd.isna(df.iloc[setup.setup_idx].atr) else 0.0
    if atr_at_choch <= 0:
        return None

    if setup.side == "long":
        limit_price = setup.zone_high
    else:
        limit_price = setup.zone_low

    entry_idx = None
    entry_price = None
    for i in range(setup.setup_idx + 1, n_max):
        bar = df.iloc[i]
        if setup.side == "long":
            if bar.low <= limit_price:
                entry_idx = i
                entry_price = min(float(bar.open), limit_price)
                if bar.open > limit_price:
                    entry_price = limit_price
                break
        else:
            if bar.high >= limit_price:
                entry_idx = i
                entry_price = max(float(bar.open), limit_price)
                if bar.open < limit_price:
                    entry_price = limit_price
                break
    if entry_idx is None:
        return None

    if setup.side == "short":
        sl = setup.sl_anchor + SL_BUFFER_ATR * atr_at_choch
    else:
        sl = setup.sl_anchor - SL_BUFFER_ATR * atr_at_choch
    R = abs(entry_price - sl)
    if R <= 0:
        return None
    if R > 3.0 * atr_at_choch:
        return None

    return simulate_exit(df, setup, entry_idx, entry_price, sl, R)


def simulate_exit(df: pd.DataFrame, setup: Setup, entry_idx: int,
                  entry: float, sl: float, R: float) -> Optional[dict]:
    n = len(df)
    side = setup.side
    qty = NOTIONAL_USD / entry
    if side == "short":
        tp_price = entry - TP_R * R
    else:
        tp_price = entry + TP_R * R

    exit_idx = None
    exit_price = None
    exit_reason = None
    bars_held = 0

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        bars_held = j - entry_idx
        if bars_held > MAX_BARS:
            exit_idx = j
            exit_price = float(bar.open)
            exit_reason = "time_stop"
            break
        if side == "short":
            if bar.high >= sl:
                exit_idx = j
                exit_price = sl
                exit_reason = "sl_hit"
                break
            if bar.low <= tp_price:
                exit_idx = j
                exit_price = tp_price
                exit_reason = "tp_3R"
                break
        else:
            if bar.low <= sl:
                exit_idx = j
                exit_price = sl
                exit_reason = "sl_hit"
                break
            if bar.high >= tp_price:
                exit_idx = j
                exit_price = tp_price
                exit_reason = "tp_3R"
                break

    if exit_idx is None:
        last = df.iloc[-1]
        exit_idx = n - 1
        exit_price = float(last.close)
        exit_reason = "data_end"

    if side == "short":
        gross = (entry - exit_price) * qty
    else:
        gross = (exit_price - entry) * qty

    # Maker on entry, taker on exit
    entry_fee = NOTIONAL_USD * MAKER_FEE
    exit_fee = (qty * exit_price) * TAKER_FEE
    funding = NOTIONAL_USD * FUNDING_PER_8H * (bars_held / 8.0)
    net = gross - entry_fee - exit_fee - funding

    return {
        "side": side,
        "zone_kind": setup.zone_kind,
        "entry_idx": entry_idx,
        "entry_time": df.index[entry_idx].isoformat(),
        "entry": entry,
        "sl": sl,
        "R": R,
        "exit_idx": exit_idx,
        "exit_time": df.index[exit_idx].isoformat(),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "gross_pnl": gross,
        "fees": entry_fee + exit_fee,
        "funding": funding,
        "net_pnl": net,
        # R-based outcome: net/R*$1000 invested
        "outcome_r": (gross - entry_fee - exit_fee - funding) / (R * qty) if R > 0 else 0.0,
        "sweep_idx": setup.sweep.idx,
        "bos_idx": setup.bos.idx,
        "choch_idx": setup.choch.idx,
        "choch_time": setup.choch.time.isoformat(),
    }


# ─────────────────────────────────────────────────────────
# Per-symbol pipeline
# ─────────────────────────────────────────────────────────

def run_symbol(sym: str) -> dict:
    df = load_candles(sym, "1h")
    df = add_atr(df)
    df = add_emas(df)
    pivots = find_swing_pivots(df)
    sweeps = detect_liquidity_sweeps(df, pivots)
    bos_events = detect_bos(df, pivots)
    obs = detect_order_blocks(df)
    brks = detect_breakers(obs)
    rbs = detect_rejection_blocks(df, pivots)
    setups = assemble_setups(df, pivots, sweeps, bos_events, obs, brks, rbs)

    out = {
        "sym": sym,
        "n_bars": len(df),
        "n_pivots": len(pivots),
        "n_sweeps": len(sweeps),
        "n_bos": len(bos_events),
        "n_obs": len(obs),
        "n_setups": len(setups),
        "trades": {v: [] for v in VARIANTS},
        "veto_counts": {"htf": 0, "wedge": 0, "both": 0, "passed_baseline": 0,
                         "passed_htf_only": 0, "passed_wedge_only": 0,
                         "passed_combo": 0},
    }

    for s in setups:
        # Try entry first; if no fill, no trade in any variant
        trade = simulate_wick_entry(df, s)
        if trade is None:
            continue
        out["veto_counts"]["passed_baseline"] += 1
        # Annotate trade with metadata
        trade["sym"] = sym
        trade["zone_kind_setup"] = s.zone_kind

        # Compute vetoes at trigger candle (CHoCH idx)
        htf_blocks = htf_a_grade_veto(df, s)
        wedge_blocks = wedge_veto(df, s)
        if htf_blocks:
            out["veto_counts"]["htf"] += 1
        if wedge_blocks:
            out["veto_counts"]["wedge"] += 1
        if htf_blocks and wedge_blocks:
            out["veto_counts"]["both"] += 1

        # Baseline: always include
        out["trades"]["baseline"].append(trade)
        # HTF-only: skip if htf_blocks
        if not htf_blocks:
            out["trades"]["htf_only"].append(trade)
            out["veto_counts"]["passed_htf_only"] += 1
        # Wedge-only: skip if wedge_blocks
        if not wedge_blocks:
            out["trades"]["wedge_only"].append(trade)
            out["veto_counts"]["passed_wedge_only"] += 1
        # Combo: skip if either fires
        if not htf_blocks and not wedge_blocks:
            out["trades"]["combo"].append(trade)
            out["veto_counts"]["passed_combo"] += 1
    return out


# ─────────────────────────────────────────────────────────
# Walk-forward bucketing
# ─────────────────────────────────────────────────────────

def quarter_of(ts_iso: str) -> Optional[str]:
    ts = pd.Timestamp(ts_iso)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    for q, (a, b) in QUARTERS.items():
        if pd.Timestamp(a) <= ts <= pd.Timestamp(b):
            return q
    return None


def summarize(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "net_$": 0.0, "ev_per_trade_$": 0.0,
                "gross_R": 0.0, "ev_per_trade_R": 0.0,
                "tp_hits": 0, "sl_hits": 0, "time_stops": 0}
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    net = sum(t["net_pnl"] for t in trades)
    gross_R = sum(t.get("outcome_r", 0.0) for t in trades)
    tp = sum(1 for t in trades if t.get("exit_reason") == "tp_3R")
    sl = sum(1 for t in trades if t.get("exit_reason") == "sl_hit")
    ts_stops = sum(1 for t in trades if t.get("exit_reason") == "time_stop")
    return {
        "n": n,
        "wr": 100.0 * wins / n,
        "net_$": net,
        "ev_per_trade_$": net / n,
        "gross_R": gross_R,
        "ev_per_trade_R": gross_R / n,
        "tp_hits": tp,
        "sl_hits": sl,
        "time_stops": ts_stops,
    }


def walk_forward(trades_all: list) -> dict:
    by_q = {q: [] for q in QUARTERS}
    for t in trades_all:
        q = quarter_of(t["entry_time"])
        if q:
            by_q[q].append(t)
    is_trades = by_q["Q1"] + by_q["Q2"]
    is_stat = summarize(is_trades)
    q3 = summarize(by_q["Q3"])
    q4 = summarize(by_q["Q4"])

    # Walk-forward gap %: (OOS - IS) / |IS|
    def gap(oos_ev: float, is_ev: float) -> float:
        if abs(is_ev) < 1e-9:
            return 0.0
        return 100.0 * (oos_ev - is_ev) / abs(is_ev)

    q3_gap = gap(q3["ev_per_trade_$"], is_stat["ev_per_trade_$"])
    q4_gap = gap(q4["ev_per_trade_$"], is_stat["ev_per_trade_$"])
    # Pass: OOS gap <= 50% (i.e., OOS at least preserves 50% of IS),
    # OOS same sign positive, n>=30 IS, n>=10 each OOS
    same_sign_q3 = (q3["ev_per_trade_$"] > 0 and is_stat["ev_per_trade_$"] > 0)
    same_sign_q4 = (q4["ev_per_trade_$"] > 0 and is_stat["ev_per_trade_$"] > 0)
    q3_preserves = same_sign_q3 and (q3["ev_per_trade_$"] >= 0.5 * is_stat["ev_per_trade_$"])
    q4_preserves = same_sign_q4 and (q4["ev_per_trade_$"] >= 0.5 * is_stat["ev_per_trade_$"])
    n_ok = (is_stat["n"] >= 30 and q3["n"] >= 10 and q4["n"] >= 10)
    pass_wf = q3_preserves and q4_preserves and n_ok

    # Verdict logic
    if pass_wf:
        verdict = "SHIP"
    elif (is_stat["ev_per_trade_$"] > 0 and q3["ev_per_trade_$"] > 0
          and q4["ev_per_trade_$"] > 0 and is_stat["n"] >= 20):
        verdict = "PILOT"
    elif is_stat["ev_per_trade_$"] > 0 and is_stat["n"] >= 30:
        verdict = "MARGINAL"
    else:
        verdict = "KILL"

    return {
        "IS_Q1Q2": is_stat,
        "Q3": q3,
        "Q4": q4,
        "q3_gap_pct": q3_gap,
        "q4_gap_pct": q4_gap,
        "n_ok": n_ok,
        "q3_preserves": q3_preserves,
        "q4_preserves": q4_preserves,
        "pass_walkforward": pass_wf,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    all_results = {}
    for sym in SYMBOLS:
        try:
            r = run_symbol(sym)
            all_results[sym] = r
            print(f"{sym}: setups={r['n_setups']}  filled={r['veto_counts']['passed_baseline']}  "
                  f"htf_block={r['veto_counts']['htf']}  wedge_block={r['veto_counts']['wedge']}  "
                  f"both_block={r['veto_counts']['both']}")
        except Exception as e:
            import traceback
            print(f"ERROR {sym}: {e}\n{traceback.format_exc()}", file=sys.stderr)

    # Aggregate trades per variant across all symbols
    agg_trades = {v: [] for v in VARIANTS}
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        for v in VARIANTS:
            agg_trades[v].extend(all_results[sym]["trades"][v])

    # Walk-forward per variant
    wf_results = {}
    for v in VARIANTS:
        wf_results[v] = walk_forward(agg_trades[v])

    # Per-symbol walk-forward (for diagnostics)
    wf_per_sym = {}
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        wf_per_sym[sym] = {}
        for v in VARIANTS:
            wf_per_sym[sym][v] = walk_forward(all_results[sym]["trades"][v])

    # Output
    out = {
        "config": {
            "symbols": SYMBOLS,
            "tf": "1h",
            "tp_R": TP_R, "max_bars": MAX_BARS,
            "sl_buffer_atr": SL_BUFFER_ATR,
            "limit_fill_window_bars": LIMIT_FILL_WINDOW_BARS,
            "maker_fee": MAKER_FEE, "taker_fee": TAKER_FEE,
            "notional_usd": NOTIONAL_USD,
            "quarters": {q: list(v) for q, v in QUARTERS.items()},
        },
        "veto_counts": {
            sym: all_results[sym]["veto_counts"] for sym in SYMBOLS if sym in all_results
        },
        "agg_walkforward": wf_results,
        "per_sym_walkforward": wf_per_sym,
    }
    (OUT_DIR / "walkforward.json").write_text(json.dumps(out, indent=2, default=str))

    # Markdown report
    md = ["# SMC1.5v3 Combo Walk-Forward Report",
          "",
          "Tests SMC1.5v2 (WICK-entry, EB exit) baseline vs HTF-A-grade veto, "
          "Wedge-breakout veto, and the combo of both filters.",
          "",
          "**Walk-forward:** Q1+Q2 IS (Oct 2025-Jan 2026), Q3 OOS (Feb 2026), "
          "Q4 OOS (Mar-Apr 2026). 4 symbols, 1h candles.",
          "",
          "**Pass criteria:** IS n>=30, OOS n>=10 each, OOS EV >= 50% IS EV "
          "(both quarters), same sign positive.",
          "",
          "## Setup detection counts",
          "",
          "| Sym | bars | pivots | sweeps | OBs | setups | filled | htf_block | wedge_block | both_block |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        r = all_results[sym]
        v = r["veto_counts"]
        md.append(f"| {sym} | {r['n_bars']} | {r['n_pivots']} | {r['n_sweeps']} | "
                  f"{r['n_obs']} | {r['n_setups']} | {v['passed_baseline']} | "
                  f"{v['htf']} | {v['wedge']} | {v['both']} |")

    md.append("")
    md.append("## Aggregate walk-forward results (all 4 symbols)")
    md.append("")
    md.append("| Variant | IS n / EV$ / EV(R) | Q3 n / EV$ / gap% | Q4 n / EV$ / gap% | Verdict |")
    md.append("|---|---|---|---|---|")
    for v in VARIANTS:
        w = wf_results[v]
        s = w["IS_Q1Q2"]; q3 = w["Q3"]; q4 = w["Q4"]
        md.append(f"| **{v}** | {s['n']} / ${s['ev_per_trade_$']:+.2f} / {s['ev_per_trade_R']:+.3f}R | "
                  f"{q3['n']} / ${q3['ev_per_trade_$']:+.2f} / {w['q3_gap_pct']:+.0f}% | "
                  f"{q4['n']} / ${q4['ev_per_trade_$']:+.2f} / {w['q4_gap_pct']:+.0f}% | "
                  f"**{w['verdict']}** |")

    md.append("")
    md.append("## Per-variant detail")
    md.append("")
    for v in VARIANTS:
        w = wf_results[v]
        md.append(f"### {v}")
        md.append("")
        md.append("| Quarter | n | WR% | net$ | EV/trade$ | EV/trade(R) | TP | SL | Time |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for q_name, q_stat in [("IS_Q1Q2", w["IS_Q1Q2"]), ("Q3", w["Q3"]), ("Q4", w["Q4"])]:
            md.append(f"| {q_name} | {q_stat['n']} | {q_stat['wr']:.1f} | "
                      f"${q_stat['net_$']:+.2f} | ${q_stat['ev_per_trade_$']:+.2f} | "
                      f"{q_stat['ev_per_trade_R']:+.3f} | "
                      f"{q_stat['tp_hits']} | {q_stat['sl_hits']} | {q_stat['time_stops']} |")
        md.append("")
        md.append(f"- IS-OOS Q3 gap: {w['q3_gap_pct']:+.0f}% | Q4 gap: {w['q4_gap_pct']:+.0f}%")
        md.append(f"- Q3 preserves edge (>=50% IS, same sign positive): "
                  f"{'YES' if w['q3_preserves'] else 'NO'}")
        md.append(f"- Q4 preserves edge: {'YES' if w['q4_preserves'] else 'NO'}")
        md.append(f"- Sample sizes meet thresholds (IS>=30, OOS>=10): "
                  f"{'YES' if w['n_ok'] else 'NO'}")
        md.append(f"- **Walk-forward verdict: {w['verdict']}**")
        md.append("")

    # Cross-comparison summary
    md.append("## Cross-comparison")
    md.append("")
    base = wf_results["baseline"]
    htf = wf_results["htf_only"]
    wedge = wf_results["wedge_only"]
    combo = wf_results["combo"]

    def cmp_ev(label, winner, loser):
        diff = winner["IS_Q1Q2"]["ev_per_trade_$"] - loser["IS_Q1Q2"]["ev_per_trade_$"]
        return f"{label} IS EV delta: {diff:+.2f}$"

    md.append(f"- Baseline IS EV$: {base['IS_Q1Q2']['ev_per_trade_$']:+.2f} (n={base['IS_Q1Q2']['n']})")
    md.append(f"- HTF-only IS EV$: {htf['IS_Q1Q2']['ev_per_trade_$']:+.2f} (n={htf['IS_Q1Q2']['n']})")
    md.append(f"- Wedge-only IS EV$: {wedge['IS_Q1Q2']['ev_per_trade_$']:+.2f} (n={wedge['IS_Q1Q2']['n']})")
    md.append(f"- Combo IS EV$: {combo['IS_Q1Q2']['ev_per_trade_$']:+.2f} (n={combo['IS_Q1Q2']['n']})")
    md.append("")
    md.append(f"- HTF-only beats baseline on IS EV: "
              f"{'YES' if htf['IS_Q1Q2']['ev_per_trade_$'] >= base['IS_Q1Q2']['ev_per_trade_$'] else 'NO'}")
    md.append(f"- Wedge-only beats baseline on IS EV: "
              f"{'YES' if wedge['IS_Q1Q2']['ev_per_trade_$'] >= base['IS_Q1Q2']['ev_per_trade_$'] else 'NO'}")
    md.append(f"- Combo beats baseline on IS EV: "
              f"{'YES' if combo['IS_Q1Q2']['ev_per_trade_$'] >= base['IS_Q1Q2']['ev_per_trade_$'] else 'NO'}")
    md.append(f"- Combo beats HTF-only on IS EV: "
              f"{'YES' if combo['IS_Q1Q2']['ev_per_trade_$'] >= htf['IS_Q1Q2']['ev_per_trade_$'] else 'NO'}")
    md.append(f"- Combo beats Wedge-only on IS EV: "
              f"{'YES' if combo['IS_Q1Q2']['ev_per_trade_$'] >= wedge['IS_Q1Q2']['ev_per_trade_$'] else 'NO'}")

    md.append("")
    md.append("## Per-symbol breakdown (walk-forward)")
    md.append("")
    md.append("| Sym | Variant | IS n / EV$ | Q3 n / EV$ | Q4 n / EV$ | Verdict |")
    md.append("|---|---|---|---|---|---|")
    for sym in SYMBOLS:
        if sym not in wf_per_sym:
            continue
        for v in VARIANTS:
            w = wf_per_sym[sym][v]
            md.append(f"| {sym} | {v} | {w['IS_Q1Q2']['n']} / ${w['IS_Q1Q2']['ev_per_trade_$']:+.2f} | "
                      f"{w['Q3']['n']} / ${w['Q3']['ev_per_trade_$']:+.2f} | "
                      f"{w['Q4']['n']} / ${w['Q4']['ev_per_trade_$']:+.2f} | {w['verdict']} |")
    md.append("")

    md.append("## Cost model")
    md.append(f"- Maker fee (entry leg): {MAKER_FEE*100:.3f}%")
    md.append(f"- Taker fee (exit leg): {TAKER_FEE*100:.3f}%")
    md.append(f"- Funding: {FUNDING_PER_8H*100:.3f}% per 8h")
    md.append(f"- Notional: ${NOTIONAL_USD:.0f}/trade")
    md.append("")

    md.append("## Strategy params (must match smc15v2_paper_engine)")
    md.append(f"- Pivot lookback: {SWING_LOOKBACK}-bar fractal")
    md.append(f"- Sweep lookback: {SWEEP_LOOKBACK} bars")
    md.append(f"- BOS lookback: {BOS_LOOKBACK} bars")
    md.append(f"- CHoCH window: {CHOCH_WINDOW}")
    md.append(f"- Pattern window: {PATTERN_WINDOW}")
    md.append(f"- Displacement: range > {DISPLACEMENT_ATR_MULT}× ATR & body > 55% range")
    md.append(f"- TP: {TP_R}R fixed | SL: sweep wick + {SL_BUFFER_ATR}× ATR | Time stop: {MAX_BARS}h")
    md.append(f"- Limit expiry: {LIMIT_FILL_WINDOW_BARS}h")
    md.append(f"- HTF veto: 1h EMA{HTF_EMA_FAST}/{HTF_EMA_SLOW} stack at CHoCH bar; A-grade = SMC full stack (always A)")
    md.append(f"- Wedge veto: detection on 1h within 8 bars back of CHoCH; "
              f"opposing-side breakout vetoes")

    (OUT_DIR / "report.md").write_text("\n".join(md))

    # Console summary
    print("\n=== AGG walk-forward by variant ===")
    for v in VARIANTS:
        w = wf_results[v]
        s = w["IS_Q1Q2"]; q3 = w["Q3"]; q4 = w["Q4"]
        print(f"  {v}: IS n={s['n']:3d} EV=${s['ev_per_trade_$']:+.2f} | "
              f"Q3 n={q3['n']:3d} EV=${q3['ev_per_trade_$']:+.2f} (gap {w['q3_gap_pct']:+.0f}%) | "
              f"Q4 n={q4['n']:3d} EV=${q4['ev_per_trade_$']:+.2f} (gap {w['q4_gap_pct']:+.0f}%) | "
              f"{w['verdict']}")
    print(f"\nReports: {OUT_DIR}/walkforward.json + report.md")


if __name__ == "__main__":
    main()

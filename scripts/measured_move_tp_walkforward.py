#!/usr/bin/env python3
"""
Measured-Move TP variants - Walk-forward backtest.

Reuses SMC1.5 setup detection from smc15_fullstack_backtest.py and wedge
detection from classical_patterns_walkforward.py / wedge_detector.py.

Tests 5 TP methods per setup:
    fixed_2R, fixed_3R, measured_move, measured_move_x_0.75, measured_move_x_1.25

SL: 1x ATR (standardized for fair compare).
Time stop: 8h.

Pattern-derived (measured-move) TP:
- SMC1.5: pattern height = sweep_extreme to opposite swing (BOS broken pivot
          for SHORT this is the broken low; for LONG the broken high).
          Project this distance from break point in trade direction.
          Implementation: distance = |sweep_extreme - bos.broken_price|.
          Project from entry (= signal close) in trade direction.
- Wedge:  pattern height = widest distance between upper and lower trendlines
          inside the wedge envelope. Project from breakout (entry) in
          trade direction.

Pass criteria for measured-move TP variant:
    must beat best fixed-R baseline by >= +0.10R EV/trade in IS
    AND maintain edge through Q3 + Q4 OOS (each >= 50% of IS, same sign).

Output:
    storage/measured_move_tp/walkforward.json
    storage/measured_move_tp/report.md
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
OUT_DIR = ROOT / "storage" / "measured_move_tp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# TP methods
TP_METHODS = [
    "fixed_2R",
    "fixed_3R",
    "measured_move",
    "measured_move_x_0.75",
    "measured_move_x_1.25",
]

# Cap on measured-move TP (in R units) so absurdly long projections don't poison stats
MEASURED_MOVE_CAP_R = 6.0
MEASURED_MOVE_FLOOR_R = 0.5

# Time stop & SL (matches spec)
TIME_STOP_HOURS = 8
SL_ATR = 1.0

# Trade economics
NOTIONAL = 1000.0
RT_TAKER_FEE_BPS = 0.118
FUNDING_RATE_PER_8H = 0.0001
ATR_PERIOD = 14
VOL_LOOKBACK = 20

# Walk-forward boundaries
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}

# SMC1.5 detection params (mirror smc15_fullstack_backtest.py)
SWING_LOOKBACK = 5
SWEEP_LOOKBACK = 20
BOS_LOOKBACK = 30
CHOCH_WINDOW = 15
PATTERN_WINDOW = 30
DISPLACEMENT_ATR_MULT = 1.0
RB_WICK_FRAC = 0.55
ENTRY_TOL_ATR = 0.10

# Wedge detection params (matches classical_patterns_walkforward.py defaults)
WEDGE_PIVOT_LB = 5
WEDGE_DETECT_WINDOW = 30
WEDGE_VOL_THRESHOLD = 1.0

MIN_IS_TRADES = 15  # slightly relaxed for SMC1.5 (small population)


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


def quarter_of(ts: pd.Timestamp) -> Optional[str]:
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    for q, (s, e) in QUARTER_BOUNDS.items():
        if pd.Timestamp(s, tz="UTC") <= ts < pd.Timestamp(e, tz="UTC"):
            return q
    return None


def find_pivots(highs: np.ndarray, lows: np.ndarray, lb: int):
    n = len(highs)
    ph, pl = [], []
    for i in range(lb, n - lb):
        h_win = highs[i - lb : i + lb + 1]
        l_win = lows[i - lb : i + lb + 1]
        if highs[i] == h_win.max() and (h_win == highs[i]).sum() == 1:
            ph.append(i)
        if lows[i] == l_win.min() and (l_win == lows[i]).sum() == 1:
            pl.append(i)
    return np.asarray(ph, dtype=int), np.asarray(pl, dtype=int)


# -----------------------------------------------------------------------------
# SMC primitives (compact reimpl of smc15_fullstack_backtest.py)
# -----------------------------------------------------------------------------
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
    direction: str  # "bsl" -> short, "ssl" -> long
    sweep_extreme: float
    swept_price: float
    time: pd.Timestamp


@dataclass
class BOS:
    idx: int
    broken_pivot_idx: int
    direction: str  # "down" or "up"
    broken_price: float
    time: pd.Timestamp


@dataclass
class CHoCH:
    idx: int
    direction: str
    pivot_idx: int


@dataclass
class OrderBlock:
    formed_idx: int
    direction: str
    high: float
    low: float
    midpoint: float
    mitigated_idx: Optional[int] = None


@dataclass
class Breaker:
    parent_dir: str
    flip_idx: int
    direction: str
    high: float
    low: float


@dataclass
class RejectionBlock:
    idx: int
    direction: str
    high: float
    low: float


@dataclass
class SMCSetup:
    sweep: Sweep
    bos: BOS
    choch: CHoCH
    zone_kind: str
    zone_high: float
    zone_low: float
    side: str  # "long" or "short"
    sl_anchor: float
    setup_idx: int


def find_swing_pivots(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> List[Pivot]:
    pivots: List[Pivot] = []
    highs = df["high"].values
    lows = df["low"].values
    times = df.index
    n = len(df)
    for i in range(lookback, n - lookback):
        win_h = highs[i - lookback : i + lookback + 1]
        win_l = lows[i - lookback : i + lookback + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            pivots.append(Pivot(i, float(highs[i]), "high", times[i]))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            pivots.append(Pivot(i, float(lows[i]), "low", times[i]))
    return pivots


def detect_sweeps(df: pd.DataFrame, pivots: List[Pivot]) -> List[Sweep]:
    sweeps: List[Sweep] = []
    pivot_highs = [p for p in pivots if p.kind == "high"]
    pivot_lows = [p for p in pivots if p.kind == "low"]
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        bar = df.iloc[i]
        recent_highs = [p for p in pivot_highs if (i - SWEEP_LOOKBACK) <= p.idx < i]
        if recent_highs:
            top = max(recent_highs, key=lambda p: p.price)
            if bar["high"] > top.price and bar["close"] < top.price:
                sweeps.append(
                    Sweep(i, top.idx, "bsl", float(bar["high"]), top.price, df.index[i])
                )
        recent_lows = [p for p in pivot_lows if (i - SWEEP_LOOKBACK) <= p.idx < i]
        if recent_lows:
            bot = min(recent_lows, key=lambda p: p.price)
            if bar["low"] < bot.price and bar["close"] > bot.price:
                sweeps.append(
                    Sweep(i, bot.idx, "ssl", float(bar["low"]), bot.price, df.index[i])
                )
    return sweeps


def detect_bos(df: pd.DataFrame, pivots: List[Pivot]) -> List[BOS]:
    out: List[BOS] = []
    pivot_highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.idx)
    pivot_lows = sorted([p for p in pivots if p.kind == "low"], key=lambda p: p.idx)
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        c = float(df["close"].iloc[i])
        recent_h = [p for p in pivot_highs if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_h):
            if c > p.price:
                out.append(BOS(i, p.idx, "up", p.price, df.index[i]))
                break
        recent_l = [p for p in pivot_lows if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_l):
            if c < p.price:
                out.append(BOS(i, p.idx, "down", p.price, df.index[i]))
                break
    return out


def detect_choch_after(
    pivots: List[Pivot], bos: BOS, window: int = CHOCH_WINDOW
) -> Optional[CHoCH]:
    end = bos.idx + window
    if bos.direction == "down":
        pre = [p for p in pivots if p.kind == "high" and p.idx <= bos.idx]
        if not pre:
            return None
        last_h = max(pre, key=lambda p: p.idx)
        post = [p for p in pivots if p.kind == "high" and bos.idx < p.idx <= end]
        for p in post:
            if p.price < last_h.price:
                return CHoCH(p.idx, "down", p.idx)
    else:
        pre = [p for p in pivots if p.kind == "low" and p.idx <= bos.idx]
        if not pre:
            return None
        last_l = max(pre, key=lambda p: p.idx)
        post = [p for p in pivots if p.kind == "low" and bos.idx < p.idx <= end]
        for p in post:
            if p.price > last_l.price:
                return CHoCH(p.idx, "up", p.idx)
    return None


def detect_obs(df: pd.DataFrame, atr: pd.Series) -> List[OrderBlock]:
    obs: List[OrderBlock] = []
    n = len(df)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    for i in range(2, n):
        a = atr.iat[i] if not pd.isna(atr.iat[i]) else 0.0
        if a <= 0:
            continue
        rng = highs[i] - lows[i]
        body = abs(closes[i] - opens[i])
        if rng < DISPLACEMENT_ATR_MULT * a or body / max(rng, 1e-12) < 0.55:
            continue
        is_bull = closes[i] > opens[i]
        for j in range(i - 1, max(i - 6, 0), -1):
            if is_bull and closes[j] < opens[j]:
                obs.append(OrderBlock(j, "bull", float(highs[j]), float(lows[j]),
                                      float((highs[j] + lows[j]) / 2.0)))
                break
            if (not is_bull) and closes[j] > opens[j]:
                obs.append(OrderBlock(j, "bear", float(highs[j]), float(lows[j]),
                                      float((highs[j] + lows[j]) / 2.0)))
                break
    # Mitigation
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


def detect_breakers(obs: List[OrderBlock]) -> List[Breaker]:
    out = []
    for ob in obs:
        if ob.mitigated_idx is None:
            continue
        out.append(Breaker(ob.direction, ob.mitigated_idx,
                           "bear" if ob.direction == "bull" else "bull",
                           ob.high, ob.low))
    return out


def detect_rbs(df: pd.DataFrame, pivots: List[Pivot]) -> List[RejectionBlock]:
    rbs: List[RejectionBlock] = []
    pivot_highs = [p for p in pivots if p.kind == "high"]
    pivot_lows = [p for p in pivots if p.kind == "low"]
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        bar = df.iloc[i]
        rng = bar["high"] - bar["low"]
        if rng <= 0:
            continue
        upper = bar["high"] - max(bar["open"], bar["close"])
        lower = min(bar["open"], bar["close"]) - bar["low"]
        if upper / rng >= RB_WICK_FRAC:
            recent = [p for p in pivot_highs if (i - SWEEP_LOOKBACK) <= p.idx < i]
            if recent:
                top = max(recent, key=lambda p: p.price)
                if bar["high"] >= top.price * 0.998:
                    rbs.append(RejectionBlock(i, "bear", float(bar["high"]), float(bar["low"])))
        if lower / rng >= RB_WICK_FRAC:
            recent = [p for p in pivot_lows if (i - SWEEP_LOOKBACK) <= p.idx < i]
            if recent:
                bot = min(recent, key=lambda p: p.price)
                if bar["low"] <= bot.price * 1.002:
                    rbs.append(RejectionBlock(i, "bull", float(bar["high"]), float(bar["low"])))
    return rbs


def assemble_smc_setups(
    df: pd.DataFrame,
    pivots: List[Pivot],
    sweeps: List[Sweep],
    bos_events: List[BOS],
    obs: List[OrderBlock],
    brks: List[Breaker],
    rbs: List[RejectionBlock],
) -> List[SMCSetup]:
    setups: List[SMCSetup] = []
    for sweep in sweeps:
        target_dir = "down" if sweep.direction == "bsl" else "up"
        side = "short" if sweep.direction == "bsl" else "long"
        cands = [b for b in bos_events
                 if b.direction == target_dir
                 and sweep.idx < b.idx <= sweep.idx + PATTERN_WINDOW]
        if not cands:
            continue
        first_bos = cands[0]
        choch = detect_choch_after(pivots, first_bos)
        if choch is None:
            continue
        if choch.idx > sweep.idx + PATTERN_WINDOW:
            continue
        zones: List[Tuple[str, float, float]] = []
        for ob in obs:
            if ob.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= ob.formed_idx <= choch.idx:
                    zones.append(("OB", ob.high, ob.low))
        for brk in brks:
            if brk.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= brk.flip_idx <= choch.idx + 5:
                    zones.append(("BRK", brk.high, brk.low))
        for rb in rbs:
            if rb.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx - 1 <= rb.idx <= sweep.idx + 1:
                    zones.append(("RB", rb.high, rb.low))
        if not zones:
            continue
        if side == "short":
            best = min(zones, key=lambda z: abs(z[2] - sweep.swept_price))
        else:
            best = min(zones, key=lambda z: abs(z[1] - sweep.swept_price))
        zk, zh, zl = best
        setups.append(SMCSetup(
            sweep=sweep,
            bos=first_bos,
            choch=choch,
            zone_kind=zk,
            zone_high=zh,
            zone_low=zl,
            side=side,
            sl_anchor=sweep.sweep_extreme,
            setup_idx=choch.idx,
        ))
    return setups


def find_smc_entry(
    df: pd.DataFrame, atr: pd.Series, setup: SMCSetup
) -> Optional[Tuple[int, float, float]]:
    """Return (entry_idx, entry_price, atr_at_entry) or None."""
    n = len(df)
    n_max = min(n, setup.setup_idx + PATTERN_WINDOW)
    a = atr.iat[setup.setup_idx] if not pd.isna(atr.iat[setup.setup_idx]) else 0.0
    if a <= 0:
        return None
    tol = ENTRY_TOL_ATR * a
    for i in range(setup.setup_idx + 1, n_max):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        if setup.side == "short":
            if bar["high"] < setup.zone_low - tol:
                continue
            if bar["low"] > setup.zone_high + tol:
                continue
            if bar["close"] < bar["open"] and bar["close"] < prev["close"]:
                ai = atr.iat[i] if not pd.isna(atr.iat[i]) else a
                return i, float(bar["close"]), float(ai)
        else:
            if bar["high"] < setup.zone_low - tol:
                continue
            if bar["low"] > setup.zone_high + tol:
                continue
            if bar["close"] > bar["open"] and bar["close"] > prev["close"]:
                ai = atr.iat[i] if not pd.isna(atr.iat[i]) else a
                return i, float(bar["close"]), float(ai)
    return None


# -----------------------------------------------------------------------------
# Wedge detection (1h, 4h)
# -----------------------------------------------------------------------------
@dataclass
class WedgeSetup:
    entry_idx: int
    side: str  # "LONG" or "SHORT"
    pattern_height: float  # widest distance between trendlines (price units)
    atr_at_entry: float


def detect_wedge_setups(
    df: pd.DataFrame, atr: pd.Series, pivot_lb: int = WEDGE_PIVOT_LB,
    detect_window: int = WEDGE_DETECT_WINDOW, vol_threshold: float = WEDGE_VOL_THRESHOLD
) -> List[WedgeSetup]:
    """Mirror of detect_wedge in classical_patterns_walkforward.py, but also
    returns the wedge pattern_height (max gap between upper / lower trendlines)
    for measured-move TP calculation."""
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    vols = df["volume"].astype(float).values
    ph, pl = find_pivots(highs, lows, pivot_lb)
    n = len(df)

    # rolling 20-bar volume avg
    v_avg = pd.Series(vols).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean().values

    setups: List[WedgeSetup] = []
    last_idx = -10**9
    for end in range(detect_window, n):
        a = atr.iat[end] if not pd.isna(atr.iat[end]) else float("nan")
        if not np.isfinite(a) or a <= 0:
            continue
        recent_ph = ph[(ph >= end - detect_window) & (ph < end)]
        recent_pl = pl[(pl >= end - detect_window) & (pl < end)]
        if len(recent_ph) < 2 or len(recent_pl) < 2:
            continue
        ph_x = recent_ph.astype(float)
        pl_x = recent_pl.astype(float)
        ph_y = highs[recent_ph]
        pl_y = lows[recent_pl]
        sh, bh = np.polyfit(ph_x, ph_y, 1)
        sl, bl = np.polyfit(pl_x, pl_y, 1)

        # Volume ok
        vol_ok = True
        if np.isfinite(v_avg[end]) and v_avg[end] > 0:
            vol_ok = (vols[end] / v_avg[end]) >= vol_threshold

        side: Optional[str] = None
        # Rising wedge - SHORT
        if sh > 0 and sl > 0 and sh < sl:
            lower_lvl = sl * end + bl
            if closes[end] < lower_lvl and vol_ok:
                side = "SHORT"
        # Falling wedge - LONG
        elif sh < 0 and sl < 0 and sh < sl:
            upper_lvl = sh * end + bh
            if closes[end] > upper_lvl and vol_ok:
                side = "LONG"
        if side is None:
            continue

        # Compute wedge pattern_height = MAX gap between upper and lower
        # trendlines across the recent_ph...recent_pl span.
        x_min = float(min(recent_ph.min(), recent_pl.min()))
        x_max = float(max(recent_ph.max(), recent_pl.max()))
        # Sample at endpoints; trendlines either widen or narrow uniformly.
        gap_at_xmin = abs((sh * x_min + bh) - (sl * x_min + bl))
        gap_at_xmax = abs((sh * x_max + bh) - (sl * x_max + bl))
        pattern_h = max(gap_at_xmin, gap_at_xmax)
        if pattern_h <= 0:
            continue
        # De-dupe close adjacent triggers
        if end - last_idx < pivot_lb:
            continue
        last_idx = end
        setups.append(WedgeSetup(
            entry_idx=int(end),
            side=side,
            pattern_height=float(pattern_h),
            atr_at_entry=float(a),
        ))
    return setups


# -----------------------------------------------------------------------------
# Trade simulator with method-driven TP
# -----------------------------------------------------------------------------
def compute_tp_R(
    method: str,
    entry: float,
    sl_dist: float,
    pattern_height: Optional[float],
) -> Optional[float]:
    """Return TP distance in R units (i.e. multiplier for sl_dist)."""
    if method == "fixed_2R":
        return 2.0
    if method == "fixed_3R":
        return 3.0
    if method.startswith("measured_move"):
        if pattern_height is None or pattern_height <= 0:
            return None
        if sl_dist <= 0:
            return None
        base_R = pattern_height / sl_dist
        if method == "measured_move":
            mult = 1.0
        elif method == "measured_move_x_0.75":
            mult = 0.75
        elif method == "measured_move_x_1.25":
            mult = 1.25
        else:
            return None
        tp_R = base_R * mult
        # Floor & cap to avoid degenerate trades
        tp_R = max(MEASURED_MOVE_FLOOR_R, min(MEASURED_MOVE_CAP_R, tp_R))
        return tp_R
    return None


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry: float,
    atr_at_entry: float,
    pattern_height: Optional[float],
    method: str,
    tf_min: int,
) -> Optional[Dict[str, Any]]:
    if entry_idx + 1 >= len(df):
        return None
    if not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
        return None
    sl_dist = SL_ATR * atr_at_entry
    tp_R = compute_tp_R(method, entry, sl_dist, pattern_height)
    if tp_R is None:
        return None
    tp_dist = tp_R * sl_dist
    bars_max = max(1, int(math.ceil(TIME_STOP_HOURS * 60 / tf_min)))
    if side in ("LONG", "long"):
        s = "LONG"
        sl_p = entry - sl_dist
        tp_p = entry + tp_dist
    else:
        s = "SHORT"
        sl_p = entry + sl_dist
        tp_p = entry - tp_dist

    end_idx = min(entry_idx + bars_max, len(df) - 1)
    exit_reason = "TIME"
    exit_price = float(df["close"].iloc[end_idx])
    exit_idx = end_idx
    mfe_R = 0.0

    for j in range(entry_idx + 1, end_idx + 1):
        bar_h = float(df["high"].iloc[j])
        bar_l = float(df["low"].iloc[j])
        if s == "LONG":
            mfe = (bar_h - entry) / sl_dist
        else:
            mfe = (entry - bar_l) / sl_dist
        if mfe > mfe_R:
            mfe_R = mfe
        if s == "LONG":
            if bar_l <= sl_p:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if bar_h >= tp_p:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break
        else:
            if bar_h >= sl_p:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if bar_l <= tp_p:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break

    if s == "LONG":
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
        "side": s,
        "exit_reason": exit_reason,
        "entry": entry,
        "exit": exit_price,
        "minutes_held": minutes_held,
        "gross_R": gross_R,
        "net_R": net_R,
        "gross_$": gross_dollars,
        "net_$": net_dollars,
        "mfe_R": mfe_R,
        "tp_R": tp_R,
        "atr": atr_at_entry,
    }


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "wr": 0.0, "ev_R": 0.0, "ev_$": 0.0,
                "net_R": 0.0, "net_$": 0.0, "avg_mfe_R": 0.0, "avg_tp_R": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_R"] > 0)
    return {
        "n": n,
        "wr": wins / n,
        "ev_R": float(np.mean([t["net_R"] for t in trades])),
        "ev_$": float(np.mean([t["net_$"] for t in trades])),
        "net_R": float(np.sum([t["net_R"] for t in trades])),
        "net_$": float(np.sum([t["net_$"] for t in trades])),
        "avg_mfe_R": float(np.mean([t["mfe_R"] for t in trades])),
        "avg_tp_R": float(np.mean([t["tp_R"] for t in trades])),
    }


# -----------------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------------
def run_smc15_setups(
    df: pd.DataFrame, atr: pd.Series
) -> List[Tuple[int, str, float, float, float]]:
    """Return [(entry_idx, side, entry_price, atr_at_entry, pattern_height), ...]"""
    pivots = find_swing_pivots(df)
    sweeps = detect_sweeps(df, pivots)
    bos_events = detect_bos(df, pivots)
    obs = detect_obs(df, atr)
    brks = detect_breakers(obs)
    rbs = detect_rbs(df, pivots)
    setups = assemble_smc_setups(df, pivots, sweeps, bos_events, obs, brks, rbs)
    out = []
    for s in setups:
        ent = find_smc_entry(df, atr, s)
        if ent is None:
            continue
        entry_idx, entry_price, ai = ent
        # Pattern height = sweep extreme to opposite swing (BOS broken pivot)
        pattern_h = abs(s.sweep.sweep_extreme - s.bos.broken_price)
        out.append((entry_idx, s.side, entry_price, ai, pattern_h))
    return out


def run_walkforward() -> Dict[str, Any]:
    all_results: Dict[str, Any] = {"per_cell": [], "walkforward": []}

    # ============= SMC1.5 (1h) ===============
    smc_tf = "1h"
    smc_tfm = tf_minutes(smc_tf)
    sym_setups: Dict[str, List[Tuple[int, str, float, float, float]]] = {}
    sym_dfs: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        p = CACHE / f"{sym}_USDT_{smc_tf}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d.index = pd.to_datetime(d.index, utc=True)
        sym_dfs[sym] = d
        atr = add_atr(d)
        sym_setups[sym] = run_smc15_setups(d, atr)

    for method in TP_METHODS:
        per_q: Dict[str, List[Dict[str, Any]]] = {q: [] for q in QUARTER_BOUNDS}
        for sym, setups in sym_setups.items():
            d = sym_dfs[sym]
            for entry_idx, side, entry, ai, ph_dist in setups:
                t = simulate_trade(d, entry_idx, side, entry, ai, ph_dist, method, smc_tfm)
                if t is None:
                    continue
                q = quarter_of(t["entry_ts"])
                if q is None:
                    continue
                per_q[q].append(t)
        is_t = per_q["Q1"] + per_q["Q2"]
        all_results["per_cell"].append({
            "setup_class": "smc15",
            "tf": smc_tf,
            "method": method,
            "params_key": f"smc15|{smc_tf}|{method}",
            "IS": aggregate(is_t),
            "Q3": aggregate(per_q["Q3"]),
            "Q4": aggregate(per_q["Q4"]),
        })

    # ============= Wedge (1h, 4h) =============
    for wedge_tf in ["1h", "4h"]:
        wedge_tfm = tf_minutes(wedge_tf)
        wedge_setups_per_sym: Dict[str, List[WedgeSetup]] = {}
        wedge_dfs: Dict[str, pd.DataFrame] = {}
        for sym in SYMBOLS:
            p = CACHE / f"{sym}_USDT_{wedge_tf}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index, utc=True)
            wedge_dfs[sym] = d
            atr = add_atr(d)
            wedge_setups_per_sym[sym] = detect_wedge_setups(d, atr)

        for method in TP_METHODS:
            per_q: Dict[str, List[Dict[str, Any]]] = {q: [] for q in QUARTER_BOUNDS}
            for sym, setups in wedge_setups_per_sym.items():
                d = wedge_dfs[sym]
                for s in setups:
                    entry_price = float(d["close"].iloc[s.entry_idx])
                    t = simulate_trade(
                        d, s.entry_idx, s.side, entry_price, s.atr_at_entry,
                        s.pattern_height, method, wedge_tfm,
                    )
                    if t is None:
                        continue
                    q = quarter_of(t["entry_ts"])
                    if q is None:
                        continue
                    per_q[q].append(t)
            is_t = per_q["Q1"] + per_q["Q2"]
            all_results["per_cell"].append({
                "setup_class": "wedge",
                "tf": wedge_tf,
                "method": method,
                "params_key": f"wedge|{wedge_tf}|{method}",
                "IS": aggregate(is_t),
                "Q3": aggregate(per_q["Q3"]),
                "Q4": aggregate(per_q["Q4"]),
            })

    # ============= Walk-forward verdicts =============
    # For each (setup_class, tf), best fixed-R baseline = better of fixed_2R / fixed_3R IS EV.
    # Each measured-move variant is judged vs baseline AND OOS pass.
    by_st_tf: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for cell in all_results["per_cell"]:
        key = (cell["setup_class"], cell["tf"])
        by_st_tf.setdefault(key, {})[cell["method"]] = cell

    verdicts: List[Dict[str, Any]] = []
    for (sc, tf), cells in sorted(by_st_tf.items()):
        # baseline = max(fixed_2R, fixed_3R) IS ev_R
        baseline_method = None
        baseline_ev = -1e9
        for fm in ("fixed_2R", "fixed_3R"):
            if fm in cells and cells[fm]["IS"]["n"] >= MIN_IS_TRADES:
                ev = cells[fm]["IS"]["ev_R"]
                if ev > baseline_ev:
                    baseline_ev = ev
                    baseline_method = fm
        for method, cell in cells.items():
            is_ev = cell["IS"]["ev_R"]
            q3_ev = cell["Q3"]["ev_R"]
            q4_ev = cell["Q4"]["ev_R"]
            is_n = cell["IS"]["n"]

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
            wf_pass = q3_pass and q4_pass

            beats_baseline = False
            improvement_R = 0.0
            if baseline_method is not None and method.startswith("measured_move"):
                improvement_R = is_ev - baseline_ev
                beats_baseline = improvement_R >= 0.10

            if method.startswith("measured_move"):
                if is_n < MIN_IS_TRADES:
                    verdict = "INSUFFICIENT"
                elif beats_baseline and wf_pass:
                    verdict = "SHIP"
                elif beats_baseline and (q3_pass or q4_pass):
                    verdict = "HOLD"
                else:
                    verdict = "KILL"
            else:
                # Baselines: standard W/F semantics
                if is_n < MIN_IS_TRADES:
                    verdict = "INSUFFICIENT"
                elif wf_pass:
                    verdict = "SHIP"
                elif q3_pass or q4_pass:
                    verdict = "HOLD"
                else:
                    verdict = "KILL"

            verdicts.append({
                "setup_class": sc,
                "tf": tf,
                "method": method,
                "IS": cell["IS"],
                "Q3": cell["Q3"],
                "Q4": cell["Q4"],
                "Q3_gap_pct": q3_gap,
                "Q4_gap_pct": q4_gap,
                "Q3_pass": q3_pass,
                "Q4_pass": q4_pass,
                "baseline_method": baseline_method,
                "baseline_IS_ev_R": baseline_ev if baseline_method else None,
                "improvement_R": improvement_R if method.startswith("measured_move") else None,
                "beats_baseline_010": beats_baseline if method.startswith("measured_move") else None,
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
            "tp_methods": TP_METHODS,
            "sl_atr": SL_ATR,
            "time_stop_hours": TIME_STOP_HOURS,
            "measured_move_floor_R": MEASURED_MOVE_FLOOR_R,
            "measured_move_cap_R": MEASURED_MOVE_CAP_R,
            "quarter_bounds": QUARTER_BOUNDS,
            "fee_bps": RT_TAKER_FEE_BPS,
            "funding_per_8h": FUNDING_RATE_PER_8H,
            "notional": NOTIONAL,
            "min_is_trades": MIN_IS_TRADES,
        },
        "per_cell": results["per_cell"],
        "walkforward": results["walkforward"],
    }
    out_json.write_text(json.dumps(payload, default=str, indent=2))

    lines: List[str] = []
    lines.append("# Measured-Move TP - Walk-forward Backtest")
    lines.append("")
    lines.append(
        "Tests 5 TP methods on SMC1.5 (1h) and Wedge (1h, 4h) setups. "
        "SL=1xATR, time stop=8h."
    )
    lines.append("")
    lines.append("Pattern-height definition:")
    lines.append("- SMC1.5: |sweep_extreme - bos.broken_price|")
    lines.append("- Wedge:  max gap between upper / lower trendlines inside the wedge")
    lines.append("")
    lines.append("Quarter splits (UTC):")
    for q, (s, e) in QUARTER_BOUNDS.items():
        lines.append(f"- {q}: {s} -> {e}")
    lines.append("")
    lines.append(
        "Pass criteria for measured-move TP variant: IS EV >= best fixed-R baseline + "
        "0.10R AND Q3/Q4 OOS each >= 50% of IS EV with same sign."
    )
    lines.append("")

    lines.append("## Walk-forward verdicts")
    lines.append("")
    lines.append(
        "| Setup | TF | Method | IS n | IS EV(R) | Q3 n | Q3 EV(R) | Q4 n | "
        "Q4 EV(R) | Q3 gap% | Q4 gap% | Improv(R) | Verdict |"
    )
    lines.append(
        "|-------|----|--------|------|----------|------|----------|------|"
        "----------|---------|---------|-----------|---------|"
    )
    for v in results["walkforward"]:
        improv = v.get("improvement_R")
        improv_s = f"{improv:+.3f}" if isinstance(improv, (int, float)) else "-"
        lines.append(
            f"| {v['setup_class']} | {v['tf']} | {v['method']} | {v['IS']['n']} | "
            f"{v['IS']['ev_R']:+.3f} | {v['Q3']['n']} | {v['Q3']['ev_R']:+.3f} | "
            f"{v['Q4']['n']} | {v['Q4']['ev_R']:+.3f} | "
            f"{v['Q3_gap_pct']:+.1f} | {v['Q4_gap_pct']:+.1f} | {improv_s} | "
            f"{v['verdict']} |"
        )
    lines.append("")

    # Per setup x tf summary
    lines.append("## Best method per Setup x TF")
    lines.append("")
    by_st_tf: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for v in results["walkforward"]:
        key = (v["setup_class"], v["tf"])
        cur = by_st_tf.get(key)
        if cur is None or v["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            by_st_tf[key] = v
    for (sc, tf), v in sorted(by_st_tf.items()):
        lines.append(
            f"- **{sc} / {tf}**: best method **{v['method']}** -> IS "
            f"{v['IS']['ev_R']:+.3f}R (n={v['IS']['n']}), Q3 "
            f"{v['Q3']['ev_R']:+.3f}R (n={v['Q3']['n']}), Q4 "
            f"{v['Q4']['ev_R']:+.3f}R (n={v['Q4']['n']}) -> **{v['verdict']}**"
        )
    lines.append("")
    n_ship = sum(1 for v in results["walkforward"]
                 if v["verdict"] == "SHIP" and v["method"].startswith("measured_move"))
    n_hold = sum(1 for v in results["walkforward"]
                 if v["verdict"] == "HOLD" and v["method"].startswith("measured_move"))
    n_kill = sum(1 for v in results["walkforward"]
                 if v["verdict"] == "KILL" and v["method"].startswith("measured_move"))
    n_insuf = sum(1 for v in results["walkforward"] if v["verdict"] == "INSUFFICIENT")
    lines.append(
        f"Measured-move totals: SHIP={n_ship}, HOLD={n_hold}, KILL={n_kill}, "
        f"INSUFFICIENT={n_insuf}"
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append(
        "- SL standardized to 1xATR for fair compare across methods. Shipped "
        "SMC1.5v2 actually uses sweep-wick SL; this study isolates the TP "
        "axis only."
    )
    lines.append(
        "- Pattern height for SMC1.5 = sweep extreme to BOS-broken pivot. "
        "Wider sweeps -> longer measured TPs; bounded by cap."
    )
    lines.append(
        "- Measured-move bounds: floor=0.5R (else trivial wins), cap=6R "
        "(else absurd projection drags WR to 0)."
    )
    lines.append(
        "- Sample sizes for SMC1.5 are small (n<50 per quarter typically) - "
        "verdicts are noisy."
    )
    lines.append("- Fees 0.118% RT taker (Delta India). Funding 0.01%/8h.")
    out_md.write_text("\n".join(lines))


def main() -> int:
    print("[MeasuredMoveTP] Running walk-forward backtest...", flush=True)
    results = run_walkforward()
    write_report(results)
    print(f"[MeasuredMoveTP] Wrote: {OUT_DIR/'walkforward.json'}", flush=True)
    print(f"[MeasuredMoveTP] Wrote: {OUT_DIR/'report.md'}", flush=True)
    print("\nWalk-forward verdicts:", flush=True)
    for v in results["walkforward"]:
        improv = v.get("improvement_R")
        improv_s = f"{improv:+.3f}" if isinstance(improv, (int, float)) else "-"
        print(
            f"  {v['setup_class']}/{v['tf']}/{v['method']}: "
            f"IS={v['IS']['ev_R']:+.3f}R(n={v['IS']['n']}) "
            f"Q3={v['Q3']['ev_R']:+.3f}R Q4={v['Q4']['ev_R']:+.3f}R "
            f"improv={improv_s} -> {v['verdict']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

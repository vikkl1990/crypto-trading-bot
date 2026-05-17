#!/usr/bin/env python3
"""SMC1.5 Entry-Variant Walk-Forward Backtest.

Goal: find which entry placement (Wick / Body / 50% / Engulfing-current) plus
optional trigger pattern (Engulfing / Dragonfly / Gravestone Doji) produces
the best out-of-sample edge for the SMC1.5 paper engine.

Variants (per spec):
  1) ENG_E   — Engulfing trigger (baseline; bullish/bearish engulfing)
  2) ENG_DR  — Engulfing OR Dragonfly Doji (long-side; for short-only signals
                still requires bearish engulfing — Dragonfly only adds long
                triggers)
  3) ENG_GR  — Engulfing OR Gravestone Doji (short-side; mirror of above)
  4) ENG_DR_GR — Engulfing OR Doji (Dragonfly long / Gravestone short)
  5) WICK    — LIMIT at OB extreme nearest to current price (maker fee)
  6) BODY    — LIMIT at OB body extreme (open or close, nearer side; maker)
  7) MID50   — LIMIT at OB midpoint (high+low)/2 (maker fee)

Walk-forward splits (1h candle data spans 2025-10-07 → 2026-04-17 UTC):
  Q1: 2025-10-07 → 2025-11-30  in-sample
  Q2: 2025-12-01 → 2026-01-31  in-sample
  Q3: 2026-02-01 → 2026-02-28  out-of-sample 1
  Q4: 2026-03-01 → 2026-04-17  out-of-sample 2

Pass criterion (per spec):
  - OOS EV >= 50% of IS EV AND same sign (positive)
  - No single quarter < -50% of average
  - Reject in-sample artifacts

Output: storage/smc15_variants/{scoreboard.csv, walkforward.json, report.md}
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
OUT_DIR = ROOT / "storage" / "smc15_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Cost model — Delta India
TAKER_FEE = 0.00059          # 0.059% per leg (market entries / all exits)
MAKER_FEE = 0.00024          # 0.024% per leg (limit entries that rest as maker)
FUNDING_PER_8H = 0.0001
NOTIONAL_USD = 1000.0

# Strategy parameters — match running SMC1.5
SWING_LOOKBACK = 5
SWEEP_LOOKBACK = 20
BOS_LOOKBACK = 30
CHOCH_WINDOW = 15
PATTERN_WINDOW = 30
ATR_LEN = 14
DISPLACEMENT_ATR_MULT = 1.0
RB_WICK_FRAC = 0.55
ENTRY_TOL_ATR = 0.10
LIMIT_FILL_WINDOW_BARS = 6   # limit must fill within 6× 1h after CHoCH
SL_BUFFER_ATR = 0.05

# EB exit config (matches running engine)
TP_R_DEFAULT = 3.0
MAX_BARS_DEFAULT = 8

# Walk-forward quarter boundaries (UTC ISO)
QUARTERS = {
    "Q1": ("2025-10-07T00:00:00+00:00", "2025-11-30T23:59:59+00:00"),
    "Q2": ("2025-12-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00"),
    "Q3": ("2026-02-01T00:00:00+00:00", "2026-02-28T23:59:59+00:00"),
    "Q4": ("2026-03-01T00:00:00+00:00", "2026-04-17T23:59:59+00:00"),
}

VARIANTS = ["ENG_E", "ENG_DR", "ENG_GR", "ENG_DR_GR", "WICK", "BODY", "MID50"]


# ─────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────

def load_candles(sym: str, tf: str = "1h") -> pd.DataFrame:
    p = CACHE_DIR / f"{sym}_USDT_{tf}.parquet"
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df = df.sort_index()
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


# ─────────────────────────────────────────────────────────
# SMC primitives — verbatim from smc15_fullstack_backtest.py
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
    direction: str
    sweep_extreme: float
    swept_price: float
    time: pd.Timestamp


@dataclass
class BOS:
    idx: int
    broken_pivot_idx: int
    direction: str
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
    direction: str
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
    parent_ob: OrderBlock
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


def find_swing_pivots(df, lookback=SWING_LOOKBACK):
    pivots = []
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


def detect_liquidity_sweeps(df, pivots):
    sweeps = []
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


def detect_bos(df, pivots):
    bos = []
    pivot_highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.idx)
    pivot_lows = sorted([p for p in pivots if p.kind == "low"], key=lambda p: p.idx)
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        c = float(df.iloc[i].close)
        recent_highs = [p for p in pivot_highs if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_highs):
            if c > p.price:
                bos.append(BOS(i, p.idx, "up", p.price, df.index[i]))
                break
        recent_lows = [p for p in pivot_lows if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_lows):
            if c < p.price:
                bos.append(BOS(i, p.idx, "down", p.price, df.index[i]))
                break
    return bos


def detect_choch_after(df, pivots, bos, window=CHOCH_WINDOW):
    n = len(df)
    end = min(n, bos.idx + window)
    if bos.direction == "down":
        pre = [p for p in pivots if p.kind == "high" and p.idx <= bos.idx]
        if not pre:
            return None
        last_high = max(pre, key=lambda p: p.idx)
        post = [p for p in pivots if p.kind == "high" and bos.idx < p.idx <= end]
        for p in post:
            if p.price < last_high.price:
                return CHoCH(p.idx, "down", p.idx, p.time)
    else:
        pre = [p for p in pivots if p.kind == "low" and p.idx <= bos.idx]
        if not pre:
            return None
        last_low = max(pre, key=lambda p: p.idx)
        post = [p for p in pivots if p.kind == "low" and bos.idx < p.idx <= end]
        for p in post:
            if p.price > last_low.price:
                return CHoCH(p.idx, "up", p.idx, p.time)
    return None


def detect_order_blocks(df):
    obs = []
    n = len(df)
    for i in range(2, n):
        bar = df.iloc[i]
        if pd.isna(bar.atr) or bar.atr <= 0 or bar.range <= 0:
            continue
        if (bar.range < DISPLACEMENT_ATR_MULT * bar.atr or bar.body / bar.range < 0.55):
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
                    time=df.index[j],
                ))
                break
            if (not is_bull) and prev.close > prev.open:
                obs.append(OrderBlock(
                    formed_idx=j, direction="bear",
                    high=float(prev.high), low=float(prev.low),
                    midpoint=float((prev.high + prev.low) / 2),
                    body_high=float(body_high), body_low=float(body_low),
                    displacement_atr=float(bar.range / bar.atr),
                    time=df.index[j],
                ))
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


def detect_breakers(obs):
    bks = []
    for ob in obs:
        if ob.mitigated_idx is None:
            continue
        flip_dir = "bear" if ob.direction == "bull" else "bull"
        bks.append(Breaker(
            parent_ob=ob,
            flip_idx=ob.mitigated_idx,
            direction=flip_dir,
            high=ob.high, low=ob.low,
            midpoint=ob.midpoint,
            body_high=ob.body_high, body_low=ob.body_low,
            time=ob.time,
        ))
    return bks


def detect_rejection_blocks(df, pivots):
    rbs = []
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
                        near_pivot_idx=top.idx, time=df.index[i],
                    ))
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
                        near_pivot_idx=bot.idx, time=df.index[i],
                    ))
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
    zone_body_high: float
    zone_body_low: float
    side: str
    sl_anchor: float
    setup_idx: int


def assemble_setups(df, pivots, sweeps, bos_events, obs, brks, rbs):
    setups = []
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
        zones = []  # (kind, high, low, mid, body_high, body_low)
        for ob in obs:
            if ob.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= ob.formed_idx <= choch.idx:
                    zones.append(("OB", ob.high, ob.low, ob.midpoint,
                                  ob.body_high, ob.body_low))
        for brk in brks:
            if brk.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= brk.flip_idx <= choch.idx + 5:
                    zones.append(("BRK", brk.high, brk.low, brk.midpoint,
                                  brk.body_high, brk.body_low))
        for rb in rbs:
            if rb.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx - 1 <= rb.idx <= sweep.idx + 1:
                    zones.append(("RB", rb.high, rb.low, rb.midpoint,
                                  rb.body_high, rb.body_low))
        if not zones:
            continue
        if side == "short":
            best = min(zones, key=lambda z: abs(z[2] - sweep.swept_price))
        else:
            best = min(zones, key=lambda z: abs(z[1] - sweep.swept_price))
        kind, zh, zl, zm, zbh, zbl = best
        setups.append(Setup(
            sweep=sweep, bos=first_bos, choch=choch,
            zone_kind=kind, zone_high=zh, zone_low=zl,
            zone_midpoint=zm, zone_body_high=zbh, zone_body_low=zbl,
            side=side, sl_anchor=sweep.sweep_extreme, setup_idx=choch.idx,
        ))
    return setups


# ─────────────────────────────────────────────────────────
# Trigger-pattern detection (for engulfing-family variants)
# ─────────────────────────────────────────────────────────

def is_bullish_engulfing(bar, prev) -> bool:
    return (bar.close > bar.open and bar.close > prev.close)


def is_bearish_engulfing(bar, prev) -> bool:
    return (bar.close < bar.open and bar.close < prev.close)


def is_dragonfly_doji(bar) -> bool:
    """Dragonfly: lower wick > 2× body, body in upper third, close >= open.
    Bullish reversal pattern."""
    if bar.range <= 0:
        return False
    body = abs(bar.close - bar.open)
    body_top = max(bar.open, bar.close)
    if body <= 0:
        body = bar.range * 0.05  # treat near-zero body as small
    if bar.lower_wick < 2 * body:
        return False
    # body in upper third
    body_pos = (body_top - bar.low) / bar.range
    if body_pos < 2.0 / 3.0:
        return False
    if bar.close < bar.open:  # must close >= open
        return False
    return True


def is_gravestone_doji(bar) -> bool:
    """Gravestone: upper wick > 2× body, body in lower third, close <= open.
    Bearish reversal pattern."""
    if bar.range <= 0:
        return False
    body = abs(bar.close - bar.open)
    body_bot = min(bar.open, bar.close)
    if body <= 0:
        body = bar.range * 0.05
    if bar.upper_wick < 2 * body:
        return False
    body_pos = (bar.high - body_bot) / bar.range
    if body_pos < 2.0 / 3.0:
        return False
    if bar.close > bar.open:
        return False
    return True


def trigger_fires(variant: str, side: str, bar, prev) -> bool:
    """Returns True if the trigger pattern for `variant` fires on `bar`."""
    if side == "long":
        eng = is_bullish_engulfing(bar, prev)
        if variant == "ENG_E":
            return eng
        if variant == "ENG_DR":
            return eng or is_dragonfly_doji(bar)
        if variant == "ENG_GR":
            return eng  # gravestone is bearish-only, so long uses engulfing
        if variant == "ENG_DR_GR":
            return eng or is_dragonfly_doji(bar)
    else:  # short
        eng = is_bearish_engulfing(bar, prev)
        if variant == "ENG_E":
            return eng
        if variant == "ENG_DR":
            return eng  # dragonfly is bullish-only
        if variant == "ENG_GR":
            return eng or is_gravestone_doji(bar)
        if variant == "ENG_DR_GR":
            return eng or is_gravestone_doji(bar)
    return False


# ─────────────────────────────────────────────────────────
# Entry simulation per variant
# ─────────────────────────────────────────────────────────

def simulate_engulfing_variant(df, setup, variant, exit_cfg) -> Optional[dict]:
    """Trigger-based market entry (taker fee)."""
    n = len(df)
    n_max = min(n, setup.setup_idx + PATTERN_WINDOW)
    atr_at_choch = float(df.iloc[setup.setup_idx].atr) if not pd.isna(df.iloc[setup.setup_idx].atr) else 0.0
    if atr_at_choch <= 0:
        return None
    tol = ENTRY_TOL_ATR * atr_at_choch
    entry_idx = None
    entry_price = None
    for i in range(setup.setup_idx + 1, n_max):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        if setup.side == "short":
            if bar.high < setup.zone_low - tol:
                continue
            if bar.low > setup.zone_high + tol:
                continue
        else:
            if bar.high < setup.zone_low - tol:
                continue
            if bar.low > setup.zone_high + tol:
                continue
        if trigger_fires(variant, setup.side, bar, prev):
            entry_idx = i
            entry_price = float(bar.close)
            break
    if entry_idx is None:
        return None
    return _finalize_simulate(df, setup, entry_idx, entry_price,
                              atr_at_choch, exit_cfg, fee_in=TAKER_FEE)


def simulate_limit_variant(df, setup, variant, exit_cfg) -> Optional[dict]:
    """Limit-order entry (maker fee). Variant determines limit price."""
    n = len(df)
    n_max = min(n, setup.setup_idx + 1 + LIMIT_FILL_WINDOW_BARS)
    atr_at_choch = float(df.iloc[setup.setup_idx].atr) if not pd.isna(df.iloc[setup.setup_idx].atr) else 0.0
    if atr_at_choch <= 0:
        return None

    # Determine limit price based on variant + side
    if variant == "WICK":
        # OB extreme nearest to current price approaching from outside.
        # Long OB: price is ABOVE the zone (after CHoCH up), pulls back into
        # zone. The edge nearest to where price is approaching from is the
        # OB HIGH. For SHORT: price below zone, approach from below → OB LOW.
        if setup.side == "long":
            limit_price = setup.zone_high
        else:
            limit_price = setup.zone_low
    elif variant == "BODY":
        # OB body extreme nearest entry side
        if setup.side == "long":
            limit_price = setup.zone_body_high
        else:
            limit_price = setup.zone_body_low
    elif variant == "MID50":
        limit_price = setup.zone_midpoint
    else:
        return None

    # Fill check: scan bars from setup_idx+1 to n_max for first bar where
    # price reaches the limit level. For LONG, fill when bar.low <= limit
    # (price falls into zone). For SHORT, fill when bar.high >= limit
    # (price rises into zone).
    entry_idx = None
    entry_price = None
    for i in range(setup.setup_idx + 1, n_max):
        bar = df.iloc[i]
        if setup.side == "long":
            if bar.low <= limit_price:
                entry_idx = i
                # Conservative fill: limit_price (best case for maker)
                # if the bar opened below the limit, assume immediate fill at open
                entry_price = min(float(bar.open), limit_price)
                # If open was above limit and bar wicked down, fill at limit
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
    return _finalize_simulate(df, setup, entry_idx, entry_price,
                              atr_at_choch, exit_cfg, fee_in=MAKER_FEE)


def _finalize_simulate(df, setup, entry_idx, entry_price,
                       atr_at_choch, exit_cfg, fee_in: float) -> Optional[dict]:
    if setup.side == "short":
        sl = setup.sl_anchor + SL_BUFFER_ATR * atr_at_choch
    else:
        sl = setup.sl_anchor - SL_BUFFER_ATR * atr_at_choch
    R = abs(entry_price - sl)
    if R <= 0:
        return None
    if R > 3.0 * atr_at_choch:
        return None
    return simulate_exit(df, entry_idx, entry_price, sl, R, setup,
                         exit_cfg, fee_in=fee_in)


def simulate_exit(df, entry_idx, entry, sl, R, setup, cfg, fee_in: float) -> Optional[dict]:
    """Simulate exits with EB config defaults: 3R fixed TP, 8h time stop."""
    n = len(df)
    side = setup.side
    bars_held = 0
    max_bars = cfg.get("max_bars", MAX_BARS_DEFAULT)
    fixed_tp_R = cfg.get("fixed_tp_R", TP_R_DEFAULT)
    qty = NOTIONAL_USD / entry
    if side == "short":
        fixed_tp_price = entry - fixed_tp_R * R
    else:
        fixed_tp_price = entry + fixed_tp_R * R

    exit_idx = None
    exit_price = None
    exit_reason = None

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        bars_held = j - entry_idx
        if bars_held > max_bars:
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
            if bar.low <= fixed_tp_price:
                exit_idx = j
                exit_price = fixed_tp_price
                exit_reason = "tp_fixed"
                break
        else:
            if bar.low <= sl:
                exit_idx = j
                exit_price = sl
                exit_reason = "sl_hit"
                break
            if bar.high >= fixed_tp_price:
                exit_idx = j
                exit_price = fixed_tp_price
                exit_reason = "tp_fixed"
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

    # Fees: entry uses fee_in (maker for limits, taker for market)
    # Exits always taker
    entry_fee = NOTIONAL_USD * fee_in
    exit_fee = (qty * exit_price) * TAKER_FEE
    fees = entry_fee + exit_fee

    hours_held = bars_held
    funding_periods = hours_held / 8.0
    funding_cost = NOTIONAL_USD * FUNDING_PER_8H * funding_periods

    net = gross - fees - funding_cost
    return {
        "side": side,
        "zone_kind": setup.zone_kind,
        "entry_idx": entry_idx,
        "entry_time": df.index[entry_idx].isoformat(),
        "entry": float(entry),
        "sl": float(sl),
        "R": float(R),
        "exit_idx": exit_idx,
        "exit_time": df.index[exit_idx].isoformat(),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "gross_pnl": float(gross),
        "fees": float(fees),
        "funding": float(funding_cost),
        "net_pnl": float(net),
        "fee_in": fee_in,
    }


def simulate_setup_for_variant(df, setup, variant, exit_cfg=None) -> Optional[dict]:
    if exit_cfg is None:
        exit_cfg = {"fixed_tp_R": TP_R_DEFAULT, "max_bars": MAX_BARS_DEFAULT}
    if variant in ("ENG_E", "ENG_DR", "ENG_GR", "ENG_DR_GR"):
        return simulate_engulfing_variant(df, setup, variant, exit_cfg)
    elif variant in ("WICK", "BODY", "MID50"):
        return simulate_limit_variant(df, setup, variant, exit_cfg)
    return None


# ─────────────────────────────────────────────────────────
# Walk-forward execution
# ─────────────────────────────────────────────────────────

def run_symbol(sym: str) -> dict:
    """Run all 7 variants on a single symbol, returning trades partitioned by quarter."""
    df = load_candles(sym, "1h")
    df = add_atr(df)
    pivots = find_swing_pivots(df)
    sweeps = detect_liquidity_sweeps(df, pivots)
    bos_events = detect_bos(df, pivots)
    obs = detect_order_blocks(df)
    brks = detect_breakers(obs)
    rbs = detect_rejection_blocks(df, pivots)
    setups = assemble_setups(df, pivots, sweeps, bos_events, obs, brks, rbs)

    quarters_ts = {
        q: (pd.Timestamp(s), pd.Timestamp(e)) for q, (s, e) in QUARTERS.items()
    }

    trades_by_variant: dict[str, dict[str, list]] = {
        v: {q: [] for q in QUARTERS} for v in VARIANTS
    }
    for variant in VARIANTS:
        for s in setups:
            tr = simulate_setup_for_variant(df, s, variant)
            if tr is None:
                continue
            tr["symbol"] = sym
            tr["variant"] = variant
            tr["setup_zone_kind"] = s.zone_kind
            entry_t = pd.Timestamp(tr["entry_time"])
            for q, (qs, qe) in quarters_ts.items():
                if qs <= entry_t <= qe:
                    trades_by_variant[variant][q].append(tr)
                    tr["quarter"] = q
                    break
    return {"sym": sym, "trades_by_variant": trades_by_variant,
            "n_setups": len(setups)}


def stats(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "net_$": 0.0, "ev_per_trade": 0.0, "gross_$": 0.0}
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    net = sum(t["net_pnl"] for t in trades)
    gross = sum(t["gross_pnl"] for t in trades)
    return {
        "n": n,
        "wr": 100.0 * wins / n,
        "net_$": net,
        "ev_per_trade": net / n,
        "gross_$": gross,
    }


def walk_forward_verdict(is_stats: dict, oos_q3: dict, oos_q4: dict) -> dict:
    """Apply pass criterion:
       - OOS EV >= 50% of IS EV (per quarter) AND same sign positive
       - No single quarter < -50% of average (Q1+Q2 IS, Q3, Q4)
    """
    is_ev = is_stats["ev_per_trade"]
    q3_ev = oos_q3["ev_per_trade"]
    q4_ev = oos_q4["ev_per_trade"]

    notes = []
    pass_q3 = False
    pass_q4 = False

    # Q3 test
    if oos_q3["n"] < 3:
        notes.append("Q3: insufficient OOS sample (<3)")
    elif is_ev <= 0:
        notes.append(f"Q3: IS EV non-positive ({is_ev:+.3f})")
    elif q3_ev <= 0:
        notes.append(f"Q3: OOS EV non-positive ({q3_ev:+.3f})")
    elif q3_ev < 0.5 * is_ev:
        notes.append(f"Q3: gap > 50% (IS={is_ev:+.3f} OOS={q3_ev:+.3f})")
    else:
        pass_q3 = True

    # Q4 test
    if oos_q4["n"] < 3:
        notes.append("Q4: insufficient OOS sample (<3)")
    elif is_ev <= 0:
        pass
    elif q4_ev <= 0:
        notes.append(f"Q4: OOS EV non-positive ({q4_ev:+.3f})")
    elif q4_ev < 0.5 * is_ev:
        notes.append(f"Q4: gap > 50% (IS={is_ev:+.3f} OOS={q4_ev:+.3f})")
    else:
        pass_q4 = True

    # No single quarter < -50% of average (compute avg EV across populated quarters)
    quarters_ev = [(q, qs["ev_per_trade"], qs["n"]) for q, qs in (
        ("IS", is_stats), ("Q3", oos_q3), ("Q4", oos_q4)) if qs["n"] > 0]
    if quarters_ev:
        avg_ev = sum(ev for _, ev, _ in quarters_ev) / len(quarters_ev)
        for q, ev, n in quarters_ev:
            if avg_ev > 0 and ev < -0.5 * avg_ev:
                notes.append(f"{q}: drawdown > 50% of avg (avg={avg_ev:+.3f}, q={ev:+.3f})")
                pass_q3 = pass_q4 = False

    verdict = "PASS" if (pass_q3 and pass_q4) else "FAIL"
    return {
        "verdict": verdict,
        "notes": notes,
        "is_ev": is_ev,
        "q3_ev": q3_ev,
        "q4_ev": q4_ev,
        "is_n": is_stats["n"],
        "q3_n": oos_q3["n"],
        "q4_n": oos_q4["n"],
        "is_net": is_stats["net_$"],
        "q3_net": oos_q3["net_$"],
        "q4_net": oos_q4["net_$"],
    }


def main() -> None:
    print("Loading symbols and running variant simulations...")
    per_sym = {}
    for sym in SYMBOLS:
        try:
            r = run_symbol(sym)
            per_sym[sym] = r
            print(f"  {sym}: {r['n_setups']} setups assembled")
        except Exception as e:
            import traceback
            print(f"  {sym}: ERROR {e}\n{traceback.format_exc()}", file=sys.stderr)
    print()

    # Aggregate by variant + quarter (across all symbols)
    agg_var_q: dict[str, dict[str, list]] = {v: {q: [] for q in QUARTERS}
                                              for v in VARIANTS}
    for sym in SYMBOLS:
        if sym not in per_sym:
            continue
        for v in VARIANTS:
            for q in QUARTERS:
                agg_var_q[v][q].extend(per_sym[sym]["trades_by_variant"][v][q])

    # Per-variant aggregate stats per quarter
    by_variant = {}
    for v in VARIANTS:
        per_q = {q: stats(agg_var_q[v][q]) for q in QUARTERS}
        is_trades = agg_var_q[v]["Q1"] + agg_var_q[v]["Q2"]
        is_stats_combined = stats(is_trades)
        # Average per-IS-quarter EV vs OOS quarter EV
        wf = walk_forward_verdict(is_stats_combined, per_q["Q3"], per_q["Q4"])
        # Total: all trades
        total_trades = is_trades + agg_var_q[v]["Q3"] + agg_var_q[v]["Q4"]
        by_variant[v] = {
            "per_quarter": per_q,
            "is_combined": is_stats_combined,
            "total": stats(total_trades),
            "walkforward": wf,
        }

    # Build scoreboard CSV
    rows = []
    for v in VARIANTS:
        bv = by_variant[v]
        for q in QUARTERS:
            s = bv["per_quarter"][q]
            rows.append({
                "variant": v, "quarter": q,
                "n": s["n"], "wr": round(s["wr"], 2),
                "net_$": round(s["net_$"], 2),
                "ev_per_trade": round(s["ev_per_trade"], 3),
            })
        rows.append({
            "variant": v, "quarter": "IS_Q1+Q2",
            "n": bv["is_combined"]["n"],
            "wr": round(bv["is_combined"]["wr"], 2),
            "net_$": round(bv["is_combined"]["net_$"], 2),
            "ev_per_trade": round(bv["is_combined"]["ev_per_trade"], 3),
        })
        rows.append({
            "variant": v, "quarter": "TOTAL",
            "n": bv["total"]["n"],
            "wr": round(bv["total"]["wr"], 2),
            "net_$": round(bv["total"]["net_$"], 2),
            "ev_per_trade": round(bv["total"]["ev_per_trade"], 3),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "scoreboard.csv", index=False)

    # Walk-forward verdict JSON
    wf_summary = {
        v: by_variant[v]["walkforward"] for v in VARIANTS
    }
    (OUT_DIR / "walkforward.json").write_text(
        json.dumps({"by_variant": wf_summary,
                    "per_variant_per_quarter": {
                        v: {q: by_variant[v]["per_quarter"][q] for q in QUARTERS}
                        for v in VARIANTS
                    }}, indent=2, default=str))

    # Markdown report
    md = ["# SMC1.5 Entry-Variant Walk-Forward Backtest\n"]
    md.append("4 symbols (BTC/ETH/SOL/XRP) × 1h candles, 6.4 months data.\n")
    md.append(f"Quarter splits:")
    for q, (qs, qe) in QUARTERS.items():
        md.append(f"  - {q}: {qs[:10]} → {qe[:10]}")
    md.append("")
    md.append("## Per-variant table\n")
    md.append("| Variant | IS Q1+Q2 n / EV | Q3 OOS n / EV | Q3 gap% | Q4 OOS n / EV | Q4 gap% | Verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        bv = by_variant[v]
        wf = bv["walkforward"]
        is_ev = bv["is_combined"]["ev_per_trade"]
        q3_ev = bv["per_quarter"]["Q3"]["ev_per_trade"]
        q4_ev = bv["per_quarter"]["Q4"]["ev_per_trade"]
        q3_gap = (q3_ev / is_ev * 100) if is_ev != 0 else 0.0
        q4_gap = (q4_ev / is_ev * 100) if is_ev != 0 else 0.0
        md.append(
            f"| {v} | n={bv['is_combined']['n']} EV=${is_ev:+.3f} | "
            f"n={bv['per_quarter']['Q3']['n']} EV=${q3_ev:+.3f} | "
            f"{q3_gap:+.0f}% | "
            f"n={bv['per_quarter']['Q4']['n']} EV=${q4_ev:+.3f} | "
            f"{q4_gap:+.0f}% | **{wf['verdict']}** |"
        )

    md.append("\n## Walk-forward notes\n")
    for v in VARIANTS:
        wf = by_variant[v]["walkforward"]
        if wf["notes"]:
            md.append(f"- **{v}** ({wf['verdict']}): {'; '.join(wf['notes'])}")
        else:
            md.append(f"- **{v}**: {wf['verdict']} (no warnings)")

    md.append("\n## Per-quarter detail\n")
    for v in VARIANTS:
        md.append(f"### {v}")
        md.append("| Quarter | n | WR% | Net$ | EV/trade |")
        md.append("|---|---|---|---|---|")
        for q in QUARTERS:
            s = by_variant[v]["per_quarter"][q]
            md.append(
                f"| {q} | {s['n']} | {s['wr']:.1f} | "
                f"${s['net_$']:+.2f} | ${s['ev_per_trade']:+.3f} |"
            )
        md.append("")

    # Best variant ranked by Q4 EV (most stringent)
    passing = [(v, by_variant[v]) for v in VARIANTS
               if by_variant[v]["walkforward"]["verdict"] == "PASS"]
    if passing:
        passing.sort(key=lambda x: x[1]["per_quarter"]["Q4"]["ev_per_trade"], reverse=True)
        best_v, best = passing[0]
        md.append(f"\n## Champion: **{best_v}** (PASS)")
        md.append(f"- IS Q1+Q2: n={best['is_combined']['n']} EV=${best['is_combined']['ev_per_trade']:+.3f}")
        md.append(f"- Q3 OOS: n={best['per_quarter']['Q3']['n']} EV=${best['per_quarter']['Q3']['ev_per_trade']:+.3f}")
        md.append(f"- Q4 OOS: n={best['per_quarter']['Q4']['n']} EV=${best['per_quarter']['Q4']['ev_per_trade']:+.3f}")
    else:
        # Closest to passing: rank by Q4 EV among variants where IS EV positive
        candidates = [(v, by_variant[v]) for v in VARIANTS
                      if by_variant[v]["is_combined"]["ev_per_trade"] > 0]
        candidates.sort(key=lambda x: x[1]["per_quarter"]["Q4"]["ev_per_trade"], reverse=True)
        if candidates:
            top_v, top = candidates[0]
            md.append(f"\n## NO walk-forward PASS — closest: **{top_v}** (FAIL)")
            md.append(f"- Q4 OOS EV=${top['per_quarter']['Q4']['ev_per_trade']:+.3f} — keep current SMC1.5 baseline")
        else:
            md.append("\n## NO variant has positive in-sample EV — keep SMC1.5 baseline")

    # Maker fee benefit quantification
    md.append("\n## Maker fee benefit\n")
    md.append("- Taker entry fee: 0.059% × $1000 = $0.59/trade")
    md.append("- Maker entry fee: 0.024% × $1000 = $0.24/trade")
    md.append("- Savings per limit-entry trade: $0.35")
    for v in ("WICK", "BODY", "MID50"):
        n = by_variant[v]["total"]["n"]
        md.append(f"- {v} total trades: {n} → fee savings = $${n * 0.35:.2f}")

    (OUT_DIR / "report.md").write_text("\n".join(md))

    # Console summary
    print("=" * 80)
    print("WALK-FORWARD VERDICT SUMMARY")
    print("=" * 80)
    for v in VARIANTS:
        bv = by_variant[v]
        wf = bv["walkforward"]
        print(f"  {v:12s}: IS n={bv['is_combined']['n']:3d} EV=${bv['is_combined']['ev_per_trade']:+.3f} "
              f"| Q3 n={bv['per_quarter']['Q3']['n']:2d} EV=${bv['per_quarter']['Q3']['ev_per_trade']:+.3f} "
              f"| Q4 n={bv['per_quarter']['Q4']['n']:2d} EV=${bv['per_quarter']['Q4']['ev_per_trade']:+.3f} "
              f"→ {wf['verdict']}")
    print()
    print(f"Outputs: {OUT_DIR}/report.md  scoreboard.csv  walkforward.json")


if __name__ == "__main__":
    main()

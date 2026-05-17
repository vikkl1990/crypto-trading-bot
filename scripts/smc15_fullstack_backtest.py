#!/usr/bin/env python3
"""SMC1.5 fullstack backtest — Liquidity Sweep + BOS + CHoCH + OB + BRK + RB.

Tests the FULL institutional SMC pattern (not the simplified OB-only SMC1).

Pattern flow (SHORT example):
  1. BSL (recent swing high) gets swept (wick above + close back inside)
  2. BOS down (close below prior swing low) confirms trend break
  3. CHoCH = first lower-high after BOS confirms character change
  4. Pullback to -OB (last bullish candle before impulse down) OR -BRK
     (a previously-bullish OB that got mitigated and flipped role)
  5. Bearish reversal candle inside OB/BRK zone fires entry
  6. SL above original sweep wick (BSL high) + 1 tick
  7. Targets: TP1 = nearest SSL (next swing low), TP2 = 2× TP1 distance

LONG = mirror.

4 exit configs × 4 symbols = 16 cells.
"""
from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
CACHE_DIR = ROOT / "storage" / "candle_cache"
OUT_DIR = ROOT / "storage" / "smc15"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Cost model — Delta India taker
TAKER_FEE = 0.00059          # 0.059% per leg
FUNDING_PER_8H = 0.0001      # 0.01% / 8h
NOTIONAL_USD = 1000.0

# Strategy parameters
SWING_LOOKBACK = 5            # pivot detection: N-bar fractal (high > N bars left & right)
SWEEP_LOOKBACK = 20           # how far back to scan for the swing being swept
BOS_LOOKBACK = 30             # how far back to find the prior pivot for BOS confirmation
CHOCH_WINDOW = 15             # window after BOS in which to find first counter-trend break
PATTERN_WINDOW = 30           # max bars from sweep → entry trigger
ATR_LEN = 14
DISPLACEMENT_ATR_MULT = 1.0   # impulse leg must be >= 1.0× ATR
OB_MAX_AGE_BARS = 30
RB_WICK_FRAC = 0.55           # rejection block: wick must be >= 55% of range
ENTRY_TOL_ATR = 0.10          # entry zone tolerance


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
# SMC primitives
# ─────────────────────────────────────────────────────────

@dataclass
class Pivot:
    idx: int
    price: float
    kind: str  # "high" or "low"
    time: pd.Timestamp


@dataclass
class Sweep:
    idx: int                # bar that did the sweep (wick beyond + close back inside)
    swept_pivot_idx: int    # the pivot that got swept
    direction: str          # "bsl" (swept high → bear setup) or "ssl" (swept low → bull)
    sweep_extreme: float    # the wick extreme (used as SL anchor)
    swept_price: float      # the pivot price that was swept
    time: pd.Timestamp


@dataclass
class BOS:
    idx: int                # bar that closed through prior pivot
    broken_pivot_idx: int
    direction: str          # "down" or "up"
    broken_price: float
    time: pd.Timestamp


@dataclass
class CHoCH:
    idx: int                # bar of the counter-trend break
    direction: str          # direction of the new trend after CHoCH ("up" or "down")
    pivot_idx: int          # the LH (or HL) pivot that confirms CHoCH
    time: pd.Timestamp


@dataclass
class OrderBlock:
    formed_idx: int
    direction: str          # "bull" (long zone) / "bear" (short zone)
    high: float
    low: float
    midpoint: float
    displacement_atr: float
    mitigated_idx: Optional[int] = None  # bar where it was mitigated (if any)
    time: pd.Timestamp = None


@dataclass
class Breaker:
    """Breaker = OB that got mitigated and now trades in OPPOSITE direction."""
    parent_ob: OrderBlock
    flip_idx: int           # bar at which we declare the flip (= mitigation bar)
    direction: str          # opposite of parent OB direction
    high: float
    low: float
    time: pd.Timestamp


@dataclass
class RejectionBlock:
    idx: int
    direction: str          # "bear" (shooting star = short setup) / "bull" (hammer = long)
    high: float
    low: float
    near_pivot_idx: int     # pivot that the wick attacked
    time: pd.Timestamp


def find_swing_pivots(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> list[Pivot]:
    """Fractal pivots: bar i is a pivot high if its high is the max in [i-lookback, i+lookback]."""
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
    """Sweep: bar wick goes beyond a recent swing extreme but body closes back inside."""
    sweeps: list[Sweep] = []
    pivot_highs = [p for p in pivots if p.kind == "high"]
    pivot_lows = [p for p in pivots if p.kind == "low"]
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        bar = df.iloc[i]
        # Recent unswept pivot highs (BSL)
        recent_highs = [p for p in pivot_highs
                        if (i - SWEEP_LOOKBACK) <= p.idx < i]
        if recent_highs:
            highest_recent = max(recent_highs, key=lambda p: p.price)
            if (bar.high > highest_recent.price and
                    bar.close < highest_recent.price):
                sweeps.append(Sweep(
                    idx=i,
                    swept_pivot_idx=highest_recent.idx,
                    direction="bsl",
                    sweep_extreme=float(bar.high),
                    swept_price=highest_recent.price,
                    time=df.index[i],
                ))
        recent_lows = [p for p in pivot_lows
                       if (i - SWEEP_LOOKBACK) <= p.idx < i]
        if recent_lows:
            lowest_recent = min(recent_lows, key=lambda p: p.price)
            if (bar.low < lowest_recent.price and
                    bar.close > lowest_recent.price):
                sweeps.append(Sweep(
                    idx=i,
                    swept_pivot_idx=lowest_recent.idx,
                    direction="ssl",
                    sweep_extreme=float(bar.low),
                    swept_price=lowest_recent.price,
                    time=df.index[i],
                ))
    return sweeps


def detect_bos(df: pd.DataFrame, pivots: list[Pivot]) -> list[BOS]:
    """BOS: bar close passes through a prior pivot."""
    bos_events: list[BOS] = []
    pivot_highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.idx)
    pivot_lows = sorted([p for p in pivots if p.kind == "low"], key=lambda p: p.idx)
    n = len(df)
    for i in range(SWING_LOOKBACK, n):
        c = float(df.iloc[i].close)
        # Bullish BOS: close above the most recent unbroken pivot high
        recent_highs = [p for p in pivot_highs if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_highs):
            if c > p.price:
                bos_events.append(BOS(i, p.idx, "up", p.price, df.index[i]))
                break
        # Bearish BOS
        recent_lows = [p for p in pivot_lows if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_lows):
            if c < p.price:
                bos_events.append(BOS(i, p.idx, "down", p.price, df.index[i]))
                break
    return bos_events


def detect_choch_after(df: pd.DataFrame, pivots: list[Pivot],
                       bos: BOS, window: int = CHOCH_WINDOW) -> Optional[CHoCH]:
    """After BOS down, CHoCH up = first higher-high.
    After BOS up, CHoCH down = first lower-low.
    But for the SMC entry pattern (sweep + BOS + CHoCH → OB retest),
    we want the SAME-direction CHoCH that confirms continuation:
      sweep BSL → BOS down → first LH formed = CHoCH (still bearish continuation
      structurally, but it's the 'character' confirmation of bearish leg).
    Implementation: after BOS down, find first pivot that is LOWER than the
    pre-BOS pivot high (= a lower-high). After BOS up, find first higher-low.
    """
    n = len(df)
    end = min(n, bos.idx + window)
    if bos.direction == "down":
        # Find first pivot high after BOS that is LOWER than the swing pivot
        # before BOS — confirms the lower-high (character break of any uptrend)
        pre_pivot = [p for p in pivots if p.kind == "high" and p.idx <= bos.idx]
        if not pre_pivot:
            return None
        last_high = max(pre_pivot, key=lambda p: p.idx)
        post_highs = [p for p in pivots if p.kind == "high"
                      and bos.idx < p.idx <= end]
        for p in post_highs:
            if p.price < last_high.price:
                return CHoCH(p.idx, "down", p.idx, p.time)
    else:  # bos.direction == "up"
        pre_pivot = [p for p in pivots if p.kind == "low" and p.idx <= bos.idx]
        if not pre_pivot:
            return None
        last_low = max(pre_pivot, key=lambda p: p.idx)
        post_lows = [p for p in pivots if p.kind == "low"
                     and bos.idx < p.idx <= end]
        for p in post_lows:
            if p.price > last_low.price:
                return CHoCH(p.idx, "up", p.idx, p.time)
    return None


def detect_order_blocks(df: pd.DataFrame) -> list[OrderBlock]:
    """OB = last opposite-color candle BEFORE a displacement leg.

    Tracks mitigation: an OB is 'mitigated' the first time price closes
    past its midpoint after formation. Mitigation timestamp recorded for
    Breaker promotion logic.
    """
    obs: list[OrderBlock] = []
    n = len(df)
    for i in range(2, n):
        bar = df.iloc[i]
        if pd.isna(bar.atr) or bar.atr <= 0 or bar.range <= 0:
            continue
        if (bar.range < DISPLACEMENT_ATR_MULT * bar.atr or
                bar.body / bar.range < 0.55):
            continue
        is_bull = bar.close > bar.open
        # Find last opposite-color candle within 5 bars back
        for j in range(i - 1, max(i - 6, 0), -1):
            prev = df.iloc[j]
            if is_bull and prev.close < prev.open:
                obs.append(OrderBlock(
                    formed_idx=j,
                    direction="bull",
                    high=float(prev.high),
                    low=float(prev.low),
                    midpoint=float((prev.high + prev.low) / 2),
                    displacement_atr=float(bar.range / bar.atr),
                    time=df.index[j],
                ))
                break
            if (not is_bull) and prev.close > prev.open:
                obs.append(OrderBlock(
                    formed_idx=j,
                    direction="bear",
                    high=float(prev.high),
                    low=float(prev.low),
                    midpoint=float((prev.high + prev.low) / 2),
                    displacement_atr=float(bar.range / bar.atr),
                    time=df.index[j],
                ))
                break
    # Compute mitigation for each OB
    closes = df.close.values
    highs = df.high.values
    lows = df.low.values
    for ob in obs:
        for k in range(ob.formed_idx + 2, n):
            if ob.direction == "bull":
                # Mitigated if price closes BELOW the OB low (full violation)
                if closes[k] < ob.low:
                    ob.mitigated_idx = k
                    break
            else:
                if closes[k] > ob.high:
                    ob.mitigated_idx = k
                    break
    return obs


def detect_breakers(obs: list[OrderBlock]) -> list[Breaker]:
    """Breaker = OB that got mitigated → flips role.
    A bull OB that fails (mitigated) becomes a -BRK (short retest zone).
    A bear OB that fails becomes a +BRK (long retest zone).
    """
    breakers: list[Breaker] = []
    for ob in obs:
        if ob.mitigated_idx is None:
            continue
        flip_dir = "bear" if ob.direction == "bull" else "bull"
        breakers.append(Breaker(
            parent_ob=ob,
            flip_idx=ob.mitigated_idx,
            direction=flip_dir,
            high=ob.high,
            low=ob.low,
            time=ob.time,
        ))
    return breakers


def detect_rejection_blocks(df: pd.DataFrame, pivots: list[Pivot]) -> list[RejectionBlock]:
    """RB = candle with dominant wick at a structural pivot.
    Bear RB: shooting star (upper wick >= 55% of range) at/above a recent pivot high.
    Bull RB: hammer (lower wick >= 55% of range) at/below a recent pivot low.
    """
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
        # Bear RB: upper wick dominant + bar.high near a recent BSL
        if upper_frac >= RB_WICK_FRAC:
            recent_highs = [p for p in pivot_highs if (i - SWEEP_LOOKBACK) <= p.idx < i]
            if recent_highs:
                top = max(recent_highs, key=lambda p: p.price)
                if bar.high >= top.price * 0.998:  # within 0.2%
                    rbs.append(RejectionBlock(
                        idx=i,
                        direction="bear",
                        high=float(bar.high),
                        low=float(bar.low),
                        near_pivot_idx=top.idx,
                        time=df.index[i],
                    ))
        if lower_frac >= RB_WICK_FRAC:
            recent_lows = [p for p in pivot_lows if (i - SWEEP_LOOKBACK) <= p.idx < i]
            if recent_lows:
                bot = min(recent_lows, key=lambda p: p.price)
                if bar.low <= bot.price * 1.002:
                    rbs.append(RejectionBlock(
                        idx=i,
                        direction="bull",
                        high=float(bar.high),
                        low=float(bar.low),
                        near_pivot_idx=bot.idx,
                        time=df.index[i],
                    ))
    return rbs


# ─────────────────────────────────────────────────────────
# Pattern matcher
# ─────────────────────────────────────────────────────────

@dataclass
class Setup:
    sweep: Sweep
    bos: BOS
    choch: CHoCH
    zone_kind: str          # "OB" or "BRK" or "RB"
    zone_high: float
    zone_low: float
    side: str               # "long" or "short"
    sl_anchor: float        # the sweep wick extreme
    setup_idx: int          # idx at which setup is FULLY ASSEMBLED (CHoCH bar)
    pre_pivots: list[Pivot] = field(default_factory=list)


def assemble_setups(df: pd.DataFrame, pivots: list[Pivot],
                    sweeps: list[Sweep], bos_events: list[BOS],
                    obs: list[OrderBlock], brks: list[Breaker],
                    rbs: list[RejectionBlock]) -> list[Setup]:
    """Walk sweeps → BOS in same direction → CHoCH → mark zone for retest."""
    setups: list[Setup] = []
    for sweep in sweeps:
        # Sweep BSL → expect bearish setup → look for BOS down within PATTERN_WINDOW
        target_dir = "down" if sweep.direction == "bsl" else "up"
        side = "short" if sweep.direction == "bsl" else "long"
        # First BOS in target direction within window after sweep
        candidate_bos = [b for b in bos_events
                         if b.direction == target_dir
                         and sweep.idx < b.idx <= sweep.idx + PATTERN_WINDOW]
        if not candidate_bos:
            continue
        first_bos = candidate_bos[0]
        choch = detect_choch_after(df, pivots, first_bos)
        if choch is None:
            continue
        if choch.idx > sweep.idx + PATTERN_WINDOW:
            continue
        # Find a candidate retest zone:
        # Short side → -OB (bear OB) formed between sweep and CHoCH
        #             OR -BRK (a bull OB that got mitigated in same window)
        #             OR -RB at the swept BSL
        zones: list[tuple[str, float, float]] = []  # (kind, high, low)
        # OBs
        for ob in obs:
            if ob.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= ob.formed_idx <= choch.idx:
                    zones.append(("OB", ob.high, ob.low))
        # BRKs (flipped role)
        for brk in brks:
            if brk.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx <= brk.flip_idx <= choch.idx + 5:
                    zones.append(("BRK", brk.high, brk.low))
        # RBs
        for rb in rbs:
            if rb.direction == ("bear" if side == "short" else "bull"):
                if sweep.idx - 1 <= rb.idx <= sweep.idx + 1:
                    zones.append(("RB", rb.high, rb.low))
        if not zones:
            continue
        # Pick the zone CLOSEST to the swept extreme (highest priority structure)
        if side == "short":
            best = min(zones, key=lambda z: abs(z[2] - sweep.swept_price))
        else:
            best = min(zones, key=lambda z: abs(z[1] - sweep.swept_price))
        zone_kind, zh, zl = best
        setups.append(Setup(
            sweep=sweep,
            bos=first_bos,
            choch=choch,
            zone_kind=zone_kind,
            zone_high=zh,
            zone_low=zl,
            side=side,
            sl_anchor=sweep.sweep_extreme,
            setup_idx=choch.idx,
        ))
    return setups


# ─────────────────────────────────────────────────────────
# Entry trigger + simulation
# ─────────────────────────────────────────────────────────

def find_entry_and_simulate(df: pd.DataFrame, setup: Setup,
                            exit_cfg: dict) -> Optional[dict]:
    """After CHoCH, walk forward looking for retest into zone + reversal candle."""
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
        # Did this bar touch the zone?
        if setup.side == "short":
            if bar.high < setup.zone_low - tol:
                continue
            if bar.low > setup.zone_high + tol:
                continue
            # Reversal: close < open AND close < prev close
            if bar.close < bar.open and bar.close < prev.close:
                entry_idx = i
                entry_price = float(bar.close)
                break
        else:  # long
            if bar.high < setup.zone_low - tol:
                continue
            if bar.low > setup.zone_high + tol:
                continue
            if bar.close > bar.open and bar.close > prev.close:
                entry_idx = i
                entry_price = float(bar.close)
                break
    if entry_idx is None:
        return None

    # SL = sweep wick extreme + small buffer
    if setup.side == "short":
        sl = setup.sl_anchor + 0.05 * atr_at_choch
    else:
        sl = setup.sl_anchor - 0.05 * atr_at_choch

    R = abs(entry_price - sl)
    if R <= 0:
        return None

    # Cap absurdly wide SLs (sweep was very far from entry → bad setup)
    # We allow up to 3× ATR distance
    if R > 3.0 * atr_at_choch:
        return None

    return simulate_exit(df, entry_idx, entry_price, sl, R, setup, exit_cfg)


def simulate_exit(df: pd.DataFrame, entry_idx: int, entry: float,
                  sl: float, R: float, setup: Setup, cfg: dict) -> Optional[dict]:
    """Walk forward bars, applying exit rules."""
    n = len(df)
    side = setup.side
    bars_held = 0
    max_bars = cfg["max_bars"]
    tp1_R = cfg.get("tp1_R")
    tp2_R = cfg.get("tp2_R")
    fixed_tp_R = cfg.get("fixed_tp_R")
    trail_engage_R = cfg.get("trail_engage_R")
    trail_lock_pct = cfg.get("trail_lock_pct", 0.0)
    tp1_filled_qty = 0.0
    realized_partial = 0.0

    qty = NOTIONAL_USD / entry
    peak_R = 0.0
    trail_sl = sl

    if side == "short":
        tp1_price = entry - (tp1_R * R) if tp1_R else None
        tp2_price = entry - (tp2_R * R) if tp2_R else None
        fixed_tp_price = entry - (fixed_tp_R * R) if fixed_tp_R else None
    else:
        tp1_price = entry + (tp1_R * R) if tp1_R else None
        tp2_price = entry + (tp2_R * R) if tp2_R else None
        fixed_tp_price = entry + (fixed_tp_R * R) if fixed_tp_R else None

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
            cur_R_at_low = (entry - bar.low) / R
            cur_R_at_high = (entry - bar.high) / R  # negative if high > entry
            peak_R = max(peak_R, cur_R_at_low)
            # check SL/trail (high reaches SL)
            if bar.high >= trail_sl:
                exit_idx = j
                exit_price = float(trail_sl)
                exit_reason = "sl_hit" if trail_sl == sl else "trail_sl"
                break
            # TP1 (partial)
            if tp1_price is not None and not tp1_filled_qty:
                if bar.low <= tp1_price:
                    # 50% out
                    tp1_qty = qty * 0.5
                    realized_partial = (entry - tp1_price) * tp1_qty
                    tp1_filled_qty = tp1_qty
                    qty = qty - tp1_qty  # remainder
                    # move SL to breakeven on remainder (textbook flow)
                    trail_sl = entry
            # TP2 (rest)
            if tp2_price is not None and tp1_filled_qty:
                if bar.low <= tp2_price:
                    exit_idx = j
                    exit_price = float(tp2_price)
                    exit_reason = "tp2"
                    break
            # Fixed TP
            if fixed_tp_price is not None:
                if bar.low <= fixed_tp_price:
                    exit_idx = j
                    exit_price = float(fixed_tp_price)
                    exit_reason = "tp_fixed"
                    break
            # Trailing
            if trail_engage_R is not None and peak_R >= trail_engage_R:
                lock_R = peak_R * trail_lock_pct
                new_sl = entry - lock_R * R
                if new_sl < trail_sl:
                    trail_sl = new_sl
        else:  # long
            cur_R_at_high = (bar.high - entry) / R
            peak_R = max(peak_R, cur_R_at_high)
            if bar.low <= trail_sl:
                exit_idx = j
                exit_price = float(trail_sl)
                exit_reason = "sl_hit" if trail_sl == sl else "trail_sl"
                break
            if tp1_price is not None and not tp1_filled_qty:
                if bar.high >= tp1_price:
                    tp1_qty = qty * 0.5
                    realized_partial = (tp1_price - entry) * tp1_qty
                    tp1_filled_qty = tp1_qty
                    qty = qty - tp1_qty
                    trail_sl = entry
            if tp2_price is not None and tp1_filled_qty:
                if bar.high >= tp2_price:
                    exit_idx = j
                    exit_price = float(tp2_price)
                    exit_reason = "tp2"
                    break
            if fixed_tp_price is not None:
                if bar.high >= fixed_tp_price:
                    exit_idx = j
                    exit_price = float(fixed_tp_price)
                    exit_reason = "tp_fixed"
                    break
            if trail_engage_R is not None and peak_R >= trail_engage_R:
                lock_R = peak_R * trail_lock_pct
                new_sl = entry + lock_R * R
                if new_sl > trail_sl:
                    trail_sl = new_sl

    if exit_idx is None:
        # Force-close at last available bar
        last = df.iloc[-1]
        exit_idx = n - 1
        exit_price = float(last.close)
        exit_reason = "data_end"

    # P&L on remaining qty
    if side == "short":
        gross_remainder = (entry - exit_price) * qty
    else:
        gross_remainder = (exit_price - entry) * qty
    gross_total = realized_partial + gross_remainder

    # Fees: entry full, partial exit, final exit
    full_qty = NOTIONAL_USD / entry
    entry_fee = NOTIONAL_USD * TAKER_FEE
    partial_fee = 0.0
    if tp1_filled_qty:
        partial_fee = (tp1_filled_qty * tp1_price) * TAKER_FEE
    final_fee = (qty * exit_price) * TAKER_FEE
    fees = entry_fee + partial_fee + final_fee

    # Funding: 0.01%/8h prorated, applies per bar (1h)
    hours_held = bars_held  # 1h candles
    funding_periods = hours_held / 8.0
    funding_cost = NOTIONAL_USD * FUNDING_PER_8H * funding_periods

    net = gross_total - fees - funding_cost

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
        "peak_R": peak_R,
        "gross_pnl": gross_total,
        "fees": fees,
        "funding": funding_cost,
        "net_pnl": net,
        "tp1_hit": bool(tp1_filled_qty),
        "sweep_idx": setup.sweep.idx,
        "bos_idx": setup.bos.idx,
        "choch_idx": setup.choch.idx,
    }


# ─────────────────────────────────────────────────────────
# Exit configs
# ─────────────────────────────────────────────────────────

EXIT_CONFIGS = {
    "EA": {  # Fixed 2R, sweep-wick SL, 4h time stop
        "fixed_tp_R": 2.0,
        "max_bars": 4,
    },
    "EB": {  # Fixed 3R, 8h
        "fixed_tp_R": 3.0,
        "max_bars": 8,
    },
    "EC": {  # TP1 1R 50% / TP2 2R rest, 8h — textbook
        "tp1_R": 1.0,
        "tp2_R": 2.0,
        "max_bars": 8,
    },
    "ED": {  # Trail 1.5R lock 33%, 12h
        "trail_engage_R": 1.5,
        "trail_lock_pct": 0.33,
        "max_bars": 12,
    },
}


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run_symbol(sym: str) -> dict:
    df = load_candles(sym, "1h")
    df = add_atr(df)
    n_bars = len(df)
    pivots = find_swing_pivots(df)
    sweeps = detect_liquidity_sweeps(df, pivots)
    bos_events = detect_bos(df, pivots)
    obs = detect_order_blocks(df)
    brks = detect_breakers(obs)
    rbs = detect_rejection_blocks(df, pivots)

    # Compute CHoCHs that follow some BOS (for reporting only)
    n_choch = 0
    for b in bos_events:
        if detect_choch_after(df, pivots, b) is not None:
            n_choch += 1

    setups = assemble_setups(df, pivots, sweeps, bos_events, obs, brks, rbs)

    counts = {
        "n_bars": n_bars,
        "n_pivots": len(pivots),
        "n_pivot_highs": sum(1 for p in pivots if p.kind == "high"),
        "n_pivot_lows": sum(1 for p in pivots if p.kind == "low"),
        "n_sweeps": len(sweeps),
        "n_bos": len(bos_events),
        "n_choch": n_choch,
        "n_obs": len(obs),
        "n_brks": len(brks),
        "n_rbs": len(rbs),
        "n_setups": len(setups),
        "setup_zone_breakdown": {},
    }
    for s in setups:
        counts["setup_zone_breakdown"][s.zone_kind] = (
            counts["setup_zone_breakdown"].get(s.zone_kind, 0) + 1)

    by_cell = {}
    for cfg_name, cfg in EXIT_CONFIGS.items():
        trades = []
        for s in setups:
            tr = find_entry_and_simulate(df, s, cfg)
            if tr is not None:
                tr["setup_zone_kind"] = s.zone_kind
                trades.append(tr)
        by_cell[cfg_name] = trades
    return {"sym": sym, "counts": counts, "by_cell": by_cell}


def summarize_cell(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "gross_$": 0.0, "net_$": 0.0,
                "ev_per_trade": 0.0, "avg_hold_bars": 0.0,
                "tp1_hit_rate": 0.0, "by_zone": {}}
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    gross = sum(t["gross_pnl"] for t in trades)
    net = sum(t["net_pnl"] for t in trades)
    avg_hold = sum(t["bars_held"] for t in trades) / n
    tp1 = sum(1 for t in trades if t.get("tp1_hit")) / n
    by_zone = {}
    for t in trades:
        zk = t["setup_zone_kind"]
        rec = by_zone.setdefault(zk, {"n": 0, "net": 0.0, "wins": 0})
        rec["n"] += 1
        rec["net"] += t["net_pnl"]
        if t["net_pnl"] > 0:
            rec["wins"] += 1
    return {
        "n": n,
        "wr": 100.0 * wins / n,
        "gross_$": gross,
        "net_$": net,
        "ev_per_trade": net / n,
        "avg_hold_bars": avg_hold,
        "tp1_hit_rate": tp1,
        "by_zone": by_zone,
    }


def verdict_for(stats: dict) -> str:
    if stats["n"] >= 30 and stats["ev_per_trade"] > 2.0:
        return "SHIP"
    if stats["n"] >= 20 and stats["ev_per_trade"] > 0:
        return "PILOT"
    return "KILL"


def main() -> None:
    all_results = {}
    for sym in SYMBOLS:
        try:
            r = run_symbol(sym)
            all_results[sym] = r
            print(f"\n{sym}: counts={r['counts']}")
        except Exception as e:
            import traceback
            print(f"ERROR {sym}: {e}\n{traceback.format_exc()}", file=sys.stderr)

    # Aggregate per cell across all symbols
    agg_cells = {}
    for cfg_name in EXIT_CONFIGS:
        trades = []
        for sym in SYMBOLS:
            if sym in all_results:
                trades.extend(all_results[sym]["by_cell"].get(cfg_name, []))
        agg_cells[cfg_name] = summarize_cell(trades)

    per_sym_cells = {}
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        per_sym_cells[sym] = {}
        for cfg_name in EXIT_CONFIGS:
            per_sym_cells[sym][cfg_name] = summarize_cell(
                all_results[sym]["by_cell"].get(cfg_name, []))

    # Scoreboard CSV
    rows = []
    for sym in SYMBOLS:
        if sym not in per_sym_cells:
            continue
        for cfg_name in EXIT_CONFIGS:
            s = per_sym_cells[sym][cfg_name]
            rows.append({
                "symbol": sym, "exit_cfg": cfg_name,
                "n": s["n"], "wr": round(s["wr"], 2),
                "gross_$": round(s["gross_$"], 2),
                "net_$": round(s["net_$"], 2),
                "ev_per_trade": round(s["ev_per_trade"], 3),
                "avg_hold_bars": round(s["avg_hold_bars"], 2),
                "tp1_hit_rate": round(s["tp1_hit_rate"], 3),
            })
    for cfg_name in EXIT_CONFIGS:
        s = agg_cells[cfg_name]
        rows.append({
            "symbol": "AGG", "exit_cfg": cfg_name,
            "n": s["n"], "wr": round(s["wr"], 2),
            "gross_$": round(s["gross_$"], 2),
            "net_$": round(s["net_$"], 2),
            "ev_per_trade": round(s["ev_per_trade"], 3),
            "avg_hold_bars": round(s["avg_hold_bars"], 2),
            "tp1_hit_rate": round(s["tp1_hit_rate"], 3),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "scoreboard.csv", index=False)

    # Markdown report
    md = ["# SMC1.5 Fullstack Backtest Report\n"]
    md.append(f"Run on 4 symbols × 1h candles, ~6 months data.\n")
    md.append("## Pattern detection sample sizes\n")
    md.append("| Sym | bars | pivots | sweeps | BOS | CHoCH | OBs | BRKs | RBs | setups |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        c = all_results[sym]["counts"]
        md.append(f"| {sym} | {c['n_bars']} | {c['n_pivots']} | {c['n_sweeps']} | "
                  f"{c['n_bos']} | {c['n_choch']} | {c['n_obs']} | {c['n_brks']} | "
                  f"{c['n_rbs']} | {c['n_setups']} |")
    md.append("\n### Setup zone breakdown")
    md.append("| Sym | OB | BRK | RB |")
    md.append("|---|---|---|---|")
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        zb = all_results[sym]["counts"]["setup_zone_breakdown"]
        md.append(f"| {sym} | {zb.get('OB',0)} | {zb.get('BRK',0)} | {zb.get('RB',0)} |")

    md.append("\n## Per-symbol scoreboard\n")
    for sym in SYMBOLS:
        if sym not in per_sym_cells:
            continue
        md.append(f"### {sym}\n")
        md.append("| cfg | n | WR% | gross$ | net$ | EV/trade | avg_hold | TP1_rate | verdict |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for cfg_name in EXIT_CONFIGS:
            s = per_sym_cells[sym][cfg_name]
            v = verdict_for(s)
            md.append(f"| {cfg_name} | {s['n']} | {s['wr']:.1f} | "
                      f"{s['gross_$']:+.2f} | {s['net_$']:+.2f} | "
                      f"{s['ev_per_trade']:+.3f} | {s['avg_hold_bars']:.1f} | "
                      f"{s['tp1_hit_rate']:.2f} | {v} |")
        md.append("")

    md.append("## Aggregate (all 4 symbols)\n")
    md.append("| cfg | n | WR% | gross$ | net$ | EV/trade | avg_hold | TP1_rate | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    best_cfg = None
    best_ev = -1e9
    for cfg_name in EXIT_CONFIGS:
        s = agg_cells[cfg_name]
        v = verdict_for(s)
        md.append(f"| {cfg_name} | {s['n']} | {s['wr']:.1f} | "
                  f"{s['gross_$']:+.2f} | {s['net_$']:+.2f} | "
                  f"{s['ev_per_trade']:+.3f} | {s['avg_hold_bars']:.1f} | "
                  f"{s['tp1_hit_rate']:.2f} | {v} |")
        if s["ev_per_trade"] > best_ev and s["n"] >= 20:
            best_ev = s["ev_per_trade"]
            best_cfg = cfg_name
    md.append("")

    # Best cell verdict
    if best_cfg:
        s = agg_cells[best_cfg]
        v = verdict_for(s)
        md.append(f"## Best cell: **{best_cfg}** → **{v}**\n")
        md.append(f"- n={s['n']} | WR={s['wr']:.1f}% | net=${s['net_$']:+.2f} | "
                  f"EV/trade=${s['ev_per_trade']:+.3f}\n")
    else:
        md.append("## Best cell: **NONE meet n>=20 threshold** → KILL\n")

    md.append("\n## Exit config legend\n")
    md.append("- **EA**: Fixed 2R TP, sweep-wick SL, 4h time stop")
    md.append("- **EB**: Fixed 3R TP, sweep-wick SL, 8h time stop")
    md.append("- **EC**: TP1=1R (50% out, BE on rest), TP2=2R, 8h time stop (textbook)")
    md.append("- **ED**: Trail at 1.5R, lock 33% MFE, 12h time stop")

    md.append("\n## Cost model")
    md.append(f"- Taker fee: {TAKER_FEE*100:.3f}% × 2 legs")
    md.append(f"- Funding: {FUNDING_PER_8H*100:.3f}% per 8h")
    md.append(f"- Notional: ${NOTIONAL_USD:.0f}/trade")

    (OUT_DIR / "backtest_report.md").write_text("\n".join(md))

    # JSON dump for downstream consumption
    summary_json = {
        "per_symbol": {sym: {"counts": all_results[sym]["counts"]}
                       for sym in SYMBOLS if sym in all_results},
        "agg_cells": {cfg: {k: v for k, v in agg_cells[cfg].items() if k != "by_zone"}
                      for cfg in EXIT_CONFIGS},
        "best_cfg": best_cfg,
        "best_verdict": verdict_for(agg_cells[best_cfg]) if best_cfg else "KILL",
    }
    (OUT_DIR / "backtest_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=str))

    print("\n=== AGG ===")
    for cfg_name in EXIT_CONFIGS:
        s = agg_cells[cfg_name]
        print(f"  {cfg_name}: n={s['n']:3d} WR={s['wr']:5.1f}% net=${s['net_$']:+8.2f} "
              f"EV/trade=${s['ev_per_trade']:+.3f} → {verdict_for(s)}")
    print(f"\nBest: {best_cfg} → {verdict_for(agg_cells[best_cfg]) if best_cfg else 'KILL'}")
    print(f"\nReports: {OUT_DIR}/backtest_report.md, scoreboard.csv, backtest_summary.json")


if __name__ == "__main__":
    main()

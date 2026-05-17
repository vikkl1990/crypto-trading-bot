#!/usr/bin/env python3
"""SMC1.5v2 (WICK-entry variant) paper engine.

Walk-forward-validated entry variant. Replaces the engulfing-trigger market
entry with a LIMIT order at the OB extreme nearest to the approaching price:
  - LONG OB → limit at OB high (top edge)
  - SHORT OB → limit at OB low (bottom edge)

Maker fee (0.024%) instead of taker (0.059%) for entry leg.

Walk-forward result (from smc15_entry_variants_backtest.py, 2025-10-07
→ 2026-04-17, BTC/ETH/SOL/XRP, 1h):
  IS Q1+Q2:  n=36  EV/trade=$+11.34
  OOS Q3:    n=3   EV/trade=$+23.21
  OOS Q4:    n=25  EV/trade=$+6.18
  Total:     n=64  WR=85.9%   net=$+632.4

Verdict: PASS walk-forward (Q3 +205% IS, Q4 +55% IS — within 50% gap rule).

Architecture mirrors scripts/smc15_paper_engine.py — independent state dir
storage/smc15v2_paper/. Polls every 5min via cron.

Pipeline:
  1. Pull 1h candles from Delta REST
  2. Compute SMC primitives (pivots, sweeps, BOS, CHoCH, OBs, BRKs, RBs)
  3. Assemble setups: sweep + BOS opp + CHoCH same dir + zone (OB/BRK/RB)
  4. After CHoCH, place a LIMIT at OB near-edge; track fill on subsequent bars
  5. SL = sweep wick + 0.05× ATR; TP = 3R; time stop = 8h after fill

State: storage/smc15v2_paper/state.json
Trades: storage/smc15v2_paper/trades.jsonl
Summary: storage/smc15v2_paper/latest.md
"""
from __future__ import annotations
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import sys as _sys_telem
_sys_telem.path.insert(0, "/home/opc/crypto-trading-bot")
from execution_v2.engine_telemetry import log_eval

ROOT = Path("/home/opc/crypto-trading-bot")
STATE_DIR = ROOT / "storage" / "smc15v2_paper"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
SUMMARY_FILE = STATE_DIR / "latest.md"
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"

SYMBOLS = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
    "SOL/USDT": "SOLUSD",
    "XRP/USDT": "XRPUSD",
}

# Cost model
TAKER_FEE = 0.00059      # exits (market)
MAKER_FEE = 0.00024      # limit entries that rest as maker
NOTIONAL_USD = 1000.0

# Strategy params (must mirror backtest exactly)
ATR_LEN = 14
SWING_LOOKBACK = 5
SWEEP_LOOKBACK = 20
BOS_LOOKBACK = 30
CHOCH_WINDOW = 15
PATTERN_WINDOW = 30
DISPLACEMENT_ATR_MULT = 1.0
RB_WICK_FRAC = 0.55
ENTRY_TOL_ATR = 0.10

# WICK-variant entry + EB exit
TP_R = 3.0
SL_BUFFER_ATR = 0.05
MAX_HOURS = 8
LIMIT_EXPIRY_HOURS = 6      # cancel pending limit if not filled within 6h
HOURS_BACK_FETCH = 24 * 21


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def fetch_1h(sym_rest: str, hours_back: int = HOURS_BACK_FETCH) -> pd.DataFrame:
    end = int(time.time())
    start = end - hours_back * 3600
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": sym_rest, "resolution": "1h", "start": start, "end": end},
            timeout=15,
        )
        rows = r.json().get("result", [])
    except Exception as e:
        print(f"  REST fetch {sym_rest} failed: {e}")
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["range"] = df.high - df.low
    df["body"] = (df.close - df.open).abs()
    df["upper_wick"] = df.high - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df.low
    return df


def add_atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.DataFrame:
    if len(df) < n + 2:
        return df
    df = df.copy()
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(n).mean()
    return df


def get_current_price(sym_rest: str) -> float | None:
    end = int(time.time())
    start = end - 300
    try:
        r = requests.get(DELTA_REST,
                         params={"symbol": sym_rest, "resolution": "1m",
                                 "start": start, "end": end},
                         timeout=8)
        rows = r.json().get("result", [])
        if not rows:
            return None
        return float(rows[-1]["close"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# SMC primitives (verbatim from backtest / smc15_paper_engine)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Pivot:
    idx: int
    price: float
    kind: str
    time_iso: str


@dataclass
class Sweep:
    idx: int
    swept_pivot_idx: int
    direction: str
    sweep_extreme: float
    swept_price: float
    time_iso: str


@dataclass
class BOS:
    idx: int
    broken_pivot_idx: int
    direction: str
    broken_price: float
    time_iso: str


@dataclass
class CHoCH:
    idx: int
    direction: str
    pivot_idx: int
    time_iso: str


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
    time_iso: str = ""


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
    time_iso: str


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
    time_iso: str


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
            pivots.append(Pivot(i, float(highs[i]), "high", times[i].isoformat()))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            pivots.append(Pivot(i, float(lows[i]), "low", times[i].isoformat()))
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
                                    float(bar.high), highest.price,
                                    df.index[i].isoformat()))
        recent_lows = [p for p in pivot_lows if (i - SWEEP_LOOKBACK) <= p.idx < i]
        if recent_lows:
            lowest = min(recent_lows, key=lambda p: p.price)
            if bar.low < lowest.price and bar.close > lowest.price:
                sweeps.append(Sweep(i, lowest.idx, "ssl",
                                    float(bar.low), lowest.price,
                                    df.index[i].isoformat()))
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
                bos_events.append(BOS(i, p.idx, "up", p.price, df.index[i].isoformat()))
                break
        recent_lows = [p for p in pivot_lows if (i - BOS_LOOKBACK) <= p.idx < i]
        for p in reversed(recent_lows):
            if c < p.price:
                bos_events.append(BOS(i, p.idx, "down", p.price, df.index[i].isoformat()))
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
                return CHoCH(p.idx, "down", p.idx, p.time_iso)
    else:
        pre = [p for p in pivots if p.kind == "low" and p.idx <= bos.idx]
        if not pre:
            return None
        last_low = max(pre, key=lambda p: p.idx)
        post = [p for p in pivots if p.kind == "low"
                and bos.idx < p.idx <= end]
        for p in post:
            if p.price > last_low.price:
                return CHoCH(p.idx, "up", p.idx, p.time_iso)
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
                    time_iso=df.index[j].isoformat(),
                ))
                break
            if (not is_bull) and prev.close > prev.open:
                obs.append(OrderBlock(
                    formed_idx=j, direction="bear",
                    high=float(prev.high), low=float(prev.low),
                    midpoint=float((prev.high + prev.low) / 2),
                    body_high=float(body_high), body_low=float(body_low),
                    displacement_atr=float(bar.range / bar.atr),
                    time_iso=df.index[j].isoformat(),
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


def detect_breakers(obs: list[OrderBlock]) -> list[Breaker]:
    breakers: list[Breaker] = []
    for ob in obs:
        if ob.mitigated_idx is None:
            continue
        flip_dir = "bear" if ob.direction == "bull" else "bull"
        breakers.append(Breaker(
            parent_high=ob.high, parent_low=ob.low,
            flip_idx=ob.mitigated_idx,
            direction=flip_dir, high=ob.high, low=ob.low,
            midpoint=ob.midpoint,
            body_high=ob.body_high, body_low=ob.body_low,
            time_iso=ob.time_iso,
        ))
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
                        near_pivot_idx=top.idx, time_iso=df.index[i].isoformat()))
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
                        near_pivot_idx=bot.idx, time_iso=df.index[i].isoformat()))
    return rbs


# ──────────────────────────────────────────────────────────────────────
# Pattern matcher
# ──────────────────────────────────────────────────────────────────────

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


def assemble_setups(df: pd.DataFrame, pivots, sweeps, bos_events,
                    obs, brks, rbs) -> list[Setup]:
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
        setups.append(Setup(sweep, first_bos, choch, kind, zh, zl, zm,
                            zbh, zbl, side, sweep.sweep_extreme, choch.idx))
    return setups


def detect_setups(df_1h: pd.DataFrame) -> list[Setup]:
    if len(df_1h) < SWING_LOOKBACK + 50:
        return []
    df = add_atr(df_1h)
    if "atr" not in df.columns:
        return []
    pivots = find_swing_pivots(df)
    sweeps = detect_liquidity_sweeps(df, pivots)
    bos_events = detect_bos(df, pivots)
    obs = detect_order_blocks(df)
    brks = detect_breakers(obs)
    rbs = detect_rejection_blocks(df, pivots)
    return assemble_setups(df, pivots, sweeps, bos_events, obs, brks, rbs)


# ──────────────────────────────────────────────────────────────────────
# Pending-limit & trade management
# ──────────────────────────────────────────────────────────────────────

def find_active_pending_limit(df_1h: pd.DataFrame, symbol: str,
                              sym_rest: str) -> Optional[dict]:
    """Identify a setup that has a CHoCH on a recent bar (within
    LIMIT_EXPIRY_HOURS of last close) and where price hasn't yet reached
    the wick limit. Returns the limit definition for queueing.

    Returns None if no fresh pending limit available.
    """
    setups = detect_setups(df_1h)
    if not setups:
        return None
    df = add_atr(df_1h)
    n = len(df)
    latest_idx = n - 1
    atr_at_latest = float(df.iloc[latest_idx].atr) if not pd.isna(df.iloc[latest_idx].atr) else 0.0
    if atr_at_latest <= 0:
        return None

    # Find candidate setups: CHoCH within LIMIT_EXPIRY_HOURS of latest close
    # AND limit price not yet hit by any bar between CHoCH and latest
    fired_limits = []
    for s in setups:
        bars_since_choch = latest_idx - s.setup_idx
        if bars_since_choch < 1 or bars_since_choch > LIMIT_EXPIRY_HOURS:
            continue

        # WICK limit: OB extreme nearest entry side
        if s.side == "long":
            limit_price = s.zone_high
        else:
            limit_price = s.zone_low

        # Compute SL based on sweep + ATR buffer at CHoCH bar
        atr_at_choch = float(df.iloc[s.setup_idx].atr) if not pd.isna(df.iloc[s.setup_idx].atr) else atr_at_latest
        if atr_at_choch <= 0:
            continue
        if s.side == "short":
            sl = s.sl_anchor + SL_BUFFER_ATR * atr_at_choch
        else:
            sl = s.sl_anchor - SL_BUFFER_ATR * atr_at_choch
        R = abs(limit_price - sl)
        if R <= 0 or R > 3.0 * atr_at_choch:
            continue

        # Check if any bar between setup_idx+1 and latest already filled
        already_filled = False
        for k in range(s.setup_idx + 1, latest_idx + 1):
            bar = df.iloc[k]
            if s.side == "long" and bar.low <= limit_price:
                already_filled = True
                break
            if s.side == "short" and bar.high >= limit_price:
                already_filled = True
                break
        if already_filled:
            # Already filled in the past — skip (would be a stale fill)
            continue

        fired_limits.append({
            "side": s.side,
            "limit_price": float(limit_price),
            "sl": float(sl),
            "R": float(R),
            "atr": float(atr_at_choch),
            "zone_kind": s.zone_kind,
            "zone_high": s.zone_high,
            "zone_low": s.zone_low,
            "sweep_extreme": s.sweep.sweep_extreme,
            "sweep_idx": s.sweep.idx,
            "bos_idx": s.bos.idx,
            "choch_idx": s.choch.idx,
            "choch_time": s.choch.time_iso,
            "sweep_time": s.sweep.time_iso,
            "bos_time": s.bos.time_iso,
            "candle_time": df.index[latest_idx].isoformat(),
        })
    if not fired_limits:
        return None
    # Prefer OB > BRK > RB
    fired_limits.sort(key=lambda f: {"OB": 0, "BRK": 1, "RB": 2}.get(f["zone_kind"], 3))
    return fired_limits[0]


def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open_trades": [], "pending_limits": [], "last_eval_candle": {},
            "history_pnl": 0.0, "history_n": 0, "history_wins": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_trade(trade: dict) -> None:
    with TRADES_FILE.open("a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def close_trade(trade: dict, exit_price: float, reason: str) -> dict:
    qty = trade["notional"] / trade["entry"]
    if trade["side"] == "long":
        gross = (exit_price - trade["entry"]) * qty
    else:
        gross = (trade["entry"] - exit_price) * qty
    # Maker on entry (limit); taker on exit (market)
    fees = trade["notional"] * MAKER_FEE + (qty * exit_price) * TAKER_FEE
    net = gross - fees
    trade["exit_price"] = exit_price
    trade["exit_reason"] = reason
    trade["closed_at"] = datetime.now(timezone.utc).isoformat()
    trade["gross_pnl_usd"] = round(gross, 4)
    trade["fees_usd"] = round(fees, 4)
    trade["net_pnl_usd"] = round(net, 4)
    trade["status"] = "closed"
    return trade


def manage_open_trade(trade: dict, sym_rest: str) -> dict:
    px = get_current_price(sym_rest)
    if px is None:
        return trade
    R = abs(trade["entry"] - trade["sl"])
    if R == 0:
        return trade
    if trade["side"] == "long":
        cur_r = (px - trade["entry"]) / R
        tp_price = trade["entry"] + TP_R * R
    else:
        cur_r = (trade["entry"] - px) / R
        tp_price = trade["entry"] - TP_R * R

    trade["peak_mfe_r"] = max(trade.get("peak_mfe_r", 0.0), cur_r)
    trade["last_price"] = px
    trade["last_check"] = datetime.now(timezone.utc).isoformat()

    opened = datetime.fromisoformat(trade["opened_at"])
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    hours_held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0
    if hours_held >= MAX_HOURS:
        return close_trade(trade, px, "time_stop")

    sl_price = trade["sl"]
    if trade["side"] == "long" and px >= tp_price:
        return close_trade(trade, tp_price, "tp_3R")
    if trade["side"] == "short" and px <= tp_price:
        return close_trade(trade, tp_price, "tp_3R")
    if trade["side"] == "long" and px <= sl_price:
        return close_trade(trade, sl_price, "sl_hit")
    if trade["side"] == "short" and px >= sl_price:
        return close_trade(trade, sl_price, "sl_hit")
    return trade


def manage_pending_limit(pending: dict, sym_rest: str,
                         state: dict) -> tuple[Optional[dict], Optional[dict]]:
    """Check if a pending limit should be filled (price reached) or expired.

    Returns (still_pending, new_open_trade) — exactly one is non-None.
    """
    px = get_current_price(sym_rest)
    if px is None:
        return pending, None

    # Expiry check
    placed = datetime.fromisoformat(pending["placed_at"])
    if placed.tzinfo is None:
        placed = placed.replace(tzinfo=timezone.utc)
    hours_pending = (datetime.now(timezone.utc) - placed).total_seconds() / 3600.0
    if hours_pending >= LIMIT_EXPIRY_HOURS:
        return None, None  # both None = expired/cancelled

    pending["last_price"] = px
    pending["last_check"] = datetime.now(timezone.utc).isoformat()

    # Fill check
    if pending["side"] == "long" and px <= pending["limit_price"]:
        # Limit filled
        entry_price = pending["limit_price"]
        trade = {
            "id": f"smc15v2_{int(time.time())}_{pending['symbol'].replace('/','')}",
            "symbol": pending["symbol"],
            "side": pending["side"],
            "entry": entry_price,
            "sl": pending["sl"],
            "atr_at_entry": pending["atr"],
            "notional": NOTIONAL_USD,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "candle_time": pending["candle_time"],
            "zone_kind": pending["zone_kind"],
            "zone_high": pending["zone_high"],
            "zone_low": pending["zone_low"],
            "sweep_extreme": pending["sweep_extreme"],
            "sweep_time": pending["sweep_time"],
            "bos_time": pending["bos_time"],
            "choch_time": pending["choch_time"],
            "limit_price": pending["limit_price"],
            "peak_mfe_r": 0.0,
            "status": "open",
        }
        return None, trade
    if pending["side"] == "short" and px >= pending["limit_price"]:
        entry_price = pending["limit_price"]
        trade = {
            "id": f"smc15v2_{int(time.time())}_{pending['symbol'].replace('/','')}",
            "symbol": pending["symbol"],
            "side": pending["side"],
            "entry": entry_price,
            "sl": pending["sl"],
            "atr_at_entry": pending["atr"],
            "notional": NOTIONAL_USD,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "candle_time": pending["candle_time"],
            "zone_kind": pending["zone_kind"],
            "zone_high": pending["zone_high"],
            "zone_low": pending["zone_low"],
            "sweep_extreme": pending["sweep_extreme"],
            "sweep_time": pending["sweep_time"],
            "bos_time": pending["bos_time"],
            "choch_time": pending["choch_time"],
            "limit_price": pending["limit_price"],
            "peak_mfe_r": 0.0,
            "status": "open",
        }
        return None, trade
    return pending, None


def eval_signals(state: dict) -> None:
    """Look for new pending limits to queue."""
    open_by_sym = {t["symbol"]: 1 for t in state["open_trades"]
                   if t.get("status") != "closed"}
    pending_by_sym = {p["symbol"]: 1 for p in state.get("pending_limits", [])}
    for sym, sym_rest in SYMBOLS.items():
        if open_by_sym.get(sym) or pending_by_sym.get(sym):
            continue
        df_1h = fetch_1h(sym_rest, hours_back=HOURS_BACK_FETCH)
        if df_1h.empty or len(df_1h) < 100:
            print(f"  {sym}: insufficient 1h data ({len(df_1h)} bars)")
            continue
        last_candle_ts = df_1h.index[-1].isoformat()
        if state["last_eval_candle"].get(sym) == last_candle_ts:
            continue
        state["last_eval_candle"][sym] = last_candle_ts
        pl = find_active_pending_limit(df_1h, sym, sym_rest)
        if not pl:
            # SMC15V2_NOSIG_TELEM_FIX (2026-05-01) — log no-signal telemetry.
            # Original patcher's anchor missed because var is 'pl' not 'setup'.
            try:
                log_eval(engine_name="smc15v2", symbol=sym, base_dir=STATE_DIR,
                         signal_fired=False,
                         state={"rejection_reason": "NO_FULL_SMC_CONFLUENCE"})
            except Exception:
                pass
            continue
        pl["symbol"] = sym
        pl["placed_at"] = datetime.now(timezone.utc).isoformat()
        state.setdefault("pending_limits", []).append(pl)
        # Telemetry: log eval state (placed pending limit)
        try:
            log_eval(engine_name="smc15v2", symbol=sym, base_dir=STATE_DIR,
                     signal_fired=True,
                     state={"side": pl["side"], "zone_kind": pl["zone_kind"],
                            "limit_price": pl["limit_price"]})
        except Exception:
            pass
        print(f"  PENDING {sym} {pl['side'].upper()} LIMIT @ {pl['limit_price']:.4f} "
              f"sl={pl['sl']:.4f} zone={pl['zone_kind']} "
              f"[{pl['zone_low']:.4f}, {pl['zone_high']:.4f}]")


def write_summary(state: dict) -> None:
    n = state.get("history_n", 0)
    wins = state.get("history_wins", 0)
    pnl = state.get("history_pnl", 0.0)
    wr = (100 * wins / n) if n else 0
    open_lines = []
    for t in state.get("open_trades", []):
        if t.get("status") == "open":
            cur = t.get("last_price", t["entry"])
            R = abs(t["entry"] - t["sl"])
            cur_r = ((cur - t["entry"]) / R) if t["side"] == "long" else ((t["entry"] - cur) / R)
            open_lines.append(
                f"- **{t['symbol']} {t['side'].upper()}** "
                f"entry=${t['entry']:.4f} cur=${cur:.4f} ({cur_r:+.2f}R, "
                f"peak {t.get('peak_mfe_r',0):+.2f}R) sl=${t['sl']:.4f} "
                f"zone={t['zone_kind']} [{t['zone_low']:.4f}, {t['zone_high']:.4f}] "
                f"sweep@{t['sweep_time'][:16]}"
            )
    pending_lines = []
    for p in state.get("pending_limits", []):
        cur = p.get("last_price", p["limit_price"])
        pending_lines.append(
            f"- **{p['symbol']} {p['side'].upper()}** LIMIT @ ${p['limit_price']:.4f} "
            f"cur=${cur:.4f} sl=${p['sl']:.4f} zone={p['zone_kind']} "
            f"placed@{p['placed_at'][:16]}"
        )
    md = f"""# SMC1.5v2 Paper Engine — WICK-entry variant (limit @ OB extreme)

_Last updated: {datetime.now(timezone.utc).isoformat()}_

## Summary

- **Closed trades:** {n}
- **Wins:** {wins} ({wr:.1f}%)
- **Net P&L:** ${pnl:+.2f}
- **Avg per trade:** ${(pnl/n if n else 0):+.2f}
- **Open positions:** {len([t for t in state.get('open_trades',[]) if t.get('status')=='open'])}
- **Pending limits:** {len(state.get('pending_limits', []))}

## Open positions

{chr(10).join(open_lines) if open_lines else '_None_'}

## Pending limits

{chr(10).join(pending_lines) if pending_lines else '_None_'}

## Walk-forward backtest baseline

Walk-forward pass 2025-10 → 2026-04 (6.4 months, 4 sym 1h):
  - IS Q1+Q2:  n=36  WR=86.1%  EV/trade=$+11.34
  - OOS Q3:    n=3   WR=100%   EV/trade=$+23.21
  - OOS Q4:    n=25  WR=84.0%  EV/trade=$+6.18
  - Total:     n=64  WR=85.9%  net=$+632.4

Verdict: PASS walk-forward. Q4 EV beats SMC1.5 baseline (+$5.15) by ~20%.

## Strategy params (must match backtest)

- Pivot lookback: {SWING_LOOKBACK}-bar fractal
- Sweep lookback: {SWEEP_LOOKBACK} bars
- BOS lookback: {BOS_LOOKBACK} bars
- CHoCH window: {CHOCH_WINDOW} bars after BOS
- Pattern window: {PATTERN_WINDOW} bars max from sweep → entry
- Displacement: candle range > {DISPLACEMENT_ATR_MULT}× ATR_14, body > 55% range
- RB wick fraction: {RB_WICK_FRAC*100:.0f}%
- Entry tol: {ENTRY_TOL_ATR}× ATR
- **Entry: LIMIT at OB nearest extreme (high for long, low for short)**
- Limit expiry: {LIMIT_EXPIRY_HOURS}h
- SL: sweep wick + {SL_BUFFER_ATR}× ATR buffer
- TP: {TP_R}R fixed
- Max R-distance: 3× ATR
- Time stop: {MAX_HOURS}h after fill
- Cost: maker 0.024% (entry) + taker 0.059% (exit) = 0.083% RT vs 0.118% (taker-only)
"""
    SUMMARY_FILE.write_text(md)


def run() -> None:
    state = load_state()
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] SMC1.5v2 WICK tick")

    # 1) Manage open trades
    still_open = []
    for trade in state.get("open_trades", []):
        if trade.get("status") == "closed":
            continue
        sym_rest = SYMBOLS.get(trade["symbol"])
        if not sym_rest:
            still_open.append(trade)
            continue
        updated = manage_open_trade(trade, sym_rest)
        if updated.get("status") == "closed":
            append_trade(updated)
            state["history_pnl"] = round(state.get("history_pnl", 0.0)
                                         + updated["net_pnl_usd"], 4)
            state["history_n"] = state.get("history_n", 0) + 1
            if updated["net_pnl_usd"] > 0:
                state["history_wins"] = state.get("history_wins", 0) + 1
            print(f"  CLOSED {updated['symbol']} {updated['side']} "
                  f"net=${updated['net_pnl_usd']:+.2f} "
                  f"reason={updated['exit_reason']}")
        else:
            still_open.append(updated)
    state["open_trades"] = still_open

    # 2) Manage pending limits (fill or expire)
    still_pending = []
    for pending in state.get("pending_limits", []):
        sym_rest = SYMBOLS.get(pending["symbol"])
        if not sym_rest:
            still_pending.append(pending)
            continue
        updated_pending, new_trade = manage_pending_limit(pending, sym_rest, state)
        if new_trade is not None:
            state["open_trades"].append(new_trade)
            print(f"  FILLED {new_trade['symbol']} {new_trade['side'].upper()} "
                  f"@ {new_trade['entry']:.4f} sl={new_trade['sl']:.4f} "
                  f"zone={new_trade['zone_kind']}")
        elif updated_pending is not None:
            still_pending.append(updated_pending)
        else:
            print(f"  EXPIRED pending limit {pending['symbol']} {pending['side'].upper()}")
    state["pending_limits"] = still_pending

    # 3) Eval signals → queue new pending limits
    eval_signals(state)

    save_state(state)
    write_summary(state)
    print(f"  open={len(state['open_trades'])}  pending={len(state.get('pending_limits',[]))}  "
          f"history n={state.get('history_n',0)} net=${state.get('history_pnl',0):+.2f}")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Walk-forward backtest: Liquidity Sweep Fade with Delta India Scalper-Offer + Maker Entry.

Re-tests the previously-killed liquidity_sweep scalp class under corrected fee math:
  - Variant A: full RT taker (old kill verdict math)
  - Variant B: taker entry + scalper-offer waiver on exit leg
  - Variant C: maker entry (post_only LIMIT) + scalper-offer waiver on exit leg

Strategy (reclaim-fade direction — fades the sweep, not joins it):
  LONG:
    - Equal-low cluster (2+ swing lows within tolerance × ATR) over last 50 bars
    - Sweep: current low < eq_low_level
    - Reclaim: current close > eq_low_level
    - Volume on reclaim: rel_vol > vol_thr
    - Body ratio: body / range >= body_thr
    - Sweep depth: (eq_low_level - low) / atr >= depth_thr
    - Maker entry: LIMIT BUY at eq_low_level + 0.5×ATR  (pulls back into reclaim zone)
  SHORT mirror.

Exit:
  - TP1 (50% size): midpoint of swept range
  - TP2 (50% size): opposite liquidity pool (next eq_high if LONG, eq_low if SHORT;
                    fallback 2.5R if no opposite pool)
  - SL: 0.5×ATR beyond the sweep wick
  - Hard time stop: 28 min for BTC/ETH (30-min scalper window minus 2-min margin)
                    13 min for SOL/XRP (15-min window minus 2-min margin)

Walk-forward quarters:
  Q1 in-sample : 2025-10-01 → 2025-12-01  (5m parquets start Nov 5; 15m has full Oct)
  Q2 in-sample : 2025-12-01 → 2026-02-01
  Q3 OOS       : 2026-02-01 → 2026-03-01
  Q4 OOS       : 2026-03-01 → 2026-05-01

Pass criteria per cell:
  - OOS Q4 EV/trade ≥ $0.10
  - Both Q3 and Q4 same sign as IS (positive)
  - Q4 gap ≤ 50% (i.e. OOS at least 50% of IS)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_v2.fee_model import FeeModel  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
CACHE_DIR = ROOT / "storage" / "candle_cache"
OUT_DIR = ROOT / "storage" / "scalper_liq_sweep"

# Symbol class → (parquet_symbol, scalper_window_min, time_stop_min)
SYMBOL_INFO = {
    # Big — BTC/ETH 30-min scalper window → 28-min time stop
    "BTC/USDT": {"parq": "BTC_USDT", "ts_min": 28, "fee_sym": "BTC/USDT"},
    "ETH/USDT": {"parq": "ETH_USDT", "ts_min": 28, "fee_sym": "ETH/USDT"},
    # Alts — fee_model only waives BTC/ETH today (ENABLE_OTHER_SYMBOLS_SCALPER=False).
    # We still backtest with the 13-min time stop AND we explicitly run a
    # "what-if alts get the offer" pass by setting force_no_scalper=False on
    # the BTC/ETH-style symbol slot via a special override — see Variant C-alt.
    "SOL/USDT": {"parq": "SOL_USDT", "ts_min": 13, "fee_sym": "SOL/USDT"},
    "XRP/USDT": {"parq": "XRP_USDT", "ts_min": 13, "fee_sym": "XRP/USDT"},
}

QUARTERS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}

# Param grid (from spec)
TF_GRID = ["5m", "15m"]
DEPTH_GRID = [0.10, 0.20, 0.30]      # sweep depth in ATR units
BODY_GRID = [0.50, 0.55, 0.60]       # body / range minimum
VOL_GRID = [1.0, 1.2, 1.5]           # rel_vol threshold
EQ_TOL_GRID = [0.20, 0.30]           # equal-level tolerance in ATR units (0.25 in prod)

# Fixed
ATR_PERIOD = 14
LOOKBACK_BARS = 50          # how far back to look for equal-level cluster
NOTIONAL = 1000.0
LIMIT_EXPIRY_BARS = {"5m": 5, "15m": 2}  # bars to wait for limit fill
SL_BUFFER_ATR = 0.5         # SL = sweep wick ± 0.5 ATR
MAKER_OFFSET_ATR = 0.5      # maker LIMIT placed 0.5 ATR inside reclaim zone
TP2_FALLBACK_RR = 2.5       # if no opposite pool, fallback TP2 distance
TP1_FRACTION = 0.5          # 50% off at midpoint

FEE_MODEL = FeeModel()

# ──────────────────────────────────────────────────────────────────────
# Loaders / indicators
# ──────────────────────────────────────────────────────────────────────
def load_parquet(sym_info: dict, tf: str) -> pd.DataFrame:
    p = CACHE_DIR / f"{sym_info['parq']}_{tf}.parquet"
    df = pd.read_parquet(p)
    df = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    df = df.sort_index()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = out["high"]; low = out["low"]; close = out["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    out["vol_ma"] = out["volume"].rolling(20, min_periods=20).mean()
    out["rel_vol"] = out["volume"] / out["vol_ma"].replace(0, np.nan)
    return out


# ──────────────────────────────────────────────────────────────────────
# Signal detection
# ──────────────────────────────────────────────────────────────────────
def detect_signal(
    df: pd.DataFrame,
    i: int,
    *,
    eq_tol_atr: float,
    depth_thr: float,
    body_thr: float,
    vol_thr: float,
) -> Optional[dict]:
    """Detect liquidity-sweep-fade signal at row i (last closed bar)."""
    if i < LOOKBACK_BARS + ATR_PERIOD + 5:
        return None
    row = df.iloc[i]
    atr = row["atr"]
    if pd.isna(atr) or atr <= 0:
        return None

    high = row["high"]; low = row["low"]
    open_ = row["open"]; close = row["close"]
    rel_vol = row["rel_vol"]
    if pd.isna(rel_vol):
        return None

    body = abs(close - open_)
    rng = high - low if high > low else atr * 0.01
    body_ratio = body / rng
    if body_ratio < body_thr:
        return None
    if rel_vol < vol_thr:
        return None

    # Window of past LOOKBACK_BARS (excluding current bar)
    win_start = max(0, i - LOOKBACK_BARS)
    if i - win_start < 10:
        return None

    tolerance = atr * eq_tol_atr
    # Use sliced numpy views directly (avoid pandas .iloc slicing overhead)
    highs = df["high"].values[win_start:i]
    lows = df["low"].values[win_start:i]
    n_lows = len(lows)
    n_highs = len(highs)

    # ── Equal lows: scan from most recent 30 bars backward ──
    eq_low_level = 0.0
    eq_low_count = 0
    j_min = max(0, n_lows - 30)
    for j in range(n_lows - 1, j_min - 1, -1):
        l_val = lows[j]
        # vectorized count of touches
        touches = int(np.sum(np.abs(lows - l_val) < tolerance)) - 1  # exclude self
        if touches >= 1 and (eq_low_level == 0 or l_val < eq_low_level):
            eq_low_level = float(l_val)
            eq_low_count = touches + 1
            break

    # ── Equal highs ──
    eq_high_level = 0.0
    eq_high_count = 0
    j_min = max(0, n_highs - 30)
    for j in range(n_highs - 1, j_min - 1, -1):
        h = highs[j]
        touches = int(np.sum(np.abs(highs - h) < tolerance)) - 1
        if touches >= 1 and h > eq_high_level:
            eq_high_level = float(h)
            eq_high_count = touches + 1
            break

    side = None
    sweep_level = 0.0
    swept_low = 0.0
    swept_high = 0.0

    # LONG fade: sweep below eq_low and reclaim above
    if eq_low_level > 0 and low < eq_low_level and close > eq_low_level:
        sweep_depth = (eq_low_level - low) / atr
        if sweep_depth >= depth_thr:
            side = "long"
            sweep_level = eq_low_level
            swept_low = low
            swept_high = eq_low_level   # the level swept upward back through

    # SHORT fade: sweep above eq_high and reclaim below
    if side is None and eq_high_level > 0 and high > eq_high_level and close < eq_high_level:
        sweep_depth = (high - eq_high_level) / atr
        if sweep_depth >= depth_thr:
            side = "short"
            sweep_level = eq_high_level
            swept_low = eq_high_level
            swept_high = high

    if side is None:
        return None

    # Maker entry: LIMIT placed inside reclaim zone (above swept low for LONG)
    # TP2 = opposite liquidity pool (eq_high for LONG / eq_low for SHORT). If no
    # opposite pool, fall back to TP2 = entry ± TP2_FALLBACK_RR × risk.
    # TP1 = midpoint between entry and TP2 (half the reward leg). The original
    # "midpoint of swept range" reading would have TP1 below the LONG entry,
    # which is degenerate; using midpoint(entry,TP2) preserves the intent of
    # taking 50% off at ~half-way to target.
    if side == "long":
        limit_price = eq_low_level + MAKER_OFFSET_ATR * atr
        taker_entry_price = float(close)
        sl = swept_low - SL_BUFFER_ATR * atr
        if eq_high_level > 0 and eq_high_level > limit_price:
            tp2 = eq_high_level
        else:
            risk = limit_price - sl
            tp2 = limit_price + TP2_FALLBACK_RR * risk
        tp1 = (limit_price + tp2) / 2.0
    else:
        limit_price = eq_high_level - MAKER_OFFSET_ATR * atr
        taker_entry_price = float(close)
        sl = swept_high + SL_BUFFER_ATR * atr
        if eq_low_level > 0 and eq_low_level < limit_price:
            tp2 = eq_low_level
        else:
            risk = sl - limit_price
            tp2 = limit_price - TP2_FALLBACK_RR * risk
        tp1 = (limit_price + tp2) / 2.0

    return {
        "i": i,
        "ts": df.index[i],
        "side": side,
        "limit_price": float(limit_price),
        "taker_entry_price": float(taker_entry_price),
        "sl": float(sl),
        "tp1_maker": float(tp1),  # midpoint(limit_price, tp2)
        "tp2": float(tp2),
        "atr": float(atr),
        "sweep_level": float(sweep_level),
        "swept_low": float(swept_low),
        "swept_high": float(swept_high),
        "eq_low_count": eq_low_count,
        "eq_high_count": eq_high_count,
    }


# ──────────────────────────────────────────────────────────────────────
# Trade simulation
# ──────────────────────────────────────────────────────────────────────
def simulate_taker_trade(
    df: pd.DataFrame,
    sig: dict,
    *,
    time_stop_min: int,
    tf_min: int,
) -> Optional[dict]:
    """Variant A/B: taker entry at signal close. Walk subsequent bars to find exit."""
    entry = sig["taker_entry_price"]
    entry_idx = sig["i"]
    side = sig["side"]
    sl = sig["sl"]
    tp2 = sig["tp2"]
    # TP1 = midpoint between entry and TP2 (taker entry is different from maker)
    tp1 = (entry + tp2) / 2.0
    bars_max = max(1, time_stop_min // tf_min)
    if bars_max <= 0:
        return None

    qty = NOTIONAL / entry
    tp1_filled = False
    realized_pnl = 0.0
    exit_reason = None
    exit_price = None
    holding_bars = 0
    end_idx = min(entry_idx + bars_max, len(df) - 1)

    for j in range(entry_idx + 1, end_idx + 1):
        b = df.iloc[j]
        bh = b["high"]; bl = b["low"]
        holding_bars = j - entry_idx

        # Order: SL > TP1 > TP2 priority — for LONG, if both SL and TP1 in same bar,
        # we conservatively take the worse case (SL first)
        if side == "long":
            hit_sl = bl <= sl
            hit_tp1 = (not tp1_filled) and bh >= tp1
            hit_tp2 = bh >= tp2
            # conservative SL-first
            if hit_sl:
                # close remaining position at SL
                remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
                realized_pnl += (sl - entry) * qty * remaining
                exit_reason = "sl_hit" if not tp1_filled else "sl_after_tp1"
                exit_price = sl
                break
            if hit_tp1 and not tp1_filled:
                realized_pnl += (tp1 - entry) * qty * TP1_FRACTION
                tp1_filled = True
                # move SL to entry (BE) for runner
                sl = entry
            if hit_tp2:
                remaining = 1.0 - TP1_FRACTION if tp1_filled else 1.0
                realized_pnl += (tp2 - entry) * qty * remaining
                exit_reason = "tp2_pool"
                exit_price = tp2
                break
        else:
            hit_sl = bh >= sl
            hit_tp1 = (not tp1_filled) and bl <= tp1
            hit_tp2 = bl <= tp2
            if hit_sl:
                remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
                realized_pnl += (entry - sl) * qty * remaining
                exit_reason = "sl_hit" if not tp1_filled else "sl_after_tp1"
                exit_price = sl
                break
            if hit_tp1 and not tp1_filled:
                realized_pnl += (entry - tp1) * qty * TP1_FRACTION
                tp1_filled = True
                sl = entry
            if hit_tp2:
                remaining = 1.0 - TP1_FRACTION if tp1_filled else 1.0
                realized_pnl += (entry - tp2) * qty * remaining
                exit_reason = "tp2_pool"
                exit_price = tp2
                break

    if exit_reason is None:
        # time stop at end
        exit_idx = min(entry_idx + bars_max, len(df) - 1)
        exit_price = float(df.iloc[exit_idx]["close"])
        if side == "long":
            remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
            realized_pnl += (exit_price - entry) * qty * remaining
        else:
            remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
            realized_pnl += (entry - exit_price) * qty * remaining
        exit_reason = "time_stop"
        holding_bars = exit_idx - entry_idx

    holding_sec = holding_bars * tf_min * 60
    return {
        "entry": float(entry),
        "exit": float(exit_price) if exit_price is not None else None,
        "side": side,
        "gross_pnl": float(realized_pnl),
        "tp1_filled": tp1_filled,
        "exit_reason": exit_reason,
        "holding_sec": holding_sec,
        "holding_bars": holding_bars,
    }


def simulate_maker_trade(
    df: pd.DataFrame,
    sig: dict,
    *,
    time_stop_min: int,
    tf_min: int,
    limit_expiry_bars: int,
) -> Optional[dict]:
    """Variant C: maker LIMIT entry. Wait for fill (optimistic touch).

    For LONG: limit at sig['limit_price']; fills if any subsequent bar's low <= limit_price.
    For SHORT: fills if high >= limit_price.
    Returns None if limit doesn't fill within `limit_expiry_bars`.
    """
    side = sig["side"]
    limit = sig["limit_price"]
    sig_idx = sig["i"]
    fill_idx = None
    for k in range(1, limit_expiry_bars + 1):
        idx = sig_idx + k
        if idx >= len(df):
            return None
        b = df.iloc[idx]
        if side == "long" and b["low"] <= limit:
            fill_idx = idx
            break
        if side == "short" and b["high"] >= limit:
            fill_idx = idx
            break
    if fill_idx is None:
        return None

    entry = limit  # filled at limit price (optimistic touch)
    sl = sig["sl"]
    tp2 = sig["tp2"]
    tp1 = sig["tp1_maker"]   # already midpoint(limit_price, tp2)
    bars_max = max(1, time_stop_min // tf_min)
    qty = NOTIONAL / entry

    tp1_filled = False
    realized_pnl = 0.0
    exit_reason = None
    exit_price = None
    holding_bars = 0
    end_idx = min(fill_idx + bars_max, len(df) - 1)

    for j in range(fill_idx + 1, end_idx + 1):
        b = df.iloc[j]
        bh = b["high"]; bl = b["low"]
        holding_bars = j - fill_idx
        if side == "long":
            hit_sl = bl <= sl
            hit_tp1 = (not tp1_filled) and bh >= tp1
            hit_tp2 = bh >= tp2
            if hit_sl:
                remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
                realized_pnl += (sl - entry) * qty * remaining
                exit_reason = "sl_hit" if not tp1_filled else "sl_after_tp1"
                exit_price = sl
                break
            if hit_tp1 and not tp1_filled:
                realized_pnl += (tp1 - entry) * qty * TP1_FRACTION
                tp1_filled = True
                sl = entry
            if hit_tp2:
                remaining = 1.0 - TP1_FRACTION if tp1_filled else 1.0
                realized_pnl += (tp2 - entry) * qty * remaining
                exit_reason = "tp2_pool"
                exit_price = tp2
                break
        else:
            hit_sl = bh >= sl
            hit_tp1 = (not tp1_filled) and bl <= tp1
            hit_tp2 = bl <= tp2
            if hit_sl:
                remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
                realized_pnl += (entry - sl) * qty * remaining
                exit_reason = "sl_hit" if not tp1_filled else "sl_after_tp1"
                exit_price = sl
                break
            if hit_tp1 and not tp1_filled:
                realized_pnl += (entry - tp1) * qty * TP1_FRACTION
                tp1_filled = True
                sl = entry
            if hit_tp2:
                remaining = 1.0 - TP1_FRACTION if tp1_filled else 1.0
                realized_pnl += (entry - tp2) * qty * remaining
                exit_reason = "tp2_pool"
                exit_price = tp2
                break

    if exit_reason is None:
        exit_idx = min(fill_idx + bars_max, len(df) - 1)
        exit_price = float(df.iloc[exit_idx]["close"])
        if side == "long":
            remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
            realized_pnl += (exit_price - entry) * qty * remaining
        else:
            remaining = 1.0 - (TP1_FRACTION if tp1_filled else 0.0)
            realized_pnl += (entry - exit_price) * qty * remaining
        exit_reason = "time_stop"
        holding_bars = exit_idx - fill_idx

    holding_sec = holding_bars * tf_min * 60
    fill_lag_bars = fill_idx - sig_idx
    return {
        "entry": float(entry),
        "exit": float(exit_price) if exit_price is not None else None,
        "side": side,
        "gross_pnl": float(realized_pnl),
        "tp1_filled": tp1_filled,
        "exit_reason": exit_reason,
        "holding_sec": holding_sec,
        "holding_bars": holding_bars,
        "fill_lag_bars": fill_lag_bars,
        "fill_idx": fill_idx,
    }


# ──────────────────────────────────────────────────────────────────────
# Quarter slicing
# ──────────────────────────────────────────────────────────────────────
def quarter_for_ts(ts: pd.Timestamp) -> Optional[str]:
    for name, (start, end) in QUARTERS.items():
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        if s <= ts < e:
            return name
    return None


# ──────────────────────────────────────────────────────────────────────
# Backtest cell
# ──────────────────────────────────────────────────────────────────────
def backtest_cell(
    df: pd.DataFrame,
    symbol: str,
    sym_info: dict,
    tf: str,
    *,
    eq_tol_atr: float,
    depth_thr: float,
    body_thr: float,
    vol_thr: float,
) -> List[dict]:
    """Walk df, detect signals, simulate all 3 variants per signal. Return list of trade dicts."""
    tf_min = 5 if tf == "5m" else 15
    time_stop_min = sym_info["ts_min"]
    fee_sym = sym_info["fee_sym"]
    limit_expiry_bars = LIMIT_EXPIRY_BARS[tf]

    trades: List[dict] = []
    last_signal_idx = -1000  # cooldown: don't fire while previous trade may still be active
    cooldown_bars = max(1, time_stop_min // tf_min) + limit_expiry_bars

    for i in range(LOOKBACK_BARS + ATR_PERIOD + 5, len(df)):
        if i - last_signal_idx < cooldown_bars:
            continue
        sig = detect_signal(
            df, i,
            eq_tol_atr=eq_tol_atr,
            depth_thr=depth_thr,
            body_thr=body_thr,
            vol_thr=vol_thr,
        )
        if sig is None:
            continue
        ts = sig["ts"]
        q = quarter_for_ts(ts)
        if q is None:
            continue
        last_signal_idx = i

        # Variant A: taker entry + full taker exit (no scalper waiver)
        ta = simulate_taker_trade(df, sig, time_stop_min=time_stop_min, tf_min=tf_min)
        if ta is None:
            continue
        feeA = FEE_MODEL.round_trip_for_trade(
            exchange="delta", entry_type="taker", exit_type="taker",
            notional_usd=NOTIONAL, symbol=fee_sym,
            holding_sec=ta["holding_sec"], force_no_scalper=True,
        )
        netA = ta["gross_pnl"] - feeA["fee_usd"]

        # Variant B: taker entry + scalper-offer waiver on exit
        feeB = FEE_MODEL.round_trip_for_trade(
            exchange="delta", entry_type="taker", exit_type="taker",
            notional_usd=NOTIONAL, symbol=fee_sym,
            holding_sec=ta["holding_sec"], force_no_scalper=False,
        )
        netB = ta["gross_pnl"] - feeB["fee_usd"]

        # Variant C: maker entry (post_only LIMIT) + scalper-offer waiver on exit
        tc = simulate_maker_trade(
            df, sig,
            time_stop_min=time_stop_min, tf_min=tf_min,
            limit_expiry_bars=limit_expiry_bars,
        )
        if tc is not None:
            feeC = FEE_MODEL.round_trip_for_trade(
                exchange="delta", entry_type="maker", exit_type="taker",
                notional_usd=NOTIONAL, symbol=fee_sym,
                holding_sec=tc["holding_sec"], force_no_scalper=False,
            )
            netC = tc["gross_pnl"] - feeC["fee_usd"]
        else:
            feeC = None
            netC = None

        trades.append({
            "symbol": symbol, "tf": tf, "ts": ts.isoformat(),
            "quarter": q, "side": sig["side"],
            "limit_price": sig["limit_price"],
            "taker_entry": sig["taker_entry_price"],
            "sl": sig["sl"], "tp1_maker": sig["tp1_maker"], "tp2": sig["tp2"],
            "atr": sig["atr"],
            "A_net$": netA, "A_gross$": ta["gross_pnl"],
            "A_fee$": feeA["fee_usd"], "A_exit_reason": ta["exit_reason"],
            "A_holding_sec": ta["holding_sec"], "A_holding_bars": ta["holding_bars"],
            "A_tp1_filled": ta["tp1_filled"], "A_scalper_applied": False,
            "B_net$": netB, "B_fee$": feeB["fee_usd"],
            "B_scalper_applied": bool(feeB["scalper_applied"]),
            "C_filled": tc is not None,
            "C_net$": netC,
            "C_gross$": tc["gross_pnl"] if tc else None,
            "C_fee$": feeC["fee_usd"] if feeC else None,
            "C_exit_reason": tc["exit_reason"] if tc else None,
            "C_holding_sec": tc["holding_sec"] if tc else None,
            "C_holding_bars": tc["holding_bars"] if tc else None,
            "C_fill_lag_bars": tc["fill_lag_bars"] if tc else None,
            "C_tp1_filled": tc["tp1_filled"] if tc else None,
            "C_scalper_applied": bool(feeC["scalper_applied"]) if feeC else False,
        })
    return trades


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────
def agg_variant(trades: List[dict], variant: str) -> Dict[str, Dict[str, float]]:
    """Return {quarter: {n, ev, wr, scalper_pct, net, tp_pct, sl_pct, time_pct, ...}}."""
    pnl_key = f"{variant}_net$"
    reason_key = f"{variant}_exit_reason" if variant in ("A", "C") else "A_exit_reason"
    sa_key = f"{variant}_scalper_applied"
    qbuckets: Dict[str, List[dict]] = {q: [] for q in QUARTERS}
    qbuckets["IS"] = []  # Q1+Q2
    qbuckets["ALL"] = []
    for t in trades:
        if variant == "C" and not t.get("C_filled"):
            continue  # variant C only counts trades where the limit filled
        if t.get(pnl_key) is None:
            continue
        q = t.get("quarter")
        if q in QUARTERS:
            qbuckets[q].append(t)
        if q in ("Q1", "Q2"):
            qbuckets["IS"].append(t)
        qbuckets["ALL"].append(t)

    out: Dict[str, Dict[str, float]] = {}
    for q, ts in qbuckets.items():
        n = len(ts)
        if n == 0:
            out[q] = {"n": 0, "ev$": 0.0, "wr": 0.0, "net$": 0.0,
                      "scalper_pct": 0.0, "tp1_rate": 0.0,
                      "tp_pct": 0.0, "sl_pct": 0.0, "time_pct": 0.0,
                      "fill_rate": 0.0}
            continue
        nets = [t[pnl_key] for t in ts]
        wins = sum(1 for x in nets if x > 0)
        sa = sum(1 for t in ts if t.get(sa_key))
        reasons = [t.get(reason_key, "") or "" for t in ts]
        # Count exit-reason buckets exclusively. "sl_after_tp1" still goes
        # to SL (final exit was SL, even if half size already booked at TP1).
        tp_n = sum(1 for r in reasons if r == "tp2_pool")
        sl_n = sum(1 for r in reasons if r.startswith("sl"))
        time_n = sum(1 for r in reasons if "time" in r)
        tp1_filled = sum(1 for t in ts if t.get(f"{variant}_tp1_filled" if variant in ("A","C") else "A_tp1_filled"))
        out[q] = {
            "n": n,
            "ev$": float(np.mean(nets)),
            "wr": wins / n,
            "net$": float(np.sum(nets)),
            "scalper_pct": sa / n,
            "tp1_rate": tp1_filled / n,
            "tp_pct": tp_n / n,
            "sl_pct": sl_n / n,
            "time_pct": time_n / n,
        }
    return out


def cell_summary(
    cell: str,
    params: dict,
    trades: List[dict],
) -> Dict[str, dict]:
    """Compute per-variant aggregates + walk-forward verdict per cell."""
    out: Dict[str, dict] = {"cell": cell, "params": params}
    for variant in ("A", "B", "C"):
        ag = agg_variant(trades, variant)
        IS = ag["IS"]; q3 = ag["Q3"]; q4 = ag["Q4"]; allq = ag["ALL"]
        is_ev = IS["ev$"] if IS["n"] else 0.0
        q3_ev = q3["ev$"] if q3["n"] else 0.0
        q4_ev = q4["ev$"] if q4["n"] else 0.0
        gap_q3 = (q3_ev / is_ev - 1.0) * 100 if abs(is_ev) > 1e-9 else float("inf")
        gap_q4 = (q4_ev / is_ev - 1.0) * 100 if abs(is_ev) > 1e-9 else float("inf")

        # Walk-forward verdict
        verdict = "KILL"
        if IS["n"] >= 25 and q4["n"] >= 10:
            same_sign = (is_ev > 0 and q3_ev > 0 and q4_ev > 0)
            ship_ok = q4_ev >= 0.10
            ratio_ok = (q4_ev / is_ev) >= 0.5 if is_ev > 0 else False
            if same_sign and ship_ok and ratio_ok:
                verdict = "PASS"
            elif (q4_ev > 0 and is_ev > 0) or (q3_ev > 0 and is_ev > 0):
                verdict = "HOLD"
        out[variant] = {
            "IS": IS, "Q3": q3, "Q4": q4, "ALL": allq,
            "Q3_gap_pct": gap_q3, "Q4_gap_pct": gap_q4,
            "verdict": verdict,
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid for fast iteration")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Param grid
    if args.quick:
        depth_grid = [0.20]
        body_grid = [0.55]
        vol_grid = [1.2]
        eq_tol_grid = [0.25]
        tf_grid = ["5m"]
    else:
        depth_grid = DEPTH_GRID
        body_grid = BODY_GRID
        vol_grid = VOL_GRID
        eq_tol_grid = EQ_TOL_GRID
        tf_grid = TF_GRID

    # Pre-load and add indicators per (symbol, tf)
    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for sym in symbols:
        info = SYMBOL_INFO[sym]
        for tf in tf_grid:
            df = load_parquet(info, tf)
            df = add_indicators(df)
            df_cache[(sym, tf)] = df
            print(f"  loaded {sym} {tf}: n={len(df)} from={df.index[0]} to={df.index[-1]}")

    # Walk grid × symbol × tf
    rows = []
    class_rows = []
    cells_seen = 0
    total_cells = len(tf_grid) * len(eq_tol_grid) * len(depth_grid) * len(body_grid) * len(vol_grid)
    print(f"\n  total param cells: {total_cells}, symbols: {len(symbols)} -> "
          f"{total_cells * len(symbols)} (sym,cell) combos")

    # Run by symbol × cell, cache trades for re-use in class aggregation
    for tf in tf_grid:
        for eq_tol in eq_tol_grid:
            for depth in depth_grid:
                for body in body_grid:
                    for vol in vol_grid:
                        cell_id = (f"tf={tf}|eq_tol={eq_tol}|depth={depth}|"
                                   f"body={body}|vol={vol}")
                        params = {
                            "tf": tf, "eq_tol_atr": eq_tol,
                            "depth_thr": depth, "body_thr": body,
                            "vol_thr": vol,
                        }
                        cells_seen += 1

                        cell_trades_by_sym: Dict[str, List[dict]] = {}
                        for sym in symbols:
                            info = SYMBOL_INFO[sym]
                            df = df_cache[(sym, tf)]
                            trades = backtest_cell(
                                df, sym, info, tf,
                                eq_tol_atr=eq_tol,
                                depth_thr=depth,
                                body_thr=body,
                                vol_thr=vol,
                            )
                            cell_trades_by_sym[sym] = trades
                            sc = cell_summary(cell_id, params, trades)
                            sc["symbol"] = sym
                            sc["tf"] = tf
                            sc["sym_class"] = "BIG" if sym in ("BTC/USDT","ETH/USDT") else "ALT"
                            sc["n_trades_total"] = len(trades)
                            rows.append(sc)

                        # Class-level aggregation reuses cell_trades_by_sym (no re-walk)
                        for cls, cls_syms in (("BIG", ["BTC/USDT","ETH/USDT"]),
                                              ("ALT", ["SOL/USDT","XRP/USDT"])):
                            cls_syms = [s for s in cls_syms if s in symbols]
                            if not cls_syms:
                                continue
                            all_trades = []
                            for sym in cls_syms:
                                all_trades.extend(cell_trades_by_sym.get(sym, []))
                            sc = cell_summary(cell_id, params, all_trades)
                            sc["sym_class"] = cls
                            sc["tf"] = tf
                            sc["n_trades_total"] = len(all_trades)
                            sc["symbols"] = cls_syms
                            class_rows.append(sc)

                        if cells_seen % 5 == 0:
                            print(f"   ...{cells_seen}/{total_cells} cells done")

    # Best per variant (filter PASS first, else best Q4 EV)
    def pick_best(variant: str, source_rows):
        passes = [r for r in source_rows if r[variant]["verdict"] == "PASS"]
        holds = [r for r in source_rows if r[variant]["verdict"] == "HOLD"]
        # rank by Q4 EV (primary) and Q3 EV (secondary), with min n filter
        def key(r):
            v = r[variant]
            return (v["Q4"]["ev$"], v["Q3"]["ev$"], v["IS"]["ev$"])
        if passes:
            best = sorted(passes, key=key, reverse=True)[0]
            return best
        if holds:
            return sorted(holds, key=key, reverse=True)[0]
        if not source_rows:
            return None
        return sorted(source_rows, key=key, reverse=True)[0]

    summary = {
        "config": {
            "symbols": symbols,
            "tfs": tf_grid,
            "eq_tol_atr_grid": eq_tol_grid,
            "depth_grid": depth_grid,
            "body_grid": body_grid,
            "vol_grid": vol_grid,
            "notional": NOTIONAL,
            "atr_period": ATR_PERIOD,
            "lookback_bars": LOOKBACK_BARS,
            "limit_expiry_bars": LIMIT_EXPIRY_BARS,
            "sl_buffer_atr": SL_BUFFER_ATR,
            "maker_offset_atr": MAKER_OFFSET_ATR,
            "tp1_fraction": TP1_FRACTION,
            "tp2_fallback_rr": TP2_FALLBACK_RR,
            "quarters": QUARTERS,
            "time_stop_min_per_class": {
                "BIG (BTC/ETH)": 28, "ALT (SOL/XRP)": 13,
            },
        },
        "per_symbol_rows": rows,
        "per_class_rows": class_rows,
        "best": {
            "by_symbol": {
                v: pick_best(v, rows)
                for v in ("A", "B", "C")
            },
            "by_class": {
                v: pick_best(v, class_rows)
                for v in ("A", "B", "C")
            },
        },
    }

    # Pass count
    n_pass = {v: sum(1 for r in rows if r[v]["verdict"] == "PASS") for v in ("A","B","C")}
    n_hold = {v: sum(1 for r in rows if r[v]["verdict"] == "HOLD") for v in ("A","B","C")}
    n_kill = {v: sum(1 for r in rows if r[v]["verdict"] == "KILL") for v in ("A","B","C")}
    summary["pass_counts"] = {"PASS": n_pass, "HOLD": n_hold, "KILL": n_kill,
                              "total_cells": len(rows)}

    out_json = out_dir / "walkforward.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    # Markdown report
    md = build_report(summary)
    (out_dir / "report.md").write_text(md)
    print(f"\n  wrote {out_json}")
    print(f"  wrote {out_dir / 'report.md'}")
    print(f"  PASS counts: A={n_pass['A']}  B={n_pass['B']}  C={n_pass['C']}")


def fmt_cell_line(r: dict, variant: str) -> str:
    v = r[variant]
    cls = r.get("sym_class") or r.get("symbol") or "?"
    p = r["params"]
    cell = (f"tf={p['tf']}|eq_tol={p['eq_tol_atr']}|depth={p['depth_thr']}|"
            f"body={p['body_thr']}|vol={p['vol_thr']}")
    return (
        f"- **{cls}**  `{cell}`\n"
        f"  - IS  n={v['IS']['n']}  EV/trade=${v['IS']['ev$']:+.3f}  WR={v['IS']['wr']*100:.1f}%\n"
        f"  - Q3  n={v['Q3']['n']}  EV/trade=${v['Q3']['ev$']:+.3f}  gap={v['Q3_gap_pct']:+.0f}%\n"
        f"  - Q4  n={v['Q4']['n']}  EV/trade=${v['Q4']['ev$']:+.3f}  gap={v['Q4_gap_pct']:+.0f}%\n"
        f"  - ALL n={v['ALL']['n']}  EV/trade=${v['ALL']['ev$']:+.3f}  scalper={v['ALL']['scalper_pct']*100:.0f}% "
        f"tp={v['ALL']['tp_pct']*100:.0f}% sl={v['ALL']['sl_pct']*100:.0f}% time={v['ALL']['time_pct']*100:.0f}%\n"
        f"  - **VERDICT: {v['verdict']}**"
    )


def build_report(summary: dict) -> str:
    pass_counts = summary["pass_counts"]
    cfg = summary["config"]
    rows = summary["per_symbol_rows"]
    class_rows = summary["per_class_rows"]
    best_sym = summary["best"]["by_symbol"]
    best_cls = summary["best"]["by_class"]

    lines = []
    lines.append("# Liquidity Sweep Fade — Walk-Forward Backtest (scalper-offer + maker entry)\n")
    lines.append("Re-test of previously-killed scalp class with corrected fee math.\n")
    lines.append("- Variant A: full taker RT (old kill verdict math)")
    lines.append("- Variant B: taker entry + scalper-offer waiver on exit")
    lines.append("- Variant C: maker entry (post_only LIMIT) + scalper-offer waiver on exit\n")

    lines.append("## Verdict counts (per-symbol cells)")
    for v in ("A","B","C"):
        lines.append(
            f"- **Variant {v}**: PASS={pass_counts['PASS'][v]}  "
            f"HOLD={pass_counts['HOLD'][v]}  KILL={pass_counts['KILL'][v]} "
            f"(of {pass_counts['total_cells']} cells)"
        )
    lines.append("")
    lines.append("## Walk-forward gates")
    lines.append("- IS n ≥ 25, Q4 n ≥ 10")
    lines.append("- All three of IS / Q3 / Q4 EV/trade > 0")
    lines.append("- Q4 EV/trade ≥ $0.10 (ship gate)")
    lines.append("- Q4 EV/trade ≥ 50% of IS EV/trade (no >50% gap)")
    lines.append("")

    lines.append("## Best per variant (per-symbol)")
    for v in ("A","B","C"):
        b = best_sym.get(v)
        if b is None:
            lines.append(f"### Variant {v} — no rows\n")
            continue
        lines.append(f"### Variant {v} (best per-symbol)\n")
        lines.append(fmt_cell_line(b, v))
        lines.append("")

    lines.append("## Best per variant (BIG vs ALT class)")
    for v in ("A","B","C"):
        b = best_cls.get(v)
        if b is None:
            lines.append(f"### Variant {v} — no class rows\n")
            continue
        lines.append(f"### Variant {v} (best per class)\n")
        lines.append(fmt_cell_line(b, v))
        lines.append("")

    # All Variant C PASS cells
    pass_C = [r for r in rows if r["C"]["verdict"] == "PASS"]
    if pass_C:
        lines.append(f"## Variant C — ALL PASS cells ({len(pass_C)})")
        for r in sorted(pass_C, key=lambda r: r["C"]["Q4"]["ev$"], reverse=True)[:25]:
            lines.append(fmt_cell_line(r, "C"))
            lines.append("")
    else:
        lines.append("## Variant C — no PASS cells")
        # show top 10 candidates by Q4 EV
        ranked = sorted(rows, key=lambda r: r["C"]["Q4"]["ev$"], reverse=True)[:10]
        lines.append("\nTop 10 by Q4 EV:")
        for r in ranked:
            lines.append(fmt_cell_line(r, "C"))
            lines.append("")

    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2, default=str))
    lines.append("```")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HTF (1h/4h) multi-strategy backtest — does HIGHER timeframe entry produce
edge after Delta India taker fees that 5m scalp cannot achieve?

Matrix:
  5 strategies × 4 exit configs × 4 symbols (BTC/ETH/SOL/XRP) = 80 cells
  ~6 months cached candles (2025-10-07 → 2026-04-17)

Cost model:
  Entry/Exit: 0.059% taker each leg = 0.118% RT
  Funding: 0.0001 × 8h periods (~0.01% per 8h, 0.03%/24h)
  No slippage modeled (fair comparison vs paper baseline)

Output: scoreboard sorted by net P&L; verdict on whether HTF beats fees.

Usage: python3 scripts/htf_multi_backtest.py
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/home/opc/crypto-trading-bot/storage/candle_cache")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TAKER_FEE = 0.00059               # Delta India taker per leg
FUNDING_8H = 0.0001               # 0.01% per 8h, neutral assumption
NOTIONAL = 1000.0                 # USD per trade
SL_PCT_FALLBACK = 0.0065          # used as risk denominator when ATR missing


# ─────────────────── data + indicators ───────────────────

def load_candles(sym: str, tf: str) -> pd.DataFrame:
    fname = sym.replace("/", "_") + f"_{tf}.parquet"
    df = pd.read_parquet(CACHE / fname)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema8"]  = df.close.ewm(span=8, adjust=False).mean()
    df["ema21"] = df.close.ewm(span=21, adjust=False).mean()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    delta = df.close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / dn.replace(0, 1e-10)))
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    sma = df.close.rolling(20).mean()
    std = df.close.rolling(20).std()
    df["bb_up"] = sma + 2 * std
    df["bb_dn"] = sma - 2 * std
    df["bb_width_pct"] = (df["bb_up"] - df["bb_dn"]) / sma * 100
    return df


# ─────────────────── strategy signals ───────────────────

def signal_S1_trend_pullback(t: int, df_1h: pd.DataFrame, df_4h: pd.DataFrame, idx_4h: int):
    """4h trend (EMA stack) + 1h pullback to ema21 + reversal candle."""
    if idx_4h < 50 or t < 30:
        return None
    r1 = df_1h.iloc[t]
    if pd.isna(r1.atr) or pd.isna(r1.ema21):
        return None
    r4 = df_4h.iloc[idx_4h]
    if pd.isna(r4.ema21) or pd.isna(r4.ema50):
        return None
    bull4 = r4.close > r4.ema21 > r4.ema50
    bear4 = r4.close < r4.ema21 < r4.ema50
    if not (bull4 or bear4):
        return None
    near = abs(r1.close - r1.ema21) < 0.5 * r1.atr
    if not near:
        return None
    prev = df_1h.iloc[t - 1]
    bull_candle = r1.close > r1.open and r1.close > prev.close
    bear_candle = r1.close < r1.open and r1.close < prev.close
    if bull4 and bull_candle:
        return ("long", r1.atr)
    if bear4 and bear_candle:
        return ("short", r1.atr)
    return None


def signal_S2_rsi_extreme(t: int, df_1h: pd.DataFrame, df_4h: pd.DataFrame, idx_4h: int):
    """4h RSI extreme + 1h rejection wick."""
    if idx_4h < 50 or t < 5:
        return None
    r1 = df_1h.iloc[t]
    if pd.isna(r1.rsi) or pd.isna(r1.atr):
        return None
    r4 = df_4h.iloc[idx_4h]
    if pd.isna(r4.rsi):
        return None
    body = abs(r1.close - r1.open)
    if body == 0:
        return None
    upper_wick = r1.high - max(r1.close, r1.open)
    lower_wick = min(r1.close, r1.open) - r1.low
    if r4.rsi > 75 and upper_wick > 2 * body and r1.close < r1.open:
        return ("short", r1.atr)
    if r4.rsi < 25 and lower_wick > 2 * body and r1.close > r1.open:
        return ("long", r1.atr)
    return None


def signal_S3_bb_squeeze(t: int, df_1h: pd.DataFrame):
    """1h BB squeeze percentile breakout with vol spike."""
    if t < 100:
        return None
    r1 = df_1h.iloc[t]
    if pd.isna(r1.bb_width_pct) or pd.isna(r1.atr):
        return None
    bbw_window = df_1h.iloc[t - 100:t].bb_width_pct.dropna()
    if len(bbw_window) < 50:
        return None
    pct10 = bbw_window.quantile(0.10)
    prev_bbw = df_1h.iloc[t - 1].bb_width_pct
    if pd.isna(prev_bbw) or prev_bbw > pct10:
        return None
    avg_vol = df_1h.iloc[max(0, t - 20):t].volume.mean()
    if avg_vol == 0 or r1.volume < 1.5 * avg_vol:
        return None
    if r1.close > r1.bb_up:
        return ("long", r1.atr)
    if r1.close < r1.bb_dn:
        return ("short", r1.atr)
    return None


def signal_S4_bos_4h(t: int, df_4h: pd.DataFrame):
    """4h break-of-structure with 1d (ema50) trend filter."""
    if t < 60:
        return None
    r4 = df_4h.iloc[t]
    if pd.isna(r4.atr) or pd.isna(r4.ema50):
        return None
    ema50_now = r4.ema50
    ema50_24h = df_4h.iloc[t - 6].ema50
    if pd.isna(ema50_24h):
        return None
    daily_up = ema50_now > ema50_24h
    daily_dn = ema50_now < ema50_24h
    swing = df_4h.iloc[t - 20:t]
    swing_high = swing.high.max()
    swing_low = swing.low.min()
    displacement = abs(r4.close - r4.open)
    if displacement < 0.5 * r4.atr:
        return None
    if daily_up and r4.close > swing_high:
        return ("long", r4.atr)
    if daily_dn and r4.close < swing_low:
        return ("short", r4.atr)
    return None


def signal_S5_range_fade(t: int, df_4h: pd.DataFrame):
    """4h range fade — multi-touch range edge with reversal candle."""
    if t < 50:
        return None
    r4 = df_4h.iloc[t]
    if pd.isna(r4.atr):
        return None
    window = df_4h.iloc[t - 30:t]
    range_high = window.high.quantile(0.95)
    range_low = window.low.quantile(0.05)
    range_size = range_high - range_low
    if range_size < 2 * r4.atr:
        return None
    band = 0.3 * r4.atr
    near_high = r4.high > range_high - band
    near_low = r4.low < range_low + band
    bear_candle = r4.close < r4.open
    bull_candle = r4.close > r4.open
    touches_high = ((window.high > range_high - band) & (window.high <= range_high)).sum()
    touches_low = ((window.low < range_low + band) & (window.low >= range_low)).sum()
    if touches_high >= 3 and near_high and bear_candle:
        return ("short", r4.atr)
    if touches_low >= 3 and near_low and bull_candle:
        return ("long", r4.atr)
    return None


# ─────────────────── trade simulation ───────────────────

@dataclass
class Trade:
    sym: str
    side: str
    tf: str
    entry_idx: int
    entry_time: pd.Timestamp
    entry: float
    sl: float
    atr_at_entry: float
    notional: float
    exit_idx: int = -1
    exit_time: pd.Timestamp = None
    exit_price: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    peak_mfe_r: float = 0.0
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    funding_usd: float = 0.0


EXIT_CONFIGS = {
    "E1_2R_fixed":   {"tp_R": 2.0, "time_stop_bars": 48},
    "E2_trail_1R":   {"trail_engage_R": 1.0, "trail_lock_pct": 0.5,  "time_stop_bars": 72},
    "E3_quick_15R":  {"tp_R": 1.5, "time_stop_bars": 24},
    "E4_wide_trail": {"trail_engage_R": 1.5, "trail_lock_pct": 0.33, "time_stop_bars": 168},
}


def simulate_exit(trade: Trade, df: pd.DataFrame, cfg: dict, max_idx: int, bar_hours: float) -> Trade:
    R = abs(trade.entry - trade.sl)
    if R == 0:
        trade.exit_idx = trade.entry_idx
        trade.exit_price = trade.entry
        trade.exit_reason = "zero_R"
        return trade
    bars_held = 0
    peak = 0.0
    trail_engaged = False
    trail_sl = trade.sl
    cap_idx = min(max_idx, trade.entry_idx + cfg.get("time_stop_bars", 48) + 5)
    closed = False
    for j in range(trade.entry_idx + 1, cap_idx):
        row = df.iloc[j]
        bars_held += 1
        if trade.side == "long":
            mfe_price = row.high - trade.entry
        else:
            mfe_price = trade.entry - row.low
        cur_mfe_r = mfe_price / R
        peak = max(peak, cur_mfe_r)
        # SL hit
        if trade.side == "long":
            if row.low <= trail_sl:
                trade.exit_idx = j
                trade.exit_time = row.name
                trade.exit_price = trail_sl
                trade.exit_reason = "trail_sl" if trail_engaged else "sl_hit"
                closed = True
                break
        else:
            if row.high >= trail_sl:
                trade.exit_idx = j
                trade.exit_time = row.name
                trade.exit_price = trail_sl
                trade.exit_reason = "trail_sl" if trail_engaged else "sl_hit"
                closed = True
                break
        # TP
        if "tp_R" in cfg:
            tp_price = trade.entry + cfg["tp_R"] * R if trade.side == "long" else trade.entry - cfg["tp_R"] * R
            if (trade.side == "long" and row.high >= tp_price) or \
               (trade.side == "short" and row.low <= tp_price):
                trade.exit_idx = j
                trade.exit_time = row.name
                trade.exit_price = tp_price
                trade.exit_reason = "tp_hit"
                closed = True
                break
        # Trail engagement
        if "trail_engage_R" in cfg and cur_mfe_r >= cfg["trail_engage_R"]:
            lock_r = peak * cfg.get("trail_lock_pct", 0.5)
            new_sl = trade.entry + lock_r * R if trade.side == "long" else trade.entry - lock_r * R
            if (trade.side == "long" and new_sl > trail_sl) or \
               (trade.side == "short" and new_sl < trail_sl):
                trail_sl = new_sl
                trail_engaged = True
        # Time stop
        if bars_held >= cfg.get("time_stop_bars", 9999):
            trade.exit_idx = j
            trade.exit_time = row.name
            trade.exit_price = row.close
            trade.exit_reason = "time_stop"
            closed = True
            break
    if not closed:
        last_j = min(cap_idx - 1, max_idx - 1)
        last_row = df.iloc[last_j]
        trade.exit_idx = last_j
        trade.exit_time = last_row.name
        trade.exit_price = last_row.close
        trade.exit_reason = "data_end"
    trade.bars_held = bars_held
    trade.peak_mfe_r = peak
    qty = trade.notional / trade.entry
    if trade.side == "long":
        gross = (trade.exit_price - trade.entry) * qty
    else:
        gross = (trade.entry - trade.exit_price) * qty
    fees = trade.notional * TAKER_FEE + (qty * trade.exit_price) * TAKER_FEE
    hours_held = bars_held * bar_hours
    funding_periods = hours_held / 8.0
    funding = trade.notional * FUNDING_8H * funding_periods
    net = gross - fees - funding
    trade.gross_pnl_usd = gross
    trade.fees_usd = fees
    trade.funding_usd = funding
    trade.net_pnl_usd = net
    return trade


# ─────────────────── orchestration ───────────────────

def collect_signals_1h(sym: str, strat: str):
    df_1h = add_indicators(load_candles(sym, "1h"))
    df_4h = add_indicators(load_candles(sym, "4h"))
    trades = []
    last_exit_idx = -1
    for t in range(60, len(df_1h)):
        if t <= last_exit_idx:
            continue
        idx_4h = max(0, df_4h.index.searchsorted(df_1h.index[t]) - 1)
        if strat == "S1":
            sig = signal_S1_trend_pullback(t, df_1h, df_4h, idx_4h)
        elif strat == "S2":
            sig = signal_S2_rsi_extreme(t, df_1h, df_4h, idx_4h)
        elif strat == "S3":
            sig = signal_S3_bb_squeeze(t, df_1h)
        else:
            return [], df_1h
        if not sig:
            continue
        side, atr = sig
        if pd.isna(atr) or atr <= 0:
            continue
        entry = df_1h.iloc[t].close
        sl = entry - atr if side == "long" else entry + atr
        tr = Trade(sym=sym, side=side, tf="1h", entry_idx=t,
                   entry_time=df_1h.index[t], entry=entry, sl=sl,
                   atr_at_entry=atr, notional=NOTIONAL)
        trades.append(tr)
        last_exit_idx = t + 50  # heuristic: skip 50 bars to avoid overlap
    return trades, df_1h


def collect_signals_4h(sym: str, strat: str):
    df_4h = add_indicators(load_candles(sym, "4h"))
    trades = []
    last_exit_idx = -1
    for t in range(60, len(df_4h)):
        if t <= last_exit_idx:
            continue
        if strat == "S4":
            sig = signal_S4_bos_4h(t, df_4h)
        elif strat == "S5":
            sig = signal_S5_range_fade(t, df_4h)
        else:
            return [], df_4h
        if not sig:
            continue
        side, atr = sig
        if pd.isna(atr) or atr <= 0:
            continue
        entry = df_4h.iloc[t].close
        sl = entry - atr if side == "long" else entry + atr
        tr = Trade(sym=sym, side=side, tf="4h", entry_idx=t,
                   entry_time=df_4h.index[t], entry=entry, sl=sl,
                   atr_at_entry=atr, notional=NOTIONAL)
        trades.append(tr)
        last_exit_idx = t + 30
    return trades, df_4h


def main():
    print("=" * 100)
    print("HTF MULTI-STRATEGY BACKTEST  —  5 strategies × 4 exits × 4 symbols")
    print("Fees: Delta India taker 0.059% × 2 legs. Funding: 0.01%/8h. Notional $1000/trade.")
    print(f"Symbols: {SYMBOLS}.   Cache range: ~6 months 1h+4h candles.")
    print("=" * 100)
    rows = []
    strat_tf = {"S1": "1h", "S2": "1h", "S3": "1h", "S4": "4h", "S5": "4h"}
    for sym in SYMBOLS:
        for strat, tf in strat_tf.items():
            if tf == "1h":
                signals, df = collect_signals_1h(sym, strat)
                bar_hours = 1.0
            else:
                signals, df = collect_signals_4h(sym, strat)
                bar_hours = 4.0
            for ec_name, ec in EXIT_CONFIGS.items():
                processed = [simulate_exit(replace(t), df, ec, len(df), bar_hours)
                             for t in signals]
                if not processed:
                    continue
                n = len(processed)
                wins = sum(1 for t in processed if t.net_pnl_usd > 0)
                gross = sum(t.gross_pnl_usd for t in processed)
                net = sum(t.net_pnl_usd for t in processed)
                fees = sum(t.fees_usd for t in processed)
                funding = sum(t.funding_usd for t in processed)
                avg_peak = sum(t.peak_mfe_r for t in processed) / n
                rows.append({
                    "sym": sym, "strat": strat, "exit": ec_name, "tf": tf,
                    "n": n, "wins": wins, "wr": 100 * wins / n,
                    "gross": gross, "net": net, "fees": fees, "funding": funding,
                    "peak": avg_peak, "ev": net / n,
                })
    rows.sort(key=lambda r: r["net"], reverse=True)
    print(f"\nTOP 20 (sym × strat × exit), sorted by net P&L:")
    print(f"  {'STRAT':6} {'EXIT':16} {'SYM':10} {'tf':3} {'n':>5} {'WR%':>6} "
          f"{'gross':>8} {'net':>8} {'fees':>7} {'fund':>6} {'peakR':>6} {'EV$':>7}")
    print("  " + "-" * 105)
    for r in rows[:20]:
        print(f"  {r['strat']:6} {r['exit']:16} {r['sym']:10} {r['tf']:3} {r['n']:>5} "
              f"{r['wr']:>5.1f}% ${r['gross']:>7.1f} ${r['net']:>7.1f} "
              f"${r['fees']:>6.1f} ${r['funding']:>5.1f} {r['peak']:>5.2f}R ${r['ev']:>6.2f}")
    # Aggregate by (strat, exit)
    print(f"\nAGGREGATE BY STRATEGY × EXIT (sum across 4 symbols):")
    print(f"  {'STRAT':6} {'EXIT':16} {'TOT_n':>7} {'WR%':>6} "
          f"{'gross':>9} {'net':>9} {'fees':>8} {'EV$':>7}")
    print("  " + "-" * 80)
    agg = {}
    for r in rows:
        key = (r["strat"], r["exit"])
        a = agg.setdefault(key, {"n":0,"wins":0,"gross":0,"net":0,"fees":0,"funding":0})
        a["n"] += r["n"]; a["wins"] += r["wins"]
        a["gross"] += r["gross"]; a["net"] += r["net"]
        a["fees"] += r["fees"]; a["funding"] += r["funding"]
    agg_list = sorted(agg.items(), key=lambda x: x[1]["net"], reverse=True)
    for (s, e), v in agg_list:
        wr = 100 * v["wins"] / v["n"] if v["n"] else 0
        ev = v["net"] / v["n"] if v["n"] else 0
        print(f"  {s:6} {e:16} {v['n']:>7d} {wr:>5.1f}% "
              f"${v['gross']:>8.1f} ${v['net']:>8.1f} ${v['fees']:>7.1f} ${ev:>6.2f}")
    # Verdict
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    profitable_min30 = [(k, v) for k, v in agg.items() if v["net"] > 0 and v["n"] >= 30]
    profitable_min10 = [(k, v) for k, v in agg.items() if v["net"] > 0 and v["n"] >= 10]
    if profitable_min30:
        print(f"✅ {len(profitable_min30)} strategy×exit cells PROFITABLE at Delta taker fees with n≥30")
        for (s, e), v in sorted(profitable_min30, key=lambda x: -x[1]["net"])[:5]:
            ev = v["net"] / v["n"]
            print(f"   {s}+{e}: net ${v['net']:+.2f} / {v['n']} trades / EV ${ev:+.2f}/trade")
    elif profitable_min10:
        print(f"⚠ {len(profitable_min10)} strategy×exit cells PROFITABLE with n<30 (small sample, validate)")
        for (s, e), v in sorted(profitable_min10, key=lambda x: -x[1]["net"])[:5]:
            ev = v["net"] / v["n"]
            print(f"   {s}+{e}: net ${v['net']:+.2f} / {v['n']} trades / EV ${ev:+.2f}/trade")
    else:
        print("❌ NO strategy×exit cell shows net profit at Delta taker fees")
        print("   → HTF approach with these algos also doesn't beat fees")
        print("   → Need either: maker fills (Wave 2) OR different algo class (e.g., funding arb)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""DT1 Day-Trading multi-timeframe backtest.

Strategy:
  Daily trend (resampled from 4h × 6) → 4h pullback to EMA21 → 15m reversal entry.

  Daily trend filter:
    - LONG bias  if daily close > daily EMA21 AND daily EMA21 > daily EMA50
    - SHORT bias if daily close < daily EMA21 AND daily EMA21 < daily EMA50
    - else SKIP

  4h setup (must align with daily trend):
    - Bull: 4h close within 0.5 × ATR_4h of 4h EMA21, 4h close > 4h EMA50
    - Bear: 4h close within 0.5 × ATR_4h of 4h EMA21, 4h close < 4h EMA50
    - "Armed" for next 6 × 15m bars after the 4h bar closes (~6h window).
      Wait — the spec says: "remains armed for next 6 × 15min bars (24 × 4h bars / 4 ≈ 6h max wait for entry)"
      That's 6 × 15m = 90 min, OR 24 × 15m = 6h. We use 24 × 15m = 6h max wait window.

  15m entry (only fires while 4h setup armed):
    - Bull: 15m close > open AND close > prev close AND vol > 1.0× 20-bar avg
    - Bear: 15m close < open AND close < prev close AND vol > 1.0× 20-bar avg
    - Entry = 15m close + 1 tick adverse slippage (taker)

  SL = 1.0 × ATR_4h.

  Exit configs tested:
    EA: TP=2R / SL=1R / time stop 4h (16 × 15m)
    EB: TP=3R / SL=1R / time stop 12h (48 × 15m)
    EC: trail engage 1R, lock 50% / time stop 12h
    ED: trail engage 1.5R, lock 33% / time stop 24h

  Costs: Delta India taker 0.059% × 2 = 0.118% RT; funding 0.01% / 8h held.
  Notional: $1000.

Verdict thresholds:
  SHIP : n ≥ 30 AND net EV/trade > +$1.50
  PILOT: n ≥ 30 AND net > 0 AND EV ≤ +$1.50
  KILL : net negative OR n < 30
"""
from __future__ import annotations
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
CACHE = ROOT / "storage" / "candle_cache"
OUT_DIR = ROOT / "storage" / "dt1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT"]
NOTIONAL = 1000.0
TAKER_FEE = 0.00059  # one-way
FUNDING_RATE_8H = 0.0001  # 0.01% per 8h period

# Strategy params
DAILY_EMA_FAST = 21
DAILY_EMA_SLOW = 50
H4_EMA_FAST = 21
H4_EMA_SLOW = 50
H4_PULLBACK_ATR = 0.5
H4_ATR_LEN = 14
SETUP_ARMED_15M_BARS = 24    # 6h (24 × 15m) wait window after 4h bar close
M15_VOL_LEN = 20
M15_VOL_MULT = 1.0
SL_ATR_MULT = 1.0
TICK_BPS = 0.0001            # 1 bp adverse slip


@dataclass
class Trade:
    symbol: str
    side: str
    exit_cfg: str
    entry_time: str
    entry: float
    sl: float
    atr_4h: float
    exit_time: str
    exit_price: float
    exit_reason: str
    bars_held_15m: int
    peak_mfe_r: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float


# ────────────────────────── data + indicators ──────────────────────────

def load_4h(sym: str) -> pd.DataFrame:
    fp = CACHE / f"{sym}_4h.parquet"
    df = pd.read_parquet(fp)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_15m(sym: str) -> pd.DataFrame:
    fp = CACHE / f"{sym}_15m.parquet"
    df = pd.read_parquet(fp)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def resample_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    daily = df_4h.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return daily


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["ema21"] = ema(daily.close, DAILY_EMA_FAST)
    daily["ema50"] = ema(daily.close, DAILY_EMA_SLOW)
    daily["bias"] = "neutral"
    long_mask = (daily.close > daily.ema21) & (daily.ema21 > daily.ema50)
    short_mask = (daily.close < daily.ema21) & (daily.ema21 < daily.ema50)
    daily.loc[long_mask, "bias"] = "long"
    daily.loc[short_mask, "bias"] = "short"
    return daily


def add_4h_indicators(h4: pd.DataFrame) -> pd.DataFrame:
    h4 = h4.copy()
    h4["ema21"] = ema(h4.close, H4_EMA_FAST)
    h4["ema50"] = ema(h4.close, H4_EMA_SLOW)
    h4["atr"] = atr(h4, H4_ATR_LEN)
    return h4


def add_15m_indicators(m15: pd.DataFrame) -> pd.DataFrame:
    m15 = m15.copy()
    m15["vol_ma"] = m15.volume.rolling(M15_VOL_LEN).mean()
    return m15


# ────────────────────────── setup detection ──────────────────────────

def detect_4h_setups(h4: pd.DataFrame, daily: pd.DataFrame) -> List[Dict]:
    """Return list of armed setups: each {start_ts, end_ts, side, atr, sl_dist}."""
    setups = []
    daily_aligned = daily[["bias"]].reindex(h4.index, method="ffill")
    h4 = h4.join(daily_aligned, rsuffix="_d")
    for i in range(60, len(h4)):
        bar = h4.iloc[i]
        if pd.isna(bar.atr) or pd.isna(bar.ema21) or pd.isna(bar.ema50):
            continue
        bias = bar.bias
        if bias not in ("long", "short"):
            continue
        dist = abs(bar.close - bar.ema21)
        if dist > H4_PULLBACK_ATR * bar.atr:
            continue
        if bias == "long":
            if bar.close <= bar.ema50:
                continue
            side = "long"
        else:
            if bar.close >= bar.ema50:
                continue
            side = "short"
        # Armed window: starts at 4h bar close, runs SETUP_ARMED_15M_BARS × 15m
        arm_start = bar.name  # 4h bar timestamp = bar close moment
        arm_end = arm_start + pd.Timedelta(minutes=15 * SETUP_ARMED_15M_BARS)
        setups.append({
            "start": arm_start,
            "end": arm_end,
            "side": side,
            "atr_4h": float(bar.atr),
            "h4_ema21": float(bar.ema21),
            "h4_close": float(bar.close),
        })
    return setups


# ────────────────────────── 15m entry + exit sim ──────────────────────────

def sim_exit(m15: pd.DataFrame, entry_idx: int, side: str, entry_price: float,
             atr_4h: float, exit_cfg: str) -> Optional[Dict]:
    """Step through 15m bars from entry_idx+1; return outcome dict."""
    R = SL_ATR_MULT * atr_4h
    if side == "long":
        sl = entry_price - R
    else:
        sl = entry_price + R

    # Configure exit
    if exit_cfg == "EA":
        tp_R, time_bars, trail_engage, trail_lock = 2.0, 16, None, None
    elif exit_cfg == "EB":
        tp_R, time_bars, trail_engage, trail_lock = 3.0, 48, None, None
    elif exit_cfg == "EC":
        tp_R, time_bars, trail_engage, trail_lock = None, 48, 1.0, 0.5
    elif exit_cfg == "ED":
        tp_R, time_bars, trail_engage, trail_lock = None, 96, 1.5, 0.33
    else:
        return None

    if side == "long":
        tp = entry_price + tp_R * R if tp_R else None
    else:
        tp = entry_price - tp_R * R if tp_R else None

    peak_mfe_r = 0.0
    trail_sl = sl

    for j in range(entry_idx + 1, min(entry_idx + 1 + time_bars, len(m15))):
        bar = m15.iloc[j]
        # MFE update
        if side == "long":
            mfe_price = bar.high - entry_price
        else:
            mfe_price = entry_price - bar.low
        cur_mfe_r = mfe_price / R if R > 0 else 0
        if cur_mfe_r > peak_mfe_r:
            peak_mfe_r = cur_mfe_r

        # Trail update
        if trail_engage is not None and peak_mfe_r >= trail_engage:
            lock_r = peak_mfe_r * trail_lock
            if side == "long":
                cand = entry_price + lock_r * R
                if cand > trail_sl:
                    trail_sl = cand
            else:
                cand = entry_price - lock_r * R
                if cand < trail_sl:
                    trail_sl = cand

        # SL hit (intra-bar)
        if side == "long" and bar.low <= trail_sl:
            return {
                "exit_price": trail_sl,
                "exit_reason": "trail_sl" if trail_sl != sl else "sl_hit",
                "exit_time": bar.name,
                "bars_held": j - entry_idx,
                "peak_mfe_r": peak_mfe_r,
            }
        if side == "short" and bar.high >= trail_sl:
            return {
                "exit_price": trail_sl,
                "exit_reason": "trail_sl" if trail_sl != sl else "sl_hit",
                "exit_time": bar.name,
                "bars_held": j - entry_idx,
                "peak_mfe_r": peak_mfe_r,
            }
        # TP hit
        if tp is not None:
            if side == "long" and bar.high >= tp:
                return {
                    "exit_price": tp,
                    "exit_reason": "tp_hit",
                    "exit_time": bar.name,
                    "bars_held": j - entry_idx,
                    "peak_mfe_r": peak_mfe_r,
                }
            if side == "short" and bar.low <= tp:
                return {
                    "exit_price": tp,
                    "exit_reason": "tp_hit",
                    "exit_time": bar.name,
                    "bars_held": j - entry_idx,
                    "peak_mfe_r": peak_mfe_r,
                }

    # Time stop
    last_idx = min(entry_idx + time_bars, len(m15) - 1)
    last = m15.iloc[last_idx]
    return {
        "exit_price": float(last.close),
        "exit_reason": "time_stop",
        "exit_time": last.name,
        "bars_held": last_idx - entry_idx,
        "peak_mfe_r": peak_mfe_r,
    }


def calc_pnl(entry: float, exit_price: float, side: str, bars_held_15m: int) -> Dict:
    qty = NOTIONAL / entry
    if side == "long":
        gross = (exit_price - entry) * qty
    else:
        gross = (entry - exit_price) * qty
    fees = NOTIONAL * TAKER_FEE + (qty * exit_price) * TAKER_FEE
    hours_held = bars_held_15m * 0.25
    funding_periods = hours_held / 8.0
    funding_cost = NOTIONAL * FUNDING_RATE_8H * funding_periods
    net = gross - fees - funding_cost
    return {"gross": gross, "fees": fees, "funding": funding_cost, "net": net}


def run_symbol(sym: str, exit_cfgs: List[str]) -> List[Trade]:
    h4 = load_4h(sym)
    m15 = load_15m(sym)
    if len(h4) < 100 or len(m15) < 200:
        print(f"  {sym}: not enough data h4={len(h4)} m15={len(m15)}")
        return []
    daily = resample_to_daily(h4)
    daily = add_daily_indicators(daily)
    h4 = add_4h_indicators(h4)
    m15 = add_15m_indicators(m15)

    setups = detect_4h_setups(h4, daily)
    print(f"  {sym}: {len(setups)} 4h setups detected (daily-aligned, on EMA21 pullback)")

    trades_by_cfg: Dict[str, List[Trade]] = {c: [] for c in exit_cfgs}
    # We allow only 1 entry per setup. Once filled, setup consumed.
    for setup in setups:
        side = setup["side"]
        atr_4h = setup["atr_4h"]
        # Find 15m bars within armed window
        win = m15[(m15.index >= setup["start"]) & (m15.index < setup["end"])]
        if len(win) < 2:
            continue
        # Need vol_ma available
        idx_pos = m15.index.get_indexer(win.index)
        entry_local_idx = None
        for k, j in enumerate(idx_pos):
            if j <= 0 or j >= len(m15):
                continue
            bar = m15.iloc[j]
            prev = m15.iloc[j - 1]
            if pd.isna(bar.vol_ma) or bar.vol_ma <= 0:
                continue
            if bar.volume < M15_VOL_MULT * bar.vol_ma:
                continue
            if side == "long":
                if bar.close > bar.open and bar.close > prev.close:
                    entry_local_idx = j
                    break
            else:
                if bar.close < bar.open and bar.close < prev.close:
                    entry_local_idx = j
                    break
        if entry_local_idx is None:
            continue
        bar = m15.iloc[entry_local_idx]
        raw_entry = float(bar.close)
        # Adverse slippage 1 tick (1 bp)
        if side == "long":
            entry = raw_entry * (1 + TICK_BPS)
        else:
            entry = raw_entry * (1 - TICK_BPS)
        # Run exit sim once per cfg
        for cfg in exit_cfgs:
            outcome = sim_exit(m15, entry_local_idx, side, entry, atr_4h, cfg)
            if outcome is None:
                continue
            pnl = calc_pnl(entry, outcome["exit_price"], side, outcome["bars_held"])
            sl = entry - SL_ATR_MULT * atr_4h if side == "long" else entry + SL_ATR_MULT * atr_4h
            trades_by_cfg[cfg].append(Trade(
                symbol=sym,
                side=side,
                exit_cfg=cfg,
                entry_time=str(bar.name),
                entry=entry,
                sl=sl,
                atr_4h=atr_4h,
                exit_time=str(outcome["exit_time"]),
                exit_price=outcome["exit_price"],
                exit_reason=outcome["exit_reason"],
                bars_held_15m=outcome["bars_held"],
                peak_mfe_r=outcome["peak_mfe_r"],
                gross_pnl=pnl["gross"],
                fees=pnl["fees"],
                funding=pnl["funding"],
                net_pnl=pnl["net"],
            ))
    return [t for cfg in exit_cfgs for t in trades_by_cfg[cfg]]


# ────────────────────────── reporting ──────────────────────────

def stats(trades: List[Trade]) -> Dict:
    if not trades:
        return {"n": 0, "wr": 0, "gross": 0, "net": 0, "ev": 0,
                "avg_mfe": 0, "avg_hold_bars": 0, "avg_hold_h": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    gross = sum(t.gross_pnl for t in trades)
    net = sum(t.net_pnl for t in trades)
    avg_mfe = sum(t.peak_mfe_r for t in trades) / n
    avg_hold = sum(t.bars_held_15m for t in trades) / n
    return {"n": n, "wr": 100 * wins / n,
            "gross": gross, "net": net, "ev": net / n,
            "avg_mfe": avg_mfe, "avg_hold_bars": avg_hold,
            "avg_hold_h": avg_hold * 0.25}


def verdict(s: Dict) -> str:
    if s["n"] < 30 or s["net"] < 0:
        return "KILL"
    if s["ev"] > 1.5:
        return "SHIP"
    return "PILOT"


def main():
    exit_cfgs = ["EA", "EB", "EC", "ED"]
    all_trades: Dict[str, Dict[str, List[Trade]]] = {}
    for sym in SYMBOLS:
        print(f"\n[{sym}]")
        try:
            trades = run_symbol(sym, exit_cfgs)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        all_trades[sym] = {c: [t for t in trades if t.exit_cfg == c] for c in exit_cfgs}
        for c in exit_cfgs:
            s = stats(all_trades[sym][c])
            print(f"    {c}: n={s['n']:3d} WR={s['wr']:5.1f}% gross=${s['gross']:8.2f} "
                  f"net=${s['net']:8.2f} ev=${s['ev']:+6.2f}/trade")

    # Aggregates per cfg across all symbols
    cfg_agg: Dict[str, Dict] = {}
    for c in exit_cfgs:
        bucket = []
        for sym in SYMBOLS:
            bucket.extend(all_trades.get(sym, {}).get(c, []))
        cfg_agg[c] = {"trades": bucket, "stats": stats(bucket)}

    # Best cfg
    best_cfg = max(cfg_agg, key=lambda c: cfg_agg[c]["stats"]["net"])
    best_stats = cfg_agg[best_cfg]["stats"]
    final_verdict = verdict(best_stats)

    # Scoreboard CSV
    scoreboard_path = OUT_DIR / "scoreboard.csv"
    with scoreboard_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "exit_cfg", "n", "wr_pct", "gross_usd",
                    "net_usd", "ev_per_trade", "avg_mfe_r", "avg_hold_h"])
        for sym in SYMBOLS:
            for c in exit_cfgs:
                s = stats(all_trades.get(sym, {}).get(c, []))
                w.writerow([sym, c, s["n"], f"{s['wr']:.1f}",
                            f"{s['gross']:.2f}", f"{s['net']:.2f}",
                            f"{s['ev']:.2f}", f"{s['avg_mfe']:.2f}",
                            f"{s['avg_hold_h']:.1f}"])
        for c in exit_cfgs:
            s = cfg_agg[c]["stats"]
            w.writerow(["AGGREGATE", c, s["n"], f"{s['wr']:.1f}",
                        f"{s['gross']:.2f}", f"{s['net']:.2f}",
                        f"{s['ev']:.2f}", f"{s['avg_mfe']:.2f}",
                        f"{s['avg_hold_h']:.1f}"])

    # Markdown report
    lines = ["# DT1 Day-Trading Backtest Report",
             "",
             "Strategy: Daily trend → 4h EMA21 pullback → 15m reversal entry.",
             "Symbols: BTC/ETH/SOL/XRP. Notional $1000. Delta India taker 0.118% RT + funding 0.01%/8h.",
             "",
             f"Data range: {pd.read_parquet(CACHE / 'BTC_USDT_4h.parquet').index.min()} — "
             f"{pd.read_parquet(CACHE / 'BTC_USDT_4h.parquet').index.max()}",
             "",
             "## Per-symbol per-exit",
             "",
             "| Symbol | Cfg | n | WR% | Gross$ | Net$ | EV$/trade | Avg MFE R | Avg Hold h |",
             "|---|---|---|---|---|---|---|---|---|"]
    for sym in SYMBOLS:
        for c in exit_cfgs:
            s = stats(all_trades.get(sym, {}).get(c, []))
            lines.append(f"| {sym} | {c} | {s['n']} | {s['wr']:.1f} | "
                         f"{s['gross']:.2f} | {s['net']:.2f} | {s['ev']:+.2f} | "
                         f"{s['avg_mfe']:.2f} | {s['avg_hold_h']:.1f} |")
    lines.append("")
    lines.append("## Aggregates per exit (all 4 symbols)")
    lines.append("")
    lines.append("| Cfg | n | WR% | Gross$ | Net$ | EV$/trade | Avg MFE R | Avg Hold h |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in exit_cfgs:
        s = cfg_agg[c]["stats"]
        lines.append(f"| {c} | {s['n']} | {s['wr']:.1f} | "
                     f"{s['gross']:.2f} | {s['net']:.2f} | {s['ev']:+.2f} | "
                     f"{s['avg_mfe']:.2f} | {s['avg_hold_h']:.1f} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Best exit: **{best_cfg}**")
    lines.append(f"- Best n: {best_stats['n']}, WR: {best_stats['wr']:.1f}%, "
                 f"net ${best_stats['net']:.2f}, EV ${best_stats['ev']:+.2f}/trade")
    lines.append(f"- **Verdict: {final_verdict}**")
    lines.append("")
    lines.append("Thresholds: SHIP = n≥30 AND EV>+$1.50. PILOT = n≥30 AND net>0 AND EV≤+$1.50. KILL otherwise.")

    rpt_path = OUT_DIR / "backtest_report.md"
    rpt_path.write_text("\n".join(lines))

    # JSON dump for re-analysis
    trades_json = {
        sym: {c: [asdict(t) for t in all_trades.get(sym, {}).get(c, [])]
              for c in exit_cfgs}
        for sym in SYMBOLS
    }
    (OUT_DIR / "trades_full.json").write_text(json.dumps(trades_json, indent=2, default=str))

    print(f"\n=== AGGREGATE per exit ===")
    for c in exit_cfgs:
        s = cfg_agg[c]["stats"]
        print(f"  {c}: n={s['n']} WR={s['wr']:.1f}% net=${s['net']:.2f} "
              f"EV=${s['ev']:+.2f}/trade  avg_MFE={s['avg_mfe']:.2f}R "
              f"hold={s['avg_hold_h']:.1f}h")
    print(f"\nBEST: {best_cfg}  VERDICT: {final_verdict}")
    print(f"Report: {rpt_path}")
    print(f"Scoreboard: {scoreboard_path}")
    return final_verdict, best_cfg, best_stats


if __name__ == "__main__":
    v, c, s = main()
    sys.exit(0)

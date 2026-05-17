#!/usr/bin/env python3
"""Per-scanner edge backtest for the 3 scanners that were silently crashing
(rsi_divergence, rsi_extreme, bb_squeeze) before the observability + ATR fix.

For each scanner:
- Walk 5m candles for BTC/ETH/SOL/XRP, ~6 months
- Apply scanner core signal logic (replicated from scalp_strategy.py)
- Simulate exits: 1R SL / 2R TP / 60-bar time stop (typical 5m scalp targets)
- Apply Delta India taker fees (0.059% × 2)
- Report: n, WR%, gross_R, net_R, gross_$, net_$, EV/trade

Goal: confirm whether unlocking these scanners adds positive net expectancy
or they bleed like structure_bounce.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/home/opc/crypto-trading-bot/storage/candle_cache")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TAKER_FEE = 0.00059
NOTIONAL = 1000.0
SL_R_PRICE_FALLBACK = 0.0065     # 0.65% if ATR-based SL invalid
TP_R = 2.0
TIME_STOP_BARS = 60               # 60 × 5m = 5h
ATR_LEN = 14
RSI_LEN = 14
BB_LEN = 20


def load_5m(sym: str) -> pd.DataFrame:
    fname = sym.replace("/", "_") + "_5m.parquet"
    df = pd.read_parquet(CACHE / fname)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta = df.close.diff()
    up = delta.clip(lower=0).rolling(RSI_LEN).mean()
    dn = (-delta.clip(upper=0)).rolling(RSI_LEN).mean()
    df["rsi"] = 100 - (100 / (1 + up / dn.replace(0, 1e-10)))
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    sma = df.close.rolling(BB_LEN).mean()
    std = df.close.rolling(BB_LEN).std()
    df["bb_up"] = sma + 2 * std
    df["bb_dn"] = sma - 2 * std
    df["bb_bandwidth"] = (df["bb_up"] - df["bb_dn"]) / sma
    df["bb_pct_b"] = (df.close - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"])
    return df


# ─────────────────── scanner core logic (replicated) ───────────────────

def signal_rsi_divergence(t: int, df: pd.DataFrame) -> tuple | None:
    """Simplified replica of _scan_rsi_divergence."""
    lookback = 12
    if t < lookback + 5:
        return None
    last = df.iloc[t]
    if pd.isna(last.atr) or pd.isna(last.rsi) or last.atr <= 0:
        return None
    window = df.iloc[t - lookback:t + 1]
    prices = window.close.values
    rsis = window.rsi.values
    if np.any(np.isnan(rsis)):
        return None
    # Bullish div: current price near recent low, RSI higher than at that low
    pmin_idx = np.argmin(prices[:-1])
    pmin = prices[pmin_idx]
    rsi_at_pmin = rsis[pmin_idx]
    if last.close < pmin * 1.005 and last.rsi > rsi_at_pmin + 5:
        return ("long", last.atr)
    # Bearish div
    pmax_idx = np.argmax(prices[:-1])
    pmax = prices[pmax_idx]
    rsi_at_pmax = rsis[pmax_idx]
    if last.close > pmax * 0.995 and last.rsi < rsi_at_pmax - 5:
        return ("short", last.atr)
    return None


def signal_rsi_extreme(t: int, df: pd.DataFrame) -> tuple | None:
    """Simplified replica of _scan_rsi_extreme."""
    if t < 5:
        return None
    last = df.iloc[t]
    prev = df.iloc[t - 1]
    if pd.isna(last.atr) or last.atr <= 0 or pd.isna(last.rsi) or pd.isna(prev.rsi):
        return None
    # OVERSOLD bounce
    if prev.rsi < 30 and last.rsi > prev.rsi and last.close > last.open:
        return ("long", last.atr)
    # OVERBOUGHT fade
    if prev.rsi > 70 and last.rsi < prev.rsi and last.close < last.open:
        return ("short", last.atr)
    return None


def signal_bb_squeeze(t: int, df: pd.DataFrame) -> tuple | None:
    """Simplified replica of _scan_bb_squeeze."""
    if t < 100:
        return None
    last = df.iloc[t]
    prev = df.iloc[t - 1]
    if pd.isna(last.atr) or last.atr <= 0:
        return None
    bw_now = last.bb_bandwidth
    bw_prev = prev.bb_bandwidth
    if pd.isna(bw_now) or pd.isna(bw_prev):
        return None
    bw_window = df.iloc[t - 100:t].bb_bandwidth.dropna()
    if len(bw_window) < 50:
        return None
    bw_25 = bw_window.quantile(0.25)
    was_squeezed = bw_prev <= bw_25
    is_expanding = bw_now > bw_prev * 1.05
    if not (was_squeezed and is_expanding):
        return None
    # Direction: vol-confirmed breakout
    avg_vol = df.iloc[max(0, t - 20):t].volume.mean()
    if avg_vol == 0 or last.volume < 1.3 * avg_vol:
        return None
    if last.close > last.bb_up:
        return ("long", last.atr)
    if last.close < last.bb_dn:
        return ("short", last.atr)
    return None


# ─────────────────── exit simulation ───────────────────

@dataclass
class Trade:
    sym: str
    side: str
    entry_idx: int
    entry: float
    sl: float
    atr: float
    notional: float = NOTIONAL
    exit_idx: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    peak_mfe_r: float = 0.0
    gross_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    net_pnl_usd: float = 0.0


def simulate_exit(trade: Trade, df: pd.DataFrame) -> Trade:
    R = abs(trade.entry - trade.sl)
    if R == 0:
        trade.exit_idx = trade.entry_idx
        trade.exit_price = trade.entry
        trade.exit_reason = "zero_R"
        return trade
    tp_price = (trade.entry + TP_R * R) if trade.side == "long" else (trade.entry - TP_R * R)
    cap = min(len(df), trade.entry_idx + TIME_STOP_BARS + 1)
    peak = 0.0
    closed = False
    for j in range(trade.entry_idx + 1, cap):
        row = df.iloc[j]
        trade.bars_held += 1
        if trade.side == "long":
            mfe = (row.high - trade.entry) / R
            if row.low <= trade.sl:
                trade.exit_idx = j; trade.exit_price = trade.sl; trade.exit_reason = "sl"; closed = True; break
            if row.high >= tp_price:
                trade.exit_idx = j; trade.exit_price = tp_price; trade.exit_reason = "tp"; closed = True; break
        else:
            mfe = (trade.entry - row.low) / R
            if row.high >= trade.sl:
                trade.exit_idx = j; trade.exit_price = trade.sl; trade.exit_reason = "sl"; closed = True; break
            if row.low <= tp_price:
                trade.exit_idx = j; trade.exit_price = tp_price; trade.exit_reason = "tp"; closed = True; break
        peak = max(peak, mfe)
        if trade.bars_held >= TIME_STOP_BARS:
            trade.exit_idx = j; trade.exit_price = row.close; trade.exit_reason = "time"; closed = True; break
    if not closed:
        last_j = cap - 1
        last_row = df.iloc[last_j]
        trade.exit_idx = last_j; trade.exit_price = last_row.close; trade.exit_reason = "end"
    trade.peak_mfe_r = peak
    qty = trade.notional / trade.entry
    if trade.side == "long":
        gross = (trade.exit_price - trade.entry) * qty
    else:
        gross = (trade.entry - trade.exit_price) * qty
    fees = trade.notional * TAKER_FEE + (qty * trade.exit_price) * TAKER_FEE
    trade.gross_pnl_usd = gross
    trade.fees_usd = fees
    trade.net_pnl_usd = gross - fees
    return trade


# ─────────────────── orchestration ───────────────────

def run_scanner(scanner_name: str, signal_fn, sym: str, df: pd.DataFrame) -> list:
    trades = []
    last_exit = -1
    for t in range(50, len(df)):
        if t <= last_exit:
            continue
        sig = signal_fn(t, df)
        if not sig:
            continue
        side, atr = sig
        if pd.isna(atr) or atr <= 0:
            continue
        last = df.iloc[t]
        entry = last.close
        sl = entry - atr if side == "long" else entry + atr
        tr = Trade(sym=sym, side=side, entry_idx=t, entry=entry, sl=sl, atr=atr)
        tr = simulate_exit(tr, df)
        trades.append(tr)
        last_exit = tr.exit_idx + 5  # cooldown
    return trades


def summarize(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0, "wr": 0, "gross": 0, "net": 0, "fees": 0, "ev": 0, "peak": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl_usd > 0)
    gross = sum(t.gross_pnl_usd for t in trades)
    fees = sum(t.fees_usd for t in trades)
    net = sum(t.net_pnl_usd for t in trades)
    avg_peak = sum(t.peak_mfe_r for t in trades) / n
    return {
        "label": label, "n": n, "wins": wins, "wr": 100 * wins / n,
        "gross": gross, "fees": fees, "net": net, "ev": net / n, "peak": avg_peak,
    }


def main():
    print("=" * 95)
    print("PER-SCANNER EDGE BACKTEST  —  rsi_divergence / rsi_extreme / bb_squeeze")
    print("Replicated core logic. 5m candles, ~6 months, BTC/ETH/SOL/XRP.")
    print("Exit: 1R SL / 2R TP / 60-bar time stop. Fees: 0.059% × 2 = 0.118% RT.")
    print("=" * 95)
    scanners = {
        "rsi_divergence": signal_rsi_divergence,
        "rsi_extreme":    signal_rsi_extreme,
        "bb_squeeze":     signal_bb_squeeze,
    }
    sym_dfs = {}
    for sym in SYMBOLS:
        try:
            sym_dfs[sym] = add_indicators(load_5m(sym))
            print(f"  {sym}: {len(sym_dfs[sym])} 5m bars")
        except Exception as e:
            print(f"  {sym}: load failed: {e}")
    print()
    rows = []
    for sname, sfn in scanners.items():
        per_sym = []
        for sym, df in sym_dfs.items():
            trades = run_scanner(sname, sfn, sym, df)
            s = summarize(trades, f"{sname} / {sym}")
            s["sym"] = sym
            s["scanner"] = sname
            per_sym.append(s)
            rows.append(s)
        # aggregate
        n_tot = sum(s["n"] for s in per_sym)
        if n_tot:
            wins = sum(s["wins"] for s in per_sym)
            gross = sum(s["gross"] for s in per_sym)
            fees = sum(s["fees"] for s in per_sym)
            net = sum(s["net"] for s in per_sym)
            print(f"{sname:18}  n={n_tot:>5}  WR={100*wins/n_tot:>5.1f}%  "
                  f"gross=${gross:>+8.2f}  fees=${fees:>7.2f}  net=${net:>+8.2f}  EV=${net/n_tot:>+6.2f}")
            for s in per_sym:
                if s["n"]:
                    print(f"  └─ {s['sym']:10} n={s['n']:>4}  WR={s['wr']:>5.1f}%  "
                          f"gross=${s['gross']:>+7.2f}  net=${s['net']:>+7.2f}  EV=${s['ev']:>+6.2f}  peak={s['peak']:.2f}R")
            print()
    # Verdict
    print("=" * 95)
    print("VERDICT")
    print("=" * 95)
    by_scanner = {}
    for r in rows:
        by_scanner.setdefault(r["scanner"], {"n":0,"net":0,"wins":0,"gross":0})
        a = by_scanner[r["scanner"]]
        a["n"] += r["n"]; a["wins"] += r["wins"]
        a["gross"] += r["gross"]; a["net"] += r["net"]
    for sname, a in sorted(by_scanner.items(), key=lambda x: -x[1]["net"]):
        if a["n"] == 0:
            print(f"  {sname}: NO SIGNALS — likely too restrictive")
            continue
        ev = a["net"] / a["n"]
        wr = 100 * a["wins"] / a["n"]
        if a["net"] > 0 and a["n"] >= 30:
            verdict = "SHIP"
        elif a["net"] > 0:
            verdict = "PILOT (small n)"
        elif a["net"] > -5 and a["n"] >= 30:
            verdict = "BREAKEVEN"
        else:
            verdict = "KILL"
        print(f"  {sname:18}: {verdict}  (n={a['n']}, WR={wr:.1f}%, net=${a['net']:+.2f}, EV=${ev:+.2f})")


if __name__ == "__main__":
    main()

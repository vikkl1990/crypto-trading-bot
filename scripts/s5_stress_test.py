#!/usr/bin/env python3
"""S5 (4h Range Fade) — STRESS TEST across regimes / cost shocks.

Read-only. Re-runs the same signal_S5_range_fade logic from htf_multi_backtest.py
across BTC/ETH/SOL/XRP, then segments the resulting trades by:

  1. Trend regime (bull / bear / choppy) — 4h EMA stack at entry
  2. Volatility regime (low/mid/high) — ATR percentile vs 100-bar window at entry
  3. Time-period quartiles (4 equal slices of the 6.3 month sample)
  4. Side (long-only vs short-only)
  5. Drawdown analysis (max DD, longest losing streak, time-to-recover)
  6. Funding shock (4× funding = 0.04%/8h)
  7. Slippage shock (0.05% adverse slippage at entry AND exit)

For each segment: n, WR%, gross, net, EV/trade, max DD vs base.

Run:  python3 /home/opc/crypto-trading-bot/scripts/s5_stress_test.py
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/home/opc/crypto-trading-bot/storage/candle_cache")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

# ─── Cost model (base) ───
TAKER_FEE = 0.00059          # Delta India taker per leg
FUNDING_8H_BASE = 0.0001     # 0.01% / 8h
FUNDING_8H_STRESS = 0.0004   # 4× shock
SLIP_STRESS = 0.0005         # 0.05% adverse at each leg
NOTIONAL = 1000.0

# ─── Best base exit (per task brief) ───
EXIT_CFG = {"trail_engage_R": 1.5, "trail_lock_pct": 0.33, "time_stop_bars": 168}


# ─────────────────── data + indicators (verbatim from htf_multi_backtest) ───────────────────

def load_candles(sym: str, tf: str) -> pd.DataFrame:
    fname = sym.replace("/", "_") + f"_{tf}.parquet"
    df = pd.read_parquet(CACHE / fname)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema21"] = df.close.ewm(span=21, adjust=False).mean()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


# ─────────────────── S5 signal (verbatim) ───────────────────

def signal_S5_range_fade(t: int, df_4h: pd.DataFrame):
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


# ─────────────────── trade objects ───────────────────

@dataclass
class Trade:
    sym: str
    side: str
    entry_idx: int
    entry_time: pd.Timestamp
    entry: float
    sl: float
    atr_at_entry: float
    notional: float = NOTIONAL
    # Context tags for segmentation
    regime: str = ""
    vol_bucket: str = ""
    # Filled after exit
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


def simulate_exit(trade: Trade, df: pd.DataFrame, cfg: dict,
                  funding_8h: float, slip_pct: float, bar_hours: float = 4.0) -> Trade:
    """Same exit logic as base. slip_pct=0.0005 means 0.05% adverse on each leg."""
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
    cap_idx = min(len(df), trade.entry_idx + cfg.get("time_stop_bars", 48) + 5)
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
        # SL
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
        last_j = min(cap_idx - 1, len(df) - 1)
        last_row = df.iloc[last_j]
        trade.exit_idx = last_j
        trade.exit_time = last_row.name
        trade.exit_price = last_row.close
        trade.exit_reason = "data_end"
    trade.bars_held = bars_held
    trade.peak_mfe_r = peak

    # Apply slippage adverse to side
    eff_entry = trade.entry * (1 + slip_pct) if trade.side == "long" else trade.entry * (1 - slip_pct)
    eff_exit = trade.exit_price * (1 - slip_pct) if trade.side == "long" else trade.exit_price * (1 + slip_pct)

    qty = trade.notional / eff_entry
    if trade.side == "long":
        gross = (eff_exit - eff_entry) * qty
    else:
        gross = (eff_entry - eff_exit) * qty
    fees = trade.notional * TAKER_FEE + (qty * eff_exit) * TAKER_FEE
    funding_periods = (bars_held * bar_hours) / 8.0
    funding = trade.notional * funding_8h * funding_periods
    net = gross - fees - funding
    trade.gross_pnl_usd = gross
    trade.fees_usd = fees
    trade.funding_usd = funding
    trade.net_pnl_usd = net
    return trade


# ─────────────────── signal collection + tagging ───────────────────

def collect_s5_signals(sym: str):
    df_4h = add_indicators(load_candles(sym, "4h"))
    trades = []
    last_exit_idx = -1
    for t in range(60, len(df_4h)):
        if t <= last_exit_idx:
            continue
        sig = signal_S5_range_fade(t, df_4h)
        if not sig:
            continue
        side, atr = sig
        if pd.isna(atr) or atr <= 0:
            continue
        r4 = df_4h.iloc[t]
        entry = r4.close
        sl = entry - atr if side == "long" else entry + atr

        # Tag regime (4h EMA stack at entry)
        if pd.notna(r4.ema21) and pd.notna(r4.ema50):
            if r4.close > r4.ema21 > r4.ema50:
                regime = "bull"
            elif r4.close < r4.ema21 < r4.ema50:
                regime = "bear"
            else:
                regime = "choppy"
        else:
            regime = "choppy"

        # Tag volatility (ATR percentile in 100-bar window)
        atr_window = df_4h.iloc[max(0, t - 100):t].atr.dropna()
        if len(atr_window) >= 50:
            p25, p75 = atr_window.quantile(0.25), atr_window.quantile(0.75)
            if atr < p25:
                vol_b = "low_vol"
            elif atr > p75:
                vol_b = "high_vol"
            else:
                vol_b = "mid_vol"
        else:
            vol_b = "mid_vol"

        tr = Trade(sym=sym, side=side, entry_idx=t, entry_time=df_4h.index[t],
                   entry=entry, sl=sl, atr_at_entry=atr,
                   regime=regime, vol_bucket=vol_b)
        trades.append(tr)
        last_exit_idx = t + 30
    return trades, df_4h


# ─────────────────── stress runners ───────────────────

def run_scenario(funding_8h: float, slip_pct: float):
    """Re-run all S5 trades with given cost stress."""
    all_trades = []
    for sym in SYMBOLS:
        sigs, df = collect_s5_signals(sym)
        for t in sigs:
            t2 = simulate_exit(replace(t), df, EXIT_CFG, funding_8h, slip_pct)
            all_trades.append(t2)
    all_trades.sort(key=lambda x: x.entry_time)
    return all_trades


# ─────────────────── stats helpers ───────────────────

def stats(trs):
    if not trs:
        return {"n": 0, "wr": 0.0, "gross": 0.0, "net": 0.0, "ev": 0.0,
                "max_dd": 0.0, "max_streak": 0}
    n = len(trs)
    wins = sum(1 for t in trs if t.net_pnl_usd > 0)
    gross = sum(t.gross_pnl_usd for t in trs)
    net = sum(t.net_pnl_usd for t in trs)
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    streak = 0
    max_streak = 0
    time_to_recover_bars = None
    underwater_since = None
    for t in trs:
        eq += t.net_pnl_usd
        if eq > peak:
            peak = eq
            if underwater_since is not None:
                time_to_recover_bars = None
                underwater_since = None
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
            if underwater_since is None:
                underwater_since = t.entry_time
        if t.net_pnl_usd <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "n": n, "wr": 100 * wins / n, "gross": gross, "net": net,
        "ev": net / n, "max_dd": max_dd, "max_streak": max_streak,
    }


def fmt_row(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"  {label:24} n=0  (no trades)"
    return (f"  {label:24} n={s['n']:>3}  WR={s['wr']:>5.1f}%  "
            f"gross=${s['gross']:>+8.2f}  net=${s['net']:>+8.2f}  "
            f"EV=${s['ev']:>+6.2f}  maxDD=${s['max_dd']:>6.2f}  "
            f"loseStreak={s['max_streak']}")


# ─────────────────── main ───────────────────

def main():
    print("=" * 110)
    print("S5 (4h Range Fade) STRESS TEST  —  E4_wide_trail exit, BTC/ETH/SOL/XRP")
    print("=" * 110)

    # ── Base run ──
    base = run_scenario(FUNDING_8H_BASE, slip_pct=0.0)
    base_s = stats(base)
    print("\n[BASE — funding 0.01%/8h, no slippage]")
    print(fmt_row("ALL", base_s))

    # 1. Regime segmentation
    print("\n[1] TREND REGIME (4h EMA stack at entry)")
    for r in ("bull", "bear", "choppy"):
        sub = [t for t in base if t.regime == r]
        print(fmt_row(r, stats(sub)))

    # 2. Vol regime
    print("\n[2] VOLATILITY REGIME (ATR percentile, 100-bar)")
    for v in ("low_vol", "mid_vol", "high_vol"):
        sub = [t for t in base if t.vol_bucket == v]
        print(fmt_row(v, stats(sub)))

    # 3. Time-period quartiles
    print("\n[3] TIME-PERIOD QUARTILES")
    if base:
        ts = sorted(base, key=lambda t: t.entry_time)
        n = len(ts)
        for q in range(4):
            lo = q * n // 4
            hi = (q + 1) * n // 4 if q < 3 else n
            sub = ts[lo:hi]
            label = (f"Q{q+1} ({sub[0].entry_time.date()}→"
                     f"{sub[-1].entry_time.date()})") if sub else f"Q{q+1}"
            print(fmt_row(label, stats(sub)))

    # 4. Side
    print("\n[4] SIDE (long vs short)")
    for s in ("long", "short"):
        sub = [t for t in base if t.side == s]
        print(fmt_row(s, stats(sub)))

    # 5. Drawdown / streak detail (already in stats), expand:
    print("\n[5] DRAWDOWN DETAIL")
    print(f"  Max DD ${base_s['max_dd']:.2f}  |  Longest losing streak: {base_s['max_streak']}")
    # Trade-by-trade equity curve markers
    eq, peak, peak_idx, trough_idx, trough = 0.0, 0.0, 0, 0, 0.0
    for i, t in enumerate(sorted(base, key=lambda t: t.entry_time)):
        eq += t.net_pnl_usd
        if eq > peak:
            peak = eq
            peak_idx = i
        dd = peak - eq
        if dd > -trough:  # most negative trough
            trough = -dd
            trough_idx = i
    print(f"  Peak-trough excursion: trade #{peak_idx} → #{trough_idx} "
          f"(span={trough_idx - peak_idx} trades)")

    # 6. Funding shock
    fund_stress = run_scenario(FUNDING_8H_STRESS, slip_pct=0.0)
    fund_s = stats(fund_stress)
    print("\n[6] FUNDING SHOCK (0.04%/8h = 4× base)")
    print(fmt_row("ALL_funding4x", fund_s))
    print(f"  vs base: net ΔΔ ${fund_s['net'] - base_s['net']:+.2f}, "
          f"EV ΔΔ ${fund_s['ev'] - base_s['ev']:+.2f}")

    # 7. Slippage shock
    slip_stress = run_scenario(FUNDING_8H_BASE, slip_pct=SLIP_STRESS)
    slip_s = stats(slip_stress)
    print("\n[7] SLIPPAGE SHOCK (0.05% adverse at entry AND exit)")
    print(fmt_row("ALL_slip5bp", slip_s))
    print(f"  vs base: net ΔΔ ${slip_s['net'] - base_s['net']:+.2f}, "
          f"EV ΔΔ ${slip_s['ev'] - base_s['ev']:+.2f}")

    # Combined shock
    combo = run_scenario(FUNDING_8H_STRESS, slip_pct=SLIP_STRESS)
    combo_s = stats(combo)
    print("\n[8] COMBINED SHOCK (4× funding + 5bp slip)")
    print(fmt_row("ALL_combo", combo_s))

    # ── Verdict ──
    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)
    base_ev = base_s["ev"]
    kill = []
    fragile = []

    # Per-regime kill check
    for r in ("bull", "bear", "choppy"):
        sub = [t for t in base if t.regime == r]
        s = stats(sub)
        if s["n"] >= 8 and s["net"] < 0:
            kill.append(f"trend regime={r} (n={s['n']}, net=${s['net']:+.2f}, EV=${s['ev']:+.2f})")
    for v in ("low_vol", "mid_vol", "high_vol"):
        sub = [t for t in base if t.vol_bucket == v]
        s = stats(sub)
        if s["n"] >= 8 and s["net"] < 0:
            kill.append(f"vol={v} (n={s['n']}, net=${s['net']:+.2f}, EV=${s['ev']:+.2f})")
    for s_side in ("long", "short"):
        sub = [t for t in base if t.side == s_side]
        s = stats(sub)
        if s["n"] >= 8 and s["net"] < 0:
            kill.append(f"side={s_side} (n={s['n']}, net=${s['net']:+.2f}, EV=${s['ev']:+.2f})")
    # Time quartile stability
    if base:
        ts = sorted(base, key=lambda t: t.entry_time)
        n = len(ts)
        for q in range(4):
            lo = q * n // 4
            hi = (q + 1) * n // 4 if q < 3 else n
            sub = ts[lo:hi]
            s = stats(sub)
            if s["n"] >= 8 and s["net"] < 0:
                kill.append(f"Q{q+1} (n={s['n']}, net=${s['net']:+.2f})")

    if fund_s["net"] < 0:
        fragile.append(f"funding 4× kills edge (net=${fund_s['net']:+.2f})")
    elif fund_s["ev"] < base_ev * 0.5:
        fragile.append(f"funding 4× halves EV (${base_ev:+.2f}→${fund_s['ev']:+.2f})")
    if slip_s["net"] < 0:
        fragile.append(f"5bp slip kills edge (net=${slip_s['net']:+.2f})")
    elif slip_s["ev"] < base_ev * 0.5:
        fragile.append(f"5bp slip halves EV (${base_ev:+.2f}→${slip_s['ev']:+.2f})")
    if combo_s["net"] < 0:
        fragile.append(f"combined shock kills edge (net=${combo_s['net']:+.2f})")

    if kill:
        print("KILL CANDIDATES (segments where S5 LOSES money, n>=8):")
        for k in kill:
            print(f"  - {k}")
    else:
        print("No segment with n>=8 loses money on base costs.")
    if fragile:
        print("FRAGILITY FLAGS:")
        for f in fragile:
            print(f"  - {f}")
    else:
        print("Edge survives funding 4× and 5bp slippage stresses.")

    if base_s["net"] <= 0:
        verdict = "FAILED"
    elif kill or len(fragile) >= 2 or combo_s["net"] < 0:
        verdict = "FRAGILE"
    else:
        verdict = "ROBUST"
    print(f"\nFINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()

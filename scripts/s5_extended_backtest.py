#!/usr/bin/env python3
"""S5 (4h Range Fade) extended parameter sweep.

Holds the rest of the S5 base implementation constant and varies one knob at
a time vs. the base config:
  Base = touches=3, range_mult=2.0, window=30, band=0.3, sl_atr=1.0,
         exit=E4_wide_trail (1.5R/33%), side=both

Variants:
  1. touches:        2, 3*, 4, 5
  2. range_mult:     1.5, 2.0*, 2.5, 3.0
  3. window:         20, 30*, 50, 80
  4. band:           0.2, 0.3*, 0.5, 0.7
  5. sl_atr:         0.7, 1.0*, 1.5, 2.0
  6. exit_cfg:       E4(1.5R/33%)*, (1.0R/50%), (2.0R/25%), (1.5R/50%)
  7. side:           long-only, short-only, both*

Cost model unchanged from htf_multi_backtest: 0.059% taker × 2 + 0.01%/8h
funding, $1000 notional. Reports top 20 by net P&L + per-axis sweeps.

Read-only. Writes nothing outside scripts/. No production code modified.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/home/opc/crypto-trading-bot/storage/candle_cache")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TAKER_FEE = 0.00059
FUNDING_8H = 0.0001
NOTIONAL = 1000.0


# ─────────────────── data + indicators (mirrors htf_multi_backtest) ───────────────────

def load_candles(sym: str, tf: str) -> pd.DataFrame:
    fname = sym.replace("/", "_") + f"_{tf}.parquet"
    df = pd.read_parquet(CACHE / fname)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


# ─────────────────── parameterized S5 signal ───────────────────

def signal_S5_param(
    t: int,
    df_4h: pd.DataFrame,
    *,
    touches: int = 3,
    range_mult: float = 2.0,
    window: int = 30,
    band_mult: float = 0.3,
    side_filter: str = "both",
):
    if t < max(50, window + 5):
        return None
    r4 = df_4h.iloc[t]
    if pd.isna(r4.atr):
        return None
    win = df_4h.iloc[t - window:t]
    range_high = win.high.quantile(0.95)
    range_low = win.low.quantile(0.05)
    range_size = range_high - range_low
    if range_size < range_mult * r4.atr:
        return None
    band = band_mult * r4.atr
    near_high = r4.high > range_high - band
    near_low = r4.low < range_low + band
    bear_candle = r4.close < r4.open
    bull_candle = r4.close > r4.open
    touches_high = ((win.high > range_high - band) & (win.high <= range_high)).sum()
    touches_low = ((win.low < range_low + band) & (win.low >= range_low)).sum()
    if touches_high >= touches and near_high and bear_candle:
        if side_filter in ("both", "short"):
            return ("short", r4.atr)
    if touches_low >= touches and near_low and bull_candle:
        if side_filter in ("both", "long"):
            return ("long", r4.atr)
    return None


# ─────────────────── trade simulation (mirrors htf_multi_backtest) ───────────────────

@dataclass
class Trade:
    sym: str
    side: str
    entry_idx: int
    entry: float
    sl: float
    atr_at_entry: float
    notional: float
    exit_idx: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    peak_mfe_r: float = 0.0
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    funding_usd: float = 0.0


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
    cap_idx = min(max_idx, trade.entry_idx + cfg.get("time_stop_bars", 168) + 5)
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
        if trade.side == "long":
            if row.low <= trail_sl:
                trade.exit_idx = j
                trade.exit_price = trail_sl
                trade.exit_reason = "trail_sl" if trail_engaged else "sl_hit"
                closed = True
                break
        else:
            if row.high >= trail_sl:
                trade.exit_idx = j
                trade.exit_price = trail_sl
                trade.exit_reason = "trail_sl" if trail_engaged else "sl_hit"
                closed = True
                break
        if "tp_R" in cfg:
            tp_price = trade.entry + cfg["tp_R"] * R if trade.side == "long" else trade.entry - cfg["tp_R"] * R
            if (trade.side == "long" and row.high >= tp_price) or \
               (trade.side == "short" and row.low <= tp_price):
                trade.exit_idx = j
                trade.exit_price = tp_price
                trade.exit_reason = "tp_hit"
                closed = True
                break
        if "trail_engage_R" in cfg and cur_mfe_r >= cfg["trail_engage_R"]:
            lock_r = peak * cfg.get("trail_lock_pct", 0.5)
            new_sl = trade.entry + lock_r * R if trade.side == "long" else trade.entry - lock_r * R
            if (trade.side == "long" and new_sl > trail_sl) or \
               (trade.side == "short" and new_sl < trail_sl):
                trail_sl = new_sl
                trail_engaged = True
        if bars_held >= cfg.get("time_stop_bars", 9999):
            trade.exit_idx = j
            trade.exit_price = row.close
            trade.exit_reason = "time_stop"
            closed = True
            break
    if not closed:
        last_j = min(cap_idx - 1, max_idx - 1)
        last_row = df.iloc[last_j]
        trade.exit_idx = last_j
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
    funding = trade.notional * FUNDING_8H * (bars_held * bar_hours / 8.0)
    trade.gross_pnl_usd = gross
    trade.fees_usd = fees
    trade.funding_usd = funding
    trade.net_pnl_usd = gross - fees - funding
    return trade


# ─────────────────── orchestration ───────────────────

# Pre-load all symbol data once.
def preload_data():
    return {sym: add_indicators(load_candles(sym, "4h")) for sym in SYMBOLS}


def run_variant(
    data: dict,
    *,
    touches: int,
    range_mult: float,
    window: int,
    band_mult: float,
    sl_atr: float,
    exit_cfg: dict,
    side_filter: str,
):
    """Returns (n, wins, gross, net, fees, funding, ev)."""
    n = wins = 0
    gross_sum = net_sum = fees_sum = fund_sum = 0.0
    for sym, df_4h in data.items():
        last_exit_idx = -1
        for t in range(60, len(df_4h)):
            if t <= last_exit_idx:
                continue
            sig = signal_S5_param(
                t, df_4h,
                touches=touches, range_mult=range_mult, window=window,
                band_mult=band_mult, side_filter=side_filter,
            )
            if not sig:
                continue
            side, atr = sig
            if pd.isna(atr) or atr <= 0:
                continue
            entry = df_4h.iloc[t].close
            sl = entry - sl_atr * atr if side == "long" else entry + sl_atr * atr
            tr = Trade(sym=sym, side=side, entry_idx=t, entry=entry, sl=sl,
                       atr_at_entry=atr, notional=NOTIONAL)
            tr = simulate_exit(tr, df_4h, exit_cfg, len(df_4h), 4.0)
            n += 1
            if tr.net_pnl_usd > 0:
                wins += 1
            gross_sum += tr.gross_pnl_usd
            net_sum += tr.net_pnl_usd
            fees_sum += tr.fees_usd
            fund_sum += tr.funding_usd
            last_exit_idx = max(tr.exit_idx, t + 1)
    return n, wins, gross_sum, net_sum, fees_sum, fund_sum


# ─────────────────── variant definitions ───────────────────

BASE = dict(
    touches=3, range_mult=2.0, window=30, band_mult=0.3, sl_atr=1.0,
    exit_cfg={"trail_engage_R": 1.5, "trail_lock_pct": 0.33, "time_stop_bars": 168},
    exit_name="E4_wide(1.5R/33%)",
    side_filter="both",
)

SWEEPS = {
    "touches":    [("touches",    v) for v in [2, 3, 4, 5]],
    "range_mult": [("range_mult", v) for v in [1.5, 2.0, 2.5, 3.0]],
    "window":     [("window",     v) for v in [20, 30, 50, 80]],
    "band_mult":  [("band_mult",  v) for v in [0.2, 0.3, 0.5, 0.7]],
    "sl_atr":     [("sl_atr",     v) for v in [0.7, 1.0, 1.5, 2.0]],
    "side":       [("side_filter", s) for s in ["long", "short", "both"]],
}

EXIT_VARIANTS = [
    ("E4_wide(1.5R/33%)",  {"trail_engage_R": 1.5, "trail_lock_pct": 0.33, "time_stop_bars": 168}),
    ("E5_quick(1.0R/50%)", {"trail_engage_R": 1.0, "trail_lock_pct": 0.50, "time_stop_bars": 168}),
    ("E6_late(2.0R/25%)",  {"trail_engage_R": 2.0, "trail_lock_pct": 0.25, "time_stop_bars": 168}),
    ("E7_balanced(1.5R/50%)", {"trail_engage_R": 1.5, "trail_lock_pct": 0.50, "time_stop_bars": 168}),
]


def variant_label(params: dict) -> str:
    parts = []
    for k in ["touches", "range_mult", "window", "band_mult", "sl_atr"]:
        v = params[k]
        base = BASE[k]
        marker = "*" if v == base else ""
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}{marker}")
        else:
            parts.append(f"{k}={v}{marker}")
    parts.append(f"exit={params['exit_name']}")
    parts.append(f"side={params['side_filter']}")
    return " | ".join(parts)


def main():
    print("=" * 110)
    print("S5 (4h Range Fade) EXTENDED BACKTEST  —  parameter sweep, single-axis variation")
    print(f"Symbols: {SYMBOLS} | 4h cached candles | Fees: 0.059% × 2 + 0.01%/8h funding")
    print("Base config (one * per axis below): touches=3, range_mult=2.0, window=30, band=0.3,")
    print("                                    sl_atr=1.0, exit=E4(1.5R/33%), side=both")
    print("=" * 110)

    data = preload_data()

    rows = []   # (label, axis, value, n, wr, gross, net, fees, funding, ev, params)

    def add_run(axis: str, value, params: dict):
        n, wins, gross, net, fees, fund = run_variant(data, **{
            k: params[k] for k in ["touches", "range_mult", "window",
                                   "band_mult", "sl_atr", "exit_cfg", "side_filter"]
        })
        wr = (100 * wins / n) if n else 0.0
        ev = (net / n) if n else 0.0
        rows.append({
            "axis": axis, "value": str(value),
            "label": variant_label(params),
            "n": n, "wins": wins, "wr": wr,
            "gross": gross, "net": net, "fees": fees, "funding": fund, "ev": ev,
            "params": params,
        })

    # Build base params helper
    def base_params(**override):
        p = dict(
            touches=BASE["touches"], range_mult=BASE["range_mult"],
            window=BASE["window"], band_mult=BASE["band_mult"],
            sl_atr=BASE["sl_atr"],
            exit_cfg=BASE["exit_cfg"], exit_name=BASE["exit_name"],
            side_filter=BASE["side_filter"],
        )
        p.update(override)
        return p

    # 1. base reference
    add_run("BASE", "base", base_params())

    # 2. each single-axis sweep
    for axis_key, variants in SWEEPS.items():
        for key, val in variants:
            params = base_params(**{key: val})
            # skip duplicate of base
            if axis_key == "side" and val == "both":
                continue
            if axis_key != "side" and val == BASE.get(key):
                continue
            add_run(axis_key, val, params)

    # 3. exit config sweep
    for ename, ecfg in EXIT_VARIANTS:
        if ename == BASE["exit_name"]:
            continue
        params = base_params(exit_cfg=ecfg, exit_name=ename)
        add_run("exit", ename, params)

    # ─────────── Top 20 by net ───────────
    rows.sort(key=lambda r: r["net"], reverse=True)
    print(f"\nTOP 20 VARIANTS — sorted by net P&L (* = base value on that axis)")
    print(f"{'rank':>4} {'axis':12} {'value':18} {'n':>4} {'WR%':>6} "
          f"{'gross':>9} {'net':>9} {'fees':>8} {'fund':>7} {'EV$':>7}")
    print("-" * 105)
    for i, r in enumerate(rows[:20], 1):
        print(f"{i:>4} {r['axis']:12} {r['value']:18} {r['n']:>4} {r['wr']:>5.1f}% "
              f"${r['gross']:>8.1f} ${r['net']:>8.1f} ${r['fees']:>7.1f} ${r['funding']:>6.1f} "
              f"${r['ev']:>6.2f}")

    # ─────────── Per-axis sweep tables ───────────
    base_row = next(r for r in rows if r["axis"] == "BASE")
    base_net = base_row["net"]

    print(f"\nBASE REFERENCE: n={base_row['n']}, WR={base_row['wr']:.1f}%, "
          f"gross=${base_row['gross']:.1f}, net=${base_net:.1f}, EV=${base_row['ev']:.2f}/trade")

    print("\nSINGLE-AXIS SWEEPS — impact of each knob (delta vs base net)")
    for axis_key in ["touches", "range_mult", "window", "band_mult", "sl_atr", "side", "exit"]:
        axis_rows = [r for r in rows if r["axis"] == axis_key]
        if not axis_rows:
            continue
        # also include base for reference
        axis_rows = sorted(axis_rows + [base_row], key=lambda r: r["value"])
        print(f"\n  {axis_key.upper()}:")
        print(f"    {'value':18} {'n':>4} {'WR%':>6} {'net':>9} {'EV$':>7} {'Δnet':>9}")
        for r in axis_rows:
            mark = " *" if r is base_row else ""
            d = r["net"] - base_net
            print(f"    {r['value']:18} {r['n']:>4} {r['wr']:>5.1f}% "
                  f"${r['net']:>8.1f} ${r['ev']:>6.2f} ${d:>+8.1f}{mark}")

    # ─────────── Optimal pick & overfitting flag ───────────
    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)

    # Filter to variants with at least 30 trades for stat confidence
    valid = [r for r in rows if r["n"] >= 30]
    if not valid:
        valid = [r for r in rows if r["n"] >= 15]
    valid.sort(key=lambda r: r["net"], reverse=True)
    best = valid[0]
    delta = best["net"] - base_net
    pct_improve = (delta / abs(base_net)) * 100 if base_net != 0 else 0
    print(f"BASE: net=${base_net:.1f}, EV=${base_row['ev']:.2f}/trade, n={base_row['n']}")
    print(f"BEST (n>=15): {best['label']}")
    print(f"      net=${best['net']:.1f}, EV=${best['ev']:.2f}/trade, n={best['n']}, WR={best['wr']:.1f}%")
    print(f"      Δnet vs base: ${delta:+.1f} ({pct_improve:+.1f}%)")
    if delta < 0.15 * abs(base_net) and delta < 200:
        print(f"   ⚠ Improvement marginal (<15% / <$200) — likely noise / overfitting risk.")
        print(f"   → Recommend KEEPING BASE config; treat best variant as not statistically meaningful.")
    elif best["n"] < 30 and base_row["n"] >= 30:
        print(f"   ⚠ Best variant has fewer trades than base — selection bias risk.")
        print(f"   → Treat as candidate, validate on out-of-sample data before deploying.")
    else:
        print(f"   ✓ Improvement looks meaningful. Candidate optimal config:")
        print(f"     {best['label']}")


if __name__ == "__main__":
    main()

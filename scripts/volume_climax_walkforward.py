#!/usr/bin/env python3
"""
Volume Climax — Walk-forward backtest.

Detects extreme volume spike at swing extreme + wick > body candles
(capitulation / euphoria reversals), simulates 4 exit configs per signal,
and walk-forward tests the best (params x exit) per timeframe across
4 quarters: Q1+Q2 in-sample, Q3 OOS, Q4 OOS.

Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive).
Reject if OOS gap > 50%.

Output:
    storage/volume_climax/walkforward.json
    storage/volume_climax/report.md

Read-only on:
    bot/signal_tracker.py, bot/signal_journey.py, bot/signal_learner.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
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
OUT_DIR = ROOT / "storage" / "volume_climax"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
TIMEFRAMES = ["5m", "15m", "1h"]

# Parameter grid (tuned on Q1+Q2 only)
GRID_REL_VOL = [2.0, 3.0, 4.0, 5.0]
GRID_LOOKBACK = [20, 50, 100]
GRID_BODY_RATIO = [0.3, 0.4, 0.5]

# Exit configs
EXITS = {
    "EA": {"tp": 1.5, "sl": 1.0, "time_min": 30, "trail": False},
    "EB": {"tp": 2.0, "sl": 1.0, "time_min": 60, "trail": False},
    "EC": {"tp": 99.0, "sl": 1.0, "time_min": 60, "trail": True,
           "trail_trigger": 0.5, "trail_lock_pct": 0.5},
    "ED": {"tp": 2.5, "sl": 1.0, "time_min": 240, "trail": False},
}

# Trade economics
NOTIONAL = 1000.0
RT_TAKER_FEE_BPS = 0.118  # round-trip Delta India taker, in pct
FUNDING_RATE_PER_8H = 0.0001  # 0.01% / 8h held
ATR_PERIOD = 14
VOL_LOOKBACK = 20

# Walk-forward boundaries
# (Data starts Nov 5 2025 for 5m, Oct 6/7 2025 for 15m/1h.)
# Use spec'd quarters; Q1 5m will simply be shorter (Nov 5 -> Nov 30).
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}


# -----------------------------------------------------------------------------
# Indicator helpers
# -----------------------------------------------------------------------------
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)


# -----------------------------------------------------------------------------
# Climax detection
# -----------------------------------------------------------------------------
def detect_climax(
    df: pd.DataFrame, min_rel_vol: float, lookback: int, body_max: float
) -> pd.DataFrame:
    """
    Returns dataframe with new columns:
        rel_vol, body, rng, body_ratio, lo_wick, up_wick,
        climax_long, climax_short, side ('LONG'|'SHORT'|None)
    """
    df = df.copy()
    vol = df["volume"].astype(float)
    df["vol_avg"] = vol.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean()
    df["rel_vol"] = vol / df["vol_avg"]

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    df["rng"] = (h - l).clip(lower=1e-12)
    df["body"] = (c - o).abs()
    df["body_ratio"] = df["body"] / df["rng"]
    df["lo_wick"] = np.minimum(o, c) - l
    df["up_wick"] = h - np.maximum(o, c)

    # New N-bar extremes (use prior N bars excluding current to avoid leak)
    df["roll_low"] = l.shift(1).rolling(lookback, min_periods=lookback).min()
    df["roll_high"] = h.shift(1).rolling(lookback, min_periods=lookback).max()
    df["new_low"] = l < df["roll_low"]
    df["new_high"] = h > df["roll_high"]

    # Lower-wick > body and recovery (close > open) -> LONG climax
    long_mask = (
        (df["rel_vol"] >= min_rel_vol)
        & df["new_low"]
        & (df["body_ratio"] < body_max)
        & (df["lo_wick"] > df["body"])
        & (c > o)
    )
    short_mask = (
        (df["rel_vol"] >= min_rel_vol)
        & df["new_high"]
        & (df["body_ratio"] < body_max)
        & (df["up_wick"] > df["body"])
        & (c < o)
    )

    df["climax_long"] = long_mask
    df["climax_short"] = short_mask
    df["side"] = np.where(long_mask, "LONG", np.where(short_mask, "SHORT", ""))
    return df


# -----------------------------------------------------------------------------
# Trade simulator
# -----------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr: float,
    exit_cfg: Dict[str, Any],
    tf_min: int,
) -> Optional[Dict[str, Any]]:
    """Simulate one trade from entry_idx (close) using forward bars."""
    if entry_idx + 1 >= len(df):
        return None
    entry = float(df["close"].iloc[entry_idx])
    if not np.isfinite(atr) or atr <= 0:
        return None

    sl_dist = exit_cfg["sl"] * atr
    tp_dist = exit_cfg["tp"] * atr
    bars_max = max(1, int(math.ceil(exit_cfg["time_min"] / tf_min)))

    if side == "LONG":
        sl_p = entry - sl_dist
        tp_p = entry + tp_dist
    else:
        sl_p = entry + sl_dist
        tp_p = entry - tp_dist

    trail_active = False
    trail_stop = None
    mfe_R = 0.0

    end_idx = min(entry_idx + bars_max, len(df) - 1)
    exit_reason = "TIME"
    exit_price = float(df["close"].iloc[end_idx])
    exit_idx = end_idx

    for j in range(entry_idx + 1, end_idx + 1):
        bar_h = float(df["high"].iloc[j])
        bar_l = float(df["low"].iloc[j])

        # Track MFE in R units
        if side == "LONG":
            mfe = (bar_h - entry) / sl_dist
        else:
            mfe = (entry - bar_l) / sl_dist
        if mfe > mfe_R:
            mfe_R = mfe

        # Trailing stop logic (EC)
        if exit_cfg.get("trail"):
            trigger = exit_cfg.get("trail_trigger", 0.5)
            lock_pct = exit_cfg.get("trail_lock_pct", 0.5)
            if not trail_active and mfe_R >= trigger:
                trail_active = True
            if trail_active:
                lock_R = mfe_R * lock_pct
                if side == "LONG":
                    new_stop = entry + lock_R * sl_dist
                    if trail_stop is None or new_stop > trail_stop:
                        trail_stop = new_stop
                else:
                    new_stop = entry - lock_R * sl_dist
                    if trail_stop is None or new_stop < trail_stop:
                        trail_stop = new_stop

        # Stop checks (intra-bar) — assume SL takes priority over TP if both touched
        if side == "LONG":
            hit_sl = bar_l <= sl_p
            hit_tp = bar_h >= tp_p
            hit_trail = trail_stop is not None and bar_l <= trail_stop
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_trail:
                exit_reason = "TRAIL"; exit_price = trail_stop; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break
        else:
            hit_sl = bar_h >= sl_p
            hit_tp = bar_l <= tp_p
            hit_trail = trail_stop is not None and bar_h >= trail_stop
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_trail:
                exit_reason = "TRAIL"; exit_price = trail_stop; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break

    # PnL — return in R units and dollars
    if side == "LONG":
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
        "side": side,
        "exit_reason": exit_reason,
        "entry": entry,
        "exit": exit_price,
        "minutes_held": minutes_held,
        "gross_R": gross_R,
        "net_R": net_R,
        "gross_$": gross_dollars,
        "net_$": net_dollars,
        "mfe_R": mfe_R,
        "atr": atr,
    }


# -----------------------------------------------------------------------------
# Per-cell run
# -----------------------------------------------------------------------------
def run_cell(
    df: pd.DataFrame,
    tf: str,
    min_rel_vol: float,
    lookback: int,
    body_max: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all 4 exit configs against a (TF x params) cell. Returns trades per exit."""
    df_signals = detect_climax(df, min_rel_vol, lookback, body_max)
    atr = add_atr(df_signals)
    tf_min = tf_minutes(tf)

    trades_by_exit: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}

    sig_idx = np.where(df_signals["side"].values != "")[0]
    for i in sig_idx:
        i = int(i)
        # Need ATR available
        atr_v = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else float("nan")
        if not np.isfinite(atr_v):
            continue
        side = df_signals["side"].iloc[i]
        for ek, ec in EXITS.items():
            t = simulate_trade(df_signals, i, side, atr_v, ec, tf_min)
            if t is not None:
                trades_by_exit[ek].append(t)
    return trades_by_exit


def quarter_of(ts: pd.Timestamp) -> Optional[str]:
    for q, (s, e) in QUARTER_BOUNDS.items():
        if pd.Timestamp(s, tz="UTC") <= ts < pd.Timestamp(e, tz="UTC"):
            return q
    return None


def aggregate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "wr": 0.0, "ev_R": 0.0, "ev_$": 0.0,
                "gross_R": 0.0, "net_R": 0.0, "gross_$": 0.0, "net_$": 0.0,
                "avg_mfe_R": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_R"] > 0)
    return {
        "n": n,
        "wr": wins / n,
        "ev_R": float(np.mean([t["net_R"] for t in trades])),
        "ev_$": float(np.mean([t["net_$"] for t in trades])),
        "gross_R": float(np.sum([t["gross_R"] for t in trades])),
        "net_R": float(np.sum([t["net_R"] for t in trades])),
        "gross_$": float(np.sum([t["gross_$"] for t in trades])),
        "net_$": float(np.sum([t["net_$"] for t in trades])),
        "avg_mfe_R": float(np.mean([t["mfe_R"] for t in trades])),
    }


def run_walkforward() -> Dict[str, Any]:
    """Full grid x WF run, picking best params per (TF, exit) on Q1+Q2."""
    all_results: Dict[str, Any] = {"per_cell": [], "walkforward": []}

    for tf in TIMEFRAMES:
        # Pre-load all symbol dfs for this TF
        sym_dfs: Dict[str, pd.DataFrame] = {}
        for sym in SYMBOLS:
            p = CACHE / f"{sym}_USDT_{tf}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index, utc=True)
            sym_dfs[sym] = d

        # For every (params) cell, run all symbols and bucket trades by quarter
        for rel_vol, lookback, body_max in product(
            GRID_REL_VOL, GRID_LOOKBACK, GRID_BODY_RATIO
        ):
            # Per-exit aggregations across all symbols
            per_q_trades: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                ek: {q: [] for q in QUARTER_BOUNDS} for ek in EXITS
            }
            for sym, d in sym_dfs.items():
                trades_by_exit = run_cell(d, tf, rel_vol, lookback, body_max)
                for ek, trades in trades_by_exit.items():
                    for t in trades:
                        q = quarter_of(t["entry_ts"])
                        if q is None:
                            continue
                        per_q_trades[ek][q].append(t)

            for ek in EXITS:
                cell_summary = {
                    "tf": tf,
                    "rel_vol": rel_vol,
                    "lookback": lookback,
                    "body_max": body_max,
                    "exit": ek,
                    "params_key": f"{tf}|rv{rel_vol}|lb{lookback}|br{body_max}|{ek}",
                }
                # IS = Q1 + Q2
                is_trades = per_q_trades[ek]["Q1"] + per_q_trades[ek]["Q2"]
                cell_summary["IS"] = aggregate(is_trades)
                cell_summary["Q3"] = aggregate(per_q_trades[ek]["Q3"])
                cell_summary["Q4"] = aggregate(per_q_trades[ek]["Q4"])
                all_results["per_cell"].append(cell_summary)

    # Walk-forward selection: per (tf, exit), choose best IS (n>=20, max ev_R)
    best_per_tf_exit: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cell in all_results["per_cell"]:
        key = (cell["tf"], cell["exit"])
        if cell["IS"]["n"] < 20:
            continue
        cur = best_per_tf_exit.get(key)
        if cur is None or cell["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            best_per_tf_exit[key] = cell

    # WF verdicts
    verdicts: List[Dict[str, Any]] = []
    for (tf, ek), cell in sorted(best_per_tf_exit.items()):
        is_ev = cell["IS"]["ev_R"]
        q3_ev = cell["Q3"]["ev_R"]
        q4_ev = cell["Q4"]["ev_R"]

        def gap_pct(oos: float, isv: float) -> float:
            if isv == 0 or not np.isfinite(isv):
                return float("inf")
            return (isv - oos) / abs(isv) * 100.0

        q3_gap = gap_pct(q3_ev, is_ev)
        q4_gap = gap_pct(q4_ev, is_ev)

        # Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive)
        def passes(oos: float) -> bool:
            return is_ev > 0 and oos > 0 and oos >= 0.5 * is_ev

        q3_pass = passes(q3_ev)
        q4_pass = passes(q4_ev)

        verdict = "SHIP" if q3_pass and q4_pass else (
            "HOLD" if q3_pass or q4_pass else "KILL"
        )

        verdicts.append({
            "tf": tf,
            "exit": ek,
            "params": {
                "rel_vol": cell["rel_vol"],
                "lookback": cell["lookback"],
                "body_max": cell["body_max"],
            },
            "IS": cell["IS"],
            "Q3": cell["Q3"],
            "Q4": cell["Q4"],
            "Q3_gap_pct": q3_gap,
            "Q4_gap_pct": q4_gap,
            "Q3_pass": q3_pass,
            "Q4_pass": q4_pass,
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

    # Trim per_cell to only non-empty IS for the json output
    payload = {
        "config": {
            "symbols": SYMBOLS,
            "timeframes": TIMEFRAMES,
            "grid_rel_vol": GRID_REL_VOL,
            "grid_lookback": GRID_LOOKBACK,
            "grid_body_ratio": GRID_BODY_RATIO,
            "exits": EXITS,
            "quarter_bounds": QUARTER_BOUNDS,
            "fee_bps": RT_TAKER_FEE_BPS,
            "funding_per_8h": FUNDING_RATE_PER_8H,
            "notional": NOTIONAL,
        },
        "per_cell": results["per_cell"],
        "walkforward": results["walkforward"],
    }
    out_json.write_text(json.dumps(payload, default=str, indent=2))

    # Markdown report
    lines: List[str] = []
    lines.append("# Volume Climax — Walk-forward Backtest")
    lines.append("")
    lines.append(
        "Detector: rel_vol >= MIN_REL_VOL on a candle that prints a new N-bar "
        "extreme, with wick > body and recovery (LONG: close>open + lower wick;"
        " SHORT: close<open + upper wick)."
    )
    lines.append("")
    lines.append("Quarter splits (UTC):")
    for q, (s, e) in QUARTER_BOUNDS.items():
        lines.append(f"- {q}: {s} -> {e}")
    lines.append("")
    lines.append("Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive).")
    lines.append("Reject if OOS gap > 50%.")
    lines.append("")
    lines.append("## Walk-forward verdicts (best params per TF x exit)")
    lines.append("")
    lines.append(
        "| TF | Exit | rel_vol | lookback | body<= | IS n | IS EV(R) | Q3 n |"
        " Q3 EV(R) | Q4 n | Q4 EV(R) | Q3 gap% | Q4 gap% | Verdict |"
    )
    lines.append(
        "|----|------|---------|----------|--------|------|----------|------|"
        "----------|------|----------|---------|---------|---------|"
    )
    for v in results["walkforward"]:
        p = v["params"]
        lines.append(
            f"| {v['tf']} | {v['exit']} | {p['rel_vol']} | {p['lookback']} | "
            f"{p['body_max']} | {v['IS']['n']} | {v['IS']['ev_R']:+.3f} | "
            f"{v['Q3']['n']} | {v['Q3']['ev_R']:+.3f} | {v['Q4']['n']} | "
            f"{v['Q4']['ev_R']:+.3f} | {v['Q3_gap_pct']:+.1f}% | "
            f"{v['Q4_gap_pct']:+.1f}% | {v['verdict']} |"
        )
    lines.append("")

    # Per-TF best summary (single best across exits)
    lines.append("## Per-TF best combo (highest IS EV, n>=20)")
    lines.append("")
    by_tf: Dict[str, Dict[str, Any]] = {}
    for v in results["walkforward"]:
        prev = by_tf.get(v["tf"])
        if prev is None or v["IS"]["ev_R"] > prev["IS"]["ev_R"]:
            by_tf[v["tf"]] = v
    for tf, v in by_tf.items():
        p = v["params"]
        lines.append(
            f"- **{tf}**: best is exit {v['exit']} @ rv={p['rel_vol']}, lb="
            f"{p['lookback']}, body<={p['body_max']} -> "
            f"IS EV {v['IS']['ev_R']:+.3f}R (n={v['IS']['n']}), "
            f"Q3 {v['Q3']['ev_R']:+.3f}R (n={v['Q3']['n']}), "
            f"Q4 {v['Q4']['ev_R']:+.3f}R (n={v['Q4']['n']}), "
            f"verdict **{v['verdict']}**."
        )
    lines.append("")

    # Notes
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- 5m data spans Nov 5 2025 -> Apr 17 2026; 15m/1h cover Oct 6 2025 ->"
        " Apr 17 2026. Q1 5m sample is therefore truncated."
    )
    lines.append(
        "- Climax detector is regime-blind by design: only price/volume math, no"
        " HTF context, no scanner overlay."
    )
    lines.append(
        "- Single-bar entry at signal close, intra-bar SL priority over TP if "
        "both touched (conservative)."
    )
    lines.append(
        "- Fees: 0.118% RT taker (Delta India). Funding: 0.01%/8h held."
    )
    lines.append(
        "- Walk-forward gap > 50% of IS or sign flip = KILL."
    )
    out_md.write_text("\n".join(lines))


def main() -> None:
    print("[VolumeClimax] Running walk-forward backtest...")
    results = run_walkforward()
    write_report(results)
    print(f"[VolumeClimax] Wrote: {OUT_DIR/'walkforward.json'}")
    print(f"[VolumeClimax] Wrote: {OUT_DIR/'report.md'}")
    # Brief stdout digest
    print("\nWalk-forward verdicts:")
    for v in results["walkforward"]:
        p = v["params"]
        print(
            f"  {v['tf']}/{v['exit']} rv={p['rel_vol']} lb={p['lookback']} "
            f"br={p['body_max']}: IS={v['IS']['ev_R']:+.3f}R(n={v['IS']['n']}) "
            f"Q3={v['Q3']['ev_R']:+.3f}R Q4={v['Q4']['ev_R']:+.3f}R -> {v['verdict']}"
        )


if __name__ == "__main__":
    main()

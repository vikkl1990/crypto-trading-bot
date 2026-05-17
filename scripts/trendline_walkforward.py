#!/usr/bin/env python3
"""
Trendline auto-detection - Walk-forward backtest.

Detects two distinct trendline-based signals (BREAK and BOUNCE), simulates
4 exit configurations per signal, and walk-forward tests the best
(params x exit) per (signal_type x TF) across 4 quarters:
    Q1+Q2 in-sample, Q3 OOS, Q4 OOS.

Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive) on BOTH OOS
quarters. HOLD if only one OOS passes; KILL if neither.

Output:
    storage/trendline/walkforward.json
    storage/trendline/report.md
    storage/trendline/wedge_overlap.json    (overlap with VETO 10k WEDGE_BREAKOUT_VETO)

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
OUT_DIR = ROOT / "storage" / "trendline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Signal type -> TF list (per spec)
SIGNAL_TFS = {
    "trendline_break": ["15m", "1h", "4h"],
    "trendline_bounce": ["15m", "1h", "4h"],
}

# Parameter grid (tuned on Q1+Q2 only)
GRID_PIVOT_LB = [5, 8, 10]                  # fractal lookback for swing detection
GRID_DETECT_WINDOW = [30, 50, 80]           # bars to consider for trendline pivots
GRID_MIN_PIVOTS = [2, 3]                    # minimum pivots required to fit a line
GRID_R2_THRESHOLD = [0.6, 0.7, 0.8]         # linear regression fit quality
GRID_BREAK_ATR = [0.3, 0.5, 0.7]            # ATR multiple for break confirmation
GRID_BOUNCE_ATR = [0.2, 0.3, 0.5]           # ATR multiple for bounce tolerance

# Four exit configs per spec
EXITS = {
    "EA": {"tp": 1.5, "sl": 1.0, "time_min": 60, "trail": False},
    "EB": {"tp": 2.0, "sl": 1.0, "time_min": 4 * 60, "trail": False},
    "EC": {
        "tp": 99.0,
        "sl": 1.0,
        "time_min": 6 * 60,
        "trail": True,
        "trail_trigger": 1.0,
        "trail_lock_pct": 0.5,
    },
    "ED": {"tp": 2.5, "sl": 1.0, "time_min": 12 * 60, "trail": False},
}

# Trade economics
NOTIONAL = 1000.0
RT_TAKER_FEE_BPS = 0.118  # round-trip Delta India taker (in pct of notional)
FUNDING_RATE_PER_8H = 0.0001
ATR_PERIOD = 14

# Walk-forward boundaries (UTC)
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)


def find_pivots(highs: np.ndarray, lows: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Strict-fractal pivot detection: pivot at i confirmed only if it's the
    unique extreme within +/- lookback bars (so it cannot be re-confirmed in
    the future at the same height)."""
    n = len(highs)
    ph: List[int] = []
    pl: List[int] = []
    for i in range(lookback, n - lookback):
        h_win = highs[i - lookback : i + lookback + 1]
        l_win = lows[i - lookback : i + lookback + 1]
        if highs[i] == h_win.max() and (h_win == highs[i]).sum() == 1:
            ph.append(i)
        if lows[i] == l_win.min() and (l_win == lows[i]).sum() == 1:
            pl.append(i)
    return np.array(ph, dtype=int), np.array(pl, dtype=int)


def fit_line(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, intercept, r2). r2=1 perfect fit, r2=0 no linear relationship."""
    if len(xs) < 2:
        return 0.0, 0.0, 0.0
    xm = xs.mean()
    ym = ys.mean()
    sxx = float(((xs - xm) ** 2).sum())
    sxy = float(((xs - xm) * (ys - ym)).sum())
    if sxx <= 0:
        return 0.0, ys[-1], 0.0
    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = slope * xs + intercept
    ss_res = float(((ys - yhat) ** 2).sum())
    ss_tot = float(((ys - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(max(0.0, min(1.0, r2)))


# -----------------------------------------------------------------------------
# Trendline detection (look-ahead-safe).
#
# At each candle i, we build trendlines using ONLY pivots whose lookback
# confirmation is in the past (j + pivot_lb < i), i.e. the pivot was confirmed
# pivot_lb bars before i. This guarantees no future leakage.
#
# A bullish trendline (support) connects swing LOWS with positive slope.
# A bearish trendline (resistance) connects swing HIGHS with negative slope.
# Both require r2 >= r2_threshold.
# -----------------------------------------------------------------------------
def detect_trendline_signals(
    df: pd.DataFrame,
    atr: pd.Series,
    signal_type: str,
    pivot_lb: int,
    detect_window: int,
    min_pivots: int,
    r2_threshold: float,
    break_atr: float,
    bounce_atr: float,
) -> List[Dict[str, Any]]:
    """Walk every candle i, attempt to fit a trendline and check break/bounce.

    Returns list of {entry_idx, side, atr, line_value (debug), trendline_type}.
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    opens = df["open"].astype(float).values
    n = len(df)
    if n < detect_window + pivot_lb + 5:
        return []

    # Pre-compute all pivots (strict fractals).
    ph_all, pl_all = find_pivots(highs, lows, pivot_lb)

    # cooldown so we don't double-fire same trendline at consecutive bars
    cooldown_bars = max(2 * pivot_lb, 5)
    last_fire_idx = -10**9

    out: List[Dict[str, Any]] = []
    for i in range(detect_window + pivot_lb + 1, n):
        if i - last_fire_idx < cooldown_bars:
            continue
        atr_v = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else float("nan")
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue

        # Pivots confirmed *before* i (lookahead-safe): pivot j is confirmed
        # at j + pivot_lb < i, so we keep j with j + pivot_lb < i.
        confirm_cutoff = i - pivot_lb - 1
        # window bounds for trendline fit (recent only)
        lo = max(0, i - detect_window)
        ph_recent = ph_all[(ph_all >= lo) & (ph_all <= confirm_cutoff)]
        pl_recent = pl_all[(pl_all >= lo) & (pl_all <= confirm_cutoff)]

        # ---- Bullish trendline (support, positive slope on lows)
        if len(pl_recent) >= min_pivots:
            xs = pl_recent.astype(float)
            ys = lows[pl_recent]
            slope, intercept, r2 = fit_line(xs, ys)
            if slope > 0 and r2 >= r2_threshold:
                line_at_i = slope * i + intercept
                if signal_type == "trendline_break":
                    # Bearish break: close drops below the support line by >= break_atr * ATR
                    delta = line_at_i - closes[i]
                    if delta >= break_atr * atr_v and closes[i] < line_at_i:
                        # Entry on close of bar i, side SHORT (reversal)
                        out.append({
                            "entry_idx": i,
                            "side": "SHORT",
                            "atr": atr_v,
                            "line": float(line_at_i),
                            "trendline": "bullish",
                            "slope": slope,
                            "r2": r2,
                            "n_pivots": int(len(pl_recent)),
                        })
                        last_fire_idx = i
                        continue
                elif signal_type == "trendline_bounce":
                    # Bullish bounce: low touches within bounce_atr * ATR of line,
                    # and close > open AND close > previous close.
                    if i >= 1:
                        touch_dist = abs(lows[i] - line_at_i)
                        if (
                            touch_dist <= bounce_atr * atr_v
                            and closes[i] > opens[i]
                            and closes[i] > closes[i - 1]
                            and lows[i] <= line_at_i + bounce_atr * atr_v
                            and lows[i] >= line_at_i - bounce_atr * atr_v
                        ):
                            out.append({
                                "entry_idx": i,
                                "side": "LONG",
                                "atr": atr_v,
                                "line": float(line_at_i),
                                "trendline": "bullish",
                                "slope": slope,
                                "r2": r2,
                                "n_pivots": int(len(pl_recent)),
                            })
                            last_fire_idx = i
                            continue

        # ---- Bearish trendline (resistance, negative slope on highs)
        if len(ph_recent) >= min_pivots:
            xs = ph_recent.astype(float)
            ys = highs[ph_recent]
            slope, intercept, r2 = fit_line(xs, ys)
            if slope < 0 and r2 >= r2_threshold:
                line_at_i = slope * i + intercept
                if signal_type == "trendline_break":
                    # Bullish break: close rises above the resistance line
                    delta = closes[i] - line_at_i
                    if delta >= break_atr * atr_v and closes[i] > line_at_i:
                        out.append({
                            "entry_idx": i,
                            "side": "LONG",
                            "atr": atr_v,
                            "line": float(line_at_i),
                            "trendline": "bearish",
                            "slope": slope,
                            "r2": r2,
                            "n_pivots": int(len(ph_recent)),
                        })
                        last_fire_idx = i
                        continue
                elif signal_type == "trendline_bounce":
                    # Bearish bounce: high touches within bounce_atr * ATR of line,
                    # and close < open AND close < previous close.
                    if i >= 1:
                        touch_dist = abs(highs[i] - line_at_i)
                        if (
                            touch_dist <= bounce_atr * atr_v
                            and closes[i] < opens[i]
                            and closes[i] < closes[i - 1]
                            and highs[i] >= line_at_i - bounce_atr * atr_v
                            and highs[i] <= line_at_i + bounce_atr * atr_v
                        ):
                            out.append({
                                "entry_idx": i,
                                "side": "SHORT",
                                "atr": atr_v,
                                "line": float(line_at_i),
                                "trendline": "bearish",
                                "slope": slope,
                                "r2": r2,
                                "n_pivots": int(len(ph_recent)),
                            })
                            last_fire_idx = i
                            continue

    return out


# -----------------------------------------------------------------------------
# Trade simulator (R-based, bar-by-bar with optional trail)
# -----------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr: float,
    exit_cfg: Dict[str, Any],
    tf_min: int,
) -> Optional[Dict[str, Any]]:
    if entry_idx >= len(df) - 1:
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

        if side == "LONG":
            mfe = (bar_h - entry) / sl_dist
        else:
            mfe = (entry - bar_l) / sl_dist
        if mfe > mfe_R:
            mfe_R = mfe

        if exit_cfg.get("trail"):
            trigger = exit_cfg.get("trail_trigger", 1.0)
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


# -----------------------------------------------------------------------------
# Per-cell run
# -----------------------------------------------------------------------------
def run_cell(
    df: pd.DataFrame,
    tf: str,
    signal_type: str,
    pivot_lb: int,
    detect_window: int,
    min_pivots: int,
    r2_threshold: float,
    break_atr: float,
    bounce_atr: float,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Run all 4 exit configs against one (signal x TF x params) cell on one symbol.

    Returns (trades_by_exit, raw_signals).
    """
    atr = add_atr(df)
    tf_min = tf_minutes(tf)

    sigs = detect_trendline_signals(
        df, atr,
        signal_type=signal_type,
        pivot_lb=pivot_lb,
        detect_window=detect_window,
        min_pivots=min_pivots,
        r2_threshold=r2_threshold,
        break_atr=break_atr,
        bounce_atr=bounce_atr,
    )

    trades_by_exit: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXITS}
    for s in sigs:
        for ek, ec in EXITS.items():
            t = simulate_trade(df, s["entry_idx"], s["side"], s["atr"], ec, tf_min)
            if t is not None:
                trades_by_exit[ek].append(t)
    return trades_by_exit, sigs


# -----------------------------------------------------------------------------
# Walk-forward: parameter selection on Q1+Q2, evaluation on Q3 / Q4
# -----------------------------------------------------------------------------
def run_walkforward() -> Dict[str, Any]:
    all_results: Dict[str, Any] = {"per_cell": [], "walkforward": []}

    # Cache loaded dfs to avoid re-reading parquet for each grid point.
    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for tf in {tf for tfs in SIGNAL_TFS.values() for tf in tfs}:
        for sym in SYMBOLS:
            p = CACHE / f"{sym}_USDT_{tf}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index, utc=True)
            df_cache[(sym, tf)] = d

    for signal_type, tfs in SIGNAL_TFS.items():
        for tf in tfs:
            # ATR is added once per (sym, tf) inside run_cell; that's fine.
            sym_dfs = {sym: df_cache[(sym, tf)] for sym in SYMBOLS if (sym, tf) in df_cache}

            # Iterate the param grid. The grid here is the cross-product of
            # six dimensions, which is large but tractable. To keep runtime
            # reasonable we drop bounce_atr from break-only grids and break_atr
            # from bounce-only grids.
            if signal_type == "trendline_break":
                grid = [
                    (plb, dw, mp, r2, batr, 0.3)  # bounce_atr is unused for break
                    for plb, dw, mp, r2, batr in product(
                        GRID_PIVOT_LB, GRID_DETECT_WINDOW, GRID_MIN_PIVOTS,
                        GRID_R2_THRESHOLD, GRID_BREAK_ATR,
                    )
                ]
            else:
                grid = [
                    (plb, dw, mp, r2, 0.5, batr)  # break_atr is unused for bounce
                    for plb, dw, mp, r2, batr in product(
                        GRID_PIVOT_LB, GRID_DETECT_WINDOW, GRID_MIN_PIVOTS,
                        GRID_R2_THRESHOLD, GRID_BOUNCE_ATR,
                    )
                ]

            for (pivot_lb, detect_window, min_pivots, r2_threshold,
                 break_atr, bounce_atr) in grid:
                per_q_trades: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                    ek: {q: [] for q in QUARTER_BOUNDS} for ek in EXITS
                }
                total_signals = 0
                # We also collect signals per-symbol for the wedge-overlap check
                # at the WF best params (we'll re-derive at report time).
                for sym, d in sym_dfs.items():
                    trades_by_exit, sigs = run_cell(
                        d, tf, signal_type,
                        pivot_lb, detect_window, min_pivots, r2_threshold,
                        break_atr, bounce_atr,
                    )
                    total_signals += len(sigs)
                    for ek, trades in trades_by_exit.items():
                        for t in trades:
                            q = quarter_of(t["entry_ts"])
                            if q is None:
                                continue
                            per_q_trades[ek][q].append(t)

                for ek in EXITS:
                    cell_summary = {
                        "signal_type": signal_type,
                        "tf": tf,
                        "pivot_lb": pivot_lb,
                        "detect_window": detect_window,
                        "min_pivots": min_pivots,
                        "r2_threshold": r2_threshold,
                        "break_atr": break_atr,
                        "bounce_atr": bounce_atr,
                        "exit": ek,
                        "params_key": (
                            f"{signal_type}|{tf}|plb{pivot_lb}|dw{detect_window}"
                            f"|mp{min_pivots}|r2{r2_threshold}|brk{break_atr}"
                            f"|bnc{bounce_atr}|{ek}"
                        ),
                        "n_signals_total": total_signals,
                    }
                    is_trades = per_q_trades[ek]["Q1"] + per_q_trades[ek]["Q2"]
                    cell_summary["IS"] = aggregate(is_trades)
                    cell_summary["Q3"] = aggregate(per_q_trades[ek]["Q3"])
                    cell_summary["Q4"] = aggregate(per_q_trades[ek]["Q4"])
                    all_results["per_cell"].append(cell_summary)

    # Walk-forward selection: per (signal_type, tf, exit), pick best IS
    # (n>=20, max ev_R) — same gate as classical_patterns_walkforward.
    best_per: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for cell in all_results["per_cell"]:
        key = (cell["signal_type"], cell["tf"], cell["exit"])
        if cell["IS"]["n"] < 20:
            continue
        cur = best_per.get(key)
        if cur is None or cell["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            best_per[key] = cell

    verdicts: List[Dict[str, Any]] = []
    for (signal_type, tf, ek), cell in sorted(best_per.items()):
        is_ev = cell["IS"]["ev_R"]
        q3_ev = cell["Q3"]["ev_R"]
        q4_ev = cell["Q4"]["ev_R"]

        def gap_pct(oos: float, isv: float) -> float:
            if isv == 0 or not np.isfinite(isv):
                return float("inf")
            return (isv - oos) / abs(isv) * 100.0

        q3_gap = gap_pct(q3_ev, is_ev)
        q4_gap = gap_pct(q4_ev, is_ev)

        def passes(oos: float) -> bool:
            return is_ev > 0 and oos > 0 and oos >= 0.5 * is_ev

        q3_pass = passes(q3_ev)
        q4_pass = passes(q4_ev)
        verdict = "SHIP" if q3_pass and q4_pass else (
            "HOLD" if q3_pass or q4_pass else "KILL"
        )

        verdicts.append({
            "signal_type": signal_type,
            "tf": tf,
            "exit": ek,
            "params": {
                "pivot_lb": cell["pivot_lb"],
                "detect_window": cell["detect_window"],
                "min_pivots": cell["min_pivots"],
                "r2_threshold": cell["r2_threshold"],
                "break_atr": cell["break_atr"],
                "bounce_atr": cell["bounce_atr"],
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
# Wedge overlap probe — for each best (signal_type x TF) verdict cell, compute
# the % of trendline signals that ALSO fire as a fresh wedge breakout (using
# the shipped scripts/wedge_detector.py). We only bother with TFs at which
# the wedge detector ships (it currently runs against 1h htf_df by default).
# -----------------------------------------------------------------------------
def compute_wedge_overlap(verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.wedge_detector import detect_wedge_breakout  # type: ignore
    except Exception as e:
        return {"error": f"wedge_detector import failed: {e}"}

    overlap_rows: List[Dict[str, Any]] = []
    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    for v in verdicts:
        sig_type = v["signal_type"]
        tf = v["tf"]
        params = v["params"]
        sym_dfs: Dict[str, pd.DataFrame] = {}
        for sym in SYMBOLS:
            key = (sym, tf)
            if key in df_cache:
                sym_dfs[sym] = df_cache[key]
                continue
            p = CACHE / f"{sym}_USDT_{tf}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index, utc=True)
            df_cache[key] = d
            sym_dfs[sym] = d

        n_total = 0
        n_overlap = 0
        per_sym: Dict[str, Dict[str, int]] = {}

        for sym, d in sym_dfs.items():
            atr = add_atr(d)
            sigs = detect_trendline_signals(
                d, atr, sig_type,
                pivot_lb=params["pivot_lb"],
                detect_window=params["detect_window"],
                min_pivots=params["min_pivots"],
                r2_threshold=params["r2_threshold"],
                break_atr=params["break_atr"],
                bounce_atr=params["bounce_atr"],
            )
            sym_total = 0
            sym_overlap = 0
            for s in sigs:
                idx = s["entry_idx"]
                # Build a slice ending at the entry bar inclusive (live bot
                # behaviour: detect_wedge_breakout examines df.iloc[-1]).
                if idx < 30:
                    continue
                slice_ = d.iloc[: idx + 1]
                try:
                    w = detect_wedge_breakout(slice_)
                except Exception:
                    w = None
                sym_total += 1
                if w is not None:
                    # We count any wedge breakout as overlap (regardless of
                    # direction). The opposing-direction overlap is what the
                    # bot uses to veto, but for "redundancy" the question is
                    # whether wedge fires at the same bar at all.
                    sym_overlap += 1
            n_total += sym_total
            n_overlap += sym_overlap
            per_sym[sym] = {"total": sym_total, "overlap": sym_overlap}

        pct = (100.0 * n_overlap / n_total) if n_total > 0 else 0.0
        overlap_rows.append({
            "signal_type": sig_type,
            "tf": tf,
            "exit": v["exit"],
            "verdict": v["verdict"],
            "n_total": n_total,
            "n_overlap": n_overlap,
            "overlap_pct": pct,
            "per_symbol": per_sym,
        })

    return {"rows": overlap_rows}


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def write_report(results: Dict[str, Any], overlap: Dict[str, Any]) -> None:
    out_json = OUT_DIR / "walkforward.json"
    out_md = OUT_DIR / "report.md"
    out_overlap = OUT_DIR / "wedge_overlap.json"

    payload = {
        "config": {
            "symbols": SYMBOLS,
            "signal_tfs": SIGNAL_TFS,
            "grid_pivot_lb": GRID_PIVOT_LB,
            "grid_detect_window": GRID_DETECT_WINDOW,
            "grid_min_pivots": GRID_MIN_PIVOTS,
            "grid_r2_threshold": GRID_R2_THRESHOLD,
            "grid_break_atr": GRID_BREAK_ATR,
            "grid_bounce_atr": GRID_BOUNCE_ATR,
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
    out_overlap.write_text(json.dumps(overlap, default=str, indent=2))

    overlap_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {
        (r["signal_type"], r["tf"], r["exit"]): r for r in overlap.get("rows", [])
    }

    lines: List[str] = []
    lines.append("# Trendline Auto-Detection - Walk-forward Backtest")
    lines.append("")
    lines.append("Detects two distinct trendline-based signals (BREAK / BOUNCE) and")
    lines.append("walk-forward tests the best (params x exit) per (signal x TF).")
    lines.append("")
    lines.append("Signal TFs:")
    for s, tfs in SIGNAL_TFS.items():
        lines.append(f"- {s}: TF={tfs}")
    lines.append("")
    lines.append("Quarter splits (UTC):")
    for q, (sd, ed) in QUARTER_BOUNDS.items():
        lines.append(f"- {q}: {sd} -> {ed}")
    lines.append("")
    lines.append("Pass criteria: Q3 EV >= 50% of IS EV AND Q4 EV >= 50% of IS EV "
                 "AND all positive.")
    lines.append("Reject if either OOS gap > 50%.")
    lines.append("")
    lines.append("Cost model: Delta India taker 0.118% RT + funding 0.01%/8h.")
    lines.append("")

    lines.append("## Walk-forward verdicts (best params per signal x TF x exit)")
    lines.append("")
    lines.append(
        "| Signal | TF | Exit | plb | dw | mp | r2 | brk | bnc | IS n | IS EV(R) | "
        "Q3 n | Q3 EV(R) | Q4 n | Q4 EV(R) | Q3 gap% | Q4 gap% | Verdict |"
    )
    lines.append(
        "|--------|----|------|-----|----|----|----|-----|-----|------|----------|"
        "------|----------|------|----------|---------|---------|---------|"
    )
    for v in results["walkforward"]:
        lines.append(
            "| {sig} | {tf} | {exit} | {plb} | {dw} | {mp} | {r2} | {brk} | {bnc} | "
            "{isn} | {isev:+.3f} | {q3n} | {q3ev:+.3f} | {q4n} | {q4ev:+.3f} | "
            "{q3gap:+.1f} | {q4gap:+.1f} | {verdict} |".format(
                sig=v["signal_type"],
                tf=v["tf"],
                exit=v["exit"],
                plb=v["params"]["pivot_lb"],
                dw=v["params"]["detect_window"],
                mp=v["params"]["min_pivots"],
                r2=v["params"]["r2_threshold"],
                brk=v["params"]["break_atr"],
                bnc=v["params"]["bounce_atr"],
                isn=v["IS"]["n"],
                isev=v["IS"]["ev_R"],
                q3n=v["Q3"]["n"],
                q3ev=v["Q3"]["ev_R"],
                q4n=v["Q4"]["n"],
                q4ev=v["Q4"]["ev_R"],
                q3gap=v["Q3_gap_pct"],
                q4gap=v["Q4_gap_pct"],
                verdict=v["verdict"],
            )
        )
    lines.append("")

    # Best-per (signal_type x TF) summary
    best_per_sig_tf: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for v in results["walkforward"]:
        key = (v["signal_type"], v["tf"])
        cur = best_per_sig_tf.get(key)
        if cur is None or v["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            best_per_sig_tf[key] = v

    lines.append("## Signal x TF summary (best exit)")
    lines.append("")
    lines.append(
        "| Signal | TF | Best Exit | IS n | IS EV(R) | Q3 n | Q3 EV(R) | Q4 n | "
        "Q4 EV(R) | Verdict |"
    )
    lines.append(
        "|--------|----|-----------|------|----------|------|----------|------|"
        "----------|---------|"
    )
    for (sig, tf), v in sorted(best_per_sig_tf.items()):
        lines.append(
            "| {sig} | {tf} | {exit} | {isn} | {isev:+.3f} | {q3n} | {q3ev:+.3f} | "
            "{q4n} | {q4ev:+.3f} | {verdict} |".format(
                sig=sig,
                tf=tf,
                exit=v["exit"],
                isn=v["IS"]["n"],
                isev=v["IS"]["ev_R"],
                q3n=v["Q3"]["n"],
                q3ev=v["Q3"]["ev_R"],
                q4n=v["Q4"]["n"],
                q4ev=v["Q4"]["ev_R"],
                verdict=v["verdict"],
            )
        )
    lines.append("")

    # Verdict counts
    ships = [v for v in results["walkforward"] if v["verdict"] == "SHIP"]
    holds = [v for v in results["walkforward"] if v["verdict"] == "HOLD"]
    kills = [v for v in results["walkforward"] if v["verdict"] == "KILL"]
    lines.append("## Verdict counts")
    lines.append("")
    lines.append(f"- SHIP: {len(ships)}")
    lines.append(f"- HOLD: {len(holds)}")
    lines.append(f"- KILL: {len(kills)}")
    lines.append("")

    # Wedge overlap
    lines.append("## Wedge overlap (vs shipped VETO 10k WEDGE_BREAKOUT_VETO)")
    lines.append("")
    lines.append("Question: of the trendline signals fired by the WF best cells, "
                 "what % also fire as a fresh wedge breakout at the same bar?")
    lines.append("")
    lines.append(
        "| Signal | TF | Exit | Verdict | n trendline | n overlap (wedge fires) | overlap % |"
    )
    lines.append(
        "|--------|----|------|---------|-------------|-------------------------|-----------|"
    )
    for v in results["walkforward"]:
        key = (v["signal_type"], v["tf"], v["exit"])
        ov = overlap_by_key.get(key)
        if ov is None:
            continue
        lines.append(
            "| {sig} | {tf} | {ek} | {verdict} | {nt} | {no} | {pct:.1f}% |".format(
                sig=v["signal_type"], tf=v["tf"], ek=v["exit"],
                verdict=v["verdict"], nt=ov["n_total"], no=ov["n_overlap"],
                pct=ov["overlap_pct"],
            )
        )
    lines.append("")

    if ships:
        lines.append("## SHIP candidates (passed both OOS quarters)")
        lines.append("")
        for v in ships:
            key = (v["signal_type"], v["tf"], v["exit"])
            ov = overlap_by_key.get(key)
            ov_str = (
                f"; wedge overlap {ov['overlap_pct']:.1f}% (n={ov['n_total']})"
                if ov else ""
            )
            lines.append(
                f"- **{v['signal_type']} {v['tf']} {v['exit']}**: IS EV "
                f"{v['IS']['ev_R']:+.3f}R (n={v['IS']['n']}), Q3 "
                f"{v['Q3']['ev_R']:+.3f}R (n={v['Q3']['n']}), Q4 "
                f"{v['Q4']['ev_R']:+.3f}R (n={v['Q4']['n']}){ov_str}"
            )
        lines.append("")

    out_md.write_text("\n".join(lines))


def main() -> int:
    print("Running trendline walk-forward...", flush=True)
    results = run_walkforward()
    print(f"per_cell={len(results['per_cell'])}  walkforward={len(results['walkforward'])}", flush=True)
    print("Computing wedge overlap for WF cells...", flush=True)
    overlap = compute_wedge_overlap(results["walkforward"])
    write_report(results, overlap)
    print(f"Wrote {OUT_DIR / 'walkforward.json'}", flush=True)
    print(f"Wrote {OUT_DIR / 'report.md'}", flush=True)
    print(f"Wrote {OUT_DIR / 'wedge_overlap.json'}", flush=True)
    ships = [v for v in results["walkforward"] if v["verdict"] == "SHIP"]
    holds = [v for v in results["walkforward"] if v["verdict"] == "HOLD"]
    kills = [v for v in results["walkforward"] if v["verdict"] == "KILL"]
    print(f"SHIP={len(ships)} HOLD={len(holds)} KILL={len(kills)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

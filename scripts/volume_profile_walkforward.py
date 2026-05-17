#!/usr/bin/env python3
"""
Volume Profile (POC / VAH / VAL) - Walk-forward backtest.

Builds session-based volume profiles (24h / 48h / 7d windows) at each candle
close, then tests 4 signal types as standalone scanners:

  Type 1 - HVN rejection: price approaches POC, rejects with reversal candle
  Type 2 - VAL bounce / VAH rejection: price tags VAL -> LONG; tags VAH -> SHORT
  Type 3 - VAH break: close > VAH with vol > 1.2x avg -> LONG continuation
            (mirror: close < VAL with vol > 1.2x -> SHORT)
  Type 4 - POC magnet: price drifts toward POC from outside the value area
            (long when price below VAL trending up; short when above VAH trending down)

Walk-forward across 4 quarters: Q1+Q2 in-sample, Q3 OOS, Q4 OOS.
Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive). KILL otherwise.

Output:
    storage/volume_profile/walkforward.json
    storage/volume_profile/report.md

Read-only on bot/* and paper engines.
"""
from __future__ import annotations

import json
import math
import sys
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
OUT_DIR = ROOT / "storage" / "volume_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
TIMEFRAMES = ["15m", "1h"]

# Profile windows (in HOURS for cross-TF compatibility)
PROFILE_WINDOWS_HOURS = [24, 48, 24 * 7]  # 24h, 48h, 7d
PROFILE_BINS = [50, 100]
VALUE_AREA_PCT = 0.70

# Type-3 break volume threshold
BREAK_VOL_THRESHOLD = 1.2

# Type-1 HVN rejection: how close to POC counts as "approach"
HVN_PROXIMITY_ATR = 0.5  # within 0.5 ATR of POC

# Type-2 VAL/VAH proximity tolerance
VA_TOL_ATR = 0.3

# Reversal candle: body sign + body_ratio > min
REVERSAL_BODY_RATIO_MIN = 0.4

# Exit configs (matches spec)
EXITS = {
    "EA": {"tp": 1.5, "sl": 1.0, "time_min": 30, "trail": False},
    "EB": {"tp": 2.0, "sl": 1.0, "time_min": 60, "trail": False},
    "EC": {
        "tp": 99.0,
        "sl": 1.0,
        "time_min": 4 * 60,
        "trail": True,
        "trail_trigger": 0.5,
        "trail_lock_pct": 0.5,
    },
    "ED": {"tp": 2.5, "sl": 1.0, "time_min": 4 * 60, "trail": False},
}

# Trade economics
NOTIONAL = 1000.0
RT_TAKER_FEE_BPS = 0.118  # round-trip Delta India taker (in pct of notional)
FUNDING_RATE_PER_8H = 0.0001
ATR_PERIOD = 14
VOL_LOOKBACK = 20

# Walk-forward boundaries (UTC) per spec
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}

# Min trades for IS to be considered
MIN_IS_TRADES = 20


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)


def quarter_of(ts: pd.Timestamp) -> Optional[str]:
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    for q, (s, e) in QUARTER_BOUNDS.items():
        if pd.Timestamp(s, tz="UTC") <= ts < pd.Timestamp(e, tz="UTC"):
            return q
    return None


# -----------------------------------------------------------------------------
# Volume profile core
# -----------------------------------------------------------------------------
def compute_profile(
    highs: np.ndarray, lows: np.ndarray, vols: np.ndarray, bins: int, va_pct: float
) -> Optional[Tuple[float, float, float, float]]:
    """Return (POC, VAH, VAL, total_vol) for a window of OHLC bars.

    Each bar's volume is distributed uniformly across its [low, high] range
    into the bin grid. POC = price midpoint of the highest-volume bin.
    Value area: expanding outward from POC bin until va_pct of total volume
    is captured. VAH = top of last bin in VA, VAL = bottom of first bin in VA.
    """
    if len(highs) == 0:
        return None
    p_min = float(np.min(lows))
    p_max = float(np.max(highs))
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return None
    edges = np.linspace(p_min, p_max, bins + 1)
    bin_w = (p_max - p_min) / bins
    if bin_w <= 0:
        return None
    counts = np.zeros(bins, dtype=float)
    total = 0.0
    for h, l, v in zip(highs, lows, vols):
        if not np.isfinite(v) or v <= 0:
            continue
        if h <= l:
            # Single bin
            idx = int(np.clip((h - p_min) / bin_w, 0, bins - 1))
            counts[idx] += v
            total += v
            continue
        # Distribute volume uniformly across the bar's range
        lo_idx = int(np.clip(np.floor((l - p_min) / bin_w), 0, bins - 1))
        hi_idx = int(np.clip(np.ceil((h - p_min) / bin_w) - 1, 0, bins - 1))
        if hi_idx < lo_idx:
            hi_idx = lo_idx
        n_bins = hi_idx - lo_idx + 1
        per_bin = v / n_bins
        counts[lo_idx : hi_idx + 1] += per_bin
        total += v
    if total <= 0:
        return None
    poc_idx = int(np.argmax(counts))
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)
    # Value area expansion: start at POC, expand to neighbour with higher vol
    target = va_pct * total
    cum = counts[poc_idx]
    lo = poc_idx
    hi = poc_idx
    while cum < target and (lo > 0 or hi < bins - 1):
        left_vol = counts[lo - 1] if lo > 0 else -1.0
        right_vol = counts[hi + 1] if hi < bins - 1 else -1.0
        if left_vol >= right_vol:
            if lo > 0:
                lo -= 1
                cum += counts[lo]
            else:
                hi += 1
                cum += counts[hi]
        else:
            if hi < bins - 1:
                hi += 1
                cum += counts[hi]
            else:
                lo -= 1
                cum += counts[lo]
    val = float(edges[lo])
    vah = float(edges[hi + 1])
    return poc, vah, val, total


def build_profiles_for_df(
    df: pd.DataFrame, tf: str, window_hours: int, bins: int, va_pct: float
) -> pd.DataFrame:
    """Add columns POC, VAH, VAL to df by computing rolling profile.

    Window is window_hours of past bars (excluding the current bar).
    """
    tfm = tf_minutes(tf)
    bars = max(1, int(round(window_hours * 60 / tfm)))
    n = len(df)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    vols = df["volume"].astype(float).values
    for i in range(bars, n):
        s = i - bars
        e = i  # exclusive of current bar (no leak)
        out = compute_profile(highs[s:e], lows[s:e], vols[s:e], bins, va_pct)
        if out is None:
            continue
        poc[i], vah[i], val[i], _ = out
    out_df = df.copy()
    out_df["poc"] = poc
    out_df["vah"] = vah
    out_df["val"] = val
    return out_df


# -----------------------------------------------------------------------------
# Signal detection per type
# -----------------------------------------------------------------------------
def detect_type1_hvn_rejection(
    df: pd.DataFrame, atr: pd.Series
) -> List[Tuple[int, str]]:
    """Type 1: price approaches POC, then rejects.

    LONG: bar.low <= POC + tol AND close > open AND body_ratio >= min
    SHORT: bar.high >= POC - tol AND close < open AND body_ratio >= min
    Counter-trend: rejection drives price away from POC.
    """
    sigs: List[Tuple[int, str]] = []
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    poc = df["poc"].values
    rng = h - l
    body = np.abs(c - o)
    body_ratio = np.where(rng > 0, body / rng, 0.0)
    n = len(df)
    for i in range(1, n):
        if not np.isfinite(poc[i]):
            continue
        a = atr.iat[i]
        if not np.isfinite(a) or a <= 0:
            continue
        tol = HVN_PROXIMITY_ATR * a
        # LONG rejection: came down to POC, closed back up
        if l[i] <= poc[i] + tol and c[i] > o[i] and body_ratio[i] >= REVERSAL_BODY_RATIO_MIN:
            # And the prior bar was approaching from above (close > poc previously)
            if c[i - 1] >= poc[i] - tol:
                sigs.append((i, "LONG"))
                continue
        if h[i] >= poc[i] - tol and c[i] < o[i] and body_ratio[i] >= REVERSAL_BODY_RATIO_MIN:
            if c[i - 1] <= poc[i] + tol:
                sigs.append((i, "SHORT"))
    return sigs


def detect_type2_va_reaction(
    df: pd.DataFrame, atr: pd.Series
) -> List[Tuple[int, str]]:
    """Type 2: VAL bounce -> LONG; VAH rejection -> SHORT.

    LONG: bar.low <= VAL + tol AND close > VAL AND close > open
    SHORT: bar.high >= VAH - tol AND close < VAH AND close < open
    """
    sigs: List[Tuple[int, str]] = []
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    vah = df["vah"].values
    val = df["val"].values
    n = len(df)
    for i in range(n):
        a = atr.iat[i]
        if not np.isfinite(a) or a <= 0:
            continue
        tol = VA_TOL_ATR * a
        if np.isfinite(val[i]):
            if l[i] <= val[i] + tol and c[i] > val[i] and c[i] > o[i]:
                sigs.append((i, "LONG"))
                continue
        if np.isfinite(vah[i]):
            if h[i] >= vah[i] - tol and c[i] < vah[i] and c[i] < o[i]:
                sigs.append((i, "SHORT"))
    return sigs


def detect_type3_va_break(df: pd.DataFrame) -> List[Tuple[int, str]]:
    """Type 3: close > VAH with vol >= threshold -> LONG; close < VAL -> SHORT."""
    sigs: List[Tuple[int, str]] = []
    c = df["close"].astype(float).values
    o = df["open"].astype(float).values
    v = df["volume"].astype(float).values
    vah = df["vah"].values
    val = df["val"].values
    # Volume rolling avg
    v_avg = pd.Series(v).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean().values
    n = len(df)
    for i in range(1, n):
        if not np.isfinite(v_avg[i]) or v_avg[i] <= 0:
            continue
        rel_v = v[i] / v_avg[i]
        if rel_v < BREAK_VOL_THRESHOLD:
            continue
        # LONG break above VAH
        if np.isfinite(vah[i]):
            # Confirm: prior close was inside VA; this close above
            prior_inside = (
                np.isfinite(vah[i - 1])
                and np.isfinite(val[i - 1])
                and val[i - 1] <= c[i - 1] <= vah[i - 1]
            )
            if c[i] > vah[i] and prior_inside and c[i] > o[i]:
                sigs.append((i, "LONG"))
                continue
        if np.isfinite(val[i]):
            prior_inside = (
                np.isfinite(vah[i - 1])
                and np.isfinite(val[i - 1])
                and val[i - 1] <= c[i - 1] <= vah[i - 1]
            )
            if c[i] < val[i] and prior_inside and c[i] < o[i]:
                sigs.append((i, "SHORT"))
    return sigs


def detect_type4_poc_magnet(
    df: pd.DataFrame, atr: pd.Series
) -> List[Tuple[int, str]]:
    """Type 4: price drifts toward POC from outside the value area.

    LONG: close < VAL (below value area) AND close > prior close (drifting up)
          AND POC > current close (POC is a target above)
    SHORT: close > VAH AND close < prior close AND POC < current close
    Mean-reversion expectation toward POC.
    """
    sigs: List[Tuple[int, str]] = []
    c = df["close"].astype(float).values
    poc = df["poc"].values
    vah = df["vah"].values
    val = df["val"].values
    n = len(df)
    for i in range(2, n):
        if not (np.isfinite(poc[i]) and np.isfinite(vah[i]) and np.isfinite(val[i])):
            continue
        a = atr.iat[i]
        if not np.isfinite(a) or a <= 0:
            continue
        # LONG magnet: below VA, drifting up
        if c[i] < val[i] and c[i] > c[i - 1] and c[i - 1] > c[i - 2] and poc[i] > c[i]:
            # And POC is at least 0.5 ATR above current close (room to run)
            if (poc[i] - c[i]) >= 0.5 * a:
                sigs.append((i, "LONG"))
                continue
        if c[i] > vah[i] and c[i] < c[i - 1] and c[i - 1] < c[i - 2] and poc[i] < c[i]:
            if (c[i] - poc[i]) >= 0.5 * a:
                sigs.append((i, "SHORT"))
    return sigs


# -----------------------------------------------------------------------------
# Trade simulator (matches volume_climax_walkforward.py shape)
# -----------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr: float,
    exit_cfg: Dict[str, Any],
    tf_min: int,
) -> Optional[Dict[str, Any]]:
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
        if side == "LONG":
            mfe = (bar_h - entry) / sl_dist
        else:
            mfe = (entry - bar_l) / sl_dist
        if mfe > mfe_R:
            mfe_R = mfe
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


def aggregate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "wr": 0.0, "ev_R": 0.0, "ev_$": 0.0,
                "net_R": 0.0, "net_$": 0.0, "avg_mfe_R": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_R"] > 0)
    return {
        "n": n,
        "wr": wins / n,
        "ev_R": float(np.mean([t["net_R"] for t in trades])),
        "ev_$": float(np.mean([t["net_$"] for t in trades])),
        "net_R": float(np.sum([t["net_R"] for t in trades])),
        "net_$": float(np.sum([t["net_$"] for t in trades])),
        "avg_mfe_R": float(np.mean([t["mfe_R"] for t in trades])),
    }


SIGNAL_TYPES = {
    "T1_hvn_rejection": detect_type1_hvn_rejection,
    "T2_va_reaction": detect_type2_va_reaction,
    "T3_va_break": detect_type3_va_break,
    "T4_poc_magnet": detect_type4_poc_magnet,
}


def detect_signals(
    df: pd.DataFrame, atr: pd.Series, sig_type: str
) -> List[Tuple[int, str]]:
    fn = SIGNAL_TYPES[sig_type]
    if sig_type == "T3_va_break":
        return fn(df)
    return fn(df, atr)


# -----------------------------------------------------------------------------
# Walk-forward driver
# -----------------------------------------------------------------------------
def run_walkforward() -> Dict[str, Any]:
    all_results: Dict[str, Any] = {"per_cell": [], "walkforward": []}

    for tf in TIMEFRAMES:
        sym_dfs: Dict[str, pd.DataFrame] = {}
        for sym in SYMBOLS:
            p = CACHE / f"{sym}_USDT_{tf}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index, utc=True)
            sym_dfs[sym] = d

        for window_h, bins in product(PROFILE_WINDOWS_HOURS, PROFILE_BINS):
            # Pre-build profile for each sym
            profiled: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
            for sym, d in sym_dfs.items():
                pf = build_profiles_for_df(d, tf, window_h, bins, VALUE_AREA_PCT)
                a = add_atr(pf)
                profiled[sym] = (pf, a)

            for sig_type in SIGNAL_TYPES:
                # Aggregate across syms
                per_q_trades: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                    ek: {q: [] for q in QUARTER_BOUNDS} for ek in EXITS
                }
                for sym, (pf, atr) in profiled.items():
                    sigs = detect_signals(pf, atr, sig_type)
                    tfm = tf_minutes(tf)
                    for idx, side in sigs:
                        a = float(atr.iat[idx]) if not pd.isna(atr.iat[idx]) else float("nan")
                        if not np.isfinite(a) or a <= 0:
                            continue
                        for ek, ec in EXITS.items():
                            t = simulate_trade(pf, idx, side, a, ec, tfm)
                            if t is None:
                                continue
                            q = quarter_of(t["entry_ts"])
                            if q is None:
                                continue
                            per_q_trades[ek][q].append(t)

                for ek in EXITS:
                    is_trades = per_q_trades[ek]["Q1"] + per_q_trades[ek]["Q2"]
                    cell = {
                        "tf": tf,
                        "window_h": window_h,
                        "bins": bins,
                        "sig_type": sig_type,
                        "exit": ek,
                        "params_key": f"{tf}|w{window_h}|b{bins}|{sig_type}|{ek}",
                        "IS": aggregate(is_trades),
                        "Q3": aggregate(per_q_trades[ek]["Q3"]),
                        "Q4": aggregate(per_q_trades[ek]["Q4"]),
                    }
                    all_results["per_cell"].append(cell)

    # Walk-forward: per (tf, sig_type, exit) pick best IS (n>=MIN_IS_TRADES, max ev_R)
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for cell in all_results["per_cell"]:
        if cell["IS"]["n"] < MIN_IS_TRADES:
            continue
        key = (cell["tf"], cell["sig_type"], cell["exit"])
        cur = best.get(key)
        if cur is None or cell["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            best[key] = cell

    verdicts: List[Dict[str, Any]] = []
    for (tf, st, ek), cell in sorted(best.items()):
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
        verdict = "SHIP" if q3_pass and q4_pass else ("HOLD" if q3_pass or q4_pass else "KILL")

        verdicts.append({
            "tf": tf,
            "sig_type": st,
            "exit": ek,
            "params": {"window_h": cell["window_h"], "bins": cell["bins"]},
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

    payload = {
        "config": {
            "symbols": SYMBOLS,
            "timeframes": TIMEFRAMES,
            "profile_windows_hours": PROFILE_WINDOWS_HOURS,
            "profile_bins": PROFILE_BINS,
            "value_area_pct": VALUE_AREA_PCT,
            "exits": EXITS,
            "quarter_bounds": QUARTER_BOUNDS,
            "fee_bps": RT_TAKER_FEE_BPS,
            "funding_per_8h": FUNDING_RATE_PER_8H,
            "notional": NOTIONAL,
            "min_is_trades": MIN_IS_TRADES,
        },
        "per_cell": results["per_cell"],
        "walkforward": results["walkforward"],
    }
    out_json.write_text(json.dumps(payload, default=str, indent=2))

    lines: List[str] = []
    lines.append("# Volume Profile (POC/VAH/VAL) - Walk-forward Backtest")
    lines.append("")
    lines.append("Four signal types tested as standalone scanners:")
    lines.append("- T1 HVN rejection: price approaches POC, prints reversal candle")
    lines.append("- T2 VA reaction: VAL bounce (LONG) / VAH rejection (SHORT)")
    lines.append("- T3 VA break: close beyond VAH/VAL with vol > 1.2x avg (continuation)")
    lines.append("- T4 POC magnet: price outside VA drifting toward POC (mean-reversion)")
    lines.append("")
    lines.append("Quarter splits (UTC):")
    for q, (s, e) in QUARTER_BOUNDS.items():
        lines.append(f"- {q}: {s} -> {e}")
    lines.append("")
    lines.append("Pass criteria: OOS EV >= 50% of IS EV AND same sign (positive). Reject otherwise.")
    lines.append("")
    lines.append("## Walk-forward verdicts (best params per TF x sig_type x exit, n>=20)")
    lines.append("")
    lines.append(
        "| TF | SigType | Exit | window_h | bins | IS n | IS EV(R) | Q3 n | Q3 EV(R) | Q4 n | Q4 EV(R) | Q3 gap% | Q4 gap% | Verdict |"
    )
    lines.append(
        "|----|---------|------|----------|------|------|----------|------|----------|------|----------|---------|---------|---------|"
    )
    for v in results["walkforward"]:
        p = v["params"]
        lines.append(
            f"| {v['tf']} | {v['sig_type']} | {v['exit']} | {p['window_h']} | {p['bins']} | "
            f"{v['IS']['n']} | {v['IS']['ev_R']:+.3f} | "
            f"{v['Q3']['n']} | {v['Q3']['ev_R']:+.3f} | {v['Q4']['n']} | "
            f"{v['Q4']['ev_R']:+.3f} | {v['Q3_gap_pct']:+.1f}% | "
            f"{v['Q4_gap_pct']:+.1f}% | {v['verdict']} |"
        )
    lines.append("")

    # Per-type summary
    lines.append("## Per signal-type best combo")
    lines.append("")
    by_type: Dict[str, Dict[str, Any]] = {}
    for v in results["walkforward"]:
        cur = by_type.get(v["sig_type"])
        if cur is None or v["IS"]["ev_R"] > cur["IS"]["ev_R"]:
            by_type[v["sig_type"]] = v
    for st in SIGNAL_TYPES:
        v = by_type.get(st)
        if v is None:
            lines.append(f"- **{st}**: 0 cells with IS n>=20.")
            continue
        p = v["params"]
        lines.append(
            f"- **{st}**: best {v['tf']}/{v['exit']} window={p['window_h']}h bins={p['bins']} -> "
            f"IS {v['IS']['ev_R']:+.3f}R (n={v['IS']['n']}), "
            f"Q3 {v['Q3']['ev_R']:+.3f}R (n={v['Q3']['n']}), "
            f"Q4 {v['Q4']['ev_R']:+.3f}R (n={v['Q4']['n']}) -> **{v['verdict']}**"
        )
    lines.append("")
    n_ship = sum(1 for v in results["walkforward"] if v["verdict"] == "SHIP")
    n_hold = sum(1 for v in results["walkforward"] if v["verdict"] == "HOLD")
    n_kill = sum(1 for v in results["walkforward"] if v["verdict"] == "KILL")
    lines.append(f"Totals: {len(results['walkforward'])} cells with IS n>=20 - "
                 f"SHIP={n_ship} HOLD={n_hold} KILL={n_kill}")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- 5m data starts Nov 5 2025; 15m/1h start Oct 6 2025. Q1 sample for 5m would be truncated (5m not tested here).")
    lines.append("- Profile windows of 7d on 15m use ~672 bars and are slow to compute; 24h/48h give better statistical density.")
    lines.append("- Conservative intra-bar fill: SL has priority over TP if both touched same bar.")
    lines.append("- Fees 0.118% RT taker (Delta India). Funding 0.01%/8h.")
    lines.append("- No regime / scanner overlay; pure price-vs-profile scanner.")
    out_md.write_text("\n".join(lines))


def main() -> None:
    print("[VolumeProfile] Running walk-forward backtest...")
    results = run_walkforward()
    write_report(results)
    print(f"[VolumeProfile] Wrote: {OUT_DIR/'walkforward.json'}")
    print(f"[VolumeProfile] Wrote: {OUT_DIR/'report.md'}")
    print("\nWalk-forward verdicts:")
    for v in results["walkforward"]:
        p = v["params"]
        print(
            f"  {v['tf']}/{v['sig_type']}/{v['exit']} w={p['window_h']}h b={p['bins']}: "
            f"IS={v['IS']['ev_R']:+.3f}R(n={v['IS']['n']}) "
            f"Q3={v['Q3']['ev_R']:+.3f}R Q4={v['Q4']['ev_R']:+.3f}R -> {v['verdict']}"
        )


if __name__ == "__main__":
    main()

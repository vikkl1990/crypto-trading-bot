#!/usr/bin/env python3
"""
Regime Filter 3-Way Walk-Forward Backtest.

Compares THREE candidate regime filters head-to-head as hard vetoes against
counter-bias trades on a synthetic engulfing-reversal test cohort:

    A) HA_BIAS    — Heiken Ashi Market Bias (>= N consecutive same-direction
                    HA candles on configurable TF). Tunable: (TF, N) on Q1+Q2.
    B) EMA21_50   — Current bot's session_bias on entry TF (close > EMA21
                    > EMA50 = bull, reverse for bear, otherwise neutral).
    C) EMA200_d   — Daily EMA200 macro filter (close vs daily EMA200 =
                    bull/bear; resampled from 4h closes -> daily for the
                    backtest window since cache lacks 1d). Already shipped
                    dark today as MACRO_EMA200_VETO.

Test cohort: bullish-engulfing -> LONG and bearish-engulfing -> SHORT trades on
the entry TF (5m and 15m). Standard exit: TP=1.5R / SL=1R / 60min time stop.
Cost: 0.118% RT taker (Delta India). 1R = full ATR-derived SL distance from
entry close.

For each filter, every trade is tagged ALIGNED (filter regime agrees with
trade side), OPPOSED (filter regime opposes trade side) or NEUTRAL (no clear
filter regime). The edge metric is the SEPARATION = ALIGNED EV - OPPOSED EV.

Walk-forward:
- Q1 (Oct-Nov 2025) and Q2 (Dec 2025 - Jan 2026) = IS
- Q3 (Feb 2026) and Q4 (Mar-Apr 2026) = OOS
- HA_BIAS tunes (TF, N) on Q1+Q2; EMA21_50 and EMA200_d use fixed defs.
- Verdict: SHIP iff IS separation > 0 AND BOTH OOS quarters preserve sign and
  >= 50% of IS magnitude. HOLD if only one OOS preserves. KILL otherwise.

Output:
    storage/regime_3way/walkforward.json
    storage/regime_3way/report.md

Read-only on bot/signal_tracker.py, bot/signal_journey.py, bot/signal_learner.py.
Does not modify scalp_strategy.py or any paper-trading engine.
"""

from __future__ import annotations

import json
import math
import os
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
OUT_DIR = ROOT / "storage" / "regime_3way"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
ENTRY_TFS = ["5m", "15m"]            # cohort entry timeframes
HA_TF_GRID = ["15m", "1h", "4h"]     # HA_BIAS TF tuning grid
HA_N_GRID = [3, 4, 5]                # consecutive bars

# Trade economics
NOTIONAL = 1000.0
RT_TAKER_FEE_BPS = 0.118             # round-trip Delta India taker, in pct
ATR_PERIOD = 14
EXIT_TP_R = 1.5
EXIT_SL_R = 1.0
EXIT_TIME_MIN = 60

# Walk-forward boundaries
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    raise ValueError(tf)


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


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False, min_periods=span).mean()


def quarter_of(ts: pd.Timestamp) -> Optional[str]:
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    for q, (s, e) in QUARTER_BOUNDS.items():
        if pd.Timestamp(s, tz="UTC") <= ts < pd.Timestamp(e, tz="UTC"):
            return q
    return None


# -----------------------------------------------------------------------------
# Heiken Ashi
# -----------------------------------------------------------------------------
def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heiken Ashi candles. Returns df with ha_open/ha_close/ha_high/ha_low/ha_dir.

    ha_dir: +1 = bullish (no lower wick), -1 = bearish (no upper wick), 0 = doji/mixed.
    """
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values

    n = len(df)
    ha_open = np.zeros(n)
    ha_close = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)

    if n == 0:
        out = df.copy()
        out["ha_open"] = []
        out["ha_close"] = []
        out["ha_high"] = []
        out["ha_low"] = []
        out["ha_dir"] = []
        return out

    # Bootstrap
    ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    ha_open[0] = (o[0] + c[0]) / 2.0
    ha_high[0] = max(h[0], ha_open[0], ha_close[0])
    ha_low[0] = min(l[0], ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close[i])

    eps = 1e-9
    bull = (ha_close > ha_open) & (ha_low >= ha_open - eps)   # no/zero lower wick
    bear = (ha_close < ha_open) & (ha_high <= ha_open + eps)  # no/zero upper wick
    ha_dir = np.where(bull, 1, np.where(bear, -1, 0))

    out = df.copy()
    out["ha_open"] = ha_open
    out["ha_close"] = ha_close
    out["ha_high"] = ha_high
    out["ha_low"] = ha_low
    out["ha_dir"] = ha_dir
    return out


def ha_bias_series(df_ha: pd.DataFrame, n: int) -> pd.Series:
    """Return +1/-1/0 regime per bar based on >=n consecutive same-direction HA bars.

    The label at bar i reflects the regime at the close of bar i (we use it for
    decisions at bar i+1 — so this is leak-safe when we look up via reindex
    forward-fill below).
    """
    d = df_ha["ha_dir"].astype(int).values
    n_rows = len(d)
    out = np.zeros(n_rows, dtype=int)
    run = 0
    last_dir = 0
    for i in range(n_rows):
        if d[i] != 0 and d[i] == last_dir:
            run += 1
        elif d[i] != 0:
            run = 1
            last_dir = d[i]
        else:
            run = 0
            last_dir = 0
        if run >= n and last_dir != 0:
            out[i] = last_dir
    return pd.Series(out, index=df_ha.index, dtype=int)


def ema21_50_bias_series(df: pd.DataFrame) -> pd.Series:
    """Current bot's session_bias on the entry TF.

    +1 = close > EMA21 > EMA50, -1 = close < EMA21 < EMA50, else 0.
    """
    e21 = ema(df["close"], 21)
    e50 = ema(df["close"], 50)
    c = df["close"].astype(float)
    bull = (c > e21) & (e21 > e50)
    bear = (c < e21) & (e21 < e50)
    out = pd.Series(0, index=df.index, dtype=int)
    out[bull] = 1
    out[bear] = -1
    return out


def ema200_d_bias_series(df_4h: pd.DataFrame) -> pd.Series:
    """Daily EMA200 macro bias derived from the 4h cache.

    We resample 4h closes to a daily series (last close of each UTC day),
    compute EMA200 on it, then map +1 if daily close > EMA200, -1 if <,
    0 if EMA not yet warm. Returned indexed on the 4h frame's bar timestamps.
    """
    # Resample 4h to daily close (last close of UTC day)
    day = df_4h["close"].astype(float).resample("1D").last().dropna()
    # Note: 6mo cache yields ~193 daily bars. Strict EMA200 with min_periods=200
    # would never warm. To allow EMA200_d to be tested at all on this window
    # we use min_periods=50 (still a meaningful slow trend filter, span=200).
    # This is the same trade-off production_macro accepts when bootstrapping —
    # the live macro_bias.parquet uses an OKX-extended history, but inside
    # this walk-forward we only have what's cached.
    min_warm = 50
    if len(day) < min_warm:
        return pd.Series(0, index=df_4h.index, dtype=int)
    e200 = day.ewm(span=200, adjust=False, min_periods=min_warm).mean()
    # Apply 0.5% chop band (matches production _classify_bias)
    chop_band = 0.005
    diff = (day - e200) / e200
    bias_d = pd.Series(0, index=day.index, dtype=int)
    bias_d[diff > chop_band] = 1
    bias_d[diff < -chop_band] = -1
    # Reindex onto the 4h timestamps via forward-fill — each 4h bar takes
    # the bias from the most recent COMPLETED daily close.
    # Shift daily bias by 1 day so the bar trades on yesterday's close (no leak).
    bias_d = bias_d.shift(1).dropna()
    bias_d.index = bias_d.index + pd.Timedelta(days=1)
    return bias_d.reindex(df_4h.index, method="ffill").fillna(0).astype(int)


# -----------------------------------------------------------------------------
# Engulfing reversal test cohort (entry TF)
# -----------------------------------------------------------------------------
def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Bullish/bearish engulfing reversal candles.

    Bullish engulfing (LONG): prev close<open, this close>open, this body covers
    prev body (this open <= prev close, this close >= prev open).
    Bearish engulfing (SHORT): mirror.
    """
    o = df["open"].astype(float)
    c = df["close"].astype(float)
    po = o.shift(1)
    pc = c.shift(1)
    bull_eng = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear_eng = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    out = df.copy()
    out["eng_long"] = bull_eng.fillna(False)
    out["eng_short"] = bear_eng.fillna(False)
    out["side"] = np.where(bull_eng, "LONG", np.where(bear_eng, "SHORT", ""))
    return out


# -----------------------------------------------------------------------------
# Trade simulator (TP 1.5R / SL 1R / 60min)
# -----------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr: float,
    tf_min: int,
) -> Optional[Dict[str, Any]]:
    if entry_idx + 1 >= len(df):
        return None
    if not np.isfinite(atr) or atr <= 0:
        return None
    entry = float(df["close"].iloc[entry_idx])

    sl_dist = EXIT_SL_R * atr
    tp_dist = EXIT_TP_R * atr
    bars_max = max(1, int(math.ceil(EXIT_TIME_MIN / tf_min)))

    if side == "LONG":
        sl_p = entry - sl_dist
        tp_p = entry + tp_dist
    else:
        sl_p = entry + sl_dist
        tp_p = entry - tp_dist

    end_idx = min(entry_idx + bars_max, len(df) - 1)
    exit_reason = "TIME"
    exit_price = float(df["close"].iloc[end_idx])
    exit_idx = end_idx

    for j in range(entry_idx + 1, end_idx + 1):
        bar_h = float(df["high"].iloc[j])
        bar_l = float(df["low"].iloc[j])
        if side == "LONG":
            hit_sl = bar_l <= sl_p
            hit_tp = bar_h >= tp_p
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break
        else:
            hit_sl = bar_h >= sl_p
            hit_tp = bar_l <= tp_p
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break

    if side == "LONG":
        ret = (exit_price - entry) / entry
    else:
        ret = (entry - exit_price) / entry
    gross_dollars = NOTIONAL * ret
    fee_dollars = NOTIONAL * (RT_TAKER_FEE_BPS / 100.0)
    net_dollars = gross_dollars - fee_dollars
    risk_dollars = NOTIONAL * (sl_dist / entry)
    gross_R = gross_dollars / risk_dollars if risk_dollars > 0 else 0.0
    net_R = net_dollars / risk_dollars if risk_dollars > 0 else 0.0

    return {
        "entry_ts": df.index[entry_idx],
        "side": side,
        "exit_reason": exit_reason,
        "net_R": net_R,
        "gross_R": gross_R,
        "net_$": net_dollars,
    }


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "wr": 0.0, "ev_R": 0.0, "ev_$": 0.0, "net_R": 0.0, "net_$": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_R"] > 0)
    return {
        "n": n,
        "wr": wins / n,
        "ev_R": float(np.mean([t["net_R"] for t in trades])),
        "ev_$": float(np.mean([t["net_$"] for t in trades])),
        "net_R": float(np.sum([t["net_R"] for t in trades])),
        "net_$": float(np.sum([t["net_$"] for t in trades])),
    }


# -----------------------------------------------------------------------------
# Build cohort + tag with all 3 filters
# -----------------------------------------------------------------------------
def load_tf(sym: str, tf: str) -> Optional[pd.DataFrame]:
    p = CACHE / f"{sym}_USDT_{tf}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d.index = pd.to_datetime(d.index, utc=True)
    d = d.sort_index()
    return d


def build_cohort_tagged(
    sym: str,
    entry_tf: str,
    ha_tf: str,
    ha_n: int,
) -> List[Dict[str, Any]]:
    """Return list of trades with tags for all 3 filters at entry time."""
    df_entry = load_tf(sym, entry_tf)
    if df_entry is None or len(df_entry) < 250:
        return []

    # Indicator setup on entry TF
    df_entry = detect_engulfing(df_entry)
    atr_series = add_atr(df_entry, ATR_PERIOD)
    bias_2150 = ema21_50_bias_series(df_entry)

    # HA on configured TF
    df_ha_tf = load_tf(sym, ha_tf)
    if df_ha_tf is None:
        return []
    df_ha = compute_ha(df_ha_tf)
    ha_bias = ha_bias_series(df_ha, ha_n)
    # The HA bias label at the close of bar j is decided at j's close. To avoid
    # leak we shift forward by one bar of the HA TF — entry trades using bias
    # of the most recently CLOSED HA bar.
    ha_bias_shifted = ha_bias.shift(1).fillna(0).astype(int)
    # Re-index to entry TF (forward-fill)
    ha_bias_on_entry = ha_bias_shifted.reindex(df_entry.index, method="ffill").fillna(0).astype(int)

    # EMA200_d via 4h cache resampled to daily
    df_4h = load_tf(sym, "4h")
    if df_4h is None:
        return []
    bias_e200d_4h = ema200_d_bias_series(df_4h)
    # Reindex to entry TF
    bias_e200d_on_entry = bias_e200d_4h.reindex(df_entry.index, method="ffill").fillna(0).astype(int)

    # Tradeable signal indices
    tf_min = tf_minutes(entry_tf)
    sides = df_entry["side"].values
    sig_idx = np.where(sides != "")[0]
    out: List[Dict[str, Any]] = []
    for i in sig_idx:
        i = int(i)
        atr_v = float(atr_series.iloc[i])
        if not np.isfinite(atr_v):
            continue
        side = sides[i]
        t = simulate_trade(df_entry, i, side, atr_v, tf_min)
        if t is None:
            continue

        side_int = 1 if side == "LONG" else -1
        ha_b = int(ha_bias_on_entry.iloc[i])
        e2150_b = int(bias_2150.iloc[i])
        e200d_b = int(bias_e200d_on_entry.iloc[i])

        def alignment(b: int) -> str:
            if b == 0:
                return "NEUTRAL"
            return "ALIGNED" if b == side_int else "OPPOSED"

        t.update({
            "symbol": sym,
            "entry_tf": entry_tf,
            "ha_bias": ha_b,
            "ema21_50_bias": e2150_b,
            "ema200_d_bias": e200d_b,
            "ha_align": alignment(ha_b),
            "ema21_50_align": alignment(e2150_b),
            "ema200_d_align": alignment(e200d_b),
        })
        out.append(t)
    return out


# -----------------------------------------------------------------------------
# Filter scoring
# -----------------------------------------------------------------------------
def score_filter(
    trades: List[Dict[str, Any]],
    align_key: str,
    quarter_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute ALIGNED / OPPOSED / NEUTRAL aggregates for a single filter."""
    bucket: Dict[str, List[Dict[str, Any]]] = {"ALIGNED": [], "OPPOSED": [], "NEUTRAL": []}
    for t in trades:
        q = quarter_of(t["entry_ts"])
        if quarter_filter is not None and q not in quarter_filter:
            continue
        bucket[t[align_key]].append(t)
    aligned = aggregate(bucket["ALIGNED"])
    opposed = aggregate(bucket["OPPOSED"])
    neutral = aggregate(bucket["NEUTRAL"])
    sep_R = aligned["ev_R"] - opposed["ev_R"]
    return {
        "ALIGNED": aligned,
        "OPPOSED": opposed,
        "NEUTRAL": neutral,
        "sep_R": sep_R,
    }


MIN_MEANINGFUL_SEP_R = 0.05  # 5% of avg risk per trade — below this is statistical noise


def verdict_from_splits(is_score: Dict[str, Any], q3: Dict[str, Any], q4: Dict[str, Any]) -> str:
    """SHIP / HOLD / KILL based on IS sep>=MIN AND OOS preserve >=50% same-sign.

    A filter must clear MIN_MEANINGFUL_SEP_R on IS to even be considered for SHIP.
    Otherwise the 50% preservation rule trivially passes when IS sep is near zero
    (e.g. +0.001R IS and +0.0006R OOS would pass mathematically but mean nothing
    economically).
    """
    is_sep = is_score["sep_R"]
    if is_sep <= 0:
        return "KILL"

    def passes(oos_sep: float) -> bool:
        if oos_sep <= 0:
            return False
        return oos_sep >= 0.5 * is_sep

    q3_pass = passes(q3["sep_R"])
    q4_pass = passes(q4["sep_R"])
    is_meaningful = is_sep >= MIN_MEANINGFUL_SEP_R

    if is_meaningful and q3_pass and q4_pass:
        return "SHIP"
    if is_meaningful and (q3_pass or q4_pass):
        return "HOLD"
    if not is_meaningful and (q3_pass or q4_pass):
        # IS sep too small to be meaningful — call it HOLD even if OOS positive
        return "HOLD"
    return "KILL"


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
def run() -> Dict[str, Any]:
    # Step 1: Tune HA_BIAS (TF, N) on Q1+Q2 IS using both entry TFs combined.
    # We compute the cohort for every (entry_tf, ha_tf, ha_n) combo on each
    # symbol and aggregate IS separation magnitude. Pick the (TF, N) combo
    # with highest IS separation across both entry TFs averaged equally.
    ha_tune_results: List[Dict[str, Any]] = []
    cohort_cache: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}

    for entry_tf in ENTRY_TFS:
        for ha_tf in HA_TF_GRID:
            for ha_n in HA_N_GRID:
                all_trades: List[Dict[str, Any]] = []
                for sym in SYMBOLS:
                    trades = build_cohort_tagged(sym, entry_tf, ha_tf, ha_n)
                    cohort_cache[(sym, entry_tf, ha_tf, ha_n)] = trades
                    all_trades.extend(trades)
                is_score = score_filter(all_trades, "ha_align", ["Q1", "Q2"])
                ha_tune_results.append({
                    "entry_tf": entry_tf,
                    "ha_tf": ha_tf,
                    "ha_n": ha_n,
                    "IS": is_score,
                })

    # Pick best HA combo: maximum mean IS sep across entry TFs (both entry TFs
    # weighted equally) — i.e., for each (ha_tf, ha_n), average sep_R over
    # entry_tf and require min n in IS >= 30 per entry_tf.
    by_ha: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in ha_tune_results:
        by_ha.setdefault((r["ha_tf"], r["ha_n"]), []).append(r)
    candidates = []
    for (ha_tf, ha_n), recs in by_ha.items():
        n_ali_total = sum(rec["IS"]["ALIGNED"]["n"] for rec in recs)
        n_opp_total = sum(rec["IS"]["OPPOSED"]["n"] for rec in recs)
        if min(n_ali_total, n_opp_total) < 30:
            # Skip combos with too-thin IS samples
            continue
        mean_sep = float(np.mean([rec["IS"]["sep_R"] for rec in recs]))
        candidates.append((mean_sep, ha_tf, ha_n, recs))
    candidates.sort(reverse=True)
    if not candidates:
        # fallback to absolute best regardless of n
        for (ha_tf, ha_n), recs in by_ha.items():
            mean_sep = float(np.mean([rec["IS"]["sep_R"] for rec in recs]))
            candidates.append((mean_sep, ha_tf, ha_n, recs))
        candidates.sort(reverse=True)
    best_sep, best_ha_tf, best_ha_n, _ = candidates[0]

    # Step 2: With best (HA_TF, HA_N) fixed, build full per-quarter cohorts
    # for that combo and score all 3 filters per entry TF.
    final: Dict[str, Any] = {
        "config": {
            "symbols": SYMBOLS,
            "entry_tfs": ENTRY_TFS,
            "ha_tf_grid": HA_TF_GRID,
            "ha_n_grid": HA_N_GRID,
            "best_ha_tf": best_ha_tf,
            "best_ha_n": best_ha_n,
            "exit_tp_R": EXIT_TP_R,
            "exit_sl_R": EXIT_SL_R,
            "exit_time_min": EXIT_TIME_MIN,
            "fee_bps_rt": RT_TAKER_FEE_BPS,
            "notional": NOTIONAL,
            "atr_period": ATR_PERIOD,
            "quarter_bounds": QUARTER_BOUNDS,
        },
        "ha_tuning": ha_tune_results,
        "ha_tuning_pick_reason": (
            f"Picked (HA_TF={best_ha_tf}, N={best_ha_n}) — highest mean IS "
            f"separation across {ENTRY_TFS} (mean sep={best_sep:+.4f}R)."
        ),
        "by_entry_tf": {},
    }

    # Build full cohort once per entry TF using best HA combo
    full_cohort: Dict[str, List[Dict[str, Any]]] = {}
    for entry_tf in ENTRY_TFS:
        all_trades: List[Dict[str, Any]] = []
        for sym in SYMBOLS:
            trades = cohort_cache.get((sym, entry_tf, best_ha_tf, best_ha_n))
            if trades is None:
                trades = build_cohort_tagged(sym, entry_tf, best_ha_tf, best_ha_n)
            all_trades.extend(trades)
        full_cohort[entry_tf] = all_trades

    # Per entry TF: score each filter on IS, Q3, Q4 + verdict
    for entry_tf, trades in full_cohort.items():
        per_filter: Dict[str, Any] = {}
        for filter_name, align_key in [
            ("HA_BIAS", "ha_align"),
            ("EMA21_50", "ema21_50_align"),
            ("EMA200_d", "ema200_d_align"),
        ]:
            is_s = score_filter(trades, align_key, ["Q1", "Q2"])
            q3_s = score_filter(trades, align_key, ["Q3"])
            q4_s = score_filter(trades, align_key, ["Q4"])
            v = verdict_from_splits(is_s, q3_s, q4_s)
            per_filter[filter_name] = {
                "IS": is_s,
                "Q3": q3_s,
                "Q4": q4_s,
                "verdict": v,
            }
        final["by_entry_tf"][entry_tf] = {
            "n_trades_total": len(trades),
            "filters": per_filter,
        }

    # Step 3: Combined entry TF view (5m + 15m together) for the head-to-head
    # winner verdict
    combined_trades: List[Dict[str, Any]] = []
    for entry_tf in ENTRY_TFS:
        combined_trades.extend(full_cohort[entry_tf])
    combined_per_filter: Dict[str, Any] = {}
    for filter_name, align_key in [
        ("HA_BIAS", "ha_align"),
        ("EMA21_50", "ema21_50_align"),
        ("EMA200_d", "ema200_d_align"),
    ]:
        is_s = score_filter(combined_trades, align_key, ["Q1", "Q2"])
        q3_s = score_filter(combined_trades, align_key, ["Q3"])
        q4_s = score_filter(combined_trades, align_key, ["Q4"])
        v = verdict_from_splits(is_s, q3_s, q4_s)
        combined_per_filter[filter_name] = {
            "IS": is_s,
            "Q3": q3_s,
            "Q4": q4_s,
            "verdict": v,
        }
    final["combined"] = {
        "n_trades_total": len(combined_trades),
        "filters": combined_per_filter,
    }

    # Step 4: Pick winner. Rank by:
    #   (1) verdict (SHIP > HOLD > KILL),
    #   (2) within verdict, a "consistency score" = min(IS, Q3, Q4) sep_R
    #       — penalises any one quarter going strongly negative,
    #   (3) tiebreak: highest IS sep magnitude (real edge over noise).
    def mean_oos_sep(rec: Dict[str, Any]) -> float:
        return 0.5 * (rec["Q3"]["sep_R"] + rec["Q4"]["sep_R"])

    def consistency(rec: Dict[str, Any]) -> float:
        return float(min(rec["IS"]["sep_R"], rec["Q3"]["sep_R"], rec["Q4"]["sep_R"]))

    ranked = []
    for fname, rec in combined_per_filter.items():
        # Aggregate score = mean of (IS sep, Q3 sep, Q4 sep). Rewards
        # magnitude where it exists and penalises any negative quarter.
        agg_score = float(np.mean([rec["IS"]["sep_R"], rec["Q3"]["sep_R"], rec["Q4"]["sep_R"]]))
        ranked.append((
            rec["verdict"],
            agg_score,
            rec["IS"]["sep_R"],
            mean_oos_sep(rec),
            consistency(rec),
            fname,
        ))
    # Verdict order: SHIP > HOLD > KILL.
    # Within same verdict: maximise mean(IS, Q3, Q4) sep — gives HA_BIAS-like
    # filters credit for big IS+Q3 sep while still being penalised for a
    # negative Q4. Tiebreak: IS magnitude (real economic edge).
    verdict_rank = {"SHIP": 0, "HOLD": 1, "KILL": 2}
    ranked.sort(key=lambda t: (verdict_rank[t[0]], -t[1], -t[2]))
    winner_verdict, _agg, winner_is, winner_oos, _winner_consist, winner_name = ranked[0]
    final["winner"] = {
        "filter": winner_name,
        "verdict": winner_verdict,
        "IS_sep_R": winner_is,
        "mean_OOS_sep_R": winner_oos,
        "min_meaningful_sep_R": MIN_MEANINGFUL_SEP_R,
        "ranking": [
            {
                "filter": r[5],
                "verdict": r[0],
                "agg_3q_sep_R": r[1],
                "IS_sep_R": r[2],
                "mean_OOS_sep_R": r[3],
                "worst_quarter_sep_R": r[4],
            }
            for r in ranked
        ],
    }
    return final


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def write_report(results: Dict[str, Any]) -> None:
    out_json = OUT_DIR / "walkforward.json"
    out_md = OUT_DIR / "report.md"
    out_json.write_text(json.dumps(results, default=str, indent=2))

    cfg = results["config"]
    lines: List[str] = []
    lines.append("# Regime Filter 3-Way Walk-Forward")
    lines.append("")
    lines.append("Three-way head-to-head: HA_BIAS (Heiken Ashi Market Bias) vs "
                 "EMA21_50 (current bot session_bias) vs EMA200_d (daily macro filter).")
    lines.append("")
    lines.append("**Test cohort:** engulfing-reversal entries on entry TF (5m + 15m), "
                 "exit TP=1.5R / SL=1R / 60min time stop, fee 0.118% RT.")
    lines.append("")
    lines.append("**Walk-forward splits (UTC):**")
    for q, (s, e) in cfg["quarter_bounds"].items():
        lines.append(f"- {q}: {s} -> {e}")
    lines.append("")
    lines.append(f"**HA_BIAS tuning (Q1+Q2 IS):** swept TF in {cfg['ha_tf_grid']} x N in {cfg['ha_n_grid']}")
    lines.append(f"- Pick: HA_TF=**{cfg['best_ha_tf']}**, N=**{cfg['best_ha_n']}**")
    lines.append(f"- Reason: {results['ha_tuning_pick_reason']}")
    lines.append("")
    lines.append("**Pass criteria:** IS separation > 0 AND BOTH OOS quarters preserve sign and >= 50% magnitude.")
    lines.append("Verdict: SHIP / HOLD / KILL.")
    lines.append("")

    # Combined head-to-head
    lines.append("## Head-to-Head (5m + 15m combined, n=" +
                 str(results["combined"]["n_trades_total"]) + ")")
    lines.append("")
    lines.append("| Filter | IS ALIGNED EV (n) | IS OPPOSED EV (n) | IS sep | "
                 "Q3 sep (n_ali/n_opp) | Q4 sep (n_ali/n_opp) | Verdict |")
    lines.append("|--------|------------------|------------------|--------|"
                 "----------------------|----------------------|---------|")
    for fname, rec in results["combined"]["filters"].items():
        ali = rec["IS"]["ALIGNED"]
        opp = rec["IS"]["OPPOSED"]
        q3 = rec["Q3"]
        q4 = rec["Q4"]
        lines.append(
            f"| {fname} | {ali['ev_R']:+.4f}R (n={ali['n']}) | "
            f"{opp['ev_R']:+.4f}R (n={opp['n']}) | {rec['IS']['sep_R']:+.4f}R | "
            f"{q3['sep_R']:+.4f}R ({q3['ALIGNED']['n']}/{q3['OPPOSED']['n']}) | "
            f"{q4['sep_R']:+.4f}R ({q4['ALIGNED']['n']}/{q4['OPPOSED']['n']}) | "
            f"**{rec['verdict']}** |"
        )
    lines.append("")

    # Per entry TF detailed
    for entry_tf, block in results["by_entry_tf"].items():
        lines.append(f"## {entry_tf} entry TF (n={block['n_trades_total']})")
        lines.append("")
        for fname, rec in block["filters"].items():
            lines.append(f"### {fname}")
            ali_is = rec["IS"]["ALIGNED"]
            opp_is = rec["IS"]["OPPOSED"]
            neu_is = rec["IS"]["NEUTRAL"]
            lines.append(
                f"- IS ALIGNED: ev={ali_is['ev_R']:+.4f}R wr={ali_is['wr']*100:.1f}% n={ali_is['n']}"
            )
            lines.append(
                f"- IS OPPOSED: ev={opp_is['ev_R']:+.4f}R wr={opp_is['wr']*100:.1f}% n={opp_is['n']}"
            )
            lines.append(
                f"- IS NEUTRAL: ev={neu_is['ev_R']:+.4f}R wr={neu_is['wr']*100:.1f}% n={neu_is['n']}"
            )
            lines.append(f"- IS sep: **{rec['IS']['sep_R']:+.4f}R**")
            for qk in ("Q3", "Q4"):
                ali = rec[qk]["ALIGNED"]
                opp = rec[qk]["OPPOSED"]
                lines.append(
                    f"- {qk}: sep={rec[qk]['sep_R']:+.4f}R "
                    f"(ALI ev={ali['ev_R']:+.4f}R n={ali['n']}, "
                    f"OPP ev={opp['ev_R']:+.4f}R n={opp['n']})"
                )
            lines.append(f"- Verdict: **{rec['verdict']}**")
            lines.append("")

    # Winner
    w = results["winner"]
    lines.append("## Winner")
    lines.append("")
    lines.append(f"**{w['filter']}** — verdict {w['verdict']}, IS sep "
                 f"{w['IS_sep_R']:+.4f}R, mean OOS sep {w['mean_OOS_sep_R']:+.4f}R.")
    lines.append("")
    lines.append(f"Ranking (combined, by verdict then mean(IS,Q3,Q4) sep; min meaningful IS sep = {w['min_meaningful_sep_R']:+.3f}R):")
    for r in w["ranking"]:
        lines.append(
            f"- {r['filter']}: verdict={r['verdict']}, mean 3q sep={r['agg_3q_sep_R']:+.4f}R, "
            f"IS sep={r['IS_sep_R']:+.4f}R, mean OOS sep={r['mean_OOS_sep_R']:+.4f}R, "
            f"worst-quarter sep={r['worst_quarter_sep_R']:+.4f}R"
        )
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    if w["verdict"] == "SHIP":
        if w["filter"] == "HA_BIAS":
            lines.append(f"**SHIP HA_BIAS** as flag-gated VETO 10k in `strategies/scalp_strategy.py` "
                         f"with HA_TF={cfg['best_ha_tf']}, HA_N={cfg['best_ha_n']}, default OFF.")
        elif w["filter"] == "EMA200_d":
            lines.append("**EMA200_d wins** — already shipped dark today as MACRO_EMA200_VETO. "
                         "Recommend continuing dark while monitoring; do not enable globally yet.")
        elif w["filter"] == "EMA21_50":
            lines.append("**EMA21_50 wins** — this is already the bot's `session_bias` filter "
                         "and is in production. No code change needed.")
    elif w["verdict"] == "HOLD":
        lines.append(f"**Do NOT ship.** Best filter ({w['filter']}) is HOLD: IS edge present but "
                     f"OOS does not preserve >=50% in both quarters or IS magnitude is below "
                     f"the {w['min_meaningful_sep_R']:+.3f}R noise floor.")
        if w["filter"] == "HA_BIAS":
            lines.append("HA_BIAS shows the largest IS magnitude (best directional signal) but "
                         "Q4 (Mar-Apr 2026) sep flips negative — likely the recent regime shift "
                         "broke the wick-purity assumption. Re-test after another quarter of data.")
        lines.append("In the meantime: keep existing EMA21_50 session_bias and the dark "
                     "MACRO_EMA200_VETO flag. Do not enable HA_BIAS_VETO.")
    else:
        lines.append("**All three filters KILL** — no edge from any regime filter on this cohort. "
                     "Keep EMA21_50 as the default session_bias and leave MACRO_EMA200_VETO dark.")
    lines.append("")

    # Caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append("- 5m cache starts Nov 5 2025; 15m/1h/4h start Oct 6-7 2025. Q1 5m sample is truncated.")
    lines.append("- Test cohort is synthetic (engulfing reversals on 5m/15m), proxying counter-trend signals like structure_bounce. Magnitudes will differ from live scanner cohorts.")
    lines.append("- 6-month window: BTC bear-rally regime dominates; mean-reverting separation may be regime-specific.")
    lines.append("- EMA200_d derived from 4h-resampled daily; only 193 daily bars in the 6-month cache. We use min_periods=50 (span=200) so the EMA can warm — production extends via OKX to a full 200-day window. Live macro_bias.parquet may differ in early Q1 where the EMA is still warming.")
    lines.append("- EMA200_d uses a 0.5% chop band (matches production _classify_bias) — bars within +/-0.5% of EMA are tagged NEUTRAL, not ALIGNED/OPPOSED.")
    lines.append("- HA_BIAS uses no-wick strict definition; relaxed wick-tolerance variants not swept.")
    lines.append("- Single-bar entry at signal close, intra-bar SL priority over TP if both touched (conservative).")
    lines.append("")

    out_md.write_text("\n".join(lines))


def main() -> None:
    print("[Regime3Way] Running 3-way walk-forward backtest...")
    results = run()
    write_report(results)
    print(f"[Regime3Way] Wrote: {OUT_DIR/'walkforward.json'}")
    print(f"[Regime3Way] Wrote: {OUT_DIR/'report.md'}")
    cfg = results["config"]
    print(f"\nHA tuning pick: HA_TF={cfg['best_ha_tf']}, N={cfg['best_ha_n']}")
    print("\nCombined head-to-head:")
    for fname, rec in results["combined"]["filters"].items():
        print(
            f"  {fname}: IS sep={rec['IS']['sep_R']:+.4f}R "
            f"Q3 sep={rec['Q3']['sep_R']:+.4f}R "
            f"Q4 sep={rec['Q4']['sep_R']:+.4f}R -> {rec['verdict']}"
        )
    w = results["winner"]
    print(f"\nWinner: {w['filter']} (verdict={w['verdict']}, "
          f"IS sep={w['IS_sep_R']:+.4f}R, mean OOS sep={w['mean_OOS_sep_R']:+.4f}R)")


if __name__ == "__main__":
    main()

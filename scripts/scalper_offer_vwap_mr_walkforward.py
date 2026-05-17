#!/usr/bin/env python3
"""VWAP Mean Reversion walk-forward — Delta scalper-offer fee math re-test.

Prior verdict on VWAP MR scalping was KILL at -$1.20/trade with full taker
round-trip fees (0.118-0.12% RT). Delta India's Scalper Offer waives the EXIT
leg fee for BTC/ETH trades closing within 30 minutes of entry. Corrected
round-trip drops from ~$1.20 to ~$0.60 per $1000 notional — the question is
whether that ~$0.60 uplift per trade is enough to revive the strategy.

This re-tests three fee variants in lockstep so the EV uplift per step is
visible:

  A — full RT taker  (today's old-kill math: 0.06% + 0.06% = 0.12% RT)
  B — scalper offer  (0.06% entry, 0.00% exit if held <= 28min)
  C — maker entry + scalper offer (0.024% entry, 0.00% exit)

Strategy:
  Trading TF       : 5m or 15m
  Bands            : VWAP ± k × stdev(close, lookback)
  LONG entry rule  : close < VWAP - k×stdev AND ranging regime
                     AND reversal candle (close > open AND close > prev close)
  SHORT entry rule : mirror (close > VWAP + k×stdev AND bearish reversal)
  Regime filter    : ATR_14 percentile rank over last 100 bars <= atr_pct_thr
  Exit             : VWAP touch (mean-reversion target) OR
                     1× ATR_14 stop on the wrong side OR
                     hard time stop at 28 minutes (under 30min scalper window)
  Sizing           : $1000 notional per trade
  Symbols          : BTC/USDT and ETH/USDT only (scalper-offer eligibility)

Walk-forward windows:
  Q1: 2025-10-01 → 2025-12-01  (in-sample, with Q2)
  Q2: 2025-12-01 → 2026-02-01  (in-sample, with Q1)
  Q3: 2026-02-01 → 2026-03-01  (out-of-sample 1)
  Q4: 2026-03-01 → 2026-05-01  (out-of-sample 2)

Note: cache starts 2025-11-05 — Q1 will be partial (Nov-only, ~25 days).

Param grid (tuned on Q1+Q2):
  TF                 5m, 15m
  k (band width)     1.0, 1.5, 2.0
  stdev lookback     20, 50 bars
  ATR-pct threshold  0.40, 0.50, 0.60

Pass criteria (per variant):
  IS EV positive AND OOS EV >= 50% of IS EV AND same sign on each Q3 and Q4.
  Strict ship gate: OOS Q4 EV per trade > +$0.10 (engine-spec rule).

Output:
  storage/scalper_vwap_mr/walkforward.json
  storage/scalper_vwap_mr/report.md

READ-ONLY on bot/signal_tracker.py, bot/signal_journey.py, bot/signal_learner.py.
Does NOT modify scalp_strategy.py. Standalone script.
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

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_v2.fee_model import FeeModel  # noqa: E402

CACHE = ROOT / "storage" / "candle_cache"
OUT_DIR = ROOT / "storage" / "scalper_vwap_mr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC", "ETH"]              # scalper-offer eligible only
TFS = ["5m", "15m"]
K_GRID = [1.0, 1.5, 2.0]
LOOKBACK_GRID = [20, 50]
ATR_PCT_GRID = [0.40, 0.50, 0.60]

# Trade economics
NOTIONAL = 1000.0
ATR_PERIOD = 14
ATR_PCT_LOOKBACK = 100
TIME_STOP_MIN = 28
SL_ATR_MULT = 1.0

# Walk-forward boundaries (cache only has Nov+ for Q1)
QUARTER_BOUNDS = {
    "Q1": ("2025-10-01", "2025-12-01"),
    "Q2": ("2025-12-01", "2026-02-01"),
    "Q3": ("2026-02-01", "2026-03-01"),
    "Q4": ("2026-03-01", "2026-05-01"),
}

VARIANTS = ["A", "B", "C"]   # A=full taker, B=scalper, C=maker+scalper


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)


def load_tf(sym: str, tf: str) -> Optional[pd.DataFrame]:
    p = CACHE / f"{sym}_USDT_{tf}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d.index = pd.to_datetime(d.index, utc=True)
    return d.sort_index()


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def add_vwap_bands(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Rolling VWAP and stdev bands.

    Use rolling typical-price weighted by volume over `lookback` bars to keep
    the VWAP responsive (a session-anchored VWAP would also be fine; we use
    rolling for consistency with non-session 24/7 crypto). Bands are k*stdev
    of close over the same window.
    """
    out = df.copy()
    typ = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float).clip(lower=1e-9)
    pv = typ * vol
    out["vwap"] = pv.rolling(lookback, min_periods=lookback).sum() / vol.rolling(lookback, min_periods=lookback).sum()
    out["stdev"] = df["close"].astype(float).rolling(lookback, min_periods=lookback).std(ddof=0)
    return out


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
# Trade simulation
# -----------------------------------------------------------------------------
FEE_MODEL = FeeModel()


def fee_for_variant(
    variant: str,
    notional_usd: float,
    symbol: str,
    holding_sec: float,
) -> Dict[str, float]:
    """Return fee_info dict for a given variant."""
    if variant == "A":
        # full RT taker, ignore scalper waiver
        return FEE_MODEL.round_trip_for_trade(
            entry_type="taker", exit_type="taker",
            notional_usd=notional_usd, symbol=symbol,
            holding_sec=holding_sec, force_no_scalper=True,
        )
    elif variant == "B":
        # taker entry, scalper offer applied to exit if eligible+within window
        return FEE_MODEL.round_trip_for_trade(
            entry_type="taker", exit_type="taker",
            notional_usd=notional_usd, symbol=symbol,
            holding_sec=holding_sec, force_no_scalper=False,
        )
    elif variant == "C":
        # maker entry + scalper offer
        return FEE_MODEL.round_trip_for_trade(
            entry_type="maker", exit_type="taker",
            notional_usd=notional_usd, symbol=symbol,
            holding_sec=holding_sec, force_no_scalper=False,
        )
    else:
        raise ValueError(variant)


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    atr: float,
    vwap: float,
    tf_min: int,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """Walk forward up to 28min, exit at VWAP touch / SL / time stop.

    Returns dict with raw/gross PnL and fee dicts for each variant.
    """
    if entry_idx + 1 >= len(df):
        return None
    if not np.isfinite(atr) or atr <= 0:
        return None
    if not np.isfinite(vwap):
        return None

    entry_price = float(df["close"].iloc[entry_idx])
    bars_max = max(1, int(math.ceil(TIME_STOP_MIN / tf_min)))
    end_idx = min(entry_idx + bars_max, len(df) - 1)

    if side == "LONG":
        sl_p = entry_price - SL_ATR_MULT * atr
        tp_p = vwap   # mean-revert target is current VWAP
    else:
        sl_p = entry_price + SL_ATR_MULT * atr
        tp_p = vwap

    exit_reason = "TIME"
    exit_price = float(df["close"].iloc[end_idx])
    exit_idx = end_idx

    for j in range(entry_idx + 1, end_idx + 1):
        bar_h = float(df["high"].iloc[j])
        bar_l = float(df["low"].iloc[j])
        if side == "LONG":
            hit_sl = bar_l <= sl_p
            hit_tp = bar_h >= tp_p
            if hit_sl and hit_tp:
                # conservative: assume SL hit first (worst case)
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break
        else:
            hit_sl = bar_h >= sl_p
            hit_tp = bar_l <= tp_p
            if hit_sl and hit_tp:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_sl:
                exit_reason = "SL"; exit_price = sl_p; exit_idx = j; break
            if hit_tp:
                exit_reason = "TP"; exit_price = tp_p; exit_idx = j; break

    # Holding seconds (entry-bar close to exit-bar close)
    entry_ts = df.index[entry_idx]
    exit_ts = df.index[exit_idx]
    holding_sec = float((exit_ts - entry_ts).total_seconds())

    if side == "LONG":
        ret = (exit_price - entry_price) / entry_price
    else:
        ret = (entry_price - exit_price) / entry_price
    gross_dollars = NOTIONAL * ret

    risk_dollars = NOTIONAL * abs(SL_ATR_MULT * atr / entry_price)
    gross_R = gross_dollars / risk_dollars if risk_dollars > 0 else 0.0

    # Compute fee for each variant
    fees = {}
    nets = {}
    scalper_applied = {}
    for v in VARIANTS:
        info = fee_for_variant(v, NOTIONAL, f"{symbol}/USDT", holding_sec)
        fees[v] = info["fee_usd"]
        nets[v] = gross_dollars - info["fee_usd"]
        scalper_applied[v] = bool(info["scalper_applied"])

    return {
        "symbol": symbol,
        "entry_ts": entry_ts.isoformat(),
        "exit_ts": exit_ts.isoformat(),
        "side": side,
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "atr": atr,
        "vwap": vwap,
        "exit_reason": exit_reason,
        "holding_sec": holding_sec,
        "gross_R": gross_R,
        "gross_$": gross_dollars,
        "fee_$": fees,
        "net_$": nets,
        "scalper_applied": scalper_applied,
        "quarter": quarter_of(entry_ts),
    }


# -----------------------------------------------------------------------------
# Cohort builder
# -----------------------------------------------------------------------------
def build_trades_for_combo(
    sym: str, tf: str, k: float, lookback: int, atr_pct_thr: float,
) -> List[Dict[str, Any]]:
    df = load_tf(sym, tf)
    if df is None or len(df) < max(lookback, ATR_PCT_LOOKBACK) + 50:
        return []

    df = add_vwap_bands(df, lookback)
    df["atr"] = add_atr(df, ATR_PERIOD)
    df["atr_pct"] = df["atr"].rolling(ATR_PCT_LOOKBACK, min_periods=ATR_PCT_LOOKBACK).rank(pct=True)

    upper = df["vwap"] + k * df["stdev"]
    lower = df["vwap"] - k * df["stdev"]

    # reversal candles
    bull_rev = (df["close"] > df["open"]) & (df["close"] > df["close"].shift(1))
    bear_rev = (df["close"] < df["open"]) & (df["close"] < df["close"].shift(1))

    # entry triggers (must be RANGING — atr_pct <= threshold)
    ranging = df["atr_pct"] <= atr_pct_thr
    long_trigger = (df["close"] < lower) & bull_rev & ranging
    short_trigger = (df["close"] > upper) & bear_rev & ranging

    tf_min = tf_minutes(tf)
    trades: List[Dict[str, Any]] = []

    # Iterate over signal candle indices
    long_idx = np.where(long_trigger.fillna(False).values)[0]
    short_idx = np.where(short_trigger.fillna(False).values)[0]

    for i in long_idx:
        atr_v = df["atr"].iloc[i]
        vwap_v = df["vwap"].iloc[i]
        if not np.isfinite(atr_v) or not np.isfinite(vwap_v):
            continue
        t = simulate_trade(df, int(i), "LONG", float(atr_v), float(vwap_v), tf_min, sym)
        if t is not None and t["quarter"] is not None:
            trades.append(t)
    for i in short_idx:
        atr_v = df["atr"].iloc[i]
        vwap_v = df["vwap"].iloc[i]
        if not np.isfinite(atr_v) or not np.isfinite(vwap_v):
            continue
        t = simulate_trade(df, int(i), "SHORT", float(atr_v), float(vwap_v), tf_min, sym)
        if t is not None and t["quarter"] is not None:
            trades.append(t)

    return trades


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def agg_for_quarters(trades: List[Dict[str, Any]], quarters: List[str], variant: str) -> Dict[str, Any]:
    sub = [t for t in trades if t["quarter"] in quarters]
    n = len(sub)
    if n == 0:
        return {"n": 0, "wr": 0.0, "ev_$": 0.0, "net_$": 0.0,
                "scalper_applied_pct": 0.0, "tp_pct": 0.0, "sl_pct": 0.0, "time_pct": 0.0}
    nets = [t["net_$"][variant] for t in sub]
    wins = sum(1 for v in nets if v > 0)
    sa = sum(1 for t in sub if t["scalper_applied"][variant]) / n
    tp = sum(1 for t in sub if t["exit_reason"] == "TP") / n
    sl = sum(1 for t in sub if t["exit_reason"] == "SL") / n
    tm = sum(1 for t in sub if t["exit_reason"] == "TIME") / n
    return {
        "n": n,
        "wr": wins / n,
        "ev_$": float(np.mean(nets)),
        "net_$": float(np.sum(nets)),
        "scalper_applied_pct": float(sa),
        "tp_pct": float(tp),
        "sl_pct": float(sl),
        "time_pct": float(tm),
    }


def gap_pct(is_ev: float, oos_ev: float) -> float:
    """Return signed % of IS that OOS preserves. >100% = OOS exceeds IS."""
    if abs(is_ev) < 1e-9:
        return 0.0
    return 100.0 * oos_ev / is_ev


# -----------------------------------------------------------------------------
# Walk-forward sweep
# -----------------------------------------------------------------------------
def main():
    print(f"[wf] Building VWAP MR cohort across grid…")
    print(f"  symbols={SYMBOLS} tfs={TFS} k={K_GRID} lookback={LOOKBACK_GRID} atr_pct={ATR_PCT_GRID}")

    # cell -> {"trades": [...], "params": {...}}
    cells: Dict[str, Dict[str, Any]] = {}
    grid = list(product(TFS, K_GRID, LOOKBACK_GRID, ATR_PCT_GRID))
    total_signals = 0

    for tf, k, lb, atr_thr in grid:
        cell_key = f"tf={tf}|k={k}|lb={lb}|atr_pct<={atr_thr}"
        all_trades = []
        for sym in SYMBOLS:
            t = build_trades_for_combo(sym, tf, k, lb, atr_thr)
            all_trades.extend(t)
        cells[cell_key] = {
            "params": {"tf": tf, "k": k, "lookback": lb, "atr_pct_thr": atr_thr},
            "trades": all_trades,
        }
        total_signals += len(all_trades)
        print(f"  {cell_key}  n={len(all_trades)}")

    print(f"[wf] total signals (sum cells): {total_signals}")

    # Build per-variant per-cell stats
    summary: Dict[str, Any] = {
        "config": {
            "symbols": SYMBOLS, "tfs": TFS, "k_grid": K_GRID,
            "lookback_grid": LOOKBACK_GRID, "atr_pct_grid": ATR_PCT_GRID,
            "notional": NOTIONAL, "time_stop_min": TIME_STOP_MIN,
            "sl_atr_mult": SL_ATR_MULT, "atr_period": ATR_PERIOD,
            "atr_pct_lookback": ATR_PCT_LOOKBACK,
            "quarters": QUARTER_BOUNDS,
        },
        "variants": {},
    }

    for variant in VARIANTS:
        per_cell = []
        for cell_key, cell in cells.items():
            tr = cell["trades"]
            is_stats = agg_for_quarters(tr, ["Q1", "Q2"], variant)
            q3_stats = agg_for_quarters(tr, ["Q3"], variant)
            q4_stats = agg_for_quarters(tr, ["Q4"], variant)
            full_stats = agg_for_quarters(tr, ["Q1", "Q2", "Q3", "Q4"], variant)
            row = {
                "cell": cell_key,
                "params": cell["params"],
                "IS_n": is_stats["n"], "IS_ev$": is_stats["ev_$"],
                "IS_wr": is_stats["wr"], "IS_net$": is_stats["net_$"],
                "Q3_n": q3_stats["n"], "Q3_ev$": q3_stats["ev_$"],
                "Q3_wr": q3_stats["wr"],
                "Q4_n": q4_stats["n"], "Q4_ev$": q4_stats["ev_$"],
                "Q4_wr": q4_stats["wr"],
                "Q3_gap%": gap_pct(is_stats["ev_$"], q3_stats["ev_$"]),
                "Q4_gap%": gap_pct(is_stats["ev_$"], q4_stats["ev_$"]),
                "all_n": full_stats["n"],
                "all_ev$": full_stats["ev_$"],
                "scalper_applied_pct": full_stats["scalper_applied_pct"],
                "tp_pct": full_stats["tp_pct"],
                "sl_pct": full_stats["sl_pct"],
                "time_pct": full_stats["time_pct"],
            }
            per_cell.append(row)

        # rank candidates
        per_cell_sorted = sorted(per_cell, key=lambda r: r["IS_ev$"], reverse=True)

        # walk-forward verdict on each cell
        for r in per_cell_sorted:
            ev_is = r["IS_ev$"]
            ev_q3 = r["Q3_ev$"]
            ev_q4 = r["Q4_ev$"]
            n_is = r["IS_n"]; n_q3 = r["Q3_n"]; n_q4 = r["Q4_n"]
            verdict = "KILL"
            if n_is >= 20 and ev_is > 0:
                # Both OOS quarters need to preserve sign and >=50% magnitude
                signs_ok = ev_q3 > 0 and ev_q4 > 0
                q3_pct = gap_pct(ev_is, ev_q3)
                q4_pct = gap_pct(ev_is, ev_q4)
                mag_ok = q3_pct >= 50.0 and q4_pct >= 50.0
                ship_gate = ev_q4 > 0.10  # engine spec: > $0.10/trade in Q4
                if signs_ok and mag_ok and ship_gate:
                    verdict = "PASS"
                elif signs_ok and (q3_pct >= 50.0 or q4_pct >= 50.0):
                    verdict = "HOLD"
                else:
                    verdict = "KILL"
            elif n_is < 20:
                verdict = "INSUFFICIENT_N"
            r["verdict"] = verdict

        summary["variants"][variant] = {
            "rows": per_cell_sorted,
            "best": per_cell_sorted[0] if per_cell_sorted else None,
            "passes": [r for r in per_cell_sorted if r["verdict"] == "PASS"],
            "holds": [r for r in per_cell_sorted if r["verdict"] == "HOLD"],
        }

    out_json = OUT_DIR / "walkforward.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[wf] wrote {out_json}")

    # report.md
    md = ["# Scalper-Offer VWAP Mean Reversion — Walk-Forward Report",
          "",
          f"*Generated against cache window — see config.quarters.*",
          "",
          "## Variants",
          "- **A** — full RT taker (0.06% × 2 = 0.12% RT)  [old kill math]",
          "- **B** — scalper offer applied (0.06% entry, 0.00% exit if BTC/ETH ≤ 28min)",
          "- **C** — maker entry + scalper offer (0.024% entry, 0.00% exit)",
          "",
          "## Param grid",
          f"- TF ∈ {TFS}",
          f"- k ∈ {K_GRID}",
          f"- stdev lookback ∈ {LOOKBACK_GRID}",
          f"- ATR-pct threshold ∈ {ATR_PCT_GRID}",
          f"- Symbols: BTC/USDT, ETH/USDT",
          f"- Hard time stop: {TIME_STOP_MIN} min, SL: {SL_ATR_MULT}× ATR_{ATR_PERIOD}",
          f"- Notional: ${NOTIONAL}",
          "",
          "## Best cell per variant (by IS EV)",
          ""]
    for v in VARIANTS:
        b = summary["variants"][v]["best"]
        if b is None:
            md.append(f"### Variant {v}\n_no cells_\n")
            continue
        md += [
            f"### Variant {v} — best cell `{b['cell']}`",
            f"- IS  : n={b['IS_n']}  EV=${b['IS_ev$']:+.3f}  WR={b['IS_wr']*100:.1f}%",
            f"- Q3  : n={b['Q3_n']}  EV=${b['Q3_ev$']:+.3f}  gap={b['Q3_gap%']:.0f}%  WR={b['Q3_wr']*100:.1f}%",
            f"- Q4  : n={b['Q4_n']}  EV=${b['Q4_ev$']:+.3f}  gap={b['Q4_gap%']:.0f}%  WR={b['Q4_wr']*100:.1f}%",
            f"- All : n={b['all_n']}  EV=${b['all_ev$']:+.3f}  scalper_applied={b['scalper_applied_pct']*100:.0f}%",
            f"- Exits: TP={b['tp_pct']*100:.0f}%  SL={b['sl_pct']*100:.0f}%  TIME={b['time_pct']*100:.0f}%",
            f"- Verdict: **{b['verdict']}**",
            "",
        ]

    md += ["## All cells, sorted by IS EV (variant B)", ""]
    rows_b = summary["variants"]["B"]["rows"]
    md += ["| cell | IS n | IS EV | Q3 EV | Q3 gap% | Q4 EV | Q4 gap% | scalper% | verdict |",
           "|------|------|-------|-------|---------|-------|---------|----------|---------|"]
    for r in rows_b:
        md.append(
            f"| `{r['cell']}` | {r['IS_n']} | ${r['IS_ev$']:+.2f} | "
            f"${r['Q3_ev$']:+.2f} | {r['Q3_gap%']:+.0f}% | "
            f"${r['Q4_ev$']:+.2f} | {r['Q4_gap%']:+.0f}% | "
            f"{r['scalper_applied_pct']*100:.0f}% | {r['verdict']} |"
        )
    md += ["",
           "## EV uplift A → B → C (best cell of each variant)",
           ""]
    for v in VARIANTS:
        b = summary["variants"][v]["best"]
        if b is None: continue
        md.append(f"- **{v}** all-quarters EV/trade = ${b['all_ev$']:+.3f}/trade   "
                  f"(IS=${b['IS_ev$']:+.3f}, Q4=${b['Q4_ev$']:+.3f})")
    md += ["",
           "## Cross-variant verdicts (best cell)",
           ""]
    for v in VARIANTS:
        passes = summary["variants"][v]["passes"]
        holds = summary["variants"][v]["holds"]
        md.append(f"- **{v}**  PASS={len(passes)}, HOLD={len(holds)}, total cells={len(summary['variants'][v]['rows'])}")
    md.append("")

    out_md = OUT_DIR / "report.md"
    out_md.write_text("\n".join(md))
    print(f"[wf] wrote {out_md}")

    # Console summary
    print("\n=== best cell per variant ===")
    for v in VARIANTS:
        b = summary["variants"][v]["best"]
        if b is None: continue
        print(f"  {v}: {b['cell']}  IS=${b['IS_ev$']:+.3f}  Q3=${b['Q3_ev$']:+.3f}  Q4=${b['Q4_ev$']:+.3f}  verdict={b['verdict']}")


if __name__ == "__main__":
    main()

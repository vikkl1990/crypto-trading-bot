#!/usr/bin/env python3
"""Counterfactual exit-guard analyzer for VN Edge Phase 5.20.8 refactor.

Replays historical trades that were killed by aggressive guards
(quick_kill / early_kill / no_proof_of_life / zombie_kill / dead_market)
through the new unified exit policy in execution/exit_guards.py and
quantifies the would-have-been P&L delta, with bootstrap CIs and a
ship/no-ship verdict.

Read-only: never touches live execution path or production state.

Usage
-----
    python3 scripts/counterfactual_exit_analyzer.py \\
        [--days 7] [--cache .rollback/wave2-backtest/candle_cache] \\
        [--out .rollback/wave2-backtest/counterfactual_report.md]

Author: VN Edge backtest authority, 2026-04-25.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make project root importable so we get the *real* exit_guards + PRODUCT_MAP.
_ROOT = Path("/home/opc/crypto-trading-bot")
sys.path.insert(0, str(_ROOT))

from execution.exit_guards import (  # noqa: E402
    fee_floor_r,
    grace_window_sec,
    should_kill_dead_signal,
    EXIT_DEAD_SIGNAL_UNIFIED,
    EXIT_STALLED_AFTER_15MIN,
)
from exchange.delta_client import PRODUCT_MAP  # noqa: E402

import asyncio  # noqa: E402

import asyncpg  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────────
DELTA_BASE = "https://api.india.delta.exchange"
RT_TAKER_FEE_PCT = 0.00059 * 2  # 5.9bp/side per design doc
POST_EXIT_WINDOW_MIN = 90
MIN_REQUIRED_CANDLES = 5
RATE_LIMIT_SLEEP_S = 0.25
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42

# Ship-criteria from design doc
KILL_DELTA_PNL_USD = 200.0
KILL_DELTA_WR_PP = 20.0

KILL_REASONS = (
    "quick_kill",
    "early_kill",
    "no_proof_of_life",
    "zombie_kill",
    "dead_market",
)

DB_CFG = dict(
    host="localhost",
    database="vnedge",
    user="vnedge",
    password="VnEdge2026db",
)

# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class Trade:
    id: str
    user_id: str
    symbol: str
    side: str  # 'long' / 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usd: float
    fees_usd: float
    entry_fee_usd: float
    opened_at: float  # unix seconds
    closed_at: float  # unix seconds
    exit_reason: str
    grade: Optional[str]
    regime: Optional[str]
    trade_type: Optional[str]
    peak_mfe_r_at_exit: float
    stop_loss: float
    leverage: float
    contract_size: float
    initial_risk: float  # per-unit (price distance entry-to-SL)
    margin: float


@dataclass
class CounterfactualResult:
    trade: Trade
    cf_exit_price: float
    cf_exit_reason: str
    cf_pnl_usd: float
    cf_age_sec: float
    cf_peak_mfe_r: float
    candles_seen: int


# ── DB ──────────────────────────────────────────────────────────────────────
async def _fetch_candidates_async(days: int) -> List[dict]:
    sql = """
    SELECT id::text AS id, user_id::text AS user_id,
           symbol, side, entry_price, exit_price,
           quantity, pnl_usd, fees_usd,
           EXTRACT(EPOCH FROM opened_at)::float8 AS opened_ts,
           EXTRACT(EPOCH FROM closed_at)::float8 AS closed_ts,
           metadata::text AS metadata
    FROM user_trades
    WHERE trade_type IN ('shadow', 'real')
      AND status = 'closed'
      AND closed_at >= NOW() - ($1 || ' days')::interval
      AND metadata::jsonb->>'exit_reason' = ANY($2::text[])
    ORDER BY closed_at DESC
    """
    conn = await asyncpg.connect(**DB_CFG)
    try:
        rows = await conn.fetch(sql, str(days), list(KILL_REASONS))
        return [dict(r) for r in rows]
    finally:
        await conn.close()


PARSE_SKIPS: List[Tuple[str, str, str]] = []  # (id, symbol, reason)


def fetch_candidates(days: int) -> List[Trade]:
    rows = asyncio.run(_fetch_candidates_async(days))
    out: List[Trade] = []
    for r in rows:
        try:
            m = json.loads(r["metadata"]) if r.get("metadata") else {}
        except Exception:
            m = {}
        try:
            entry = float(r["entry_price"] or 0)  # noqa: E501
            sl = float(m.get("stop_loss") or 0)
            if entry <= 0 or sl <= 0:
                PARSE_SKIPS.append(
                    (str(r["id"])[:8], r["symbol"],
                     f"missing entry/sl in metadata (entry={entry}, sl={sl})")
                )
                continue
            initial_risk = abs(entry - sl)
            out.append(Trade(
                id=str(r["id"]),
                user_id=str(r["user_id"])[:8],
                symbol=r["symbol"],
                side=r["side"].lower(),
                entry_price=entry,
                exit_price=float(r["exit_price"] or entry),
                quantity=float(r["quantity"] or 0),
                pnl_usd=float(r["pnl_usd"] or 0),
                fees_usd=float(r["fees_usd"] or 0),
                entry_fee_usd=float(m.get("entry_fee_usd") or (r["fees_usd"] or 0) / 2),
                opened_at=float(r["opened_ts"]),
                closed_at=float(r["closed_ts"]),
                exit_reason=str(m.get("exit_reason") or ""),
                grade=m.get("grade"),
                regime=m.get("regime"),
                trade_type=m.get("trade_type"),
                peak_mfe_r_at_exit=float(m.get("peak_mfe_r") or 0),
                stop_loss=sl,
                leverage=float(m.get("leverage") or 1),
                contract_size=float(m.get("contract_size") or 1),
                initial_risk=initial_risk,
                margin=float(m.get("margin") or 0),
            ))
        except (TypeError, ValueError) as e:
            PARSE_SKIPS.append((str(r.get("id", "?"))[:8], r.get("symbol", "?"),
                                f"parse error: {e}"))
    return out


# ── Delta candles ───────────────────────────────────────────────────────────
def _delta_symbol(symbol: str) -> Optional[str]:
    info = PRODUCT_MAP.get(symbol)
    if not info:
        return None
    return info.get("symbol")


def fetch_candles(symbol: str, start: int, end: int, cache_dir: Path) -> List[Dict[str, Any]]:
    """Fetch 1m candles from Delta India PROD. Cached per symbol-window."""
    delta_sym = _delta_symbol(symbol)
    if not delta_sym:
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{delta_sym}_{start}_{end}.json"
    cache_path = cache_dir / key
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass  # corrupt cache → re-fetch

    qs = urllib.parse.urlencode({
        "symbol": delta_sym,
        "resolution": "1m",
        "start": start,
        "end": end,
    })
    url = f"{DELTA_BASE}/v2/history/candles?{qs}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vnedge-cf-analyzer"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())
    except Exception as e:
        print(f"  delta fetch error {delta_sym} [{start}-{end}]: {e}", file=sys.stderr)
        return []

    candles = payload.get("result") or []
    # Sort ascending by time (Delta returns descending).
    candles.sort(key=lambda c: c["time"])
    cache_path.write_text(json.dumps(candles))
    time.sleep(RATE_LIMIT_SLEEP_S)
    return candles


# ── Counterfactual replay ──────────────────────────────────────────────────
def _r_from_price(side: str, entry: float, price: float, risk: float) -> float:
    if risk <= 0:
        return 0.0
    if side == "long":
        return (price - entry) / risk
    return (entry - price) / risk


def _qty_units(t: Trade) -> float:
    """How many price-units (e.g. SOL, ETH) the position holds.

    `quantity` is in lots; convert to base-units via contract_size.
    Falls back to margin*leverage/entry if quantity is missing.
    """
    if t.quantity > 0 and t.contract_size > 0:
        return t.quantity * t.contract_size
    if t.margin > 0 and t.leverage > 0 and t.entry_price > 0:
        return (t.margin * t.leverage) / t.entry_price
    return 0.0


def _exit_pnl_usd(t: Trade, exit_price: float) -> float:
    """Counterfactual NET P&L: gross (price move on units) minus
    actual entry fee (already paid) + new exit fee at taker rate."""
    units = _qty_units(t)
    if units <= 0:
        return 0.0
    if t.side == "long":
        gross = (exit_price - t.entry_price) * units
    else:
        gross = (t.entry_price - exit_price) * units
    notional_exit = exit_price * units
    exit_fee = notional_exit * (RT_TAKER_FEE_PCT / 2.0)  # one-side taker
    return gross - t.entry_fee_usd - exit_fee


def replay_trade(t: Trade, candles: List[Dict[str, Any]]) -> Optional[CounterfactualResult]:
    """Walk forward through 1m post-exit candles applying the NEW guard set.

    Order of guards per minute (matching live ordering in user_real_manager):
      1. SL hit (intra-bar high/low)
      2. dead_market  (age>=180s, quiet/sideways regime, peak<0.08R, current<-0.10R)
      3. unified dead_signal (Phase 5.20.8 — replaces quick/early/zombie/proof)
      4. mfe_pullback (peak>=0.30R AND current<=peak*0.40)
      5. exhaustion_wick (peak>=0.30R AND current>0 AND wick>60% body)
      6. time_decay 30min SCALP / 60min others
    """
    if len(candles) < MIN_REQUIRED_CANDLES:
        return None

    risk = t.initial_risk
    if risk <= 0:
        return None

    peak = float(t.peak_mfe_r_at_exit or 0)
    age0 = max(0.0, t.closed_at - t.opened_at)
    last_close: float = t.exit_price

    fired_reason: Optional[str] = None
    fired_price: Optional[float] = None
    fired_age: float = age0
    candles_used = 0

    for c in candles:
        ts = float(c["time"])
        # Only consume candles strictly after closed_at (the exit moment).
        if ts < t.closed_at:
            continue
        candles_used += 1
        bar_high = float(c["high"])
        bar_low = float(c["low"])
        bar_open = float(c["open"])
        bar_close = float(c["close"])
        last_close = bar_close

        # 1. SL hit — first because intra-bar prices can sweep through SL.
        sl_hit = (
            (t.side == "long" and bar_low <= t.stop_loss) or
            (t.side == "short" and bar_high >= t.stop_loss)
        )
        if sl_hit:
            fired_reason = "sl_hit"
            fired_price = t.stop_loss
            fired_age = age0 + (ts - t.closed_at) + 60
            break

        # Update peak using intra-bar extreme.
        if t.side == "long":
            bar_peak_r = _r_from_price("long", t.entry_price, bar_high, risk)
        else:
            bar_peak_r = _r_from_price("short", t.entry_price, bar_low, risk)
        if bar_peak_r > peak:
            peak = bar_peak_r

        # current_r is on close at the end of the minute.
        current_r = _r_from_price(t.side, t.entry_price, bar_close, risk)
        # Effective age at end of this bar.
        age_now = age0 + (ts - t.closed_at) + 60

        # 2. Dead market (kept guard — quiet regimes only).
        rg = (t.regime or "").lower()
        if (age_now >= 180 and rg in ("quiet", "low_liquidity", "mean_reversion", "")
                and peak < 0.08 and current_r < -0.10):
            fired_reason = "dead_market"
            fired_price = bar_close
            fired_age = age_now
            break

        # 3. Unified dead-signal (the new policy).
        kill = should_kill_dead_signal(
            age_sec=age_now,
            current_r=current_r,
            peak_mfe_r=peak,
            grade=t.grade,
            entry=t.entry_price,
            sl=t.stop_loss,
            trade_type=t.trade_type,
            regime=t.regime,
        )
        if kill:
            fired_reason = kill
            fired_price = bar_close
            fired_age = age_now
            break

        # 4. MFE pullback (kept guard).
        if peak >= 0.30 and current_r <= peak * 0.40:
            fired_reason = "mfe_pullback"
            fired_price = bar_close
            fired_age = age_now
            break

        # 5. Exhaustion wick (kept guard).
        if peak >= 0.30 and current_r > 0:
            body = abs(bar_close - bar_open)
            rng = bar_high - bar_low
            if rng > 0 and body > 0:
                if t.side == "long":
                    upper_wick = bar_high - max(bar_close, bar_open)
                    if upper_wick > body * 0.6:
                        fired_reason = "exhaustion_wick"
                        fired_price = bar_close
                        fired_age = age_now
                        break
                else:
                    lower_wick = min(bar_close, bar_open) - bar_low
                    if lower_wick > body * 0.6:
                        fired_reason = "exhaustion_wick"
                        fired_price = bar_close
                        fired_age = age_now
                        break

        # 6. Time decay backstop.
        max_age = 1800 if (t.trade_type or "").upper() == "SCALP" else 3600
        if age_now > max_age:
            fired_reason = f"time_decay_{int(age_now/60)}m"
            fired_price = bar_close
            fired_age = age_now
            break

    if fired_reason is None:
        # Survived the whole window — exit at last close as "window_end".
        fired_reason = "window_end"
        fired_price = last_close
        fired_age = age0 + POST_EXIT_WINDOW_MIN * 60

    pnl_cf = _exit_pnl_usd(t, fired_price)

    return CounterfactualResult(
        trade=t,
        cf_exit_price=float(fired_price),
        cf_exit_reason=fired_reason,
        cf_pnl_usd=float(pnl_cf),
        cf_age_sec=float(fired_age),
        cf_peak_mfe_r=float(peak),
        candles_seen=candles_used,
    )


# ── Stats ──────────────────────────────────────────────────────────────────
def bootstrap_ci(deltas: List[float], n: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    """Return (mean, low_2.5pct, high_97.5pct) of bootstrap resampled SUM."""
    if not deltas:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    k = len(deltas)
    sums: List[float] = []
    for _ in range(n):
        s = sum(deltas[rng.randrange(k)] for _ in range(k))
        sums.append(s)
    sums.sort()
    lo = sums[int(0.025 * n)]
    hi = sums[int(0.975 * n)]
    mean = sum(deltas)
    return (mean, lo, hi)


def win_rate(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    return 100.0 * sum(1 for p in pnls if p > 0) / len(pnls)


# ── Reporting ──────────────────────────────────────────────────────────────
def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):.2f}"


def fmt_signed(x: float) -> str:
    return f"{'+' if x >= 0 else ''}{fmt_money(x) if abs(x) >= 0.005 else '$0.00'}"


def render_report(
    results: List[CounterfactualResult],
    skipped: List[Tuple[Trade, str]],
    days: int,
    runtime_s: float,
) -> str:
    n = len(results)
    actual_pnls = [r.trade.pnl_usd for r in results]
    cf_pnls = [r.cf_pnl_usd for r in results]
    deltas = [b - a for a, b in zip(actual_pnls, cf_pnls)]

    actual_total = sum(actual_pnls)
    cf_total = sum(cf_pnls)
    delta_total = cf_total - actual_total

    actual_wr = win_rate(actual_pnls)
    cf_wr = win_rate(cf_pnls)

    actual_avg = actual_total / n if n else 0.0
    cf_avg = cf_total / n if n else 0.0

    mean_d, ci_lo, ci_hi = bootstrap_ci(deltas)

    # Cohort: original_reason → cf_reason
    transitions = collections.defaultdict(lambda: {"n": 0, "actual": 0.0, "cf": 0.0})
    for r in results:
        key = (r.trade.exit_reason, r.cf_exit_reason)
        d = transitions[key]
        d["n"] += 1
        d["actual"] += r.trade.pnl_usd
        d["cf"] += r.cf_pnl_usd

    # Cohort: grade × regime
    grade_regime = collections.defaultdict(lambda: {"n": 0, "actual": 0.0, "cf": 0.0})
    for r in results:
        key = (r.trade.grade or "?", r.trade.regime or "?")
        d = grade_regime[key]
        d["n"] += 1
        d["actual"] += r.trade.pnl_usd
        d["cf"] += r.cf_pnl_usd

    # Verdict
    pass_pnl = delta_total > KILL_DELTA_PNL_USD
    pass_wr = (cf_wr - actual_wr) > KILL_DELTA_WR_PP
    pass_ci = ci_lo > 0
    n_pass = sum([pass_pnl, pass_wr, pass_ci])
    if n_pass >= 2 and delta_total > 0:
        verdict_emoji, verdict = "[GREEN]", "SHIP IT"
    elif delta_total > 0 and n_pass >= 1:
        verdict_emoji, verdict = "[YELLOW]", "CAUTION"
    else:
        verdict_emoji, verdict = "[RED]", "DON'T SHIP"

    end_dt = datetime.utcnow()
    start_dt_str = (end_dt.replace(microsecond=0)
                    .replace(hour=0, minute=0, second=0)
                    .replace(day=end_dt.day - days if end_dt.day > days else 1)
                    .strftime("%Y-%m-%d"))
    end_dt_str = end_dt.strftime("%Y-%m-%d")

    out: List[str] = []
    out.append("# Wave 2 Counterfactual Analysis -- Phase 5.20.8 Exit Refactor")
    out.append("")
    out.append(f"Window: last {days} days ({start_dt_str} -> {end_dt_str})")
    out.append(f"Candidate trades processed: {n}  (skipped: {len(skipped)})")
    out.append(f"Runtime: {runtime_s:.1f}s")
    out.append("")
    out.append("## Aggregate Outcome")
    out.append("")
    out.append("| Metric | Actual (current) | Counterfactual (refactored) | Delta |")
    out.append("|---|---|---|---|")
    out.append(f"| Total P&L | {fmt_money(actual_total)} | {fmt_money(cf_total)} | {fmt_signed(delta_total)} |")
    out.append(f"| Win rate | {actual_wr:.1f}% | {cf_wr:.1f}% | {('+' if cf_wr>=actual_wr else '')}{cf_wr-actual_wr:.1f} pp |")
    out.append(f"| Avg P&L / trade | {fmt_money(actual_avg)} | {fmt_money(cf_avg)} | {fmt_signed(cf_avg-actual_avg)} |")
    out.append("")
    out.append(f"Bootstrap 95% CI on Delta P&L (n={BOOTSTRAP_RESAMPLES}): "
               f"[{fmt_money(ci_lo)}, {fmt_money(ci_hi)}]   mean={fmt_signed(mean_d)}")
    out.append("")

    out.append("## Exit Reason Transitions")
    out.append("")
    out.append("| original -> counterfactual | n | actual_pnl | cf_pnl | delta |")
    out.append("|---|---|---|---|---|")
    for (orig, cf), d in sorted(transitions.items(), key=lambda kv: -kv[1]["n"]):
        out.append(f"| {orig} -> {cf} | {d['n']} | {fmt_money(d['actual'])} "
                   f"| {fmt_money(d['cf'])} | {fmt_signed(d['cf']-d['actual'])} |")
    out.append("")

    out.append("## Per-cohort Delta (grade x regime)")
    out.append("")
    out.append("| grade | regime | n | actual_pnl | cf_pnl | delta | avg_delta |")
    out.append("|---|---|---|---|---|---|---|")
    for (g, rg), d in sorted(grade_regime.items(),
                             key=lambda kv: -(kv[1]["cf"] - kv[1]["actual"])):
        avg_d = (d["cf"] - d["actual"]) / max(1, d["n"])
        out.append(f"| {g} | {rg} | {d['n']} | {fmt_money(d['actual'])} | "
                   f"{fmt_money(d['cf'])} | {fmt_signed(d['cf']-d['actual'])} | "
                   f"{fmt_signed(avg_d)} |")
    out.append("")

    if skipped:
        out.append("## Skipped trades")
        out.append("")
        out.append("| trade_id | symbol | reason |")
        out.append("|---|---|---|")
        for t, why in skipped[:50]:
            out.append(f"| {t.id[:8]} | {t.symbol} | {why} |")
        if len(skipped) > 50:
            out.append(f"| ... | ... | (+{len(skipped)-50} more) |")
        out.append("")

    out.append("## Verdict")
    out.append("")
    out.append(f"### {verdict_emoji} {verdict}")
    out.append("")
    out.append("Per design-doc kill criteria:")
    out.append("")
    out.append(f"- Delta P&L > +${KILL_DELTA_PNL_USD:.0f} over {days} days?  "
               f"{'PASS' if pass_pnl else 'FAIL'}  (actual: {fmt_signed(delta_total)})")
    out.append(f"- Delta WR > +{KILL_DELTA_WR_PP:.0f}pp?                "
               f"{'PASS' if pass_wr else 'FAIL'}  (actual: {('+' if cf_wr>=actual_wr else '')}{cf_wr-actual_wr:.1f}pp)")
    out.append(f"- Lower bootstrap bound > $0?           "
               f"{'PASS' if pass_ci else 'FAIL'}  (actual: {fmt_money(ci_lo)})")
    out.append("")

    rec_lines = []
    if verdict == "SHIP IT":
        rec_lines.append(
            "The refactored exit policy delivers a positive P&L delta with a lower "
            "bootstrap bound above zero. Recommend proceeding to live A/B with "
            "standard 24-48h shadow validation before full rollout."
        )
    elif verdict == "CAUTION":
        rec_lines.append(
            "Counterfactual P&L is positive but at least one kill-criterion failed. "
            "The directional signal is right but the magnitude is borderline. "
            "Recommend extending the analysis window to 14 days or running a "
            "longer shadow A/B (>=72h) before promoting to live capital."
        )
    else:
        rec_lines.append(
            "The refactor either degrades aggregate P&L or fails the statistical "
            "significance bar. Do NOT ship as-is. Investigate cohort-level "
            "regressions (see grade x regime table) and tune patience / fee-floor "
            "constants before re-running."
        )
    out.append("Recommendation: " + " ".join(rec_lines))
    out.append("")
    return "\n".join(out)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--cache", default=".rollback/wave2-backtest/candle_cache",
                    type=str)
    ap.add_argument("--out", default=".rollback/wave2-backtest/counterfactual_report.md",
                    type=str)
    args = ap.parse_args()

    cache_dir = (_ROOT / args.cache).resolve()
    out_path = (_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1] Fetching candidates from DB (last {args.days}d)...")
    trades = fetch_candidates(args.days)
    print(f"    -> {len(trades)} candidates ({len(PARSE_SKIPS)} dropped at parse for missing fields)")

    results: List[CounterfactualResult] = []
    skipped: List[Tuple[Trade, str]] = []

    # Surface parse-time drops so the report shows the full universe.
    for tid, sym, reason in PARSE_SKIPS:
        skipped.append((Trade(
            id=tid, user_id="", symbol=sym, side="?", entry_price=0.0,
            exit_price=0.0, quantity=0.0, pnl_usd=0.0, fees_usd=0.0,
            entry_fee_usd=0.0, opened_at=0.0, closed_at=0.0,
            exit_reason="(parse-drop)", grade=None, regime=None,
            trade_type=None, peak_mfe_r_at_exit=0.0, stop_loss=0.0,
            leverage=0.0, contract_size=0.0, initial_risk=0.0, margin=0.0,
        ), reason))

    for i, t in enumerate(trades, 1):
        when = datetime.utcfromtimestamp(t.closed_at).strftime("%H:%M")
        print(f"[{i:>2}/{len(trades)}] {t.symbol} {t.side} closed {when} "
              f"({t.exit_reason}, peak={t.peak_mfe_r_at_exit:.3f}R)...", end=" ")

        if t.symbol not in PRODUCT_MAP:
            print("SKIP (no PRODUCT_MAP entry)")
            skipped.append((t, "no PRODUCT_MAP entry"))
            continue

        start = int(t.closed_at)
        end = start + POST_EXIT_WINDOW_MIN * 60
        candles = fetch_candles(t.symbol, start, end, cache_dir)
        if len(candles) < MIN_REQUIRED_CANDLES:
            print(f"SKIP ({len(candles)} candles < {MIN_REQUIRED_CANDLES})")
            skipped.append((t, f"only {len(candles)} candles available"))
            continue

        res = replay_trade(t, candles)
        if res is None:
            print("SKIP (replay returned None)")
            skipped.append((t, "replay returned None (bad risk?)"))
            continue
        results.append(res)
        delta = res.cf_pnl_usd - t.pnl_usd
        print(f"-> {res.cf_exit_reason} cf_pnl={fmt_money(res.cf_pnl_usd)} "
              f"d={fmt_signed(delta)} (peak={res.cf_peak_mfe_r:.3f}R)")

    runtime = time.time() - t0
    print()
    print(f"Processed {len(results)}, skipped {len(skipped)}, runtime={runtime:.1f}s")

    md = render_report(results, skipped, args.days, runtime)
    out_path.write_text(md)
    print(f"\nReport saved: {out_path}\n")
    print("=" * 78)
    print(md)


if __name__ == "__main__":
    main()

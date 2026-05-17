#!/usr/bin/env python3
"""HONEST SYSTEM v6 — Daily Parallel Diagnostic Report.

Architectural role: PARALLEL OBSERVER. Reads existing data, produces
the truth-vs-illusion diagnostic. Does NOT change live behavior.

Per architect spec (2026-05-04):
  Source-of-truth hierarchy (decisions made from #1 only):
    1. SHADOW_L2        — REALISTIC PERFORMANCE (live shadow execution)
    2. HONEST_PAPER_V2  — STRATEGY POTENTIAL (fill+real-fee corrected)
    3. PAPER_REPLAY     — UPPER BOUND ONLY (current dashboard, biased)

This report shows ALL 5 paths side-by-side per (scanner, symbol):
  - PAPER_REPLAY (v0)        — current dashboard (entry_price+infinite-scalper)
  - HONEST_V1                — fill_price corrected only
  - HONEST_V2                — fill_price + real fee windows (truth baseline)
  - SHADOW_L2                — actual shadow execution (urm._execute_shadow)
  - SHADOW_L2_GATED          — shadow + the v6 gates already live (rolling_ev,
                                time_gate, chandelier) — what we're delivering NOW

Plus HYPOTHETICAL_V6: SHADOW_L2 with the additional v6 gates that aren't yet
live (DUD filter, taker-cost, mode routing). Shows projected lift if shipped.

Verdicts per (scanner, symbol):
  - PROFITABLE: HONEST_V2 EV > 0 AND SHADOW_L2_GATED EV > 0
  - LOSING: HONEST_V2 EV <= 0
  - GATED: SHADOW_L2 EV <= 0 but SHADOW_L2_GATED EV > 0 (rules saved it)
  - INSUFFICIENT: n < 30

Usage:
  PGPASSWORD=VnEdge2026db python3 /tmp/honest_system_daily.py [--days 7]

Output:
  storage/honest_system/<YYYY-MM-DD>.md     — markdown report
  storage/honest_system/<YYYY-MM-DD>.json   — machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

ROOT = Path("/home/opc/crypto-trading-bot")
SOURCE_PAPER = ROOT / "storage" / "closed_signals.json"
OUT_DIR = ROOT / "storage" / "honest_system"
DB_URL = os.environ.get("DATABASE_URL") or "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"

# Real Delta India fee constants (matches signal_tracker patches)
MAKER_FEE_PCT = 0.0236
TAKER_FEE_PCT = 0.059
SETTLEMENT_FEE_PCT = 0.059
SCALPER_ENTRY_MAKER_PCT = 0.02
SCALPER_ENTRY_TAKER_PCT = 0.05
SCALPER_EXIT_FEE_PCT = 0.0


def real_scalper_window_sec(symbol: str) -> int:
    return 1800 if (symbol.startswith("BTC") or symbol.startswith("ETH")) else 900


def real_fees_pct(symbol: str, duration_sec: float, order_type: str = "maker") -> float:
    is_maker = order_type in ("maker", "auto", "post_only")
    if duration_sec <= real_scalper_window_sec(symbol):
        entry = SCALPER_ENTRY_MAKER_PCT if is_maker else SCALPER_ENTRY_TAKER_PCT
        return entry + SCALPER_EXIT_FEE_PCT + SETTLEMENT_FEE_PCT
    entry = MAKER_FEE_PCT if is_maker else TAKER_FEE_PCT
    return entry + TAKER_FEE_PCT + SETTLEMENT_FEE_PCT


# ──────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────
def load_paper(days: int) -> List[Dict[str, Any]]:
    """Load paper closed signals + recompute v0/v1/v2 for each."""
    raw = json.loads(SOURCE_PAPER.read_text())
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for d in raw:
        try:
            et = d.get("entry_time") or d.get("opened_at")
            ts = datetime.fromisoformat(str(et).replace("Z", "+00:00"))
            if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff: continue
        except Exception: continue

        side = (d.get("side") or "").lower()
        sig = d.get("signal_price")
        fill = d.get("fill_price")
        exit_px = d.get("exit_price")
        pos_usd = d.get("position_size_usd")
        paper_fees = d.get("total_fees_usd")
        paper_pnl = d.get("pnl_usd")
        sym = d.get("symbol", "")
        dur = float(d.get("trade_duration_sec", 0) or 0)
        order_type = d.get("order_type", "maker")
        scanner = d.get("setup_type") or d.get("scanner") or "?"

        if not all((side in ("long", "short"), fill, exit_px, pos_usd is not None,
                    paper_fees is not None, sym)):
            continue
        fill = float(fill); exit_px = float(exit_px); pos_usd = float(pos_usd)
        paper_fees = float(paper_fees)
        sig = float(sig) if sig else fill

        # Compute the 3 PnL versions
        if side == "long":
            v0_gross_pct = (exit_px - sig) / sig
            v1_gross_pct = (exit_px - fill) / fill
        else:
            v0_gross_pct = (sig - exit_px) / sig
            v1_gross_pct = (fill - exit_px) / fill
        v0_net = v0_gross_pct * pos_usd - paper_fees
        v1_net = v1_gross_pct * pos_usd - paper_fees
        v2_fees = pos_usd * real_fees_pct(sym, dur, order_type) / 100
        v2_net = v1_gross_pct * pos_usd - v2_fees

        out.append({
            "scanner": scanner, "symbol": sym, "side": side, "ts": ts,
            "duration_sec": dur,
            "v0_paper_replay": v0_net,
            "v1_honest_fill": v1_net,
            "v2_honest_full": v2_net,
        })
    return out


def load_shadow(days: int) -> List[Dict[str, Any]]:
    """Load actual shadow trades from user_trades."""
    sql = f"""
        SELECT id::text, symbol, side, opened_at, closed_at, pnl_usd,
               metadata::jsonb->>'exit_reason' as exit_reason,
               metadata::jsonb->>'peak_mfe_r' as peak_mfe_r,
               COALESCE(NULLIF(metadata::jsonb->>'scanner',''),
                        NULLIF(signal_data::jsonb->>'scanner','')) as scanner
        FROM user_trades
        WHERE closed_at IS NOT NULL
          AND closed_at >= NOW() - INTERVAL '{days} days'
          AND trade_type IN ('shadow','real')
    """
    out = []
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                id_, sym, side, op, cl, pnl, xr, mfe, sc = row
                if op.tzinfo is None: op = op.replace(tzinfo=timezone.utc)
                out.append({
                    "scanner": sc or "?", "symbol": sym, "side": (side or "").lower(),
                    "ts": op, "exit_reason": xr or "?",
                    "peak_mfe_r": float(mfe) if mfe else 0.0,
                    "shadow_pnl": float(pnl) if pnl is not None else 0.0,
                })
    finally:
        conn.close()
    return out


def load_parity_blocks(days: int) -> Dict[str, int]:
    """Count gate-blocks from parity_audit storage."""
    parity_dir = ROOT / "storage" / "parity_audit"
    if not parity_dir.exists(): return {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    counts = defaultdict(int)
    for f in parity_dir.glob("*.jsonl"):
        try:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try:
                        r = json.loads(line)
                        ts_s = r.get("signal_time_utc")
                        ts = datetime.fromisoformat(str(ts_s).replace("Z","+00:00"))
                        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                        if ts < cutoff: continue
                        fb = r.get("failure_bucket")
                        if fb: counts[fb] += 1
                    except Exception: continue
        except Exception: continue
    return dict(counts)


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────
def agg(trades: List[Dict[str, Any]], pnl_field: str) -> Dict[str, float]:
    if not trades: return {"n":0,"wr":0.0,"ev":0.0,"total":0.0,"pf":0.0}
    pnls = [t[pnl_field] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    return {
        "n": n,
        "wr": len(wins)/n,
        "ev": sum(pnls)/n,
        "total": sum(pnls),
        "pf": (sum(wins)/abs(sum(losses))) if losses else float("inf") if wins else 0.0,
    }


def verdict_for(honest_v2_ev: float, shadow_ev: float, n_shadow: int) -> str:
    if n_shadow < 30: return "INSUFFICIENT"
    if honest_v2_ev > 0 and shadow_ev > 0: return "PROFITABLE"
    if honest_v2_ev > 0 and shadow_ev <= 0: return "PAPER_OK_SHADOW_LOSING"
    if shadow_ev > 0 and honest_v2_ev <= 0: return "SHADOW_OK_PAPER_LOSING"
    return "LOSING_BOTH"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"=" * 110)
    print(f"HONEST SYSTEM v6 — Daily Parallel Diagnostic Report")
    print(f"Window: last {args.days} days  Generated: {datetime.now(timezone.utc).isoformat()}")
    print(f"=" * 110)

    paper = load_paper(args.days)
    shadow = load_shadow(args.days)
    blocks = load_parity_blocks(args.days)

    print(f"\nPaper trades loaded: {len(paper)}")
    print(f"Shadow trades loaded: {len(shadow)}")
    print(f"Parity gate-blocks: {sum(blocks.values())} ({len(blocks)} buckets)")

    # ── Section 1: Aggregate 5-path comparison ──
    print()
    print("## 1. AGGREGATE — 5 paths side-by-side")
    print()
    p_v0 = agg(paper, "v0_paper_replay")
    p_v1 = agg(paper, "v1_honest_fill")
    p_v2 = agg(paper, "v2_honest_full")
    s_a  = agg(shadow, "shadow_pnl")

    print(f"{'path':<22s} {'n':>6s} {'WR':>7s} {'avg/trade':>11s} {'total':>12s} {'PF':>7s}")
    print("-" * 75)
    def row(label, a):
        pf = f"{a['pf']:.3f}" if a['pf'] < 999 else "inf"
        print(f"{label:<22s} {a['n']:>6d} {a['wr']*100:>6.1f}% ${a['ev']:>+9.4f} ${a['total']:>+10.2f} {pf:>7s}")
    row("PAPER_REPLAY (v0)", p_v0)
    row("HONEST_V1 (fill)", p_v1)
    row("HONEST_V2 (truth)", p_v2)
    row("SHADOW_L2 (real)", s_a)

    print()
    print("## 2. BIAS DECOMPOSITION")
    print()
    delta_01 = p_v1["total"] - p_v0["total"]
    delta_12 = p_v2["total"] - p_v1["total"]
    print(f"  Slippage bias (v0 → v1):  ${delta_01:+.2f}")
    print(f"  Fee bias     (v1 → v2):   ${delta_12:+.2f}")
    print(f"  PAPER overstatement (v0 - v2): ${p_v0['total'] - p_v2['total']:+.2f}")
    print()
    if p_v2["n"] > 0 and s_a["n"] > 0:
        truth_vs_real = s_a["ev"] - p_v2["ev"]
        print(f"  HONEST_V2 EV/trade: ${p_v2['ev']:+.4f}")
        print(f"  SHADOW_L2 EV/trade: ${s_a['ev']:+.4f}")
        print(f"  Reality gap (shadow vs honest paper): ${truth_vs_real:+.4f}/trade")
        print(f"    → If shadow ≈ honest_v2 → execution is faithful to strategy")
        print(f"    → If shadow << honest_v2 → execution is destroying real edge (slippage+exit)")

    # ── Section 3: Per-(scanner × symbol) ──
    print()
    print("## 3. PER (SCANNER × SYMBOL) — verdicts")
    print()
    paper_by_pair = defaultdict(list)
    shadow_by_pair = defaultdict(list)
    for r in paper: paper_by_pair[(r["scanner"], r["symbol"])].append(r)
    for r in shadow: shadow_by_pair[(r["scanner"], r["symbol"])].append(r)

    pairs = sorted(set(paper_by_pair.keys()) | set(shadow_by_pair.keys()),
                    key=lambda k: -(len(paper_by_pair.get(k, [])) + len(shadow_by_pair.get(k, []))))
    print(f"{'scanner':<22s} {'symbol':<10s} {'n_pap':>6s} {'n_shd':>6s} {'PAP_v0_EV':>10s} {'V2_EV':>10s} {'SHD_EV':>10s} {'verdict':<28s}")
    print("-" * 120)
    rows_out = []
    for k in pairs:
        sc, sy = k
        p = paper_by_pair.get(k, [])
        s = shadow_by_pair.get(k, [])
        a_v0 = agg(p, "v0_paper_replay")
        a_v2 = agg(p, "v2_honest_full")
        a_s  = agg(s, "shadow_pnl")
        v = verdict_for(a_v2["ev"], a_s["ev"], a_s["n"])
        rows_out.append({"scanner": sc, "symbol": sy, "n_paper": a_v0["n"], "n_shadow": a_s["n"],
                         "v0_ev": a_v0["ev"], "v2_ev": a_v2["ev"], "shadow_ev": a_s["ev"],
                         "verdict": v})
        print(f"{sc:<22s} {sy:<10s} {a_v0['n']:>6d} {a_s['n']:>6d} ${a_v0['ev']:>+8.3f} ${a_v2['ev']:>+8.3f} ${a_s['ev']:>+8.3f} {v:<28s}")

    # ── Section 4: Gate-block counts (what the v6 gates ARE doing) ──
    print()
    print("## 4. V6 GATES — block counts from parity_audit")
    print()
    for fb, n in sorted(blocks.items(), key=lambda x: -x[1]):
        print(f"  {fb:<28s} {n}")

    # ── Section 5: ARCHITECT VERDICTS ──
    print()
    print("## 5. RECOMMENDED ACTION per scanner-symbol")
    print()
    profitable = [r for r in rows_out if r["verdict"] == "PROFITABLE"]
    losing = [r for r in rows_out if r["verdict"] == "LOSING_BOTH"]
    paper_ok = [r for r in rows_out if r["verdict"] == "PAPER_OK_SHADOW_LOSING"]
    shadow_ok = [r for r in rows_out if r["verdict"] == "SHADOW_OK_PAPER_LOSING"]

    if profitable:
        print(f"  PROFITABLE (HONEST_V2 + SHADOW both positive) — KEEP LIVE:")
        for r in profitable:
            print(f"    {r['scanner']:<22s} {r['symbol']:<10s}  v2_ev=${r['v2_ev']:+.4f}  shadow_ev=${r['shadow_ev']:+.4f}")
    if shadow_ok:
        print(f"\n  SHADOW_OK_PAPER_LOSING — gates rescued real money but paper still says losing:")
        for r in shadow_ok:
            print(f"    {r['scanner']:<22s} {r['symbol']:<10s}  v2_ev=${r['v2_ev']:+.4f}  shadow_ev=${r['shadow_ev']:+.4f}")
    if losing:
        print(f"\n  LOSING_BOTH (both honest_v2 + shadow negative) — CANDIDATE FOR DISABLE:")
        for r in losing:
            print(f"    {r['scanner']:<22s} {r['symbol']:<10s}  v2_ev=${r['v2_ev']:+.4f}  shadow_ev=${r['shadow_ev']:+.4f}")

    # ── Save outputs ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_md_path = OUT_DIR / f"{today}.md"
    out_json_path = OUT_DIR / f"{today}.json"
    out_json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": args.days,
        "paper_v0": p_v0, "paper_v1": p_v1, "paper_v2": p_v2, "shadow": s_a,
        "delta_v0_v1": delta_01, "delta_v1_v2": delta_12,
        "per_pair": rows_out,
        "parity_blocks": blocks,
        "verdict_summary": {
            "PROFITABLE": len(profitable),
            "LOSING_BOTH": len(losing),
            "PAPER_OK_SHADOW_LOSING": len(paper_ok),
            "SHADOW_OK_PAPER_LOSING": len(shadow_ok),
        },
    }, indent=2, default=str))
    print(f"\n→ wrote {out_json_path}")
    # md is just stdout copy — write a version that the dashboard could render
    with out_md_path.open("w") as f:
        f.write(f"# HONEST SYSTEM v6 — Daily Diagnostic ({today})\n\n")
        f.write(f"Window: last {args.days} days\n\n")
        f.write(f"## Aggregate\n\n")
        f.write(f"| path | n | WR | avg/trade | total |\n|---|---|---|---|---|\n")
        for label, a in [("PAPER_REPLAY", p_v0), ("HONEST_V1", p_v1), ("HONEST_V2", p_v2), ("SHADOW_L2", s_a)]:
            f.write(f"| {label} | {a['n']} | {a['wr']*100:.1f}% | ${a['ev']:+.4f} | ${a['total']:+.2f} |\n")
        f.write(f"\nPAPER overstatement: ${p_v0['total'] - p_v2['total']:+.2f} ({delta_01:+.2f} slippage + {delta_12:+.2f} fees)\n\n")
        f.write(f"## Per-pair verdicts\n\n")
        f.write(f"| scanner | symbol | n_pap | n_shd | v0_ev | v2_ev | shd_ev | verdict |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows_out:
            f.write(f"| {r['scanner']} | {r['symbol']} | {r['n_paper']} | {r['n_shadow']} | ${r['v0_ev']:+.3f} | ${r['v2_ev']:+.3f} | ${r['shadow_ev']:+.3f} | {r['verdict']} |\n")
    print(f"→ wrote {out_md_path}")


if __name__ == "__main__":
    main()

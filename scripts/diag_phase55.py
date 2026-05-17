#!/usr/bin/env python3
"""Phase 5.5 Multi-Diagnosis.

Breaks down Phase 5.5 cohort by per-feature markers so we can attribute
net improvement to:
  N1 — Percentage-based maker offset (entry_exec_mode)
  N2 — Iterative SL retry (n2_sl_retries)
  N3 — Dynamic BE-lock floor (n3_min_lock_r)

Plus overall KPIs: WR, avg NET, fee % of gross, peak distribution.

Usage:
  python3 diag_phase55.py [--hours 12]
"""

import argparse
import os
import sys
import subprocess
from collections import defaultdict

PG = "PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -A -F '|' -t -c"


def run_psql(q):
    """Shell out to psql to avoid psycopg2 dependency on VM."""
    import json
    cmd = f'{PG} "{q}"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: {res.stderr}", file=sys.stderr)
        sys.exit(1)
    return [ln for ln in res.stdout.strip().splitlines() if ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=12)
    args = ap.parse_args()

    q = f"""
    SELECT
      id,
      symbol,
      side,
      pnl_usd,
      fees_usd,
      quantity,
      entry_price,
      COALESCE(metadata::jsonb->>'grade','?') AS grade,
      COALESCE(metadata::jsonb->>'regime','?') AS regime,
      COALESCE(metadata::jsonb->>'exit_reason','?') AS exit_reason,
      COALESCE((metadata::jsonb->>'peak_mfe_r')::numeric, 0) AS peak_r,
      COALESCE((metadata::jsonb->>'entry_exec_mode')::text, '?') AS entry_mode,
      COALESCE((metadata::jsonb->>'n2_sl_retries')::int, 0) AS n2_retries,
      COALESCE((metadata::jsonb->>'n3_min_lock_r')::numeric, 0.30) AS n3_lock,
      COALESCE((metadata::jsonb->>'fee_pct_of_gross')::numeric, 0) AS fee_pct,
      EXTRACT(EPOCH FROM (closed_at - opened_at)) AS dur_s
    FROM user_trades
    WHERE closed_at >= NOW() - INTERVAL '{args.hours} hours'
      AND COALESCE(metadata::jsonb->>'phase','') = '5.5'
      AND pnl_usd IS NOT NULL
    ORDER BY opened_at ASC
    """
    rows = run_psql(q)

    if not rows:
        print(f"No Phase 5.5 trades in last {args.hours}h.")
        print("Run again after trades accumulate under 5.5 phase tag.")
        return

    # Parse all trades
    trades = []
    for ln in rows:
        parts = ln.split('|')
        if len(parts) < 15:
            continue
        try:
            trades.append({
                'id': parts[0],
                'symbol': parts[1],
                'side': parts[2],
                'pnl': float(parts[3] or 0),
                'fees': float(parts[4] or 0),
                'qty': float(parts[5] or 0),
                'entry': float(parts[6] or 0),
                'grade': parts[7],
                'regime': parts[8],
                'exit_reason': parts[9],
                'peak_r': float(parts[10] or 0),
                'entry_mode': parts[11] or '?',
                'n2_retries': int(parts[12] or 0),
                'n3_lock': float(parts[13] or 0.30),
                'fee_pct': float(parts[14] or 0),
                'dur_s': float(parts[15] or 0) if len(parts) > 15 else 0,
            })
        except (ValueError, IndexError):
            continue

    n = len(trades)
    net = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)

    print()
    print("=" * 78)
    print(f"  PHASE 5.5 MULTI-DIAGNOSIS — {n} trades in last {args.hours}h")
    print("=" * 78)
    print(f"  OVERALL  NET: ${net:+.2f}  |  W:{wins} L:{losses}  |  WR: {100*wins/max(n,1):.1f}%")
    print(f"  Avg NET/trade: ${net/n:+.3f}  |  Avg duration: {sum(t['dur_s'] for t in trades)/n:.0f}s")
    print()

    # ========== N1: ENTRY MODE BREAKDOWN ==========
    print("── N1 · Entry Execution Mode ──")
    by_mode = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0, 'fees': 0.0})
    for t in trades:
        m = t['entry_mode']
        by_mode[m]['n'] += 1
        by_mode[m]['pnl'] += t['pnl']
        by_mode[m]['fees'] += t['fees']
        if t['pnl'] > 0:
            by_mode[m]['wins'] += 1
    print(f"  {'MODE':<18} {'n':<4} {'net':<10} {'avg':<8} {'wr':<6} {'avg_fees':<10}")
    print(f"  {'-'*18} {'-'*4} {'-'*10} {'-'*8} {'-'*6} {'-'*10}")
    for mode, d in sorted(by_mode.items(), key=lambda x: -x[1]['n']):
        avg = d['pnl'] / max(d['n'], 1)
        wr = 100 * d['wins'] / max(d['n'], 1)
        avg_fees = d['fees'] / max(d['n'], 1)
        print(f"  {mode:<18} {d['n']:<4} ${d['pnl']:>+7.2f}  ${avg:>+5.3f}  {wr:>4.0f}%   ${avg_fees:.3f}")

    maker_count = sum(d['n'] for m, d in by_mode.items() if 'maker' in m.lower())
    taker_count = by_mode.get('market_taker', {'n': 0})['n']
    print(f"\n  Maker fill rate: {100*maker_count/max(n,1):.1f}% (target ≥60% post-N1)")
    print(f"  Market_taker fallthrough: {100*taker_count/max(n,1):.1f}%")
    print()

    # ========== N2: SL RETRY ATTEMPTS ==========
    print("── N2 · SL Retry Usage ──")
    by_retry = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'peak': 0.0})
    for t in trades:
        r = t['n2_retries']
        by_retry[r]['n'] += 1
        by_retry[r]['pnl'] += t['pnl']
        by_retry[r]['peak'] += t['peak_r']
    print(f"  {'retries':<10} {'n':<4} {'net':<10} {'avg':<8} {'avg_peak':<10}")
    print(f"  {'-'*10} {'-'*4} {'-'*10} {'-'*8} {'-'*10}")
    for r in sorted(by_retry.keys()):
        d = by_retry[r]
        avg = d['pnl'] / max(d['n'], 1)
        avg_peak = d['peak'] / max(d['n'], 1)
        label = '0 (primary)' if r == 0 else f'{r} retries'
        print(f"  {label:<10} {d['n']:<4} ${d['pnl']:>+7.2f}  ${avg:>+5.3f}  {avg_peak:.2f}R")

    retry_needed = sum(d['n'] for r, d in by_retry.items() if r > 0)
    print(f"\n  Retry needed on {100*retry_needed/max(n,1):.1f}% of SL updates")
    print(f"  → Higher = more fast-tape; each retry is a race we won under N2")
    print()

    # ========== N3: DYNAMIC LOCK FLOOR ==========
    print("── N3 · Dynamic BE-Lock Floor (min_lock_r) ──")
    lock_buckets = defaultdict(lambda: {'n': 0, 'pnl': 0.0})
    for t in trades:
        lr = t['n3_lock']
        if lr < 0.32:
            b = '0.30 (static fallback)'
        elif lr < 0.40:
            b = '0.30–0.39 (slightly wider)'
        elif lr < 0.50:
            b = '0.40–0.49 (tight SL trade)'
        else:
            b = '0.50+ (very tight SL)'
        lock_buckets[b]['n'] += 1
        lock_buckets[b]['pnl'] += t['pnl']
    print(f"  {'bucket':<28} {'n':<4} {'net':<10} {'avg':<8}")
    print(f"  {'-'*28} {'-'*4} {'-'*10} {'-'*8}")
    for b, d in sorted(lock_buckets.items()):
        avg = d['pnl'] / max(d['n'], 1)
        print(f"  {b:<28} {d['n']:<4} ${d['pnl']:>+7.2f}  ${avg:>+5.3f}")
    print()

    # ========== FEE DRAG COHORT ==========
    print("── Fee drag (% of gross) distribution ──")
    fee_buckets = {'<50%': 0, '50-75%': 0, '75-100%': 0, '>100%': 0}
    for t in trades:
        fp = t['fee_pct']
        if fp < 50: fee_buckets['<50%'] += 1
        elif fp < 75: fee_buckets['50-75%'] += 1
        elif fp < 100: fee_buckets['75-100%'] += 1
        else: fee_buckets['>100%'] += 1
    for b, cnt in fee_buckets.items():
        pct = 100 * cnt / max(n, 1)
        bar = '█' * int(pct / 3)
        print(f"  {b:<10} {cnt:>3} ({pct:>4.1f}%)  {bar}")
    print()

    # ========== EXIT REASON BREAKDOWN ==========
    print("── Exit reason × outcome ──")
    by_reason = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0})
    for t in trades:
        r = t['exit_reason']
        by_reason[r]['n'] += 1
        by_reason[r]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_reason[r]['wins'] += 1
    print(f"  {'exit_reason':<22} {'n':<4} {'net':<10} {'avg':<8} {'wr':<6}")
    print(f"  {'-'*22} {'-'*4} {'-'*10} {'-'*8} {'-'*6}")
    for r, d in sorted(by_reason.items(), key=lambda x: x[1]['pnl']):
        avg = d['pnl'] / max(d['n'], 1)
        wr = 100 * d['wins'] / max(d['n'], 1)
        print(f"  {r:<22} {d['n']:<4} ${d['pnl']:>+7.2f}  ${avg:>+5.3f}  {wr:>4.0f}%")
    print()

    # ========== ATTRIBUTION SUMMARY ==========
    print("=" * 78)
    print("VERDICT (data-driven)")
    print("=" * 78)
    mt_count = by_mode.get('market_taker', {'n': 0})['n']
    maker_aggr_count = sum(d['n'] for m, d in by_mode.items() if 'aggr_1bp' in m or 'aggr_2bp' in m)
    if n >= 10:
        if maker_aggr_count >= n * 0.40:
            print(f"  ✅ N1 WORKING: {100*maker_aggr_count/n:.0f}% maker_aggr rate (target ≥40%)")
        else:
            print(f"  ⚠️  N1 BORDERLINE: only {100*maker_aggr_count/n:.0f}% maker_aggr (target ≥40%)")
        if retry_needed > 0:
            retry_success = sum(1 for t in trades if t['n2_retries'] > 0 and t['pnl'] > -1.0)
            print(f"  ✅ N2 ACTIVE: {retry_needed} trades needed retry, {retry_success} recovered")
        else:
            print(f"  ℹ️  N2 UNTESTED: no trails needed retry (slow tape)")
        n3_dynamic = sum(1 for t in trades if t['n3_lock'] > 0.32)
        if n3_dynamic > 0:
            print(f"  ✅ N3 ACTIVE: {n3_dynamic} trades used dynamic lock floor >0.30R")
        else:
            print(f"  ℹ️  N3 UNTESTED: all trades had default 0.30R lock (wide SLs)")
        if net/n > -0.30:
            print(f"  ✅ OVERALL: avg NET/trade ${net/n:+.3f} — on track for break-even")
        else:
            print(f"  ⚠️  OVERALL: avg NET/trade ${net/n:+.3f} — below break-even threshold")
    else:
        print(f"  ⏳ NEED MORE DATA: only {n} trades in window, need ≥10 for confident attribution")
    print()


if __name__ == "__main__":
    main()

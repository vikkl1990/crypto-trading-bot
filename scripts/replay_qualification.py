#!/usr/bin/env python3
"""T2.2 — Paper-signal qualification replay.

Replays the last N days of paper-tracked signals through demo's
qualify_signal gates and produces a per-gate rejection report bucketed
by paper's actual outcome.

Answers the question: "Of the paper signals that WON, how many would
demo's qualification have rejected, and at which gate?"

If a specific gate is rejecting the majority of paper winners, that gate
is the admission leak. If no single gate dominates, the gap is in
EXECUTION (fill quality, MFE capture, exit timing), not admission.

Usage:
    ./scripts/replay_qualification.py [--days 7] [--user admin|niranjan]

Output:
    - Summary table of rejection rates per gate
    - Breakdown: how many paper WINNERS each gate rejected (admission leak)
    - Breakdown: how many paper LOSERS each gate rejected (correctly filtered)
    - Net edge: for each gate, (winners_rejected * avg_paper_win) -
                (losers_rejected * avg_paper_loss) = $$$ left on table

Design:
    - Parses closed_signals_archive.jsonl (single source of truth for paper)
    - Reconstructs a signal dict matching what orchestrator would pass
    - Walks the 11 qualification gates in order (first-fail wins)
    - Does NOT hit DB or Delta (pure replay, safe to run anytime)
    - Ignores time-dependent gates (cohort_veto, daily_limit, max_open)
      because these depend on real-time state we can't reconstruct;
      flagged in output as "NOT CHECKED"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Default paths (override with env)
PAPER_JOURNAL = os.environ.get(
    "PAPER_JOURNAL",
    "/home/opc/crypto-trading-bot/storage/closed_signals_archive.jsonl",
)

# Must match execution/user_real_manager.py qualify_signal defaults
DEFAULT_ML_THRESHOLD = 0.55
DEFAULT_MIN_CONFIDENCE = 45  # Phase 5.3.2
DEFAULT_GRADE_ALLOWED = ("A+", "A", "B")  # Phase 5.3.2: allow B gated on ml
DEFAULT_GRADE_B_ML_FLOOR = 0.60  # B requires ml >= 0.60 (tighter than default 0.55)
DEFAULT_MAX_SL_PCT = 0.04  # T4.2 margin guard

# PRODUCT_MAP entries with demo_id > 0 (testnet-supported symbols only).
# Mirrors exchange/delta_client.py as of 2026-04-23.
DEMO_SUPPORTED = {
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "SHIB/USDT",
}


# ── Gate checks (each returns (passed, reject_reason_or_empty)) ────────

def check_venue(sig: dict) -> Tuple[bool, str]:
    sym = sig.get("symbol", "")
    if sym not in DEMO_SUPPORTED:
        return False, f"user_venue_unsupported:{sym}"
    return True, ""


def check_confidence(sig: dict) -> Tuple[bool, str]:
    conf = float(sig.get("confidence", 0) or 0)
    if conf < DEFAULT_MIN_CONFIDENCE:
        return False, f"user_conf_floor:{conf}<{DEFAULT_MIN_CONFIDENCE}"
    return True, ""


def check_grade(sig: dict) -> Tuple[bool, str]:
    g = sig.get("grade", "C") or "C"
    if g not in DEFAULT_GRADE_ALLOWED:
        return False, f"user_grade_filter:{g}"
    # Phase 5.3.2: Grade B needs ml >= 0.60 compensating floor.
    if g == "B":
        meta = sig.get("metadata") or {}
        ml = float(meta.get("ml_probability", 0) or sig.get("ml_probability", 0) or 0)
        if ml > 0 and ml < DEFAULT_GRADE_B_ML_FLOOR:
            return False, f"user_grade_B_ml_floor:{ml:.3f}<{DEFAULT_GRADE_B_ML_FLOOR}"
    return True, ""


def check_ml(sig: dict) -> Tuple[bool, str]:
    meta = sig.get("metadata") or {}
    ml_prob = meta.get("ml_probability")
    if ml_prob is None:
        # Fall back to top-level (paper sometimes logs it here)
        ml_prob = sig.get("ml_probability")
    if ml_prob is not None and float(ml_prob) > 0 and float(ml_prob) < DEFAULT_ML_THRESHOLD:
        return False, f"user_ml_floor:{ml_prob:.3f}<{DEFAULT_ML_THRESHOLD}"
    return True, ""


def check_margin_guard(sig: dict) -> Tuple[bool, str]:
    ep = float(sig.get("entry_price", 0) or 0)
    sl = float(sig.get("stop_loss", 0) or 0)
    if ep > 0 and sl > 0:
        sl_pct = abs(ep - sl) / ep
        if sl_pct > DEFAULT_MAX_SL_PCT:
            return False, f"user_margin_guard:sl={sl_pct*100:.2f}%>4%"
    return True, ""


GATES = [
    ("venue", check_venue),
    ("confidence", check_confidence),
    ("grade", check_grade),
    ("ml", check_ml),
    ("margin_guard", check_margin_guard),
]

NOT_CHECKED = [
    "user_enabled", "user_live_balance_floor", "user_cb_tripped",
    "user_daily_limit", "user_symbol_filter", "user_max_open",
    "user_duplicate", "user_cohort_veto", "user_live_emergency_halt",
]


def replay_signal(sig: dict) -> Tuple[str, str]:
    """Run sig through gates in order. Return (first_fail_gate, reason).
    If all pass, returns ('qualified', '')."""
    for gate_name, fn in GATES:
        passed, reason = fn(sig)
        if not passed:
            return gate_name, reason
    return "qualified", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--journal", type=str, default=PAPER_JOURNAL)
    args = ap.parse_args()

    if not os.path.exists(args.journal):
        print(f"ERROR: journal not found: {args.journal}", file=sys.stderr)
        sys.exit(1)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    # Per-gate accumulators
    # gate -> {won: n, lost: n, won_pnl_sum: $, lost_pnl_sum: $}
    gate_stats = defaultdict(lambda: {"won": 0, "lost": 0, "won_pnl": 0.0, "lost_pnl": 0.0})
    qualified_stats = {"won": 0, "lost": 0, "won_pnl": 0.0, "lost_pnl": 0.0}
    total_signals = 0
    parse_errors = 0

    with open(args.journal) as f:
        for line in f:
            try:
                d = json.loads(line)
                ct = d.get("closed_at") or d.get("exit_time")
                if not ct:
                    continue
                t = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if t < cutoff:
                    continue
                pnl = float(d.get("pnl_usd", d.get("pnl", 0)) or 0)
                won = pnl > 0
                # Build a signal-like dict matching what qualify_signal sees
                sig = {
                    "symbol": d.get("symbol", ""),
                    "side": d.get("side", ""),
                    "confidence": d.get("confidence", 0),
                    "grade": d.get("grade", "C"),
                    "entry_price": d.get("entry_price", 0),
                    "stop_loss": d.get("stop_loss", 0),
                    "metadata": d.get("metadata", {}) or {},
                }
                # Flatten ml_probability if it was stored top-level
                if "ml_probability" in d and "ml_probability" not in sig["metadata"]:
                    sig["metadata"]["ml_probability"] = d["ml_probability"]

                gate, _reason = replay_signal(sig)
                total_signals += 1
                if gate == "qualified":
                    if won:
                        qualified_stats["won"] += 1
                        qualified_stats["won_pnl"] += pnl
                    else:
                        qualified_stats["lost"] += 1
                        qualified_stats["lost_pnl"] += pnl
                else:
                    if won:
                        gate_stats[gate]["won"] += 1
                        gate_stats[gate]["won_pnl"] += pnl
                    else:
                        gate_stats[gate]["lost"] += 1
                        gate_stats[gate]["lost_pnl"] += pnl
            except Exception:
                parse_errors += 1
                continue

    # ── Report ────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"T2.2 PAPER → DEMO QUALIFICATION REPLAY (last {args.days} days)")
    print("=" * 80)
    print(f"  Journal:        {args.journal}")
    print(f"  Total signals:  {total_signals}")
    print(f"  Parse errors:   {parse_errors}")
    print()

    print("── QUALIFIED (would have been executed) ──")
    q = qualified_stats
    total_q = q["won"] + q["lost"]
    wr = 100 * q["won"] / max(total_q, 1)
    print(f"  n={total_q}  wins={q['won']}  losses={q['lost']}  wr={wr:.1f}%")
    print(f"  paper_net=${q['won_pnl'] + q['lost_pnl']:+.2f}  "
          f"(wins ${q['won_pnl']:+.2f} / losses ${q['lost_pnl']:+.2f})")
    print()

    print("── REJECTED (per-gate breakdown) ──")
    print(f"  {'GATE':<20} {'winners':<10} {'losers':<10} {'edge_left':<15} {'net_impact':<12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15} {'-'*12}")

    # Sort by worst leak first (most winner-$ left on table)
    rows = []
    for gate, s in gate_stats.items():
        # "edge_left" = what we SHOULD have captured from winners rejected by this gate
        # "net_impact" = if we removed this gate entirely, net change in $
        edge_left = s["won_pnl"]
        net_impact = s["won_pnl"] + s["lost_pnl"]  # would gain winners but also absorb losers
        rows.append((edge_left, gate, s, net_impact))
    rows.sort(reverse=True)

    total_winners_blocked = 0
    total_winner_dollars_blocked = 0.0
    total_losers_blocked = 0
    total_loser_dollars_blocked = 0.0
    for edge_left, gate, s, net_impact in rows:
        print(f"  {gate:<20} {s['won']:<10} {s['lost']:<10} "
              f"${s['won_pnl']:>+10.2f}    ${net_impact:>+9.2f}")
        total_winners_blocked += s["won"]
        total_winner_dollars_blocked += s["won_pnl"]
        total_losers_blocked += s["lost"]
        total_loser_dollars_blocked += s["lost_pnl"]

    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15} {'-'*12}")
    print(f"  {'TOTAL BLOCKED':<20} {total_winners_blocked:<10} {total_losers_blocked:<10} "
          f"${total_winner_dollars_blocked:>+10.2f}    "
          f"${total_winner_dollars_blocked + total_loser_dollars_blocked:>+9.2f}")
    print()

    print("── Gates NOT replay-checkable (time/state-dependent) ──")
    for g in NOT_CHECKED:
        print(f"    {g}")
    print("  (These require real-time DB state — duplicate check, open_count,")
    print("   cohort_veto, live_halt. Real demo behavior also excludes these.)")
    print()

    # ── Diagnostic verdict ─────────────────────────────────────────
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    if not rows:
        print("  No rejected signals in window. Demo is already accepting everything paper sees.")
    else:
        worst_gate = rows[0][1]
        worst_winners = rows[0][2]["won"]
        worst_winner_pnl = rows[0][2]["won_pnl"]
        total_winners = qualified_stats["won"] + total_winners_blocked
        pct_blocked = 100 * worst_winners / max(total_winners, 1)

        print(f"  Worst leak: '{worst_gate}' gate rejected {worst_winners} winners "
              f"(${worst_winner_pnl:+.2f})")
        print(f"  That's {pct_blocked:.1f}% of all paper winners in the window.")
        print()

        if worst_winner_pnl > 50 and pct_blocked > 30:
            print(f"  → ADMISSION is likely the bigger leak. Consider loosening '{worst_gate}'.")
            print(f"    But validate losers_absorbed impact: removing gate adds "
                  f"${rows[0][2]['lost_pnl']:+.2f} of losses.")
            print(f"    Net impact if removed: ${rows[0][3]:+.2f}")
        elif worst_winner_pnl < 20:
            print("  → Gates are NOT the bottleneck. The leak is in EXECUTION.")
            print("    Focus on: maker fill rate, MFE capture, exit timing, multiplexing.")
        else:
            print("  → Mixed signal. No single gate dominates.")
            print("    Likely a combination of admission + execution gaps.")
    print()


if __name__ == "__main__":
    main()

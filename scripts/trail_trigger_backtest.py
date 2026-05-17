#!/usr/bin/env python3
"""Trail-trigger backtest — compare 0.3R / 0.4R / 0.5R variants.

Hypothesis: shadow's primary config requires peak_mfe_r >= 0.5R to engage
trail, but paper's signal_tracker engages at ~0.3R. Today's data showed
paper hitting trail_profit 92% of the time vs shadow's 0% in the same
window. This backtest validates whether lowering shadow's trail trigger
would close the gap WITHOUT breaking other exits.

Tests 5 variants over the last 7 days of historical paper signals
(uses the existing phase3 engine + ConfigurableExitPolicy):

  baseline_05R         current production primary (trail=0.5R, lock=80%)
  proposed_03R_lock80  the proposed fix (trail=0.3R, lock=80%)
  proposed_03R_lock60  more aggressive lock (0.3R + 60% lock)
  alt_04R_lock80       middle ground (0.4R + 80% lock)
  no_dead_kill_03R     0.3R trail, no dead_signal kill, no stall

Outputs storage/backtest_trail/trail_trigger_TS.md with leaderboard.
"""
import sys
import datetime as datetime_mod
import pathlib
import logging
from datetime import timezone, timedelta

ROOT = pathlib.Path("/home/opc/crypto-trading-bot")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "storage" / "backtest_trail"
TS = datetime_mod.datetime.utcnow()
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHED_SYMBOLS = {
    "AVAX/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT",
    "LINK/USDT", "SOL/USDT", "XRP/USDT",
}

# 5 variants to test — each has identical max_age + dead_kill behavior
# except where the variant name indicates a difference.
VARIANTS = [
    {"id": "baseline_05R",         "max_age_sec": 600, "trail_trigger": 0.5, "trail_lock": 0.80,
     "dead_kill_R": -0.10, "stall_kill_R": -0.05, "tp_R": None,
     "_note": "current production primary"},
    {"id": "proposed_03R_lock80",  "max_age_sec": 600, "trail_trigger": 0.3, "trail_lock": 0.80,
     "dead_kill_R": -0.10, "stall_kill_R": -0.05, "tp_R": None,
     "_note": "lowered trail to match paper's signal_tracker"},
    {"id": "proposed_03R_lock60",  "max_age_sec": 600, "trail_trigger": 0.3, "trail_lock": 0.60,
     "dead_kill_R": -0.10, "stall_kill_R": -0.05, "tp_R": None,
     "_note": "0.3R + tighter 60% lock"},
    {"id": "alt_04R_lock80",       "max_age_sec": 600, "trail_trigger": 0.4, "trail_lock": 0.80,
     "dead_kill_R": -0.10, "stall_kill_R": -0.05, "tp_R": None,
     "_note": "middle ground 0.4R"},
    {"id": "no_dead_kill_03R",     "max_age_sec": 600, "trail_trigger": 0.3, "trail_lock": 0.80,
     "dead_kill_R":  None, "stall_kill_R":  None, "tp_R": None,
     "_note": "0.3R + remove dead_kill + stall guards"},
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7,
                    help="Lookback days of paper signals (default 7)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Reuse the phase3 sweep machinery
    from phase3_historical_sweep import make_configurable_exit_policy
    from backtest.execution_replay.engine import load_signals, run_backtest
    from backtest.execution_replay.fill_model import TakerFillModel
    from backtest.execution_replay.metrics import compute_metrics

    signals_path = ROOT / "storage" / "closed_signals.json"
    since = datetime_mod.datetime.now(timezone.utc) - timedelta(days=args.days)
    print(f"Loading paper signals from last {args.days}d (since {since.isoformat()})...")
    signals = load_signals(signals_path, since=since)
    raw_n = len(signals)

    def _sym(s):
        if isinstance(s, dict):
            return s.get("symbol", "")
        return getattr(s, "symbol", "")

    signals = [s for s in signals if _sym(s) in CACHED_SYMBOLS]
    print(f"  Loaded {raw_n} signals → {len(signals)} after cache-symbol filter ({sorted(CACHED_SYMBOLS)})")
    if not signals:
        print("No signals, abort.")
        return 1

    fill_model = TakerFillModel()
    results = []

    for cfg in VARIANTS:
        print(f"\n=== {cfg['id']} — {cfg.get('_note','')} ===")
        try:
            ep = make_configurable_exit_policy(cfg)
            trades = run_backtest(
                signals=signals,
                fill_model=fill_model,
                default_funding_rate_8h=0.0,
                exit_policy=ep,
            )
            metrics = compute_metrics(trades)

            try:
                pnl_col = trades["sim_net_usd"] if "sim_net_usd" in trades.columns else trades.get("pnl_usd")
                wins = int((pnl_col > 0).sum()) if pnl_col is not None else 0
            except Exception:
                wins = 0
            try:
                ep_col = trades["exit_policy_name"] if "exit_policy_name" in trades.columns else None
                replayed = int((ep_col != "historical").sum()) if ep_col is not None else 0
            except Exception:
                replayed = 0

            n = getattr(metrics, "n_trades", 0) or len(trades)
            wr_pct = float(getattr(metrics, "win_rate", 0) or 0) * 100.0
            if wr_pct > 100.0:
                wr_pct = float(getattr(metrics, "win_rate", 0) or 0)

            # Per-exit-reason breakdown — most useful: how often does trail_profit fire?
            try:
                if "replayed_exit_reason" in trades.columns:
                    reason_counts = trades["replayed_exit_reason"].value_counts().to_dict()
                else:
                    reason_counts = {}
            except Exception:
                reason_counts = {}

            results.append({
                "cfg_id": cfg["id"],
                "note": cfg.get("_note", ""),
                "n": n,
                "wins": wins,
                "wr": wr_pct,
                "net": getattr(metrics, "total_pnl_usd", 0),
                "avg": getattr(metrics, "avg_pnl_per_trade", 0),
                "pf": getattr(metrics, "profit_factor", 0),
                "max_dd": getattr(metrics, "max_drawdown_usd", 0),
                "replayed": replayed,
                "trail_count": int(reason_counts.get("trail_profit", 0)),
                "td_count": sum(int(reason_counts.get(k, 0))
                                for k in reason_counts if k.startswith("time_decay")),
                "dead_count": int(reason_counts.get("dead_signal_unified", 0)),
                "sl_count": int(reason_counts.get("sl_hit", 0)),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR for {cfg['id']}: {e}")
            results.append({
                "cfg_id": cfg["id"], "note": cfg.get("_note", ""),
                "n": 0, "wins": 0, "wr": 0, "net": 0, "avg": 0, "pf": 0, "max_dd": 0,
                "replayed": 0, "trail_count": 0, "td_count": 0, "dead_count": 0, "sl_count": 0,
                "error": str(e)[:120],
            })

    # Sort by net PnL desc
    results.sort(key=lambda r: -r["net"])

    lines = [
        f"# Trail-trigger Backtest — last {args.days}d",
        f"Generated: {TS.isoformat()}Z",
        f"",
        f"**Hypothesis**: shadow's primary uses trail_trigger=0.5R but paper's signal_tracker effectively engages trail at ~0.3R (per 04-27 paper-vs-shadow audit showing paper trail_profit 92% vs shadow 0% in late-night window).",
        f"",
        f"**Test**: 5 variants × {len(signals)} historical paper signals (last {args.days}d, cache symbols {sorted(CACHED_SYMBOLS)}).",
        f"",
        "## Leaderboard (sorted by Net PnL desc)",
        "",
        "| Config | Note | n | wins | WR | Net | Avg | PF | MaxDD | trail_n | time_decay_n | dead_kill_n | sl_n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        err = r.get("error", "")
        if err:
            lines.append(f"| `{r['cfg_id']}` | {r['note']} | ERR | | | | | | | | | | {err} |")
            continue
        lines.append(
            f"| `{r['cfg_id']}` | {r['note']} | {r['n']} | {r['wins']} | "
            f"{r['wr']:.1f}% | ${r['net']:.2f} | ${r['avg']:.3f} | "
            f"{r['pf']:.2f} | ${r['max_dd']:.2f} | "
            f"{r['trail_count']} | {r['td_count']} | {r['dead_count']} | {r['sl_count']} |"
        )

    if results:
        baseline = next((r for r in results if r["cfg_id"] == "baseline_05R"), None)
        winner = results[0]
        lines.extend([
            "",
            f"## Summary",
            "",
        ])
        if baseline and winner["cfg_id"] != "baseline_05R":
            uplift_net = winner["net"] - baseline["net"]
            uplift_pct = (winner["net"] / baseline["net"] - 1) * 100 if baseline["net"] != 0 else None
            lines.append(f"- **Winner**: `{winner['cfg_id']}` — net ${winner['net']:.2f} vs baseline ${baseline['net']:.2f} = ${uplift_net:+.2f} uplift ({uplift_pct:+.1f}% if not None else 'n/a')")
            lines.append(f"- Trail-profit fires: winner {winner['trail_count']} vs baseline {baseline['trail_count']}")
            lines.append(f"- Dead-kill fires: winner {winner['dead_count']} vs baseline {baseline['dead_count']}")
        elif baseline:
            lines.append(f"- Baseline IS the winner — proposed change does NOT improve on history.")
        lines.append("")
        lines.append("## Decision rule")
        lines.append("")
        lines.append("- If a proposed variant beats baseline by ≥10% Net AND has ≥20% more trail_profit fires AND PF stays ≥ baseline → SHIP that variant.")
        lines.append("- Else → keep baseline; the gap I observed today was variance, not signal.")

    out_file = OUT_DIR / f"trail_trigger_{TS.strftime('%Y%m%d_%H%M%S')}.md"
    out_file.write_text("\n".join(lines))
    print(f"\nWrote: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

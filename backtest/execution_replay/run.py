"""
Run execution-replay backtest.

Default behavior: load last 30 days of paper signals, run under 3 fill models
(paper / maker / taker), compare side-by-side.

Usage:
    python3 -m backtest.execution_replay.run --days 30
    python3 -m backtest.execution_replay.run --days 7 --json out.json
    python3 -m backtest.execution_replay.run --model taker --days 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from backtest.execution_replay.engine import load_signals, run_backtest
from backtest.execution_replay.fill_model import (
    TakerFillModel, MakerFillModel, PaperFillModel,
)
from backtest.execution_replay.metrics import (
    compute_metrics, format_report, format_comparison, format_admit_report,
)
from backtest.execution_replay.cohort_analysis import run_all_breakdowns
from backtest.execution_replay.admit_policy import (
    AdmitPolicy, AdmitAll, CohortFilterPolicy,
)
from backtest.execution_replay.exit_policy import (
    ExitPolicy, get_policy as get_exit_policy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_backtest")


DEFAULT_SIGNALS_PATH = Path(__file__).parent.parent.parent / "storage" / "closed_signals.json"


def run(
    signals_path: Path,
    days: int,
    default_funding_rate_8h: float,
    models_to_run: list,
    json_out: Path | None = None,
    show_cohorts: bool = True,
    admit_policy: AdmitPolicy | None = None,
    admit_policy_name: str = "all",
    exit_policy: ExitPolicy | None = None,
    exit_policy_name: str = "historical",
) -> None:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    logger.info("Loading paper signals from %s (since %s)...",
                signals_path, since.date())

    signals = load_signals(signals_path, since=since)
    logger.info("Loaded %d signals in window", len(signals))

    if not signals:
        logger.warning("No signals in window — nothing to backtest")
        return

    if admit_policy is not None and admit_policy_name != "all":
        logger.info("Applying admit policy: %s", admit_policy_name)
    if exit_policy is not None and exit_policy_name != "historical":
        logger.info("Applying exit policy: %s (candle-walk replay)", exit_policy_name)

    results = []
    labels = []
    for label, model in models_to_run:
        logger.info("Running backtest with %s fill model...", label)
        df = run_backtest(signals, fill_model=model,
                          default_funding_rate_8h=default_funding_rate_8h,
                          admit_policy=admit_policy,
                          exit_policy=exit_policy)
        if df.empty:
            logger.warning("No trades produced by %s model", label)
            continue
        # Admit-policy report (only when policy is non-default).
        if admit_policy_name != "all" and "admit_decision" in df.columns:
            print(format_admit_report(label, admit_policy_name, df))
        # For risk metrics, keep the existing semantics: only admitted
        # rows contribute to P&L / WR / Sharpe. Rejected rows have
        # net_pnl_usd == 0 and would silently dilute averages otherwise.
        admitted_df = df[df.get("admit_decision", "admitted") == "admitted"] \
            if "admit_decision" in df.columns else df
        if admitted_df.empty:
            logger.warning("No admitted trades under %s model + %s policy",
                           label, admit_policy_name)
            continue
        m = compute_metrics(admitted_df)
        results.append((label, admitted_df, m))
        labels.append(label)
        print(format_report(label, m))

    # Comparison table
    if len(results) >= 2:
        print(format_comparison(
            labels=[r[0] for r in results],
            results=[r[2] for r in results],
        ))

    # Cohort breakdowns — focus on realistic models (maker/taker), skip paper.
    if show_cohorts:
        for label, df, _m in results:
            if label == "paper":
                continue
            print(run_all_breakdowns(df, label))

    # Dump first result's trade-level detail as JSON if requested
    if json_out and results:
        payload = {
            "metadata": {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "signals_loaded": len(signals),
                "window_days": days,
                "funding_rate_8h": default_funding_rate_8h,
            },
            "models": [
                {
                    "label": label,
                    "metrics": m.to_dict(),
                    "trades": df.to_dict(orient="records"),
                }
                for label, df, m in results
            ],
        }
        with open(json_out, "w") as f:
            json.dump(payload, f, default=str, indent=2)
        logger.info("Wrote JSON report → %s", json_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS_PATH,
                    help="Path to closed_signals.json")
    ap.add_argument("--days", type=int, default=30,
                    help="Lookback window in days")
    ap.add_argument("--funding-rate", type=float, default=0.0001,
                    help="Default 8h funding rate (0.01% = 0.0001)")
    ap.add_argument("--model", choices=["paper", "maker", "taker", "all"],
                    default="all",
                    help="Which fill model(s) to run")
    ap.add_argument("--taker-latency-bps", type=float, default=2.0,
                    help="Extra bps beyond half-spread for taker slippage")
    ap.add_argument("--maker-miss", type=float, default=0.40,
                    help="Maker entry-miss probability")
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="Write detailed per-trade results to JSON")
    ap.add_argument("--no-cohorts", action="store_true",
                    help="Skip cohort breakdowns (faster output)")
    ap.add_argument(
        "--admit-policy", choices=["all", "cohort_filter"], default="all",
        help="Signal admission policy. 'all' = admit everything (default). "
             "'cohort_filter' = apply Wave 6.C Lever 2 rules "
             "(reject atr_pct<=0.000225 OR vwap_zone=='penalty').",
    )
    ap.add_argument(
        "--exit-policy", choices=["historical", "legacy", "phase58", "lever3"],
        default="historical",
        help="Exit policy. 'historical' = read exit_reason from signal record (default, fastest). "
             "'legacy' = re-simulate via LegacyExitPolicy (cascade as of 2026-04-23). "
             "'phase58' = Wave 2 unified guard. 'lever3' = trail-lock + raised time gates.",
    )
    args = ap.parse_args()

    models = []
    if args.model in ("paper", "all"):
        models.append(("paper", PaperFillModel()))
    if args.model in ("maker", "all"):
        models.append((
            "maker",
            MakerFillModel(
                entry_miss_prob=args.maker_miss,
                exit_miss_prob=args.maker_miss * 0.75,  # exits miss less (indicator-fire)
            )
        ))
    if args.model in ("taker", "all"):
        models.append((
            "taker",
            TakerFillModel(extra_latency_bps=args.taker_latency_bps)
        ))

    if args.admit_policy == "cohort_filter":
        admit_policy: AdmitPolicy = CohortFilterPolicy()
    else:
        admit_policy = AdmitAll()

    exit_policy = get_exit_policy(args.exit_policy)

    run(
        signals_path=args.signals,
        days=args.days,
        default_funding_rate_8h=args.funding_rate,
        models_to_run=models,
        json_out=args.json_out,
        show_cohorts=not args.no_cohorts,
        admit_policy=admit_policy,
        admit_policy_name=args.admit_policy,
        exit_policy=exit_policy,
        exit_policy_name=args.exit_policy,
    )


if __name__ == "__main__":
    main()

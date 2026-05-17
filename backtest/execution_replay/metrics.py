"""
Risk-adjusted metrics for backtest output.

Industry-standard calculations:
  - Sharpe ratio (annualized): (mean_daily_return - rf) / std_daily_return × sqrt(365)
  - Sortino ratio: same but only downside std in denominator
  - Max drawdown: largest peak-to-trough equity decline
  - Calmar ratio: CAGR / |max_drawdown|
  - Profit factor: sum(wins) / |sum(losses)|

All metrics are computed from per-trade P&L aggregated by day.
Risk-free rate defaults to 0 (crypto typical convention).

Bootstrap confidence intervals provided on key metrics for honesty.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 365  # crypto = 24/7


@dataclass
class RiskMetrics:
    n_trades: int
    n_days: int
    total_pnl_usd: float
    avg_pnl_per_trade: float
    avg_pnl_per_day: float
    win_rate: float
    winners_avg: float
    losers_avg: float
    profit_factor: float
    sharpe_annualized: float
    sortino_annualized: float
    max_drawdown_usd: float
    max_drawdown_pct_of_peak: float
    calmar_annualized: float
    best_day: float
    worst_day: float
    best_trade: float
    worst_trade: float
    # Bootstrap 95% CI on Sharpe (honest uncertainty)
    sharpe_ci_low: float
    sharpe_ci_high: float
    # Phase 5.20-D (2026-04-25) — Wilson interval on win rate (binomial proportion)
    # AND bootstrap CI on profit factor. Audit: previously only Sharpe had CI;
    # WR and PF were point estimates. On small samples this implied false precision.
    win_rate_ci_low: float = 0.0
    win_rate_ci_high: float = 0.0
    profit_factor_ci_low: float = 0.0
    profit_factor_ci_high: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _aggregate_to_daily(df: pd.DataFrame) -> pd.Series:
    """Given trade DataFrame with 'exit_time' and 'net_pnl_usd', sum by day."""
    if len(df) == 0:
        return pd.Series(dtype=float)
    df = df.copy()
    df["day"] = pd.to_datetime(df["exit_time"], utc=True).dt.floor("D")
    return df.groupby("day")["net_pnl_usd"].sum().sort_index()


def _wilson_ci(wins: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval on a binomial proportion (win rate).
    More accurate than normal approximation on small samples.
    Returns (lo_pct, hi_pct).

    Phase 5.20-D — addresses audit gap: WR was reported without CI.
    """
    if n == 0:
        return 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (
        max(0.0, (center - halfwidth) * 100),
        min(100.0, (center + halfwidth) * 100),
    )


def _bootstrap_pf_ci(
    pnls: np.ndarray, n_boot: int = 1000, alpha: float = 0.05
) -> Tuple[float, float]:
    """Bootstrap 95% CI on profit factor."""
    if len(pnls) < 5:
        return 0.0, float("inf")
    rng = np.random.default_rng(seed=43)
    n = len(pnls)
    pfs = []
    for _ in range(n_boot):
        sample = rng.choice(pnls, size=n, replace=True)
        wins = sample[sample > 0]
        losses = sample[sample < 0]
        if losses.sum() < 0:
            pfs.append(wins.sum() / abs(losses.sum()))
    if not pfs:
        return 0.0, float("inf")
    pfs_arr = np.array(pfs)
    return (
        float(np.quantile(pfs_arr, alpha / 2)),
        float(np.quantile(pfs_arr, 1 - alpha / 2)),
    )


def _bootstrap_sharpe_ci(
    daily_returns: np.ndarray, n_boot: int = 1000, alpha: float = 0.05
) -> Tuple[float, float]:
    """95% bootstrap CI on annualized Sharpe."""
    if len(daily_returns) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=42)  # reproducible
    n = len(daily_returns)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(daily_returns, size=n, replace=True)
        if sample.std() > 1e-9:
            s = (sample.mean() / sample.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
            sharpes.append(s)
    if not sharpes:
        return (float("nan"), float("nan"))
    sharpes_arr = np.array(sharpes)
    return (
        float(np.quantile(sharpes_arr, alpha / 2)),
        float(np.quantile(sharpes_arr, 1 - alpha / 2)),
    )


def compute_metrics(trades_df: pd.DataFrame) -> RiskMetrics:
    """
    Compute full risk-adjusted metrics from a backtest DataFrame.

    Required columns: exit_time, net_pnl_usd
    """
    if len(trades_df) == 0:
        return RiskMetrics(
            n_trades=0, n_days=0,
            total_pnl_usd=0, avg_pnl_per_trade=0, avg_pnl_per_day=0,
            win_rate=0, winners_avg=0, losers_avg=0, profit_factor=0,
            sharpe_annualized=0, sortino_annualized=0,
            max_drawdown_usd=0, max_drawdown_pct_of_peak=0,
            calmar_annualized=0,
            best_day=0, worst_day=0, best_trade=0, worst_trade=0,
            sharpe_ci_low=0, sharpe_ci_high=0,
        )

    pnls = trades_df["net_pnl_usd"].values.astype(float)
    n_trades = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total = float(np.sum(pnls))

    # Daily aggregation
    daily = _aggregate_to_daily(trades_df)
    n_days = len(daily)
    daily_returns = daily.values.astype(float) if n_days > 0 else np.array([])

    # Sharpe (annualized)
    if len(daily_returns) >= 2 and daily_returns.std() > 1e-9:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0

    # Sortino (downside-only std)
    if len(daily_returns) >= 2:
        downside = daily_returns[daily_returns < 0]
        if len(downside) >= 1 and downside.std() > 1e-9:
            sortino = (daily_returns.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sortino = float("inf") if daily_returns.mean() > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown on cumulative equity.
    # For losing strategies the equity curve never exceeds the starting $0,
    # so "peak_pct" is undefined (division by ~0). Two-metric approach:
    #   max_dd_usd: always meaningful (largest peak-to-trough decline in $)
    #   max_dd_pct_of_peak: only meaningful if peak > 0 (else NaN)
    if n_days >= 1:
        equity = np.cumsum(daily_returns)
        equity_with_start = np.concatenate([[0], equity])
        running_max = np.maximum.accumulate(equity_with_start)
        drawdowns = equity_with_start - running_max
        max_dd = float(drawdowns.min())  # negative
        peak = float(running_max.max())
        if peak > 1.0:  # meaningful peak
            max_dd_pct = _safe_div(abs(max_dd), peak, 0) * 100
        else:
            # Losing strategy never achieved a real peak — pct meaningless.
            # Report cumulative loss as % of initial equity ($1 reference).
            max_dd_pct = float("nan")
    else:
        max_dd = 0.0
        max_dd_pct = 0.0

    # Calmar = annualized return / |max DD|
    if n_days >= 1 and max_dd != 0:
        annualized_return = (total / n_days) * TRADING_DAYS_PER_YEAR
        calmar = annualized_return / abs(max_dd)
    else:
        calmar = 0.0

    # Bootstrap Sharpe CI
    sharpe_lo, sharpe_hi = _bootstrap_sharpe_ci(daily_returns)
    # Phase 5.20-D — Wilson CI on WR + bootstrap CI on PF
    wr_lo, wr_hi = _wilson_ci(int(len(wins)), int(n_trades))
    pf_lo, pf_hi = _bootstrap_pf_ci(pnls)

    return RiskMetrics(
        n_trades=n_trades,
        n_days=n_days,
        total_pnl_usd=total,
        avg_pnl_per_trade=float(np.mean(pnls)),
        avg_pnl_per_day=float(total / n_days) if n_days else 0,
        win_rate=float(len(wins) / n_trades * 100),
        winners_avg=float(np.mean(wins)) if len(wins) else 0,
        losers_avg=float(np.mean(losses)) if len(losses) else 0,
        profit_factor=float(_safe_div(np.sum(wins), abs(np.sum(losses)), 0)),
        sharpe_annualized=float(sharpe),
        sortino_annualized=float(sortino),
        max_drawdown_usd=max_dd,
        max_drawdown_pct_of_peak=max_dd_pct,
        calmar_annualized=float(calmar),
        best_day=float(daily.max()) if n_days else 0,
        worst_day=float(daily.min()) if n_days else 0,
        best_trade=float(pnls.max()),
        worst_trade=float(pnls.min()),
        sharpe_ci_low=sharpe_lo,
        sharpe_ci_high=sharpe_hi,
        win_rate_ci_low=wr_lo,
        win_rate_ci_high=wr_hi,
        profit_factor_ci_low=pf_lo,
        profit_factor_ci_high=pf_hi,
    )


def format_report(label: str, m: RiskMetrics) -> str:
    """Human-readable metrics report."""
    lines = [
        "",
        "=" * 70,
        f"  BACKTEST RESULT: {label}",
        "=" * 70,
        f"  Trades:                {m.n_trades}",
        f"  Days:                  {m.n_days}",
        f"  Total NET P&L:         ${m.total_pnl_usd:+,.2f}",
        f"  Avg/trade:             ${m.avg_pnl_per_trade:+.3f}",
        f"  Avg/day:               ${m.avg_pnl_per_day:+.2f}",
        "",
        f"  Win rate:              {m.win_rate:.1f}%",
        f"  Avg winner:            ${m.winners_avg:+.3f}",
        f"  Avg loser:             ${m.losers_avg:+.3f}",
        f"  Profit factor:         {m.profit_factor:.2f}",
        "",
        f"  Sharpe (annualized):   {m.sharpe_annualized:.2f}",
        f"     95% CI:            [{m.sharpe_ci_low:.2f}, {m.sharpe_ci_high:.2f}]",
        f"  Sortino (annualized):  {m.sortino_annualized:.2f}",
        f"  Calmar (annualized):   {m.calmar_annualized:.2f}",
        "",
        f"  Max drawdown:          ${m.max_drawdown_usd:+.2f}  ({'n/a (no peak)' if (m.max_drawdown_pct_of_peak != m.max_drawdown_pct_of_peak) else f'{m.max_drawdown_pct_of_peak:.1f}% of peak'})",
        f"  Best day:              ${m.best_day:+.2f}",
        f"  Worst day:             ${m.worst_day:+.2f}",
        f"  Best trade:            ${m.best_trade:+.2f}",
        f"  Worst trade:           ${m.worst_trade:+.2f}",
        "=" * 70,
    ]
    return "\n".join(lines)


def format_comparison(labels: List[str], results: List[RiskMetrics]) -> str:
    """Side-by-side comparison table."""
    if not results:
        return "(no results)"
    headers = ["Metric"] + labels
    rows = [
        ["n_trades"]       + [str(r.n_trades) for r in results],
        ["days"]           + [str(r.n_days)   for r in results],
        ["Total NET $"]    + [f"{r.total_pnl_usd:+.2f}" for r in results],
        ["Avg/trade $"]    + [f"{r.avg_pnl_per_trade:+.3f}" for r in results],
        ["Avg/day $"]      + [f"{r.avg_pnl_per_day:+.2f}" for r in results],
        ["Win rate %"]     + [f"{r.win_rate:.1f}"  for r in results],
        ["Profit factor"]  + [f"{r.profit_factor:.2f}" for r in results],
        ["Sharpe (ann)"]   + [f"{r.sharpe_annualized:.2f}" for r in results],
        ["Sharpe CI low"]  + [f"{r.sharpe_ci_low:.2f}"  for r in results],
        ["Sharpe CI high"] + [f"{r.sharpe_ci_high:.2f}" for r in results],
        ["Sortino (ann)"]  + [f"{r.sortino_annualized:.2f}" for r in results],
        ["Max DD $"]       + [f"{r.max_drawdown_usd:+.2f}"  for r in results],
        ["Max DD %peak"]   + [f"{r.max_drawdown_pct_of_peak:.1f}"  for r in results],
        ["Calmar (ann)"]   + [f"{r.calmar_annualized:.2f}" for r in results],
    ]
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    def fmt_row(cells):
        return "  " + " | ".join(str(c).rjust(widths[i]) for i, c in enumerate(cells))
    sep = "  " + "-+-".join("-" * w for w in widths)
    return "\n".join([
        "",
        "=" * (sum(widths) + 3 * len(widths)),
        "  SIDE-BY-SIDE COMPARISON",
        "=" * (sum(widths) + 3 * len(widths)),
        fmt_row(headers),
        sep,
    ] + [fmt_row(r) for r in rows] + [""])


def format_admit_report(label: str, policy_name: str, df: "pd.DataFrame") -> str:
    """
    Admit-policy summary printed when a non-default `--admit-policy` is in
    effect. Shows total signals, admit/reject split by reason, and a
    counterfactual: what was the historical paper P&L of the rejected
    signals? — i.e. "what we would have forfeited by filtering this
    cohort out, using the historical exit data as ground truth."

    Parameters
    ----------
    label : fill-model label ('paper' / 'maker' / 'taker') — informational.
    policy_name : admit-policy name ('cohort_filter', etc.) — for header.
    df : DataFrame from run_backtest, must contain the `admit_decision` and
         `admit_reason` columns (i.e. produced with admit_policy != None).

    Note: the counterfactual is built from `paper_pnl_usd` (historical P&L
    captured at signal-close time), NOT `sim_net_usd` — rejected rows have
    sim_net_usd == 0 by construction. This is the most faithful answer to
    "what did this policy cost / save us?" because the historical P&L is
    what the live bot actually realized on those signals.
    """
    if "admit_decision" not in df.columns:
        return ""

    total = len(df)
    admitted = df[df["admit_decision"] == "admitted"]
    rejected = df[df["admit_decision"] == "rejected"]
    n_admitted = len(admitted)
    n_rejected = len(rejected)

    # Reason breakdown — group rejection_reason by prefix
    # ("cohort_filter_atr:..." → "cohort_filter_atr")
    by_reason = {}
    for r in rejected.get("admit_reason", []):
        if not isinstance(r, str):
            continue
        # Collapse the floating ATR value into one bucket
        prefix = r.split(":", 1)[0] if ":" in r else r
        by_reason[prefix] = by_reason.get(prefix, 0) + 1

    # Counterfactual P&L of the rejected cohort, using historical paper P&L.
    forfeited_paper_pnl = 0.0
    forfeited_wins = 0
    forfeited_losses = 0
    if n_rejected > 0 and "paper_pnl_usd" in rejected.columns:
        paper_pnls = rejected["paper_pnl_usd"].astype(float)
        forfeited_paper_pnl = float(paper_pnls.sum())
        forfeited_wins = int((paper_pnls > 0).sum())
        forfeited_losses = int((paper_pnls < 0).sum())

    pct_admitted = (n_admitted / total * 100) if total else 0.0
    pct_rejected = (n_rejected / total * 100) if total else 0.0

    lines = [
        "",
        "=" * 70,
        f"  ADMIT POLICY REPORT [{label} / {policy_name}]",
        "=" * 70,
        f"  Total signals:         {total}",
        f"  Admitted:              {n_admitted}  ({pct_admitted:.1f}%)",
        f"  Rejected:              {n_rejected}  ({pct_rejected:.1f}%)",
        "",
        "  Rejection reasons:",
    ]
    if by_reason:
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason}:  {count}")
    else:
        lines.append("    (none)")
    lines += [
        "",
        "  COUNTERFACTUAL — historical paper P&L of REJECTED signals",
        "  (positive = filter saved money; negative = filter cost money)",
        f"    Forfeited paper P&L:  ${-forfeited_paper_pnl:+.2f}  "
        "(sign-flipped: '+' means filter is net-positive)",
        f"    Rejected wins:        {forfeited_wins}",
        f"    Rejected losses:      {forfeited_losses}",
        "=" * 70,
    ]
    return "\n".join(lines)

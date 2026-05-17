"""Walk-Forward Harness Template — Phase 5.21 Research Lab.

Reusable W/F engine for any tuning study. Each TUNE concept implements
a Strategy class with:
  - param_grid()       — iterable of param dicts for the sweep
  - simulate(df, p)    — given candle df + param dict, return list of Trade

The harness handles:
  - Cache loading (storage/candle_cache/<SYM>_USDT_<TF>.parquet)
  - Quarter split (Q1+Q2 IS / Q3 OOS / Q4 OOS)
  - Per-cell × per-fee-variant aggregation (variants A/B/C)
  - PASS/HOLD/KILL verdict per the discipline:
      IS EV positive AND OOS gap ≤50% AND Q4 EV per-trade > $0.10
  - JSON + markdown report output

Usage:
    from wf_harness import WalkForwardEngine, Strategy, Trade

    class VolumeGateStudy(Strategy):
        def param_grid(self):
            for thr in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
                yield {"vol_threshold": thr}

        def simulate(self, df, params):
            # iterate df, emit Trade for each entry that passes vol_threshold
            # ... your strategy logic ...
            return trades

    engine = WalkForwardEngine(
        study=VolumeGateStudy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=Path("storage/wf_studies/volume_gate"),
    )
    engine.run()  # writes results JSON + report.md

The engine is read-only against the cache. Output goes to its own out_dir.
NEVER imports from execution_v2.user_real_manager or strategies.scalp_strategy
to avoid contaminating the live bot.
"""
from __future__ import annotations

import json
import math
import sys
import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
ROOT = Path("/home/opc/crypto-trading-bot")
CACHE_DIR = ROOT / "storage" / "candle_cache"

# Walk-forward windows (UTC). Standard schedule across all studies.
QUARTERS: Dict[str, Tuple[str, str]] = {
    "Q1": ("2025-10-01", "2025-12-01"),  # IS (partial — cache starts 2025-11-05)
    "Q2": ("2025-12-01", "2026-02-01"),  # IS
    "Q3": ("2026-02-01", "2026-03-01"),  # OOS-1
    "Q4": ("2026-03-01", "2026-05-01"),  # OOS-2
}
IS_QUARTERS = ["Q1", "Q2"]
OOS_QUARTERS = ["Q3", "Q4"]

# Fee variants — Delta India, scalper-offer eligibility per symbol.
# Maker = 0.02% × 1.18 GST.  Taker = 0.05% × 1.18 GST.  Scalper-offer waives
# exit fee for BTC/ETH ≤30min, others ≤15min.
FEE_VARIANTS = {
    "A_full_taker":      {"entry_pct": 0.00059, "exit_pct": 0.00059, "scalper": False},
    "B_scalper_taker":   {"entry_pct": 0.00059, "exit_pct": 0.00000, "scalper": True},
    "C_maker_scalper":   {"entry_pct": 0.000236, "exit_pct": 0.00000, "scalper": True},
}

# Pass criteria
PASS_OOS_GAP_MAX = 0.50      # gap = (IS_EV - OOS_EV) / |IS_EV|; must be ≤ 50%
PASS_Q4_EV_MIN = 0.10        # Q4 EV per trade must exceed $0.10
PASS_IS_EV_MIN = 0.01        # IS EV must be positive (> $0.01 per trade)


# ─────────────────────────────────────────────────────────────────────
# Trade schema
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    """Standard trade record produced by a strategy.simulate(...) call.

    All fields required. Strategies must compute entry/exit prices (not
    signals); the harness applies fee variants to gross_pnl_usd to derive
    net per variant.
    """
    symbol: str
    side: str                  # 'long' | 'short'
    entry_price: float
    exit_price: float
    notional_usd: float        # for fee math (entry_price × position_size × contract)
    holding_sec: int
    entry_ts: pd.Timestamp     # for quarter assignment
    exit_ts: pd.Timestamp
    exit_reason: str = ""      # 'tp' | 'sl' | 'time_stop' | etc.
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def gross_pnl_usd(self) -> float:
        """Gross P&L (no fees) per trade."""
        if self.side == "long":
            return (self.exit_price - self.entry_price) / self.entry_price * self.notional_usd
        return (self.entry_price - self.exit_price) / self.entry_price * self.notional_usd


# ─────────────────────────────────────────────────────────────────────
# Strategy interface
# ─────────────────────────────────────────────────────────────────────
class Strategy(ABC):
    """Plug-in interface for any TUNE study.

    Subclass + implement param_grid() and simulate(). The harness handles
    everything else.
    """
    name: str = "unnamed_study"

    @abstractmethod
    def param_grid(self) -> Iterator[Dict[str, Any]]:
        """Yield dict per cell to test."""
        ...

    @abstractmethod
    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        """Run the strategy over the candle df with these params.

        df is the FULL candle history (not pre-split by quarter — strategy
        gets the full series so it can compute indicators that need history).
        Strategy must populate Trade.entry_ts so the harness can split into
        quarters AFTER trade generation.
        """
        ...

    def cell_id(self, params: Dict[str, Any]) -> str:
        """Stable string id for a param cell (used as JSON key)."""
        return "_".join(f"{k}={v}" for k, v in sorted(params.items()))


# ─────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────────────
@dataclass
class CellResult:
    """Aggregate stats for one (symbol × tf × params × variant) cell."""
    n_total: int = 0
    is_ev: float = 0.0
    is_n: int = 0
    q3_ev: float = 0.0
    q3_n: int = 0
    q4_ev: float = 0.0
    q4_n: int = 0
    oos_ev: float = 0.0
    oos_n: int = 0
    is_gross_sum: float = 0.0
    is_net_sum: float = 0.0
    oos_gross_sum: float = 0.0
    oos_net_sum: float = 0.0
    win_rate_is: float = 0.0
    win_rate_oos: float = 0.0
    verdict: str = "INSUFFICIENT_DATA"
    gap_pct: Optional[float] = None  # (IS_EV - OOS_EV) / |IS_EV|


class WalkForwardEngine:
    """Run a Strategy across param_grid × symbols × timeframes × fee variants."""

    def __init__(
        self,
        study: Strategy,
        symbols: List[str],
        timeframes: List[str],
        out_dir: Path,
        fee_variants: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> None:
        self.study = study
        self.symbols = symbols
        self.timeframes = timeframes
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fee_variants = fee_variants or list(FEE_VARIANTS.keys())
        self.verbose = verbose
        self._cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    # ──── Cache I/O ──────────────────────────────────────────────────
    def load_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        key = (symbol, tf)
        if key in self._cache:
            return self._cache[key]
        path = CACHE_DIR / f"{symbol}_USDT_{tf}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No cache: {path}")
        df = pd.read_parquet(path)
        # Standardize: ensure datetime index (UTC)
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()
        self._cache[key] = df
        return df

    # ──── Quarter assignment ─────────────────────────────────────────
    @staticmethod
    def quarter_of(ts: pd.Timestamp) -> Optional[str]:
        """Return Q1/Q2/Q3/Q4 or None if outside all windows."""
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        for q, (start, end) in QUARTERS.items():
            s = pd.Timestamp(start, tz="UTC")
            e = pd.Timestamp(end, tz="UTC")
            if s <= ts < e:
                return q
        return None

    # ──── Fee application ────────────────────────────────────────────
    @staticmethod
    def apply_fees(trade: Trade, variant: str) -> float:
        """Return net P&L for this trade under the given fee variant."""
        cfg = FEE_VARIANTS[variant]
        gross = trade.gross_pnl_usd
        entry_fee = trade.notional_usd * cfg["entry_pct"]
        # Scalper-offer eligibility per symbol
        if cfg["scalper"]:
            window = 1800 if trade.symbol in ("BTC", "ETH") else 900
            eligible = trade.holding_sec <= window
            exit_fee = 0.0 if eligible else trade.notional_usd * 0.00059
        else:
            exit_fee = trade.notional_usd * cfg["exit_pct"]
        return gross - entry_fee - exit_fee

    # ──── Aggregation ────────────────────────────────────────────────
    def aggregate(self, trades: List[Trade], variant: str) -> CellResult:
        """Compute IS / Q3 / Q4 / OOS stats + verdict for one cell."""
        result = CellResult()
        result.n_total = len(trades)

        if not trades:
            return result

        # Bucket trades by quarter
        by_q: Dict[str, List[Tuple[Trade, float]]] = {}
        for t in trades:
            q = self.quarter_of(t.entry_ts)
            if q is None:
                continue
            net = self.apply_fees(t, variant)
            by_q.setdefault(q, []).append((t, net))

        is_trades = [(t, n) for q in IS_QUARTERS for (t, n) in by_q.get(q, [])]
        q3_trades = by_q.get("Q3", [])
        q4_trades = by_q.get("Q4", [])
        oos_trades = q3_trades + q4_trades

        def stats(buf: List[Tuple[Trade, float]]) -> Tuple[float, float, float, int, float]:
            if not buf:
                return (0.0, 0.0, 0.0, 0, 0.0)
            n = len(buf)
            net_sum = sum(n_pnl for _, n_pnl in buf)
            gross_sum = sum(t.gross_pnl_usd for t, _ in buf)
            ev = net_sum / n
            wr = sum(1 for _, n_pnl in buf if n_pnl > 0) / n
            return (ev, gross_sum, net_sum, n, wr)

        is_ev, is_gross, is_net, is_n, is_wr = stats(is_trades)
        q3_ev, _,        q3_net,  q3_n, _    = stats(q3_trades)
        q4_ev, _,        q4_net,  q4_n, _    = stats(q4_trades)
        oos_ev, oos_gross, oos_net, oos_n, oos_wr = stats(oos_trades)

        result.is_ev = is_ev;     result.is_n = is_n
        result.q3_ev = q3_ev;     result.q3_n = q3_n
        result.q4_ev = q4_ev;     result.q4_n = q4_n
        result.oos_ev = oos_ev;   result.oos_n = oos_n
        result.is_gross_sum = is_gross
        result.is_net_sum = is_net
        result.oos_gross_sum = oos_gross
        result.oos_net_sum = oos_net
        result.win_rate_is = is_wr
        result.win_rate_oos = oos_wr

        # Verdict
        if is_n < 10 or oos_n < 5:
            result.verdict = "INSUFFICIENT_DATA"
        elif is_ev < PASS_IS_EV_MIN:
            result.verdict = "KILL_IS_NEGATIVE"
        elif q4_ev < PASS_Q4_EV_MIN:
            result.verdict = "KILL_Q4_BELOW_FLOOR"
        else:
            gap = (is_ev - oos_ev) / abs(is_ev) if is_ev != 0 else float("inf")
            result.gap_pct = gap
            if gap > PASS_OOS_GAP_MAX:
                result.verdict = "KILL_OOS_GAP_TOO_BIG"
            elif q3_ev > 0 and q4_ev > 0:
                result.verdict = "PASS"
            else:
                result.verdict = "HOLD_OOS_MIXED_SIGN"
        return result

    # ──── Main runner ────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """Iterate full grid, return aggregated results, write outputs."""
        t0 = _time.time()
        all_results: Dict[str, Any] = {
            "study": self.study.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "fee_variants": self.fee_variants,
            "cells": {},
        }
        cells_run = 0
        cells_pass = 0

        for symbol in self.symbols:
            for tf in self.timeframes:
                try:
                    df = self.load_candles(symbol, tf)
                except FileNotFoundError as e:
                    if self.verbose:
                        print(f"  [{symbol} {tf}] SKIP — {e}")
                    continue
                if self.verbose:
                    print(f"\n=== {symbol} {tf} ({len(df)} bars) ===")

                for params in self.study.param_grid():
                    cells_run += 1
                    cell_id = self.study.cell_id(params)
                    trades = self.study.simulate(df, params)
                    if self.verbose:
                        print(f"  {cell_id}: {len(trades)} trades", end="")

                    for variant in self.fee_variants:
                        result = self.aggregate(trades, variant)
                        key = f"{symbol}|{tf}|{cell_id}|{variant}"
                        all_results["cells"][key] = asdict(result)
                        if result.verdict == "PASS":
                            cells_pass += 1
                            if self.verbose:
                                print(f"  ✓ PASS({variant})", end="")
                    if self.verbose:
                        print()

        all_results["finished_at"] = datetime.now(timezone.utc).isoformat()
        all_results["wall_sec"] = round(_time.time() - t0, 1)
        all_results["cells_run"] = cells_run
        all_results["cells_pass"] = cells_pass

        # Write outputs
        json_path = self.out_dir / "walkforward.json"
        json_path.write_text(json.dumps(all_results, indent=2, default=str))

        report_path = self.out_dir / "report.md"
        report_path.write_text(self._render_report(all_results))

        if self.verbose:
            print(f"\n=== DONE ({all_results['wall_sec']}s) ===")
            print(f"Cells run: {cells_run}  PASS: {cells_pass}")
            print(f"Wrote: {json_path}")
            print(f"Wrote: {report_path}")
        return all_results

    # ──── Report rendering ───────────────────────────────────────────
    def _render_report(self, results: Dict[str, Any]) -> str:
        """Markdown summary of W/F results."""
        lines = [
            f"# {results['study']} — Walk-Forward Report",
            f"",
            f"Generated: {results['finished_at']}",
            f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}",
            f"Symbols: {', '.join(results['symbols'])}",
            f"Timeframes: {', '.join(results['timeframes'])}",
            f"Fee variants: {', '.join(results['fee_variants'])}",
            f"",
            f"## Pass criteria",
            f"- IS EV per trade > ${PASS_IS_EV_MIN}",
            f"- Q4 EV per trade > ${PASS_Q4_EV_MIN}",
            f"- OOS gap ≤ {int(PASS_OOS_GAP_MAX*100)}%",
            f"- Q3 and Q4 both same sign as IS",
            f"",
            f"## Top 20 cells by Q4 EV (PASS only)",
            f"",
            f"| symbol | tf | cell | variant | IS_n | IS_EV | Q3_EV | Q4_EV | gap% | WR_oos |",
            f"|---|---|---|---|---|---|---|---|---|---|",
        ]
        passed = [
            (k, v) for k, v in results["cells"].items() if v["verdict"] == "PASS"
        ]
        passed.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        for k, v in passed[:20]:
            sym, tf, cell, variant = k.split("|", 3)
            gap = f"{v['gap_pct']*100:.0f}" if v["gap_pct"] is not None else "—"
            lines.append(
                f"| {sym} | {tf} | {cell} | {variant} | "
                f"{v['is_n']} | ${v['is_ev']:.3f} | ${v['q3_ev']:.3f} | "
                f"${v['q4_ev']:.3f} | {gap}% | {v['win_rate_oos']*100:.0f}% |"
            )
        if not passed:
            lines.append("| (none) | | | | | | | | | |")

        # Verdict distribution
        verdicts: Dict[str, int] = {}
        for v in results["cells"].values():
            verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
        lines.extend([
            f"",
            f"## Verdict distribution",
            f"",
            f"| Verdict | Count |",
            f"|---|---|",
        ])
        for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
            lines.append(f"| {v} | {n} |")
        return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────
# Stub strategy for smoke test (NOT for actual W/F use)
# ─────────────────────────────────────────────────────────────────────
class _StubBuyAndHoldStrategy(Strategy):
    """Trivial strategy — buys at the start of each week, sells 6h later.

    Used only to verify the harness plumbing works end-to-end. Real studies
    must implement their own simulate() with proper signal logic.
    """
    name = "stub_buy_and_hold"

    def param_grid(self):
        for hold_hours in [3, 6, 12]:
            yield {"hold_hours": hold_hours}

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        hold = int(params["hold_hours"])
        trades: List[Trade] = []
        # Pick one entry per week (Monday 12:00 UTC) — simple deterministic schedule
        symbol = df.attrs.get("symbol", "BTC")  # set by harness if needed
        if df.empty:
            return trades
        # Generate weekly entries
        df_local = df.copy()
        df_local["dow"] = df_local.index.dayofweek
        df_local["hour"] = df_local.index.hour
        candidates = df_local[(df_local["dow"] == 0) & (df_local["hour"] == 12)]
        for ts, row in candidates.iterrows():
            try:
                exit_ts = ts + pd.Timedelta(hours=hold)
                if exit_ts > df.index[-1]:
                    break
                # Find closest exit bar
                exit_idx = df.index.get_indexer([exit_ts], method="nearest")[0]
                if exit_idx <= 0:
                    continue
                exit_row = df.iloc[exit_idx]
                trades.append(Trade(
                    symbol=symbol,
                    side="long",
                    entry_price=float(row["close"]),
                    exit_price=float(exit_row["close"]),
                    notional_usd=1000.0,
                    holding_sec=hold * 3600,
                    entry_ts=ts,
                    exit_ts=df.index[exit_idx],
                    exit_reason="time_stop",
                ))
            except Exception:
                continue
        return trades


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== W/F Harness Template — Smoke Test ===\n")
    print("Running stub buy-and-hold strategy across BTC/ETH @ 1h to verify")
    print("plumbing. NOT a real W/F study — just validates the engine.\n")

    engine = WalkForwardEngine(
        study=_StubBuyAndHoldStrategy(),
        symbols=["BTC", "ETH"],
        timeframes=["1h"],
        out_dir=ROOT / "storage" / "wf_studies" / "_smoke_test",
    )
    results = engine.run()

    print("\n=== Smoke Test Summary ===")
    print(f"  Cells run: {results['cells_run']}")
    print(f"  Cells pass: {results['cells_pass']}")
    print(f"  Wall time: {results['wall_sec']}s")

    # Show the verdict distribution
    verdicts: Dict[str, int] = {}
    for v in results["cells"].values():
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    print(f"\n  Verdict distribution:")
    for v, n in verdicts.items():
        print(f"    {v}: {n}")
    print(f"\n  See: {ROOT}/storage/wf_studies/_smoke_test/")
    print("\nHarness is ready. Plug in TUNE concept by subclassing Strategy.")

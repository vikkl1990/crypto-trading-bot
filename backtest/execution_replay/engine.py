"""
Signal-replay backtest engine.

Given paper's captured signals (closed_signals.json), re-simulates each
trade under a different execution model (taker, maker, paper, or custom)
and produces a DataFrame of trades + risk-adjusted metrics.

The core idea:
  - Paper's `entry_price` = signal_price at emit time (bar close).
  - Paper's `exit_price` = indicator-fire tick price (idealized).
  - Paper's `pnl_usd` assumes these frictionless fills.

The backtest replaces paper's fills with a realistic fill model, then
recomputes P&L from the same trajectory (entry signal, peak, exit logic).

INPUTS it reads from each closed signal:
  - symbol, side, entry_price, exit_price     → replayed through fill model
  - contract_size, contracts (position size)
  - entry_time, exit_time                     → funding hours
  - scanner / grade / ml_probability           → for cohort analysis
  - mfe_r / exit_r                             → peak MFE (for analysis)

OUTPUTS per trade:
  - sim_entry_price, sim_exit_price
  - sim_gross_usd, sim_fees_usd, sim_funding_usd, sim_net_usd
  - paper_net_usd (from the source file — for comparison)
  - execution_gap_usd (paper_net - sim_net)

Current limitations (will address in full version):
  - Single-snapshot funding (no 8h reset)
  - Assumes `exit_price` from paper reflects realistic "next achievable" exit
  - Doesn't yet replay the trail logic tick-by-tick
  - No book-walk for large orders
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

from backtest.execution_replay.fill_model import (
    TakerFillModel, MakerFillModel, PaperFillModel, FillResult,
)
from backtest.execution_replay.cost_model import compute_round_trip_cost, TradeCost
from backtest.execution_replay.admit_policy import AdmitPolicy, AdmitAll
from backtest.execution_replay.exit_policy import (
    ExitPolicy, HistoricalExitPolicy, TradeState, get_policy as get_exit_policy,
)
from backtest.execution_replay.candle_cache import get_candles_window

logger = logging.getLogger(__name__)


class FillModelProto(Protocol):
    """Minimal interface for plug-in fill models."""
    def fill_entry(self, symbol: str, side: str, signal_price: float) -> FillResult: ...
    def fill_exit(self, symbol: str, side: str, signal_price: float) -> FillResult: ...


@dataclass
class SimulatedTrade:
    # Source signal metadata (from paper)
    trade_id:           str
    symbol:             str
    side:               str  # long / short
    scanner:            str
    grade:              str
    ml_probability:     float
    regime:             str

    # Paper's recorded values
    paper_entry_price:  float
    paper_exit_price:   float
    paper_pnl_usd:      float
    peak_mfe_r:         float
    exit_reason:        str

    # Timing
    entry_time:         datetime
    exit_time:          datetime
    hours_held:         float

    # Position sizing
    contracts:          float  # lots
    contract_size:      float
    notional_usd:       float

    # Simulated fills
    sim_entry_price:    float
    sim_exit_price:     float
    sim_entry_fill_type: str
    sim_exit_fill_type:  str
    sim_entry_slip_bps:  float
    sim_exit_slip_bps:   float

    # Simulated P&L
    sim_gross_usd:      float
    sim_entry_fee_usd:  float
    sim_exit_fee_usd:   float
    sim_funding_usd:    float
    sim_net_usd:        float

    # Gap vs paper
    execution_gap_usd:  float  # positive = sim worse than paper

    # Exit-policy outcome (gap (a), 2026-04-25).
    #   exit_policy_name: which policy decided the exit
    #   replayed_exit_reason: what the policy said (vs historical_exit_reason)
    #   replayed_exit_price: where the exit landed under the policy
    #   historical_exit_reason: original from closed_signals.json (always set)
    #   historical_exit_price: original from closed_signals.json (always set)
    # When exit_policy is None or 'historical' the replayed_* fields equal
    # the historical_* fields (preserves engine's pre-gap-a behavior).
    exit_policy_name:        str = "historical"
    replayed_exit_reason:    str = ""
    replayed_exit_price:     float = 0.0
    historical_exit_reason:  str = ""
    historical_exit_price:   float = 0.0
    bars_walked:             int = 0   # candles processed by exit_policy

    # Admit-policy outcome (gap (d), 2026-04-25).
    #   admit_decision: "admitted" | "rejected"
    #   admit_reason:    None when admitted; rejection_reason str when rejected
    # When admit_decision == "rejected" all sim_* / cost / gap fields are zero
    # and `paper_*` fields preserve what the historical bot actually did, so the
    # row remains the counterfactual record (net_pnl_usd=0 for rejected; the
    # delta vs paper_pnl_usd is "what we forfeited by filtering this trade out").
    admit_decision:     str = "admitted"
    admit_reason:       Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        return d


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _coerce_float(x, default=0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def load_signals(path: str | Path, since: Optional[datetime] = None) -> List[dict]:
    """
    Load closed paper signals from JSON file.
    Optionally filter to signals closed after `since`.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list")
    out = []
    for t in data:
        if not isinstance(t, dict):
            continue
        xt = _parse_dt(t.get("exit_time", ""))
        if xt is None:
            continue
        if since and xt < since:
            continue
        out.append(t)
    return out


def simulate_trade(
    signal: dict,
    fill_model: FillModelProto,
    default_funding_rate_8h: Optional[float] = None,
    admit_policy: Optional[AdmitPolicy] = None,
    exit_policy: Optional[ExitPolicy] = None,
) -> Optional[SimulatedTrade]:
    """
    Replay one paper signal through the given fill model + cost model.
    Returns None if signal has insufficient data.

    If `admit_policy` is provided AND it returns a rejection_reason, this
    returns a SimulatedTrade with admit_decision="rejected", zero P&L /
    cost / gap fields, and the original paper_* fields preserved (so the
    row remains a counterfactual record). Default `admit_policy=None`
    preserves existing behavior — every signal that parses cleanly is
    admitted.
    """
    try:
        symbol   = signal.get("symbol", "")
        side     = (signal.get("side") or "").lower()
        scanner  = signal.get("setup_type") or signal.get("scanner") or ""
        grade    = signal.get("grade", "?")
        regime   = signal.get("metadata", {}).get("regime", "?") if isinstance(signal.get("metadata"), dict) else "?"
        ml_prob  = _coerce_float(signal.get("ml_probability"))

        paper_entry = _coerce_float(signal.get("entry_price"))
        paper_exit  = _coerce_float(signal.get("exit_price"))
        paper_pnl   = _coerce_float(signal.get("pnl_usd"))
        peak_mfe    = _coerce_float(signal.get("mfe_r") or signal.get("peak_mfe_r"))
        exit_reason = signal.get("exit_reason") or signal.get("reason") or "?"

        entry_t = _parse_dt(signal.get("entry_time", ""))
        exit_t  = _parse_dt(signal.get("exit_time",  ""))
        if not (entry_t and exit_t) or paper_entry <= 0 or paper_exit <= 0:
            return None

        hours = max((exit_t - entry_t).total_seconds() / 3600.0, 0.0)

        contracts     = _coerce_float(signal.get("contracts") or signal.get("quantity"), 1.0)
        contract_size = _coerce_float(signal.get("contract_size"), 1.0)
        notional      = contracts * contract_size * paper_entry
        if notional <= 0:
            # Fall back to paper_stake × leverage for position notional
            stake = _coerce_float(signal.get("paper_stake"), 100.0)
            lev   = _coerce_float(signal.get("leverage"), 1.0)
            notional = stake * lev
            contracts = notional / (paper_entry * max(contract_size, 1.0))

        # Admit decision (gap (d), 2026-04-25). Done AFTER parsing the
        # signal (so the row carries the same metadata as an admitted row)
        # but BEFORE the fill model runs (so a rejected signal never
        # incurs simulated slippage/fees). All sim_* / cost / gap fields
        # are zero on rejection.
        if admit_policy is not None:
            rejection_reason = admit_policy.should_reject(signal)
            if rejection_reason:
                return SimulatedTrade(
                    trade_id=str(signal.get("trade_id") or signal.get("id") or ""),
                    symbol=symbol, side=side,
                    scanner=scanner, grade=grade, regime=regime,
                    ml_probability=ml_prob,
                    paper_entry_price=paper_entry,
                    paper_exit_price=paper_exit,
                    paper_pnl_usd=paper_pnl,
                    peak_mfe_r=peak_mfe,
                    exit_reason=exit_reason,
                    entry_time=entry_t,
                    exit_time=exit_t,
                    hours_held=hours,
                    contracts=contracts,
                    contract_size=contract_size,
                    notional_usd=notional,
                    sim_entry_price=0.0,
                    sim_exit_price=0.0,
                    sim_entry_fill_type="rejected",
                    sim_exit_fill_type="rejected",
                    sim_entry_slip_bps=0.0,
                    sim_exit_slip_bps=0.0,
                    sim_gross_usd=0.0,
                    sim_entry_fee_usd=0.0,
                    sim_exit_fee_usd=0.0,
                    sim_funding_usd=0.0,
                    sim_net_usd=0.0,
                    execution_gap_usd=0.0,
                    admit_decision="rejected",
                    admit_reason=rejection_reason,
                    exit_policy_name="n/a-rejected",
                    replayed_exit_reason="",
                    replayed_exit_price=0.0,
                    historical_exit_reason=exit_reason,
                    historical_exit_price=paper_exit,
                    bars_walked=0,
                )

        # === EXIT POLICY REPLAY (gap (a), 2026-04-25) ===
        # If exit_policy is provided AND not HistoricalExitPolicy, walk
        # forward through 1m candles applying the policy. The replayed
        # exit reason + price REPLACE the historical ones for the rest
        # of this function (cost calc operates on replayed prices).
        # On candle cache miss OR <2 candles in window: fall back to
        # historical exit (preserves pre-gap-a behavior; bars_walked=0).
        historical_exit_reason = exit_reason
        historical_exit_price = paper_exit
        replayed_exit_reason = exit_reason
        replayed_exit_price = paper_exit
        bars_walked = 0
        exit_policy_name = "historical"
        if exit_policy is not None and getattr(exit_policy, "name", "") != "historical":
            exit_policy_name = getattr(exit_policy, "name", "unknown")
            entry_unix_ts = int(entry_t.timestamp())
            meta_for_tt = signal.get("metadata") if isinstance(signal.get("metadata"), dict) else {}
            trade_type_name = (meta_for_tt or {}).get("trade_type", "") or "SCALP"
            max_window = 1800 if str(trade_type_name).upper() == "SCALP" else 3600
            candles_df = get_candles_window(symbol, entry_unix_ts, max_window_sec=max_window)
            if candles_df is not None and len(candles_df) >= 2:
                sl_initial = float((meta_for_tt or {}).get("stop_loss", 0) or 0)
                if sl_initial <= 0:
                    sl_initial = float(signal.get("stop_loss", 0) or 0)
                if sl_initial <= 0:
                    sl_initial = paper_entry * (0.99 if side == "long" else 1.01)
                tp_initial = float((meta_for_tt or {}).get("take_profit", 0) or 0)
                initial_risk = abs(paper_entry - sl_initial)
                state = TradeState(
                    entry_price=paper_entry,
                    side=side,
                    stop_loss=sl_initial,
                    take_profit=tp_initial,
                    initial_risk=initial_risk,
                    grade=str(grade),
                    regime=str(regime),
                    trade_type=str(trade_type_name),
                )
                entry_ms = entry_unix_ts * 1000
                fired = False
                for idx in range(len(candles_df)):
                    row = candles_df.iloc[idx]
                    ts_ms = int(row["timestamp"])
                    state.age_sec = max(0.0, (ts_ms - entry_ms) / 1000.0)
                    if initial_risk > 0:
                        hi = float(row["high"])
                        lo = float(row["low"])
                        cl = float(row["close"])
                        if side == "long":
                            bar_r = (hi - paper_entry) / initial_risk
                            state.current_r = (cl - paper_entry) / initial_risk
                        else:
                            bar_r = (paper_entry - lo) / initial_risk
                            state.current_r = (paper_entry - cl) / initial_risk
                        if bar_r > state.peak_mfe_r:
                            state.peak_mfe_r = bar_r
                    candle = {
                        "timestamp": ts_ms,
                        "open":  float(row["open"]),
                        "high":  float(row["high"]),
                        "low":   float(row["low"]),
                        "close": float(row["close"]),
                    }
                    if side == "long" and state.stop_loss > 0 and float(row["low"]) <= state.stop_loss:
                        candle["sl_hit"] = True
                    elif side != "long" and state.stop_loss > 0 and float(row["high"]) >= state.stop_loss:
                        candle["sl_hit"] = True
                    state.prev_bodies.append(abs(float(row["close"]) - float(row["open"])))
                    if len(state.prev_bodies) > 5:
                        state.prev_bodies = state.prev_bodies[-5:]
                    state.bar_count = idx + 1
                    bars_walked = idx + 1
                    reason = exit_policy.should_exit(state, candle)
                    if reason:
                        replayed_exit_reason = reason
                        if reason in ("sl_hit", "trail_profit"):
                            replayed_exit_price = state.stop_loss
                        else:
                            replayed_exit_price = float(row["close"])
                        fired = True
                        break
                if not fired:
                    replayed_exit_reason = "window_end"
                    replayed_exit_price = float(candles_df.iloc[-1]["close"])
                # Recompute hours based on replayed exit time
                try:
                    replay_exit_ms = int(candles_df.iloc[bars_walked - 1]["timestamp"])
                    hours = max((replay_exit_ms - entry_ms) / (3600.0 * 1000.0), 0.0)
                except Exception:
                    pass
            paper_exit = replayed_exit_price
            exit_reason = replayed_exit_reason

        # Simulate entry fill
        entry_fill = fill_model.fill_entry(symbol, side, paper_entry)
        if not entry_fill.filled:
            # Maker miss with no fallback — trade never happened.
            return None

        # Simulate exit fill
        exit_fill = fill_model.fill_exit(symbol, side, paper_exit)
        if not exit_fill.filled:
            # Can't close — treat as still open (skip from P&L)
            return None

        # Gross P&L from simulated fills
        if side == "long":
            gross = (exit_fill.fill_price - entry_fill.fill_price) * contracts * contract_size
        else:  # short
            gross = (entry_fill.fill_price - exit_fill.fill_price) * contracts * contract_size

        # Costs
        cost = compute_round_trip_cost(
            notional_usd=notional,
            side=side,
            entry_fill_type=entry_fill.fill_type,
            exit_fill_type=exit_fill.fill_type,
            hours_held=hours,
            funding_rate_8h=default_funding_rate_8h,
        )

        net = gross - cost.total_cost_usd
        gap = paper_pnl - net  # positive = paper's win was bigger than sim's

        return SimulatedTrade(
            trade_id=str(signal.get("trade_id") or signal.get("id") or ""),
            symbol=symbol, side=side,
            scanner=scanner, grade=grade, regime=regime,
            ml_probability=ml_prob,
            paper_entry_price=paper_entry,
            paper_exit_price=paper_exit,
            paper_pnl_usd=paper_pnl,
            peak_mfe_r=peak_mfe,
            exit_reason=exit_reason,
            entry_time=entry_t,
            exit_time=exit_t,
            hours_held=hours,
            contracts=contracts,
            contract_size=contract_size,
            notional_usd=notional,
            sim_entry_price=entry_fill.fill_price,
            sim_exit_price=exit_fill.fill_price,
            sim_entry_fill_type=entry_fill.fill_type,
            sim_exit_fill_type=exit_fill.fill_type,
            sim_entry_slip_bps=entry_fill.slippage_bps,
            sim_exit_slip_bps=exit_fill.slippage_bps,
            sim_gross_usd=gross,
            sim_entry_fee_usd=cost.entry_fee_usd,
            sim_exit_fee_usd=cost.exit_fee_usd,
            sim_funding_usd=cost.funding_usd,
            sim_net_usd=net,
            execution_gap_usd=gap,
            admit_decision="admitted",
            admit_reason=None,
            exit_policy_name=exit_policy_name,
            replayed_exit_reason=replayed_exit_reason,
            replayed_exit_price=replayed_exit_price,
            historical_exit_reason=historical_exit_reason,
            historical_exit_price=historical_exit_price,
            bars_walked=bars_walked,
        )
    except Exception as e:
        logger.debug("simulate_trade failed for trade_id=%s: %s",
                     signal.get("trade_id"), e)
        return None


def run_backtest(
    signals: List[dict],
    fill_model: FillModelProto,
    default_funding_rate_8h: Optional[float] = None,
    admit_policy: Optional[AdmitPolicy] = None,
    exit_policy: Optional[ExitPolicy] = None,
) -> pd.DataFrame:
    """
    Run backtest on a list of paper signals.
    Returns DataFrame with one row per simulated trade.

    `admit_policy` (optional) — if provided, signals it rejects appear as
    rows with admit_decision="rejected" and zero sim_* fields. Use the
    `admit_decision` column to split admitted vs rejected for cohort
    accounting (see `metrics.compute_metrics_with_admit_breakdown`).
    """
    trades: List[SimulatedTrade] = []
    skipped = 0
    for s in signals:
        sim = simulate_trade(
            s, fill_model, default_funding_rate_8h,
            admit_policy=admit_policy,
            exit_policy=exit_policy,
        )
        if sim is None:
            skipped += 1
            continue
        trades.append(sim)

    logger.info("Backtest: %d signals → %d simulated (%d skipped)",
                len(signals), len(trades), skipped)

    if not trades:
        return pd.DataFrame(columns=["net_pnl_usd", "exit_time"])  # empty-compatible

    df = pd.DataFrame([t.to_dict() for t in trades])
    # Standardize column name for metrics.compute_metrics
    df["net_pnl_usd"] = df["sim_net_usd"]
    return df

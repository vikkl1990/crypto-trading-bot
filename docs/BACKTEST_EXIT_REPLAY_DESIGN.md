# Backtest Exit-Logic Replay — Design Doc
**Author:** Backtest Engineer (Agent 13)
**Date:** 2026-04-25
**Status:** Design — not yet implemented. Sister doc: `.rollback/backtest_capability_survey_20260425_1625.md`.
**Owner sign-off needed:** architect, before any code lands.
**Implements:** Gap (a) from the capability survey — exit-logic replay.
**Defers:** Gaps (b)-(f) — sizing replay, maker-mode policy, cohort filter, mark alignment, lever combinations. They get separate design docs once (a) is shipped.

---

## 1. Why this exists

The current `backtest/execution_replay/` engine replays signals through different fill models but reads the historical `exit_reason` straight off disk. That is fine for "would taker fees have killed the edge" questions and useless for everything else.

Today (2026-04-25) we tried to use Edge Validator (Agent 1) to gate three changes — Wave 2 unified guard, Lever 3 trail-lock, and the proposed mark-alignment patch — and were forced to write three separate ad-hoc scripts (`counterfactual_exit_analyzer.py`, `lever3_flip_analysis.py`, `paper_vs_shadow_gap.py`) because the engine could not answer "what would have happened if the exit policy had been different." Each script reimplements pieces of the same loop: pull a trade, walk forward through 1m candles, evaluate a guard at each tick, compute counterfactual P&L. That is a backtest engine waiting to be asked.

This design promotes that loop into a first-class engine capability through an `ExitPolicy` abstraction. Once it lands, Edge Validator can demand "every PR that touches `execution/exit_guards.py` must include a backtest run showing the new policy's net P&L delta has a 95% CI lower bound above zero on at least 7 days of recent data." That is the gate we have been missing.

## 2. Scope

**In scope.** New module `backtest/execution_replay/exit_policy.py`. Modification of `backtest/execution_replay/engine.py:simulate_trade(...)` to accept an optional `ExitPolicy` and walk forward through cached candles. Candle-cache loader/aligner shared with future gaps. A fidelity-validation harness that runs `LegacyExitPolicy` on 7d of historical signals and asserts per-trade simulated outcomes match historical record within ±2% on net P&L and exact match on exit_reason for ≥95% of trades.

**Out of scope.** Re-deriving the entry decision (still trust the historical signal). Re-deriving sizing or fill type — those are gaps (b) and (c). Re-computing peak_mfe_r against a different mark — that is gap (e). Composing multiple policies into a "lever stack" — gap (f).

**Non-goal.** This design does **not** propose changing any production code in `execution/exit_guards.py` or `user_real_manager.py`. The backtest engine imports them; production stays the source of truth. If we ever need to backtest a hypothetical policy that does not exist in production, we add a new `ExitPolicy` subclass, not a fork of the live module.

## 3. Architecture

```
backtest/execution_replay/
├── engine.py                    [MODIFIED]
├── fill_model.py                [unchanged]
├── cost_model.py                [unchanged]
├── metrics.py                   [unchanged]
├── cohort_analysis.py           [unchanged]
├── candle_cache.py              [NEW — shared loader for gaps a,b,e]
└── exit_policy.py               [NEW — this design]
```

### 3.1 The `ExitPolicy` abstraction

```python
# backtest/execution_replay/exit_policy.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class TradeState:
    """Mutable state walked forward by the engine."""
    symbol:        str
    side:          Literal["long", "short"]
    entry_price:   float
    stop_loss:     float
    initial_risk:  float          # |entry - stop_loss|
    grade:         Optional[str]
    regime:        Optional[str]
    trade_type:    Optional[str]  # SCALP / INTRADAY / RUNNER
    age_sec:       float          # seconds since entry
    current_price: float
    current_r:     float          # signed (entry to current) / risk
    peak_mfe_r:    float          # max favorable R seen so far


@dataclass
class CandleTick:
    """One 1-minute candle being evaluated."""
    ts:     float    # unix seconds, bar OPEN time
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class ExitDecision:
    """Output of policy.should_exit(...)."""
    exit_reason: str   # short string like 'sl_hit' / 'dead_signal_unified'
    exit_price:  float # the price the policy would close at


class ExitPolicy(ABC):
    """Stateless decision module. One instance, called per trade per tick."""
    name: str = "abstract"

    @abstractmethod
    def should_exit(
        self,
        state:  TradeState,
        candle: CandleTick,
    ) -> Optional[ExitDecision]:
        """
        Return ExitDecision if the trade should close at this candle, else None.

        IMPORTANT: SL hit is checked FIRST inside this method using bar
        high/low (intra-bar fills). Other guards use bar close.
        """
```

That is the entire interface. Three properties matter.

**Stateless.** A policy holds no per-trade memory. All trade context is on `TradeState`, which the engine owns and updates each tick. This means one policy instance can serve thousands of trades in parallel; equally it means a policy can never accidentally leak data between trades. Phase 4.5 / Phase 5.1 / Phase 5.20.8 are all expressible without any cross-trade state.

**Single decision per tick.** The policy returns at most one `ExitDecision` per call. If none of the policy's guards fire at this minute, return `None` and the engine advances to the next candle. This forces every policy to declare priority order in one place rather than scattering kill conditions across helper functions.

**SL is the policy's job, not the engine's.** Earlier drafts of this design had the engine check SL before delegating to the policy. That was wrong — Lever 3's whole point is that the policy *moves* the SL when peak ≥ 0.20R, so the engine cannot know whether to short-circuit. By making SL the policy's first check, every policy can either honor the historical SL, override it, or trail it.

### 3.2 Concrete subclasses (initial three)

**`LegacyExitPolicy`** — the pre-Phase-5.20.8 cascade. Order: SL hit → quick_kill (age > 30s, peak ≤ 0.02R, current < -0.05R) → no_proof_of_life (A+/A only, age > 90s, peak < 0.05R, current < 0) → early_kill (B/C, age > 60s, peak < kill_mfe, current < 0) → dead_market (age ≥ 180s, quiet regime, peak < 0.08R, current < -0.10R) → zombie_kill (age > 600s, peak < 0.10R, current < 0) → mfe_pullback (peak ≥ 0.30R, current ≤ peak * 0.40) → exhaustion_wick (peak ≥ 0.30R, wick > 60% of body) → time_decay (1800s SCALP / 3600s other). This is the **fidelity baseline** — its job is to reproduce historical outcomes and prove the simulator is faithful before any new policy is trusted.

**`Phase58ExitPolicy`** — the Wave 2 unified guard that landed today. Imports `should_kill_dead_signal` from `execution.exit_guards` directly so the policy under test is always the live module (no copy-paste drift). Order: SL hit → dead_market → unified `should_kill_dead_signal` → mfe_pullback → exhaustion_wick → time_decay. This is the **policy we want to gate before promoting further**.

**`Lever3ExitPolicy`** — Phase 5.20.8 plus Lever 3's trail-lock. When `peak_mfe_r ≥ 0.20`, mutates `state.stop_loss` to `entry ± 0.20 * initial_risk` (long uses `+`, short uses `-`) before any other check. Then runs the Phase58 cascade with raised time gates (`quick_kill_min_age = 120` instead of 30, `no_proof_of_life` window doubled). This is the **policy admin runs today on his own user**; we need to gate before promoting to all users.

These three together cover every exit-logic A/B we are running this week. New policies (e.g. Phase 5.21 if it materialises) drop in as additional subclasses with no engine change.

### 3.3 Modified `simulate_trade(...)`

```python
def simulate_trade(
    signal:                  dict,
    fill_model:              FillModelProto,
    exit_policy:             Optional[ExitPolicy] = None,
    candle_loader:           Optional[CandleLoader] = None,
    default_funding_rate_8h: Optional[float] = None,
    max_replay_minutes:      int = 240,
) -> Optional[SimulatedTrade]:
```

When `exit_policy is None`, behavior is **identical to today** — read historical exit, simulate fills, done. This is the back-compat path; existing callers (CLI, tests) keep working unchanged.

When `exit_policy is not None`, the engine takes the candle path:

1. Build `TradeState` from the signal (entry_price, side, grade, regime, trade_type, stop_loss, initial_risk).
2. Call `candle_loader.candles_for(symbol, entry_time, entry_time + max_replay_minutes*60)` to get the 1m bars covering the trade.
3. For each candle in order: update `state.age_sec`, `state.current_price=close`, `state.current_r`, `state.peak_mfe_r` using the bar's intra-minute extreme. Call `exit_policy.should_exit(state, candle)`. On `ExitDecision`, record the decision and break.
4. If no candle fired a decision, exit at the last candle's close with reason `replay_window_end`.
5. Pass the policy's exit price into `fill_model.fill_exit(symbol, side, decision.exit_price)` — fill simulation is orthogonal to exit decision and we keep that separation strict.
6. Compute gross/fees/funding the same way as today using sim entry and the new sim exit.

The output `SimulatedTrade` row gains three new fields: `replayed_exit_reason` (from the policy), `replayed_exit_age_sec` (when the policy fired), `replayed_peak_mfe_r` (peak across the walked candles). The historical equivalents (`exit_reason`, `peak_mfe_r`) are preserved for delta analysis.

### 3.4 The `CandleLoader` interface

A thin wrapper around what `counterfactual_exit_analyzer` already does plus a parquet-cache fast path:

```python
class CandleLoader:
    def __init__(self, parquet_dir: Path, rest_cache_dir: Path):
        self.parquet_dir   = parquet_dir   # storage/candle_cache/*.parquet
        self.rest_cache_dir = rest_cache_dir # JSON file cache for REST gaps

    def candles_for(self, symbol: str, t_start: float, t_end: float) -> list[CandleTick]:
        """Return 1m candles in [t_start, t_end]. Tries parquet first, falls back
        to Delta REST + file cache for gaps. Always returns sorted ascending."""
```

The parquet fast-path (`storage/candle_cache/*_1m.parquet`) covers our 13 priority symbols at no API cost. For symbols not yet in the parquet cache, the loader falls back to the same Delta REST fetch + per-window JSON cache that `counterfactual_exit_analyzer.fetch_candles` already implements — we lift that function verbatim. First run on a fresh window backfills the cache; subsequent runs are local-only.

## 4. Validation — fidelity check (CRITICAL)

A new replay engine that produces different per-trade outcomes from the historical record could be (a) right and exposing a real production bug or (b) wrong and silently misleading every Edge Validator gate that follows. We need a clear test that distinguishes the two.

The validation harness is a separate CLI: `python3 -m backtest.execution_replay.validate_fidelity --days 7`. It:

1. Loads the last 7 days of closed signals.
2. Runs each through the engine **twice**: once with `exit_policy=None` (today's behavior, the historical-baseline) and once with `exit_policy=LegacyExitPolicy()` (replay the same historical guard cascade).
3. For each trade, computes `(replayed_exit_reason == historical_exit_reason)` and `abs(replayed_pnl - historical_pnl) / max(abs(historical_pnl), 1.0)`.
4. Asserts at least 95% of trades have matching exit reasons AND median P&L deviation under 2%, mean under 5%. (The 5% mean tolerance accounts for trades where intra-minute price paths differ from the live tick stream — 1m candles cannot perfectly recover sub-minute SL hits when the live monitor checked every 250ms.)
5. Prints a per-mismatch table so we can see exactly where the simulator deviates and decide whether each deviation is a real bug or a known limitation.

Until this harness passes, **no new ExitPolicy is trusted by Edge Validator.** That is non-negotiable; the value of the whole capability depends on the simulator being demonstrably faithful to live behavior.

Known limitations that may surface in the fidelity harness:
- 1m bars cannot reconstruct sub-minute SL sweeps. Trades where live SL fired between bar open and the next bar's open will show `replayed_exit_age` lagging by up to 60s.
- The live `dead_market` check uses the regime as known *at exit time*; the historical signal stores regime as known *at entry time*. If regime classification flipped during the trade, replay will diverge.
- Funding rate snapshots use a single 8h rate for the entire hold. For trades crossing a funding boundary, replayed funding cost may be off by up to 2x.

These limitations are documented up-front so they cannot become silent surprises later.

## 5. CLI integration

Extend `backtest/execution_replay/run.py` with `--exit-policy {historical, legacy, phase58, lever3}`. `historical` (default) preserves today's behavior exactly. The other three select an `ExitPolicy` subclass. When a non-historical policy is selected, the report adds a section "Replayed vs Historical Exit Outcomes" with per-trade exit-reason deltas and aggregate P&L delta with bootstrap CI (reusing the existing `metrics._bootstrap_pf_ci` machinery).

Example admin workflow tomorrow morning:

```bash
# Baseline
python3 -m backtest.execution_replay.run --days 7 --model paper --exit-policy historical
# Wave 2 candidate
python3 -m backtest.execution_replay.run --days 7 --model paper --exit-policy phase58
# Lever 3 candidate
python3 -m backtest.execution_replay.run --days 7 --model paper --exit-policy lever3
# Side-by-side
python3 -m backtest.execution_replay.run --days 7 --model paper --exit-policy historical,phase58,lever3
```

The third row of the comparison table now shows "exit policy" alongside "fill model" — Edge Validator's gate becomes "the policy with the highest lower-CI Sharpe under taker fills wins."

## 6. Effort and rollout plan

**Day 1 (≈260 LOC).** Land `exit_policy.py` (ABC + `LegacyExitPolicy`). Land `candle_cache.py` (parquet + REST loader). Plumb the optional `exit_policy` arg into `simulate_trade` with the candle-walking branch. No CLI change yet. Unit tests for the ABC and a single `LegacyExitPolicy` test trade.

**Day 2 (≈220 LOC).** Land `Phase58ExitPolicy` and `Lever3ExitPolicy`. Land the fidelity-validation harness (`validate_fidelity.py`). Run on 7d of data; iterate on `LegacyExitPolicy` until the 95%/2% bar passes. This is the day where the simulator either earns its keep or is sent back for repair.

**Day 3 (≈170 LOC).** Land CLI integration. Run all three policies on 7d / 14d / 30d windows. Hand the comparison report to Edge Validator and to the architect for review. If the report passes the same statistical bar that `counterfactual_exit_analyzer.py` enforces (delta > $200, delta WR > 20pp, lower CI > 0), we can deprecate the ad-hoc scripts.

**Total: ~650 LOC, 2-3 working days.** The estimate is dominated by Day 2 — if the fidelity harness reveals more deviations than expected, that day stretches.

## 7. Open questions for the architect

1. **Fidelity bar.** Is 95% exit-reason agreement plus 5% mean P&L deviation strict enough to gate new policies? I picked these from typical industry tolerances on 1m-resolution simulators; happy to tighten to 99%/2% if you want, but that probably needs sub-minute candle reconstruction from `delta_ws` ticks rather than 1m OHLC.

2. **Scope of LegacyExitPolicy.** The pre-Phase-5.20.8 cascade evolved over many phases (4.1, 4.5, 5.1, 5.3.1, 5.4, 5.5.3, 5.6-G). Reproducing every micro-phase exactly is a fool's errand — I propose freezing `LegacyExitPolicy` at "the cascade as of 2026-04-23 close" since that is what the closed_signals window we replay against actually used. Confirm or correct.

3. **Where do mark-alignment fixes live?** The fidelity harness will show non-trivial deviations from intra-minute SL sweeps and from mark-divergence (paper synthetic vs L2 mid). These are real production behaviors, not simulator bugs. I propose treating them as known limitations documented in the validation report rather than blocking the engine on solving them — gap (e) is a separate workstream. Agree?

4. **Should the engine carry both replayed AND historical exit outcomes?** I lean strongly yes — the delta IS the deliverable. But it doubles the column count of `SimulatedTrade`. Confirm before I commit to the schema.

5. **Defer or include `replay_window_end` exits in P&L?** When the policy never fires, we currently propose to exit at the last candle's close. An alternative is to drop these trades from the metrics with a logged warning. I prefer "exit at window end" because dropping selectively introduces a survivorship bias in the comparison. Confirm.

---

**Pending sign-off, this design becomes the implementation plan for the first PR family from Agent 13.** All other gaps wait their turn.

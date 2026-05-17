"""
Exit-policy abstraction for the execution-replay backtest engine.

The engine historically read `exit_reason` directly from each closed signal
and replayed only entry/exit fills. That blocked Edge Validator (Agent 1)
from gating ANY exit-policy change — Wave 2 unified guard, Lever 3
trail-lock, mark-alignment patch — because the question "what would have
happened with a different exit policy?" required candle-by-candle replay.

This module promotes the exit-decision into a first-class engine concern.

API
---
    TradeState              mutable state during candle-by-candle replay
    ExitPolicy              Protocol — must define should_exit(state, candle)
    HistoricalExitPolicy    no-op — returns the historical exit_reason from
                            the source signal (preserves engine's pre-gap-a
                            behavior bit-for-bit)
    LegacyExitPolicy        re-simulates the cascade frozen at 2026-04-23 close
                            (production policy at the closed_signals window)
    Phase58ExitPolicy       Wave 2 unified guard via execution.exit_guards
    Lever3ExitPolicy        trail-lock at peak >= 0.20R + raised time gates
                            (admin's current production policy)

The `should_exit` contract:
    Returns None to continue.
    Returns str (exit_reason) to exit at this candle's close.
    May mutate `state.stop_loss` to implement trail behavior.

Notes
-----
* Pure functions / data classes. No I/O, no side effects beyond state.
* Defensive: bad inputs (None, missing keys) silently no-op rather than
  raise — the engine should always make progress.
* Frozen Legacy: the `LegacyExitPolicy` is frozen at the 2026-04-23 close
  cascade. Future production changes go in NEW subclasses (Lever3,
  Phase58, etc.) so the historical fidelity check (legacy vs historical
  exit_reason >= 95% agreement) remains stable.

Author: Backtest Engineer (Agent 13) + architect (this session)
Implements: gap (a) from .rollback/backtest_capability_survey_20260425_1625.md
            See docs/BACKTEST_EXIT_REPLAY_DESIGN.md for full design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


# ── Constants matching production user_real_manager.py exit cascade ─────────

# _EARLY_KILL config (per trade_type) — matches user_real_manager.py:_EARLY_KILL
_EARLY_KILL = {
    "SCALP":    (60, 0.10),
    "INTRADAY": (90, 0.08),
    "RUNNER":   (0,  0.0),    # disabled
}

# Regimes treated as "active" for time-gate softening (mirrors production)
_ACTIVE_REGIMES = frozenset({
    "trending", "trending_up", "trending_down", "breakout", "high_volatility",
})

# Regimes treated as "chop" for dead_market gate (mirrors production)
_CHOP_REGIMES = frozenset({"quiet", "low_liquidity", "mean_reversion", ""})


# ──────────────────────────────────────────────────────────────────────────
# TradeState — mutable state during candle-by-candle replay
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TradeState:
    """Mutable trade state during candle-by-candle replay.

    Constructed at trade entry; mutated by the engine + ExitPolicy at each
    candle. The policy may mutate `stop_loss` (trail behavior). All other
    fields are managed by the engine.
    """
    entry_price:    float
    side:           str           # "long" or "short"
    stop_loss:      float          # MUTABLE — policies may trail
    take_profit:    float
    initial_risk:   float          # |entry - sl_initial|; stable
    grade:          str            # "A+", "A", "B", "C", "?"
    regime:         str            # "sideways", "trending_up", etc.
    trade_type:     str            # "SCALP", "INTRADAY", "RUNNER"

    # Engine-updated each tick
    age_sec:        float = 0.0
    peak_mfe_r:     float = 0.0
    current_r:      float = 0.0
    bar_count:      int = 0        # candles processed

    # Optional context for exhaustion checks (set when prev candles available)
    prev_bodies:    list = field(default_factory=list)  # last 3 candle body sizes


# ──────────────────────────────────────────────────────────────────────────
# ExitPolicy Protocol + concrete implementations
# ──────────────────────────────────────────────────────────────────────────
class ExitPolicy(Protocol):
    """Decides if a trade should exit at the current candle.

    Returns None to continue, str (exit_reason) to exit.
    May mutate state.stop_loss for trail behaviors.
    """
    name: str

    def should_exit(self, state: TradeState, candle: Dict[str, Any]) -> Optional[str]:
        ...


class HistoricalExitPolicy:
    """No-op policy: returns the historical exit_reason on the FIRST candle.

    Preserves engine's pre-gap-a behavior. Used as the default and as the
    baseline for fidelity comparison ("what does the historical record say").
    """
    name: str = "historical"

    def should_exit(self, state: TradeState, candle: Dict[str, Any]) -> Optional[str]:
        # The engine reads historical exit_reason directly when this policy is
        # in use; this method is a placeholder to satisfy the Protocol.
        return None


@dataclass
class LegacyExitPolicy:
    """Frozen at the production cascade as of 2026-04-23 close.

    Order of guards (matches `user_real_manager._monitor_trade` lines
    ~1855-2050 at that commit):

        0.   quick_kill          age > 30s,  peak <= 0.02R, current < -0.05R
        1.   SL hit / trail_profit
        2.   no_proof_of_life    A+/A only: age > 90s (180s in trend),
                                 peak < 0.05R, current < 0
        2'.  early_kill          B/C only: age > _EARLY_KILL[type].sec,
                                 peak < _EARLY_KILL[type].mfe, current < 0
        3.   dead_market         age >= 180s, quiet/MR regime,
                                 peak < 0.08R, current < -0.10R
        3b.  zombie_kill         age > 600s, peak < 0.10R, current < 0
        4.   no_momentum         RUNNER only: age >= 600-900s,
                                 peak < 0.15-0.20R
        5.   mfe_pullback        peak >= 0.30R, current <= 40% of peak
        5b.  exhaustion_wick     age >= 60s, peak >= 0.30R,
                                 reversal wick > 60% of body
        6.   time_decay          age > 1800s SCALP / 3600s other
    """
    name: str = "legacy"

    def should_exit(self, state: TradeState, candle: Dict[str, Any]) -> Optional[str]:
        age = state.age_sec
        peak = state.peak_mfe_r
        current = state.current_r

        # Guard 1 — SL hit / trail_profit (always first; safety net)
        # SL hit detection happens in the engine via candle low/high; this
        # policy doesn't decide SL hits, the engine signals them via a
        # special candle key. If the engine sets candle["sl_hit"] = True,
        # we honor it.
        if candle.get("sl_hit"):
            # If SL has trailed past entry into profit, label as trail_profit.
            in_profit = (
                (state.side == "long" and state.stop_loss > state.entry_price) or
                (state.side != "long" and state.stop_loss < state.entry_price)
            )
            return "trail_profit" if in_profit else "sl_hit"

        # Guard 0 — quick_kill (all grades)
        if age > 30 and peak <= 0.02 and current < -0.05:
            return "quick_kill"

        # Guard 2 — no_proof_of_life (A+/A only, regime-aware time gate)
        grade_u = (state.grade or "").upper()
        rg = (state.regime or "").lower()
        if grade_u in ("A+", "A"):
            pol_sec = 180 if rg in _ACTIVE_REGIMES else 90
            if age > pol_sec and peak < 0.05 and current < 0:
                return "no_proof_of_life"
        else:
            # Guard 2' — early_kill (B/C only)
            kill_sec, kill_mfe = _EARLY_KILL.get(
                (state.trade_type or "SCALP").upper(), (60, 0.10)
            )
            if kill_sec > 0 and age > kill_sec and peak < kill_mfe and current < 0:
                return "early_kill"

        # Guard 3 — dead_market (chop regimes only)
        if age >= 180 and rg in _CHOP_REGIMES and peak < 0.08 and current < -0.10:
            return "dead_market"

        # Guard 3b — zombie_kill
        if age > 600 and peak < 0.10 and current < 0:
            return "zombie_kill"

        # Guard 4 — no_momentum (RUNNER only, regime-aware)
        if (state.trade_type or "").upper() == "RUNNER":
            nm_sec, nm_mfe = (900, 0.15) if rg in _ACTIVE_REGIMES else (600, 0.20)
            if age >= nm_sec and peak < nm_mfe:
                return "no_momentum"

        # Guard 5 — mfe_pullback
        if peak >= 0.30 and current <= peak * 0.40:
            return "mfe_pullback"

        # Guard 5b — exhaustion_wick (requires candle data)
        if age >= 60 and peak >= 0.30 and current > 0:
            try:
                op = float(candle.get("open", 0))
                cl = float(candle.get("close", 0))
                hi = float(candle.get("high", 0))
                lo = float(candle.get("low", 0))
                body = abs(cl - op)
                if body > 0:
                    if state.side == "long":
                        upper_wick = hi - max(op, cl)
                        if upper_wick > body * 0.6:
                            return "exhaustion_wick"
                    else:  # short
                        lower_wick = min(op, cl) - lo
                        if lower_wick > body * 0.6:
                            return "exhaustion_wick"
                # Shrink check — 3 consecutive shrinking bodies
                if len(state.prev_bodies) >= 2:
                    bodies = state.prev_bodies[-2:] + [body]
                    if bodies[0] > bodies[1] > bodies[2]:
                        return "exhaustion_shrink"
            except (TypeError, ValueError):
                pass

        # Guard 6 — time_decay
        max_age = 1800 if (state.trade_type or "").upper() == "SCALP" else 3600
        if age > max_age:
            mins = int(age / 60)
            return f"time_decay_{mins}m"

        return None


@dataclass
class Phase58ExitPolicy:
    """Wave 2 unified dead-signal guard (FALSIFIED 2026-04-25 by counterfactual).

    Retained as a backtest policy for regression testing — proves that the
    re-simulation infra correctly reproduces the falsification. The architect
    chose NOT to ship this policy; admin runs `lever3` instead.

    Delegates to `execution.exit_guards.should_kill_dead_signal` — the same
    function user_real_manager would call if any user had `exit_policy='refactored'`.
    """
    name: str = "phase58"

    def should_exit(self, state: TradeState, candle: Dict[str, Any]) -> Optional[str]:
        # SL hit always takes precedence (safety net)
        if candle.get("sl_hit"):
            in_profit = (
                (state.side == "long" and state.stop_loss > state.entry_price) or
                (state.side != "long" and state.stop_loss < state.entry_price)
            )
            return "trail_profit" if in_profit else "sl_hit"

        try:
            from execution.exit_guards import should_kill_dead_signal
            return should_kill_dead_signal(
                age_sec=state.age_sec,
                current_r=state.current_r,
                peak_mfe_r=state.peak_mfe_r,
                grade=state.grade,
                entry=state.entry_price,
                sl=state.stop_loss,
                trade_type=state.trade_type,
                regime=state.regime,
            )
        except ImportError:
            return None


@dataclass
class Lever3ExitPolicy:
    """Lever 3 = trail-lock at peak >= 0.20R + raised quick_kill/no_proof_of_life.

    Currently live for admin (exit_policy='lever3'). This is the policy
    Wave 6.C is testing.

    Behavior:
      1. SL hit (always first)
      2. TRAIL-LOCK: when peak_mfe_r >= 0.20R, mutate state.stop_loss to
                     entry + 0.20*R_distance (long) or entry - 0.20*R_distance (short)
                     (raise-only — never lowers an existing SL)
      3. quick_kill min_age raised 30s -> 120s
      4. no_proof_of_life _pol_sec doubled (90s->180s chop, 180s->360s trend)
      5. All other Legacy guards apply unchanged
    """
    name: str = "lever3"
    trail_lock_threshold_r: float = 0.20
    quick_kill_min_age:     float = 120.0
    pol_sec_multiplier:     float = 2.0

    def should_exit(self, state: TradeState, candle: Dict[str, Any]) -> Optional[str]:
        # Apply trail-lock BEFORE checking SL hit so a freshly-raised SL
        # might be hit on the same candle.
        if state.peak_mfe_r >= self.trail_lock_threshold_r:
            risk_dist = state.initial_risk
            if risk_dist > 0:
                if state.side == "long":
                    target_sl = state.entry_price + self.trail_lock_threshold_r * risk_dist
                    if target_sl > state.stop_loss:
                        state.stop_loss = target_sl
                else:
                    target_sl = state.entry_price - self.trail_lock_threshold_r * risk_dist
                    if target_sl < state.stop_loss:
                        state.stop_loss = target_sl

        # Re-check SL hit on this candle using updated stop_loss
        try:
            hi = float(candle.get("high", 0))
            lo = float(candle.get("low", 0))
            if state.side == "long" and lo > 0 and lo <= state.stop_loss:
                in_profit = state.stop_loss > state.entry_price
                return "trail_profit" if in_profit else "sl_hit"
            if state.side != "long" and hi > 0 and hi >= state.stop_loss:
                in_profit = state.stop_loss < state.entry_price
                return "trail_profit" if in_profit else "sl_hit"
        except (TypeError, ValueError):
            pass

        if candle.get("sl_hit"):
            in_profit = (
                (state.side == "long" and state.stop_loss > state.entry_price) or
                (state.side != "long" and state.stop_loss < state.entry_price)
            )
            return "trail_profit" if in_profit else "sl_hit"

        age = state.age_sec
        peak = state.peak_mfe_r
        current = state.current_r

        # Lever 3 modified Guard 0 — quick_kill min_age 30 -> 120
        if age > self.quick_kill_min_age and peak <= 0.02 and current < -0.05:
            return "quick_kill"

        # Lever 3 modified Guard 2 — no_proof_of_life _pol_sec doubled
        grade_u = (state.grade or "").upper()
        rg = (state.regime or "").lower()
        if grade_u in ("A+", "A"):
            base_pol_sec = 180 if rg in _ACTIVE_REGIMES else 90
            pol_sec = base_pol_sec * self.pol_sec_multiplier
            if age > pol_sec and peak < 0.05 and current < 0:
                return "no_proof_of_life"
        else:
            kill_sec, kill_mfe = _EARLY_KILL.get(
                (state.trade_type or "SCALP").upper(), (60, 0.10)
            )
            if kill_sec > 0 and age > kill_sec and peak < kill_mfe and current < 0:
                return "early_kill"

        # All other Legacy guards apply unchanged
        if age >= 180 and rg in _CHOP_REGIMES and peak < 0.08 and current < -0.10:
            return "dead_market"
        if age > 600 and peak < 0.10 and current < 0:
            return "zombie_kill"
        if (state.trade_type or "").upper() == "RUNNER":
            nm_sec, nm_mfe = (900, 0.15) if rg in _ACTIVE_REGIMES else (600, 0.20)
            if age >= nm_sec and peak < nm_mfe:
                return "no_momentum"
        if peak >= 0.30 and current <= peak * 0.40:
            return "mfe_pullback"
        if age >= 60 and peak >= 0.30 and current > 0:
            try:
                op = float(candle.get("open", 0))
                cl = float(candle.get("close", 0))
                hi = float(candle.get("high", 0))
                lo = float(candle.get("low", 0))
                body = abs(cl - op)
                if body > 0:
                    if state.side == "long":
                        upper_wick = hi - max(op, cl)
                        if upper_wick > body * 0.6:
                            return "exhaustion_wick"
                    else:
                        lower_wick = min(op, cl) - lo
                        if lower_wick > body * 0.6:
                            return "exhaustion_wick"
            except (TypeError, ValueError):
                pass
        max_age = 1800 if (state.trade_type or "").upper() == "SCALP" else 3600
        if age > max_age:
            mins = int(age / 60)
            return f"time_decay_{mins}m"

        return None


# ──────────────────────────────────────────────────────────────────────────
# Factory — pick a policy by name (used by run.py CLI)
# ──────────────────────────────────────────────────────────────────────────
_POLICY_REGISTRY = {
    "historical": HistoricalExitPolicy,
    "legacy":     LegacyExitPolicy,
    "phase58":    Phase58ExitPolicy,
    "lever3":     Lever3ExitPolicy,
}


def get_policy(name: str) -> ExitPolicy:
    """Factory — returns an ExitPolicy instance by name. Defaults to historical."""
    cls = _POLICY_REGISTRY.get((name or "historical").lower(), HistoricalExitPolicy)
    return cls()


__all__ = [
    "TradeState",
    "ExitPolicy",
    "HistoricalExitPolicy",
    "LegacyExitPolicy",
    "Phase58ExitPolicy",
    "Lever3ExitPolicy",
    "get_policy",
]

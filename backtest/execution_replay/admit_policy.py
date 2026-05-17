"""
Admit-policy abstraction for the execution-replay backtest engine.

The engine historically admitted every closed signal in the input file as
a simulated trade. That is correct when we want to ask "what fill model
would have made this set of trades profitable?" but wrong when the
question is "what if production had REJECTED some of those signals at
the admit gate?" — i.e. the cohort filter (Wave 6.C Lever 2) and any
future filter rule.

This module introduces an `AdmitPolicy` Protocol and two concrete
implementations:

    AdmitAll              — preserves today's behavior (default).
    CohortFilterPolicy    — replays Wave 6.C Lever 2:
                              reject if atr_pct_of_price <= 0.000225
                              reject if metadata.vwap_zone == 'penalty'

The policy receives the raw `signal` dict (one entry from
`closed_signals.json`) and returns:

    None                  → admit (engine simulates as usual)
    str (rejection_reason)→ reject (engine emits a SimulatedTrade row
                              with admit_decision='rejected', zero P&L,
                              for counterfactual accounting)

Notes:
  * The CohortFilterPolicy thresholds default to the production values
    as of 2026-04-25. Change them by passing constructor args.
  * Defensive: missing/malformed `entry_price`, `atr`, `metadata`, or
    `vwap_zone` all silently fall through to admit. This mirrors the
    production behavior in `execution/user_real_manager.py:qualify_signal`
    — bad data must NEVER auto-reject.
  * The Protocol is duck-typed; any object with a `should_reject(signal)`
    method works. This lets gap (a) future-extend with policies that hold
    state (e.g. a stateful "max-N-per-symbol-per-hour" rule).

Author: Backtest Engineer (Agent 13)
Created: 2026-04-25 — implements gap (d) from
         `.rollback/backtest_capability_survey_20260425_1625.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class AdmitPolicy(Protocol):
    """Decides whether a signal should be admitted to simulation.

    Returns None to admit, str (rejection_reason) to reject.
    """

    def should_reject(self, signal: dict) -> Optional[str]:
        ...


class AdmitAll:
    """Default policy — admits everything (matches today's engine behavior)."""

    name: str = "all"

    def should_reject(self, signal: dict) -> Optional[str]:
        return None


@dataclass
class CohortFilterPolicy:
    """Wave 6.C Lever 2 — reject by atr_pct + vwap_zone.

    Mirrors the production filter in
    `execution/user_real_manager.py:qualify_signal` (the
    `cohort_filter_enabled` branch, ~line 740 as of 2026-04-25).

    Defaults match production:
      atr_pct_threshold   = 0.000225  (reject when atr_pct <= threshold)
      vwap_penalty_value  = "penalty" (case-insensitive equality match)
    """

    atr_pct_threshold: float = 0.000225
    vwap_penalty_value: str = "penalty"
    name: str = "cohort_filter"

    def should_reject(self, signal: dict) -> Optional[str]:
        meta = signal.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        # Rule 1: atr_pct_of_price (atr / entry_price) <= threshold
        # Production matches: float() with try/except, only applies if
        # entry_price > 0 and atr > 0. Bad data falls through (admit).
        try:
            entry = float(signal.get("entry_price", 0) or 0)
            atr = float(meta.get("atr", 0) or 0)
            if entry > 0 and atr > 0:
                atr_pct = atr / entry
                if atr_pct <= self.atr_pct_threshold:
                    return f"cohort_filter_atr:{atr_pct:.6f}<={self.atr_pct_threshold}"
        except (TypeError, ValueError):
            # Mirror production: skip rule on bad numeric data, do NOT reject.
            pass

        # Rule 2: vwap_zone == 'penalty' (case-insensitive)
        vwap_zone = str(meta.get("vwap_zone", "") or "").lower()
        if vwap_zone == self.vwap_penalty_value:
            return "cohort_filter_vwap_penalty"

        return None

"""Unit tests for backtest/execution_replay/admit_policy.py.

Covers gap (d) — cohort-filter admit policy.

Verifies:
  AdmitAll
    - returns None for empty / minimal / full signal dicts
  CohortFilterPolicy
    - atr_pct exactly at threshold → reject (uses <= comparison)
    - atr_pct just below threshold → reject
    - atr_pct just above threshold → admit (None)
    - vwap_zone == 'penalty' → reject (case-insensitive)
    - vwap_zone == 'PENALTY' → reject (case-insensitive)
    - vwap_zone == 'clear' → admit
    - vwap_zone missing → admit (rule skipped)
  Defensive parsing
    - missing entry_price / atr / metadata → admit (don't reject on bad data)
    - non-numeric entry_price / atr → admit
    - metadata is None / not a dict → admit
"""
from __future__ import annotations

import os
import sys

# Allow `pytest tests/test_admit_policy.py` from the repo root without an
# installed package — same pattern as test_exit_guards.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from backtest.execution_replay.admit_policy import (
    AdmitAll,
    AdmitPolicy,  # Protocol — imported to assert duck-typing
    CohortFilterPolicy,
)


# ── AdmitAll ────────────────────────────────────────────────────────────────
class TestAdmitAll:
    def test_admits_empty_dict(self):
        assert AdmitAll().should_reject({}) is None

    def test_admits_minimal_signal(self):
        assert AdmitAll().should_reject({"symbol": "BTC/USDT"}) is None

    def test_admits_signal_with_filter_match_fields(self):
        # Even if the signal WOULD be rejected by cohort_filter, AdmitAll
        # must still admit it.
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.001, "vwap_zone": "penalty"},
        }
        assert AdmitAll().should_reject(sig) is None

    def test_implements_admit_policy_protocol(self):
        # Duck-typing check — Protocol membership.
        policy: AdmitPolicy = AdmitAll()
        assert hasattr(policy, "should_reject")


# ── CohortFilterPolicy: ATR rule ───────────────────────────────────────────
class TestCohortFilterPolicy_ATR:
    def test_atr_pct_exactly_at_threshold_rejects(self):
        # entry=100, atr=0.0225 → atr_pct = 0.000225 == threshold
        # Production uses <= so this MUST reject.
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.0225, "vwap_zone": "clear"},
        }
        result = CohortFilterPolicy().should_reject(sig)
        assert result is not None
        assert result.startswith("cohort_filter_atr:")
        assert "0.000225" in result

    def test_atr_pct_just_below_threshold_rejects(self):
        # entry=100, atr=0.020 → atr_pct = 0.000200 < 0.000225
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.020, "vwap_zone": "clear"},
        }
        result = CohortFilterPolicy().should_reject(sig)
        assert result is not None
        assert result.startswith("cohort_filter_atr:")

    def test_atr_pct_just_above_threshold_admits(self):
        # entry=100, atr=0.025 → atr_pct = 0.000250 > 0.000225
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.025, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_custom_threshold_changes_decision(self):
        # Same atr_pct = 0.000200 — would reject at default 0.000225 but
        # admits at custom 0.000150.
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.020, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy(atr_pct_threshold=0.000150).should_reject(sig) is None

    def test_realistic_eth_signal_admits(self):
        # ETH @ 2309.65 with atr 1.05 → atr_pct = 0.000455 (well above thresh)
        sig = {
            "entry_price": 2309.65,
            "metadata": {"atr": 1.05, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None


# ── CohortFilterPolicy: VWAP rule ──────────────────────────────────────────
class TestCohortFilterPolicy_VWAP:
    def test_vwap_zone_penalty_lowercase_rejects(self):
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.025, "vwap_zone": "penalty"},
        }
        assert CohortFilterPolicy().should_reject(sig) == "cohort_filter_vwap_penalty"

    def test_vwap_zone_penalty_uppercase_rejects(self):
        # Production lowercases the field before comparing — so 'PENALTY'
        # must reject too.
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.025, "vwap_zone": "PENALTY"},
        }
        assert CohortFilterPolicy().should_reject(sig) == "cohort_filter_vwap_penalty"

    def test_vwap_zone_mixed_case_rejects(self):
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.025, "vwap_zone": "Penalty"},
        }
        assert CohortFilterPolicy().should_reject(sig) == "cohort_filter_vwap_penalty"

    def test_vwap_zone_clear_admits(self):
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.025, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_vwap_zone_marginal_admits(self):
        # Real production values include "marginal" / "noise" — only
        # exact 'penalty' triggers rejection.
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.025, "vwap_zone": "marginal"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_vwap_zone_noise_admits(self):
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.025, "vwap_zone": "noise"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_vwap_zone_missing_admits(self):
        sig = {"entry_price": 100.0, "metadata": {"atr": 0.025}}
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_atr_rule_evaluated_before_vwap_rule(self):
        # When BOTH rules would reject, the ATR rule wins (production
        # checks ATR first).
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.020, "vwap_zone": "penalty"},
        }
        result = CohortFilterPolicy().should_reject(sig)
        assert result is not None
        assert result.startswith("cohort_filter_atr:")


# ── CohortFilterPolicy: defensive parsing ──────────────────────────────────
class TestCohortFilterPolicy_Defensive:
    def test_empty_signal_admits(self):
        assert CohortFilterPolicy().should_reject({}) is None

    def test_missing_entry_price_skips_atr_rule_and_admits(self):
        # No entry_price → ATR rule skipped; vwap_zone absent → admit.
        sig = {"metadata": {"atr": 0.001}}
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_missing_atr_skips_atr_rule_and_admits(self):
        sig = {"entry_price": 100.0, "metadata": {}}
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_missing_metadata_admits(self):
        sig = {"entry_price": 100.0}
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_metadata_none_admits(self):
        sig = {"entry_price": 100.0, "metadata": None}
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_metadata_not_a_dict_admits(self):
        # Production reads `meta = signal.get("metadata") or {}` and
        # then `.get(...)` on it. We mirror that defensively — if
        # metadata is some odd type (string, list, int) we fall back
        # to an empty dict and admit.
        sig = {"entry_price": 100.0, "metadata": "garbage"}
        assert CohortFilterPolicy().should_reject(sig) is None

        sig = {"entry_price": 100.0, "metadata": ["list", "not", "dict"]}
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_zero_entry_price_skips_atr_rule(self):
        # Production requires entry_price > 0 for the atr_pct calc.
        sig = {
            "entry_price": 0.0,
            "metadata": {"atr": 0.001, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_zero_atr_skips_atr_rule(self):
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": 0.0, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_negative_entry_price_skips_atr_rule(self):
        sig = {
            "entry_price": -100.0,
            "metadata": {"atr": 0.001, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_non_numeric_entry_price_admits(self):
        sig = {
            "entry_price": "not-a-number",
            "metadata": {"atr": 0.001, "vwap_zone": "clear"},
        }
        # Production wraps in try/except (TypeError, ValueError) and
        # falls through. We do the same.
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_non_numeric_atr_admits(self):
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": "n/a", "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_none_atr_skips_atr_rule(self):
        # `None` is treated as 0 by the `or 0` guard.
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": None, "vwap_zone": "clear"},
        }
        assert CohortFilterPolicy().should_reject(sig) is None

    def test_bad_atr_does_not_block_vwap_rule(self):
        # If ATR is malformed but VWAP rule would reject, VWAP STILL
        # fires — bad data on one rule doesn't disable the other.
        # Mirrors production behaviour.
        sig = {
            "entry_price": 100.0,
            "metadata": {"atr": "garbage", "vwap_zone": "penalty"},
        }
        assert CohortFilterPolicy().should_reject(sig) == "cohort_filter_vwap_penalty"


# ── CohortFilterPolicy: integration with engine semantics ─────────────────
class TestCohortFilterPolicy_Integration:
    """Verify the policy plays nicely when the engine wraps it."""

    def test_returns_str_or_none_only(self):
        # Engine code does `if rejection_reason:` so the policy MUST
        # return either None or a non-empty string. Test a variety
        # of inputs to verify no other types leak through.
        policy = CohortFilterPolicy()
        signals = [
            {},
            {"symbol": "BTC/USDT"},
            {"entry_price": 100.0, "metadata": {"atr": 0.001}},
            {"entry_price": 100.0, "metadata": {"vwap_zone": "penalty"}},
            {"entry_price": 100.0, "metadata": {"vwap_zone": "clear"}},
            {"entry_price": 0.0, "metadata": {"atr": 0.0}},
        ]
        for sig in signals:
            result = policy.should_reject(sig)
            assert result is None or (isinstance(result, str) and result)

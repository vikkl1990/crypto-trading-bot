-- Migration 016 — Phase 6.C — Wave 6.C Lever 1 + Lever 2
--
-- Adds two per-user flags for the paper-vs-shadow gap closure work:
--
-- 1. cohort_filter_enabled (BOOLEAN, default FALSE) — Lever 2
--    When TRUE, qualify_signal applies the two free filter rules
--    discovered by the paper-shadow gap analyzer:
--      a. reject if atr_pct_of_price <= 0.000225
--      b. reject if vwap_zone == 'penalty'
--    Both rules have $0 paper-winner cost in 24h sample.
--
-- 2. shadow_simulated_balance (NUMERIC, default NULL) — Lever 1
--    When set AND user is shadow_live, compute_size uses this value
--    instead of _cached_balance ($100 default). This raises the sizing
--    ceiling so shadow positions become realistic-live-mode-sized
--    instead of fee-tax-dominated $10-22 margins.
--    NULL = no override = current behavior (~$10-22 margins).
--    Recommended Wave 6.C live-A/B value: 1000.0 → margins land $40-220.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS cohort_filter_enabled BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS shadow_simulated_balance NUMERIC DEFAULT NULL
        CHECK (shadow_simulated_balance IS NULL OR shadow_simulated_balance > 0);

COMMENT ON COLUMN users.cohort_filter_enabled IS
    'Phase 6.C Lever 2 — when TRUE, qualify_signal applies discovered filter rules (atr_pct + vwap_penalty). See docs/PRE_SIGNAL_ENGINE_DESIGN.md Layer 0.';

COMMENT ON COLUMN users.shadow_simulated_balance IS
    'Phase 6.C Lever 1 — for shadow_live users, override _cached_balance with this value when sizing. Raises ceiling so positions become realistic-live-sized. NULL = no override.';

-- Migration 015 — Phase 5.20.8 — Per-user exit_policy flag for Wave 4 A/B
--
-- Adds a per-user opt-in for the unified dead-signal guard introduced in
-- Phase 5.20.8 (see docs/EXIT_GUARD_REFACTOR_5_8.md).
--
-- exit_policy values:
--   'current'    — legacy cascade: quick_kill / no_proof_of_life / early_kill / zombie_kill
--                  (the policy that produced 12% WR in 24h shadow data)
--   'refactored' — unified dead-signal guard with fee-floor invariant
--                  + ATR-relative grace window
--
-- Default is 'current' so behavior is unchanged for users without an
-- explicit setting. The Wave 3 A/B will set admin -> 'refactored' and
-- niranjan -> 'current' explicitly.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS exit_policy text NOT NULL DEFAULT 'current'
        CHECK (exit_policy IN ('current', 'refactored'));

-- Backfill existing rows to 'current' explicitly (safety; default already covers this)
UPDATE users SET exit_policy = 'current' WHERE exit_policy IS NULL;

COMMENT ON COLUMN users.exit_policy IS
    'Phase 5.20.8 — per-user exit guard policy. current=legacy cascade, refactored=unified dead-signal guard. See docs/EXIT_GUARD_REFACTOR_5_8.md';

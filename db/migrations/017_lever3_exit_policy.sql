-- Migration 017 — Phase 6.C — Wave 6.C Lever 3 — exit_policy='lever3'
--
-- Adds 'lever3' as a valid value for users.exit_policy.
--
-- exit_policy values now:
--   'current'    = legacy cascade (default)
--   'refactored' = Wave 2 unified guard (FALSIFIED — kept for code branch but no users)
--   'lever3'     = Trail-lock at peak ≥ 0.20R + raised quick_kill/no_proof_of_life
--                  time gates (120s/180s/360s vs 30s/90s/180s)
--
-- See Lever 3 finding: 17/17 paper-WIN → shadow-LOSS flips are EXIT_KILL.
-- All 17 paper trades exited via trail_profit; all 17 shadow trades killed
-- by aggressive guards. Trail-lock + raised time gates port paper's
-- behavior to shadow.

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_exit_policy_check;
ALTER TABLE users ADD CONSTRAINT users_exit_policy_check
    CHECK (exit_policy IN ('current', 'refactored', 'lever3'));

COMMENT ON COLUMN users.exit_policy IS
    'Phase 5.20.8/6.C — exit guard policy. current=legacy, refactored=Wave 2 unified guard (falsified, unused), lever3=trail-lock + raised time gates per docs/EXIT_GUARD_REFACTOR_5_8.md and Lever 3 analysis.';

-- Migration 010 — Phase 5.19.1: cohort blacklist pause flag
-- Date: 2026-04-25
--
-- Adds a per-user `cohort_blacklist_paused_until` timestamp. When set
-- to a future time, the user_real_manager.qualify_signal cohort check
-- is bypassed for that user.
--
-- Use case: parallel A/B/C maker mode test — admin needs to participate
-- in ALL signals to accumulate per-mode samples evenly. Without this
-- pause, recent shadow losses on ETH would block admin from ETH signals
-- and bias the test toward symbols where admin is unblocked.
--
-- Set + clear:
--   UPDATE users SET cohort_blacklist_paused_until = NOW() + INTERVAL '48 hours'
--    WHERE email = 'admin@vnedge.com';   -- pause for 48h
--   UPDATE users SET cohort_blacklist_paused_until = NULL
--    WHERE email = 'admin@vnedge.com';   -- resume

ALTER TABLE users
ADD COLUMN IF NOT EXISTS cohort_blacklist_paused_until TIMESTAMPTZ;

-- Pause admin's blacklist for 48h to enable clean multimode A/B/C
UPDATE users
SET cohort_blacklist_paused_until = NOW() + INTERVAL '48 hours'
WHERE email = 'admin@vnedge.com';

-- Sanity check
SELECT email, bot_mode, maker_patience_mode,
       cohort_blacklist_paused_until,
       NOW() < cohort_blacklist_paused_until AS pause_active
FROM users
WHERE bot_mode IS NOT NULL
ORDER BY email;

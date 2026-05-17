-- Migration 009 — Phase 5.19: maker mode multimode A/B/C test
-- Date: 2026-04-25
--
-- The existing maker_patience_mode column already supports the new value
-- 'multimode' (it's a VARCHAR with no enum constraint). This migration:
--   1. Sets admin@vnedge.com to 'multimode' so they participate in the
--      3-arm parallel test (per-trade hash → standard | patient | l2_aware)
--   2. Keeps niranjan_139@yahoo.co.in on 'standard' as the production
--      control arm and to preserve clean Fix 1b cohort analytics.
--   3. Documents the values for future reference.
--
-- Mode descriptions:
--   'standard'   — 500/350ms probe, 1.0x bp offset
--   'patient'    — 2500/1750ms probe, 2.5x bp offset
--   'aggressive' — legacy alias for 'patient' (backward compat only)
--   'l2_aware'   — walk L2 book to first level with cum_depth >= our_size
--   'multimode'  — per-trade A/B/C: hash signal_id → one of the three above
--
-- A/B/C test plan:
--   admin (shadow_live) → multimode → 33% standard / 33% patient / 33% l2_aware
--   niranjan (demo) → standard → control + clean Fix 1b cohort
--
-- Verdict comparison after 30+ trades per mode (~6-12h):
--   maker_fill_rate, slippage_bps, NET pnl, profit_factor per mode

UPDATE users
SET maker_patience_mode = 'multimode'
WHERE email = 'admin@vnedge.com';

-- Sanity check: print final distribution
-- (purely informational — psql will not error on a SELECT)
SELECT email, bot_mode, maker_patience_mode
FROM users
WHERE bot_mode IS NOT NULL
ORDER BY email;

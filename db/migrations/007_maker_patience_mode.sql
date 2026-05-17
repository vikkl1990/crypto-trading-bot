-- Migration 007 — Phase 5.14 maker patience mode (A/B execution discipline)
-- Date: 2026-04-25
--
-- Adds per-user `maker_patience_mode` column for A/B testing of how
-- aggressively to wait for maker fills. Calibration showed 100% maker
-- miss rate on standard timings (122/122 fills are taker). Patient mode
-- gives the resting limit order more time to fill before falling through.
--
-- Values:
--   'standard'   — existing behavior (500ms tier1 / 350ms tier2, 1x bp offsets)
--   'patient'    — 5x probe time + 2.5x bp offsets (2500ms / 1750ms)
--   'aggressive' — 10x probe time + 3.5x bp offsets (5000ms / 3500ms)
--
-- A/B plan:
--   admin@vnedge.com           → 'patient'  (test the lever)
--   niranjan_139@yahoo.co.in   → 'standard' (control)
-- Compare 24-48h: maker fill rate %, slippage bps, NET P&L per trade.
-- Promote winning mode to default if statistically significant.

ALTER TABLE users
ADD COLUMN IF NOT EXISTS maker_patience_mode VARCHAR(20) DEFAULT 'standard';

-- Initialize: admin gets patient (test arm), all others standard (control).
UPDATE users
SET maker_patience_mode = 'patient'
WHERE email = 'admin@vnedge.com';

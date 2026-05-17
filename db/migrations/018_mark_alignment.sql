-- Migration 018 — Lever 3 Fix #3: Shadow Exit-Mark Alignment
-- Date: 2026-04-25
--
-- Adds per-user mark_alignment_enabled flag for the Lever 3 Fix #3
-- A/B. When TRUE, shadow's _monitor_trade reads paper's
-- highest_price/lowest_price watermark from signal_tracker._active
-- as the primary peak_mfe_r source, with the existing snapshot/
-- candle/tick ratchets retained as cold-start fallback.
--
-- Why: paper updates highest_price synchronously on every WS tick,
-- shadow polls _ws_prices once per 500 ms iteration. Phase 4.3/5.2
-- back-fills are time-gated and disabled in the first 0-60 s after
-- entry — exactly when Lever 3 Fix #1's trail-lock @ 0.20 R fires.
-- See docs/MARK_ALIGNMENT_DESIGN.md.
--
-- A/B plan: admin@vnedge.com = test arm (TRUE), niranjan_139 = control
-- (FALSE). 48 h soak with auto-revert watchdog (mig 013) on shadow WR.

ALTER TABLE users
ADD COLUMN IF NOT EXISTS mark_alignment_enabled BOOLEAN DEFAULT FALSE;

UPDATE users SET mark_alignment_enabled = TRUE
WHERE email = 'admin@vnedge.com';

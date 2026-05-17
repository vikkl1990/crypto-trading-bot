-- Migration 023: Bybit live trading support
-- Created 2026-04-26
-- Purpose: per-user Bybit execution mode independent of bot_mode (which is Delta).

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS bybit_mode VARCHAR(16) NOT NULL DEFAULT 'paper';

-- Allowed values: paper | shadow | live
-- paper:  no Bybit dispatch (bot-wide paper signals only)
-- shadow: bybit_shadow_simulator cron mirrors signals (already shipped)
-- live:   real orders placed on Bybit via private WebSocket

CREATE INDEX IF NOT EXISTS idx_users_bybit_live
    ON users (bybit_mode)
    WHERE bybit_mode = 'live' AND is_active = TRUE;

COMMENT ON COLUMN users.bybit_mode IS
  'Bybit execution mode: paper | shadow | live. Independent of bot_mode (Delta).';

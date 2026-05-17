-- Migration 011 — Phase 5.20-B1: GLOBAL kill switch
-- Date: 2026-04-25
--
-- Adds a single `bot_state.kill_switch_engaged` flag that the bot checks
-- at the top of every entry path (admission gate). When set:
--   - All new entries are rejected with reason 'kill_switch_engaged'
--   - Optionally: all open positions are emergency-closed at market
--
-- Engage:
--   UPDATE bot_state SET kill_switch_engaged=TRUE,
--                       kill_switch_reason='manual halt by ops',
--                       kill_switch_engaged_at=NOW(),
--                       kill_switch_close_open=TRUE;
--
-- Release:
--   UPDATE bot_state SET kill_switch_engaged=FALSE;
--
-- The bot polls this every 5 seconds (cheap query, in-memory cache).

CREATE TABLE IF NOT EXISTS bot_state (
    id                          SERIAL PRIMARY KEY,
    kill_switch_engaged         BOOLEAN DEFAULT FALSE,
    kill_switch_reason          VARCHAR(255),
    kill_switch_engaged_at      TIMESTAMPTZ,
    kill_switch_engaged_by      VARCHAR(255),
    kill_switch_close_open      BOOLEAN DEFAULT FALSE,
    -- Future: per-strategy kill flags
    updated_at                  TIMESTAMPTZ DEFAULT NOW()
);

-- Singleton row
INSERT INTO bot_state (id, kill_switch_engaged) VALUES (1, FALSE)
ON CONFLICT (id) DO NOTHING;

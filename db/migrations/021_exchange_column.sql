-- Migration 021: exchange tagging on user_trades
-- Created 2026-04-26
-- Purpose: enable side-by-side comparison of strategy performance across exchanges
-- (Delta India is current; Bybit added today as second shadow venue).

ALTER TABLE user_trades
    ADD COLUMN IF NOT EXISTS exchange VARCHAR(32) NOT NULL DEFAULT 'delta_india';

CREATE INDEX IF NOT EXISTS idx_user_trades_exchange_recent
    ON user_trades (exchange, opened_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_trades_exchange_type
    ON user_trades (exchange, trade_type, closed_at DESC)
    WHERE closed_at IS NOT NULL;

COMMENT ON COLUMN user_trades.exchange IS
  'Origin venue. Values: delta_india (default, all historical), bybit, okx (future).';

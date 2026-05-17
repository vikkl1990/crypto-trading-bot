-- Migration 012 — Phase 5.20-C: ops audit + EOD recon + decay tracking
-- Date: 2026-04-25
--
-- Three new tables for operational rigor:
--   1. ops_actions     — every operator-initiated change (kill, pause, threshold)
--   2. eod_recon       — daily P&L reconciliation snapshots
--   3. config_snapshot — bot config state at every restart

CREATE TABLE IF NOT EXISTS ops_actions (
    id              SERIAL PRIMARY KEY,
    action_at       TIMESTAMPTZ DEFAULT NOW(),
    action_type     VARCHAR(50) NOT NULL,   -- kill_engage, kill_release, pause_user, threshold_change
    target          VARCHAR(255),            -- user email, symbol, etc.
    actor           VARCHAR(255),            -- cli, dashboard, scheduler, auto-revert
    prev_value      TEXT,
    new_value       TEXT,
    reason          TEXT
);
CREATE INDEX IF NOT EXISTS idx_ops_actions_at ON ops_actions(action_at DESC);

CREATE TABLE IF NOT EXISTS eod_recon (
    recon_date      DATE PRIMARY KEY,
    bot_pnl_usd     DOUBLE PRECISION,        -- sum from user_trades (closed today)
    delta_pnl_usd   DOUBLE PRECISION,        -- from Delta statement (NULL if testnet)
    diff_usd        DOUBLE PRECISION,        -- bot - delta
    diff_pct        DOUBLE PRECISION,
    n_trades        INTEGER,
    fees_usd        DOUBLE PRECISION,
    funding_usd     DOUBLE PRECISION,
    reconciled_at   TIMESTAMPTZ DEFAULT NOW(),
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS config_snapshot (
    id              SERIAL PRIMARY KEY,
    captured_at     TIMESTAMPTZ DEFAULT NOW(),
    bot_pid         INTEGER,
    git_sha         VARCHAR(40),             -- best-effort from git
    config_hash     VARCHAR(64),
    config_json     JSONB                    -- full config snapshot
);
CREATE INDEX IF NOT EXISTS idx_config_snap_at ON config_snapshot(captured_at DESC);

-- Strategy decay tracking: per-scanner rolling Sharpe + WR
CREATE TABLE IF NOT EXISTS strategy_decay (
    id              SERIAL PRIMARY KEY,
    measured_at     TIMESTAMPTZ DEFAULT NOW(),
    scanner         VARCHAR(50),
    window_days     INTEGER,
    n_trades        INTEGER,
    win_rate_pct    DOUBLE PRECISION,
    avg_pnl_usd     DOUBLE PRECISION,
    sharpe_ann      DOUBLE PRECISION,
    sharpe_vs_baseline_pct DOUBLE PRECISION,  -- how much current Sharpe vs 90d baseline
    is_degraded     BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_decay_at ON strategy_decay(measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_decay_scanner ON strategy_decay(scanner, measured_at DESC);

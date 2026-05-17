-- Migration 013 — Phase 5.20-D: Auto-revert system
-- Date: 2026-04-25
--
-- Detects post-deploy degradation and automatically rolls back the offending
-- change. Inspired by SRE principle: "if your deploy made things worse, revert
-- before paging a human".
--
-- Tables:
--   deployment_baseline — captured on every bot start; metric snapshot for comparison
--   deployment_revert_log — every auto-revert action with reason + metric evidence
--
-- Trigger conditions (any one):
--   - WR drops > 10pp over 30+ trades vs prior 30 trades
--   - Avg P&L/trade drops > $0.50 over 30+ trades vs prior baseline
--   - Latency p95 increases > 200ms
--   - SL_REVERT spike (>5/hr)
--   - Crash count > 1 in 1h

CREATE TABLE IF NOT EXISTS deployment_baseline (
    id                          SERIAL PRIMARY KEY,
    captured_at                 TIMESTAMPTZ DEFAULT NOW(),
    bot_pid                     INTEGER,
    git_sha                     VARCHAR(40),
    config_hash                 VARCHAR(64),
    -- Phase / version markers from active code
    active_phases               JSONB,
    -- Pre-deploy metrics (last 30 trades before this restart)
    baseline_n_trades           INTEGER,
    baseline_wr_pct             DOUBLE PRECISION,
    baseline_avg_pnl_usd        DOUBLE PRECISION,
    baseline_sharpe_7d          DOUBLE PRECISION,
    baseline_latency_p95_ms     DOUBLE PRECISION,
    baseline_sl_revert_per_hr   DOUBLE PRECISION,
    -- Status
    is_active                   BOOLEAN DEFAULT TRUE,
    superseded_by_id            INTEGER REFERENCES deployment_baseline(id),
    notes                       TEXT
);
CREATE INDEX IF NOT EXISTS idx_deploy_baseline_at ON deployment_baseline(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_deploy_baseline_active
  ON deployment_baseline(is_active) WHERE is_active = TRUE;

CREATE TABLE IF NOT EXISTS deployment_revert_log (
    id                  SERIAL PRIMARY KEY,
    detected_at         TIMESTAMPTZ DEFAULT NOW(),
    baseline_id         INTEGER REFERENCES deployment_baseline(id),
    trigger_reason      VARCHAR(100),       -- 'wr_drop', 'pnl_drop', 'latency_spike', etc.
    trigger_detail      JSONB,               -- numbers that triggered it
    revert_action       VARCHAR(100),       -- 'engaged_kill_switch', 'reverted_phase_X', 'manual_required'
    reverted_at         TIMESTAMPTZ,
    operator_notified   BOOLEAN DEFAULT FALSE,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_revert_log_at ON deployment_revert_log(detected_at DESC);

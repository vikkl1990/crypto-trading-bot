-- Migration 020: kill_switch_close_audit
-- Created 2026-04-26 (Silent Failure #13 fix per docs/KILL_SWITCH_CLOSE_OPEN_LISTENER_v1.md)
--
-- Records every kill_switch_close_open invocation for forensic audit.

CREATE TABLE IF NOT EXISTS kill_switch_close_audit (
    id                SERIAL PRIMARY KEY,
    triggered_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    triggered_by      VARCHAR(64),
    reason            TEXT,
    positions_closed  INT NOT NULL DEFAULT 0,
    positions_failed  INT NOT NULL DEFAULT 0,
    detail            JSONB
);

CREATE INDEX IF NOT EXISTS idx_kill_switch_close_audit_recent
    ON kill_switch_close_audit (triggered_at DESC);

COMMENT ON TABLE kill_switch_close_audit IS
  'Audit trail for kill_switch_close_open listener (orchestrator).';

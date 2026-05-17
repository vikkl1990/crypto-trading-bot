-- Migration 019: auto_revert_events table
-- Created 2026-04-26 (Architect-call #14: ALERTS row population for Overview Strip)
--
-- Stores discrete events emitted by the auto_revert_detector + cohort_drift_sentinel
-- + circuit-breaker trips + ML-calibration warnings. These events feed:
--   - Overview Strip ALERTS row (last 60min)
--   - Co-pilot "Decisions Queued For You" panel (future)
--   - Daily ops digest

CREATE TABLE IF NOT EXISTS auto_revert_events (
    id              SERIAL PRIMARY KEY,
    event_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type      VARCHAR(64) NOT NULL,
        -- 'auto_revert' | 'cohort_drift' | 'cb_trip' | 'ml_calibration_drift' | 'silent_failure'
    severity        VARCHAR(16) NOT NULL DEFAULT 'warn',
        -- 'info' | 'ok' | 'warn' | 'error' | 'critical'
    title           VARCHAR(160) NOT NULL,
    detail          TEXT,
    cohort          VARCHAR(160),       -- optional: e.g. "trending_up/structure_bounce/long/asia"
    metric_key      VARCHAR(64),        -- optional: e.g. "wr_pct", "pf"
    metric_before   FLOAT,
    metric_after    FLOAT,
    user_email      VARCHAR(255),       -- optional: per-user events
    acknowledged    BOOLEAN DEFAULT FALSE,
    ack_by          VARCHAR(64),
    ack_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_auto_revert_events_recent
    ON auto_revert_events (event_at DESC);
CREATE INDEX IF NOT EXISTS idx_auto_revert_events_unacked
    ON auto_revert_events (acknowledged, severity, event_at DESC)
    WHERE acknowledged = FALSE;

COMMENT ON TABLE auto_revert_events IS
  'Discrete operational events for the architect (Overview Strip ALERTS row).';

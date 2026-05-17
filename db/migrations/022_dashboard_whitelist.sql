-- Migration 022: dashboard_whitelist
-- Created 2026-04-26
-- Purpose: DB-managed IP/CIDR allowlist for dashboard port 8080.
-- Sync'd to iptables every 5 minutes by scripts/dashboard_whitelist_sync.sh.

CREATE TABLE IF NOT EXISTS dashboard_whitelist (
    id              SERIAL PRIMARY KEY,
    ip_cidr         INET NOT NULL,           -- single IP (1.2.3.4) or CIDR (1.2.3.0/24)
    label           VARCHAR(80) NOT NULL,    -- human-readable, required
    granted_by      VARCHAR(64) NOT NULL DEFAULT 'cli',
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,             -- nullable = permanent
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    notes           TEXT,
    UNIQUE (ip_cidr)
);

CREATE INDEX IF NOT EXISTS idx_dashboard_whitelist_active
    ON dashboard_whitelist (enabled, expires_at)
    WHERE enabled = TRUE;

-- Seed with current architect IP so they don't lose access on first sync
INSERT INTO dashboard_whitelist (ip_cidr, label, granted_by, notes)
VALUES ('171.76.82.182/32', 'architect-home', 'migration_022_seed',
        'Architect primary IP — was previously hardcoded in iptables')
ON CONFLICT (ip_cidr) DO NOTHING;

COMMENT ON TABLE dashboard_whitelist IS
  'DB-managed IP allowlist for dashboard port 8080. Synced to iptables every 5 min.';

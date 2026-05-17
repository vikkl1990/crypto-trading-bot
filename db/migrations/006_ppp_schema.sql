-- Migration 006 — Peak Probability Predictor (PPP) schema
-- 2026-04-24 — VN Edge quant architect collaboration
--
-- PPP is a binary classifier that predicts peak_mfe_r >= 0.30 BEFORE entry.
-- It runs as an admission gate: below-threshold signals rejected before
-- per-user qualification. Shipped per docs/PPP_V2_IMPLEMENTATION_PLAN.md.
--
-- This migration creates:
--   1. signal_features  — one row per emitted signal, features + label + PPP decision
--   2. ppp_model_runs   — registry of trained models with OOF metrics
--   3. users.ppp_enabled / users.ppp_mode — per-user feature flag + mode
--
-- Idempotent: all DDL uses IF NOT EXISTS guards.

-- ==========================================================================
-- signal_features: feature snapshot + label + PPP decision per signal
-- ==========================================================================

CREATE TABLE IF NOT EXISTS signal_features (
    signal_id           VARCHAR(64) PRIMARY KEY,
    emitted_at          TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    side                VARCHAR(10) NOT NULL,

    -- Signal-time metadata (denormalized for fast queries)
    scanner_type        VARCHAR(40),
    grade               VARCHAR(4),
    confidence          INTEGER,
    regime              VARCHAR(30),
    ml_probability      DOUBLE PRECISION,

    -- Full feature vector (flexible schema evolution)
    features            JSONB NOT NULL,

    -- Label (backfilled after trade closes)
    peak_mfe_r          DOUBLE PRECISION,
    will_peak_30r       BOOLEAN,
    label_captured      BOOLEAN DEFAULT FALSE,
    label_trade_id      VARCHAR(64),
    label_captured_at   TIMESTAMPTZ,

    -- Prediction logging (each phase may populate; most recent wins)
    ppp_heuristic_score DOUBLE PRECISION,
    ppp_lr_score        DOUBLE PRECISION,
    ppp_lgb_score       DOUBLE PRECISION,

    -- Decision (what the gate DID at emit time)
    ppp_decision        VARCHAR(20),        -- 'admit' | 'reject' | 'failopen' | 'log_only'
    ppp_threshold       DOUBLE PRECISION,
    ppp_reason          VARCHAR(200),
    ppp_model_type      VARCHAR(20),        -- 'heuristic' | 'lr' | 'lgb'
    ppp_latency_ms      DOUBLE PRECISION,

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sigf_emitted
    ON signal_features(emitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_sigf_symbol_side
    ON signal_features(symbol, side, emitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_sigf_label
    ON signal_features(label_captured, will_peak_30r);

-- Partial index for unlabeled rows (what we're backfilling)
CREATE INDEX IF NOT EXISTS idx_sigf_unlabeled
    ON signal_features(emitted_at)
    WHERE label_captured = FALSE;

-- Partial index for decisions (for A/B analysis)
CREATE INDEX IF NOT EXISTS idx_sigf_decision
    ON signal_features(ppp_decision, emitted_at DESC)
    WHERE ppp_decision IS NOT NULL;

-- ==========================================================================
-- ppp_model_runs: registry of every trained PPP model
-- ==========================================================================

CREATE TABLE IF NOT EXISTS ppp_model_runs (
    run_id                  SERIAL PRIMARY KEY,
    model_type              VARCHAR(20) NOT NULL,  -- 'heuristic' | 'lr' | 'lgb'
    trained_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Training window
    training_window_start   TIMESTAMPTZ,
    training_window_end     TIMESTAMPTZ,
    n_training_samples      INTEGER,
    n_positives             INTEGER,

    -- OOF metrics (from TimeSeriesSplit cross-validation)
    oof_precision           DOUBLE PRECISION,
    oof_recall              DOUBLE PRECISION,
    oof_roc_auc             DOUBLE PRECISION,
    oof_pr_auc              DOUBLE PRECISION,

    -- Decision threshold selected from OOF
    chosen_threshold        DOUBLE PRECISION,

    -- Per-feature importance (JSONB: {feature_name: importance_score})
    feature_importances     JSONB,

    -- Model artifact location
    model_path              VARCHAR(255),

    -- Deployment state
    champion                BOOLEAN DEFAULT FALSE,  -- currently serving
    challenger              BOOLEAN DEFAULT FALSE,  -- in A/B canary
    retired_at              TIMESTAMPTZ,

    notes                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_ppp_runs_trained
    ON ppp_model_runs(trained_at DESC);

CREATE INDEX IF NOT EXISTS idx_ppp_runs_champion
    ON ppp_model_runs(champion)
    WHERE champion = TRUE;

-- ==========================================================================
-- users: per-user PPP feature flag + mode
-- ==========================================================================

-- ppp_enabled: master toggle (false = PPP bypasses for this user)
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS ppp_enabled BOOLEAN DEFAULT FALSE;

-- ppp_mode: 'off' | 'log' | 'enforce'
--   off     = PPP doesn't run for this user
--   log     = PPP runs, score logged, but signal admitted regardless
--   enforce = below-threshold signals rejected
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS ppp_mode VARCHAR(20) DEFAULT 'log';

-- When was PPP mode last changed (audit trail)
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS ppp_mode_updated_at TIMESTAMPTZ;

-- ==========================================================================
-- Verification query (not executed — just for reference)
-- ==========================================================================
--
-- SELECT
--   (SELECT COUNT(*) FROM signal_features) as sig_features_rows,
--   (SELECT COUNT(*) FROM ppp_model_runs) as ppp_runs,
--   (SELECT COUNT(*) FROM users WHERE ppp_mode IS NOT NULL) as users_with_ppp_mode;

-- Migration 008 — Phase 5.16: PPP regressor advisory column
-- Date: 2026-04-25
--
-- Adds advisory column for the regression PPP (predicts continuous peak_mfe_r)
-- alongside the existing binary classifier score (ppp_lr_score).
-- Both run in parallel for champion/challenger; regressor is log-only initially.

ALTER TABLE signal_features
ADD COLUMN IF NOT EXISTS ppp_regressor_score DOUBLE PRECISION;

ALTER TABLE signal_features
ADD COLUMN IF NOT EXISTS ppp_regressor_decision VARCHAR(20);
-- 'admit' | 'reject' | 'failopen' — what the regressor WOULD have decided
-- using its default_admit_threshold_r (set at training time)

ALTER TABLE signal_features
ADD COLUMN IF NOT EXISTS ppp_regressor_threshold DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_sigf_regressor_score
ON signal_features(ppp_regressor_score)
WHERE ppp_regressor_score IS NOT NULL;

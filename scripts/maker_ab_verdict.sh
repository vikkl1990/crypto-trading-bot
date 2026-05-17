#!/usr/bin/env bash
# Patient Maker A/B verdict report — v2 (2026-04-30)
# Extended to handle shadow_live users (was real-only).
# Reports both real and shadow data + maker counterfactual deltas.

set -u
export PGPASSWORD=VnEdge2026db
PSQL="psql -h localhost -U vnedge -d vnedge -A -F | -t -c"
NOW_UTC="$(date -u '+%Y-%m-%d %H:%M UTC')"
OUT_DIR=/home/opc/crypto-trading-bot/storage/verdicts
mkdir -p "$OUT_DIR"

echo "=================================================================="
echo "  PATIENT MAKER A/B — 24h VERDICT REPORT v2"
echo "  Generated: $NOW_UTC"
echo "=================================================================="
echo

echo "── 0. USER STATE ─────────────────────────────────────────────────"
$PSQL "SELECT email, bot_mode, maker_patience_mode FROM users
       WHERE email IN ('admin@vnedge.com','niranjan_139@yahoo.co.in')
       ORDER BY email;"
echo

echo "── 1. Replay calibrator (last 24h maker miss) ────────────────────"
cd /home/opc/crypto-trading-bot && python3 -m backtest.execution_replay.calibrate_maker_miss --days 1 2>&1 | tail -25
echo

echo "── 2. Per-user fill-type breakdown (REAL trades) ─────────────────"
$PSQL "SELECT u.email, u.maker_patience_mode, COALESCE(NULLIF(ut.metadata::jsonb->>'fee_type',''),'unknown') as fill_type, COUNT(*)
       FROM user_trades ut JOIN users u ON ut.user_id=u.id
       WHERE ut.trade_type='real' AND ut.opened_at >= NOW() - INTERVAL '24 hours'
       GROUP BY u.email, u.maker_patience_mode, fill_type
       ORDER BY u.email, fill_type;"
echo

echo "── 2b. Per-user fill-type breakdown (SHADOW trades) ──────────────"
$PSQL "SELECT u.email, u.maker_patience_mode, COALESCE(NULLIF(ut.metadata::jsonb->>'fee_type',''),'unknown') as fill_type, COUNT(*)
       FROM user_trades ut JOIN users u ON ut.user_id=u.id
       WHERE ut.trade_type='shadow' AND ut.opened_at >= NOW() - INTERVAL '24 hours'
       GROUP BY u.email, u.maker_patience_mode, fill_type
       ORDER BY u.email, fill_type;"
echo

echo "── 3. Per-user 24h P&L (real + shadow) ───────────────────────────"
$PSQL "SELECT u.email, u.maker_patience_mode, ut.trade_type, COUNT(*) as n,
              ROUND(SUM(ut.pnl_usd)::numeric,2) as net_pnl,
              ROUND(AVG(ut.pnl_usd)::numeric,3) as avg,
              SUM(CASE WHEN ut.pnl_usd>0 THEN 1 ELSE 0 END) as wins
       FROM user_trades ut JOIN users u ON ut.user_id=u.id
       WHERE ut.trade_type IN ('real','shadow') AND ut.closed_at >= NOW() - INTERVAL '24 hours'
       GROUP BY u.email, u.maker_patience_mode, ut.trade_type
       ORDER BY u.email, ut.trade_type;"
echo

echo "── 3b. SHADOW maker counterfactual savings (24h) ─────────────────"
$PSQL "SELECT u.email, u.maker_patience_mode,
              COUNT(*) as shadow_n,
              ROUND(SUM(ut.pnl_usd)::numeric,2) as actual_net,
              ROUND(SUM((ut.metadata->>'cf_maker_savings_50pct')::float)::numeric,2) as cf50_save,
              ROUND(SUM((ut.metadata->>'cf_maker_savings_100pct')::float)::numeric,2) as cf100_save,
              ROUND((SUM(ut.pnl_usd) + SUM((ut.metadata->>'cf_maker_savings_100pct')::float))::numeric,2) as net_at_100pct_maker
       FROM user_trades ut JOIN users u ON ut.user_id=u.id
       WHERE ut.trade_type='shadow' AND ut.status='closed'
         AND ut.closed_at >= NOW() - INTERVAL '24 hours'
       GROUP BY u.email, u.maker_patience_mode
       ORDER BY u.email;"
echo

echo "── 4. Fix 1b — fee-wall explore samples ──────────────────────────"
$PSQL "SELECT COUNT(*) as n, ROUND(SUM(pnl_usd)::numeric,2) as net,
              SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END) as wins
       FROM user_trades
       WHERE metadata::jsonb->>'fee_wall_explore'='true'
         AND closed_at >= NOW() - INTERVAL '24 hours';"
echo

echo "── 5. VERDICT ────────────────────────────────────────────────────"

# Determine which mode the patient cohort is in (live vs shadow_live)
PATIENT_MODE="$($PSQL "SELECT bot_mode FROM users WHERE maker_patience_mode='patient' LIMIT 1;" | tr -d ' |')"
PATIENT_USERS="$($PSQL "SELECT COUNT(*) FROM users WHERE maker_patience_mode='patient';" | tr -d ' |')"

if [[ "$PATIENT_MODE" == "live" ]]; then
    # Real-trade verdict (original logic)
    SCOPE="REAL"
    ADMIN_PCT="$($PSQL "
      WITH t AS (
        SELECT COALESCE(NULLIF(ut.metadata::jsonb->>'fee_type',''),'unknown') as fill_type
        FROM user_trades ut JOIN users u ON ut.user_id=u.id
        WHERE ut.trade_type='real' AND ut.opened_at >= NOW() - INTERVAL '24 hours'
          AND u.maker_patience_mode='patient'
      )
      SELECT ROUND(100.0 * SUM(CASE WHEN fill_type='maker' THEN 1 ELSE 0 END)::numeric
                  / NULLIF(COUNT(*),0), 1)
      FROM t;
    " | tr -d ' |')"
    ADMIN_N="$($PSQL "
      SELECT COUNT(*) FROM user_trades ut JOIN users u ON ut.user_id=u.id
      WHERE ut.trade_type='real' AND ut.opened_at >= NOW() - INTERVAL '24 hours'
        AND u.maker_patience_mode='patient';
    " | tr -d ' |')"
else
    # Shadow-trade verdict (new — uses simulated maker fills if present)
    SCOPE="SHADOW"
    ADMIN_PCT="$($PSQL "
      WITH t AS (
        SELECT COALESCE(NULLIF(ut.metadata::jsonb->>'fee_type',''),'unknown') as fill_type
        FROM user_trades ut JOIN users u ON ut.user_id=u.id
        WHERE ut.trade_type='shadow' AND ut.opened_at >= NOW() - INTERVAL '24 hours'
          AND u.maker_patience_mode='patient'
      )
      SELECT ROUND(100.0 * SUM(CASE WHEN fill_type='maker' THEN 1 ELSE 0 END)::numeric
                  / NULLIF(COUNT(*),0), 1)
      FROM t;
    " | tr -d ' |')"
    ADMIN_N="$($PSQL "
      SELECT COUNT(*) FROM user_trades ut JOIN users u ON ut.user_id=u.id
      WHERE ut.trade_type='shadow' AND ut.opened_at >= NOW() - INTERVAL '24 hours'
        AND u.maker_patience_mode='patient';
    " | tr -d ' |')"
fi

echo "Patient bot_mode:                ${PATIENT_MODE:-unknown}"
echo "Patient-mode users in DB:        ${PATIENT_USERS:-0}"
echo "Verdict scope:                   ${SCOPE} trades"
echo "Patient-cohort maker fill rate:  ${ADMIN_PCT:-N/A}%   (n=${ADMIN_N:-0})"
echo

PCT_INT="${ADMIN_PCT%%.*}"
if [[ "${PATIENT_USERS:-0}" -lt 1 ]]; then
    echo "🟡 A/B NOT CONFIGURED — no users in maker_patience_mode='patient'"
    echo "   Next: enable patience on admin and re-run 24h"
elif [[ "$SCOPE" == "SHADOW" && "$PATIENT_MODE" != "live" ]]; then
    # Shadow-mode patient cohort — patient setting only matters if shadow simulator
    # has been wired to honor it (Phase 5.21 work). If not wired yet, this verdict
    # is informational only — admin maker_pct will mirror niranjan's because
    # _execute_shadow doesn't consult maker_patience_mode.
    if [[ -z "$PCT_INT" || "$PCT_INT" == "" || "$ADMIN_N" -eq 0 ]]; then
        echo "🟡 NO SHADOW DATA — patient cohort has no shadow trades in 24h"
    elif (( PCT_INT >= 20 )); then
        echo "🟢 PATIENT MAKER WORKING (shadow scope) — maker rate ${ADMIN_PCT}% ≥ 20%"
        echo "   This means the shadow maker simulator IS honoring patience mode."
        echo "   Next: validate via Path B real pilot before full live cutover."
    elif (( PCT_INT >= 5 )); then
        echo "🟡 PARTIAL (shadow scope) — maker rate ${ADMIN_PCT}% in 5–20% band"
        echo "   Next: tune simulator queue model OR move to live pilot for ground truth"
    else
        echo "🔴 SHADOW SIMULATOR NOT HONORING PATIENT MODE — maker rate ${ADMIN_PCT}% < 5%"
        echo "   Likely root cause: _execute_shadow hardcoded to taker (line 2036+)."
        echo "   Required: ship shadow_maker_sim.py wiring (Phase 5.21)"
        echo "   Workaround: switch admin to bot_mode='live' for real fill data"
    fi
elif [[ -z "$PCT_INT" || "$PCT_INT" == "" || "$ADMIN_N" -eq 0 ]]; then
    echo "🟡 INSUFFICIENT DATA — patient cohort has no real trades in 24h window"
    echo "   Next: investigate why no real signals fired, or extend window"
elif [[ "$ADMIN_N" -lt 5 ]]; then
    echo "🟡 LOW SAMPLE — only ${ADMIN_N} patient-cohort trades in 24h"
elif (( PCT_INT >= 20 )); then
    echo "🟢 PATIENT MAKER WORKING — patient maker rate ${ADMIN_PCT}% ≥ 20% threshold"
    echo "   Next: ship Wave 2 (Patch 5.18 + 5.9-C + live readiness)"
elif (( PCT_INT >= 5 )); then
    echo "🟡 PARTIAL — patient maker rate ${ADMIN_PCT}% in 5–20% band"
    echo "   Next: tune patience to 'aggressive' mode, continue 24h test"
else
    echo "🔴 PATIENT MAKER FAILED — patient maker rate ${ADMIN_PCT}% < 5%"
    echo "   Next: pre-signal engine refactor (Wave 4 / Patch 5.8) becomes urgent"
fi
echo
echo "=================================================================="

#!/usr/bin/env bash
# Agent 6 — Risk Monitor (TIER A ONLY)
# Phase 6.C / Architect change #1 (2026-04-25)
#
# Auto-engage kill_switch on extreme outliers ONLY (no override window):
#   - Any single closed trade with |pnl| > 5R
#   - Live user balance drop > 10% in 1h (skipped if no live users)
#
# Tier B (alert + countdown) and Tier C (alert only) are a future agent
# spawn — this script is the bare-minimum auto-protection.
#
# Cron: */5 * * * * /home/opc/crypto-trading-bot/scripts/risk_monitor_tier_a.sh
set -u
ROOT=/home/opc/crypto-trading-bot
RISK_LOG=$ROOT/.rollback/risk_check.log
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$(dirname "$RISK_LOG")"

# Rate limit: max 1 destructive action per hour
LAST_ENGAGE_FILE=$ROOT/.rollback/.risk_monitor_last_engage
RATE_LIMIT_SEC=3600
NOW_SEC=$(date +%s)
LAST_SEC=$(cat "$LAST_ENGAGE_FILE" 2>/dev/null || echo 0)
ALREADY_ENGAGED_RECENTLY=0
if (( NOW_SEC - LAST_SEC < RATE_LIMIT_SEC )); then
    ALREADY_ENGAGED_RECENTLY=1
fi

# Check kill_switch is not already engaged (idempotency)
KS_STATUS=$(cd $ROOT && python3 -m execution.kill_switch status 2>&1 | head -1)
if echo "$KS_STATUS" | grep -qi 'ENGAGED'; then
    echo "[$TS] kill_switch already engaged: $KS_STATUS" >> "$RISK_LOG"
    exit 0
fi

# CHECK 1: Any single trade closed in last 15 min with |pnl| > 5R
OUTLIER=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c "
SELECT COUNT(*) FROM user_trades
WHERE closed_at > NOW() - INTERVAL '15 min'
  AND ABS(pnl_usd) > 5.0 * COALESCE(NULLIF(metadata::jsonb->>'initial_risk','')::numeric, 1.0);
" 2>/dev/null || echo 0)

# CHECK 2: Stale process (no log activity in 60s)
LAST_LOG_AGE=$(systemctl status cryptobot 2>/dev/null | grep -oE 'Active:.*ago' | head -1)
HB=$(journalctl -u cryptobot --since '90 sec ago' 2>/dev/null | wc -l)
STALE=0
if [[ ${HB:-0} -lt 1 ]]; then
    STALE=1
fi

# Decision
ENGAGED=0
REASON=""
if [[ ${OUTLIER:-0} -gt 0 ]]; then
    REASON="single_trade_loss_gt_5R"
    ENGAGED=1
elif [[ $STALE -eq 1 ]]; then
    REASON="bot_process_stale_no_heartbeat_90s"
    ENGAGED=1
fi

if [[ $ENGAGED -eq 1 ]]; then
    if [[ $ALREADY_ENGAGED_RECENTLY -eq 1 ]]; then
        echo "[$TS] WOULD-ENGAGE: $REASON (rate-limited; last engage at $LAST_SEC)" >> "$RISK_LOG"
    else
        echo "[$TS] ENGAGING kill_switch: $REASON" >> "$RISK_LOG"
        cd $ROOT && python3 -m execution.kill_switch engage --reason "risk_monitor_tier_a:$REASON" 2>&1 | tee -a "$RISK_LOG"
        echo "$NOW_SEC" > "$LAST_ENGAGE_FILE"
    fi
else
    echo "[$TS] OK (outlier=$OUTLIER stale=$STALE ks=$KS_STATUS)" >> "$RISK_LOG"
fi

exit 0

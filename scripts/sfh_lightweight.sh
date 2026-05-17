#!/usr/bin/env bash
# Agent 5 — Silent Failure Hunter (LIGHTWEIGHT)
# Phase 6.C / Architect change #2 (2026-04-25)
#
# Hourly cron — catches the most common silent failures without LLM cost.
# Heavy 6-scan LLM version runs every 6h (separate cron).
#
# Cron: 5 * * * * /home/opc/crypto-trading-bot/scripts/sfh_lightweight.sh
set -u
ROOT=/home/opc/crypto-trading-bot
ALERT_FILE=$ROOT/storage/sfh_alerts.log
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ALERTS=()

# 1. signal_data populated check (today's known bug — should be fixed but verify)
COUNT_EMPTY=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c \
  "SELECT COUNT(*) FROM user_trades WHERE signal_data='{}' AND opened_at > NOW() - INTERVAL '1 hour';" 2>/dev/null || echo 0)
if [[ ${COUNT_EMPTY:-0} -gt 0 ]]; then
    ALERTS+=("RED: $COUNT_EMPTY trades opened in last 1h have empty signal_data")
fi

# 2. Import errors in journal (cron python path issues etc.)
JOURNAL_ERRORS=$(journalctl --since '1 hour ago' 2>/dev/null | grep -ciE 'modulenotfounderror|importerror' || echo 0)
if [[ ${JOURNAL_ERRORS:-0} -gt 0 ]]; then
    ALERTS+=("RED: $JOURNAL_ERRORS import errors in journal last 1h")
fi

# 3. Required metadata keys present on recent shadow trades
KEY_GAPS=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c \
  "SELECT COUNT(*) FROM user_trades
   WHERE trade_type='shadow' AND opened_at > NOW() - INTERVAL '1 hour'
     AND NOT (metadata::jsonb ? 'fee_drag_r' AND metadata::jsonb ? 'grade'
              AND metadata::jsonb ? 'regime' AND metadata::jsonb ? 'margin');" 2>/dev/null || echo 0)
if [[ ${KEY_GAPS:-0} -gt 0 ]]; then
    ALERTS+=("YELLOW: $KEY_GAPS shadow trades missing core metadata keys")
fi

# 4. paper_signal_id wired (Phase 6.C Lever 3 Fix #3 verification)
NO_PAPER_ID=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c \
  "SELECT COUNT(*) FROM user_trades
   WHERE trade_type='shadow' AND opened_at > NOW() - INTERVAL '1 hour'
     AND COALESCE(NULLIF(signal_data::jsonb->>'paper_signal_id',''),'') = '';" 2>/dev/null || echo 0)
if [[ ${NO_PAPER_ID:-0} -gt 0 ]]; then
    ALERTS+=("YELLOW: $NO_PAPER_ID shadow trades missing paper_signal_id (mark alignment dead for these)")
fi

# 5. Bot heartbeat present (last 5 min log activity)
HB_COUNT=$(journalctl -u cryptobot --since '5 min ago' 2>/dev/null | wc -l)
if [[ ${HB_COUNT:-0} -lt 3 ]]; then
    ALERTS+=("RED: bot log activity suspiciously low in last 5 min ($HB_COUNT lines)")
fi

# 6. Auto-revert and watchdog crons fired in window
RECON_RAN=$(journalctl --since '1 hour ago' 2>/dev/null | grep -c 'auto_revert' || echo 0)
if [[ ${RECON_RAN:-0} -eq 0 ]]; then
    ALERTS+=("YELLOW: auto_revert cron has not run in last 1h")
fi

# Write alerts (only if any)
if [[ ${#ALERTS[@]} -gt 0 ]]; then
    {
        echo "[$TS] SFH LIGHTWEIGHT — ${#ALERTS[@]} alert(s):"
        for a in "${ALERTS[@]}"; do echo "  - $a"; done
        echo ""
    } >> "$ALERT_FILE"
fi

exit 0

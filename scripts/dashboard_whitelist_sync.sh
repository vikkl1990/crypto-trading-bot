#!/usr/bin/env bash
# scripts/dashboard_whitelist_sync.sh
# Phase 6.D security (2026-04-26)
#
# Cron wrapper that calls dashboard_whitelist.py to sync DB → iptables every 5 min.
# Idempotent — only acts on rules with comment 'dashboard-whitelist'.
#
# Cron: */5 * * * * /home/opc/crypto-trading-bot/scripts/dashboard_whitelist_sync.sh

set -u
ROOT=/home/opc/crypto-trading-bot
LOG=$ROOT/storage/dashboard_whitelist.log
PYTHON=/home/opc/miniconda3/bin/python3.13

mkdir -p "$(dirname $LOG)"

cd $ROOT
$PYTHON -m scripts.dashboard_whitelist apply >> $LOG 2>&1

# Trim log if >5MB
if [ -f "$LOG" ] && [ "$(stat -c %s $LOG)" -gt 5242880 ]; then
    tail -n 1000 $LOG > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
fi

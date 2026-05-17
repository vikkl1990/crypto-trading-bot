#!/bin/bash
# Pull engine state files + backtest reports from VM1 (READ-ONLY)
# Runs every minute via cron. Errors are silent so cron doesn't spam mail.

set -u
VM1="opc@150.230.171.48"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
DEST="/home/opc/engines_dashboard/data"
LOG="/home/opc/engines_dashboard/logs/sync.log"

mkdir -p "$DEST"

# Engine paper state dirs (live forward state)
for dir in s5_paper smc1_paper smc15_paper smc15v2_paper; do
    rsync -az --delete --timeout=30 \
        -e "ssh $SSH_OPTS" \
        "$VM1:/home/opc/crypto-trading-bot/storage/$dir/" \
        "$DEST/$dir/" 2>>"$LOG" || true
done

# Backtest report dirs (slower-changing reference docs)
for dir in smc15 smc15_variants scalp_research scanner_refinements ema200_data macd_div post_exit; do
    rsync -az --delete --timeout=30 \
        -e "ssh $SSH_OPTS" \
        "$VM1:/home/opc/crypto-trading-bot/storage/$dir/" \
        "$DEST/$dir/" 2>>"$LOG" || true
done

# Stamp last sync time
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$DEST/.last_sync"

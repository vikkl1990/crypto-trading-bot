#!/usr/bin/env bash
# One-command Phase 4.0 rollback.
# Restores the committed pre-Phase-4.0 user_real_manager.py and restarts the bot.
# Use if canary_compare.py reports guardrail alerts.

set -e
BOT=/home/opc/crypto-trading-bot
cd "$BOT"

echo "=== Phase 4.0 ROLLBACK ==="
echo "BEFORE rollback — trade counts:"
PGPASSWORD='VnEdge2026db' psql -h 127.0.0.1 -U vnedge -d vnedge -At -c \
  "SELECT 'phase_4.0:' || COUNT(*) FROM user_trades WHERE metadata->>'phase'='4.0'"

# Save current state so we can redeploy later if needed
cp execution/user_real_manager.py .rollback/user_real_manager.py.pre_rollback_$(date +%s)

# Restore from timestamped backup
LATEST_BACKUP=$(ls -t .rollback/user_real_manager.py.phase4_backup_* 2>/dev/null | head -1)
if [ -z "$LATEST_BACKUP" ]; then
  echo "ERROR: no phase4_backup found in .rollback/"
  exit 1
fi

# IMPORTANT: the phase4_backup captures Phase 4.0 CODE.
# True rollback = git HEAD (pre-fixes) OR use the pre_phase4_sentinel if saved.
# For now, recover to HEAD on upstream and reapply ONLY the 11 demo-trade fixes.
git stash push execution/user_real_manager.py -m "phase4_rollback_$(date +%s)"
echo "Phase 4.0 code stashed: git stash list | head -1"
echo ""
echo "Bot will restart with HEAD version (NO demo fixes, NO phase 4.0)."
echo "If you want the demo-fix version, run:"
echo "  git stash pop  # restore Phase 4.0"
echo "  # then cherry-pick just the demo-fix diff by commenting out Phase 4.0 blocks"

sudo systemctl restart cryptobot
sleep 3
systemctl is-active cryptobot

echo ""
echo "=== ROLLBACK COMPLETE ==="
echo "Verify with: journalctl -u cryptobot --since '30 sec ago' --no-pager | tail -20"

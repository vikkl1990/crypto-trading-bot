#!/bin/bash
# VN Edge: Trigger ML model retrain on VM4
# Cron: 0 4 * * 1 (Monday 4am UTC = weekly)
#
# 2026-05-02 FIX: previous version called `python3 ml_training/run_trainer.py --auto`
# which failed silently for 2+ weeks ("--auto" not a valid flag). Replaced with
# direct batch invocation of live_outcome_trainer.py which has a proper __main__
# entry, exits cleanly, and writes live_model_results.json + the .joblib model.
# This is the TRUE retrain path for the live-outcome shared model.
#
# Per-scanner models (candidate_*.json) are trained by a separate process
# scheduled internally on VM4 — those have been updating daily and don't need
# this trigger. We only fix the live_outcome path here.

set -e
VM4_HOST="opc@10.0.2.4"
SSH_KEY="$HOME/.ssh/cryptobot_oci"
TIMEOUT_SEC=600

echo "[$(date)] Triggering live_outcome ML retrain on VM4..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$VM4_HOST" \
    "cd /home/opc/crypto-trading-bot && timeout ${TIMEOUT_SEC} python3 -m ml_training.live_outcome_trainer 2>&1" | tail -40

# After training, sync new models to VM1
sleep 5
$(dirname "$0")/sync_ml_models.sh
echo "[$(date)] Retrain + sync complete"

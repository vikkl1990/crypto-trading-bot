#!/usr/bin/env bash
# DAY-2 PATCH ORCHESTRATOR — applies all three held patches in sequence with
# safety checks, syntax verification, restart, and 30-min sanity watch.
#
# Usage:
#   ./apply_day2_patches.sh                     # DRY-RUN (default)
#   ./apply_day2_patches.sh --apply              # actually apply + restart
#   ./apply_day2_patches.sh --rollback <ts>      # restore .bak files

set -uo pipefail

ROOT=/home/opc/crypto-trading-bot
TS=$(date -u +%s)
BAK_TAG="day2_${TS}"
MODE_FLAG="${1:-dry-run}"
LOG=/tmp/day2_apply_${TS}.log

# Convert flag to internal mode for python patcher
case "$MODE_FLAG" in
    --apply)    DAY2_MODE="apply" ;;
    --rollback) DAY2_MODE="rollback" ;;
    *)          DAY2_MODE="dry-run" ;;
esac
export DAY2_MODE

export DAY2_F_STRAT=$ROOT/strategies/scalp_strategy.py
export DAY2_F_URM=$ROOT/execution/user_real_manager.py
export DAY2_F_EXIT=$ROOT/execution/exit_guards.py
export DAY2_F_REGIME=$ROOT/strategies/regime_filter.py
F_ENV=$ROOT/.env

PATCHER=$ROOT/scripts/day2_patcher.py

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

abort() {
    log "❌ ABORT: $*"
    log "Backups (if any) preserved with .bak.${BAK_TAG} suffix — restore with --rollback ${TS}"
    exit 1
}

# ─────────────────────────────────────────────────────────────────────
preflight() {
    log "=== PRE-FLIGHT CHECKS ==="

    local today=$(date -u +%Y%m%d)
    local verdicts=$(ls $ROOT/storage/verdicts/maker_ab_${today}*.txt 2>/dev/null | wc -l)
    if [ "$verdicts" -eq 0 ]; then
        log "⚠ WARN: no verdict file for today"
    else
        log "✓ Verdict file present: $(ls $ROOT/storage/verdicts/maker_ab_${today}*.txt | tail -1)"
    fi

    local pid=$(pgrep -f "python.*main.py" | head -1)
    if [ -z "$pid" ]; then
        abort "bot is not running"
    fi
    log "✓ Bot running, PID=$pid"

    local open_n=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -A -F '|' -t -c \
        "SELECT COUNT(*) FROM user_trades WHERE status='open' AND trade_type='shadow';" | tr -d ' |')
    log "✓ Open shadow trades: $open_n (will reconcile after restart)"

    log "Pre-flight OK"
    echo
}

# ─────────────────────────────────────────────────────────────────────
backup_files() {
    if [ "$DAY2_MODE" != "apply" ]; then return; fi
    log "Backing up files with tag .bak.${BAK_TAG}"
    cp "$DAY2_F_STRAT"  "$DAY2_F_STRAT.bak.${BAK_TAG}"
    cp "$DAY2_F_URM"    "$DAY2_F_URM.bak.${BAK_TAG}"
    cp "$DAY2_F_EXIT"   "$DAY2_F_EXIT.bak.${BAK_TAG}"
    cp "$DAY2_F_REGIME" "$DAY2_F_REGIME.bak.${BAK_TAG}"
    [ -f "$F_ENV" ] && cp "$F_ENV" "$F_ENV.bak.${BAK_TAG}"
}

# ─────────────────────────────────────────────────────────────────────
run_patch() {
    local n=$1; local label=$2
    log "=== PATCH $n: $label ==="
    DAY2_PATCH=$n python3 "$PATCHER"
    if [ "$DAY2_MODE" == "apply" ]; then
        case $n in
            1) python3 -m py_compile "$DAY2_F_STRAT" || abort "P$n syntax error" ;;
            2) python3 -m py_compile "$DAY2_F_URM" && python3 -m py_compile "$DAY2_F_EXIT" \
                 || abort "P$n syntax error" ;;
            3) python3 -m py_compile "$DAY2_F_URM" || abort "P$n syntax error" ;;
            4) python3 -m py_compile "$DAY2_F_REGIME" || abort "P$n syntax error" ;;
            5) python3 -m py_compile "$DAY2_F_STRAT" || abort "P$n syntax error" ;;
            6) python3 -m py_compile "$DAY2_F_STRAT" || abort "P$n syntax error" ;;
        esac
    fi
    log "✓ Patch $n done"
    echo
}

# ─────────────────────────────────────────────────────────────────────
update_env() {
    if [ "$DAY2_MODE" != "apply" ]; then
        log "DRY-RUN: would set SHADOW_MAKER_SIM_ENABLED=true in $F_ENV"
        return
    fi
    if grep -q "^SHADOW_MAKER_SIM_ENABLED=" "$F_ENV" 2>/dev/null; then
        sed -i 's/^SHADOW_MAKER_SIM_ENABLED=.*/SHADOW_MAKER_SIM_ENABLED=true  # MAKER_SIM_WIRING_5_21/' "$F_ENV"
        log "✓ .env: existing flag set to true"
    else
        echo "SHADOW_MAKER_SIM_ENABLED=true  # MAKER_SIM_WIRING_5_21" >> "$F_ENV"
        log "✓ .env: SHADOW_MAKER_SIM_ENABLED=true added"
    fi
}

# ─────────────────────────────────────────────────────────────────────
restart_bot() {
    log "=== RESTART BOT ==="
    if [ "$DAY2_MODE" != "apply" ]; then
        log "DRY-RUN: would kill bot and start new"
        return
    fi

    # FIX 2026-05-01: trust the bot's own pidfile, not pgrep race.
    # Use the actual python.*main.py match (with leading slash to filter bash wrappers).
    local old_pid=""
    if [ -f "$ROOT/.bot.pid" ]; then
        old_pid=$(cat "$ROOT/.bot.pid" 2>/dev/null | tr -d '[:space:]')
        log "Bot pidfile says PID=$old_pid"
    fi
    if [ -z "$old_pid" ] || ! kill -0 "$old_pid" 2>/dev/null; then
        # Pidfile missing or stale; fall back to pgrep with strict pattern
        old_pid=$(pgrep -f "/python.*main\.py" | head -1)
        log "Pidfile invalid; pgrep found PID=$old_pid"
    fi

    if [ -z "$old_pid" ]; then
        abort "couldn't identify bot PID for restart"
    fi

    log "Killing PID $old_pid (graceful TERM, then KILL after 8s)"
    kill "$old_pid" 2>/dev/null
    for i in 1 2 3 4 5 6 7 8; do
        sleep 1
        if ! kill -0 "$old_pid" 2>/dev/null; then break; fi
    done
    if kill -0 "$old_pid" 2>/dev/null; then
        log "Graceful kill failed; SIGKILL"
        kill -9 "$old_pid" 2>/dev/null
        sleep 2
    fi
    rm -f "$ROOT/.bot.pid"
    log "Pidfile cleared"

    cd "$ROOT" && nohup /home/opc/miniconda3/bin/python3.13 main.py > "/tmp/bot_${TS}.log" 2>&1 < /dev/null & disown
    sleep 12

    # Verify fresh PID and log shows clean startup
    local new_pid=$(pgrep -f "/python.*main\.py" | head -1)
    if [ -z "$new_pid" ] || [ "$new_pid" = "$old_pid" ]; then
        abort "bot restart failed — new_pid=$new_pid old_pid=$old_pid"
    fi
    if grep -qE "ERROR: Bot already running" "/tmp/bot_${TS}.log" 2>/dev/null; then
        abort "bot startup blocked by stale pidfile"
    fi
    log "✓ Bot restarted: old=$old_pid → new=$new_pid  Log: /tmp/bot_${TS}.log"
}

# ─────────────────────────────────────────────────────────────────────
sanity_watch() {
    log "=== 30-MIN SANITY WATCH ==="
    if [ "$DAY2_MODE" != "apply" ]; then
        log "DRY-RUN: would watch for 30 min"
        return
    fi
    local start_iso=$(date -u +"%Y-%m-%d %H:%M:%S+00:00")
    log "Starting watch at $start_iso — reports every 5 min × 6"

    for i in 1 2 3 4 5 6; do
        sleep 300
        log "--- T+$((i*5))min ---"
        PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -A -F '|' -t -c \
            "SELECT u.email, COALESCE(NULLIF(ut.metadata::jsonb->>'fee_type',''),'?') as ft, COUNT(*),
                    ROUND(SUM(ut.pnl_usd) FILTER (WHERE ut.status='closed')::numeric, 2) as net
             FROM user_trades ut JOIN users u ON u.id=ut.user_id
             WHERE u.email IN ('admin@vnedge.com','niranjan_139@yahoo.co.in')
               AND ut.trade_type='shadow' AND ut.opened_at >= '$start_iso'
             GROUP BY u.email, ft ORDER BY u.email, ft;" 2>&1 | tee -a "$LOG"
    done

    log "=== POST-WATCH GUIDANCE ==="
    log "Expected: admin maker% > niranjan maker% (patient mode differential)"
    log "Expected: structure_bounce trade count drops ~30% (volume gate)"
    log "Expected: peak_floor_stall + time_decay_5m exits appear in mix"
    log "If admin == niranjan after 30 min: check SHADOW_MAKER_SIM_ENABLED in .env"
    log "If trade count drops >50%: volume gate too aggressive — rollback"
}

# ─────────────────────────────────────────────────────────────────────
rollback() {
    local rb_ts="${2:-}"
    if [ -z "$rb_ts" ]; then
        log "Available backups:"
        ls -1 $ROOT/strategies/*.bak.day2_* $ROOT/execution/*.bak.day2_* 2>/dev/null | head -10
        abort "specify timestamp: --rollback <ts>"
    fi
    log "=== ROLLBACK to day2_${rb_ts} ==="
    for f in $DAY2_F_STRAT $DAY2_F_URM $DAY2_F_EXIT $DAY2_F_REGIME; do
        if [ -f "${f}.bak.day2_${rb_ts}" ]; then
            cp "${f}.bak.day2_${rb_ts}" "$f"
            log "✓ Restored $f"
        fi
    done
    if [ -f "$F_ENV.bak.day2_${rb_ts}" ]; then
        cp "$F_ENV.bak.day2_${rb_ts}" "$F_ENV"
        log "✓ Restored .env"
    fi
    log "Restart bot to load rolled-back code"
}

# ─────────────────────────────────────────────────────────────────────
log "=========================================="
log "  DAY-2 PATCH ORCHESTRATOR"
log "  Mode: $MODE_FLAG"
log "  Timestamp: $TS"
log "=========================================="
echo

case "$MODE_FLAG" in
    --rollback)
        rollback "$@"
        ;;
    --apply|dry-run)
        preflight
        backup_files
        log "── W/F-validated patches (require backtest evidence to ship) ──"
        run_patch 1 "Wyckoff volume gate"
        run_patch 2 "Threshold tweaks (max_age/stall/relaxed)"
        run_patch 3 "Maker-sim wiring"
        log "── Subtraction-only patches (no W/F needed — disable known losers) ──"
        run_patch 4 "A+ size multiplier 1.3 → 1.0"
        run_patch 5 "high_volatility hard-veto for structure_bounce"
        run_patch 6 "Asia-early veto extension (UTC 4-7)"
        update_env
        if [ "$DAY2_MODE" == "apply" ]; then
            restart_bot
            sanity_watch
        fi
        log "=== DONE ==="
        log "Log: $LOG"
        if [ "$DAY2_MODE" != "apply" ]; then
            log "DRY-RUN complete — re-run with --apply to actually execute"
        fi
        ;;
    *)
        echo "Usage: $0 [--apply | --rollback <ts>]"
        echo "       (no args = dry-run)"
        exit 1
        ;;
esac

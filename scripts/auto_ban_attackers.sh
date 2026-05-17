#!/usr/bin/env bash
# scripts/auto_ban_attackers.sh
# Phase 6.C security (2026-04-25)
#
# Scans journalctl for attack patterns + auto-bans repeat offenders via iptables.
# Lightweight fail2ban replacement (no daemon, just cron + grep).
#
# Cron: */10 * * * * /home/opc/crypto-trading-bot/scripts/auto_ban_attackers.sh

set -u
ROOT=/home/opc/crypto-trading-bot
STATE=$ROOT/storage/auto_ban_state
LOG=$ROOT/storage/auto_ban.log
WINDOW_MIN=60          # look at last 60 min of journal
THRESHOLD=3            # ban after 3 hits in window
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Allowlist — never ban these (architect IP + loopback + private nets)
ALLOWLIST_REGEX='^(171\.76\.82\.182|127\.0\.0\.1|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)'

mkdir -p "$STATE" "$(dirname "$LOG")"

# ── 1. Find attacker IPs in last $WINDOW_MIN minutes ────────────────
# Patterns we care about:
#   - "Error handling request from <IP>" (aiohttp malformed requests)
#   - "BadHttpMessage" (raw bad HTTP)
#   - Future: failed auth, rate-limit triggers
ATTACKERS=$(sudo journalctl -u cryptobot --since "$WINDOW_MIN min ago" --no-pager 2>/dev/null \
    | grep -oE 'from [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' \
    | awk '{print $2}' \
    | grep -vE "$ALLOWLIST_REGEX" \
    | sort | uniq -c | sort -rn \
    | awk -v t=$THRESHOLD '$1 >= t {print $2}')

if [[ -z "$ATTACKERS" ]]; then
    # No attackers above threshold — silent exit (no log spam)
    exit 0
fi

# ── 2. Check current iptables banned set; ban only new IPs ──────────
BANNED_FILE=$STATE/banned_ips.txt
touch "$BANNED_FILE"

NEW_BANS=()
for ip in $ATTACKERS; do
    if ! grep -qFx "$ip" "$BANNED_FILE"; then
        # Verify not already in iptables (defense in depth)
        if ! sudo iptables -L INPUT -n | grep -q "DROP.*$ip "; then
            sudo iptables -I INPUT 1 -s "$ip" -m comment --comment "auto-ban-$TS" -j DROP
            echo "$ip" >> "$BANNED_FILE"
            NEW_BANS+=("$ip")
        fi
    fi
done

# ── 3. If new bans applied, persist iptables + log ──────────────────
if [[ ${#NEW_BANS[@]} -gt 0 ]]; then
    sudo iptables-save | sudo tee /etc/sysconfig/iptables > /dev/null 2>&1
    {
        echo "[$TS] AUTO_BAN — ${#NEW_BANS[@]} new IP(s) banned (window=${WINDOW_MIN}min, threshold=${THRESHOLD}):"
        for ip in "${NEW_BANS[@]}"; do
            hits=$(sudo journalctl -u cryptobot --since "$WINDOW_MIN min ago" --no-pager 2>/dev/null \
                | grep -c "from $ip")
            echo "  - $ip  ($hits hits in last ${WINDOW_MIN}min)"
        done
        echo ""
    } >> "$LOG"
fi

exit 0

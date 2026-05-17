#!/usr/bin/env bash
# Agent 8 — Data Integrity Engineer (LIGHTWEIGHT DAILY)
# Phase 6.C (2026-04-25)
#
# Daily 02:00 UTC scan of last 24h trades for missing required metadata.
# Generates a markdown report; flags trigger heavy LLM agent for backfill.
#
# Cron: 0 2 * * * /home/opc/crypto-trading-bot/scripts/data_integrity_daily.sh
set -u
ROOT=/home/opc/crypto-trading-bot
TS=$(date -u +%Y%m%d)
REPORT=$ROOT/.rollback/data_integrity_${TS}.md

mkdir -p "$(dirname "$REPORT")"

cat > "$REPORT" <<EOF
# Data Integrity Report — $TS

Run: $(date -u)
Agent: 8 / Data Integrity Engineer (lightweight bash)

## Required Field Coverage (last 24h shadow + real)
EOF

REQUIRED_KEYS="grade regime ml_prob fee_type margin leverage stop_loss take_profit tick_size contract_size product_id initial_risk"

for key in $REQUIRED_KEYS; do
    cov=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c "
    SELECT
        COUNT(*) FILTER (WHERE metadata::jsonb ? '$key' AND COALESCE(NULLIF(metadata::jsonb->>'$key',''),'') != '') ||
        '/' ||
        COUNT(*) ||
        ' (' ||
        ROUND(100.0 * COUNT(*) FILTER (WHERE metadata::jsonb ? '$key' AND COALESCE(NULLIF(metadata::jsonb->>'$key',''),'') != '') / NULLIF(COUNT(*),0), 1) ||
        '%)'
    FROM user_trades
    WHERE trade_type IN ('shadow','real') AND opened_at > NOW() - INTERVAL '24 hours';
    " 2>/dev/null)
    echo "- $key: $cov" >> "$REPORT"
done

cat >> "$REPORT" <<'EOF'

## signal_data populated check (Phase 6.C signal_data fix verification)
EOF

EMPTY_SD=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c "
SELECT COUNT(*) FROM user_trades
WHERE trade_type IN ('shadow','real') AND opened_at > NOW() - INTERVAL '24 hours'
  AND signal_data::text = '{}';" 2>/dev/null || echo 0)
TOTAL_SD=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c "
SELECT COUNT(*) FROM user_trades
WHERE trade_type IN ('shadow','real') AND opened_at > NOW() - INTERVAL '24 hours';" 2>/dev/null || echo 0)
echo "- empty signal_data: $EMPTY_SD / $TOTAL_SD" >> "$REPORT"

cat >> "$REPORT" <<'EOF'

## paper_signal_id linkage (Phase 6.C Lever 3 Fix #3 verification)
EOF

NO_PAPER_ID=$(PGPASSWORD=VnEdge2026db psql -h localhost -U vnedge -d vnedge -At -c "
SELECT COUNT(*) FROM user_trades
WHERE trade_type='shadow' AND opened_at > NOW() - INTERVAL '24 hours'
  AND COALESCE(NULLIF(signal_data::jsonb->>'paper_signal_id',''),'') = '';" 2>/dev/null || echo 0)
echo "- shadow trades missing paper_signal_id: $NO_PAPER_ID" >> "$REPORT"

cat >> "$REPORT" <<'EOF'

## Recommendations
EOF

# Decide if a heavy LLM agent should be spawned for backfill
if [[ ${EMPTY_SD:-0} -gt 0 ]]; then
    echo "- 🔴 Spawn Data Integrity Engineer (heavy) to backfill empty signal_data" >> "$REPORT"
fi
if [[ ${NO_PAPER_ID:-0} -gt 5 ]]; then
    echo "- 🔴 Spawn investigator to trace why paper_signal_id is missing" >> "$REPORT"
fi
if [[ ${EMPTY_SD:-0} -eq 0 && ${NO_PAPER_ID:-0} -eq 0 ]]; then
    echo "- 🟢 No findings; metadata coverage healthy" >> "$REPORT"
fi

echo "" >> "$REPORT"
echo "Report saved: $REPORT" >> "$REPORT"

exit 0

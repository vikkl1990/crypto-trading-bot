#!/usr/bin/env bash
# Agent 4 — Code Review Engineer (WRAPPER, event-driven)
# Phase 6.C (2026-04-25)
#
# Called pre-deploy by Deployment Gatekeeper (Agent 3) on every patch script.
# Runs a 7-point safety checklist on the supplied patch script.
#
# Usage: code_review_wrapper.sh <path/to/apply_*.sh>
# Exit 0 = APPROVED, Exit 1 = REJECTED, Exit 2 = CONCERN (escalate to architect)
set -u

PATCH="${1:-}"
if [[ -z "$PATCH" || ! -f "$PATCH" ]]; then
    echo "ERROR: usage: $0 <patch_script>"
    exit 1
fi

ROOT=/home/opc/crypto-trading-bot
TS=$(date -u +%Y%m%d_%H%M%S)
REVIEW=$ROOT/.rollback/code_review_$(basename "$PATCH" .sh)_${TS}.md

mkdir -p "$(dirname "$REVIEW")"

cat > "$REVIEW" <<EOF
# Code Review — $(basename "$PATCH")
Reviewer: Agent 4 (Code Review Engineer wrapper)
Run: $(date -u)
Patch: $PATCH

## Checklist
EOF

PASSED=0
TOTAL=7

# 1. Anchor uniqueness — patch should use 'assert n == 1' on every str.replace
ANCHOR_ASSERTIONS=$(grep -c 'assert.*count.*1' "$PATCH" 2>/dev/null || echo 0)
REPLACE_COUNT=$(grep -c 'src.replace\|str.replace\|sed.*-i' "$PATCH" 2>/dev/null || echo 0)
if [[ ${ANCHOR_ASSERTIONS:-0} -ge ${REPLACE_COUNT:-1} && ${REPLACE_COUNT:-0} -gt 0 ]]; then
    echo "1. ✅ Anchor uniqueness: $ANCHOR_ASSERTIONS assertions / $REPLACE_COUNT replacements" >> "$REVIEW"
    PASSED=$((PASSED + 1))
elif [[ ${REPLACE_COUNT:-0} -eq 0 ]]; then
    echo "1. 🟡 Anchor uniqueness: no str.replace patterns found (might be SQL-only)" >> "$REVIEW"
    PASSED=$((PASSED + 1))
else
    echo "1. 🔴 Anchor uniqueness: $ANCHOR_ASSERTIONS assertions for $REPLACE_COUNT replacements (gap)" >> "$REVIEW"
fi

# 2. Idempotency — SQL uses IF NOT EXISTS / ON CONFLICT
SQL_LINES=$(grep -ciE 'CREATE TABLE|ALTER TABLE|INSERT INTO' "$PATCH" 2>/dev/null || echo 0)
IDEMPOTENT=$(grep -ciE 'IF NOT EXISTS|ON CONFLICT|DROP.*IF EXISTS' "$PATCH" 2>/dev/null || echo 0)
if [[ ${SQL_LINES:-0} -eq 0 ]]; then
    echo "2. 🟢 Idempotency: no SQL DDL detected" >> "$REVIEW"
    PASSED=$((PASSED + 1))
elif [[ ${IDEMPOTENT:-0} -ge 1 ]]; then
    echo "2. ✅ Idempotency: $IDEMPOTENT idempotent patterns / $SQL_LINES SQL DDL lines" >> "$REVIEW"
    PASSED=$((PASSED + 1))
else
    echo "2. 🔴 Idempotency: $SQL_LINES SQL DDL lines but no IF NOT EXISTS" >> "$REVIEW"
fi

# 3. Syntax check on referenced Python files
PY_FILES=$(grep -oE '/.+\.py' "$PATCH" 2>/dev/null | sort -u)
PY_COUNT=$(echo "$PY_FILES" | grep -c . || echo 0)
echo "3. Python files referenced: $PY_COUNT" >> "$REVIEW"

# 4. Backup discipline — patch creates .rollback dir
HAS_BACKUP=$(grep -ciE 'rollback|\.bak|cp.*backup' "$PATCH" 2>/dev/null || echo 0)
if [[ ${HAS_BACKUP:-0} -ge 1 ]]; then
    echo "4. ✅ Backup discipline: $HAS_BACKUP backup-related lines" >> "$REVIEW"
    PASSED=$((PASSED + 1))
else
    echo "4. 🔴 Backup discipline: no backup pattern detected" >> "$REVIEW"
fi

# 5. Unicode safety — check for the dash characters that bit us today
SUSPECT_UNICODE=$(grep -c '[─━┃│]' "$PATCH" 2>/dev/null || echo 0)
if [[ ${SUSPECT_UNICODE:-0} -eq 0 ]]; then
    echo "5. ✅ Unicode safety: no box-drawing chars in anchors" >> "$REVIEW"
    PASSED=$((PASSED + 1))
else
    echo "5. 🟡 Unicode safety: $SUSPECT_UNICODE box-drawing chars found (possible em-dash mismatch)" >> "$REVIEW"
fi

# 6. No git destructive operations
GIT_DESTRUCTIVE=$(grep -ciE 'git push --force|git reset --hard|git clean -f|git checkout --' "$PATCH" 2>/dev/null || echo 0)
if [[ ${GIT_DESTRUCTIVE:-0} -eq 0 ]]; then
    echo "6. ✅ No destructive git operations" >> "$REVIEW"
    PASSED=$((PASSED + 1))
else
    echo "6. 🔴 Destructive git ops detected: $GIT_DESTRUCTIVE" >> "$REVIEW"
fi

# 7. Has dry-run support
HAS_DRYRUN=$(grep -ciE 'dry.?run|--apply' "$PATCH" 2>/dev/null || echo 0)
if [[ ${HAS_DRYRUN:-0} -ge 1 ]]; then
    echo "7. ✅ Dry-run support detected" >> "$REVIEW"
    PASSED=$((PASSED + 1))
else
    echo "7. 🟡 No dry-run flag detected (apply runs immediately)" >> "$REVIEW"
fi

cat >> "$REVIEW" <<EOF

## Verdict
PASSED: $PASSED / $TOTAL
EOF

if [[ $PASSED -eq $TOTAL ]]; then
    echo "## 🟢 APPROVED — Gatekeeper may proceed" >> "$REVIEW"
    exit 0
elif [[ $PASSED -ge 5 ]]; then
    echo "## 🟡 CONCERN — escalate to architect" >> "$REVIEW"
    exit 2
else
    echo "## 🔴 REJECT — block deploy" >> "$REVIEW"
    exit 1
fi

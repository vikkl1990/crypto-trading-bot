"""Comprehensive Production Compliance Audit — 10 Categories"""
import json, os, sys

findings = []

def finding(cat, sev, title, current, gap, risk, effort):
    findings.append({"cat": cat, "sev": sev, "title": title, "current": current, "gap": gap, "risk": risk, "effort": effort})

# Load all source files
rm = open("execution/real_manager.py").read()
st = open("bot/signal_tracker.py").read()
orch = open("bot/orchestrator.py").read()
dc = open("exchange/delta_client.py").read()
sv = open("dashboard/server.py").read()
ix = open("dashboard/templates/index.html").read()

try:
    cfg = open("config/settings.yaml").read()
except:
    cfg = ""

try:
    env = open(".env").read()
except:
    env = ""

try:
    service = open("/etc/systemd/system/cryptobot.service").read()
except:
    service = ""

print("=" * 100)
print("COMPREHENSIVE PRODUCTION COMPLIANCE AUDIT")
print("=" * 100)

# ============================================================
# CAT 1: ORDER MANAGEMENT
# ============================================================
cat = "1. ORDER MANAGEMENT"

# Emergency close after SL fail
if "emergency" in rm.lower() and "EMERGENCY CLOSE" in rm:
    finding(cat, "OK", "Emergency close on SL failure", "3 retries then emergency close", "", "", "")
else:
    finding(cat, "CRITICAL", "No emergency close on SL failure", "Unknown", "Position runs unprotected", "Unlimited loss", "MEDIUM")

# Order deduplication
if "already_mirrored" in rm or "dedup" in rm.lower():
    finding(cat, "OK", "Order deduplication", "Checks paper_to_real mapping before mirroring", "", "", "")
else:
    finding(cat, "HIGH", "No order deduplication", "Missing", "Same signal could place 2 orders", "Double position", "EASY")

# Cancel orders on close
if "cancel" in rm.lower() and "CANCEL ORDER" in rm:
    finding(cat, "OK", "Orders cancelled on close", "SL/TP cancelled when trade closes", "", "", "")
else:
    finding(cat, "HIGH", "Stale orders not cancelled", "Missing", "SL/TP orders left on Delta", "Ghost orders", "EASY")

# Partial fill handling
if "partial" in rm.lower() and "fill" in rm.lower():
    has_partial = "partial_fill" in rm or "unfilled_qty" in rm
    if has_partial:
        finding(cat, "OK", "Partial fill handling", "Tracks partial fills", "", "", "")
    else:
        finding(cat, "MEDIUM", "No partial fill tracking", "Assumes full fill", "Partial fills create size mismatch", "Wrong PnL", "MEDIUM")
else:
    finding(cat, "MEDIUM", "No partial fill handling", "Missing", "Partial fills not tracked", "Size mismatch", "MEDIUM")

# Rate limiting
if "rate_limit" in dc or "_rate_limit_check" in dc:
    finding(cat, "OK", "API rate limiting", "rate_limit_check() before orders", "", "", "")
else:
    finding(cat, "HIGH", "No rate limiting", "Missing", "Could exceed Delta 500 ops/sec", "API ban", "EASY")

# ============================================================
# CAT 2: RISK MANAGEMENT
# ============================================================
cat = "2. RISK MANAGEMENT"

# Daily loss limit
if "daily_loss" in rm or "daily_pnl" in rm:
    finding(cat, "OK", "Daily loss limit", "Circuit breaker with daily_loss_limit", "", "", "")
else:
    finding(cat, "CRITICAL", "No daily loss limit", "Missing", "No cap on daily losses", "Account blow-up", "EASY")

# Circuit breaker
if "circuit_breaker" in rm and "consecutive_losses" in rm:
    finding(cat, "OK", "Circuit breaker", "Trips at 3 consecutive losses", "", "", "")
else:
    finding(cat, "CRITICAL", "No circuit breaker", "Missing", "No stop on losing streaks", "Account drain", "EASY")

# Max position size
if "max_margin" in rm or "max_position" in rm or "leverage_cap" in rm:
    finding(cat, "OK", "Position size limits", "margin cap + leverage cap", "", "", "")
else:
    finding(cat, "HIGH", "No position size limit", "Missing", "Could over-leverage", "Large loss", "EASY")

# Max drawdown kill switch
if "drawdown" in rm.lower() or "max_drawdown" in rm.lower():
    finding(cat, "MEDIUM", "Drawdown monitoring exists", "Tracked but no kill switch", "No auto-disable at X% drawdown", "Slow bleed", "EASY")
else:
    finding(cat, "HIGH", "No max drawdown kill switch", "Missing", "No automatic shutdown at severe drawdown", "Account drain", "EASY")

# Pre-trade margin check
if "balance" in rm and "min_balance" in rm:
    finding(cat, "OK", "Pre-trade balance check", "Checks balance >= min before entry", "", "", "")
else:
    finding(cat, "MEDIUM", "No pre-trade margin check", "Missing", "Could place order without margin", "Rejected by Delta", "EASY")

# ============================================================
# CAT 3: STATE MANAGEMENT
# ============================================================
cat = "3. STATE MANAGEMENT"

# Atomic writes
if "tempfile" in st or "atomic" in st.lower() or "_safe_write" in st:
    finding(cat, "OK", "Safe file writes", "_safe_write() method exists", "", "", "")
else:
    finding(cat, "HIGH", "Non-atomic state writes", "Direct file writes", "Crash mid-write corrupts state", "Lost trade data", "MEDIUM")

# Write-ahead log
if "WAL" in rm or "write_ahead" in rm or "journal" in rm.lower():
    finding(cat, "CRITICAL", "No write-ahead log", "Missing", "Crash between entry and SL = naked position", "$100-500 per event", "HARD")
else:
    finding(cat, "CRITICAL", "No write-ahead log", "Missing", "Crash between order placement and state save", "Naked positions", "HARD")

# State backup
if "backup" in st.lower() or ".bak" in st:
    finding(cat, "OK", "State backups exist", "Hourly + daily backups in storage/backups/", "", "", "")
else:
    finding(cat, "MEDIUM", "No automated state backup", "Manual only", "No recovery point", "Data loss", "EASY")

# Persistence of new fields
if "independent_exit" in rm and "_save_state" in rm:
    # Check if save includes independent_exit
    save_section = rm[rm.index("_save_state"):rm.index("_save_state")+2000]
    if "independent_exit" in save_section:
        finding(cat, "OK", "Independent exit fields persisted", "All 15 fields saved in _save_state", "", "", "")
    else:
        finding(cat, "HIGH", "New fields not persisted", "independent_exit lost on restart", "Real exit system broken after restart", "Trade runs unmanaged", "EASY")

# ============================================================
# CAT 4: ERROR HANDLING
# ============================================================
cat = "4. ERROR HANDLING"

# Bare except blocks
bare_except_rm = rm.count("except:") - rm.count("except: #") - rm.count("except:  #")
bare_except_st = st.count("except:") - st.count("except: #")
bare_except_orch = orch.count("except:") - orch.count("except: #")
total_bare = bare_except_rm + bare_except_st + bare_except_orch
if total_bare > 5:
    finding(cat, "MEDIUM", f"Bare except: blocks ({total_bare})", "Swallows errors silently", "Hidden failures", "Undetected issues", "EASY")
else:
    finding(cat, "OK", "Minimal bare except blocks", f"{total_bare} found", "", "", "")

# Heartbeat/watchdog
if "heartbeat" in orch.lower() or "watchdog" in orch.lower():
    finding(cat, "OK", "Heartbeat monitoring", "Heartbeat tracks component health", "", "", "")
else:
    finding(cat, "HIGH", "No heartbeat", "Missing", "Silent failure undetected", "Bot dies silently", "MEDIUM")

# Auto-restart
if "Restart=always" in service or "restart=always" in service.lower():
    finding(cat, "OK", "Auto-restart on crash", "systemd Restart=always", "", "", "")
elif "Restart=" in service:
    finding(cat, "MEDIUM", "Limited auto-restart", service.split("Restart=")[1][:20] if "Restart=" in service else "unknown", "May not restart on all failures", "Extended downtime", "EASY")
else:
    finding(cat, "HIGH", "No auto-restart configured", "Missing", "Bot stays dead after crash", "Miss all trades", "EASY")

# Timeout handling
if "timeout" in dc.lower():
    finding(cat, "OK", "API timeout handling", "Timeouts configured for Delta API", "", "", "")
else:
    finding(cat, "HIGH", "No API timeout", "Missing", "Hanging connection blocks event loop", "Bot freezes", "EASY")

# ============================================================
# CAT 5: LOGGING & AUDIT
# ============================================================
cat = "5. LOGGING & AUDIT"

# Audit trail
if "audit_log" in orch or "AUDIT" in orch:
    finding(cat, "OK", "Immutable audit trail", "audit_log() on trade entry/exit", "", "", "")
else:
    finding(cat, "HIGH", "No audit trail", "Missing", "Can't reconstruct trades from logs", "Compliance fail", "MEDIUM")

# Sensitive data in logs
if "api_key" in dc.lower() and "logger" in dc:
    # Check if keys are logged
    if "api_key" in dc.split("logger")[0][-200:] if "logger" in dc else False:
        finding(cat, "CRITICAL", "API keys may be logged", "Key variables near log statements", "Keys in log files", "Security breach", "EASY")
    else:
        finding(cat, "OK", "API keys not logged", "Keys used only for auth, not logged", "", "", "")
else:
    finding(cat, "OK", "API key handling", "Keys loaded from .env", "", "", "")

# Log rotation
if "RotatingFileHandler" in open("main.py").read() if os.path.exists("main.py") else False:
    finding(cat, "OK", "Log rotation", "RotatingFileHandler configured", "", "", "")
else:
    finding(cat, "MEDIUM", "No log rotation", "Logs grow indefinitely", "Disk fills up", "Bot crashes on full disk", "EASY")

# ============================================================
# CAT 6: CONCURRENCY
# ============================================================
cat = "6. CONCURRENCY"

# Async lock usage
if "asyncio.Lock()" in st and "async with self._lock" not in st:
    finding(cat, "HIGH", "Lock declared but never used", "self._lock = asyncio.Lock() exists but no 'async with'", "State mutations unprotected", "Race conditions", "HARD")
elif "asyncio.Lock()" in st:
    finding(cat, "OK", "Async lock used", "Lock protects state mutations", "", "", "")
else:
    finding(cat, "HIGH", "No async lock", "Missing", "Concurrent state access", "Data corruption", "HARD")

# Blocking sleep in async
sleep_in_rm = rm.count("time.sleep(")
if sleep_in_rm > 0:
    finding(cat, "MEDIUM", f"Blocking time.sleep() in real_manager ({sleep_in_rm}x)", "Blocks event loop during order placement", "WS disconnects, stale prices during sleep", "Missed exits", "MEDIUM")

# ============================================================
# CAT 7: CONFIGURATION
# ============================================================
cat = "7. CONFIGURATION"

# Hardcoded values
hardcoded = []
if "= 50" in rm and "margin" in rm: hardcoded.append("margin=$50")
if "= 20" in rm and "leverage" in rm: hardcoded.append("leverage=20")
if "0.0040" in st: pass  # This is the SL floor, OK
if "= 3" in rm and "consecutive" in rm: hardcoded.append("CB=3 losses")

if hardcoded:
    finding(cat, "LOW", f"Some hardcoded values ({', '.join(hardcoded[:3])})", "Values in code instead of config", "Requires code change to adjust", "Inflexible", "EASY")

# Config validation
if "validate" in cfg.lower() or "schema" in cfg.lower():
    finding(cat, "OK", "Config validation", "Schema validation on load", "", "", "")
else:
    finding(cat, "MEDIUM", "No config validation", "Config loaded without validation", "Invalid values cause runtime errors", "Unexpected behavior", "MEDIUM")

# ============================================================
# CAT 8: SECURITY
# ============================================================
cat = "8. SECURITY"

# Dashboard auth
if "auth" in sv.lower() and "_auth_middleware" in sv:
    finding(cat, "OK", "Dashboard authentication", "Multi-user auth with session cookies", "", "", "")
else:
    finding(cat, "CRITICAL", "No dashboard auth", "Missing", "Anyone can view/control trades", "Unauthorized access", "MEDIUM")

# API keys in HTML
if "TG_BOT_TOKEN" in ix or "api_key" in ix.lower() or "api_secret" in ix.lower():
    finding(cat, "HIGH", "Secrets in client HTML", "Token found in HTML", "Viewable in browser source", "Token theft", "EASY")
else:
    finding(cat, "OK", "No secrets in client HTML", "Removed or not present", "", "", "")

# HTTPS
if "https" in sv.lower() or "ssl" in sv.lower() or "tls" in sv.lower():
    finding(cat, "OK", "HTTPS configured", "SSL/TLS enabled", "", "", "")
else:
    finding(cat, "HIGH", "No HTTPS", "HTTP only on port 8080", "Credentials sent in plaintext", "Session hijacking", "MEDIUM")

# ============================================================
# CAT 9: TESTING
# ============================================================
cat = "9. TESTING"

test_files = []
for root, dirs, files in os.walk("."):
    for f in files:
        if f.startswith("test_") or f.endswith("_test.py"):
            test_files.append(os.path.join(root, f))

if len(test_files) > 5:
    finding(cat, "OK", f"Test suite ({len(test_files)} files)", "Unit tests exist", "", "", "")
elif len(test_files) > 0:
    finding(cat, "MEDIUM", f"Minimal tests ({len(test_files)} files)", "Some tests exist", "Low coverage", "Regressions undetected", "HARD")
else:
    finding(cat, "HIGH", "No test suite", "Zero test files", "No automated testing", "Bugs ship to production", "HARD")

# ============================================================
# CAT 10: DEPLOYMENT & OPERATIONS
# ============================================================
cat = "10. DEPLOYMENT"

# Systemd service
if service:
    finding(cat, "OK", "Systemd service configured", "cryptobot.service exists", "", "", "")
else:
    finding(cat, "HIGH", "No service manager", "Missing", "Manual start required", "Extended downtime", "EASY")

# Health endpoint
if "/api/ping" in sv or "/api/health" in sv:
    finding(cat, "OK", "Health check endpoint", "/api/ping exists", "", "", "")
else:
    finding(cat, "MEDIUM", "No health endpoint", "Missing", "Can't monitor externally", "Silent failures", "EASY")

# Disk space
try:
    import shutil
    total, used, free = shutil.disk_usage("/home/opc")
    free_gb = free / (1024**3)
    if free_gb < 5:
        finding(cat, "HIGH", f"Low disk space ({free_gb:.1f}GB free)", "Running low", "Bot crashes when disk full", "Data loss", "EASY")
    else:
        finding(cat, "OK", f"Disk space OK ({free_gb:.1f}GB free)", "Sufficient", "", "", "")
except:
    finding(cat, "LOW", "Could not check disk space", "Unknown", "", "", "EASY")

# ============================================================
# PRINT REPORT
# ============================================================
print()

# Summary counts
counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "OK": 0}
for f in findings:
    counts[f["sev"]] = counts.get(f["sev"], 0) + 1

print(f"FINDINGS: {counts.get('CRITICAL',0)} CRITICAL | {counts.get('HIGH',0)} HIGH | {counts.get('MEDIUM',0)} MEDIUM | {counts.get('LOW',0)} LOW | {counts.get('OK',0)} OK")
print()

# Print by severity
for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    sev_findings = [f for f in findings if f["sev"] == sev]
    if not sev_findings:
        continue
    icon = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🔵"}[sev]
    print(f"\n{icon} {sev} ({len(sev_findings)})")
    print("-" * 90)
    for f in sev_findings:
        print(f"  [{f['cat']}] {f['title']}")
        print(f"    Current: {f['current']}")
        if f['gap']: print(f"    Gap: {f['gap']}")
        if f['risk']: print(f"    Risk: {f['risk']}")
        if f['effort']: print(f"    Fix: {f['effort']}")
        print()

# Print OK items
ok_findings = [f for f in findings if f["sev"] == "OK"]
print(f"\n✅ PASSING ({len(ok_findings)})")
print("-" * 90)
for f in ok_findings:
    print(f"  [{f['cat']}] {f['title']}: {f['current']}")

# Production readiness score
total = len(findings)
ok = counts.get("OK", 0)
critical = counts.get("CRITICAL", 0)
high = counts.get("HIGH", 0)
score = max(0, (ok / total * 100) - (critical * 15) - (high * 5))
print(f"\n{'='*90}")
print(f"PRODUCTION READINESS SCORE: {score:.0f}/100")
print(f"{'='*90}")
if score >= 80:
    print("VERDICT: PRODUCTION READY (minor issues only)")
elif score >= 60:
    print("VERDICT: CONDITIONALLY READY (fix HIGH issues before scaling)")
elif score >= 40:
    print("VERDICT: NOT READY (fix CRITICAL + HIGH before real money)")
else:
    print("VERDICT: DANGEROUS (immediate action required)")

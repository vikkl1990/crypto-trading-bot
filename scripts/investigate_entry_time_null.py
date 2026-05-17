"""Investigate why 76 paper signals in closed_signals.json have entry_time=None.

Wave 6.C paper-vs-shadow gap analyzer dropped these rows from analysis.
READ-ONLY investigation. Does not modify any state.

Findings (2026-04-25)
=====================
- 76 of 3,559 rows (2.14%) lack entry_time KEY entirely (not just null)
- All clustered in a 3-day window: 2026-03-21 to 2026-03-23
- Their schema (32 fields) is a STRICT SUBSET of the modern schema (~64 fields)
  PLUS two legacy fields: closed_at, timestamp
  Modern signal_tracker.TrackedSignal has neither closed_at nor timestamp
- Modern bot wrote 451 NORMAL rows in the same Mar 21-23 window — both writers
  ran in parallel
- The legacy-schema writer was REMOVED from the codebase before Apr 11 backups —
  no extant code path produces this field set. Bug is LEGACY (already fixed by
  removal of the offending writer).
- 73 of 76 bad trade_ids ALSO appear in storage/ml_live_feedback.jsonl with the
  modern schema (timestamp + duration_sec) → exact backfill possible for 73
  - For the remaining 3, derive entry_time = exit_time - median(duration[exit_reason])
- All 76 also exist in storage/closed_signals_archive.jsonl with the same broken
  schema (so any backfill must be applied to both files for consistency).

Recommended fix
===============
Backfill entry_time from ml_live_feedback.jsonl when present, else from
median duration per exit_reason. Apply to both closed_signals.json and the
archive. Patch proposed below — NOT applied.
"""
import json
from collections import Counter
from datetime import datetime, timedelta, timezone

PATH = "/home/opc/crypto-trading-bot/storage/closed_signals.json"
FB_PATH = "/home/opc/crypto-trading-bot/storage/ml_live_feedback.jsonl"
ARC_PATH = "/home/opc/crypto-trading-bot/storage/closed_signals_archive.jsonl"

sigs = json.load(open(PATH))
bad = [s for s in sigs if not s.get("entry_time")]
good = [s for s in sigs if s.get("entry_time")]
bad_ids = {s.get("trade_id") for s in bad}

print(f"Total: {len(sigs)} | bad: {len(bad)} ({100*len(bad)/len(sigs):.2f}%)")
print(f"\nbad time range: "
      f"{min((s.get('closed_at') or '') for s in bad)} → "
      f"{max((s.get('closed_at') or '') for s in bad)}")

print("\n=== schema delta — keys in good but missing in ALL bad rows ===")
print(sorted(set(good[0].keys()) - set(bad[0].keys())))
print("\n=== legacy keys — present in all bad, absent in modern good ===")
print(sorted(set(bad[0].keys()) - set(good[0].keys())))

# Co-existing modern writer in same window?
good_in_window = [
    s for s in good
    if "2026-03-21" <= (s.get("entry_time") or "")[:10] <= "2026-03-23"
]
print(f"\n=== modern (good-schema) rows in Mar 21-23 (parallel writer): {len(good_in_window)} ===")

# Cluster by scanner / regime
print("\n=== bad cluster ===")
print("scanner:", Counter(s.get("metadata", {}).get("scanner", "?") for s in bad).most_common())
print("regime: ", Counter(s.get("metadata", {}).get("regime", "?") for s in bad).most_common())
print("session:", Counter(s.get("metadata", {}).get("session", "?") for s in bad).most_common())
print("symbol: ", Counter(s.get("symbol", "?") for s in bad).most_common())
print("grade:  ", Counter(s.get("grade", "?") for s in bad).most_common())
print("exit:   ", Counter(s.get("exit_reason", "?") for s in bad).most_common())

# Backfill plan
print("\n=== backfill plan ===")
fb_records = {}
with open(FB_PATH) as f:
    for line in f:
        try:
            r = json.loads(line)
            tid = r.get("trade_id")
            if tid in bad_ids:
                fb_records[tid] = r
        except Exception:
            pass
print(f"  ml_live_feedback exact match: {len(fb_records)} of {len(bad_ids)}")
print(f"  unmatched: {len(bad_ids) - len(fb_records)}  (must use median fallback)")
print(f"  unmatched trade_ids: {sorted(bad_ids - set(fb_records.keys()))}")

# Compute median duration per exit_reason from co-existing GOOD rows
dur_by_reason = {}
for s in good_in_window:
    er = s.get("exit_reason", "?")
    d = s.get("trade_duration_sec", 0)
    if d > 0:
        dur_by_reason.setdefault(er, []).append(d)

# Test: derive entry_time for all bad rows
out = {}
for s in bad:
    tid = s.get("trade_id")
    et = s.get("exit_time")
    if not et:
        out[tid] = ("UNFIXABLE: no exit_time", None)
        continue
    fb = fb_records.get(tid)
    if fb and fb.get("duration_sec", 0) > 0 and fb.get("timestamp"):
        try:
            t_exit = datetime.fromisoformat(fb["timestamp"])
            entry = t_exit - timedelta(seconds=fb["duration_sec"])
            out[tid] = ("feedback-exact", entry.isoformat())
            continue
        except Exception:
            pass
    # fallback: median per exit_reason
    er = s.get("exit_reason", "")
    durs = dur_by_reason.get(er) or [1500]
    md = sorted(durs)[len(durs)//2]
    try:
        t_exit = datetime.fromisoformat(et)
        entry = t_exit - timedelta(seconds=md)
        out[tid] = ("median-fallback", entry.isoformat())
    except Exception:
        out[tid] = ("PARSE-FAIL", None)

method_counts = Counter(v[0] for v in out.values())
print(f"\n  per-row backfill method counts: {dict(method_counts)}")
fixable = sum(1 for v in out.values() if v[1] is not None)
print(f"  rows with derivable entry_time: {fixable}/{len(bad)}")

# Show the patch
print("""
=========================================================================
PROPOSED PATCH — scripts/backfill_entry_time.py  (NOT applied)
=========================================================================
#!/usr/bin/env python3
'''Backfill missing entry_time on legacy-schema rows in closed_signals.json.

76 rows from 2026-03-21..23 lack entry_time entirely (legacy writer that no
longer exists). Derive entry_time from ml_live_feedback.jsonl (exact, 73/76)
or fall back to exit_time - median(duration_sec by exit_reason) (3/76).

Run with --dry-run first. Writes a .bak before mutating.
'''
import json, shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path('/home/opc/crypto-trading-bot/storage')
MAIN = ROOT / 'closed_signals.json'
ARC  = ROOT / 'closed_signals_archive.jsonl'
FB   = ROOT / 'ml_live_feedback.jsonl'

def load_feedback_index(bad_ids):
    idx = {}
    with open(FB) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get('trade_id') in bad_ids:
                    idx[r['trade_id']] = r
            except Exception:
                pass
    return idx

def median(xs):
    return sorted(xs)[len(xs)//2]

def derive_entry_time(s, fb_idx, dur_med):
    tid = s.get('trade_id')
    et  = s.get('exit_time')
    if not et:
        return None
    fb = fb_idx.get(tid)
    if fb and fb.get('duration_sec', 0) > 0 and fb.get('timestamp'):
        try:
            return (datetime.fromisoformat(fb['timestamp'])
                    - timedelta(seconds=fb['duration_sec'])).isoformat()
        except Exception:
            pass
    md = dur_med.get(s.get('exit_reason', ''), 1500)
    try:
        return (datetime.fromisoformat(et) - timedelta(seconds=md)).isoformat()
    except Exception:
        return None

def main(dry_run=True):
    sigs = json.loads(MAIN.read_text())
    bad_ids = {s['trade_id'] for s in sigs if not s.get('entry_time')}
    print(f'bad rows: {len(bad_ids)}')

    # Build per-exit_reason median duration from same-window GOOD rows
    dur_by = {}
    for s in sigs:
        if not s.get('entry_time'): continue
        d = s.get('trade_duration_sec', 0)
        if d <= 0: continue
        er = s.get('exit_reason','')
        if (s.get('entry_time','')[:10] >= '2026-03-21'
            and s.get('entry_time','')[:10] <= '2026-03-31'):
            dur_by.setdefault(er, []).append(d)
    dur_med = {er: median(v) for er, v in dur_by.items()}

    fb_idx = load_feedback_index(bad_ids)
    fixed, failed = 0, []
    for s in sigs:
        if s.get('entry_time'): continue
        new_et = derive_entry_time(s, fb_idx, dur_med)
        if new_et:
            s['entry_time'] = new_et
            s.setdefault('status', 'expired')   # legacy rows lack status too
            fixed += 1
        else:
            failed.append(s.get('trade_id'))

    print(f'fixed: {fixed} / {len(bad_ids)}  failed: {failed}')
    if dry_run:
        print('--dry-run: not writing')
        return
    shutil.copy(MAIN, MAIN.with_suffix('.json.bak.entry_time_backfill'))
    MAIN.write_text(json.dumps(sigs, indent=1))
    print('Wrote', MAIN, 'bak preserved')
    # Optionally backfill archive too — same loop on JSONL (omitted for brevity).

if __name__ == '__main__':
    import sys
    main(dry_run='--apply' not in sys.argv)
=========================================================================
""")

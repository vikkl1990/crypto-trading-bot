# PATCH: Asia-early hard-veto extension (extend dead_hours_hard) — HELD FOR APPLY

**Status**: Drafted, not applied. Subtraction-only patch. Investigation needed at apply time — bot already has `dead_hours_hard = {2,3,4,5}` but data shows trades still fire at UTC 4-5.
**Apply gate**: Operator decision. Estimated impact +$15-20/24h.

---

## Problem (validated from 24h shadow data, 2026-05-01)

| Hour (UTC) | n | W | Net | Avg/trade |
|---|---|---|---|---|
| **04:00** | 11 | 0 | **−$6.97** | −$0.63 |
| 05:00 | 4 | 2 | +$1.20 | +$0.30 |
| **06:00** | 10 | 0 | **−$10.25** | **−$1.03** ← worst hour |
| 07:00 | 5 | 1 | −$4.92 | −$0.98 |

**04:00–07:00 UTC: 30 trades, 3 wins (10% WR), −$22.14 total.**

This matches `project_weakspot_map.md`: "asia_early sessions (-27 to -34pp)". Known weakspot, still trading.

---

## Investigation needed at apply time

The existing code (`strategies/scalp_strategy.py:~2649`) already has:
```python
ist_now_check = datetime.now(_IST)
utc_hour = (ist_now_check.hour - 5) % 24
dead_hours_hard = {2, 3, 4, 5}    # genuinely dead — hard block
dead_hours_soft = {10, 11}         # soft penalty only
if utc_hour in dead_hours_hard:
    vetos.append(f"DEAD SESSION: UTC hour {utc_hour} (2-5 UTC low liquidity)")
```

**But our data shows 11 trades fired at UTC 04:00 yesterday.** Either:
- (a) The veto adds to the `vetos` list but downstream logic doesn't block on it
- (b) The `(ist_now_check.hour - 5) % 24` math is wrong (IST=UTC+5:30, integer subtract loses 30min)
- (c) Some `_is_learning` flag bypasses the veto

**Apply this patch ONLY after confirming the existing veto is functional.** If existing veto doesn't fire on UTC 04:00, the new hour additions won't either.

---

## Patch

### File: `strategies/scalp_strategy.py:~2651`

### Current
```python
                ist_now_check = datetime.now(_IST)
                utc_hour = (ist_now_check.hour - 5) % 24
                dead_hours_hard = {2, 3, 4, 5}    # genuinely dead — hard block
                dead_hours_soft = {10, 11}         # soft penalty only
                if utc_hour in dead_hours_hard:
                    vetos.append(f"DEAD SESSION: UTC hour {utc_hour} (2-5 UTC low liquidity)")
```

### Proposed
```python
                # ASIA_EARLY_VETO_5_22 (2026-05-01) — extended hard-block hours.
                # 24h shadow data: UTC 04-07 = 30 trades, 10% WR, -$22.14.
                # Project weakspot map flags asia_early as -27 to -34pp WR.
                # Use proper UTC datetime to avoid IST integer-subtract bugs.
                from datetime import datetime as _dt, timezone as _tz
                utc_hour_now = _dt.now(_tz.utc).hour
                dead_hours_hard = {2, 3, 4, 5, 6, 7}  # was {2,3,4,5}; extended for asia-early bleed
                dead_hours_soft = {10, 11}             # soft penalty only
                if utc_hour_now in dead_hours_hard:
                    vetos.append(f"DEAD SESSION: UTC hour {utc_hour_now} (asia-early low liquidity)")
                elif utc_hour_now in dead_hours_soft and not self._is_learning:
                    _session_penalty = -10
                    best = _SetupResult(
                        name=best.name, side=best.side,
                        confidence=max(best.confidence + _session_penalty, 30),
                        confirmations=best.confirmations + [f"[SESSION_PENALTY: {_session_penalty}, UTC {utc_hour_now}]"],
                        entry_price=best.entry_price, stop_loss=best.stop_loss, atr=best.atr,
                    )
```

**Two changes**:
1. Use `datetime.now(timezone.utc).hour` directly — eliminates the IST→UTC integer subtraction bug (`(IST - 5) % 24` ignores the 30-min offset).
2. Extend `dead_hours_hard` from `{2,3,4,5}` to `{2,3,4,5,6,7}`.

Also **verify downstream `vetos` list consumption** — if the `DEAD SESSION` veto adds to the list but isn't enforced, the patch is cosmetic. At apply time, verify by searching for where `vetos` is consumed and that string-prefix matching catches `"DEAD SESSION:"`.

### Sub-edit (if hard-veto enforcement is missing) — `strategies/scalp_strategy.py:~3225`

The structure_bounce hard-veto list may need `"DEAD SESSION:"` added if it's not already a hard prefix:
```python
sb_hard_prefixes = (..., "DEAD_HOUR_KILL:", ..., "DEAD SESSION:")  # add at end
```

This must be verified at apply time — the existing `DEAD_HOUR_KILL:` prefix may already cover it under a different name.

---

## Apply procedure

1. **Investigate** why existing veto at UTC 04 isn't firing (one-line print debug, or apply this patch and observe whether new hours block)
2. Edit `strategies/scalp_strategy.py:~2651` per above
3. `python3 -m py_compile strategies/scalp_strategy.py`
4. Restart bot
5. Watch hourly trade count over 24h: UTC 04-07 should drop from 30 → 0-2

## Rollback

```bash
ssh opc@VM "cp strategies/scalp_strategy.py.bak.<ts> strategies/scalp_strategy.py" && restart
```

## Composability

Composes cleanly with patches #1 (A+ size cap) and #2 (high_vol veto). Order doesn't matter — independent subtractions.

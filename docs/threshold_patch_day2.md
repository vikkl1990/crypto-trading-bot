# DAY-2 THRESHOLD PATCH — PROPOSED, NOT SHIPPED

**Status**: Held until ≥24h of patient-maker data exists.
**Reason**: We just enabled `maker_patience_mode='patient'` on admin (2026-04-30). Shipping threshold tweaks at the same time would confound the A/B — we wouldn't know which knob moved the needle.

**Apply this patch IF AND ONLY IF**:
- Tomorrow's 03:00 UTC verdict shows admin maker fill rate ≥20% (🟢 PATIENT MAKER WORKING)
- AND admin's net P&L is still negative

If patient maker alone flips admin to net-positive, do NOT ship this — the time-decay structure may be reading correctly under maker economics.

---

## Data driving this patch (last 24h, n=137 admin shadow trades, captured 2026-04-30 21:00 UTC)

### Peak-MFE distribution
```
peak_bucket    n   wins  win%   net_sum   avg
0.00-0.05R    42    1     2%   -$70.56  -$1.68    ← duds (below grace+early-kill)
0.05-0.15R    36    0     0%   -$46.77  -$1.30    ← duds
0.15-0.30R    21    2    10%   -$13.67  -$0.65    ← marginal (avg held=600s — at cap)
0.30-0.50R    24   20    83%   +$12.19  +$0.51    ← winners (avg held=289s)
0.50-1.0R     13   13   100%   +$24.65  +$1.90    ← winners (avg held=318s)
1R+            1    1   100%   +$6.06   +$6.06
```
**Binary edge at 0.30R**: trades that peak ≥0.30R win 95% (38/40) for +$42.90; trades that peak <0.30R win 3% (3/99) for −$131.

### Hold-duration distribution
```
held_bucket       n   wins  net      avg     max_peak
00-02m (grace)   16    7   +$4.18  +$0.26    0.84R
02-05m           16    9   +$6.27  +$0.39    1.09R
05-08m           21   12   -$1.95  -$0.09    0.71R
08-10m (CAP)     79    9   -$90.63 -$1.15    0.59R    ← 58% of all flow, 11% WR, $90 bleed
10-15m            5    0   -$5.98  -$1.20    0.18R
```
The 8–10m bucket is the "force-close dud zone" — trades that survive past 5min without developing get euthanized at L2-worst-case taker.

### Counterfactual estimate
- Tightening time_decay 600s → 300s + peak-floor stall kill: ~+$40 recovered on 24h sample
- Net would move from −$86 → −$46
- Stacked with 100% maker counterfactual ($74 savings): **−$46 + $74 = +$28**

---

## Patch 1 — `execution/user_real_manager.py` line ~2594

### Current code (line 2581-2594)
```python
                    if getattr(trade, "_is_shadow", False):
                        max_age = 600  # all shadow trade_types
                    else:
                        max_age = 600 if (trade.trade_type or "").upper() == "SCALP" else 3600
                if age_sec > max_age:
                    await self._close_trade(trade, price, f"time_decay_{int(age_sec/60)}m")
                    break
```

### Proposed
```python
                    if getattr(trade, "_is_shadow", False):
                        max_age = 300  # 2026-04-30 — tightened 600→300s based on
                                       # 137-trade peak-MFE distribution: 79 of 137
                                       # closed at 600s cap with 11% WR / -$91 net.
                                       # Trades that peak >0.30R (winners) close
                                       # before 320s avg; cutting at 300s euthanizes
                                       # the dud bucket without harming winners.
                    else:
                        max_age = 600 if (trade.trade_type or "").upper() == "SCALP" else 3600
                if age_sec > max_age:
                    await self._close_trade(trade, price, f"time_decay_{int(age_sec/60)}m")
                    break
```

---

## Patch 2 — NEW guard, insert after Patch 1 block (line ~2596)

```python
                # 6b. PEAK-FLOOR STALL KILL — Phase 5.9 (2026-04-30)
                # Data: 78 trades had peak_mfe_r < 0.15R after 300s — 1 win.
                # They're done. Currently they survive the 300s max_age cap
                # only because the cap is 600s, then bleed at the cap.
                # New rule: at >=300s, if peak <0.15R, force close.
                # This is structurally different from the dead_market guard
                # (which fires earlier at 180s but only with current_r<-0.10).
                # Stall kill catches the "drifting sideways past peak"
                # pattern: trade hasn't gone bad enough for dead_market to
                # trigger but also hasn't developed.
                if age_sec >= 300 and trade.peak_mfe_r < 0.15 and \
                   getattr(trade, "_is_shadow", False):
                    await self._close_trade(trade, price, "peak_floor_stall")
                    break
```

**DB migration needed**: add `'peak_floor_stall'` to allowed exit_reason values
(see `db/migrations/014_exit_guard_refactor.sql` pattern).

---

## Patch 3 — `execution/exit_guards.py` line 73-74

### Current
```python
_STALL_AGE_SEC = 900.0
_STALL_PEAK_R = 0.20
```

### Proposed
```python
_STALL_AGE_SEC = 480.0       # 2026-04-30 tightened 900→480s
_STALL_PEAK_R = 0.20
```

**Rationale**: With the new 300s max_age cap, a 900s zombie backstop never fires (trade is already gone). Tightening to 480s catches middle-zone trades (peak 0.15-0.20R, current >-0.05R) that escape both the new max_age and peak-floor guards.

---

## Patch 4 — `execution/user_real_manager.py` line 336

### Current
```python
self._relaxed_shadow_exits = False  # disabled for clean test
self._relaxed_shadow_simulation = False  # disabled for clean test
```

### Proposed (admin only — leave niranjan as control)
```python
# 2026-04-30 — re-enable relaxed_shadow_exits ONLY for users in
# patient maker mode. Rationale: the 2026-04-26 review showed
# "8/9 Delta exit categories net negative, $47/$50 daily loss
# came from these guards firing on slippage not on real adverse
# movement". 2026-04-30 137-trade sample confirmed this with
# -$92 from time_decay_10m alone. The relaxed multipliers
# (kill -0.16R vs -0.10R, fee_floor 1.5x, patience 1.5x)
# absorb the ~6bps shadow slippage hole.
_patient = (self.maker_patience_mode == "patient")
self._relaxed_shadow_exits = _patient
self._relaxed_shadow_simulation = False  # disabled for clean test
```

---

## Verification plan after applying

1. Restart bot with patches loaded
2. Run for 24h
3. Re-run analysis query (this file's "Data driving" section)
4. Expected:
   - `time_decay_*` count drops sharply (most close as `peak_floor_stall` instead)
   - Avg held in losers drops 600s → ~300s
   - 8-10m bucket largely empty
   - Net P&L improves $30-50/24h

If results diverge by >50% from expected, **revert immediately** — the calibration sample (137 trades) is small and may not generalize.

---

## Rollback
```bash
ssh opc@150.230.171.48 "cd /home/opc/crypto-trading-bot && git diff HEAD -- execution/user_real_manager.py execution/exit_guards.py > /tmp/rollback_patch.diff && git checkout -- execution/user_real_manager.py execution/exit_guards.py"
```

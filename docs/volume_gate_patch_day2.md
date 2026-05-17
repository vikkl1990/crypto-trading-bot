# VOLUME-CONFIRM GATE PATCH — HELD FOR DAY-2 APPLY

**Status**: Documented. NOT applied. Apply tomorrow alongside the threshold tweak + maker-sim wiring patches as a single Day-2 batch.

**Wyckoff principle being honored**: *"Low volume = trap. Volume confirms price."*

---

## Problem (from 24h shadow data)

`structure_bounce` is 97.8% of admin's flow (133/137 trades). Of those 133:
- **42 trades had peak_mfe_r < 0.05R** ("never moved at all") for −$70.56
- **36 trades had peak 0.05–0.15R** for −$46.77

These ~78 silent-then-dead trades represent the classic Wyckoff "low volume trap" — price tagged a level, scored well on rejection geometry, but nobody actually committed in the trade direction.

## Root cause

`scalp_strategy.py` line 5410-5421 — **volume is a score modifier, not a hard gate**:

```python
if best_vol > 1.5:        score += 15   # (passes regardless)
elif best_vol > 1.2:      score += 10
elif best_vol > 0.9:      score += 5
else:                     score -= 5    # ← PENALIZED but not BLOCKED
```

A trade with `best_vol = 0.6` (40% below median = clearly thin participation) only loses 5 score points. It can still admit if structure_zone + wick + confluence scored high enough. **Wyckoff says this is exactly the trap.**

## The patch

### Current (line 5410-5421)
```python
        # --- STEP 3: Volume spike on or near rejection candle ---
        rejection_bar = df.iloc[-rejection_bar_idx]
        rej_vol = float(rejection_bar.get("rel_vol", 1.0))
        if np.isnan(rej_vol):
            rej_vol = 1.0
        curr_vol = float(last.get("rel_vol", 1.0))
        if np.isnan(curr_vol):
            curr_vol = 1.0
        best_vol = max(rej_vol, curr_vol)

        if best_vol > 1.5:
            confs.append(f"Volume spike {best_vol:.1f}×")
            score += 15
        elif best_vol > 1.2:
            confs.append(f"Volume {best_vol:.1f}×")
            score += 10
        elif best_vol > 0.9:
            score += 5
        else:
            # Low volume at structure = weak bounce, still allow but penalize
            score -= 5
```

### Proposed
```python
        # --- STEP 3: Volume spike on or near rejection candle ---
        # Phase 5.21 (2026-04-30) — Wyckoff "volume confirms price" hard gate.
        # Was: low vol penalized -5 but admitted. Was admitting ~78/133 trades
        # in the 0.0-0.15R peak bucket per 24h shadow data — all dud (1 win/78).
        # New: volume below 1.0× median = HARD VETO. Wyckoff: low volume = trap.
        rejection_bar = df.iloc[-rejection_bar_idx]
        rej_vol = float(rejection_bar.get("rel_vol", 1.0))
        if np.isnan(rej_vol):
            rej_vol = 1.0
        curr_vol = float(last.get("rel_vol", 1.0))
        if np.isnan(curr_vol):
            curr_vol = 1.0
        best_vol = max(rej_vol, curr_vol)

        # HARD GATE: low volume = trap. No scalp_strategy entry on weak participation.
        # 1.0 = at-median; below that price is moving without participation.
        if best_vol < 1.0:
            return None

        if best_vol > 1.5:
            confs.append(f"Volume spike {best_vol:.1f}×")
            score += 15
        elif best_vol > 1.2:
            confs.append(f"Volume {best_vol:.1f}×")
            score += 10
        else:  # 1.0 ≤ best_vol ≤ 1.2 — minimum acceptable
            confs.append(f"Volume {best_vol:.1f}× (at-median)")
            score += 5
```

## Optional companion patch — direction-aware volume

The current `rel_vol` is direction-blind (just `bar_volume / 20-bar median`). A more rigorous Wyckoff implementation reads volume *in the trade direction*:
- For LONG: rejection bar must close > open AND have above-median volume (bullish commit)
- For SHORT: close < open AND above-median volume (bearish commit)

This is more code (need to compute per-side volume on the rejection bar), so deferred to Phase 5.22. The simpler hard gate above gets us 80% of the value.

## Estimated impact

| Metric | Before (24h actual) | After (estimate) |
|---|---|---|
| structure_bounce trades fired | 133 | ~85-95 (cuts ~30%) |
| 0.0-0.15R bucket trades | 78 | ~40-50 (cuts ~50%) |
| Net 24h | −$85.48 | **~−$40 to −$30** |
| Avg fee % of gross | 854% | likely lower (fewer micro-loss trades) |

**Why the loser cut is asymmetric**: low-volume entries already populate the dud bucket (peak<0.15R = 78 trades, 1 win); winners predominantly had volume confirmation already. So we cut mostly losers.

**Risk**: A few legitimate winners might be filtered (false positives where volume was just-below-median but structure was real). That's the cost of discipline. Quote your own framework: *"First survive."*

## Apply procedure

1. **Day-2 batch**: apply this AFTER 03:00 UTC verdict + alongside threshold patch + maker-sim wiring patch.
2. Single edit to `strategies/scalp_strategy.py` at line ~5418 (insert 2 lines before the existing `if best_vol > 1.5` block).
3. Restart bot to load.
4. Monitor first 50 trades — confirm:
   - structure_bounce trade count drops ~30%
   - 0.0-0.15R peak bucket drops ~50%
   - Win rate moves from 27% → ~35-40%
   - Net 24h moves from −$85 → −$40 (or better with maker sim + threshold tweaks composed)

## Rollback

Revert the 2-line insertion. Trivial.

```bash
ssh opc@VM "cd /home/opc/crypto-trading-bot && git checkout -- strategies/scalp_strategy.py"
```

## Composability note

This patch composes cleanly with:
- **`threshold_patch_day2.md`** — kills duds at peak<0.15R after 300s. Volume gate prevents them entering in the first place. Belt-and-suspenders.
- **`shadow_maker_sim_wiring_patch.md`** — recovers fee economics on the trades that DO enter. Fewer trades × maker fees = bigger Δ.

Stack of all three: estimated −$85 → ~−$10 to +$15 net 24h (rough; actual depends on regime).

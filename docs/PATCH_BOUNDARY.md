# PATCH ERA REGISTRY

This file tracks code-behavior boundaries — moments when the bot's execution
or signal-generation logic changed materially. Production data BEFORE a
boundary is structurally different from data AFTER. Use these timestamps
to filter dashboards/queries when comparing performance.

**ANY trade with `closed_at` between two boundaries belongs to that era.**

| Era | Started | Patches Live | Description |
|---|---|---|---|
| `pre_5_22` | (since beginning) | none | Original execution logic — no maker fills, full taker fees, 600s time_decay, A+ at 1.3× size |
| `post_5_22` | **2026-05-01 13:16:22 UTC** | volume_gate, threshold_tweaks, maker_sim_wiring, A+_size_cap, high_vol_veto, asia_early_veto | First batch of 6 patches applied via `apply_day2_patches.sh --apply` |
| `post_5_22_ml` | **2026-05-01 19:36:25 UTC** | + ml_threshold_lower (0.05), + live_outcome retrain pipeline fixed | Lowered ML threshold + first successful live_outcome retrain in 20 days |
| `post_5_22_d` | **2026-05-02 05:54:00 UTC** | + Batch D: peak_floor_stall fix, symbol blacklist (now removed in E), low-ML×taker skip, LiveOutcomeScorer wired (init only) |
| `post_5_22_e` | **2026-05-02 ~07:30 UTC** | + Batch E: blacklist REMOVED, patient probe extended (4s/2.75s), low-p_fill skip, patch_era wired into DB metadata, metadata passthrough live |

**Boundary timestamps (UTC, exact):**
- `2026-05-01 13:16:22` — first apply restart, PID 4130807 → 297137 (then 286693 → 286699 → 297137 chain)
- `2026-05-01 19:36:25` — live_outcome retrain completion; subsequent restart for ML threshold
- `2026-05-02 05:09:00` — `liq_grab_ob_fvg` paper engine cron entry added (no bot restart needed; standalone)

## How to filter trades by era

```sql
-- Pre-patch trades (baseline)
SELECT ... FROM user_trades
WHERE closed_at < '2026-05-01 13:16:22 UTC';

-- Post-patch (full Wave 1 — 6 production patches live)
SELECT ... FROM user_trades
WHERE closed_at >= '2026-05-01 13:16:22 UTC'
  AND closed_at < '2026-05-01 19:36:25 UTC';

-- Post-ML-threshold (Wave 1 + ML threshold lowered)
SELECT ... FROM user_trades
WHERE closed_at >= '2026-05-01 19:36:25 UTC';

-- OR via metadata.patch_era field (after patch_era_tag patch is applied):
SELECT metadata->>'patch_era' as era, COUNT(*), AVG(pnl_usd)
FROM user_trades
WHERE closed_at >= NOW() - INTERVAL '7 days'
GROUP BY era;
```

## Why this matters

Aggregate stats over multi-era data are misleading. Example dashboard reading
on 2026-05-01 night showed:
  Total: 1,327 trades, −$681 net, 28% WR

Filtering to `post_5_22` only:
  Post-patch: 92 trades, −$34 net, 27% WR, **48% maker fills** (vs 0% pre-patch)

Same dashboard. Same data. Different story when filtered correctly.

## Maintenance protocol

When applying a new code patch that changes execution or signal behavior:

1. **Capture pre-restart baseline timestamp** (UTC, exact)
2. **Apply patch + restart bot**
3. **Capture post-restart timestamp** (UTC, exact — first new PID startup)
4. **Append a new row to the table above** with the patch list and description
5. (Optional) **Bump `CURRENT_PATCH_ERA`** in `bot/patch_era.py` so new trades get tagged

Trades silently span eras — no migration needed for old data. Filtering is
done at query/dashboard time.

| `post_5_22_f` | **2026-05-02 ~07:46 UTC** | + Batch F: asia_early veto enforcement (added DEAD SESSION: to sb_hard_prefixes), ML thresholds reverted to original 0.48-0.55 |

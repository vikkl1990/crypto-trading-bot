"""Single source of truth for the bot's current patch era.

Used by `_record_trade_db()` to stamp every new trade with `patch_era` so
dashboards and analysis queries can filter by era without parsing timestamps.

When applying a new behavior-changing patch:
  1. Update CURRENT_PATCH_ERA below to a new string
  2. Append the era to docs/PATCH_BOUNDARY.md
  3. Restart bot to pick up the new value
  4. New trades will auto-tag with the new era

Old trades retain their previous era tag (or NULL if pre-tagging).
"""
from __future__ import annotations

# Bump this string when applying a new behavior-changing patch.
# See docs/PATCH_BOUNDARY.md for the full registry.
CURRENT_PATCH_ERA: str = "post_5_22_l"

# Eras (newest first):
#   post_5_22_o    — 2026-05-02 ~16:55 UTC — Patch O Phase O.4: paper-permissive ML threshold (move from qualify_signal to _execute_shadow). Restores paper feed coverage; shadow execution unchanged.
#   post_5_22_l    — 2026-05-02 ~13:30 UTC — Patch L: Shadow Signal Bridge (paper engines → shadow execution via file queue, per-engine breaker modes A/B/C)
#   post_5_22_jk   — 2026-05-02 ~12:30 UTC — Patches J+K: realized-WR circuit breaker (40% floor, 60min pause) + ATR percentile regime gate (chop allergy for SB)
#   post_5_22_h    — 2026-05-02 ~10:30 UTC — Patch H: GRADE_BY_REGIME_SIZE_MULT (27 cohort multipliers; sideways×A+ → 0.50×, trending_down×A+ → 0.30×, high_vol×B → 1.20×)
#   post_5_22_g    — 2026-05-02 ~09:30 UTC — Patch G: ML dashboard auth middleware (X-API-Key on POSTs)
#   post_5_22_f    — 2026-05-02 ~07:50 UTC — Batch F: DEAD SESSION added to sb_hard_prefixes; ML thresholds reverted to 0.48-0.55
#   post_5_22_e    — 2026-05-02 ~07:30 UTC — Batch E: BATCH_E_5_22 (unblacklist + maker-rate push + patch_era wired into DB)
#   post_5_22_d    — 2026-05-02 05:54 UTC — Batch D: peak_floor_stall fix, blacklist (now removed), low-ML×taker skip, LiveOutcomeScorer wired
#   post_5_22_ml   — 2026-05-01 19:36 UTC — + ML threshold lower (0.43-0.50) + retrain pipeline fix
#   post_5_22      — 2026-05-01 13:16 UTC — + 6 prod patches (vol gate, threshold tweak, maker sim, A+ size cap, high_vol veto, asia early veto)
#   pre_5_22       — original baseline before any 2026-05-01 patches

# W/F Studies Registry

**Purpose**: Single source of truth for what's been tested, what passed, what was killed. Stops re-running same studies. Each row is a study with its hypothesis, verdict, and ship status.

**Maintainer rule**: Append a new row whenever a W/F study runs. Don't delete rows — failed studies are valuable evidence too.

| Date | Study | Symbols | Cell variants | Verdict | Top Q4 EV | Shipped? | Notes |
|---|---|---|---|---|---|---|---|
| 2026-04-29 | scalper_offer_vwap_mr_walkforward | BTC, ETH | 36 × 3 fee variants = 108 | **PASS Variant C** | +$0.341 | ✅ paper engine | Baseline reference |
| 2026-04-29 | scalper_offer_liq_sweep_walkforward | All 4 | 432 × 3 variants = 1,296 | KILL all | — | ❌ | All variants killed; no maker-economics revival |
| 2026-04-30 | time_decay_sweep | All 4 | 9 max_age × 4 sym × 3 var = 108 | KILL_IS_NEGATIVE × 108 | (best −$0.15) | ❌ | Tightening doesn't fix entry quality |
| 2026-04-30 | ema_momentum_relax | All 4 | 5 EMA pairs × 4 sym × 3 var = 120 | KILL_IS_NEGATIVE × 120 | (best −$0.09) | ❌ | EMA cross has no IS edge in any variant |
| 2026-04-30 | liquidity_sweep_relax | All 4 | 12 cells × 4 sym × 3 var = 144 | KILL — but 35 IS-positive cells found | (best +$0.56 IS, −$0.70 Q4) | ❌ | IS-positive but Q4 collapses — overfit |
| 2026-04-30 | bos_choch_relax | All 4 | 12 cells × 4 sym × 3 var = 144 | KILL_IS_NEGATIVE × 144 | (best −$0.15) | ❌ | No edge with simplified detector |
| 2026-04-30 | liquidity_sweep_htf | All 4 | 144 | **PASS — 7 cells (SOL only)** | +$0.293 | ✅ paper engine | HTF regime filter unlocks SOL |
| 2026-05-01 | liq_grab_ob_fvg | All 4 | 32 × 4 sym × 3 var = 384 | **PASS — 12 cells (SOL only)** | +$0.569 | ✅ paper engine | 5-step SMC: sweep + OB reclaim + FVG fill |
| 2026-05-02 | liq_grab_per_pair_relaxed | All 4 | 128 × 4 sym × 3 var = 1,536 | **PASS — 17 cells (SOL only)** | +$0.690 (tp_atr=1.5 better than 2.0) | ✅ updated paper engine to tp=1.5 | Confirms BTC/ETH/XRP have no SMC edge |
| 2026-05-02 | demand_zone_retest | BTC, ETH | 162 × 2 sym × 3 var = 972 | KILL × 972 (100% KILL_IS_NEGATIVE) | best −$0.002 (ETH variant C) | ❌ | Q3→Q4 deteriorating; pattern decays in current chop |
| 2026-05-02 | break_block_reversal | BTC, ETH | 162 × 2 sym × 3 var = 972 | KILL × 972 | best +$0.34 IS / 107% gap (BTC variant C) | ❌ | Asymmetric: SHORT-after-bullish-break carries all the edge; SHORT-only variant queued as future study |
| 2026-05-02 | bos_choch_retest_gate | All 4 | 36 × 4 sym × 3 var = 432 | KILL × 432 (0 cells with positive Q4 EV) | best ETH +$0.82 IS / −$1.26 Q4 | ❌ disabled in scanner | NameError protected the bot — 2 W/F studies converged on KILL = true regime collapse |
| 2026-05-02 | absorption_bubble | All 4 | 162 × 4 sym × 3 var = 1,944 | **PASS — 21 cells (ETH only)** | +$0.640 (lookback=30 sweep=0.3 displ=0.65 vol=1.5 tp=1.5R) | 🟡 candidate (pending Patch L bridge soak + ship) | First reversal-class W/F PASS; both LONG and SHORT carry edge; maker mandatory |
| 2026-05-02 | fvg_mtf_cascade | All 4 | 108 × 4 sym × 3 var = 1,296 | **PARKED — 17 PASS but ETH+SOL=0** | best +$1.28 (XRP n=21) / +$0.16 (BTC clean) | 🅿️ parked, re-test post-Patch M | htf_align=True/False = identical PASS rate (1h FVG is implicit trend filter); maker-only |
| 2026-05-02 | chop_vwap_mr | All 4 | 162 × 4 sym × 3 var = 1,944 | KILL × 1944 (100% KILL_IS_NEGATIVE) | — | ❌ | Mean-revert with 4h ATR rank gate failed across all params; chop strategy needs different mechanic |

## Verdict legend

- **PASS** — IS EV > $0.01, Q4 EV > $0.10, OOS gap ≤ 50%, Q3 + Q4 same sign as IS
- **HOLD_OOS_MIXED_SIGN** — Q3 vs Q4 disagree; regime-dependent edge
- **KILL_IS_NEGATIVE** — no in-sample edge
- **KILL_Q4_BELOW_FLOOR** — IS positive but Q4 EV < $0.10
- **KILL_OOS_GAP_TOO_BIG** — IS edge halves in OOS (overfit)
- **INSUFFICIENT_DATA** — < 10 IS or < 5 OOS trades
- **PARKED** — IS-positive cells exist but frequency too low and/or not all symbols passed; queued for re-test post-architectural-change (e.g. trend oracle)

## Cumulative findings (updated 2026-05-02)

1. **4 W/F-validated strategies**: liq_sweep_htf (SOL), liq_grab_ob_fvg (SOL), scalper_vwap_mr (BTC/ETH), **absorption_bubble (ETH) — NEW**
2. **0 W/F-validated XRP strategies** — biggest open gap. Need different setup family (likely catalyst-driven)
3. **Retest pattern family is dead in current chop** — 3 independent W/F studies (Demand Zone, Break Block, bos_choch retest gate) converged on KILL with same Q3 regime-collapse signature. Defer until Patch M (trend oracle) lands and re-test as gated patterns.
4. **Reversal pattern family survives chop** — Absorption Bubbles ETH PASSED with both LONG and SHORT edge. Reversal/exhaustion mechanics fade extremes that chop produces; retest mechanics chase continuation that chop denies.
5. **MTF self-gating works structurally but is too sparse standalone** — FVG MTF Cascade had 17 PASS cells but 0.3-0.8 trades/day per pair; promising as trend-oracle-gated candidate, not as standalone engine.
6. **EMA-based trend filter adds ZERO lift on top of pattern-implicit gates** (FVG MTF htf_align A/B test). Patch M trend oracle should target REGIME-NAIVE scanners (structure_bounce), BYPASS for self-gating ones (liq_grab_ob_fvg, liq_sweep_htf, FVG MTF, smc1/15/v2), COUNTER for reversals.
7. **Variant C (maker entry + scalper offer) dominates passing cells** — fee economics matter as much as signal quality. Taker-only deploys uniformly KILL.
8. **Single-pattern W/F testing has hit diminishing returns.** Next leverage = orchestration layer (Patch M trend oracle + Patch N sequence state machine), not more single-bar pattern hunts.

## Pending studies (queued)

| Priority | Study | Effort | Why |
|---|---|---|---|
| HIGH | **Re-W/F Demand Zone, Break Block, bos_choch retest WITH Patch M trend gate as precondition** | 4-6h after Patch M lands | Hypothesis: KILL'd patterns FLIP to PASS when restricted to trending regimes |
| HIGH | Absorption Bubbles per-pair re-tune for BTC/SOL/XRP (ETH already PASS) | 3-4h | Extend the working ETH cell to other pairs with custom params |
| MED | SHORT-only Break Block Reversal variant (asymmetric edge surfaced today) | 3-4h | BTC top cell showed SHORT carrying all the edge (+$0.52, 77% WR, n=22) |
| MED | FVG MTF Cascade as candidate generator inside Patch M's gated bundle | 2h after Patch M lands | Re-test with trend oracle ranking instead of standalone shipping |
| HIGH | Live A/B: candidate vs live_outcome model | 30 min code + 2-3 days data | Validates +7pp AUC claim |
| MED | Volume gate effectiveness | 1h | Verify whether the patch shipped actually filters losers |
| MED | Per-symbol time-stop tuning | 2-3h | Possible exit improvement |
| LOW | XRP-specific catalyst-driven strategy | 2-4 weeks research | XRP coverage gap — needs different setup family |
| LOW | Symbol×regime tuning sweep | 4-6h | May unlock BTC/ETH for SMC class with relaxed regime filter |

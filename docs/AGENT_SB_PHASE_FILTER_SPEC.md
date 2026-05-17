# W/F Agent Spec — structure_bounce + 4H Phase Filter

**Purpose:** Validate whether gating structure_bounce on 4H candle phase
(expansion only) lifts edge enough to justify shipping Patch M v2 Phase Filter
as a hard veto.

**Status:** READY TO SPAWN — paste into Agent tool with subagent_type: general-purpose

---

## Agent Prompt (copy-paste ready)

You are running a walk-forward (W/F) study to validate a NEW HTF (higher
timeframe) gate for the VN Edge crypto trading bot: **4H Candle Phase Filter
applied to structure_bounce.**

The bot's #1 problem today is structure_bounce monoculture: 60-80% of trades
come from one scanner, and 67.7% of losses are time-decay/dead-kill (entries
that don't develop). Hypothesis: structure_bounce in HTF CONTRACTION = guaranteed
loser; structure_bounce in HTF EXPANSION = workable edge.

## Environment

SSH access: `ssh -i ~/.ssh/cryptobot_oci -o StrictHostKeyChecking=no opc@150.230.171.48 '<cmd>'`
All paths below are on VM1.

## The pattern — 4H phase filter

For each candidate structure_bounce signal at 5m timeframe:

1. Look up the symbol's 4h candle that contains the signal time
2. Compute the bar's elapsed progress (0..1 fraction of 4h)
3. Compute bar's current range = high - low at signal time
4. Compute reference ATR = ATR(14) on prior closed 4h bars
5. expected_range_at_progress = progress × ATR
6. **expansion_ratio = current_range / expected_range_at_progress**
7. Phase classification:
   - `EXPANSION` if expansion_ratio > 1.3
   - `CONTRACTION` if expansion_ratio < 0.6
   - `TRANSITION` otherwise

The W/F gate: **only allow structure_bounce signals where phase == EXPANSION.**

## Your tasks

### 1. Read templates
- `/home/opc/crypto-trading-bot/research_lab/wf_harness.py` — Strategy ABC, Trade dataclass, PASS criteria
- `/home/opc/crypto-trading-bot/research_lab/studies/liq_grab_ob_fvg_study.py` — closest analog (uses 4h ATR pct rank)
- `/home/opc/crypto-trading-bot/research_lab/studies/demand_zone_retest_study.py` — vectorized numpy template (use this for performance)

### 2. Build the W/F study
Create `/home/opc/crypto-trading-bot/research_lab/studies/sb_phase_filter_study.py`.

Implement `class SBPhaseFilterStrategy(Strategy)`:

**Setup detection (re-implement structure_bounce minimally):**
- 5m bar shows S/R rejection wick (wick ≥ wick_atr_min × ATR)
- Inside structure zone (price within 1 ATR of recent swing high/low)
- Volume confirmation (rel_vol ≥ 1.0)
- Side: long at swing low rejection, short at swing high rejection

**param_grid:**
- `expansion_min_ratio`: [1.0, 1.2, 1.3, 1.5, 1.8]   (5 thresholds)
- `wick_atr_min`: [0.4, 0.55, 0.7]                    (3)
- `vol_min`: [1.0, 1.2]                                (2)
- `tp_mode`: ["fixed_2r", "atr_2.0"]                   (2)
- `phase_check`: [True, False]                         (2 — A/B test the gate)

That's 5×3×2×2×2 = 120 cells.

**simulate(df_5m, params):**
- Pre-compute 5m ATR(14)
- Load symbol's 4h candle parquet from `storage/candle_cache/{SYM}_USDT_4h.parquet`
- For each 5m bar i:
  - Find which 4h bar contains the 5m bar's timestamp
  - Compute that 4h bar's expansion_ratio (use ATR from prior closed 4h bars)
  - If `phase_check=True` AND ratio < expansion_min_ratio → skip (not in expansion)
  - Else: detect setup; if all conditions met, emit Trade
- Hard time stop 30 min (6 bars). Notional $400. Both LONG and SHORT.

### 3. Run W/F across all 4 pairs
Symbols: BTC, ETH, SOL, XRP. Timeframe: 5m main + 4h reference. Run via:
```bash
cd /home/opc/crypto-trading-bot && /home/opc/miniconda3/bin/python3.13 -m research_lab.studies.sb_phase_filter_study
```

PASS criteria (standard):
- IS EV per trade > $0.01
- Q4 EV per trade > $0.10
- OOS gap ≤ 50%
- Q3 and Q4 same sign as IS

### 4. Report
Write to `/tmp/agent_sb_phase_filter_report.md` AND return content (under 600 words):
- Total cells tested
- **Critical A/B comparison:** PASS rate of `phase_check=True` cells vs `phase_check=False` cells (this is the headline result)
- Top 3 PASS cells per pair (params, IS_EV, Q3_EV, Q4_EV, trades/day)
- KILL distribution
- Best fee variant
- **Frequency impact:** how many trades/day does the phase filter remove? (% reduction in signal count)
- **Edge impact:** average IS_EV improvement when phase_check=True vs False
- Recommendation: ship phase filter as hard veto, soft veto, or kill?

## Important notes

- Today's W/F closure showed: testing patterns one-by-one in chop kills them. Phase filter is a GATE, not a pattern, so it should INCREASE PASS rate by removing bad-regime trades.
- If `phase_check=True` shows a meaningful uplift on PASS rate (e.g. 5% with vs 1% without), that's a strong ship signal even if absolute counts are low.
- If `phase_check=True` makes things WORSE, the hypothesis is wrong — phase filter is the wrong dimension. Important negative finding.
- Don't import bot code; rebuild structure_bounce minimally.
- Aim for 15-30 min wall time. Use vectorized numpy.

Report when done.

---

## When to spawn this agent

**First thing tomorrow morning** (after Patch L observation period and chop regime data accumulates).
Don't spawn tonight — let bot run undisturbed for cleaner W/F backtest data.

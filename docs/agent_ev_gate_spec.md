# W/F Agent Spec — EV / Market Edge Gate

**Purpose:** Validate whether replacing the bot's static ML threshold with an Expected-Value (EV) gate AND a Market-Edge gate (model probability vs cohort baseline WR) improves edge enough to justify shipping Patch P.

**Status:** READY TO SPAWN

**Why this matters:** Today's 6 W/F kills (demand_zone, break_block, bos_choch_retest, chop_vwap_mr, sb_phase_filter, ema_angle_momentum) all measured "predicted WR" but never "edge over baseline." A 70% predicted WR signal in a 75%-baseline cohort has NEGATIVE edge — current ML threshold can't see that. EV/Edge gate addresses this structurally.

---

## Agent Prompt (copy-paste ready)

You are running a walk-forward (W/F) study to validate an EV-based and Market-Edge-based admission gate for the VN Edge crypto trading bot. This replaces the bot's static ML threshold (currently `pair_thresholds: 0.40-0.48` in settings.yaml).

The hypothesis: **trade only when EV > 0.04 AND/OR (p_model - p_mkt) > 0.04 where p_mkt = rolling 7-day cohort baseline WR.**

This is structurally different from today's six KILL'd filter studies (demand_zone, break_block, bos_choch_retest, chop_vwap_mr, sb_phase_filter, ema_angle_momentum). All those measured "is this signal good?" — this study measures "does this signal have ALPHA over baseline?"

## Environment
SSH access: `ssh -i ~/.ssh/cryptobot_oci -o StrictHostKeyChecking=no opc@150.230.171.48 '<cmd>'`
All paths below are on VM1.

## The formulas

**EV (Expected Value):**
```
EV = p * b - (1 - p)
```
- p = ml_probability (from existing model)
- b = RR = (TP - entry) / (entry - SL)  [for long; mirror for short]
- Admit if EV > ev_threshold (test 0.0, 0.02, 0.04, 0.06, 0.08)

**Market Edge:**
```
edge = p_model - p_mkt
```
- p_model = ml_probability
- p_mkt = rolling 7-day WR for the cohort (regime × side × scanner)
- Admit if edge > edge_threshold (test 0.02, 0.04, 0.06, 0.08, 0.10)

## Your tasks

### 1. Read templates
- `/home/opc/crypto-trading-bot/research_lab/wf_harness.py` — Strategy ABC, Trade dataclass, PASS criteria
- `/home/opc/crypto-trading-bot/research_lab/studies/sb_phase_filter_study.py` — recent A/B-style study (gate ON vs OFF) — use as structural template
- `/home/opc/crypto-trading-bot/research_lab/studies/demand_zone_retest_study.py` — vectorized numpy template for performance

### 2. Build the W/F study
Create `/home/opc/crypto-trading-bot/research_lab/studies/ev_edge_gate_study.py`.

Implement `class EVEdgeGateStrategy(Strategy)`:

**Setup detection (rebuild structure_bounce minimally):**
- 5m bar shows S/R rejection wick (wick ≥ 0.55 × ATR)
- Inside structure zone (price within 1 ATR of recent swing high/low — 20-bar window)
- Volume confirmation (rel_vol ≥ 1.0)
- Side: long at swing low rejection, short at swing high rejection

**Compute per-signal:**
- p = simulated ml_probability (use a proxy: e.g., 0.5 + score-derived adjustment from confluence count, OR replay actual ml_prob from closed_signals.json if available)
- b = RR = abs((TP - entry) / (entry - SL))
- EV = p * b - (1 - p)
- For market edge: maintain rolling 7-day cohort WR per (regime × side × scanner). Use deque of last N trades per cohort.
- edge = p - p_mkt

**param_grid:**
- `gate_mode`: ["none", "ev_only", "edge_only", "ev_and_edge", "ev_or_edge"]   (5 modes — A/B test the gate types)
- `ev_threshold`: [0.0, 0.02, 0.04, 0.06, 0.08]   (5)
- `edge_threshold`: [0.02, 0.04, 0.06, 0.08]   (4)
- `tp_rr`: [1.5, 2.0]   (2)
- `cohort_window_n`: [50, 100, 200]   (3 — rolling baseline window size)

5×5×4×2×3 = 600 cells. **REDUCE to ~120 cells** by fixing `tp_rr=2.0` and `cohort_window_n=100`. Then 5×5×4×1×1 = 100 cells.

**simulate(df, params):**
- Pre-compute 5m ATR(14), rolling SR levels, wicks, vol_ratio
- Iterate bars; for each setup detection:
  - Compute EV and edge (if not first 100 trades, when baseline isn't ready yet — admit those for warmup)
  - Apply gate per `gate_mode`
  - If admitted, emit Trade
- Hard time stop 30 min (6 bars). Notional $400. Both LONG and SHORT.

### 3. Run W/F across all 4 pairs
Symbols: BTC, ETH, SOL, XRP. Timeframe: 5m. Run via:
```bash
cd /home/opc/crypto-trading-bot && /home/opc/miniconda3/bin/python3.13 -m research_lab.studies.ev_edge_gate_study
```

PASS criteria:
- IS EV per trade > $0.01
- Q4 EV per trade > $0.10
- OOS gap ≤ 50%
- Q3 and Q4 same sign as IS

### 4. Report
Write to `/tmp/agent_ev_edge_gate_report.md` AND return content (under 700 words):

**Critical comparisons (the headline):**
- A/B PASS rate: each `gate_mode` (none / ev_only / edge_only / ev_and_edge / ev_or_edge)
- A/B avg IS_EV uplift: each gate mode vs `none`
- A/B trade-count reduction: each gate mode vs `none`

**Critical question to answer:** does EV gate or Market Edge gate add structural alpha (PASS rate strictly higher than `none`), unlike all 6 filter studies today?

Then standard sections:
- Total cells, PASS / KILL distribution
- Top 3 PASS cells per pair (params, IS_EV, Q3_EV, Q4_EV, trades/day)
- Best fee variant
- Best gate_mode (likely)
- Recommendation: ship as Patch P with which gate mode, threshold values, or kill?

## Important notes

- Today's filter studies all KILLED with the same Q3 collapse signature. EV/Edge gate is structurally different — it asks "does this signal have alpha over baseline" rather than "is this signal predicted to win." Should give different result.
- If `gate_mode=ev_only` PASSES while `gate_mode=none` KILLS, the gate adds REAL alpha. Strong ship signal.
- If both KILL, the framework needs reconsideration.
- Vectorize aggressively for performance. Aim 20-30 min wall time.
- Don't import bot code; rebuild structure_bounce minimally.

Report when done.

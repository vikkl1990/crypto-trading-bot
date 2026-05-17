# W/F Agent Spec — Fractional Kelly Sizing

**Purpose:** Validate whether replacing the bot's empirical Patch H 27-cohort matrix with mathematically-optimal Fractional Kelly sizing (α=0.4) improves risk-adjusted returns.

**Status:** READY TO SPAWN

**Why this matters:** Patch H grade×regime sizing is hand-tuned multipliers (0.25-1.20× per cohort). Kelly is one provably-optimal formula. Today's shadow data showed Patch H is doing useful work but is empirical, not mathematical. Kelly might do better and is cleaner.

---

## Agent Prompt (copy-paste ready)

You are running a walk-forward (W/F) study to validate replacing the bot's empirical Patch H sizing matrix with Fractional Kelly Criterion sizing.

The hypothesis: **f = α * (p*b - q) / b where α = 0.4, p = ml_probability, b = RR, q = 1-p.**

Patch H currently uses 27 hardcoded cohort multipliers in `execution/user_real_manager.py:GRADE_BY_REGIME_SIZE_MULT`. Kelly replaces all of them with one formula.

**Critical:** This is a SIZING study, not a SELECTION study. The trade SET is the same; only POSITION SIZE differs. The W/F validates whether Kelly produces better RISK-ADJUSTED returns (Sharpe, max DD) than Patch H matrix at the same trade set.

## Environment
SSH access: `ssh -i ~/.ssh/cryptobot_oci -o StrictHostKeyChecking=no opc@150.230.171.48 '<cmd>'`
All paths below are on VM1.

## The formulas

**Full Kelly:**
```
f* = (p * b - q) / b
```
- p = ml_probability
- b = RR = (TP - entry) / (entry - SL)
- q = 1 - p
- If f* < 0, don't trade (negative edge)

**Fractional Kelly:**
```
f = α * f*    where α ∈ (0, 1]
```
α = 0.25 to 0.5 is "professional" — caps Kelly's aggressiveness to avoid ruin from over-betting on miscalibrated probabilities.

**Position size in dollars:**
```
size_usd = bankroll * f
```
- Cap at min/max bankroll fraction (e.g., 1% min, 25% max)

## Your tasks

### 1. Read templates
- `/home/opc/crypto-trading-bot/research_lab/wf_harness.py` — Strategy ABC, Trade dataclass, PASS criteria
- `/home/opc/crypto-trading-bot/research_lab/studies/demand_zone_retest_study.py` — vectorized numpy template

### 2. Build the W/F study
Create `/home/opc/crypto-trading-bot/research_lab/studies/kelly_sizing_study.py`.

Implement `class KellySizingStrategy(Strategy)`:

**Setup detection (rebuild structure_bounce minimally — same as ev_edge_gate study):**
- 5m bar shows S/R rejection wick (wick ≥ 0.55 × ATR)
- Inside structure zone (1 ATR of recent swing)
- Volume confirmation (rel_vol ≥ 1.0)
- Side: long at swing low rejection, short at swing high rejection

**Compute per-signal:**
- p = ml_probability proxy (same approach as ev_edge_gate study)
- b = RR = abs((TP - entry) / (entry - SL))
- f_star = (p * b - (1-p)) / b
- f = α * f_star, capped to [0.01, 0.25] of bankroll

**Sizing modes (the A/B):**
- `sizing_mode = "patch_h"`: simulate Patch H matrix (use lookup table of 27 cohorts)
- `sizing_mode = "fixed"`: fixed $400 notional (current default in studies)
- `sizing_mode = "kelly"`: fractional Kelly with α from grid

**param_grid:**
- `sizing_mode`: ["patch_h", "fixed", "kelly"]   (3 — A/B/C)
- `alpha`: [0.25, 0.4, 0.5]   (3 — only used when sizing_mode=kelly)
- `bankroll_initial`: [1000, 5000]   (2 — bankroll size scaling)
- `tp_rr`: [1.5, 2.0]   (2)

3×3×2×2 = 36 cells (but `alpha` only matters for kelly; `patch_h` and `fixed` collapse to 6 unique non-kelly cells × 2 tp × 2 bank = 24 + kelly=12, total 36 effective).

**simulate(df, params):**
- For each setup, compute the position size based on `sizing_mode`
- Track running bankroll (compound)
- Emit Trade with `notional_usd = bankroll * f`
- Record: notional, realized P&L per trade, Sharpe, Sortino, max DD over the IS/OOS windows

**Special metrics (beyond standard PASS):**
- **Sharpe per cell**: mean_return / std_return * sqrt(252)
- **Sortino per cell**: mean_return / downside_std * sqrt(252)
- **Max drawdown per cell**: max peak-to-trough %
- **Profit Factor per cell**: sum(positive_returns) / abs(sum(negative_returns))

### 3. Run W/F across all 4 pairs
Symbols: BTC, ETH, SOL, XRP. Timeframe: 5m. Run via:
```bash
cd /home/opc/crypto-trading-bot && /home/opc/miniconda3/bin/python3.13 -m research_lab.studies.kelly_sizing_study
```

PASS criteria (standard + new):
- IS EV per trade > $0.01
- Q4 EV per trade > $0.10
- OOS gap ≤ 50%
- Q3 and Q4 same sign as IS
- **NEW: max drawdown < 25%** (Kelly without cap can blow up; reject ruinous configs)

### 4. Report
Write to `/tmp/agent_kelly_sizing_report.md` AND return content (under 700 words):

**Critical A/B comparison (the headline):**

| Sizing mode | Avg IS_EV | Sharpe | Profit Factor | Max DD |
|---|---|---|---|---|
| patch_h | $X | Y | Z | W% |
| fixed | $X | Y | Z | W% |
| kelly α=0.4 | $X | Y | Z | W% |

Then:
- Total cells, PASS / KILL counts per sizing_mode
- Best Kelly α (0.25 vs 0.4 vs 0.5)
- Per-pair best sizing_mode
- Recommendation: ship Kelly as Patch Q (replacing Patch H), keep Patch H, or stay on fixed?

## Important notes

- Kelly's strength is in PROPER sizing — bad signals get small bets, good signals get large. The PASS rate may be similar to Patch H, but Sharpe/PF should be higher if Kelly is correctly priced.
- If `sizing_mode=kelly` shows HIGHER Sharpe + LOWER max DD than `patch_h`, that's the signal to ship.
- If max drawdown blows up (e.g. >40%), the model probabilities aren't calibrated enough for Kelly. Add tighter alpha cap.
- Vectorize compounding loop (use cumulative product if possible).
- Aim 20-30 min wall time.

Report when done.

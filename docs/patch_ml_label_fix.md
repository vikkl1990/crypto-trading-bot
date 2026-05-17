# PATCH: Switch candidate_trainer label_mode from "mfe" → "trade" — HELD FOR APPLY

**Status**: Drafted, NOT applied. Apply only after the W/F validation study below confirms the new label produces equal-or-better OOS performance vs current.

**Hypothesis**: MFE-based labels (trade is "win" if `mfe_r ≥ 0.3R`) are misaligned with what the bot actually realizes (trade is "win" if `pnl_usd > 0`). Audit of 4,483 records shows **34.9% mismatch rate** — 1,461 trades that closed profitably are labeled "loss" by MFE, while 104 trades that hit MFE peak still closed losing.

The infrastructure for the fix already exists. We just need to flip a default.

---

## Evidence (from today's `ml_live_feedback.jsonl` audit)

| Outcome | n | % |
|---|---|---|
| MFE win + PnL win (label correct) | 1,894 | 42.2% |
| MFE loss + PnL loss (label correct) | 1,024 | 22.8% |
| **MFE win + PnL loss** (mislabeled — bot let winner turn into loser) | 104 | 2.3% |
| **MFE loss + PnL win** (mislabeled — bot exits profitably under 0.3R peak) | **1,461** | **32.6%** |

**The MFE label calls 32.6% of the bot's actual winners "losses".** The model is being trained to predict the wrong target.

---

## The fix is one default change (the infrastructure already exists)

`ml_training/candidate_trainer.py:793` already supports two label modes:

```python
def build_dataset_with_veto_labels(
    self,
    df: pd.DataFrame,
    symbol: str,
    scanner_func=None,
    label_mode: str = "mfe",        # ← currently default
    mfe_threshold_r: float = 0.2,
    ...
):
    """
    Label modes:
    - "mfe": Will MFE exceed threshold_r within max_bars?
            (default, claims to "generalize best" — but doesn't match production)
    - "trade": Did full simulated trade win?
              (uses LiveTrackerRunner — matches live exit logic since Phase 4.6 / 2026-04-16)
    """
```

The `"trade"` mode calls `_simulate_trade_outcome()` (line 436) which since Phase 4.6 invokes **the actual live signal_tracker in a sandbox**. Quote from its docstring:

> "Backend selection via ML_SIM_BACKEND env var: 'tracker' (default as of 2026-04-16): LiveTrackerRunner — calls the actual live signal_tracker in a sandbox. Structurally correct, matches live exit logic."

**So `label_mode="trade"` already produces realistic labels.** It just isn't the default.

---

## Patch

### Edit 1 — `ml_training/candidate_trainer.py:793`

**Current:**
```python
        label_mode: str = "mfe",
```

**Proposed:**
```python
        # ML_LABEL_FIX_5_22 (2026-05-02) — was "mfe" (default).
        # 4,483-record audit found 34.9% label mismatch with realized P&L.
        # "trade" mode invokes LiveTrackerRunner which matches live exit
        # logic (Phase 4.6 / 2026-04-16) — produces labels that match
        # what the bot actually realizes.
        label_mode: str = "trade",
```

### Edit 2 — `ml_training/candidate_trainer.py:1618` (same default in higher-level wrapper)

**Current:**
```python
        label_mode: str = "mfe",
```

**Proposed:** same as Edit 1.

### Edit 3 — `ml_training/candidate_trainer.py:1814` (third call site for default)

**Current:**
```python
        label_mode: str = "mfe",
```

**Proposed:** same as Edit 1.

---

## Required pre-apply W/F validation

Per the discipline established 2026-04-30, **DO NOT APPLY** without W/F evidence that "trade" labels produce ≥ as-good OOS performance as MFE.

### Study design

Run the W/F harness with both label modes on each scanner. Compare:
- IS EV per trade (MFE vs trade labels)
- OOS Q3 EV
- OOS Q4 EV
- OOS gap %

Pass criteria for shipping the fix:
- `trade` label OOS Q4 EV ≥ MFE label OOS Q4 EV (per scanner, on majority of symbols)
- `trade` label retains monotonic calibration buckets
- Top-quartile WR remains > 50% after switch

### Study scaffold (research_lab/studies/ml_label_comparison_study.py)

```python
from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

class MLLabelComparisonStudy(Strategy):
    """Compare candidate_trainer with label_mode='mfe' vs label_mode='trade'.

    Per cell, retrain candidate_trainer with the chosen mode and
    measure OOS performance on shadow-replayed trades.
    """
    name = "ml_label_comparison"

    def param_grid(self):
        for label_mode in ["mfe", "trade"]:
            for scanner in ["structure_bounce", "bos_choch", "liquidity_sweep"]:
                yield {"label_mode": label_mode, "scanner": scanner}

    def simulate(self, df, params):
        # 1. Train candidate_trainer with params["label_mode"]
        # 2. Apply trained model to OOS trades
        # 3. Return list of Trade with expected vs actual outcomes
        ...
```

This is a non-trivial study — needs full candidate_trainer wired into the harness. Defer until time permits or punt to live A/B (run 7 days with both modes on different cohorts).

---

## Alternative: live A/B test (faster than W/F study)

Since `live_outcome_trainer.py` already trains on realized P&L AND has a fresh model (re-trained today), we can compare:

| Cohort | Decision rule | Source |
|---|---|---|
| **A** | Use existing per-scanner candidate model (MFE-trained) | candidate_*.json |
| **B** | Use new live_outcome model (PnL-trained) | live_model_results.json |

Wire both into `MLScorer` and tag predictions with `_model_source: candidate` vs `_model_source: live_outcome`. After 7 days, compare:
- Per-cohort WR, net P&L
- ROC AUC against realized outcome
- Calibration plots

This is what the verdict pipeline `lever_verdict_*.md` could surface daily.

---

## Apply procedure (only after validation)

1. Edit `candidate_trainer.py` per the 3 edits above (~3 lines changed)
2. `python3 -m py_compile ml_training/candidate_trainer.py`
3. Trigger retrain on VM4: `/home/opc/crypto-trading-bot/scripts/trigger_ml_retrain.sh`
4. (Optional) Trigger candidate_trainer manually if it has a separate cron
5. Verify new candidate_*.json files report `"label_mode": "trade"` in their metadata
6. Run 24h with new models loaded
7. Compare top-quartile WR pre/post on `ml_live_feedback.jsonl`

## Rollback

```bash
ssh opc@VM4 "cd /home/opc/crypto-trading-bot && git checkout -- ml_training/candidate_trainer.py"
# Then re-trigger retrain to regenerate models with old labels
```

---

## Composability with other ML fixes

This patch composes cleanly with:
1. **The cron-trigger fix shipped today** (live_outcome_trainer now re-runs weekly)
2. **The 6 production patches shipped today** (volume gate, threshold tweaks, A+ size cap, high_vol veto, asia_early veto, maker sim wiring) — these reduce the execution-side noise that makes labels harder to predict

Stack: better execution + better labels + working retrain pipeline = the ML system might finally produce the "USEFUL" verdict in production rather than just in training metrics.

---

## Why I'm holding this rather than shipping

1. **The W/F harness doesn't yet have a "train ML and evaluate" mode** — would need to extend the harness (1 day work)
2. **The live A/B alternative exists** — wire both candidate + live_outcome into MLScorer with a hash-based 50/50 split. Cleaner than full W/F.
3. **Today's freshly-retrained live_outcome model needs 24h baseline** before any other ML changes
4. **This is the second-order fix** — the 6 production patches shipped today address the FIRST-order problem (execution destroying signals). Fix execution first, validate ML-via-realized-P&L afterward.

**Recommended sequencing**:
- Day 1 (today): retrain pipeline fix shipped ✓
- Day 2 (tomorrow): observe new live_outcome model in shadow data, confirm AUC stays >0.55
- Day 3-4: build live A/B test (candidate-model vs live_outcome-model on 50/50 cohort)
- Day 5: based on live A/B result, decide whether to ship this label-fix patch

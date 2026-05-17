# W/F Harness Template — Usage Guide

**File**: `research_lab/wf_harness.py`
**Status**: Smoke-tested 2026-04-30 (6 cells × 3 fee variants = 18, 0.3s wall, KILL_IS_NEGATIVE as expected for stub).

The harness is the **single discipline gate** between live observation and code apply. Any TUNE concept should be implemented as a `Strategy` subclass, run through the harness, and only ship if the verdict is PASS.

## How it works

1. You implement `param_grid()` (yields dicts) and `simulate(df, params)` (returns list of `Trade`)
2. Harness loads cached candles for each (symbol, tf), runs your `simulate()` per cell
3. Trades are bucketed by entry_ts into Q1+Q2 (IS) / Q3 (OOS-1) / Q4 (OOS-2)
4. Per fee variant (A/B/C), harness aggregates EV and computes verdict
5. Output: `walkforward.json` + `report.md` in your study's out_dir

## Discipline (encoded in verdict logic)

| Criterion | Threshold | Constant |
|---|---|---|
| Min IS sample | 10 trades | (hardcoded) |
| Min OOS sample | 5 trades | (hardcoded) |
| IS EV per trade | > $0.01 | `PASS_IS_EV_MIN` |
| Q4 EV per trade | > $0.10 | `PASS_Q4_EV_MIN` |
| OOS gap | ≤ 50% | `PASS_OOS_GAP_MAX` |
| Q3 and Q4 sign | both positive | (hardcoded) |

If ANY fail → verdict ∈ {KILL_IS_NEGATIVE, KILL_Q4_BELOW_FLOOR, KILL_OOS_GAP_TOO_BIG, HOLD_OOS_MIXED_SIGN, INSUFFICIENT_DATA}.

## Plugging in each TUNE concept

### TUNE study #1 — Volume gate (Wyckoff)

```python
# research_lab/studies/volume_gate_study.py
import pandas as pd
from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

class VolumeGateStudy(Strategy):
    name = "volume_gate"

    def param_grid(self):
        for thr in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            yield {"vol_threshold": thr}

    def simulate(self, df, params):
        # Replicate structure_bounce minimal version + volume gate.
        # Iterate bars, check rejection geometry, gate on vol_threshold.
        # Return list of Trade.
        thr = params["vol_threshold"]
        trades = []
        # ... [your structure_bounce logic with thr-gated volume] ...
        return trades

if __name__ == "__main__":
    engine = WalkForwardEngine(
        study=VolumeGateStudy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=Path("/home/opc/crypto-trading-bot/storage/wf_studies/volume_gate"),
    )
    engine.run()
```

**Cell count**: 6 thresholds × 4 symbols × 1 TF × 3 fee variants = 72 cell-variants
**Expected wall time**: ~30-60s on cached candles
**Pass criteria**: at least one threshold should pass W/F across BTC and ETH

### TUNE study #2 — Time-decay tightening

```python
class TimeDecayStudy(Strategy):
    name = "time_decay_tightening"

    def param_grid(self):
        for max_age in [180, 240, 300, 360, 420, 480, 600]:
            for stall_age in [240, 300, 360, 480, 600]:
                if stall_age <= max_age:
                    yield {"max_age": max_age, "stall_age": stall_age}

    def simulate(self, df, params):
        # Generate trades using current structure_bounce logic + new exit rules.
        # Apply time_decay at params["max_age"], peak_floor_stall at params["stall_age"].
        ...
```

**Cell count**: ~30 cells × 4 symbols × 3 variants = 360
**Expected wall time**: ~2-5 min

### TUNE study #3 — HTF directional veto (trendline)

```python
class HtfVetoStudy(Strategy):
    name = "htf_directional_veto"

    def param_grid(self):
        for veto_on in [False, True]:
            for ema_fast in [8, 13]:
                for ema_slow in [21, 50]:
                    yield {"veto_on": veto_on, "ema_fast": ema_fast, "ema_slow": ema_slow}

    def simulate(self, df, params):
        # Strategy needs 4h candles too — load via self.load_candles() or
        # accept multi-tf via subclass override.
        ...
```

**Note**: For multi-TF studies, override `simulate(self, df_5m, df_4h, params)` and add a multi-TF loader. The base harness only loads one TF per cell.

### TUNE study #4 — Dual-TF EMA momentum (your 5/8/13, 9/21, 20/50 concepts)

```python
class DualTfEmaStudy(Strategy):
    name = "dual_tf_ema_momentum"

    def param_grid(self):
        # Grid over fast/slow EMA pairs × confirmation TF
        for fast in [5, 8, 9]:
            for slow in [13, 21, 50]:
                for confirm_tf in ["5m", "15m"]:
                    if slow > fast:
                        yield {"fast": fast, "slow": slow, "confirm_tf": confirm_tf}

    def simulate(self, df, params):
        # 1m base + confirm_tf agreement + EMA cross logic
        ...
```

**Cell count**: ~12 grid × 4 sym × 3 variants = 144

### TUNE study #5 — Scanner relaxation (5 silent scanners)

One Strategy class per scanner. Each replicates the scanner's gate logic + the relaxation parameter, generates trades, validates.

## How to run a study

```bash
ssh opc@VM "cd /home/opc/crypto-trading-bot && \
  python3 -m research_lab.studies.volume_gate_study"
```

Output written to `storage/wf_studies/<study_name>/`:
- `walkforward.json` — full per-cell results
- `report.md` — top-20 PASS cells + verdict distribution

## Reading the report

The report shows top 20 PASS cells sorted by Q4 EV. If 0 PASS:
- Check verdict distribution — INSUFFICIENT_DATA means strategy generated <10 IS trades (gate too tight)
- KILL_IS_NEGATIVE means concept doesn't have edge in IS — abandon
- KILL_Q4_BELOW_FLOOR means edge erodes in OOS — overfit to IS
- KILL_OOS_GAP_TOO_BIG means edge halves in OOS — overfit
- HOLD_OOS_MIXED_SIGN means Q3 disagrees with Q4 — regime-dependent edge, risky to ship

## Iteration loop

1. Run study → get verdict distribution
2. If 0 PASS: tighten OR loosen param grid based on which failure mode dominates
3. If 1+ PASS: pick best cell, examine for over-specificity (suspect if only one symbol passes)
4. Cross-validate top cell on different symbols if not already in grid
5. Ship via Day-2-style orchestrator IFF cross-validation holds

## What NOT to do

- ❌ Apply held PRs based on live shadow data alone — use the harness
- ❌ Tune param grid to pass a study (overfitting)
- ❌ Ship single-symbol PASS without checking other symbols
- ❌ Skip OOS gap criterion ("but IS looked great") — that's the discipline rule

## Existing studies that already use this pattern

- `scripts/scalper_offer_vwap_mr_walkforward.py` (PASS — Variant C)
- `scripts/scalper_offer_liq_sweep_walkforward.py` (KILL — all variants)

These predate this harness but follow the same structure. Future studies should use `research_lab.wf_harness` for consistency.

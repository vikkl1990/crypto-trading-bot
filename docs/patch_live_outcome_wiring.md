# PATCH: Wire LiveOutcomeScorer into scalp_strategy — HELD FOR APPLY

**Status**: Module deployed at `bot/live_outcome_scorer.py`, smoke-tested. **Wiring NOT applied.** This doc is the held PR.

**Apply gate**: Recommend after 24h baseline observation (no decision change initially — measurement only).

---

## What this does

Adds a SECOND ML model (live_outcome, trained on realized P&L) to every signal alongside the existing candidate model (per-scanner, MFE-trained). Both predictions land in `signal.metadata`. **No decision logic changes** — pure measurement, fail-open.

After 24h of dual predictions, we'll have ~150 trades to compare:
- Candidate-only decision rule
- Live_outcome-only decision rule
- Combined (both gates)

The 2026-05-02 backtest on 4,530 records showed live_outcome adds **+7.03 pp AUC** over candidate, with **100% WR / +$14.52/trade at live_outcome > 0.65** (n=14 in last 7d).

---

## The wiring (3 edits)

### Edit 1 — Add scorer instance at __init__

`strategies/scalp_strategy.py:~448` (right after existing `self._ml_scorer = MLScorer(...)`):

```python
        # LIVE_OUTCOME_WIRING_5_22 (2026-05-02) — second ML model for measurement.
        # Loads model_live_shared.joblib (trained on realized P&L, +7pp AUC over
        # candidate model in 4,530-record backtest). Phase 1: measurement only —
        # writes live_outcome_prob to metadata; doesn't change decisions.
        # See docs/patch_live_outcome_wiring.md.
        from bot.live_outcome_scorer import LiveOutcomeScorer
        try:
            self._live_outcome_scorer = LiveOutcomeScorer(enabled=True)
            logger.warning("LiveOutcomeScorer initialized: %s",
                           self._live_outcome_scorer.get_stats())
        except Exception as _e:
            logger.error("LiveOutcomeScorer init failed: %s", _e)
            self._live_outcome_scorer = None
```

### Edit 2 — Add scoring call in `_build_signal()` or right after `_ml_scorer.score_candidate()`

`strategies/scalp_strategy.py:~3766` (after `ml_result = self._ml_scorer.score_candidate(...)`):

```python
            # LIVE_OUTCOME_WIRING_5_22 — score same signal with live_outcome model.
            # Fail-open: any error returns 0.5 neutral, doesn't block trade.
            live_outcome_result = {"probability": 0.5, "verdict": "DISABLED",
                                    "model_age_hours": None, "error": None}
            if getattr(self, "_live_outcome_scorer", None) is not None:
                try:
                    live_outcome_result = self._live_outcome_scorer.score({
                        "regime":           regime,
                        "scanner":          best_sr.scanner_name,
                        "side":             "long" if best.side == OrderSide.LONG else "short",
                        "session":          getattr(self, "_current_session", "us"),
                        "symbol":           symbol,
                        "trade_type":       (best.trade_type if hasattr(best, "trade_type") else "SCALP"),
                        "confidence":       best.confidence,
                        "atr_ratio":        getattr(self, "_atr_ratio", 1.0),
                        "ml_probability":   ml_result.get("probability", 0.5),
                        "leverage":         20,    # Phase 1: placeholder (size unknown until later)
                        "position_size_usd": 5000, # Phase 1: placeholder
                    })
                except Exception as _e:
                    logger.error("LiveOutcomeScorer.score failed: %s", _e)
                    live_outcome_result["error"] = str(_e)
```

### Edit 3 — Stamp metadata

`strategies/scalp_strategy.py:~4079` (where existing `signal.metadata["ml_probability"]` is set):

```python
        # LIVE_OUTCOME_WIRING_5_22 — write live_outcome predictions to metadata.
        signal.metadata["live_outcome_prob"]    = round(live_outcome_result.get("probability", 0.5), 4)
        signal.metadata["live_outcome_verdict"] = live_outcome_result.get("verdict", "?")
        signal.metadata["live_outcome_age_h"]   = round(live_outcome_result.get("model_age_hours") or 0, 1)
        if live_outcome_result.get("error"):
            signal.metadata["live_outcome_error"] = live_outcome_result["error"][:120]
```

---

## What gets measured

Every signal in `ml_live_feedback.jsonl` will gain 3-4 fields:
- `live_outcome_prob` (float in [0, 1])
- `live_outcome_verdict` (STRONG_TAKE | TAKE | WEAK | AVOID | NEUTRAL | DISABLED | ERROR)
- `live_outcome_age_h` (hours since model trained — staleness signal)
- `live_outcome_error` (only present if call failed)

After 24h, query:

```sql
SELECT
  CASE
    WHEN (metadata->>'live_outcome_prob')::float >= 0.65 THEN '5_strong'
    WHEN (metadata->>'live_outcome_prob')::float >= 0.55 THEN '4_take'
    WHEN (metadata->>'live_outcome_prob')::float >= 0.50 THEN '3_weak'
    ELSE '2_avoid'
  END as live_band,
  COUNT(*),
  SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END) as wins,
  ROUND(SUM(pnl_usd)::numeric,2) as net,
  ROUND(AVG(pnl_usd)::numeric,3) as avg
FROM user_trades
WHERE trade_type='shadow' AND closed_at >= NOW() - INTERVAL '24 hours'
  AND metadata->>'live_outcome_prob' IS NOT NULL
GROUP BY live_band ORDER BY live_band;
```

Expected: live_band='5_strong' shows higher WR + better avg P&L than '2_avoid'. If yes, ship Phase 2 wiring (gate or sizer).

---

## Phase 1 limitations (acknowledged)

| Limitation | Mitigation |
|---|---|
| `position_size_usd` is unknown at signal time (depends on ML probability and tier — chicken-and-egg) | Use placeholder 5000. Live_outcome model has position_norm as top feature — predictions will be biased toward "max position" outcome. Phase 2 can re-score retroactively. |
| `leverage` not yet computed at signal time | Use placeholder 20 (max). Same issue as above but smaller magnitude. |
| Model file may be missing on first run | Scorer fails open: returns prob=0.5, doesn't block trade |
| Model staleness (>48h) | scorer logs `model_age_hours` — if > 168h (1 week), surfaces as warning |

---

## Apply procedure

1. **Pre-check**: confirm module exists + smoke-tests OK
   ```bash
   ssh opc@VM "cd /home/opc/crypto-trading-bot && python3 -m bot.live_outcome_scorer | tail -10"
   ```
2. Edit `strategies/scalp_strategy.py` per above (3 edits)
3. `python3 -m py_compile strategies/scalp_strategy.py`
4. Restart bot: use the proper-restart pattern (`scripts/apply_day2_patches.sh` style — kill via pidfile, sleep 8, remove pidfile, nohup new)
5. Tail bot log for `LiveOutcomeScorer initialized:` warning
6. Wait 1h. Verify trades have `live_outcome_prob` in metadata via psql query above.

## Rollback

Single-edit revert:

```bash
ssh opc@VM "cp strategies/scalp_strategy.py.bak.<ts> strategies/scalp_strategy.py" && restart
```

OR set `enabled=False` at scorer init to silently disable without code revert.

## Composability

Composes cleanly with all 8 patches shipped today (volume gate, threshold tweaks, A+ size cap, high_vol veto, asia_early veto, maker sim wiring, ML threshold lower, ML retrain pipeline fix). Adds metadata only; doesn't conflict with any of them.

## Phase 2 (after 24h baseline)

Based on observed `live_outcome_prob` × `pnl_usd` correlation in the 24h sample:

- **If lift confirmed**: add `live_outcome_prob >= 0.50` as additional admit gate (combined with candidate threshold). Should marginally tighten flow but improve avg P&L.
- **If lift weak**: keep as measurement only. Use for post-hoc analysis and exit_reason re-bucketing.
- **If anti-predictive**: investigate (likely model staleness or feature drift). Disable scorer.

---

## Why wait 24h before applying

**The cron retrain pipeline fix shipped today retrained `live_outcome` for the first time in 20 days.** I want to be sure the new model behaves consistently in shadow data before adding ANOTHER moving piece. 24h baseline isolates threshold-lower's effect from this wiring's effect.

**Alternative**: apply now, monitor with degradation criteria. The wiring is fail-open — worst case is `live_outcome_prob` is missing from metadata.

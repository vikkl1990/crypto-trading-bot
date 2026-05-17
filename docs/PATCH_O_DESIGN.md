# Patch O — Paper-Layer Permissive Recording

**Status:** DESIGN-ONLY — no code shipped. Implementation deferred until architect approval.
**Sentinel:** PATCH_O_5_22
**Era stamp on activation:** `post_5_22_o`
**Related:** Patch M (trend oracle), Patch N (sequence state machine), Patch L (shadow bridge)

---

## 1. The architectural mistake we made today

Today we shipped 6 patches (Batch D, E, F, Patches G, H, J/K, L). Each veto we added did TWO things:
1. ✅ Blocked execution (intended)
2. ❌ Blocked paper-feed recording (unintended consequence)

Result: paper feed dropped **78% in 24 hours** (May 1: 153 records → May 2: 60 records). ML model now sees a CENSORED dataset — it can never learn whether our newly-strict vetoes are correctly blocking losers or wrongly blocking winners.

**The architect's insight: paper is for LEARNING. Execution is for DECIDING. Don't conflate them.**

## 2. The current (broken) flow

```
Scanner detects setup
    ↓
ML scoring (assigns ml_prob)
    ↓
Veto stack runs (HTF, regime, asia_early, LOW_PFILL_SKIP, threshold gates...)
    ↓
IF vetoes pass → broadcast → ml_live_feedback.jsonl + shadow execution
IF vetoes fail → SILENT (no paper record, no learning, ML model never sees signal)
```

**Selection bias.** When we tighten vetoes:
- Execution: fewer trades (intended)
- Paper feed: also fewer records (unintended)
- ML retraining: trains on ever-narrower sample
- Verdict: model loses calibration on the rejected population

## 3. The correct flow

```
Scanner detects setup
    ↓
ML scoring (assigns ml_prob)
    ↓
Veto stack runs — but only TAGS the signal, doesn't drop it
    ↓
ALL candidates → ml_live_feedback.jsonl  (with vetoes_applied list, would_execute bool)
    ↓
Execution router (shadow/real)
    ├── If vetoes_applied empty + ML pass → execute shadow trade
    └── Else → skip execution but signal is RECORDED for learning
```

**Single principle:** Paper layer is **observational** (write-only, full coverage). Execution layer is **decision-making** (filtered).

## 4. Where each gate currently lives — and where it SHOULD live

| Gate | Currently in | Effect on paper feed | Should be in |
|---|---|---|---|
| LOW_PFILL_SKIP (Batch E) | `_execute_shadow` | Paper record DOES happen, but no shadow trade | ✅ already correct (just tag) |
| LOW_ML × TAKER skip (Batch D) | `_execute_shadow` | Same | ✅ same |
| ML threshold (PAIR_ML_THRESHOLDS) | `qualify_signal` | **Paper record blocked** ← problem | Move to execution; record always |
| HTF STRICT, REGIME MISMATCH, asia_early DEAD SESSION | `qualify_signal` | **Paper record blocked** ← problem | Move to execution; record always |
| Patch J circuit breaker | `_execute_shadow` (Patch JK) | Paper record DOES happen | ✅ correct, just needs tagging |
| Patch K ATR regime gate | `_execute_shadow` (Patch JK) | Same | ✅ same |
| Patch H grade×regime sizing | sizing only — doesn't block | ✅ correct | ✅ no change |
| Bridge LOW_PFILL_SKIP (in consumer) | `_execute_shadow` (via consumer) | Same | ✅ correct |

**Verdict:** The ONLY gates currently corrupting the paper feed are the ones in `qualify_signal` — particularly the ML threshold and HTF/REGIME/SESSION vetoes.

The execution-layer gates (LOW_PFILL_SKIP, circuit breaker) already DO record paper signals — they just block the trade. The paper feed is silent today because qualify_signal is the choke point.

## 5. New ml_live_feedback.jsonl schema

### Existing fields (kept)
```json
{
  "timestamp": "...",
  "trade_id": "...",
  "symbol": "BTC/USDT",
  "side": "long",
  "setup_type": "structure_bounce",
  "trade_type": "INTRADAY",
  "regime": "sideways",
  "session": "asia_early",
  "confidence": 65,
  "grade": "B",
  "ml_probability": 0.62,
  "ml_verdict": "TAKE",
  "ml_model_version": "...",
  "entry_price": 78050.0,
  "stop_loss": 77900.0,
  ...
}
```

### NEW fields (Patch O additions)
```json
{
  ...existing fields...,

  // === Veto audit (the new contribution) ===
  "vetoes_applied": ["DEAD SESSION", "LOW_PFILL_SKIP"],
  "vetoes_hard": ["DEAD SESSION"],
  "vetoes_soft": ["LOW_PFILL_SKIP"],
  "would_execute": false,
  "blocked_reason": "DEAD SESSION (asia_early)",

  // === Execution outcome ===
  "did_execute": false,
  "shadow_trade_id": null,        // populated only if did_execute=true
  "real_trade_id": null,           // populated only if real-mode executed

  // === Maker-sim transparency ===
  "p_fill_maker_sim": 0.18,
  "expected_fee_type": "taker",
  "maker_mode_used": "patient",

  // === ML threshold context ===
  "ml_threshold_at_signal": 0.50,  // what threshold was active at signal time
  "ml_pass_threshold": true,        // ml_probability >= ml_threshold_at_signal

  // === Patch era (already in shadow trades) ===
  "patch_era": "post_5_22_l"
}
```

### Field semantics

- `vetoes_applied`: ALL vetoes that fired, regardless of severity
- `vetoes_hard`: subset that would have hard-killed execution
- `vetoes_soft`: subset that would have applied confidence penalty only
- `would_execute`: True iff vetoes_hard is empty AND ml_pass_threshold (and any other hard gate)
- `did_execute`: True iff shadow/real trade was actually placed
- These two should match in normal operation; mismatch = downstream gate fired (e.g. balance, position cap)

## 6. Implementation phases

### Phase O.1 — Schema extension (1-2h)
- Update ml_live_feedback writer to accept new fields
- Default missing fields to null/empty list (backward-compat)
- Smoke test: existing paper feed still parses

### Phase O.2 — Refactor qualify_signal to return tagged result (3-4h)
- Change return type: `Optional[Signal]` → `QualifyResult(signal, vetoes_applied, would_execute, blocked_reason)`
- Callers check `would_execute` to decide whether to broadcast/execute
- Callers always emit a paper record via the new schema
- Gate the refactor: feature flag `PATCH_O_ENABLED=true` — if false, fall back to old behavior

### Phase O.3 — Wire paper-record-always (2-3h)
- New `_record_paper_candidate(qualify_result)` function in scalp_strategy
- Called regardless of would_execute outcome
- Writes to ml_live_feedback.jsonl with full schema
- Existing paper-engine writers (cron) update similarly via shadow_bridge.publish_signal augmentation

### Phase O.4 — Move ML threshold to execution layer (1-2h)
- ML threshold check currently in qualify_signal — REMOVE from there
- Add to `_execute_shadow` and bridge consumer
- Paper records get ml_pass_threshold tag; execution layer enforces

### Phase O.5 — 24h soak + ML retrain validation (24h)
- Compare paper feed volume Patch O ON vs OFF
- Expected: 2-4× increase in records (vetoed signals now recorded)
- ML retrain on full dataset; validate AUC vs current censored model
- If AUC ≥ 0.55 (current baseline), proceed; if < 0.50, investigate

### Phase O.6 — Roll out execution-layer veto refactor (2-3h)
- Move HTF STRICT, REGIME MISMATCH, DEAD SESSION from qualify_signal to `_execute_shadow`
- Each veto becomes a fail-fast check in execution path
- Paper feed continues recording all candidates

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| ml_live_feedback.jsonl grows 3-5× faster | Add daily rotation + DB sink option in Phase O.5 |
| Existing ML training pipeline breaks on new schema | Phase O.1 ships schema with backward-compat defaults; trainer updates non-blocking |
| Execution layer becomes too verbose with multiple gates | Encapsulate as `ExecutionGateChain` — single iterable of gates |
| Selection bias persists if some scanners short-circuit before qualify_signal | Audit each scanner; ensure they all reach qualify_signal even on weak setups |
| Paper feed signal-to-noise drops | ML trainer will weight by realized PnL, not just count — quality matters more than quantity |

## 8. Verification gates

| Phase | Gate |
|---|---|
| O.1 | Existing paper feed parses without error after schema change |
| O.2 | qualify_signal output unchanged when PATCH_O_ENABLED=false |
| O.3 | New paper records appear with vetoes_applied populated correctly |
| O.4 | Paper feed volume INCREASES day-over-day (back to May 1 levels minimum) |
| O.5 | ML retrain AUC ≥ baseline (0.55) on the new full-coverage dataset |
| O.6 | Shadow trade volume UNCHANGED (vetoes still apply at execution) — only paper records change |

## 9. Out of scope

- ❌ Don't refactor scanners themselves
- ❌ Don't change ML scoring (trainer reads from feed; trainer logic separate)
- ❌ Don't change Bybit demo path
- ❌ Don't change real execution path (other than gate location)

## 10. Effort estimate

| Phase | Wall time |
|---|---|
| O.1 schema | 1-2h |
| O.2 refactor qualify_signal | 3-4h |
| O.3 wire paper-record-always | 2-3h |
| O.4 move ML threshold to execution | 1-2h |
| O.5 soak + ML retrain validation | 24h |
| O.6 execution-layer veto refactor | 2-3h |
| **Total** | **~12-16h work + 1 day soak** |

## 11. Why this is the highest-leverage architectural change

**Compared to Patches M and N:**
- Patch M (trend oracle): adds ANOTHER veto on top of existing stack
- Patch N (sequence state machine): adds ANOTHER signal source
- **Patch O: rebuilds the foundation so M and N's data is actually USABLE**

If we ship M and N before O, we add more filters and more signals — but ML still trains on a censored sample. Patch O makes M, N, and all future patches safer because their effects are MEASURABLE in the paper feed.

**Patch O should ship FIRST, before any other architectural addition.**

## 12. Decision gates for the architect

Before starting Phase O.1, confirm:

1. **Default for `PATCH_O_ENABLED`: false (opt-in) or true (default-on)?** Recommend `false` for first 24h, then `true`.
2. **Storage strategy for the larger paper feed: rotate JSONL daily, or push to Postgres table?** Recommend daily rotation for v1, DB migration as Phase R3 of refactor doc.
3. **Should standalone paper engines (liq_grab_ob_fvg etc.) ALSO emit pre-veto records?** They don't have vetoes today — but if they grow them, same principle applies. Recommend yes from day 1.
4. **ML retrain trigger:** retrain on every Sunday with the new full-coverage dataset, or wait for 14d of accumulated coverage? Recommend wait for 14d.

---

## Appendix — code skeleton

```python
# strategies/scalp_strategy.py

@dataclass
class QualifyResult:
    """Replaces Optional[Signal] return from qualify_signal."""
    signal: Optional[Signal]              # the candidate signal (always populated even if vetoed)
    vetoes_applied: List[str]             # ALL vetoes that fired
    vetoes_hard: List[str]                # subset that would block execution
    vetoes_soft: List[str]                # subset that's confidence-penalty only
    would_execute: bool                   # True iff no hard vetoes + ml pass
    blocked_reason: Optional[str]         # human-readable; first hard veto if blocked


def qualify_signal(self, ...) -> QualifyResult:
    """Refactored: always returns a QualifyResult, never None."""
    # ... existing veto computation ...

    qr = QualifyResult(
        signal=candidate_signal,
        vetoes_applied=vetos,
        vetoes_hard=hard_vetos,
        vetoes_soft=soft_vetos,
        would_execute=(len(hard_vetos) == 0),
        blocked_reason=hard_vetos[0] if hard_vetos else None,
    )

    # ALWAYS record to paper feed — regardless of would_execute
    if PATCH_O_ENABLED:
        self._record_paper_candidate(qr)

    return qr


def _record_paper_candidate(self, qr: QualifyResult) -> None:
    """Patch O — write the candidate to ml_live_feedback regardless of vetoes."""
    sig = qr.signal
    record = {
        # ... all existing fields from current paper record ...
        "vetoes_applied": qr.vetoes_applied,
        "vetoes_hard": qr.vetoes_hard,
        "vetoes_soft": qr.vetoes_soft,
        "would_execute": qr.would_execute,
        "blocked_reason": qr.blocked_reason,
        "did_execute": False,    # caller updates this if execution happens
        "shadow_trade_id": None,
        "patch_era": _PATCH_ERA_FOR_TRADES,
    }
    self._paper_writer.append(record)


# Caller (in orchestrator or scanner loop):
qr = strategy.qualify_signal(...)
if qr.would_execute:
    await self._execute_shadow(qr.signal, ...)
    # update paper record's did_execute=True via correlation key
```

---

**End of Patch O design doc. Ready for architect approval before Phase O.1 implementation.**

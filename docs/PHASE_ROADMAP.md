# Phase Roadmap — VN Edge Bot

**Current state**: Phase 5.22 (post-Batch-F).  Two paper engines live for SOL; one for BTC/ETH. 6-week shadow data accumulated; 3 W/F-validated strategies.

**Next major decision**: shadow_live → live cutover. Requires Phase D feature parity.

---

## ✅ COMPLETED (this session)

### Phase 5.22 — Today's session (2026-05-01 → 2026-05-02)

- 21 patches across 5 bot restarts
- 6 production patches (volume gate, threshold tweaks, maker sim, A+ size cap, high_vol veto, asia_early veto)
- ML threshold tuning (lower → revert)
- ML retrain pipeline fix (was silently broken 20 days)
- LiveOutcomeScorer wired into bot (Phase 1: measurement only)
- 2 new paper engines deployed (liq_sweep_htf, liq_grab_ob_fvg)
- 3 W/F-validated SOL strategies
- patch_era stamping every trade
- Architecture cleanup (real_manager.py archived, 4 fix scripts archived, CURRENT_PHASE bumped)

---

## 🟡 PHASE A — SOL Paper Engine Optimization (1 week)

**Goal**: Tune the 2 SOL paper engines with live data to maximize Q4 EV.

**Data needed**: ~50 closed trades per engine for statistical confidence.

**Tasks**:
- [ ] After 50+ closed liq_grab_ob_fvg trades, measure live vs W/F predicted EV
- [ ] After 50+ closed liq_sweep_htf trades, same
- [ ] Run W/F study comparing tp_atr 1.5 vs 1.8 vs 2.0 on live data
- [ ] Run W/F study on disp_atr 0.5 vs 0.6 vs 0.7
- [ ] If significant lift, lock new params

**Estimated impact**: +$0.10-0.20/trade per engine

---

## 🟡 PHASE B — BTC/ETH Strategy Diversification (2-3 weeks)

**Goal**: Add a 2nd validated strategy per BTC and ETH (currently only scalper_vwap_mr).

**Why**: BTC/ETH still rely on a single strategy class. Diversification reduces regime risk.

**Tasks**:
- [ ] Research candidates: order_block_entry_relax, momentum_ride, post_impulse
- [ ] Build W/F study per candidate (focus BTC + ETH only)
- [ ] If any pass, deploy as 2nd paper engine for those pairs
- [ ] Re-validate combined-engine performance

**Estimated impact**: 2× signal density → +30-50% per-day P&L

---

## 🟡 PHASE C — XRP Coverage Gap (2-4 weeks)

**Goal**: Build XRP-specific strategy. XRP currently has ZERO W/F-validated strategies despite being 10% of admin's flow.

**Hypothesis**: XRP needs catalyst-driven setups (news/regulatory events), not the SMC chop-fade family that works on SOL.

**Tasks**:
- [ ] Research XRP microstructure (volume profile, news cycle, regulatory event impact)
- [ ] Build candidate strategies: news-event front-running, post-catalyst momentum, range-bound mean revert
- [ ] W/F validate each
- [ ] If passes, deploy XRP-specific paper engine

**Estimated impact**: +$5-15/24h once validated

---

## 🟡 PHASE D — Live Cutover (1 month)

**Goal**: Go from shadow_live → live for at least one user (probably admin first as small-stake test).

**Blockers**: Need to port 12 features from `.archive/legacy/real_manager.py` into `user_real_manager.py`:

| Feature | What it does | When needed |
|---|---|---|
| `partial_close_real()` | Lock 75% / run 25% on TP1 hit | High |
| `update_exchange_sl()` | Server-side stop on Delta | Critical for safety |
| `mirror_paper_exit()` | Paper-to-real exit mirroring | Standard |
| `handle_exchange_fill()` | WebSocket fill handler | Latency-sensitive |
| `handle_exchange_position_close()` | WS close handler | Standard |
| `_emergency_close_all()` | Panic button | Operational |
| `mirror_paper_trade()` | Entry mirroring | Standard |
| `execute_independent_real_trade()` | Direct exec API | Optional |
| `sync_with_paper()` | Paper/real sync | Optional |
| `sync_exchange_positions()` | Periodic reconciliation | Standard |
| `_get_nautilus()` | Nautilus framework integration | Optional |
| Server-side bracket orders | OCO at exchange | Optional |

**Tasks**:
- [ ] Port `partial_close_real` + `update_exchange_sl` (critical)
- [ ] Port `handle_exchange_fill` + `handle_exchange_position_close` (WebSocket events)
- [ ] Port `_emergency_close_all` (panic button)
- [ ] Add bot-side circuit breaker (max daily loss)
- [ ] Smoke-test on Delta testnet
- [ ] Switch admin to `bot_mode='live'` with small stake ($100)
- [ ] Monitor for 7 days
- [ ] If validated, scale stake

**Estimated impact**: Validates production economics for real money

---

## 🟡 PHASE E — Multi-Exchange (1-2 months)

**Goal**: Diversify across Bybit + OKX in addition to Delta India.

**Why**: Single-exchange risk (technical issues, regulatory). Bybit has thinner spreads on some pairs.

**Current state**: Bybit demo trades fire (saw 1,269 records in DB). Architecture supports it but isn't scaled.

**Tasks**:
- [ ] Audit Bybit vs Delta fill quality on shadow data
- [ ] Build per-exchange routing logic (signal → best execution venue)
- [ ] Deploy paper engine on Bybit
- [ ] Add OKX integration

---

## 🟡 PHASE F — ML as Hard Gate (2 weeks after Phase A)

**Goal**: After live_outcome A/B validation, use it as a hard gate (skip trades where prob < threshold) instead of just a ranker.

**Pre-req**: 14+ days of fresh live_outcome model + Phase A live data.

**Tasks**:
- [ ] Run definitive A/B: candidate-only vs candidate+live_outcome ≥ 0.50 gate
- [ ] If A/B shows live_outcome adds ≥ $0.20/trade, ship as gate
- [ ] Tune the threshold per-pair
- [ ] Monitor for 7 days

**Estimated impact**: +$15-25/24h if validation succeeds

---

## 🚧 ORTHOGONAL: Architecture Debt (anytime)

These don't block any phase but should be addressed for codebase health:

| Item | Effort |
|---|---|
| Consolidate 4 HTF vetoes → 1 evaluator | 4-6h |
| Remove 10 #DISABLED# blocks | 30min |
| Storage backup hygiene (2.4GB) | 1h |
| Rotate closed_signals_archive.jsonl (25MB) | 1h |
| Bump bb_squeeze ML model (40 days stale) | 1h |
| Fix degradation_check.sh false-positive whitelist | 30min |

---

## SEQUENCING RECOMMENDATION

```
Week 1:  Phase A (SOL optimization, data-driven)
Week 2:  Phase F prep (live_outcome A/B)  +  Phase B research (BTC/ETH 2nd strategy)
Week 3:  Phase B execution (deploy 2nd BTC/ETH engine)
Week 4:  Phase D start (port partial_close_real + update_exchange_sl)
Week 5-6: Phase C XRP research (parallel)
Week 7-8: Phase D live cutover (admin small-stake)
Week 9+:  Phase E multi-exchange OR Phase F ML-as-gate (data-driven choice)
```

---

## DAILY ARCHITECT RHYTHM (the user explicitly asked for this earlier)

Every UTC 05:30 (the operator's morning IST 11:00):
1. Pull `storage/verdicts/lever_verdict_*.md` — yesterday's verdicts
2. Pull `storage/code_review/rollback_diff_*.md` — overnight reverts
3. Pull `storage/venue_perf/*.md` — exchange health
4. Run shadow loss review query
5. Surface top 3 attack surfaces
6. Output one-page brief

**Status**: Not yet automated. Held as manual responsibility for the operator until automated.

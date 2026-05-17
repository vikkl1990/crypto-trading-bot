# SHADOW MAKER SIM WIRING PATCH — HELD FOR DAY-2 APPLY

**Status**: Module deployed (`execution_v2/shadow_maker_sim.py`), tested standalone. Wiring NOT applied to `user_real_manager.py`. Apply tomorrow after the 03:00 UTC verdict has fired with the current taker-only baseline.

**Why hold**: Two changes at once (patient mode + maker simulator) confounds the A/B. We need one clean 24h baseline showing the simulator differentiating admin vs niranjan. Today's verdict will fire 🔴 SHADOW SIMULATOR NOT HONORING PATIENT MODE — that's the correct signal that we have work to do.

---

## Calibration validated (smoke test 2026-04-30 21:43 UTC)

```
BTC/USDT @ 1-tick spread, 12 lots, 100-lot top depth:
  standard   51%   (target 40-50%)  ✓
  patient    67%   (target 55-65%)  ✓
  l2_aware   63%   (target 50-60%)  ✓

SOL/USDT @ 2-tick spread, 50 lots, 50-lot top depth:
  standard   19%   (target 15-25%)  ✓
  patient    21%   (target 20-30%)  ✓ (low because size = 100% of depth)

A/B differential on BTC: +16pp patient over standard
A/B differential on SOL: +2pp  patient over standard (capacity-limited)
```

**Determinism**: same signal_id always produces same outcome (SHA-256 based). ✓

---

## Patch 1 — `execution/user_real_manager.py:2036` `_execute_shadow()`

### Current code (lines 2065-2077)
```python
        # Taker fill simulation (conservative — real maker rate will be better)
        if order_side == "buy":
            shadow_fill = float(book["asks"][0][0])
        else:
            shadow_fill = float(book["bids"][0][0])

        # Slippage model (gap G): simulate adverse slippage on top of L2 worst-case.
        # ... (existing slip block)

        # Recalc SL from shadow fill (preserve R-distance)
        if shadow_fill != entry_price and entry_price > 0:
            sl = sl + (shadow_fill - entry_price)

        # Fee: 0.05% × 1.18 GST = 0.059% taker
        notional = shadow_fill * lots * trade_contract_size
        shadow_entry_fee = notional * 0.00059
```

### Proposed
```python
        # Phase 5.21 (2026-04-30) — MAKER FILL SIMULATION.
        # Was: hardcoded taker worst-case (0.059%). Made admin-vs-niranjan
        # patience A/B silent because shadow path didn't read maker_patience_mode.
        # Now: probabilistic maker fill via shadow_maker_sim, gated by
        # SHADOW_MAKER_SIM_ENABLED env var (default off for safe rollout).
        from execution_v2.shadow_maker_sim import simulate_entry_fill

        _maker_sim_enabled = (
            _os.getenv("SHADOW_MAKER_SIM_ENABLED", "false").lower() == "true"
        )
        _signal_id_for_sim = str(
            signal.get("id") or signal.get("signal_id")
            or f"{symbol}_{order_side}_{int(time.time()*1000)}"
        )

        if _maker_sim_enabled:
            _fill = simulate_entry_fill(
                order_side=order_side,
                book=book,
                symbol=symbol,
                our_lots=float(lots),
                tick_size=float(tick_size or 0.01),
                patience_mode=getattr(self, "maker_patience_mode", "standard"),
                signal_id=_signal_id_for_sim,
            )
            shadow_fill = _fill.fill_price
            _fee_type_used = _fill.fee_type
            _fee_pct_used = _fill.fee_pct
            # Stamp metadata for post-hoc A/B analysis
            if not isinstance(meta, dict):
                meta = {}
            meta["maker_sim_enabled"] = True
            meta["maker_sim_filled"] = _fill.filled_as_maker
            meta["maker_sim_mode"] = _fill.mode_used
            meta["maker_sim_p_fill"] = round(_fill.p_fill, 4)
            meta["maker_sim_spread_ticks"] = _fill.spread_ticks
        else:
            # Legacy path — taker worst-case (preserve current behavior for rollback)
            if order_side == "buy":
                shadow_fill = float(book["asks"][0][0])
            else:
                shadow_fill = float(book["bids"][0][0])
            _fee_type_used = "taker"
            _fee_pct_used = 0.00059

        # Slippage model — only applies when sim returns taker (maker fills don't slip)
        try:
            _slip_bps = float(_os.getenv("SHADOW_SLIPPAGE_BPS", "0"))
        except Exception:
            _slip_bps = 0.0
        if _slip_bps > 0 and _fee_type_used == "taker":
            _slip_factor = _slip_bps / 10000.0
            if order_side == "buy":
                shadow_fill = shadow_fill * (1 + _slip_factor)
            else:
                shadow_fill = shadow_fill * (1 - _slip_factor)

        # Recalc SL from shadow fill (preserve R-distance)
        if shadow_fill != entry_price and entry_price > 0:
            sl = sl + (shadow_fill - entry_price)

        # Fee: variable based on maker/taker outcome
        notional = shadow_fill * lots * trade_contract_size
        shadow_entry_fee = notional * _fee_pct_used
```

### Trade record update — line 2126
```python
            entry_fee_usd=float(shadow_entry_fee),
            fee_type=_fee_type_used,    # ← was hardcoded "taker"
```

---

## Patch 2 — `execution/user_real_manager.py:3005` `_close_shadow()`

Same treatment for the exit fill. The scalper-offer math stays in place; the simulator just decides whether the close is `maker` or `taker` for fee calculation purposes.

```python
        # Phase 5.21 — MAKER EXIT SIMULATION (mirrors entry sim)
        from execution_v2.shadow_maker_sim import simulate_exit_fill

        _maker_sim_enabled_x = (
            _os.getenv("SHADOW_MAKER_SIM_ENABLED", "false").lower() == "true"
        )
        _entry_was_maker = (str(getattr(trade, "fee_type", "") or "") == "maker")

        if _maker_sim_enabled_x and book and book.get("bids") and book.get("asks"):
            _exit_fill = simulate_exit_fill(
                side=trade.side,
                book=book,
                symbol=trade.symbol,
                our_lots=float(trade.position_size),
                tick_size=float(trade.tick_size or 0.01),
                patience_mode=getattr(self, "maker_patience_mode", "standard"),
                signal_id=trade.trade_id,
                entry_was_maker=_entry_was_maker,
                holding_sec=_holding_sec,
            )
            actual_exit = _exit_fill.fill_price
            _exit_fee_type = _exit_fill.fee_type
            _base_exit_fee_pct = _exit_fill.fee_pct
        elif book and book.get("bids") and book.get("asks"):
            # Legacy taker exit
            if trade.side == "long":
                actual_exit = float(book["bids"][0][0])
            else:
                actual_exit = float(book["asks"][0][0])
            _exit_fee_type = "taker"
            _base_exit_fee_pct = 0.00059
        else:
            actual_exit = float(exit_price)
            _exit_fee_type = "taker"
            _base_exit_fee_pct = 0.00059

        # Existing scalper-offer block follows — but use _base_exit_fee_pct instead
        # of hardcoded 0.00059:
        _exit_notional = actual_exit * trade.position_size * _cs
        _scalper_window = (30 * 60) if _is_btc_eth else (15 * 60)
        _scalper_eligible = _holding_sec <= _scalper_window
        if _scalper_eligible:
            exit_fee_usd = 0.0   # scalper offer waives, regardless of maker/taker
        else:
            exit_fee_usd = _exit_notional * _base_exit_fee_pct  # ← was hardcoded 0.00059
        # Stamp metadata
        meta_close = ut_meta or {}
        meta_close["exit_fee_type"] = _exit_fee_type
        meta_close["exit_was_maker"] = (_exit_fee_type == "maker")
```

---

## Apply procedure

1. **Pre-check**: Confirm baseline verdict has fired at 03:00 UTC and is on file:
   ```bash
   ls -la /home/opc/crypto-trading-bot/storage/verdicts/maker_ab_2026*
   ```
2. **Apply patches** to `user_real_manager.py` (both blocks above).
3. **Restart bot WITHOUT** the env var:
   ```bash
   ssh opc@VM "kill <PID> && sleep 4 && cd /home/opc/crypto-trading-bot && \
     nohup python3.13 main.py > /tmp/bot_<ts>.log 2>&1 & disown"
   ```
4. **Verify legacy path unchanged** by inspecting first 5 shadow trades — `fee_type` should still be `taker` (env var off).
5. **Enable simulator**:
   ```bash
   # In /home/opc/crypto-trading-bot/.env append:
   echo "SHADOW_MAKER_SIM_ENABLED=true" >> /home/opc/crypto-trading-bot/.env
   ```
6. **Restart again** to load the env var.
7. **Watch first 20 trades** — admin should show ~50-65% maker fills, niranjan ~25-50%.
8. If differential is <10pp after 50 trades, recalibrate `_BASE_P_BY_SYMBOL` in `shadow_maker_sim.py` against actual paper-side maker fill data.

## Rollback

```bash
# Disable instantly without restart:
sed -i 's/^SHADOW_MAKER_SIM_ENABLED=true/SHADOW_MAKER_SIM_ENABLED=false/' \
    /home/opc/crypto-trading-bot/.env
# Then restart bot. Or kill SHADOW_MAKER_SIM_ENABLED line entirely.
```

The patch is gated by env var — fully reversible without code changes.

## Expected impact (estimate)

Based on smoke-test calibration + last 24h's 138 admin trades:

| Scenario | Maker fill % | Net 24h |
|---|---|---|
| Current (taker only) | 0% | −$88 |
| With sim, admin (patient) | ~55-60% (BTC/ETH only — neither runs much SOL) | **−$45 to −$30** |
| With sim, niranjan (standard) | ~40-45% | −$60 to −$50 |
| **A/B delta admin-niranjan** | **+15pp** | **+$15 to +$25** in admin's favor |

If observed delta is in this range, the simulator is honoring patience mode correctly and we have ground truth for the live cutover decision. If admin matches niranjan, the calibration's wrong.

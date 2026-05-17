# PATCH: high_volatility hard-veto for structure_bounce — HELD FOR APPLY

**Status**: Drafted, not applied. Subtraction-only patch (no W/F needed — disables known-loser regime cohort).
**Apply gate**: Operator decision. Estimated impact +$25-35/24h.

---

## Problem (validated from 24h shadow data, 2026-05-01)

structure_bounce fires in **high_volatility** regimes where mean-reversion at S/R levels has no edge — levels get blown through, no bounce.

| Regime | n | WR | Σ Net | Avg/trade |
|---|---|---|---|---|
| sideways | 81 (59%) | 27% | −$43.52 | −$0.54 |
| **high_volatility** | **23** | **9%** | **−$37.06** | **−$1.61** ← worst per-trade |
| breakout | 7 | 14% | −$7.98 | −$1.14 |
| trending_up | 1 | 0% | −$1.78 | — |
| mean_reversion | 26 | 42% | −$0.91 | −$0.04 |

**high_volatility per-trade loss is 3× the sideways average.** structure_bounce should not fire in this regime.

The existing hard-veto list (`strategies/scalp_strategy.py:3225`):
```python
sb_hard_prefixes = ("CHOCH CONFLICT:", "REGIME MISMATCH:", "REGIME SIDE:", "STRICT 4H VETO:",
                    "AP_SB_SHORT_KILL:", "BAD_EDGE_SCANNER_KILL:", "DEAD_HOUR_KILL:",
                    "MACRO_EMA200_VETO:", "MACD_DIV_VETO:", "HTF_HARD_VETO_AGRADE:",
                    "VOLUME_CLIMAX_VETO:", "WEDGE_BREAKOUT_VETO:")
```

`HIGH_VOL_REGIME_VETO_SB:` is NOT in the list. There IS a `REGIME MISMATCH:` veto but it doesn't currently fire on high_volatility for structure_bounce (else 23 trades wouldn't have admitted).

---

## Patch

Two edits needed: (a) add the new hard-veto prefix to the list, (b) emit the veto when conditions match.

### Edit 1 — `strategies/scalp_strategy.py:~3225` (add prefix)

### Current
```python
        sb_hard_prefixes = ("CHOCH CONFLICT:", "REGIME MISMATCH:", "REGIME SIDE:", "STRICT 4H VETO:", "AP_SB_SHORT_KILL:", "BAD_EDGE_SCANNER_KILL:", "DEAD_HOUR_KILL:", "MACRO_EMA200_VETO:", "MACD_DIV_VETO:", "HTF_HARD_VETO_AGRADE:", "VOLUME_CLIMAX_VETO:", "WEDGE_BREAKOUT_VETO:")
```

### Proposed
```python
        # HIGH_VOL_VETO_SB_5_22 (2026-05-01) — added HIGH_VOL_REGIME_VETO_SB
        # to the hard-veto list. 23 high_volatility structure_bounce trades
        # in 24h cost -$37 (avg -$1.61/trade, 9% WR). Mean-reversion has no
        # edge when levels get blown through.
        sb_hard_prefixes = ("CHOCH CONFLICT:", "REGIME MISMATCH:", "REGIME SIDE:", "STRICT 4H VETO:", "AP_SB_SHORT_KILL:", "BAD_EDGE_SCANNER_KILL:", "DEAD_HOUR_KILL:", "MACRO_EMA200_VETO:", "MACD_DIV_VETO:", "HTF_HARD_VETO_AGRADE:", "VOLUME_CLIMAX_VETO:", "WEDGE_BREAKOUT_VETO:", "HIGH_VOL_REGIME_VETO_SB:")
```

### Edit 2 — `strategies/scalp_strategy.py` — emit the veto

Insert new veto check right after the `is_sb` flag is set (line ~3221):

```python
        is_sb = best_sr.scanner_name == "structure_bounce"

        # HIGH_VOL_VETO_SB_5_22 — block structure_bounce in high_volatility regime
        # Data 2026-05-01: 23 such trades = 9% WR, -$1.61 avg, -$37/24h.
        _regime_now = (locals().get('regime') or '').lower()
        if is_sb and _regime_now == "high_volatility":
            vetos.append(f"HIGH_VOL_REGIME_VETO_SB: structure_bounce blocked in high_volatility regime")
```

**Implementation note**: `regime` should already be in scope (from `_regime_filter.detect_regime()` at line ~1451). If not, fetch via `getattr(self, '_last_regime', '')` or similar. Check actual variable name at apply time.

---

## Apply procedure

1. Edit `strategies/scalp_strategy.py` per above (2 edits)
2. `python3 -m py_compile strategies/scalp_strategy.py`
3. Restart bot
4. Watch eval_trace / metadata for `HIGH_VOL_REGIME_VETO_SB:` veto firings
5. After 24h: high_volatility cohort trade count should drop from 23 → 0-2

## Rollback

```bash
ssh opc@VM "cp strategies/scalp_strategy.py.bak.<ts> strategies/scalp_strategy.py" && restart
```

## Composability

Composes cleanly with patches #1 (A+ size cap) and #3 (Asia-early veto). Order doesn't matter.

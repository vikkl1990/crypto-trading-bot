# PATCH: A+ size multiplier 1.3 → 1.0 — HELD FOR APPLY

**Status**: Drafted, not applied. Subtraction-only patch (no W/F needed — disables a known-loser amplifier).
**Apply gate**: Operator decision. Estimated impact +$25-35/24h.

---

## Problem (validated from 24h shadow data, 2026-05-01)

The bot's confidence-size multiplier amplifies size on cohorts that demonstrably lose money:

| Grade | ML band | n | WR | Σ Net | Avg/trade |
|---|---|---|---|---|---|
| **A+** | **0.80+** | **32** | **22%** | **−$38.43** | **−$1.20** |
| A+ | 0.70-0.79 | 34 | 26% | −$19.12 | −$0.56 |
| A+ | 0.60-0.69 | 14 | 21% | −$8.63 | −$0.62 |
| A+ | <0.60 | 10 | 30% | −$9.15 | −$0.92 |
| **A+ TOTAL** | (all) | **90** | **23%** | **−$75.33** |
| A | (all) | 14 | 36% | −$3.58 |
| B | (all) | 22 | 32% | −$10.34 |
| C | (all) | 12 | 17% | −$1.99 |

A+ is **65% of total 24h loss** ($75/$91). Higher confidence = worse performance. Calibration is anti-predictive at the top.

The size multiplier (`strategies/regime_filter.py:520-528`):
```python
if tier == "strong":
    if confidence >= 90: return 1.3   # A+ likely
    return 1.1                          # A
elif tier == "valid": return 1.0        # B
elif tier == "weak":  return 0.6        # C
return 0.5
```

So A+ trades get **1.3× size**, B trades get **1.0×**. The multiplier is amplifying losses on A+ by 30% over B-grade size.

**Note**: Avg margin in 24h data: A+ ≈ $45, B ≈ $34, C ≈ $19 — confirms 1.3× is being applied.

---

## Patch

### File: `strategies/regime_filter.py:520-528`

### Current
```python
def calc_confidence_size_multiplier(confidence: int, tier: str) -> float:
    """Scale position size based on signal confidence and tier.

    Strong signals (80+) get full or boosted size.
    Valid signals (65-79) get normal size.
    Weak signals (50-64) get reduced size.
    """
    if tier == "strong":
        if confidence >= 90:
            return 1.3  # exceptional signal
        return 1.1
    elif tier == "valid":
        return 1.0
    elif tier == "weak":
        return 0.6
    return 0.5  # near_miss or rejected shouldn't trade but just in case
```

### Proposed
```python
def calc_confidence_size_multiplier(confidence: int, tier: str) -> float:
    """Scale position size based on signal confidence and tier.

    Strong signals (80+) get full or boosted size.
    Valid signals (65-79) get normal size.
    Weak signals (50-64) get reduced size.
    """
    if tier == "strong":
        # AB_SIZE_CAP_5_22 (2026-05-01) — was 1.3 boost on conf>=90 (A+).
        # 24h shadow data: A+ x ML 0.80+ = 32 trades, 22% WR, -$1.20 avg.
        # ML calibration is anti-predictive at top. Capping all "strong"
        # at 1.0 removes the A+ amplifier without changing trade selection.
        # Estimated saving: $25-35/24h.
        return 1.0
    elif tier == "valid":
        return 1.0
    elif tier == "weak":
        return 0.6
    return 0.5  # near_miss or rejected shouldn't trade but just in case
```

**Conservative variant** (cap A+ but preserve A boost): change `if confidence >= 90: return 1.3` → `return 1.0` and keep the `return 1.1` for non-A+ "strong" tier. The proposed full-cap is more aggressive but matches the data.

---

## Apply procedure

1. Edit `strategies/regime_filter.py:520-528` per above
2. `python3 -m py_compile strategies/regime_filter.py`
3. Restart bot: `kill <pid> && nohup python3.13 main.py > /tmp/bot_<ts>.log 2>&1 & disown`
4. Watch first 24h: avg A+ margin should drop from ~$45 to ~$36 (matches B-grade)
5. Watch first 24h: A+ net P&L should improve proportionally to size reduction

## Rollback

```bash
ssh opc@VM "cp strategies/regime_filter.py.bak.<ts> strategies/regime_filter.py" && restart
```

## Composability

Composes cleanly with patches #2 (high_vol veto) and #3 (Asia-early veto). Order doesn't matter — each is an independent subtraction.

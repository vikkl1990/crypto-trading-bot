# Patch M — Trend Oracle (CM EMA Trend Bars adaptation)

**Status:** DESIGN-ONLY — no code shipped. Implementation deferred until either:
(a) absorption_bubbles + fvg_mtf_cascade W/F results land, OR
(b) architect explicitly approves implementation

**Sentinel:** PATCH_M_5_22
**Era stamp on activation:** `post_5_22_m`

---

## 1. Why this exists

Today's empirical evidence (5 W/F studies):

| Pattern | W/F Verdict | Failure mode |
|---|---|---|
| Demand Zone Retest | KILL (0/972) | Universal Q3 collapse |
| Break Block Reversal | KILL (0/972) | Universal Q3 collapse |
| bos_choch retest gate | KILL (0/432) | OOS gap > 100% on every IS-positive cell |
| Absorption Bubbles | TBD | (running) |
| FVG MTF Cascade | TBD | (running) |

3 of 3 retest patterns died with the SAME signature: Q3 (Feb 2026 chop) destroys what Q1/Q2 (Q4 trending) validated. Per Agent 1's diagnosis: *"Two converging studies = true regime collapse, not a tuning miss."*

**The bot is over-fit to trending regimes. Every SMC pattern works in trends, dies in chop.** What's missing isn't more patterns — it's a regime oracle that prevents trend-following patterns from running in chop in the first place.

The user identified this independently via their CM EMA Trend Bars message: *"a custom Pine Script indicator that plots colored bars/trend lines based on multiple EMAs (likely EMA 8/21/50/200) that change color based on trend strength and direction."* That's exactly the gate we need.

## 2. What it does

A small in-process module exposing one function:

```python
get_trend_state(symbol, timeframe="1h") -> TrendState
```

Returns a dataclass:

```python
@dataclass
class TrendState:
    color: str                # "green" | "red" | "neutral"
    strength: float           # 0.0 to 1.0
    bars_in_state: int        # how long current color held
    aligned_emas: int         # how many EMAs are stacked correctly (0-4)
    last_flip_ts: pd.Timestamp
    is_strong_trend: bool     # color != neutral AND strength >= 0.6
```

Rules (CM EMA Bars logic, adapted):
- Compute EMA 8, 21, 50, 200 on the requested timeframe
- **Green** when: EMA8 > EMA21 > EMA50 > EMA200 AND price > EMA8
- **Red** when: EMA8 < EMA21 < EMA50 < EMA200 AND price < EMA8
- **Neutral** otherwise (EMAs not fully stacked = no clear trend)
- **Strength** = fraction of last N bars (default 20) that held the same color

## 3. Where it gates

Per Patch JK lesson (qualify_signal bypass in shadow_live), the trend oracle MUST be checked in BOTH places:

### 3a. scalp_strategy.py qualify_signal
Inserted BEFORE existing veto checks. Same prefix-based hard kill pattern as PATCH_J/K:

```python
# PATCH_M veto — trend oracle
try:
    if _patch_m_trend is not None:
        ts = _patch_m_trend.get_trend_state(symbol, timeframe="1h")
        side = best_sr.side.value
        if not _patch_m_trend_aligned(ts, side):
            vetos.append(f"TREND_ORACLE_M: {symbol} 1h={ts.color} strength={ts.strength:.2f} vs {side} — misaligned")
except Exception:
    pass
```

Add `"TREND_ORACLE_M:"` to `sb_hard_prefixes` tuple.

### 3b. user_real_manager.py _execute_shadow
Same gate, called before any other Patch JK gates. So shadow trades coming from paper-mirror OR from the Patch L bridge BOTH pass through this filter.

### 3c. shadow_signal_consumer.py (Patch L bridge)
The consumer's `_check_breaker(source_engine)` extends to also call trend oracle:
```python
def _check_trend_alignment(symbol: str, side: str) -> Optional[str]:
    """Returns veto reason if trend not aligned with side, else None."""
```

This means **bridged signals from W/F-validated paper engines also get gated by the trend oracle** — preventing them from running in chop.

## 4. Per-scanner / per-engine alignment policy

Not all scanners want the same gate behavior:

| Scanner / Engine | Mode | Why |
|---|---|---|
| structure_bounce | **REQUIRE alignment** (trend OR neutral, not opposite) | Trend-following scanner, dies in chop |
| liq_grab_ob_fvg | **REQUIRE alignment** | SMC pattern, needs trending regime |
| liq_sweep_htf | **REQUIRE alignment** | Same |
| smc1, smc15, smc15v2 | **REQUIRE alignment** | Same |
| chop_vwap_mr (if it passes W/F) | **REQUIRE neutral or counter-trend** | Mean-revert wants chop |
| absorption_bubbles (if it passes) | **REQUIRE counter-trend OR neutral** | Reversal pattern wants exhaustion |

Configurable via `bot/trend_oracle.py` constant:

```python
TREND_ALIGNMENT_POLICY = {
    "structure_bounce":     "ALIGNED",       # green for long, red for short
    "liq_grab_ob_fvg":      "ALIGNED",
    "liq_sweep_htf":        "ALIGNED",
    "smc1":                 "ALIGNED",
    "smc15":                "ALIGNED",
    "smc15v2":              "ALIGNED",
    "chop_vwap_mr":         "NEUTRAL",        # only fires when EMAs not stacked
    "absorption_bubbles":   "COUNTER_OR_NEUTRAL",  # reversal play
}
```

## 5. Module structure (mirrors existing pattern from regime_gate.py)

```
bot/trend_oracle.py
├── TrendState dataclass
├── _ema(arr, period) → np.ndarray (vectorized)
├── _color_from_emas(price, e8, e21, e50, e200) → "green"|"red"|"neutral"
├── class TrendOracle:
│   ├── _cache: dict[(symbol, tf) -> (mtime, TrendState)]
│   ├── _compute(symbol, tf) → TrendState   (loads parquet + computes)
│   ├── get_trend_state(symbol, tf="1h") → TrendState
│   ├── is_aligned(symbol, side, tf="1h", policy="ALIGNED") → bool
│   └── stats() → dict (for dashboards)
├── get_trend_oracle() → singleton
└── TREND_ALIGNMENT_POLICY dict (per-scanner config)
```

Cache TTL: 1 hour (1h candles update every hour, so 1h cache is conservative).
Data source: `storage/candle_cache/{SYM}_USDT_1h.parquet` (already exists, used by regime_gate).

## 6. Implementation phases

### Phase M.1 — Build module + smoke test (1h)
- Write `bot/trend_oracle.py` (~150 lines)
- Smoke-test in REPL: `get_trend_state("BTC/USDT", "1h")` returns expected color
- Verify against TradingView CM EMA Bars on BTC/USDT 1h chart

### Phase M.2 — Wire into shadow path (1h)
- Patch `_execute_shadow` to gate via `is_aligned(symbol, side, scanner_name)`
- Patch `shadow_signal_consumer._check_breaker` to also call trend oracle
- Default policy: SOFT VETO (log but don't block) for first 24h to gather data

### Phase M.3 — Validate then enforce (after 24h soak)
- Compare WR with/without trend gate via cohort analysis
- If trend gate would have blocked X% of losses without sacrificing winners → flip to HARD VETO
- Update sb_hard_prefixes to include `"TREND_ORACLE_M:"`

### Phase M.4 — Re-W/F killed patterns WITH trend gate (4-6h, post-soak)
- Re-run Demand Zone Retest, Break Block Reversal, bos_choch retest with trend gate as precondition
- Hypothesis: cells that KILLED universally will now PASS in trending segments
- If true: revive these patterns gated by trend oracle, ship as new paper engines via Patch L bridge

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| EMA-based oracle has lag (200-period EMA on 1h = 200h = 8 days lookback) | Use shorter EMAs (8/21/50) as primary, EMA200 as TREND filter only |
| Cache staleness if 1h candle parquet sync breaks | Add max_age check; fall back to "neutral" if data > 2h old |
| False neutrals during early-trend transition | Phase M.2 ships as SOFT VETO for 24h to measure FP rate before enforcement |
| Scanner-policy config drift | Centralize in single `TREND_ALIGNMENT_POLICY` dict in `bot/trend_oracle.py` |
| Breaks Patch L bridge | Add try/except in consumer call, fail-open default |

## 8. Verification gates (architect discipline)

Before shipping any phase:

| Phase | Verification gate |
|---|---|
| M.1 | Smoke test produces sensible colors for BTC, ETH, SOL, XRP on 1h. Compare visually to TradingView indicator. |
| M.2 | Restart bot, observe `TREND_ORACLE_M: ...` log lines fire on next signal. Verify SOFT VETO mode doesn't block (logs only). |
| M.3 | Cohort analysis report shows >70% precision on blocking losers (vs blocking too many winners) before flipping to HARD. |
| M.4 | W/F results show clear PASS for ≥1 of the 3 previously-killed patterns when gated by trend oracle. |

## 9. Out of scope

- ❌ Real-time tick-by-tick trend updates (1h cache is sufficient)
- ❌ Multi-timeframe trend confluence (1h is the gate; LTF is the entry)
- ❌ Adaptive EMAs (Pine Script CM EMA Bars uses fixed lengths — match it)
- ❌ ML-based trend detection (keep simple, deterministic)

## 10. Effort estimate

| Phase | Wall time | Files touched |
|---|---|---|
| M.1 | 1h | 1 new file |
| M.2 | 1h | 2 patched (scalp_strategy, user_real_manager, shadow_signal_consumer) |
| M.3 | 24h soak + 1h analysis | 1 sed-style flip on policy |
| M.4 | 4-6h W/F | 3 study scripts re-run |
| **Total** | **2-3 days end-to-end** | **6 files** |

## 11. Decision gates for the architect

Before starting Phase M.1, confirm:

1. **Do we want 1h as the trend timeframe?** (Alternative: 4h for slower trends, less noise)
2. **EMA lengths: 8/21/50/200 (CM EMA default)?** Or shorter set for crypto?
3. **Default policy for unknown scanners: ALIGNED, NEUTRAL, or BYPASS?** (Recommend ALIGNED — most strategies want trend bias.)
4. **SOFT VETO for 24h before HARD?** Or trust the design and HARD-veto from M.2?

Recommendation: 1h timeframe, 8/21/50/200, default ALIGNED, SOFT for 24h.

---

## Appendix — Code skeleton (Phase M.1)

```python
# bot/trend_oracle.py
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

CACHE_DIR = Path("/home/opc/crypto-trading-bot/storage/candle_cache")
CACHE_TTL_SEC = 3600  # 1h candles update hourly
EMA_PERIODS = (8, 21, 50, 200)
STRENGTH_LOOKBACK = 20


@dataclass
class TrendState:
    color: str             # "green" | "red" | "neutral"
    strength: float        # 0..1
    bars_in_state: int
    aligned_emas: int      # 0..4
    last_flip_ts: Optional[pd.Timestamp]
    is_strong_trend: bool

    @classmethod
    def neutral(cls):
        return cls("neutral", 0.0, 0, 0, None, False)


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values


def _color_from_emas(price, e8, e21, e50, e200) -> Tuple[str, int]:
    """Return (color, aligned_emas_count)."""
    if np.isnan(e8) or np.isnan(e21) or np.isnan(e50) or np.isnan(e200):
        return "neutral", 0
    if price > e8 > e21 > e50 > e200:
        return "green", 4
    if price < e8 < e21 < e50 < e200:
        return "red", 4
    # Partial alignment
    aligned = sum([
        e8 > e21, e21 > e50, e50 > e200,
    ])
    if aligned >= 3 and price > e8:
        return "green", aligned
    aligned_down = sum([
        e8 < e21, e21 < e50, e50 < e200,
    ])
    if aligned_down >= 3 and price < e8:
        return "red", aligned_down
    return "neutral", max(aligned, aligned_down)


class TrendOracle:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], Tuple[float, TrendState]] = {}

    def _compute(self, symbol: str, tf: str) -> TrendState:
        sym_clean = symbol.replace("/", "_")
        path = CACHE_DIR / f"{sym_clean}_{tf}.parquet"
        if not path.exists():
            return TrendState.neutral()
        df = pd.read_parquet(path).sort_index().tail(300)
        if len(df) < EMA_PERIODS[-1] + 5:
            return TrendState.neutral()
        c = df["close"].astype(float).values
        e8 = _ema(c, 8); e21 = _ema(c, 21); e50 = _ema(c, 50); e200 = _ema(c, 200)
        # Per-bar color (vectorized for strength calc)
        colors = []
        for i in range(len(c)):
            color, _ = _color_from_emas(c[i], e8[i], e21[i], e50[i], e200[i])
            colors.append(color)
        cur_color, cur_aligned = _color_from_emas(c[-1], e8[-1], e21[-1], e50[-1], e200[-1])
        # Strength = fraction of last STRENGTH_LOOKBACK bars matching cur_color
        recent = colors[-STRENGTH_LOOKBACK:]
        strength = recent.count(cur_color) / max(1, len(recent))
        # Bars in current state
        bars_in_state = 0
        for cc in reversed(colors):
            if cc == cur_color:
                bars_in_state += 1
            else:
                break
        last_flip_ts = None
        if bars_in_state < len(df):
            try:
                last_flip_ts = df.index[-bars_in_state]
            except Exception:
                pass
        return TrendState(
            color=cur_color,
            strength=round(strength, 3),
            bars_in_state=bars_in_state,
            aligned_emas=cur_aligned,
            last_flip_ts=last_flip_ts,
            is_strong_trend=(cur_color != "neutral" and strength >= 0.6),
        )

    def get_trend_state(self, symbol: str, tf: str = "1h") -> TrendState:
        now = time.time()
        with self._lock:
            cached = self._cache.get((symbol, tf))
            if cached and (now - cached[0]) < CACHE_TTL_SEC:
                return cached[1]
        ts = self._compute(symbol, tf)
        with self._lock:
            self._cache[(symbol, tf)] = (now, ts)
        return ts

    def is_aligned(self, symbol: str, side: str, scanner: str = "",
                   tf: str = "1h") -> Tuple[bool, str]:
        """Return (allowed, veto_reason). veto_reason is empty if allowed."""
        policy = TREND_ALIGNMENT_POLICY.get(scanner, "ALIGNED")
        if policy == "BYPASS":
            return True, ""
        ts = self.get_trend_state(symbol, tf)
        side_l = side.lower()
        if policy == "ALIGNED":
            if side_l == "long" and ts.color == "green":
                return True, ""
            if side_l == "short" and ts.color == "red":
                return True, ""
            return False, (f"TREND_ORACLE_M: {symbol} 1h={ts.color} "
                           f"strength={ts.strength:.2f} vs {side_l} — misaligned")
        if policy == "NEUTRAL":
            if ts.color == "neutral" or not ts.is_strong_trend:
                return True, ""
            return False, (f"TREND_ORACLE_M: {symbol} 1h={ts.color} "
                           f"strength={ts.strength:.2f} — too trendy for chop strategy")
        if policy == "COUNTER_OR_NEUTRAL":
            if ts.color == "neutral":
                return True, ""
            if side_l == "long" and ts.color == "red" and ts.is_strong_trend:
                return True, ""
            if side_l == "short" and ts.color == "green" and ts.is_strong_trend:
                return True, ""
            return False, (f"TREND_ORACLE_M: {symbol} 1h={ts.color} "
                           f"strength={ts.strength:.2f} vs {side_l} — not counter-trend reversal")
        return True, ""


TREND_ALIGNMENT_POLICY: Dict[str, str] = {
    "structure_bounce":     "ALIGNED",
    "liq_grab_ob_fvg":      "ALIGNED",
    "liq_sweep_htf":        "ALIGNED",
    "smc1":                 "ALIGNED",
    "smc15":                "ALIGNED",
    "smc15v2":              "ALIGNED",
    "chop_vwap_mr":         "NEUTRAL",
    "absorption_bubbles":   "COUNTER_OR_NEUTRAL",
    "fvg_mtf_cascade":      "ALIGNED",
}


_ORACLE: Optional[TrendOracle] = None


def get_trend_oracle() -> TrendOracle:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = TrendOracle()
    return _ORACLE
```

---

**End of Patch M design doc. ~250 lines code skeleton. Ready for architect review + Phase M.1 implementation when approved.**

---

## ADDENDUM (2026-05-02 ~15:35 UTC) — TREND_ALIGNMENT_POLICY refinement

After today's W/F closure, the FVG MTF Cascade study surfaced a critical input:

> **htf_align effect (FVG MTF W/F):** True: 7/357 PASS (2.0%). False: 10/492 PASS (2.0%). **Identical PASS rate** — the 1h FVG existence is already an implicit trend filter; EMA21 slope adds no lift.

**Implication:** Adding an EMA-based trend gate on top of patterns that ALREADY have implicit trend filters provides ZERO lift. The Patch M trend oracle's value is in filtering REGIME-NAIVE scanners, not double-gating regime-aware ones.

**Revised TREND_ALIGNMENT_POLICY (replaces v1):**

| Scanner / Engine | v1 Policy | v2 Policy (post-FVG-MTF finding) | Reason |
|---|---|---|---|
| structure_bounce | ALIGNED | **ALIGNED** | Regime-naive — needs trend gate (this is the target) |
| liq_grab_ob_fvg | ALIGNED | **BYPASS** | Already self-gates via 5-step OB+FVG sequence |
| liq_sweep_htf | ALIGNED | **BYPASS** | HTF gate is in the name (4h ATR pct rank ≤ 0.6) |
| smc1, smc15, smc15v2 | ALIGNED | **BYPASS** | Multi-step SMC sequence self-gates |
| fvg_mtf_cascade | ALIGNED | **BYPASS** | 1h FVG presence IS the trend filter (W/F-proven) |
| absorption_bubbles | COUNTER_OR_NEUTRAL | **COUNTER_OR_NEUTRAL** | Reversal — gate against strong trend (don't fade strong moves) |
| chop_vwap_mr | NEUTRAL | (KILLED today, irrelevant) | — |

**Net effect:** Patch M becomes a LASER-FOCUSED filter on  (the bot's monoculture), not a blanket gate that double-filters everything. This dramatically reduces risk of over-blocking and matches the empirical finding.

## ADDENDUM 2 (2026-05-02 ~15:35 UTC) — Today's W/F closure context

| Study | Verdict | Implication for Patch M |
|---|---|---|
| Demand Zone Retest (KILL) | Q3 collapse | Re-test post-Patch M as gated pattern |
| Break Block Reversal (KILL) | Q3 collapse | Re-test post-Patch M (especially asymmetric SHORT variant) |
| bos_choch retest gate (KILL) | Q3 collapse | Re-test post-Patch M (or retire concept) |
| Absorption Bubbles (PASS ETH) | Reversal works in chop | Patch M policy: COUNTER — protect from strong trend |
| FVG MTF Cascade (PARKED) | Self-gates via 1h FVG | BYPASS Patch M — adding EMA gate proven to add zero lift |
| chop_vwap_mr (KILL) | Mean-revert variant doesn't work | — |

**Architect's read:** Patch M is the foundational filter that lets KILL'd patterns become viable (when re-tested as regime-gated) without slowing down the patterns that already work. The empirical case strengthens.

---

## ADDENDUM v2 (2026-05-03 ~02:55 UTC) — Phase Filter as second dimension

User-surfaced concept: **HTF candle phase filter** (expansion vs contraction)
as a separate signal-quality gate. After review, this is COMPLEMENTARY to the
EMA-based trend oracle (v1), not a replacement.

### Why both dimensions matter

| Dimension | What it answers | Mechanism | Lag | Hostile case |
|---|---|---|---|---|
| **Trend Oracle (v1)** | "Is HTF trending UP / DOWN / sideways?" | Multi-EMA stacking (8/21/50/200) | Some (smoothed) | Counter-trend entries |
| **Phase Filter (v2)** | "Is current HTF bar EXPANDING / CONTRACTING?" | Range / progress-adjusted ATR | None (current bar) | Entries inside dead chop bar |

A signal could pass BOTH for highest conviction.

### Counter-evidence to consider (FVG MTF Cascade W/F finding)

> htf_align effect (FVG MTF W/F): True 7/357 PASS (2.0%), False 10/492 PASS (2.0%) — **identical**.
> EMA-slope alignment added zero lift on top of FVG-implicit trend filter.

**Implication:** EMA-based trend filter alone may not add value where the pattern
ALREADY has implicit MTF context. Phase filter measures something STRUCTURALLY DIFFERENT
(volatility expansion, not trend direction) — so it MAY add value where EMA wouldn't.
Need W/F validation before claiming the lift.

### Unified HTF Context Oracle (v2 architecture)

Replace the v1 single-purpose `TrendOracle` with a multi-dimension `HTFContextOracle`:

```python
@dataclass
class HTFContext:
    # Trend dimension (v1)
    color: str                        # "green" | "red" | "neutral"
    trend_strength: float             # 0..1
    aligned_emas: int                 # 0..4
    bars_in_trend: int

    # Phase dimension (v2 — NEW)
    phase: str                        # "expansion" | "contraction" | "transition"
    expansion_ratio: float            # current_range / (progress × atr)
    bar_progress: float               # 0..1 = fraction of current 4h bar elapsed

    # Convenience flags
    is_strong_trend: bool             # color != neutral AND strength >= 0.6
    is_expanding: bool                # phase == "expansion"
    is_aligned_and_expanding: bool    # both conditions for highest-conviction entries
```

### Phase Filter algorithm

```python
def compute_phase(symbol, tf="4h") -> tuple[str, float]:
    """Returns (phase_label, expansion_ratio)."""
    df = load_candles(symbol, tf, lookback=20)
    cur_bar = df.iloc[-1]
    cur_atr = atr(df.iloc[:-1], 14).iloc[-1]   # ATR from CLOSED bars only
    bar_open_ts = df.index[-1]
    age_hours = (now_utc() - bar_open_ts).total_seconds() / 3600.0
    bar_duration_hours = 4.0  # for 4h bars
    progress = min(age_hours, bar_duration_hours) / bar_duration_hours
    cur_range = cur_bar.high - cur_bar.low
    expected_range = progress * cur_atr
    if expected_range <= 0:
        return "transition", 0.0
    ratio = cur_range / expected_range

    if ratio > 1.3:
        return "expansion", ratio
    elif ratio < 0.6:
        return "contraction", ratio
    else:
        return "transition", ratio
```

### Per-scanner policy v2 (refined)

| Scanner | v1 policy (Trend Oracle) | v2 addition (Phase Filter) |
|---|---|---|
| structure_bounce | ALIGNED | + REQUIRE phase=expansion |
| liq_grab_ob_fvg | BYPASS (self-gates) | BYPASS (self-gates) |
| liq_sweep_htf | BYPASS | BYPASS |
| smc1, smc15, smc15v2 | BYPASS | BYPASS |
| absorption_bubbles | COUNTER_OR_NEUTRAL | + REQUIRE phase=contraction (reversal at exhaustion) |
| chop_vwap_mr | NEUTRAL | + REQUIRE phase=contraction (mean revert needs chop) |
| fvg_mtf_cascade | BYPASS | BYPASS |

### Verification gates added for v2

| Phase | Gate |
|---|---|
| M.0 (design — done) | Document the dual-dimension architecture |
| M.1 (build) | Smoke test phase computation against TradingView 4h chart for BTC/ETH |
| M.2 (W/F validate) | Run "structure_bounce + phase=expansion" study before wiring as live veto |
| M.3 (soft veto soak 24h) | Measure FP rate (entries blocked that would have won) |
| M.4 (hard veto) | Flip after FP rate < 30% |

### Effort estimate revision

| Phase | v1 (Trend Oracle only) | v2 (Trend + Phase) |
|---|---|---|
| Design | 1-2h ✅ done | + 1h (this addendum) ✅ done |
| Build | 1h | 1.5h (added compute_phase + dataclass field) |
| W/F validate | n/a | 30-60 min agent run BEFORE deploy |
| Wire (soft) | 1h | 1.5h (per-scanner phase policy) |
| Soak 24h + analyze | 24h | 24h |
| Hard veto rollout | 1h | 1h |
| **Total** | ~3h work + 24h soak | ~5h work + 24h soak |

### Key risks (v2)

| Risk | Mitigation |
|---|---|
| Phase computation depends on accurate "current 4h bar age" | Use bar's open timestamp from candle_cache; compute age = now - open |
| Edge case: bar just opened (progress < 0.1), no signal possible yet | Skip phase check if progress < 0.1; default to "transition" (allow) |
| Phase oracle stale if 4h candle parquet sync breaks | Add max_age check (default 30 min); fall back to "transition" |
| Hard-veto deadlock (lessons from Patch JK!) | NO lazy-seed from DB; compute LIVE every call; no permanent state |

**Critical guardrail:** Patch M v2 must NOT have any persistent "tripped" state.
Every `is_aligned()` call computes from candle cache + current time. No Patch JK-style deadlock possible.

### Decision gates for the architect (revised for v2)

Before starting Phase M.1 build:

1. **Build v2 (Trend + Phase) or v1 only first?** Recommend v2 — extra effort is small.
2. **Phase thresholds: 1.3/0.6 (recommended), or stricter?** Run W/F to tune.
3. **Run W/F study FIRST?** YES — don't build before validating. (This is the lesson from today.)

---

**End of v2 addendum. Patch M now has dual-dimension HTF Context Oracle as the design.**

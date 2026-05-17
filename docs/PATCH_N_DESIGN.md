# Patch N — SMC Reversal Sequence State Machine

**Status:** DESIGN-ONLY — no code shipped. Implementation deferred until:
(a) Patch M (Trend Oracle) lands as the regime gate prerequisite
(b) 3 in-flight W/F agents return their verdicts
(c) Architect explicit approval for Phase N.1

**Sentinel:** PATCH_N_5_22
**Era stamp on activation:** `post_5_22_n`

---

## 1. Why this exists

Today's empirical evidence:

| Single-pattern W/F | Verdict |
|---|---|
| Demand Zone Retest | KILL (universal Q3 collapse) |
| Break Block Reversal | KILL (universal Q3 collapse) |
| bos_choch retest gate | KILL (OOS gap >100%) |
| Absorption Bubbles | TBD (running) |
| FVG MTF Cascade | TBD (running) |

The architect-grade insight, formalized:

> **Single-bar SMC pattern detection is at diminishing returns. The leverage is multi-bar SEQUENCE detection.**

The full SMC reversal narrative ("End of Downtrend" image) is a **6-step sequence over many bars**, not a single setup. Testing each step independently gives misleading results — the steps are only meaningful in order, and ALL together form the high-probability reversal.

**Patch N implements the sequence as a state machine** that watches each symbol bar-by-bar, advances state only when each step's criteria are met, and publishes a single signal when the full narrative completes.

## 2. The 6-step SMC Reversal sequence

From the architect's reference (Real Market Edge "End of Downtrend"):

```
[Downtrend in progress] → [LIQUIDITY SWEEP] → [CHoCH] → [BOS] → [POI RETEST] → [FIB CONFIRMATION] → [ENTRY/RALLY]
```

**For LONG entries (downtrend → uptrend flip); mirror for SHORT (uptrend → downtrend flip):**

### Step 0 — Pre-condition: established downtrend
- 1H Trend Oracle (Patch M) returns `color="red"` AND `bars_in_state ≥ 30` (mature downtrend, not just a pullback)
- This is the regime gate — sequence only starts in a mature trend

### Step 1 — Range / Equal Lows formation
- Last 20-50 5m bars contain ≥2 lows within `equal_low_atr` × ATR (default 0.3)
- These are the resting liquidity zones (stop clusters)
- State: `IDLE` → `EQUAL_LOWS_DETECTED`

### Step 2 — Liquidity Sweep
- A 5m bar's low penetrates the equal-lows zone by ≥ `sweep_atr` × ATR (default 0.2)
- Wick ≥ `wick_ratio` × bar range (default 0.4) — confirms sweep was rejected
- State: `EQUAL_LOWS_DETECTED` → `SWEPT`
- Timeout: if no CHoCH within `swept_window_bars` (default 8 bars), revert to `IDLE`

### Step 3 — CHoCH (Change of Character)
- A 5m bar closes above the most recent prior swing high (the "internal swing" from before the sweep)
- Body ≥ `choch_body_atr` × ATR (default 0.5)
- State: `SWEPT` → `CHOCH_CONFIRMED`
- Record `choch_close_price` for retest reference
- Timeout: if no BOS within `choch_window_bars` (default 12 bars), revert to `IDLE`

### Step 4 — BOS (Break of Structure)
- A subsequent 5m bar closes above the NEXT swing high after CHoCH
- Confirms the new bullish structure (multiple BOS = stronger confirmation, optional)
- State: `CHOCH_CONFIRMED` → `BOS_CONFIRMED`
- Record `bos_close_price`, `poi_zone` (the OB that was reclaimed during CHoCH-to-BOS run)
- Timeout: if no retest within `bos_window_bars` (default 20 bars), revert to `IDLE`

### Step 5 — POI Retest
- Price returns into `poi_zone` (the proximal OB or reclaimed level identified in step 4)
- Defined as: `low ≤ poi_high AND high ≥ poi_low`
- State: `BOS_CONFIRMED` → `POI_RETESTED`
- Compute Fibonacci retracement levels:
  - `fib_anchor_low` = sweep_bar low
  - `fib_anchor_high` = bos_close_price
  - Levels: 0.62, 0.70, 0.79 retrace
- Timeout: if no Fib + confirmation within `retest_window_bars` (default 10), revert to `IDLE`

### Step 6 — Fib Confirmation + Entry
- Current bar is in the Fib zone (price between 0.62 and 0.79 retrace levels)
- Bullish confirmation candle: close > open, body ≥ `confirm_body_atr` × ATR (default 0.5)
- (Optional) volume confluence: rel_vol ≥ 1.3
- State: `POI_RETESTED` → `ARMED`
- Publish signal via `bot/shadow_bridge.publish_signal()`:
  - `entry_price` = confirmation candle close
  - `stop_loss` = sweep low − 0.3 × ATR
  - `take_profit` = `bos_close_price + (bos_close_price − stop_loss)` (1:R based on swing)
  - `source_engine` = "smc_reversal_sequence"
  - `extra_meta` = full sequence trace (sweep_ts, choch_ts, bos_ts, poi_zone, fib_levels)
- After publish: state → `IDLE`, sequence reset

## 3. State machine diagram

```
                    Trend Oracle (Patch M) says: 1h trend is RED, bars_in_state >= 30
                                              │
                                              ▼
                                          [IDLE]
                                              │
                              detect 2+ lows within equal_low_atr × ATR
                                              │
                                              ▼
                                  [EQUAL_LOWS_DETECTED]
                                              │
                              sweep candle penetrates equal lows + reject wick
                                              │
                                              ▼
                                          [SWEPT]
                                  ┌──────────┼──────────┐
                                  │          │
                              timeout    CHoCH bar closes above prior swing high
                                  │          │
                                  ▼          ▼
                              [IDLE]   [CHOCH_CONFIRMED]
                                              │
                                  ┌──────────┼──────────┐
                                  │          │
                              timeout    BOS bar closes above next swing high
                                  │          │
                                  ▼          ▼
                              [IDLE]   [BOS_CONFIRMED]
                                              │
                                  ┌──────────┼──────────┐
                                  │          │
                              timeout    price retraces into POI zone
                                  │          │
                                  ▼          ▼
                              [IDLE]   [POI_RETESTED]
                                              │
                                  ┌──────────┼──────────┐
                                  │          │
                              timeout    fib_zone + confirmation candle
                                  │          │
                                  ▼          ▼
                              [IDLE]      [ARMED]
                                              │
                                  publish_signal() → shadow_bridge → shadow trade
                                              │
                                              ▼
                                          [IDLE]
```

## 4. Module structure (mirrors regime_gate.py / circuit_breaker.py / trend_oracle.py)

```
bot/sequence_oracle.py
├── SequenceState dataclass (immutable per-tick snapshot)
├── SequenceContext dataclass (mutable per-symbol state)
├── EqualLowsDetector  (vectorized helper)
├── SwingHighFinder    (vectorized helper)
├── PoiZoneCalculator  (uses last bullish OB before BOS)
├── FibLevelCalculator (0.62 / 0.7 / 0.79 retrace bands)
├── class SMCReversalStateMachine:
│   ├── _states: dict[(symbol, side) -> SequenceContext]
│   ├── on_5m_close(symbol, df_5m, atr_5m) → Optional[SignalEnvelope]
│   ├── _step_idle_to_equal_lows(...)
│   ├── _step_equal_lows_to_swept(...)
│   ├── _step_swept_to_choch(...)
│   ├── _step_choch_to_bos(...)
│   ├── _step_bos_to_retest(...)
│   ├── _step_retest_to_armed(...)
│   ├── _emit_signal(ctx) → publish_signal(...)
│   └── stats() → dict (per-symbol state, last transitions)
├── _trend_oracle_gate(symbol, side, mode="strict") → bool
└── get_smc_state_machine() → singleton

storage/sequence_oracle/
├── state_snapshot.json  (persisted per-symbol state for restart safety)
└── transitions.jsonl    (audit log of every state change)
```

## 5. Integration points

### 5a. Trigger: bar-close hook in main bot
Add a one-line call inside the bot's existing 5m candle-close handler:

```python
# In bot/orchestrator.py or wherever 5m close fires
try:
    from bot.sequence_oracle import get_smc_state_machine
    sm = get_smc_state_machine()
    sm.on_5m_close(symbol, df_5m_recent, atr_5m)
except Exception as e:
    self._log.debug("sequence_oracle.on_5m_close failed: %s", e)
```

State machine processes one symbol per call. Async-safe (per-symbol locks).

### 5b. Output: signals via Patch L bridge
When state reaches `ARMED`, the state machine calls:

```python
publish_signal(
    source_engine="smc_reversal_sequence",
    symbol=symbol, side=side,
    entry_price=entry, stop_loss=sl, take_profit=tp,
    ml_probability=0.75,           # high prior given full sequence completion
    grade="A+",                     # full SMC = highest grade
    setup_type="smc_reversal",
    confidence=92.0,
    regime="reversal",
    extra_meta={
        "sequence_trace": ctx.to_audit_dict(),  # full 6-step audit
        "fib_zone": [fib_062, fib_079],
        "trend_oracle_at_arm": trend_oracle.get_trend_state(symbol).color,
    },
)
```

### 5c. Trend Oracle gate
Before any state transition, check trend oracle:

```python
def _can_advance(self, symbol: str, side: str) -> bool:
    """Don't advance state if regime conditions don't support it."""
    oracle = get_trend_oracle()
    ts = oracle.get_trend_state(symbol, "1h")
    if side == "long":
        # LONG sequence requires mature DOWNTREND that's flipping
        return ts.color == "red" and ts.bars_in_state >= 30
    else:
        return ts.color == "green" and ts.bars_in_state >= 30
```

If `_can_advance` returns False at ANY step, sequence resets to IDLE. This means **the state machine only fires when a real regime flip is happening** — exactly the gate that's been missing.

### 5d. Patch JK circuit breaker integration
The bridged signal goes through `_check_breaker("smc_reversal_sequence")` per Patch L mode B (own per-engine breaker). If 20 consecutive sequence completions show < 40% WR, breaker pauses for 60min. Self-protecting.

## 6. Persistence (restart safety)

State machine maintains in-memory state, but persists snapshot to disk every minute:

```python
storage/sequence_oracle/state_snapshot.json
{
  "BTC/USDT_long": {
    "state": "BOS_CONFIRMED",
    "sweep_bar_idx": 12345,
    "sweep_low_price": 78050.0,
    "choch_close_price": 78250.0,
    "bos_close_price": 78400.0,
    "poi_zone": [78100.0, 78180.0],
    "bars_since_state_change": 7,
    "side": "long",
    "ts_iso": "2026-05-02T15:30:00+00:00"
  },
  ...
}
```

On bot restart, restore state from disk → no false IDLE resets that would miss in-progress sequences.

Also write append-only audit log:

```python
storage/sequence_oracle/transitions.jsonl
{"ts":"...", "symbol":"BTC/USDT", "side":"long", "from":"SWEPT", "to":"CHOCH_CONFIRMED", "reason":"close 78250 > prior_swing_high 78230"}
```

This audit trail is invaluable for debugging WHY the state machine did or didn't fire on a given setup.

## 7. Configuration / tunables

```python
# bot/sequence_oracle.py
DEFAULTS = {
    "equal_low_atr": 0.3,
    "sweep_atr": 0.2,
    "wick_ratio": 0.4,
    "choch_body_atr": 0.5,
    "confirm_body_atr": 0.5,
    "swept_window_bars": 8,
    "choch_window_bars": 12,
    "bos_window_bars": 20,
    "retest_window_bars": 10,
    "trend_oracle_min_bars_in_state": 30,
    "min_atr_for_sweep_usd": 1.0,   # skip illiquid pairs
}

# Per-symbol overrides (e.g. SOL needs tighter sweep_atr due to noise)
SYMBOL_OVERRIDES = {
    "SOL/USDT": {"sweep_atr": 0.15, "equal_low_atr": 0.25},
    "XRP/USDT": {"sweep_atr": 0.25, "equal_low_atr": 0.35},
}
```

## 8. Verification gates (architect discipline)

Before shipping any phase:

| Phase | Verification gate |
|---|---|
| N.1 | State machine in dry-run mode for 24h. Audit `transitions.jsonl` to confirm states advance correctly. ZERO publish_signal calls in dry-run. |
| N.2 | Verify state machine produces ≥ 3 ARMED states across all 4 pairs in 24h. If 0 → params too tight; if >50 → params too loose. |
| N.3 | Compare ARMED states to manual chart inspection — verify the bot saw what a trader would see. Adjust params before live signals. |
| N.4 | Enable signal publishing. First 5 ARMED signals: manual review before next 5. Then auto-fire. |
| N.5 | After 50 trades: cohort report. PASS if WR > 60% AND avg PnL > $1.50 (high bar — full SMC sequence should outperform basic scanners). |

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| State machine missed a transition due to bar order issue | Audit log shows EVERY transition with timestamp; reproducible from candle data |
| Sequence too rare to be useful (< 1 trade/day) | Phase N.2 verification gate measures ARMED frequency before signal go-live |
| Symbol state corruption on partial restart | Persisted snapshot restored on startup; falls back to IDLE if corrupt |
| Trend Oracle (Patch M) not yet shipped | Patch N has hard dependency on Patch M; design doc explicit about this |
| Sequence completes but trend has already reversed | TTL on each state ensures stale sequences expire (max 50 bars total = 4h on 5m) |
| False positive: sweep then immediate continuation (no real flip) | CHoCH requirement (close above prior swing) filters most; trend_oracle gate filters more |
| Performance: per-bar evaluation across 20 symbols | Vectorized helpers + cached swing-high computation; ~5ms per symbol per close |

## 10. Implementation phases

| Phase | Wall time | Deliverable |
|---|---|---|
| **N.0** | (now, this doc) | Design spec + skeleton |
| **N.1** | 4-6h | Build module + dry-run mode (no signal publish) |
| **N.2** | 24h soak | Verify ARMED frequency in dry-run |
| **N.3** | 1-2h | Visual validation on TradingView vs ARMED states |
| **N.4** | 24h | Enable signal publish for ADMIN only, MANUAL approval per signal |
| **N.5** | 7-day soak | Auto-fire enabled, monitor WR + EV |
| **Total** | **~2 weeks end-to-end** | Production-grade SMC reversal sequence detector |

## 11. Why this is the high-leverage move (vs more single-pattern W/F)

| Approach | Today's score | Tomorrow's expected score |
|---|---|---|
| Single-pattern W/F (Demand Zone, Break Block, etc.) | 0/3 PASS | Same — Q3 chop kills all single-bar patterns |
| Sequence state machine + trend oracle gate | NEW | Plausibly 1-3 trades/day at high WR (full SMC sequence is rare but high quality) |

**Single patterns die because they ignore context. The state machine IS the context.** Each state transition embeds prior state — by the time we reach ARMED, we KNOW a sweep happened, AND a CHoCH formed, AND a BOS confirmed, AND price retraced to POI, AND it's in the Fib zone, AND there's a confirmation candle. That's 6 layers of confluence vs the 1-3 of any single scanner.

## 12. Decision gates for the architect

Before starting Phase N.1, confirm:

1. **Patch M ships first?** (Strong recommend yes — trend oracle is a hard dependency)
2. **Initial scope: LONG-side reversal only?** (Or both LONG-end-of-downtrend AND SHORT-end-of-uptrend from day 1?)
3. **Default ALIGNED policy:** sequence only fires when current trend is OPPOSITE of intended entry side. Confirm.
4. **Persistence: snapshot every 60s OR every state change?** (Recommend every state change — cheaper writes, audit trail richer)
5. **Per-symbol grade weighting:** SOL gets `grade=A` (already validated for SMC), BTC/ETH get `grade=B` until proven? Or all `A+` since full sequence?

Recommendation:
- Patch M first
- Both sides from day 1
- ALIGNED policy strict
- Snapshot per state change
- All symbols `grade=A+` (full sequence is rare and high-quality)

---

## Appendix — Code skeleton (Phase N.1)

```python
# bot/sequence_oracle.py
from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

STORAGE_DIR = Path("/home/opc/crypto-trading-bot/storage/sequence_oracle")
SNAPSHOT_PATH = STORAGE_DIR / "state_snapshot.json"
AUDIT_PATH = STORAGE_DIR / "transitions.jsonl"


class State(str, Enum):
    IDLE = "IDLE"
    EQUAL_LOWS_DETECTED = "EQUAL_LOWS_DETECTED"
    SWEPT = "SWEPT"
    CHOCH_CONFIRMED = "CHOCH_CONFIRMED"
    BOS_CONFIRMED = "BOS_CONFIRMED"
    POI_RETESTED = "POI_RETESTED"
    ARMED = "ARMED"


@dataclass
class SequenceContext:
    symbol: str
    side: str                          # "long" | "short"
    state: State = State.IDLE
    bars_since_state_change: int = 0
    # Step-1 state
    equal_low_price: Optional[float] = None
    equal_low_count: int = 0
    # Step-2 state
    sweep_bar_idx: Optional[int] = None
    sweep_low_price: Optional[float] = None
    # Step-3 state
    choch_close_price: Optional[float] = None
    prior_swing_high: Optional[float] = None
    # Step-4 state
    bos_close_price: Optional[float] = None
    poi_zone: Optional[Tuple[float, float]] = None
    # Step-5 state
    fib_levels: Optional[Tuple[float, float, float]] = None  # 0.62, 0.7, 0.79
    last_state_change_ts: Optional[str] = None


# Defaults — tunable
DEFAULTS = {
    "equal_low_atr": 0.3,
    "sweep_atr": 0.2,
    "wick_ratio": 0.4,
    "choch_body_atr": 0.5,
    "confirm_body_atr": 0.5,
    "swept_window_bars": 8,
    "choch_window_bars": 12,
    "bos_window_bars": 20,
    "retest_window_bars": 10,
    "trend_oracle_min_bars_in_state": 30,
}


class SMCReversalStateMachine:
    def __init__(self, dry_run: bool = True):
        self._lock = threading.Lock()
        self._states: Dict[Tuple[str, str], SequenceContext] = {}
        self._dry_run = dry_run
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_snapshot()

    def on_5m_close(self, symbol: str, df_5m: pd.DataFrame,
                    atr_5m: float) -> Optional[Dict]:
        """Process new 5m candle close for both sides. Returns ARMED signal dict or None."""
        signals = []
        for side in ("long", "short"):
            sig = self._process_one(symbol, side, df_5m, atr_5m)
            if sig:
                signals.append(sig)
        return signals[0] if signals else None

    def _process_one(self, symbol: str, side: str,
                     df_5m: pd.DataFrame, atr_5m: float) -> Optional[Dict]:
        key = (symbol, side)
        ctx = self._states.get(key) or SequenceContext(symbol=symbol, side=side)

        # Trend oracle gate — only advance if regime supports the flip
        if not self._trend_oracle_allows(symbol, side):
            if ctx.state != State.IDLE:
                self._transition(ctx, State.IDLE, "trend_oracle_gate_failed")
            self._states[key] = ctx
            return None

        # Per-state evaluator
        prev_state = ctx.state
        if ctx.state == State.IDLE:
            self._step_idle_to_equal_lows(ctx, df_5m, atr_5m)
        elif ctx.state == State.EQUAL_LOWS_DETECTED:
            self._step_equal_lows_to_swept(ctx, df_5m, atr_5m)
        elif ctx.state == State.SWEPT:
            self._step_swept_to_choch(ctx, df_5m, atr_5m)
        elif ctx.state == State.CHOCH_CONFIRMED:
            self._step_choch_to_bos(ctx, df_5m, atr_5m)
        elif ctx.state == State.BOS_CONFIRMED:
            self._step_bos_to_retest(ctx, df_5m, atr_5m)
        elif ctx.state == State.POI_RETESTED:
            self._step_retest_to_armed(ctx, df_5m, atr_5m)

        # Bump bars-since counter and check timeouts
        if ctx.state == prev_state:
            ctx.bars_since_state_change += 1
            self._check_timeout(ctx)
        else:
            ctx.bars_since_state_change = 0

        self._states[key] = ctx
        self._save_snapshot()

        # If ARMED, emit signal and reset
        if ctx.state == State.ARMED and not self._dry_run:
            sig = self._emit_signal(ctx, df_5m, atr_5m)
            self._transition(ctx, State.IDLE, "armed_signal_emitted")
            return sig
        return None

    # ── Step implementations (sketch) ──
    def _step_idle_to_equal_lows(self, ctx, df, atr):
        # Scan last 50 bars for ≥2 lows within equal_low_atr × ATR
        ...

    def _step_equal_lows_to_swept(self, ctx, df, atr):
        # Detect sweep candle: low penetrates equal_low by ≥ sweep_atr × ATR + wick reject
        ...

    def _step_swept_to_choch(self, ctx, df, atr):
        # Detect bar that closes above prior_swing_high with body ≥ choch_body_atr × ATR
        ...

    def _step_choch_to_bos(self, ctx, df, atr):
        # Detect bar that closes above next swing high after CHoCH
        ...

    def _step_bos_to_retest(self, ctx, df, atr):
        # Detect price returning into poi_zone
        ...

    def _step_retest_to_armed(self, ctx, df, atr):
        # Detect price in fib zone + bullish confirmation candle
        ...

    def _trend_oracle_allows(self, symbol: str, side: str) -> bool:
        try:
            from bot.trend_oracle import get_trend_oracle
            oracle = get_trend_oracle()
            ts = oracle.get_trend_state(symbol, "1h")
            min_bars = DEFAULTS["trend_oracle_min_bars_in_state"]
            if side == "long":
                return ts.color == "red" and ts.bars_in_state >= min_bars
            else:
                return ts.color == "green" and ts.bars_in_state >= min_bars
        except Exception:
            return True   # fail open

    def _emit_signal(self, ctx, df, atr) -> Dict:
        from bot.shadow_bridge import publish_signal
        last = df.iloc[-1]
        entry = float(last["close"])
        if ctx.side == "long":
            sl = ctx.sweep_low_price - 0.3 * atr
        else:
            sl = ctx.sweep_low_price + 0.3 * atr  # mirror for short
        risk = abs(entry - sl)
        tp = entry + risk if ctx.side == "long" else entry - risk
        sig_id = publish_signal(
            source_engine="smc_reversal_sequence",
            symbol=ctx.symbol, side=ctx.side,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            ml_probability=0.75, grade="A+",
            setup_type="smc_reversal",
            confidence=92.0, regime="reversal",
            extra_meta={
                "sequence_trace": asdict(ctx),
                "fib_zone": list(ctx.fib_levels) if ctx.fib_levels else None,
            },
        )
        return {"sig_id": sig_id, "ctx": ctx}

    def _transition(self, ctx, new_state: State, reason: str):
        old_state = ctx.state
        ctx.state = new_state
        ctx.last_state_change_ts = pd.Timestamp.utcnow().isoformat()
        # Append to audit log
        with open(AUDIT_PATH, "a") as fh:
            fh.write(json.dumps({
                "ts": ctx.last_state_change_ts,
                "symbol": ctx.symbol, "side": ctx.side,
                "from": old_state.value, "to": new_state.value,
                "reason": reason,
            }) + "\n")

    def _check_timeout(self, ctx):
        timeouts = {
            State.SWEPT: DEFAULTS["swept_window_bars"],
            State.CHOCH_CONFIRMED: DEFAULTS["choch_window_bars"],
            State.BOS_CONFIRMED: DEFAULTS["bos_window_bars"],
            State.POI_RETESTED: DEFAULTS["retest_window_bars"],
        }
        max_bars = timeouts.get(ctx.state)
        if max_bars and ctx.bars_since_state_change >= max_bars:
            self._transition(ctx, State.IDLE, f"timeout_{max_bars}bars")

    def _save_snapshot(self):
        snap = {
            f"{k[0]}_{k[1]}": {**asdict(v), "state": v.state.value}
            for k, v in self._states.items()
        }
        SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2, default=str))

    def _load_snapshot(self):
        if not SNAPSHOT_PATH.exists():
            return
        try:
            data = json.loads(SNAPSHOT_PATH.read_text())
            for k, v in data.items():
                sym, side = k.rsplit("_", 1)
                ctx = SequenceContext(symbol=sym, side=side)
                for f, val in v.items():
                    if hasattr(ctx, f):
                        if f == "state":
                            setattr(ctx, f, State(val))
                        else:
                            setattr(ctx, f, val)
                self._states[(sym, side)] = ctx
        except Exception:
            pass


_SM: Optional[SMCReversalStateMachine] = None


def get_smc_state_machine(dry_run: bool = True) -> SMCReversalStateMachine:
    global _SM
    if _SM is None:
        _SM = SMCReversalStateMachine(dry_run=dry_run)
    return _SM
```

---

**End of Patch N design doc. ~430 lines code skeleton + state machine spec. Ready for architect review + Phase N.1 implementation when Patch M is live.**

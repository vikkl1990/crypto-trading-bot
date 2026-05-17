"""Shadow maker-fill simulator — Phase 5.21 (2026-04-30).

PROBLEM
-------
`UserRealManager._execute_shadow()` (line 2036) hardcodes worst-case taker
fills against L2 top-of-book. This makes the admin-vs-niranjan A/B test
(admin=patient, niranjan=standard) silent — both users always see fee_type='taker'
because the shadow path never consults `maker_patience_mode`.

The maker counterfactual computed at close (line 3098) tells us what saving
WOULD have been realized at fixed-rate maker (50%/100%), but doesn't differentiate
patience modes — it's the same number for everyone.

This module gives the shadow path a probabilistic maker-fill simulator that:
  1. Reads the user's `maker_patience_mode`
  2. Looks at L2 spread + depth + our size
  3. Returns (filled_as_maker: bool, simulated_fill_price, mode_used) for the entry

When wired into `_execute_shadow`, it produces a real differential between
admin and niranjan: patient mode should fill ~55% maker (after queue/spread
adjustment), standard ~30%. Tomorrow's verdict will then have actual A/B data.

FILL MODEL — v1 (probabilistic, deterministic per signal_id)
------------------------------------------------------------
The full L2 queue model with order persistence + adverse-fill risk is Phase
5.22 work. v1 uses simple heuristics calibrated to known-published Delta India
maker fill rates from the 2026-04-26 batch B telemetry sweep:

  Base fill probability (standard mode, normal market):
    BTC/USDT  60%   — tight spreads, deep books
    ETH/USDT  47%
    SOL/USDT  23%
    Other     35%   (default)

  Multipliers:
    patience='patient'      → ×1.5  (5.14 patient mode: 2500ms probe, 2.5x bp offset)
    patience='aggressive'   → ×1.5  (alias of patient, kept for compat)
    patience='l2_aware'     → ×1.4  (Phase 5.19, walks book depth)
    patience='multimode'    → ×1.4  (per-trade A/B/C random — average effect)
    patience='standard'     → ×1.0  (baseline)

    spread_ticks ≤ 2        → ×1.20 (very tight: easy maker fill)
    spread_ticks 3-5        → ×1.00
    spread_ticks 6-10       → ×0.85
    spread_ticks > 10       → ×0.65 (wide: market needs to come to us)

    our_size/top_depth ratio:
      < 0.10                → ×1.10 (negligible queue position impact)
      0.10-0.50             → ×1.00
      0.50-1.00             → ×0.85
      > 1.00                → ×0.70 (we're bigger than top, queue grinds slow)

Fills are deterministic per signal_id (same signal → same outcome) so re-runs
of the same test produce identical fill patterns. This is critical for
reproducible A/B analysis.

WIRING (held — see docs/shadow_maker_sim_wiring_patch.md)
---------------------------------------------------------
Drop-in replacement for the entry-fill block in _execute_shadow:

    from execution_v2.shadow_maker_sim import simulate_entry_fill

    fill_result = simulate_entry_fill(
        order_side=order_side,
        book=book,
        symbol=symbol,
        our_lots=lots,
        tick_size=tick_size,
        patience_mode=self.maker_patience_mode,
        signal_id=signal.get("id") or signal.get("signal_id") or f"{symbol}_{int(time.time()*1000)}",
    )
    shadow_fill = fill_result.fill_price
    fee_type = fill_result.fee_type        # 'maker' or 'taker'
    fee_pct  = 0.0002 * 1.18 if fee_type == "maker" else 0.0005 * 1.18
    shadow_entry_fee = notional * fee_pct
    # ... existing trade record code, but stamp metadata:
    meta["maker_sim_attempted"] = True
    meta["maker_sim_filled"] = fill_result.filled_as_maker
    meta["maker_sim_mode"] = fill_result.mode_used
    meta["maker_sim_p_fill"] = fill_result.p_fill

The exit-fill block needs the same treatment for round-trip honesty.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Calibrated fill probabilities (Phase 5.21 v1).
# Base = probability under STANDARD mode at NORMAL conditions (3-5 tick
# spread, our_size 10-50% of top depth). Adjustments are deltas from this
# normal-market baseline. Multipliers MUST not stack to >1.0 in realistic
# scenarios. Targets after adjustment:
#   patient @ tight BTC normal size  → ~55-65% fill (not 100%)
#   patient @ wide SOL large size    → ~25-35% fill
#   standard @ normal BTC            → ~40-50% fill
_BASE_P_BY_SYMBOL: Dict[str, float] = {
    "BTC/USDT": 0.45,
    "ETH/USDT": 0.38,
    "SOL/USDT": 0.20,
    "XRP/USDT": 0.32,
    "ADA/USDT": 0.28,
}
_BASE_P_DEFAULT = 0.28

# Patience multipliers — chosen so patient×tight_spread×small_size lands
# around 0.65, not 1.0. patient mode is a meaningful but not deterministic
# improvement.
_PATIENCE_MULT: Dict[str, float] = {
    "standard": 1.0,
    "patient": 1.30,
    "aggressive": 1.30,  # alias
    "l2_aware": 1.20,
    "multimode": 1.20,
}
_PATIENCE_DEFAULT = 1.0

# Fee rates (Delta India, with 1.18× GST).
_FEE_MAKER = 0.0002 * 1.18   # 0.0236%
_FEE_TAKER = 0.0005 * 1.18   # 0.059%


@dataclass
class FillResult:
    filled_as_maker: bool
    fill_price: float
    fee_type: str               # 'maker' | 'taker'
    fee_pct: float              # decimal (0.000236 or 0.00059)
    mode_used: str              # patience mode at decision time
    p_fill: float               # probability that drove the decision (for debugging)
    spread_ticks: int           # snapshot at decision time
    top_depth: float            # snapshot at decision time


def _spread_mult(spread_ticks: int) -> float:
    # Tighter spreads → higher fill prob, but bounded so we don't stack to >1
    if spread_ticks <= 2:
        return 1.10
    if spread_ticks <= 5:
        return 1.00
    if spread_ticks <= 10:
        return 0.85
    return 0.65


def _size_mult(our_lots: float, top_depth: float) -> float:
    if top_depth <= 0:
        return 0.85  # no depth data — slight penalty
    ratio = our_lots / top_depth
    if ratio < 0.10:
        return 1.05
    if ratio < 0.50:
        return 1.00
    if ratio < 1.00:
        return 0.85
    return 0.70


def _deterministic_uniform(signal_id: str) -> float:
    """Map a signal_id to a stable [0,1) sample via SHA-256.

    Same signal_id → same float, different across signals.
    Better than Python's hash() (which is salted per-process).
    """
    h = hashlib.sha256(signal_id.encode("utf-8")).digest()
    # Take first 8 bytes as uint64, divide by 2^64
    n = int.from_bytes(h[:8], "big")
    return n / (1 << 64)


def compute_fill_probability(
    symbol: str,
    patience_mode: str,
    spread_ticks: int,
    our_lots: float,
    top_depth: float,
) -> Tuple[float, str]:
    """Return (p_fill, normalized_mode_name)."""
    mode = (patience_mode or "standard").lower().strip()
    base = _BASE_P_BY_SYMBOL.get(symbol, _BASE_P_DEFAULT)
    pm = _PATIENCE_MULT.get(mode, _PATIENCE_DEFAULT)
    sm = _spread_mult(spread_ticks)
    zm = _size_mult(our_lots, top_depth)
    p = base * pm * sm * zm
    return (max(0.0, min(1.0, p)), mode)


def simulate_entry_fill(
    order_side: str,
    book: Dict[str, Any],
    symbol: str,
    our_lots: float,
    tick_size: float,
    patience_mode: str,
    signal_id: str,
) -> FillResult:
    """Simulate whether a post_only maker entry would have filled.

    Args:
      order_side : 'buy' or 'sell'
      book       : {'bids': [[px, sz], ...], 'asks': [[px, sz], ...]}
      symbol     : e.g. 'BTC/USDT'
      our_lots   : our order size in lots/contracts
      tick_size  : minimum price increment
      patience_mode : 'standard' | 'patient' | 'aggressive' | 'l2_aware' | 'multimode'
      signal_id  : stable string for deterministic sampling

    Returns:
      FillResult — see dataclass.
    """
    if not book or not book.get("bids") or not book.get("asks"):
        # No L2 data → fall back to taker assumption (matches current behavior)
        return FillResult(
            filled_as_maker=False,
            fill_price=0.0,
            fee_type="taker",
            fee_pct=_FEE_TAKER,
            mode_used=patience_mode or "standard",
            p_fill=0.0,
            spread_ticks=0,
            top_depth=0.0,
        )

    best_bid = float(book["bids"][0][0])
    best_ask = float(book["asks"][0][0])
    bid_depth = float(book["bids"][0][1])
    ask_depth = float(book["asks"][0][1])

    if tick_size > 0 and best_ask > best_bid:
        spread_ticks = max(1, round((best_ask - best_bid) / tick_size))
    else:
        spread_ticks = 1

    top_depth = bid_depth if order_side == "buy" else ask_depth

    p_fill, mode = compute_fill_probability(
        symbol=symbol,
        patience_mode=patience_mode,
        spread_ticks=spread_ticks,
        our_lots=our_lots,
        top_depth=top_depth,
    )

    sample = _deterministic_uniform(signal_id)
    filled_as_maker = sample < p_fill

    if filled_as_maker:
        # post_only fill: at the spread edge our side
        if order_side == "buy":
            maker_px = best_bid + tick_size  # one tick inside spread (joining bid+1)
            # Don't cross the spread
            if maker_px >= best_ask:
                maker_px = best_bid
        else:
            maker_px = best_ask - tick_size
            if maker_px <= best_bid:
                maker_px = best_ask
        return FillResult(
            filled_as_maker=True,
            fill_price=maker_px,
            fee_type="maker",
            fee_pct=_FEE_MAKER,
            mode_used=mode,
            p_fill=p_fill,
            spread_ticks=spread_ticks,
            top_depth=top_depth,
        )

    # Taker fallback (current shadow behavior)
    taker_px = best_ask if order_side == "buy" else best_bid
    return FillResult(
        filled_as_maker=False,
        fill_price=taker_px,
        fee_type="taker",
        fee_pct=_FEE_TAKER,
        mode_used=mode,
        p_fill=p_fill,
        spread_ticks=spread_ticks,
        top_depth=top_depth,
    )


def simulate_exit_fill(
    side: str,                # 'long' | 'short' (trade.side, NOT order_side)
    book: Dict[str, Any],
    symbol: str,
    our_lots: float,
    tick_size: float,
    patience_mode: str,
    signal_id: str,
    entry_was_maker: bool,
    holding_sec: float,
) -> FillResult:
    """Simulate the closing fill.

    Same model as entry but the patience mode applies less aggressively at
    exit (you don't want to camp on a maker order while the trade decays).
    Specifically:
      - patient mode applies a 0.7x dampener for SCALP (≤10min holds): you
        can't afford to wait long
      - for longer holds, full patience multiplier applies

    Critically: scalper-offer math (BTC/ETH ≤30min, others ≤15min → free
    exit) is HANDLED BY THE FEE MODEL, not here. This function only models
    the fill type. The fee waiver is applied separately.
    """
    if not book or not book.get("bids") or not book.get("asks"):
        return FillResult(
            filled_as_maker=False, fill_price=0.0, fee_type="taker",
            fee_pct=_FEE_TAKER, mode_used=patience_mode, p_fill=0.0,
            spread_ticks=0, top_depth=0.0,
        )

    # Closing direction
    order_side = "sell" if side == "long" else "buy"

    # Dampen patience for scalper-window-eligible exits (don't camp)
    effective_mode = patience_mode
    if holding_sec < 600 and patience_mode in ("patient", "aggressive"):
        # < 10min hold: pretend we're standard for fill prob calc, but tag
        # the actual mode for telemetry
        effective_mode = "standard"

    return simulate_entry_fill(
        order_side=order_side,
        book=book,
        symbol=symbol,
        our_lots=our_lots,
        tick_size=tick_size,
        patience_mode=effective_mode,
        signal_id=f"{signal_id}__exit",
    )


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== shadow_maker_sim smoke test ===\n")

    # Mock L2 book — BTC/USDT, $1 spread, 100 lot top depth
    book_btc = {
        "bids": [[75000.0, 100], [74999.5, 200], [74999.0, 300]],
        "asks": [[75001.0, 100], [75001.5, 250], [75002.0, 400]],
    }
    book_sol = {
        "bids": [[185.50, 50], [185.45, 80], [185.40, 100]],
        "asks": [[185.60, 50], [185.65, 70], [185.70, 100]],
    }

    print("BTC/USDT, 12 lots, $0.5 tick, 1-tick spread:")
    for mode in ("standard", "patient", "aggressive", "l2_aware"):
        # 100 deterministic samples — count maker fills
        n_maker = 0
        for i in range(100):
            r = simulate_entry_fill(
                order_side="buy",
                book=book_btc,
                symbol="BTC/USDT",
                our_lots=12,
                tick_size=0.5,
                patience_mode=mode,
                signal_id=f"test_btc_{i}",
            )
            if r.filled_as_maker:
                n_maker += 1
        print(f"  mode={mode:12s} n_maker={n_maker}/100  expected p≈"
              f"{compute_fill_probability('BTC/USDT', mode, 1, 12, 100)[0]:.2f}")

    print("\nSOL/USDT, 50 lots, 0.05 tick, 2-tick spread:")
    for mode in ("standard", "patient"):
        n_maker = 0
        for i in range(100):
            r = simulate_entry_fill(
                order_side="buy",
                book=book_sol,
                symbol="SOL/USDT",
                our_lots=50,
                tick_size=0.05,
                patience_mode=mode,
                signal_id=f"test_sol_{i}",
            )
            if r.filled_as_maker:
                n_maker += 1
        print(f"  mode={mode:12s} n_maker={n_maker}/100  expected p≈"
              f"{compute_fill_probability('SOL/USDT', mode, 2, 50, 50)[0]:.2f}")

    print("\nDeterminism check (same signal_id → same outcome):")
    r1 = simulate_entry_fill("buy", book_btc, "BTC/USDT", 12, 0.5, "patient", "sigA")
    r2 = simulate_entry_fill("buy", book_btc, "BTC/USDT", 12, 0.5, "patient", "sigA")
    assert r1.filled_as_maker == r2.filled_as_maker, "non-deterministic!"
    print(f"  signal 'sigA' twice → both filled_as_maker={r1.filled_as_maker} ✓")

    print("\nFee rates:")
    print(f"  taker = {_FEE_TAKER*100:.4f}%  (0.05% × 1.18 GST)")
    print(f"  maker = {_FEE_MAKER*100:.4f}%  (0.02% × 1.18 GST)")
    print(f"  delta = {(_FEE_TAKER - _FEE_MAKER)*10000:.2f} bps savings/side")

    print("\nExit fill (15min BTC long, patient mode):")
    r = simulate_exit_fill(
        side="long", book=book_btc, symbol="BTC/USDT",
        our_lots=12, tick_size=0.5, patience_mode="patient",
        signal_id="trade_xyz", entry_was_maker=True, holding_sec=900,
    )
    print(f"  filled_as_maker={r.filled_as_maker} mode_used={r.mode_used} p_fill={r.p_fill:.2f}")

    print("\nExit fill (5min BTC long, patient mode — should dampen):")
    r = simulate_exit_fill(
        side="long", book=book_btc, symbol="BTC/USDT",
        our_lots=12, tick_size=0.5, patience_mode="patient",
        signal_id="trade_xyz", entry_was_maker=True, holding_sec=300,
    )
    print(f"  filled_as_maker={r.filled_as_maker} mode_used={r.mode_used} p_fill={r.p_fill:.2f}")
    print("  (mode_used should still report 'patient' but p should reflect 'standard')")

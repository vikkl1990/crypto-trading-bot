"""
Fill models: convert a paper signal's idealized entry/exit prices into
realistic fills that a demo/live executor would actually receive.

Two base models:
  - TakerFillModel: crosses the spread, pays slippage on entry AND exit.
    Configurable slippage in bps (default 5bp per side).
  - MakerFillModel: rests at limit, has miss probability. On miss,
    optionally falls through to taker.

Both models support per-symbol slippage overrides (meme coins are worse
than BTC).

Future extensions:
  - Book-walk model using captured L2 depth from delta_ws
  - Latency model (paper exits at indicator-fire tick; demo exits 200ms later)
  - Partial-fill model for larger orders

Unit convention:
  bps = basis points = 0.01% = 1/10000
  slippage_bps=5 → 0.05% price adjustment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# Per-symbol typical spread on Delta India testnet (from L2_INIT observations)
# Values in bps. Higher = less liquid = more slippage expected.
TYPICAL_SPREAD_BPS: Dict[str, float] = {
    "BTC/USDT":  0.6,    # 0.5 tick on ~78000 ≈ 0.6bp
    "ETH/USDT":  2.2,    # 0.05 on 2330
    "SOL/USDT": 13.0,    # 0.011 on 86
    "XRP/USDT": 14.0,    # 0.0002 on 1.44
    "LTC/USDT": 18.0,
    "ADA/USDT": 40.0,    # thin book
    "DOT/USDT": 20.0,
    "TAO/USDT": 12.0,
    "DOGE/USDT": 10.0,
    "LINK/USDT": 10.0,
    "AVAX/USDT": 20.0,
    "PEPE/USDT": 50.0,   # meme
    "SHIB/USDT": 40.0,
    "WIF/USDT":  55.0,
    "SUI/USDT":  15.0,
    "NEAR/USDT": 20.0,
    "BONK/USDT": 60.0,
    "TRUMP/USDT": 30.0,
    "POPCAT/USDT": 50.0,
    "MEME/USDT":  60.0,
}
DEFAULT_SPREAD_BPS = 20.0  # unknown symbol fallback


@dataclass
class FillResult:
    """Outcome of a simulated fill."""
    filled: bool
    fill_price: float
    slippage_bps: float      # how many bp worse than signal_price
    fill_type: str           # 'taker' | 'maker' | 'maker_miss_fell_through'


@dataclass
class TakerFillModel:
    """
    Market-taker fill: always fills, but pays spread + slippage.

    Entry:
        long:  fills at signal_price × (1 + slip/10000)
        short: fills at signal_price × (1 - slip/10000)

    Exit:
        long (sell to close):  fills at signal_price × (1 - slip/10000)
        short (buy to close):  fills at signal_price × (1 + slip/10000)

    Slippage defaults:
        entry_slip_bps = half_spread + extra_latency_bps
        exit_slip_bps  = same

    Configurable per-symbol via `per_symbol_override`.
    """
    extra_latency_bps: float = 2.0   # always-on latency penalty beyond spread
    per_symbol_override: Dict[str, float] = field(default_factory=dict)

    def _slip_bps(self, symbol: str) -> float:
        if symbol in self.per_symbol_override:
            return self.per_symbol_override[symbol]
        # Half the bid-ask spread (taker crosses it) + latency
        half_spread = TYPICAL_SPREAD_BPS.get(symbol, DEFAULT_SPREAD_BPS) / 2.0
        return half_spread + self.extra_latency_bps

    def fill_entry(self, symbol: str, side: str, signal_price: float) -> FillResult:
        bps = self._slip_bps(symbol)
        factor = 1.0 + bps / 10000.0 if side == "long" else 1.0 - bps / 10000.0
        return FillResult(
            filled=True,
            fill_price=signal_price * factor,
            slippage_bps=bps,
            fill_type="taker",
        )

    def fill_exit(self, symbol: str, side: str, signal_price: float) -> FillResult:
        # Exit sign is opposite of entry (closing flip)
        bps = self._slip_bps(symbol)
        factor = 1.0 - bps / 10000.0 if side == "long" else 1.0 + bps / 10000.0
        return FillResult(
            filled=True,
            fill_price=signal_price * factor,
            slippage_bps=bps,
            fill_type="taker",
        )


@dataclass
class MakerFillModel:
    """
    Passive maker: rests at limit. May miss if market doesn't come back.

    Simplification for MVP:
        entry_miss_prob: probability the resting order never fills
        exit_miss_prob: same for exit
        fallback_to_taker: if miss, optionally execute at taker price

    When filled as maker, slippage = 0 (we got our price).
    When fallback fires, slippage = taker slippage.
    """
    entry_miss_prob: float = 0.40
    exit_miss_prob: float = 0.30
    fallback_to_taker: bool = True
    fallback_model: Optional[TakerFillModel] = None

    def __post_init__(self):
        if self.fallback_model is None:
            self.fallback_model = TakerFillModel()

    def _miss(self, prob: float, seed_salt: int) -> bool:
        # Deterministic pseudorandom from a salt — reproducible backtests.
        # Simple hash-based approach so no RNG state leaks across tests.
        import hashlib
        h = int(hashlib.md5(f"{seed_salt}_{prob}".encode()).hexdigest()[:8], 16)
        return (h / 0xFFFFFFFF) < prob

    def fill_entry(self, symbol: str, side: str, signal_price: float) -> FillResult:
        # Deterministic miss check (seed from price+symbol)
        seed = int(signal_price * 10000) + hash(symbol) % 1000
        if self._miss(self.entry_miss_prob, seed):
            if self.fallback_to_taker:
                r = self.fallback_model.fill_entry(symbol, side, signal_price)
                return FillResult(
                    filled=True, fill_price=r.fill_price,
                    slippage_bps=r.slippage_bps,
                    fill_type="maker_miss_fell_through",
                )
            return FillResult(
                filled=False, fill_price=0.0, slippage_bps=0.0,
                fill_type="maker_miss",
            )
        # Maker fill at the signal price exactly
        return FillResult(
            filled=True, fill_price=signal_price, slippage_bps=0.0,
            fill_type="maker",
        )

    def fill_exit(self, symbol: str, side: str, signal_price: float) -> FillResult:
        seed = int(signal_price * 10000) + hash(symbol) % 1000 + 7
        if self._miss(self.exit_miss_prob, seed):
            if self.fallback_to_taker:
                r = self.fallback_model.fill_exit(symbol, side, signal_price)
                return FillResult(
                    filled=True, fill_price=r.fill_price,
                    slippage_bps=r.slippage_bps,
                    fill_type="maker_miss_fell_through",
                )
            return FillResult(
                filled=False, fill_price=0.0, slippage_bps=0.0,
                fill_type="maker_miss",
            )
        return FillResult(
            filled=True, fill_price=signal_price, slippage_bps=0.0,
            fill_type="maker",
        )


@dataclass
class PaperFillModel:
    """
    Frictionless paper fill. Used for reconstructing paper's own P&L curve
    within the same framework (sanity check).
    """

    def fill_entry(self, symbol: str, side: str, signal_price: float) -> FillResult:
        return FillResult(
            filled=True, fill_price=signal_price, slippage_bps=0.0, fill_type="paper",
        )

    def fill_exit(self, symbol: str, side: str, signal_price: float) -> FillResult:
        return FillResult(
            filled=True, fill_price=signal_price, slippage_bps=0.0, fill_type="paper",
        )

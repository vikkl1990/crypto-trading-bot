"""
Cost model: computes fees and funding for a simulated trade.

Delta India fee schedule (perpetuals, 2026-04):
    Taker: 0.05% × 1.18 GST = 0.059% of notional
    Maker: 0.02% × 1.18 GST = 0.0236% of notional

Funding:
    Fires every 8h at IST 05:30 / 13:30 / 21:30.
    funding_paid = notional × funding_rate × (hours_held / 8)
    Sign: longs pay positive rate (cost), shorts collect.

MVP uses a SINGLE entry snapshot for rate (linear interpolation across
full hold). Full version will reset rate at each 8h boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Delta India fee constants (2026-04)
TAKER_RATE = 0.0005        # 0.05%
MAKER_RATE = 0.0002        # 0.02%
GST_MULT   = 1.18          # 18% Indian GST applied on the fee
TAKER_FEE  = TAKER_RATE * GST_MULT  # 0.059%
MAKER_FEE  = MAKER_RATE * GST_MULT  # 0.0236%

# Default funding rate if unavailable (8-hour rate, neutral fallback)
DEFAULT_FUNDING_RATE_8H = 0.0001  # 0.01% / 8h


@dataclass
class TradeCost:
    """Itemized cost of a round-trip trade."""
    entry_fee_usd:  float
    exit_fee_usd:   float
    funding_usd:    float  # signed: + = cost, - = income
    total_cost_usd: float  # entry_fee + exit_fee + funding


def fee_per_side(notional_usd: float, fill_type: str) -> float:
    """
    Compute one-side fee. fill_type ∈ {taker, maker, paper, maker_miss_fell_through}.
    Paper mode returns 0. Maker-miss-fallthrough is taker.
    """
    if fill_type == "paper":
        return 0.0
    if fill_type == "maker":
        return notional_usd * MAKER_FEE
    # taker, maker_miss_fell_through, or anything else
    return notional_usd * TAKER_FEE


def compute_funding(
    notional_usd: float,
    side: str,
    hours_held: float,
    funding_rate_8h: Optional[float] = None,
) -> float:
    """
    Funding cost for one trade held for `hours_held`.
    Returns signed USD: + = cost (long pays), - = income (short collects).

    MVP: linear interpolation of single rate across full hold.
    """
    if hours_held <= 0 or notional_usd <= 0:
        return 0.0
    rate = funding_rate_8h if funding_rate_8h is not None else DEFAULT_FUNDING_RATE_8H
    # abs() in case paper stored a negative snapshot
    periods = hours_held / 8.0
    raw = notional_usd * rate * periods
    return raw if side == "long" else -raw


def compute_round_trip_cost(
    notional_usd:   float,
    side:           str,
    entry_fill_type: str,
    exit_fill_type:  str,
    hours_held:     float = 0.0,
    funding_rate_8h: Optional[float] = None,
) -> TradeCost:
    """
    Full round-trip cost including both fills and funding.

    Notional is ENTRY notional (approximate — slight difference from exit
    notional in a winner/loser but immaterial at our scale).
    """
    e_fee = fee_per_side(notional_usd, entry_fill_type)
    x_fee = fee_per_side(notional_usd, exit_fill_type)
    fund  = compute_funding(notional_usd, side, hours_held, funding_rate_8h)
    return TradeCost(
        entry_fee_usd=e_fee,
        exit_fee_usd=x_fee,
        funding_usd=fund,
        total_cost_usd=e_fee + x_fee + fund,
    )

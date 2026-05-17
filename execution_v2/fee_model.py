"""Fee model with Delta India scalper-offer support.

Delta India fee schedule (perpetuals):
  - Taker fee: 0.05-0.06% (we use 0.06% baseline)
  - Maker fee: 0.02-0.04% (we use 0.024% baseline maker rebate-like rate)

Scalper Offer (Delta India promo):
  - For BTC/ETH: exit fee waived if trade closes within 30 minutes of entry
  - For other (eligible) symbols: exit fee waived within 15 minutes
  - Entry leg always pays full fee (taker or maker)
  - Only the EXIT leg gets waived

Calling conventions:
  fm = FeeModel()
  result = fm.round_trip_for_trade(
      exchange="delta",
      entry_type="taker",
      exit_type="taker",
      notional_usd=1000.0,
      symbol="BTC/USDT",
      holding_sec=900,
  )
  result["fee_usd"]          -> dollar fees for the round trip (entry + exit)
  result["fee_rate"]         -> total RT rate as decimal fraction
  result["scalper_eligible"] -> True if symbol+window qualifies
  result["scalper_applied"]  -> True if exit waived (eligible AND within window)
"""
from __future__ import annotations
from typing import Dict, Tuple


# Delta India fee schedule (decimal fractions)
DELTA_TAKER = 0.0006          # 0.06%
DELTA_MAKER = 0.00024         # 0.024%
# Settlement fees waived intraday (we ignore here — round-trip only)

# Scalper-offer windows (seconds)
SCALPER_WINDOW_BTC_ETH = 30 * 60   # 30 minutes
SCALPER_WINDOW_OTHER = 15 * 60     # 15 minutes
SCALPER_BIG_SYMBOLS = {"BTC/USDT", "ETH/USDT", "BTCUSD", "ETHUSD"}
# Only BTC/ETH waiver is publicly confirmed at the time of writing.
# Set ENABLE_OTHER_SYMBOLS_SCALPER=True if/when the offer extends.
ENABLE_OTHER_SYMBOLS_SCALPER = False


def _leg_rate(exchange: str, leg_type: str) -> float:
    leg_type = (leg_type or "taker").lower()
    if exchange != "delta":
        # Default to delta for now; extend if/when other exchanges added
        pass
    if leg_type == "maker":
        return DELTA_MAKER
    return DELTA_TAKER


def is_scalper_eligible(symbol: str, holding_sec: float) -> Tuple[bool, bool]:
    """Return (symbol_eligible, within_window).

    symbol_eligible — True if Delta lists the symbol under scalper offer.
    within_window  — True if holding duration <= permitted window.
    """
    sym = (symbol or "").upper()
    if sym in SCALPER_BIG_SYMBOLS:
        return True, holding_sec <= SCALPER_WINDOW_BTC_ETH
    if ENABLE_OTHER_SYMBOLS_SCALPER:
        return True, holding_sec <= SCALPER_WINDOW_OTHER
    return False, False


class FeeModel:
    """Compute per-leg and round-trip fees with Delta scalper-offer awareness."""

    def __init__(self,
                 taker: float = DELTA_TAKER,
                 maker: float = DELTA_MAKER):
        self.taker = taker
        self.maker = maker

    def leg_fee_usd(self, leg_type: str, notional_usd: float) -> float:
        rate = self.maker if (leg_type or "taker").lower() == "maker" else self.taker
        return float(notional_usd) * rate

    def round_trip_for_trade(
        self,
        exchange: str = "delta",
        entry_type: str = "taker",
        exit_type: str = "taker",
        notional_usd: float = 1000.0,
        symbol: str = "BTC/USDT",
        holding_sec: float = 0.0,
        force_no_scalper: bool = False,
    ) -> Dict[str, float]:
        """Compute round-trip fee with optional scalper-offer waiver on exit leg."""
        entry_rate = self.maker if (entry_type or "taker").lower() == "maker" else self.taker
        exit_rate_full = self.maker if (exit_type or "taker").lower() == "maker" else self.taker

        sym_elig, within = is_scalper_eligible(symbol, holding_sec)
        scalper_applied = bool(sym_elig and within and not force_no_scalper)

        entry_fee = notional_usd * entry_rate
        exit_fee = 0.0 if scalper_applied else (notional_usd * exit_rate_full)

        total_fee = entry_fee + exit_fee
        total_rate = total_fee / notional_usd if notional_usd > 0 else 0.0

        return {
            "fee_usd": total_fee,
            "fee_rate": total_rate,
            "entry_fee_usd": entry_fee,
            "exit_fee_usd": exit_fee,
            "entry_rate": entry_rate,
            "exit_rate_charged": 0.0 if scalper_applied else exit_rate_full,
            "exit_rate_full": exit_rate_full,
            "scalper_eligible": sym_elig,
            "scalper_within_window": within,
            "scalper_applied": scalper_applied,
        }


# Convenience module-level instance for ad-hoc use
_default = FeeModel()


def round_trip_for_trade(*args, **kwargs) -> Dict[str, float]:
    return _default.round_trip_for_trade(*args, **kwargs)

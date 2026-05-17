"""Per-scanner real-trade policy — single source of truth for live execution gates.

Architect spec (2026-05-03 ship batch 1 + 2 / Phase 4B):

Per (scanner, symbol) policy fields:
  live_enabled           — bool. False blocks the pair from live/shadow execution.
  shadow_only            — bool. If True (and live_enabled=True), block real but allow shadow.
  entry_type             — "maker_only" | "maker_preferred" | "maker_taker_dynamic"
  max_entry_wait_bars    — int. For maker entries, cancel limit after N bars unfilled.
  taker_min_expected_move_rt_multiplier — float. For maker_taker_dynamic, taker fill
                                          allowed only if expected_move >= N × RT_cost.
                                          (RT_cost is round-trip fees, computed from
                                           FeeModel given symbol + holding window.)
  exit_method            — "vwap_touch" | "fast_exit_volfade" | "tp_rr" | "default"
  exit_max_hold_bars     — int. Hard time cap.
  exit_tp_rr             — float. For exit_method="tp_rr", target R-multiple.
  no_averaging           — bool. Block adding to existing position.
  kill_reason            — str. Free-form reason for live_enabled=False (audit trail).
  deployed_at            — ISO date. When the policy was last edited.

Pairs not in POLICIES default to DEFAULT_POLICY (live_enabled=True, no constraints).

USAGE:
    from bot.scanner_real_policy import live_enabled, policy_for, taker_cost_gate

    # In urm._execute_shadow:
    if not live_enabled(scanner, symbol):
        return  # SCANNER_POLICY block

    admit, diag = taker_cost_gate(scanner, symbol, entry_price, sl_price, tp_price, notional)
    if not admit:
        return  # TAKER_COST_GATE block

    pol = policy_for(scanner, symbol)
    # pol may have entry_type, max_entry_wait_bars, exit_method, etc.
    # Caller decides how to enforce based on policy.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────
# POLICIES — keyed by (scanner_name, symbol)
# ──────────────────────────────────────────────────────────────────────
# Pairs with live_enabled=False are blocked from real/shadow execution at the
# urm._execute_shadow gate. Paper engines may still run for ML training data.
POLICIES: Dict[Tuple[str, str], Dict[str, Any]] = {

    # ── SOL liq_grab_ob_fvg + liquidity_sweep_htf → DISABLED ──────────
    # W/F evidence: exit_wf_tp1_be_trail_continuation 2026-05-03
    #   baseline EV_oos = $-0.576/trade over 1143 trades on SOL
    #   all exit-policy variants made it WORSE
    #   rolling-EV auto-disable wants to skip 75-98% of signals
    #   → entry detector is broken in OOS; needs rework, not exit tuning
    # Policy block kept rich for re-enable readiness (Phase 4B framework).
    ("liq_grab_ob_fvg", "SOL/USDT"): {
        "live_enabled": False,
        "shadow_only": True,
        "entry_type": "maker_taker_dynamic",
        "taker_min_expected_move_rt_multiplier": 3.0,
        "exit_method": "tp_rr",
        "exit_tp_rr": 2.0,
        "exit_max_hold_bars": 25,
        "kill_reason": "wf_2026_05_03: oos_ev=$-0.576/trade; all exit variants KILL",
        "deployed_at": "2026-05-03",
    },
    ("liquidity_sweep_htf", "SOL/USDT"): {
        "live_enabled": False,
        "shadow_only": True,
        "entry_type": "maker_taker_dynamic",
        "taker_min_expected_move_rt_multiplier": 3.0,
        "exit_method": "tp_rr",
        "exit_tp_rr": 2.0,
        "exit_max_hold_bars": 25,
        "kill_reason": "wf_2026_05_03: oos_ev=$-0.576/trade; all exit variants KILL",
        "deployed_at": "2026-05-03",
    },

    # ── scalper_vwap_mr × {BTC, ETH} — maker-only, fast exit ──────────
    # W/F: exit_wf_vwap_touch 2026-05-03 baseline ALREADY OPTIMAL
    #   BTC: +$0.031/trade, ETH: +$0.118/trade. All exit variants KILL.
    # Per architect: maker-only, no market fallback, max wait 1-2 bars,
    # exit VWAP touch OR 5-8 bars.
    ("scalper_vwap_mr", "BTC/USDT"): {
        "live_enabled": True,
        "entry_type": "maker_only",
        "max_entry_wait_bars": 2,
        "exit_method": "vwap_touch",
        "exit_max_hold_bars": 8,
        "deployed_at": "2026-05-03",
    },
    ("scalper_vwap_mr", "ETH/USDT"): {
        "live_enabled": True,
        "entry_type": "maker_only",
        "max_entry_wait_bars": 2,
        "exit_method": "vwap_touch",
        "exit_max_hold_bars": 8,
        "deployed_at": "2026-05-03",
    },

    # ── absorption_bubble × ETH — maker-preferred, fast exit ──────────
    # W/F: exit_wf_absorption_fast 2026-05-03 winner = absorption_fast_exit_6bar
    # ALREADY shipped to live engine (TP=1.0R, vol-fade exit, max 6 bars).
    ("absorption_bubble", "ETH/USDT"): {
        "live_enabled": True,
        "entry_type": "maker_preferred",
        "exit_method": "fast_exit_volfade",
        "exit_max_hold_bars": 6,
        "no_averaging": True,
        "deployed_at": "2026-05-03",
    },

    # ── structure_bounce × 8 symbols — chandelier trail enabled ──────
    # CHANDELIER_TRAIL_5_22 (2026-05-03) — port from signal_tracker.
    # Backtest verified: +$1,945 lift over 14 days at activation_R=0.4 / trail_atr=0.6
    # (flips structure_bounce from -$1,262 → +$683). See chandelier_shadow_backtest.
    # Applies to live_enabled=True scanners (rolling_ev gate may still block individual signals).
    # Same 8 symbols as rolling_ev_monitor scope.
    ("structure_bounce", "BTC/USDT"): {
        "live_enabled": True,
        "entry_type": "default",
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        # TIME_GATE_5_22 (2026-05-03) — block UTC hours that historically lost
        # most money for structure_bounce. From shadow_loss_review 7d data:
        # 7, 16, 17, 19, 21 UTC were biggest loss concentration.
        # 14, 15 UTC were positive. Other hours mixed.
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },
    ("structure_bounce", "ETH/USDT"): {
        "live_enabled": True,
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },
    ("structure_bounce", "SOL/USDT"): {
        "live_enabled": True,
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },
    ("structure_bounce", "XRP/USDT"): {
        "live_enabled": True,
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },
    ("structure_bounce", "DOT/USDT"): {
        "live_enabled": True,
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },
    ("structure_bounce", "DOGE/USDT"): {
        "live_enabled": True,
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },
    ("structure_bounce", "LINK/USDT"): {
        "live_enabled": True,
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },
    ("structure_bounce", "LTC/USDT"): {
        "live_enabled": True,
        "exit_method": "chandelier_trail",
        "chandelier_activation_R": 0.4,
        "chandelier_trail_atr": 0.6,
        "block_utc_hours": [7, 16, 17, 19, 21],
        "deployed_at": "2026-05-03",
    },

    # Other scanners default via policy_for() fallback (live_enabled=True).
    # Notable: structure_bounce is allowed BUT gated by rolling_ev_monitor
    # for 8 symbols (BTC/ETH/SOL/XRP/DOT/DOGE/LINK/LTC). See bot/rolling_ev_monitor.py.
}


DEFAULT_POLICY: Dict[str, Any] = {
    "live_enabled": True,
    "shadow_only": False,
    "entry_type": "default",
    "exit_method": "default",
    "policy": "default",
}


def live_enabled(scanner: str, symbol: str) -> bool:
    """Hot-path gate. Returns True if the (scanner, symbol) is allowed to
    execute live (real or shadow trades). Returns False to BLOCK execution."""
    p = POLICIES.get((scanner, symbol))
    if p is None:
        return True
    return bool(p.get("live_enabled", True))


def shadow_only(scanner: str, symbol: str) -> bool:
    """Returns True if pair is allowed to run in shadow mode but blocked from
    real money. Useful for graduated rollout."""
    p = POLICIES.get((scanner, symbol))
    if p is None:
        return False
    return bool(p.get("shadow_only", False))


def policy_for(scanner: str, symbol: str) -> Dict[str, Any]:
    """Return the full policy dict for (scanner, symbol), or DEFAULT_POLICY."""
    return dict(POLICIES.get((scanner, symbol), DEFAULT_POLICY))


def entry_type(scanner: str, symbol: str) -> str:
    """Return entry_type for (scanner, symbol). One of:
       maker_only | maker_preferred | maker_taker_dynamic | default."""
    return policy_for(scanner, symbol).get("entry_type", "default")


def exit_method(scanner: str, symbol: str) -> str:
    """Return exit_method for (scanner, symbol). One of:
       vwap_touch | fast_exit_volfade | tp_rr | default."""
    return policy_for(scanner, symbol).get("exit_method", "default")


def exit_max_hold_bars(scanner: str, symbol: str) -> Optional[int]:
    v = policy_for(scanner, symbol).get("exit_max_hold_bars")
    return int(v) if v is not None else None


def all_disabled_pairs() -> list:
    """List all (scanner, symbol) pairs currently blocked from live."""
    return [k for k, v in POLICIES.items() if not v.get("live_enabled", True)]


def time_blocked(scanner: str, symbol: str,
                 current_hour_utc: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    """TIME_GATE_5_22 (2026-05-03): block signals during configured UTC hours.

    Per shadow_loss_review 7d: certain UTC hours concentrate losses.
    For structure_bounce: 7, 16, 17, 19, 21 UTC = bad. 14, 15 UTC = good.

    Returns (blocked, diag).
    """
    pol = policy_for(scanner, symbol)
    blocked_hours = pol.get("block_utc_hours")
    if not blocked_hours:
        return False, {"reason": "no_time_rule"}
    if current_hour_utc is None:
        from datetime import datetime, timezone
        current_hour_utc = datetime.now(timezone.utc).hour
    if current_hour_utc in blocked_hours:
        return True, {
            "reason": "blocked_utc_hour",
            "current_hour_utc": current_hour_utc,
            "blocked_hours": blocked_hours,
        }
    return False, {
        "reason": "time_ok",
        "current_hour_utc": current_hour_utc,
    }


# ──────────────────────────────────────────────────────────────────────
# Taker-cost gate (Phase 4B)
# ──────────────────────────────────────────────────────────────────────
# Fee constants (Delta India + GST). Match bot/rolling_ev_monitor + execution_v2/fee_model.
FEE_TAKER_PCT = 0.00059      # 0.05% × 1.18 GST
FEE_MAKER_PCT = 0.000236     # 0.02% × 1.18 GST
SCALPER_WINDOW_SEC_BTCETH = 1800
SCALPER_WINDOW_SEC_OTHER = 900


def round_trip_cost_usd(symbol: str, notional: float, *,
                        entry_type: str = "taker",
                        assume_scalper_eligible: bool = True) -> float:
    """Compute estimated round-trip cost ($) for a $notional trade.

    For maker entry + scalper exit (BTC/ETH ≤30min, others ≤15min):
        RT = notional × FEE_MAKER_PCT  (exit fee waived under scalper offer)
    For taker entry + scalper exit:
        RT = notional × FEE_TAKER_PCT
    For taker entry + taker exit (no scalper):
        RT = notional × FEE_TAKER_PCT × 2

    `assume_scalper_eligible=True` — assume trade closes within scalper window."""
    entry_pct = FEE_MAKER_PCT if entry_type == "maker" else FEE_TAKER_PCT
    if assume_scalper_eligible:
        exit_pct = 0.0
    else:
        exit_pct = FEE_TAKER_PCT
    return notional * (entry_pct + exit_pct)


def expected_move_usd(entry_price: float, tp_price: float, side: str, notional: float) -> float:
    """Compute expected dollar move from entry → TP for a $notional position.

    For LONG:  move_pct = (tp - entry) / entry
    For SHORT: move_pct = (entry - tp) / entry
    move_$    = move_pct × notional
    Returns 0.0 if tp_price is None/invalid (no expected-move calc possible)."""
    if not tp_price or not entry_price or entry_price <= 0:
        return 0.0
    if side.lower() == "long":
        move_pct = (tp_price - entry_price) / entry_price
    else:
        move_pct = (entry_price - tp_price) / entry_price
    return float(move_pct * notional)


def taker_cost_gate(
    scanner: str, symbol: str,
    entry_price: float, sl_price: float, tp_price: float,
    side: str, notional: float = 1000.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Gate that allows/blocks a signal based on whether expected_move ≥ N × RT cost.

    Only enforced for pairs with `entry_type=maker_taker_dynamic` AND a
    `taker_min_expected_move_rt_multiplier` config field. Other pairs pass through.

    Returns (admit, diag). admit=True means signal should be executed.
    """
    pol = policy_for(scanner, symbol)
    if pol.get("entry_type") != "maker_taker_dynamic":
        return True, {"reason": "no_taker_cost_rule"}
    n_mult = pol.get("taker_min_expected_move_rt_multiplier")
    if n_mult is None:
        return True, {"reason": "no_taker_cost_rule"}
    rt_cost = round_trip_cost_usd(symbol, notional, entry_type="taker",
                                   assume_scalper_eligible=True)
    move = expected_move_usd(entry_price, tp_price, side, notional)
    if move <= 0:
        return False, {
            "reason": "no_expected_move",
            "tp_price": tp_price,
            "rt_cost_usd": round(rt_cost, 4),
        }
    threshold = n_mult * rt_cost
    admit = move >= threshold
    return admit, {
        "reason": "ok" if admit else "expected_move_below_threshold",
        "expected_move_usd": round(move, 4),
        "threshold_usd": round(threshold, 4),
        "rt_cost_usd": round(rt_cost, 4),
        "n_mult": n_mult,
    }


def summary() -> Dict[str, Any]:
    return {
        "n_policies": len(POLICIES),
        "n_disabled": sum(1 for v in POLICIES.values() if not v.get("live_enabled", True)),
        "n_shadow_only": sum(1 for v in POLICIES.values() if v.get("shadow_only", False)),
        "n_with_taker_cost_gate": sum(
            1 for v in POLICIES.values()
            if v.get("entry_type") == "maker_taker_dynamic"
            and v.get("taker_min_expected_move_rt_multiplier") is not None
        ),
        "disabled_pairs": [f"{s}|{sy}" for (s, sy), v in POLICIES.items()
                            if not v.get("live_enabled", True)],
    }

#!/usr/bin/env python3
"""Day-2 patcher — applies all three held patches via anchor-based replace.

Called from apply_day2_patches.sh. Reads paths + mode from env vars.

Env vars:
  DAY2_F_STRAT  — strategies/scalp_strategy.py
  DAY2_F_URM    — execution/user_real_manager.py
  DAY2_F_EXIT   — execution/exit_guards.py
  DAY2_F_REGIME — strategies/regime_filter.py
  DAY2_MODE     — 'apply' or 'dry-run'
  DAY2_PATCH    — '1' (volume gate), '2' (thresholds), '3' (maker sim),
                  '4' (A+ size cap), '5' (high_vol veto), '6' (asia_early veto)

Each patch is idempotent (sentinel-checked) and prints PATCH_OK or PATCH_SKIP.
Exits non-zero on any error.
"""
import os
import sys

MODE = os.environ.get("DAY2_MODE", "dry-run")
PATCH = os.environ.get("DAY2_PATCH", "1")
F_STRAT = os.environ.get("DAY2_F_STRAT", "")
F_URM = os.environ.get("DAY2_F_URM", "")
F_EXIT = os.environ.get("DAY2_F_EXIT", "")
F_REGIME = os.environ.get("DAY2_F_REGIME", "")


def apply_replace(path: str, old: str, new: str, label: str, sentinel: str = "") -> str:
    """Returns 'PATCHED' / 'SKIP_ALREADY' / aborts on missing anchor.

    Idempotency: if `sentinel` (a unique marker that ONLY appears in the new
    content) is already present in src, skip. Otherwise the old block must
    be present (else anchor error) and we replace.
    """
    src = open(path).read()
    if sentinel and sentinel in src:
        return "SKIP_ALREADY"
    if old not in src:
        print(f"ANCHOR_NOT_FOUND for {label} in {path}", file=sys.stderr)
        sys.exit(2)
    src2 = src.replace(old, new, 1)
    if MODE == "apply":
        open(path, "w").write(src2)
        return "PATCHED"
    else:
        return f"DRY-RUN (+{len(new) - len(old)} chars)"


def patch_1_volume_gate():
    old = '''        if best_vol > 1.5:
            confs.append(f"Volume spike {best_vol:.1f}\xd7")
            score += 15
        elif best_vol > 1.2:
            confs.append(f"Volume {best_vol:.1f}\xd7")
            score += 10
        elif best_vol > 0.9:
            score += 5
        else:
            # Low volume at structure = weak bounce, still allow but penalize
            score -= 5'''
    new = '''        # WYCKOFF_VOLUME_GATE_5_21 (2026-04-30) - hard veto on low volume.
        # Was: -5 score penalty. Was admitting ~78/133 trades in 0.0-0.15R
        # peak bucket per 24h shadow data - 1 win/78. Wyckoff: low vol = trap.
        if best_vol < 1.0:
            return None

        if best_vol > 1.5:
            confs.append(f"Volume spike {best_vol:.1f}\xd7")
            score += 15
        elif best_vol > 1.2:
            confs.append(f"Volume {best_vol:.1f}\xd7")
            score += 10
        else:
            confs.append(f"Volume {best_vol:.1f}\xd7 (at-median)")
            score += 5'''
    print(f"[P1 volume_gate] {apply_replace(F_STRAT, old, new, 'volume_gate', 'WYCKOFF_VOLUME_GATE_5_21')}")


def patch_2_thresholds():
    # 2A — max_age 600 -> 300 + new peak_floor_stall guard
    old_2a = '''                    if getattr(trade, "_is_shadow", False):
                        max_age = 600  # all shadow trade_types
                    else:
                        max_age = 600 if (trade.trade_type or "").upper() == "SCALP" else 3600
                if age_sec > max_age:
                    await self._close_trade(trade, price, f"time_decay_{int(age_sec/60)}m")
                    break'''
    new_2a = '''                    if getattr(trade, "_is_shadow", False):
                        # THRESHOLD_TWEAK_5_21 (2026-04-30) - 600->300s.
                        # 79/137 closed at 600s cap with 11% WR / -$91 net.
                        # Winners peak before 320s avg; cutting at 300s
                        # euthanizes duds without harming winners.
                        max_age = 300
                    else:
                        max_age = 600 if (trade.trade_type or "").upper() == "SCALP" else 3600
                if age_sec > max_age:
                    await self._close_trade(trade, price, f"time_decay_{int(age_sec/60)}m")
                    break

                # 6b. THRESHOLD_TWEAK_5_21 - peak-floor stall kill at 300s (shadow only).
                # 78/137 trades had peak<0.15R after 300s - 1 win. They are done.
                if age_sec >= 300 and float(trade.peak_mfe_r) < 0.15 and \\
                   getattr(trade, "_is_shadow", False):
                    await self._close_trade(trade, price, "peak_floor_stall")
                    break'''
    print(f"[P2A max_age+stall] {apply_replace(F_URM, old_2a, new_2a, 'p2a', 'THRESHOLD_TWEAK_5_21 (2026-04-30) - 600->300s')}")

    # 2B — _relaxed_shadow_exits gating on patient mode
    old_2b = '''        self._relaxed_shadow_exits = False  # disabled for clean test
        self._relaxed_shadow_simulation = False  # disabled for clean test'''
    new_2b = '''        # THRESHOLD_TWEAK_5_21 - gate relaxed_shadow_exits on patient mode.
        # Absorbs the 6bps shadow slippage hole; admin (patient) gets relaxed,
        # niranjan (standard) stays as control.
        _patient = (self.maker_patience_mode == "patient")
        self._relaxed_shadow_exits = _patient
        self._relaxed_shadow_simulation = False  # still disabled for clean test'''
    print(f"[P2B relaxed_shadow] {apply_replace(F_URM, old_2b, new_2b, 'p2b', 'THRESHOLD_TWEAK_5_21 - gate relaxed_shadow_exits')}")

    # 2C — exit_guards _STALL_AGE_SEC 900 -> 480
    old_2c = "_STALL_AGE_SEC = 900.0"
    new_2c = "_STALL_AGE_SEC = 480.0   # THRESHOLD_TWEAK_5_21 - was 900s, tightened to match new 300s max_age"
    print(f"[P2C STALL_AGE] {apply_replace(F_EXIT, old_2c, new_2c, 'p2c', 'THRESHOLD_TWEAK_5_21 - was 900s')}")


def patch_3_maker_sim():
    # 3A — entry block
    old_3a = '''        # Taker fill simulation (conservative — real maker rate will be better)
        if order_side == "buy":
            shadow_fill = float(book["asks"][0][0])
        else:
            shadow_fill = float(book["bids"][0][0])'''
    new_3a = '''        # MAKER_SIM_WIRING_5_21 (2026-04-30) - probabilistic maker fill via sim.
        # Gated by SHADOW_MAKER_SIM_ENABLED env var (default false). When false,
        # falls through to legacy taker-only behavior.
        # FIX 2026-05-01: import os locally — _os is a local in __init__ and
        # not in scope inside _execute_shadow. Use a unique alias to avoid
        # masking the existing _os usage (which is wrapped in try/except).
        import os as _os_p3
        from execution_v2.shadow_maker_sim import simulate_entry_fill
        _maker_sim_enabled = (
            _os_p3.getenv("SHADOW_MAKER_SIM_ENABLED", "false").lower() == "true"
        )
        _signal_id_for_sim = str(
            signal.get("id") or signal.get("signal_id")
            or f"{symbol}_{order_side}_{int(time.time()*1000)}"
        )
        if _maker_sim_enabled:
            _fill = simulate_entry_fill(
                order_side=order_side, book=book, symbol=symbol,
                our_lots=float(lots), tick_size=float(tick_size or 0.01),
                patience_mode=getattr(self, "maker_patience_mode", "standard"),
                signal_id=_signal_id_for_sim,
            )
            shadow_fill = _fill.fill_price
            _fee_type_used = _fill.fee_type
            _fee_pct_used = _fill.fee_pct
            if not isinstance(meta, dict):
                meta = {}
            meta["maker_sim_enabled"] = True
            meta["maker_sim_filled"] = _fill.filled_as_maker
            meta["maker_sim_mode"] = _fill.mode_used
            meta["maker_sim_p_fill"] = round(_fill.p_fill, 4)
        else:
            if order_side == "buy":
                shadow_fill = float(book["asks"][0][0])
            else:
                shadow_fill = float(book["bids"][0][0])
            _fee_type_used = "taker"
            _fee_pct_used = 0.00059'''
    print(f"[P3A entry_fill] {apply_replace(F_URM, old_3a, new_3a, 'p3a', 'MAKER_SIM_WIRING_5_21 (2026-04-30)')}")

    # 3B — entry fee uses sim-aware percent
    old_3b = '''        # Fee: 0.05% \xd7 1.18 GST = 0.059% taker
        notional = shadow_fill * lots * trade_contract_size
        shadow_entry_fee = notional * 0.00059'''
    new_3b = '''        # Fee: variable based on maker/taker outcome (MAKER_SIM_WIRING_5_21)
        notional = shadow_fill * lots * trade_contract_size
        shadow_entry_fee = notional * _fee_pct_used'''
    print(f"[P3B entry_fee] {apply_replace(F_URM, old_3b, new_3b, 'p3b', 'MAKER_SIM_WIRING_5_21)')}")

    # 3C — fee_type on trade record (carefully scoped)
    src = open(F_URM).read()
    sentinel_after = '''            entry_fee_usd=float(shadow_entry_fee),
            funding_rate_at_entry=float(_funding_rate_snap),
        )'''
    idx = src.find(sentinel_after)
    if idx == -1:
        print("[P3C] ANCHOR (entry_fee_usd block) NOT FOUND", file=sys.stderr)
        sys.exit(2)
    pre = src[max(0, idx - 500):idx]
    if 'fee_type="taker",' not in pre and "fee_type='taker'," not in pre:
        print("[P3C] fee_type=taker not in pre-block (already patched?)")
        return
    new_pre = pre.replace(
        'fee_type="taker",',
        'fee_type=_fee_type_used,    # MAKER_SIM_WIRING_5_21',
        1,
    )
    src2 = src[:max(0, idx - 500)] + new_pre + src[idx:]
    if MODE == "apply":
        open(F_URM, "w").write(src2)
        print("[P3C fee_type_field] PATCHED")
    else:
        print("[P3C fee_type_field] DRY-RUN")


# ──────────────────────────────────────────────────────────────────────
# Patch 4 — A+ size cap (subtraction-only, no W/F needed)
# ──────────────────────────────────────────────────────────────────────
def patch_4_a_plus_size_cap():
    """Cap A+ size multiplier 1.3 → 1.0. See docs/patch_a_plus_size_cap.md."""
    old = '''    if tier == "strong":
        if confidence >= 90:
            return 1.3  # exceptional signal
        return 1.1'''
    new = '''    if tier == "strong":
        # AB_SIZE_CAP_5_22 (2026-05-01) — was 1.3 boost on conf>=90 (A+).
        # 24h shadow data: A+ x ML 0.80+ = 32 trades, 22% WR, -$1.20 avg.
        # ML calibration is anti-predictive at top. Capping all "strong"
        # at 1.0 removes the A+ amplifier without changing trade selection.
        # Estimated saving: $25-35/24h.
        return 1.0'''
    print(f"[P4 a_plus_size_cap] {apply_replace(F_REGIME, old, new, 'p4', 'AB_SIZE_CAP_5_22')}")


# ──────────────────────────────────────────────────────────────────────
# Patch 5 — high_volatility hard-veto for structure_bounce
# ──────────────────────────────────────────────────────────────────────
def patch_5_high_vol_veto():
    """Add HIGH_VOL_REGIME_VETO_SB to structure_bounce hard-veto list + emit logic.
    See docs/patch_high_vol_veto.md."""
    # Edit 5A — extend the hard-prefix tuple
    old_5a = '''        sb_hard_prefixes = ("CHOCH CONFLICT:", "REGIME MISMATCH:", "REGIME SIDE:", "STRICT 4H VETO:", "AP_SB_SHORT_KILL:", "BAD_EDGE_SCANNER_KILL:", "DEAD_HOUR_KILL:", "MACRO_EMA200_VETO:", "MACD_DIV_VETO:", "HTF_HARD_VETO_AGRADE:", "VOLUME_CLIMAX_VETO:", "WEDGE_BREAKOUT_VETO:")'''
    new_5a = '''        # HIGH_VOL_VETO_SB_5_22 (2026-05-01) — added HIGH_VOL_REGIME_VETO_SB
        # to the hard-veto list. 23 high_volatility structure_bounce trades
        # in 24h cost -$37 (avg -$1.61/trade, 9% WR). Mean-reversion has no
        # edge when levels get blown through.
        sb_hard_prefixes = ("CHOCH CONFLICT:", "REGIME MISMATCH:", "REGIME SIDE:", "STRICT 4H VETO:", "AP_SB_SHORT_KILL:", "BAD_EDGE_SCANNER_KILL:", "DEAD_HOUR_KILL:", "MACRO_EMA200_VETO:", "MACD_DIV_VETO:", "HTF_HARD_VETO_AGRADE:", "VOLUME_CLIMAX_VETO:", "WEDGE_BREAKOUT_VETO:", "HIGH_VOL_REGIME_VETO_SB:")'''
    print(f"[P5A hard_prefix] {apply_replace(F_STRAT, old_5a, new_5a, 'p5a', 'HIGH_VOL_VETO_SB_5_22')}")

    # Edit 5B — emit the veto right after `is_sb` flag is set.
    # Anchor on the assignment + the next existing comment block.
    old_5b = '''        is_sb = best_sr.scanner_name == "structure_bounce"

        # For structure_bounce: only HTF, CHOCH, and REGIME MISMATCH are hard vetos'''
    new_5b = '''        is_sb = best_sr.scanner_name == "structure_bounce"

        # HIGH_VOL_VETO_SB_5_22 — block structure_bounce in high_volatility regime.
        # Data 2026-05-01: 23 such trades = 9% WR, -$1.61 avg, -$37/24h.
        try:
            _regime_now_p5 = (locals().get('regime') or getattr(self, '_last_regime_str', '') or '').lower()
        except Exception:
            _regime_now_p5 = ''
        if is_sb and _regime_now_p5 == "high_volatility":
            vetos.append(f"HIGH_VOL_REGIME_VETO_SB: structure_bounce blocked in high_volatility regime")

        # For structure_bounce: only HTF, CHOCH, and REGIME MISMATCH are hard vetos'''
    print(f"[P5B emit_veto] {apply_replace(F_STRAT, old_5b, new_5b, 'p5b', 'HIGH_VOL_VETO_SB_5_22 — block structure_bounce')}")


# ──────────────────────────────────────────────────────────────────────
# Patch 6 — Asia-early veto extension
# ──────────────────────────────────────────────────────────────────────
def patch_6_asia_early_veto():
    """Extend dead_hours_hard from {2,3,4,5} to {2,3,4,5,6,7} and fix
    IST→UTC integer conversion bug.  See docs/patch_asia_early_veto.md."""
    old = '''                ist_now_check = datetime.now(_IST)
                utc_hour = (ist_now_check.hour - 5) % 24
                dead_hours_hard = {2, 3, 4, 5}    # genuinely dead — hard block
                dead_hours_soft = {10, 11}         # soft penalty only
                if utc_hour in dead_hours_hard:
                    vetos.append(f"DEAD SESSION: UTC hour {utc_hour} (2-5 UTC low liquidity)")'''
    new = '''                # ASIA_EARLY_VETO_5_22 (2026-05-01) — extended hard-block hours.
                # 24h shadow data: UTC 04-07 = 30 trades, 10% WR, -$22.14.
                # Project weakspot map flags asia_early as -27 to -34pp WR.
                # Use proper UTC datetime to avoid IST integer-subtract bugs.
                from datetime import datetime as _dt_p6, timezone as _tz_p6
                utc_hour = _dt_p6.now(_tz_p6.utc).hour
                dead_hours_hard = {2, 3, 4, 5, 6, 7}  # was {2,3,4,5}; extended for asia-early bleed
                dead_hours_soft = {10, 11}             # soft penalty only
                if utc_hour in dead_hours_hard:
                    vetos.append(f"DEAD SESSION: UTC hour {utc_hour} (asia-early low liquidity)")'''
    print(f"[P6 asia_early] {apply_replace(F_STRAT, old, new, 'p6', 'ASIA_EARLY_VETO_5_22')}")


if __name__ == "__main__":
    if PATCH == "1":
        patch_1_volume_gate()
    elif PATCH == "2":
        patch_2_thresholds()
    elif PATCH == "3":
        patch_3_maker_sim()
    elif PATCH == "4":
        patch_4_a_plus_size_cap()
    elif PATCH == "5":
        patch_5_high_vol_veto()
    elif PATCH == "6":
        patch_6_asia_early_veto()
    else:
        print(f"Unknown DAY2_PATCH={PATCH}", file=sys.stderr)
        sys.exit(1)

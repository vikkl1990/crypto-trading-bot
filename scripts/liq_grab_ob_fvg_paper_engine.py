#!/usr/bin/env python3
"""SOL Liquidity Grab + OB Reclaim + FVG Fill paper engine — 5-step SMC.

W/F-validated 2026-05-02 (storage/wf_studies/liq_grab_ob_fvg/):
  Variant C (maker entry + scalper offer):
    cell `disp_atr=0.7 fill_disp=0.4 reclaim_lookback=5 regime_max=0.6 sweep_atr=0.1`
    IS Q1+Q2 :  n=97,  EV/trade=$+0.027
    OOS Q3   :         EV/trade=$+1.669
    OOS Q4   :         EV/trade=$+0.569 (vs basic liq_sweep_htf $+0.238 — 2.4× better)
    WR_oos:    58%
  Verdict: PASS walk-forward (12 PASS cells, all SOL/Variant C).

Setup spec (5-step SMC confluence — LONG; mirror for SHORT):
  Step 1 — LIQUIDITY GRAB: 5m bar sweeps below 20-bar low by ≥0.1× ATR
                            AND closes bearish (close < open)
                            AND body ≥ 0.7× ATR (strong displacement)
  Step 2 — OB IDENTIFY: scan back 5 bars from sweep, find LAST BULLISH
                         candle (close > open) → "proximal Order Block"
                         (range = OB_high to OB_low)
  Step 3 — OB RECLAIM: within next 5 bars after sweep, a bar must close
                        ≥ OB_high (price reclaims the OB level)
  Step 4 — FVG FILL: a Fair Value Gap (3-candle pattern: bar j+1 low > bar j-1 high)
                      forms in the reclaim sequence AND is FILLED with a
                      bullish displacement candle (close > open, body ≥ 0.4× ATR)
  Step 5 — ENTRY: at the FVG-fill candle close → maker LIMIT at OB_high level
  STOP : 1× ATR below entry  (target stop loss preserves R)
  TP   : 2× ATR above entry  (1:2 RR target)
  TIME : 13min hard stop (SOL is non-BTC/ETH; scalper window=15min)

Symbols: SOL/USDT only (W/F-validated symbol; family=liquid_majors).

Paper engine architecture mirrors scalper_vwap_mr_paper_engine.py and
liq_sweep_htf_paper_engine.py.  State dir: storage/liq_grab_ob_fvg_paper/

Phase-2 candidate IF this engine produces live results consistent with
Q4 W/F prediction (~$+0.57/trade, ~58% WR over ~30+ trades): promote
to production scanner via the SMC stack.

Independent — does NOT touch scalp_strategy.py, the live bot, or any
signal_tracker / signal_journey / signal_learner code.  Default OFF
unless explicitly enabled via cron.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

import sys as _sys_telem
_sys_telem.path.insert(0, "/home/opc/crypto-trading-bot")
from execution_v2.engine_telemetry import log_eval

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_v2.fee_model import FeeModel  # noqa: E402

STATE_DIR = ROOT / "storage" / "liq_grab_ob_fvg_paper"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
SUMMARY_FILE = STATE_DIR / "latest.md"
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"

SYMBOLS = {
    "SOL/USDT": "SOLUSD",
}

# Strategy params (LOCKED to W/F-passing cell)
TF = "5m"
TF_RES = "5m"
TF_MIN = 5
DISP_ATR = 0.7              # sweep candle body ≥ 0.7× ATR
SWEEP_ATR = 0.1             # extreme breach ≥ 0.1× ATR
RECLAIM_LOOKBACK = 5        # bars after sweep to find OB reclaim
FILL_DISP = 0.4             # FVG-fill candle body ≥ 0.4× ATR
OB_LOOKBACK = 5             # bars before sweep to scan for OB
REGIME_MAX = 0.60           # 4h ATR pct rank gate
SL_ATR_MULT = 1.0
TP_ATR_MULT = 1.5            # 2026-05-02 W/F refined: Q4 EV +bash.69 vs 2.0 +bash.57, WR 63% vs 58%
TIME_STOP_MIN = 13          # 15min scalper window − 2min margin
NOTIONAL_USD = 1000.0
LIMIT_EXPIRY_MIN = TF_MIN   # 5min — cancel pending limit if not filled
ROLL_LOOKBACK = 20
ATR_PERIOD = 14
HOURS_BACK_5M = 24 * 3      # 3 days of 5m
HOURS_BACK_4H = 24 * 30     # 30 days of 4h


# ──────────────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────────────
def fetch_candles(sym_rest: str, resolution: str, hours_back: int) -> pd.DataFrame:
    end = int(time.time())
    start = end - hours_back * 3600
    try:
        r = requests.get(DELTA_REST,
                         params={"symbol": sym_rest, "resolution": resolution,
                                 "start": start, "end": end},
                         timeout=10)
        rows = r.json().get("result", [])
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def get_current_price(sym_rest: str) -> Optional[float]:
    end = int(time.time())
    start = end - 600
    try:
        r = requests.get(DELTA_REST,
                         params={"symbol": sym_rest, "resolution": "1m",
                                 "start": start, "end": end},
                         timeout=10)
        rows = r.json().get("result", [])
    except Exception:
        return None
    if not rows:
        return None
    return float(rows[-1].get("close", 0)) or None


# ──────────────────────────────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high, low, close = out["high"], out["low"], out["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    out["roll_high"] = high.rolling(ROLL_LOOKBACK).max().shift(1)
    out["roll_low"]  = low.rolling(ROLL_LOOKBACK).min().shift(1)
    return out


def get_htf_atr_pct(df_4h: pd.DataFrame) -> float:
    if df_4h.empty or len(df_4h) < 50:
        return float("nan")
    df = df_4h.copy()
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    pct_rank = df["atr"].rolling(100, min_periods=50).rank(pct=True)
    last = pct_rank.iloc[-1]
    return float(last) if not pd.isna(last) else float("nan")


# ──────────────────────────────────────────────────────────────────────
# SMC pattern detection (5-step)
# ──────────────────────────────────────────────────────────────────────
def find_proximal_ob_long(df: pd.DataFrame, sweep_idx: int, lookback: int) -> Optional[Tuple[float, float]]:
    """For LONG setup: find last BULLISH bar before sweep — proximal OB."""
    for j in range(sweep_idx - 1, max(0, sweep_idx - lookback - 1), -1):
        bar = df.iloc[j]
        if float(bar["close"]) > float(bar["open"]):
            return (float(bar["high"]), float(bar["low"]))
    return None


def find_proximal_ob_short(df: pd.DataFrame, sweep_idx: int, lookback: int) -> Optional[Tuple[float, float]]:
    """For SHORT setup: find last BEARISH bar before sweep."""
    for j in range(sweep_idx - 1, max(0, sweep_idx - lookback - 1), -1):
        bar = df.iloc[j]
        if float(bar["close"]) < float(bar["open"]):
            return (float(bar["high"]), float(bar["low"]))
    return None


def detect_fvg_long(df: pd.DataFrame, start_idx: int, end_idx: int
                     ) -> Optional[Tuple[int, float, float]]:
    """Bullish FVG: bar j+1.low > bar j-1.high (3-candle gap-up)."""
    for j in range(start_idx + 1, min(end_idx, len(df) - 1)):
        prev_high = float(df.iloc[j - 1]["high"])
        next_low = float(df.iloc[j + 1]["low"])
        if next_low > prev_high:
            return (j, prev_high, next_low)
    return None


def detect_fvg_short(df: pd.DataFrame, start_idx: int, end_idx: int
                      ) -> Optional[Tuple[int, float, float]]:
    """Bearish FVG: bar j+1.high < bar j-1.low (3-candle gap-down)."""
    for j in range(start_idx + 1, min(end_idx, len(df) - 1)):
        prev_low = float(df.iloc[j - 1]["low"])
        next_high = float(df.iloc[j + 1]["high"])
        if next_high < prev_low:
            return (j, next_high, prev_low)
    return None


def fvg_filled_long(df: pd.DataFrame, fvg_idx: int, gap_low: float, gap_high: float,
                     end_idx: int, atr: float, fill_disp: float) -> Optional[int]:
    """Bar that fills bullish FVG with bullish displacement."""
    for j in range(fvg_idx + 1, min(end_idx, len(df))):
        bar = df.iloc[j]
        if float(bar["low"]) <= gap_high:
            body = abs(float(bar["close"]) - float(bar["open"]))
            if float(bar["close"]) > float(bar["open"]) and body >= fill_disp * atr:
                return j
    return None


def fvg_filled_short(df: pd.DataFrame, fvg_idx: int, gap_low: float, gap_high: float,
                      end_idx: int, atr: float, fill_disp: float) -> Optional[int]:
    for j in range(fvg_idx + 1, min(end_idx, len(df))):
        bar = df.iloc[j]
        if float(bar["high"]) >= gap_low:
            body = abs(float(bar["close"]) - float(bar["open"]))
            if float(bar["close"]) < float(bar["open"]) and body >= fill_disp * atr:
                return j
    return None


# ──────────────────────────────────────────────────────────────────────
# Signal detection
# ──────────────────────────────────────────────────────────────────────
def detect_signal(df_5m: pd.DataFrame, htf_atr_pct: float) -> Optional[dict]:
    """Detect the 5-step SMC confluence on the most recent CLOSED bars.

    Returns dict {side, limit_price, sl, tp, atr, sweep_low, ob_high,
                  ob_low, fvg_low, fvg_high, signal_ts} or None.
    """
    if pd.isna(htf_atr_pct) or htf_atr_pct > REGIME_MAX:
        return None

    if len(df_5m) < ROLL_LOOKBACK + RECLAIM_LOOKBACK + 10:
        return None
    df = add_indicators(df_5m)
    n = len(df)

    # We need to look back enough for the full pattern to play out.
    # The sweep bar must be old enough that reclaim + FVG fill can complete.
    # Earliest sweep_idx we can confirm: n - (RECLAIM_LOOKBACK + 5) - 1
    earliest_sweep = max(ROLL_LOOKBACK + 2, n - (RECLAIM_LOOKBACK + 5) - 1)
    latest_sweep = n - (RECLAIM_LOOKBACK + 3)  # need room for fill
    if latest_sweep <= earliest_sweep:
        return None

    # Walk backwards from latest_sweep — first match wins (most recent setup)
    for sweep_idx in range(latest_sweep, earliest_sweep - 1, -1):
        sweep_bar = df.iloc[sweep_idx]
        atr = sweep_bar["atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        roll_high = sweep_bar["roll_high"]
        roll_low = sweep_bar["roll_low"]
        if pd.isna([roll_high, roll_low]).any():
            continue

        o = float(sweep_bar["open"])
        h = float(sweep_bar["high"])
        l = float(sweep_bar["low"])
        c = float(sweep_bar["close"])
        body = abs(c - o)

        # ─── LONG: bearish sweep below low ───
        sweep_long = (l < roll_low - SWEEP_ATR * atr) and (c < o) and (body >= DISP_ATR * atr)
        if sweep_long:
            ob = find_proximal_ob_long(df, sweep_idx, lookback=OB_LOOKBACK)
            if ob is None: continue
            ob_high, ob_low = ob

            # OB reclaim
            reclaim_idx = None
            for j in range(sweep_idx + 1, min(sweep_idx + 1 + RECLAIM_LOOKBACK, n)):
                if float(df.iloc[j]["close"]) >= ob_high:
                    reclaim_idx = j; break
            if reclaim_idx is None: continue

            # FVG in [sweep_idx, reclaim_idx + 2]
            fvg = detect_fvg_long(df, sweep_idx, reclaim_idx + 2)
            if fvg is None: continue
            fvg_idx, gap_low, gap_high = fvg

            # FVG filled with bullish displacement
            fill_idx = fvg_filled_long(df, fvg_idx, gap_low, gap_high,
                                        min(reclaim_idx + 5, n), atr, FILL_DISP)
            if fill_idx is None: continue
            # Only consider this signal if fill_idx is the LAST closed bar
            # (we just got the confirmation — fresh signal)
            if fill_idx != n - 1:
                continue

            entry = float(df.iloc[fill_idx]["close"])
            sl = entry - SL_ATR_MULT * atr
            tp = entry + TP_ATR_MULT * atr
            return {
                "side": "long",
                "limit_price": entry,
                "sl": sl, "tp": tp,
                "atr": float(atr),
                "sweep_low": l,
                "ob_high": ob_high, "ob_low": ob_low,
                "fvg_low": gap_low, "fvg_high": gap_high,
                "htf_atr_pct": htf_atr_pct,
                "signal_ts": df.index[fill_idx].isoformat(),
                "reclaim_lag": fill_idx - sweep_idx,
            }

        # ─── SHORT: bullish sweep above high ───
        sweep_short = (h > roll_high + SWEEP_ATR * atr) and (c > o) and (body >= DISP_ATR * atr)
        if sweep_short:
            ob = find_proximal_ob_short(df, sweep_idx, lookback=OB_LOOKBACK)
            if ob is None: continue
            ob_high, ob_low = ob

            reclaim_idx = None
            for j in range(sweep_idx + 1, min(sweep_idx + 1 + RECLAIM_LOOKBACK, n)):
                if float(df.iloc[j]["close"]) <= ob_low:
                    reclaim_idx = j; break
            if reclaim_idx is None: continue

            fvg = detect_fvg_short(df, sweep_idx, reclaim_idx + 2)
            if fvg is None: continue
            fvg_idx, gap_low, gap_high = fvg

            fill_idx = fvg_filled_short(df, fvg_idx, gap_low, gap_high,
                                         min(reclaim_idx + 5, n), atr, FILL_DISP)
            if fill_idx is None: continue
            if fill_idx != n - 1:
                continue

            entry = float(df.iloc[fill_idx]["close"])
            sl = entry + SL_ATR_MULT * atr
            tp = entry - TP_ATR_MULT * atr
            return {
                "side": "short",
                "limit_price": entry,
                "sl": sl, "tp": tp,
                "atr": float(atr),
                "sweep_high": h,
                "ob_high": ob_high, "ob_low": ob_low,
                "fvg_low": gap_low, "fvg_high": gap_high,
                "htf_atr_pct": htf_atr_pct,
                "signal_ts": df.index[fill_idx].isoformat(),
                "reclaim_lag": fill_idx - sweep_idx,
            }

    return None


# ──────────────────────────────────────────────────────────────────────
# State / IO  (mirror liq_sweep_htf_paper_engine.py)
# ──────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open_trades": [], "pending_limits": [], "last_eval_candle": {},
            "history_pnl": 0.0, "history_n": 0, "history_wins": 0,
            "history_scalper_applied": 0}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_trade(trade: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with TRADES_FILE.open("a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def close_trade(trade: dict, exit_price: float, reason: str) -> dict:
    side = trade["side"]; entry = trade["entry"]; notional = trade["notional"]
    holding_sec = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(trade["opened_at"])).total_seconds()
    gross_pct = (exit_price - entry) / entry if side == "long" else (entry - exit_price) / entry
    gross_usd = gross_pct * notional
    fee_model = FeeModel()
    fee_info = fee_model.round_trip_for_trade(
        exchange="delta", entry_type="maker", exit_type="taker",
        notional_usd=notional, symbol=trade["symbol"],
        holding_sec=holding_sec, force_no_scalper=False,
    )
    net_usd = gross_usd - fee_info["fee_usd"]
    trade["closed_at"] = datetime.now(timezone.utc).isoformat()
    trade["exit_price"] = exit_price
    trade["close_reason"] = reason
    trade["holding_sec"] = round(holding_sec, 1)
    trade["gross_pnl_usd"] = round(gross_usd, 4)
    trade["fee_usd"] = round(fee_info["fee_usd"], 4)
    trade["fee_pct"] = round(fee_info["fee_pct"] * 100, 4)
    trade["scalper_applied"] = bool(fee_info["scalper_applied"])
    trade["net_pnl_usd"] = round(net_usd, 4)
    trade["status"] = "closed"
    return trade


def manage_open_trade(trade: dict, sym_rest: str) -> dict:
    px = get_current_price(sym_rest)
    if px is None: return trade
    opened = datetime.fromisoformat(trade["opened_at"])
    if opened.tzinfo is None: opened = opened.replace(tzinfo=timezone.utc)
    age_min = (datetime.now(timezone.utc) - opened).total_seconds() / 60.0

    R = abs(trade["entry"] - trade["sl"])
    cur_r = ((px - trade["entry"]) / R) if trade["side"] == "long" else \
            ((trade["entry"] - px) / R)
    trade["peak_mfe_r"] = max(trade.get("peak_mfe_r", 0.0), cur_r)
    trade["last_price"] = px
    trade["last_check"] = datetime.now(timezone.utc).isoformat()

    # Hard time stop FIRST — preserve scalper-offer eligibility
    if age_min >= TIME_STOP_MIN:
        return close_trade(trade, px, f"time_stop_{TIME_STOP_MIN}min")
    if trade["side"] == "long" and px <= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")
    if trade["side"] == "short" and px >= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")
    if trade["side"] == "long" and px >= trade["tp"]:
        return close_trade(trade, trade["tp"], "tp_hit")
    if trade["side"] == "short" and px <= trade["tp"]:
        return close_trade(trade, trade["tp"], "tp_hit")
    return trade


def manage_pending_limit(pending: dict, sym_rest: str
                         ) -> tuple[Optional[dict], Optional[dict]]:
    px = get_current_price(sym_rest)
    if px is None: return pending, None
    placed = datetime.fromisoformat(pending["placed_at"])
    if placed.tzinfo is None: placed = placed.replace(tzinfo=timezone.utc)
    age_min = (datetime.now(timezone.utc) - placed).total_seconds() / 60.0
    if age_min >= LIMIT_EXPIRY_MIN: return None, None

    pending["last_price"] = px
    pending["last_check"] = datetime.now(timezone.utc).isoformat()

    fill = False
    if pending["side"] == "long" and px <= pending["limit_price"]: fill = True
    if pending["side"] == "short" and px >= pending["limit_price"]: fill = True
    if not fill: return pending, None

    entry = pending["limit_price"]
    trade = {
        "id": f"liqgrab_{int(time.time())}_{pending['symbol'].replace('/', '')}",
        "symbol": pending["symbol"], "side": pending["side"],
        "entry": entry, "sl": pending["sl"], "tp": pending["tp"],
        "atr_at_entry": pending["atr"],
        "ob_high_at_entry": pending["ob_high"],
        "ob_low_at_entry": pending["ob_low"],
        "fvg_low_at_entry": pending["fvg_low"],
        "fvg_high_at_entry": pending["fvg_high"],
        "htf_atr_pct_at_entry": pending["htf_atr_pct"],
        "reclaim_lag_at_entry": pending.get("reclaim_lag"),
        "notional": NOTIONAL_USD,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "candle_time": pending["signal_ts"],
        "limit_price": pending["limit_price"],
        "peak_mfe_r": 0.0, "status": "open",
    }
    # PATCH_L_5_22 — bridge new trade to shadow execution queue
    try:
        from bot.shadow_bridge import publish_signal as _bridge_pub
        _bridge_pub(
            source_engine="liq_grab_ob_fvg",
            symbol=trade["symbol"],
            side=trade["side"],
            entry_price=trade["entry"],
            stop_loss=trade["sl"],
            take_profit=trade.get("tp"),
            ml_probability=0.70,
            grade="A",
            setup_type="liq_grab_ob_fvg",
            confidence=80.0,
            regime="bridge_engine",
            extra_meta={
                "ob_high": trade.get("ob_high_at_entry"),
                "fvg_low": trade.get("fvg_low_at_entry"),
                "engine_trade_id": trade.get("id"),
            },
        )
    except Exception:
        pass  # fail open — paper engine continues regardless
    return None, trade


# ──────────────────────────────────────────────────────────────────────
# Main eval
# ──────────────────────────────────────────────────────────────────────
def eval_signals(state: dict) -> None:
    open_by_sym = {t["symbol"] for t in state.get("open_trades", [])}
    pending_by_sym = {p["symbol"] for p in state.get("pending_limits", [])}
    last_eval = state.setdefault("last_eval_candle", {})

    for sym, sym_rest in SYMBOLS.items():
        if sym in open_by_sym or sym in pending_by_sym:
            continue
        df_5m = fetch_candles(sym_rest, "5m", HOURS_BACK_5M)
        if df_5m.empty or len(df_5m) < ROLL_LOOKBACK + RECLAIM_LOOKBACK + 10:
            print(f"  {sym}: insufficient 5m data ({len(df_5m)} bars)")
            continue
        df_4h = fetch_candles(sym_rest, "4h", HOURS_BACK_4H)
        htf_atr_pct = get_htf_atr_pct(df_4h)

        last_closed_ts = df_5m.index[-1].isoformat()
        if last_eval.get(sym) == last_closed_ts:
            try:
                log_eval(engine_name="liq_grab_ob_fvg", symbol=sym, base_dir=STATE_DIR,
                         signal_fired=False,
                         state={"rejection_reason": "ALREADY_EVALUATED",
                                "htf_atr_pct": htf_atr_pct})
            except Exception: pass
            continue
        last_eval[sym] = last_closed_ts

        sig = detect_signal(df_5m, htf_atr_pct)

        try:
            _state = {"htf_atr_pct": round(htf_atr_pct, 3) if not pd.isna(htf_atr_pct) else None,
                      "rejection_reason": None if sig else "NO_SMC_5STEP_CONFLUENCE"}
            if sig:
                _state.update({k: sig[k] for k in
                               ("side", "limit_price", "sl", "tp",
                                "ob_high", "ob_low", "fvg_low", "fvg_high")})
            log_eval(engine_name="liq_grab_ob_fvg", symbol=sym, base_dir=STATE_DIR,
                     signal_fired=(sig is not None), state=_state)
        except Exception: pass

        if not sig: continue

        pending = {
            "symbol": sym, "side": sig["side"],
            "limit_price": sig["limit_price"], "sl": sig["sl"], "tp": sig["tp"],
            "atr": sig["atr"],
            "ob_high": sig["ob_high"], "ob_low": sig["ob_low"],
            "fvg_low": sig["fvg_low"], "fvg_high": sig["fvg_high"],
            "htf_atr_pct": sig["htf_atr_pct"],
            "reclaim_lag": sig.get("reclaim_lag"),
            "signal_ts": sig["signal_ts"],
            "placed_at": datetime.now(timezone.utc).isoformat(),
        }
        state.setdefault("pending_limits", []).append(pending)
        print(f"  {sym}: SMC SIGNAL {sig['side']} limit={sig['limit_price']:.4f} "
              f"sl={sig['sl']:.4f} tp={sig['tp']:.4f} ob=[{sig['ob_low']:.4f},{sig['ob_high']:.4f}] "
              f"fvg=[{sig['fvg_low']:.4f},{sig['fvg_high']:.4f}] htf_pct={htf_atr_pct:.2f}")


def write_summary(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    n = state.get("history_n", 0); w = state.get("history_wins", 0)
    pnl = state.get("history_pnl", 0.0); sa = state.get("history_scalper_applied", 0)
    open_n = len(state.get("open_trades", [])); pending_n = len(state.get("pending_limits", []))
    wr = (100 * w / n) if n else 0; ev = (pnl / n) if n else 0
    sa_pct = (100 * sa / n) if n else 0
    SUMMARY_FILE.write_text(
        f"# SOL Liquidity Grab + OB Reclaim + FVG Fill Paper Engine\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"## Cumulative Results\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Closed trades | {n} |\n"
        f"| Wins | {w} ({wr:.1f}%) |\n"
        f"| Net PnL | ${pnl:.2f} |\n"
        f"| EV per trade | ${ev:.3f} |\n"
        f"| Scalper offer applied | {sa}/{n} ({sa_pct:.0f}%) |\n"
        f"| Open trades | {open_n} |\n"
        f"| Pending limits | {pending_n} |\n\n"
        f"## W/F Reference\n\n"
        f"Q4 OOS EV per trade (predicted): $+0.569\n"
        f"WR_oos (predicted): 58%\n"
    )


def main():
    state = load_state()
    print(f"=== liq_grab_ob_fvg paper engine — {datetime.now(timezone.utc).isoformat()} ===")

    new_open = []
    for trade in state.get("open_trades", []):
        sym_rest = SYMBOLS.get(trade["symbol"])
        if sym_rest is None:
            new_open.append(trade); continue
        managed = manage_open_trade(trade, sym_rest)
        if managed.get("status") == "closed":
            append_trade(managed)
            state["history_pnl"] = round(state.get("history_pnl", 0)
                                         + managed["net_pnl_usd"], 4)
            state["history_n"] = state.get("history_n", 0) + 1
            if managed["net_pnl_usd"] > 0:
                state["history_wins"] = state.get("history_wins", 0) + 1
            if managed.get("scalper_applied"):
                state["history_scalper_applied"] = state.get("history_scalper_applied", 0) + 1
            print(f"  CLOSED {managed['symbol']} {managed['side']} "
                  f"@${managed['exit_price']:.4f} {managed['close_reason']} "
                  f"net=${managed['net_pnl_usd']:.4f}")
        else:
            new_open.append(managed)
    state["open_trades"] = new_open

    new_pending = []
    for pending in state.get("pending_limits", []):
        sym_rest = SYMBOLS.get(pending["symbol"])
        if sym_rest is None: continue
        still, opened = manage_pending_limit(pending, sym_rest)
        if opened is not None:
            state["open_trades"].append(opened)
            print(f"  FILLED {opened['symbol']} {opened['side']} @${opened['entry']:.4f}")
        elif still is not None:
            new_pending.append(still)
    state["pending_limits"] = new_pending

    eval_signals(state)
    save_state(state)
    write_summary(state)
    print(f"=== done. open={len(state['open_trades'])} pending={len(state['pending_limits'])} ===")


if __name__ == "__main__":
    main()

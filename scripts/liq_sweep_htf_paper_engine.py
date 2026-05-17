#!/usr/bin/env python3
"""SOL Liquidity-Sweep + HTF Regime paper engine.

Walk-forward-validated 2026-04-30 (storage/wf_studies/liquidity_sweep_htf):
  Variant C (maker entry + scalper offer):
    cell `disp_atr=0.6 sweep_atr=0.2 regime_max=0.6`
    IS Q1+Q2 :  n=122, EV/trade=$+0.292
    OOS Q3   :  n=?,   EV/trade=$+1.201 (regime gate cleaner in Q3)
    OOS Q4   :  n=?,   EV/trade=$+0.238 (PASS — above $0.10 floor)
    WR_oos: 59%
  Verdict: PASS walk-forward (both OOS positive, Q4 beats $0.10 ship gate).

Total of 7 PASS cells found, all SOL/USDT Variant C, regime_max ≤ 0.8.
This engine pins the most robust cell (largest IS_n, balanced disp_atr).

Strategy spec (mirrors W/F simulator exactly):
  Trading TF        : 5m
  HTF context       : 4h ATR_14 percentile rank ≤ 0.60 (skip high-vol chop)
  LONG entry rule   : 5m bar sweeps below 20-bar low by ≥ 0.2 × ATR_5m
                      AND closes above prev bar low
                      AND bullish reversal (close > open)
                      AND body ≥ 0.6 × ATR_5m
                      → LIMIT BUY at signal-bar close (maker entry)
  SHORT entry rule  : mirror at 20-bar high
  Stop              : 1 × ATR_5m on the wrong side
  Target            : 2 × ATR_5m (≈ 2R)
  Time stop         : 13 minutes (SOL is non-BTC/ETH; scalper window=15min)
  Notional          : $1000
  Symbols           : SOL/USDT only (W/F-validated symbol)

Paper engine architecture mirrors scalper_vwap_mr_paper_engine.py.
  State dir:      storage/liq_sweep_htf_paper/
  State file:     state.json
  Trades file:    trades.jsonl
  Summary file:   latest.md
  Cron:           */5 * * * *

Independent — does NOT touch scalp_strategy.py, the live bot, or any
signal_tracker / signal_journey / signal_learner code.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import sys as _sys_telem
_sys_telem.path.insert(0, "/home/opc/crypto-trading-bot")
from execution_v2.engine_telemetry import log_eval

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_v2.fee_model import FeeModel  # noqa: E402

STATE_DIR = ROOT / "storage" / "liq_sweep_htf_paper"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
SUMMARY_FILE = STATE_DIR / "latest.md"
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"

SYMBOLS = {
    "SOL/USDT": "SOLUSD",
}

# Strategy params (locked to W/F-passing cell)
TF = "5m"
TF_RES = "5m"
TF_MIN = 5
DISP_ATR = 0.6           # body ≥ 0.6 × ATR
SWEEP_ATR = 0.2          # extreme breach ≥ 0.2 × ATR
REGIME_MAX = 0.60        # 4h ATR pct rank gate
SL_ATR_MULT = 1.0
TP_ATR_MULT = 2.0
TIME_STOP_MIN = 13       # 15min scalper window − 2min margin (non-BTC/ETH)
NOTIONAL_USD = 1000.0
LIMIT_EXPIRY_MIN = TF_MIN  # 5min — cancel pending limit if not filled
ROLL_LOOKBACK = 20
ATR_PERIOD = 14
HOURS_BACK_5M = 24 * 3   # 3 days of 5m = 864 bars (>> 20 lookback)
HOURS_BACK_4H = 24 * 30  # 30 days of 4h = 180 bars (> 100 ATR-pct lookback)


# ──────────────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────────────
def fetch_candles(sym_rest: str, resolution: str, hours_back: int) -> pd.DataFrame:
    end = int(time.time())
    start = end - hours_back * 3600
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": sym_rest, "resolution": resolution,
                    "start": start, "end": end},
            timeout=10,
        )
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
    """Fetch latest 1m close as proxy for current price."""
    end = int(time.time())
    start = end - 600  # 10 min lookback
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": sym_rest, "resolution": "1m",
                    "start": start, "end": end},
            timeout=10,
        )
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
    # Rolling 20-bar high/low (excluding current bar)
    out["roll_high"] = high.rolling(ROLL_LOOKBACK).max().shift(1)
    out["roll_low"]  = low.rolling(ROLL_LOOKBACK).min().shift(1)
    return out


def get_htf_atr_pct(df_4h: pd.DataFrame) -> float:
    """Compute the LATEST 4h ATR percentile rank.

    Returns a float in [0, 1] or NaN if insufficient data.
    """
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
# Signal detection (mirrors W/F simulator)
# ──────────────────────────────────────────────────────────────────────
def detect_signal(df_5m: pd.DataFrame, htf_atr_pct: float) -> Optional[dict]:
    """Look for a fresh signal on the LAST CLOSED 5m bar.

    Returns dict {side, limit_price, sl, tp, atr, signal_ts} or None.
    """
    if pd.isna(htf_atr_pct):
        return None
    if htf_atr_pct > REGIME_MAX:
        return None  # 4h vol too high — skip

    if len(df_5m) < ROLL_LOOKBACK + 5:
        return None
    df = add_indicators(df_5m)
    i = len(df) - 1
    row = df.iloc[i]
    if any(pd.isna(row[c]) for c in ("atr", "roll_high", "roll_low")):
        return None
    atr = float(row["atr"])
    if atr <= 0:
        return None

    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    body = abs(c - o)
    if body < DISP_ATR * atr:
        return None  # weak displacement

    roll_high = float(row["roll_high"])
    roll_low  = float(row["roll_low"])
    prev = df.iloc[i - 1]
    prev_low, prev_high = float(prev["low"]), float(prev["high"])

    side = None
    if l < roll_low - SWEEP_ATR * atr and c > prev_low and c > o:
        side = "long"
        sl = c - SL_ATR_MULT * atr
        tp = c + TP_ATR_MULT * atr
    elif h > roll_high + SWEEP_ATR * atr and c < prev_high and c < o:
        side = "short"
        sl = c + SL_ATR_MULT * atr
        tp = c - TP_ATR_MULT * atr
    else:
        return None

    return {
        "side": side,
        "limit_price": float(c),  # maker limit at signal-bar close
        "sl": float(sl),
        "tp": float(tp),
        "atr": atr,
        "htf_atr_pct": htf_atr_pct,
        "signal_ts": df.index[i].isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────
# State / IO
# ──────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "open_trades": [],
        "pending_limits": [],
        "last_eval_candle": {},
        "history_pnl": 0.0,
        "history_n": 0,
        "history_wins": 0,
        "history_scalper_applied": 0,
    }


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_trade(trade: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with TRADES_FILE.open("a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def close_trade(trade: dict, exit_price: float, reason: str) -> dict:
    """Compute fees + net PnL using fee_model with scalper-offer logic."""
    side = trade["side"]
    entry = trade["entry"]
    notional = trade["notional"]
    holding_sec = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(trade["opened_at"])).total_seconds()

    # Gross PnL on notional
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
    trade["fee_pct"] = round(fee_info["fee_rate"] * 100, 4)
    trade["scalper_applied"] = bool(fee_info["scalper_applied"])
    trade["net_pnl_usd"] = round(net_usd, 4)
    trade["status"] = "closed"
    return trade


# ──────────────────────────────────────────────────────────────────────
# Trade lifecycle
# ──────────────────────────────────────────────────────────────────────
def manage_open_trade(trade: dict, sym_rest: str) -> dict:
    px = get_current_price(sym_rest)
    if px is None:
        return trade

    opened = datetime.fromisoformat(trade["opened_at"])
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    age_min = (datetime.now(timezone.utc) - opened).total_seconds() / 60.0

    R = abs(trade["entry"] - trade["sl"])
    cur_r = ((px - trade["entry"]) / R) if trade["side"] == "long" else \
            ((trade["entry"] - px) / R)
    trade["peak_mfe_r"] = max(trade.get("peak_mfe_r", 0.0), cur_r)
    trade["last_price"] = px
    trade["last_check"] = datetime.now(timezone.utc).isoformat()

    # 1. Hard time stop FIRST — preserve scalper-offer eligibility
    if age_min >= TIME_STOP_MIN:
        return close_trade(trade, px, f"time_stop_{TIME_STOP_MIN}min")

    # 2. SL hit
    if trade["side"] == "long" and px <= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")
    if trade["side"] == "short" and px >= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")

    # 3. TP hit (2× ATR target)
    if trade["side"] == "long" and px >= trade["tp"]:
        return close_trade(trade, trade["tp"], "tp_hit")
    if trade["side"] == "short" and px <= trade["tp"]:
        return close_trade(trade, trade["tp"], "tp_hit")

    return trade


def manage_pending_limit(pending: dict, sym_rest: str
                         ) -> tuple[Optional[dict], Optional[dict]]:
    """Return (still_pending, new_open_trade). Exactly one is non-None."""
    px = get_current_price(sym_rest)
    if px is None:
        return pending, None

    placed = datetime.fromisoformat(pending["placed_at"])
    if placed.tzinfo is None:
        placed = placed.replace(tzinfo=timezone.utc)
    age_min = (datetime.now(timezone.utc) - placed).total_seconds() / 60.0
    if age_min >= LIMIT_EXPIRY_MIN:
        return None, None  # expired

    pending["last_price"] = px
    pending["last_check"] = datetime.now(timezone.utc).isoformat()

    fill = False
    if pending["side"] == "long" and px <= pending["limit_price"]:
        fill = True
    if pending["side"] == "short" and px >= pending["limit_price"]:
        fill = True
    if not fill:
        return pending, None

    entry = pending["limit_price"]
    trade = {
        "id": f"liqswphtf_{int(time.time())}_{pending['symbol'].replace('/', '')}",
        "symbol": pending["symbol"],
        "side": pending["side"],
        "entry": entry,
        "sl": pending["sl"],
        "tp": pending["tp"],
        "atr_at_entry": pending["atr"],
        "htf_atr_pct_at_entry": pending["htf_atr_pct"],
        "notional": NOTIONAL_USD,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "candle_time": pending["signal_ts"],
        "limit_price": pending["limit_price"],
        "peak_mfe_r": 0.0,
        "status": "open",
    }
    # PATCH_L_5_22 — bridge new trade to shadow execution queue
    try:
        from bot.shadow_bridge import publish_signal as _bridge_pub
        _bridge_pub(
            source_engine="liq_sweep_htf",
            symbol=trade["symbol"],
            side=trade["side"],
            entry_price=trade["entry"],
            stop_loss=trade["sl"],
            take_profit=trade.get("tp"),
            ml_probability=0.65,
            grade="A",
            setup_type="liq_sweep_htf",
            confidence=75.0,
            regime="bridge_engine",
            extra_meta={"engine_trade_id": trade.get("id")},
        )
    except Exception:
        pass  # fail open — paper engine continues regardless
    return None, trade


# ──────────────────────────────────────────────────────────────────────
# Main eval loop
# ──────────────────────────────────────────────────────────────────────
def eval_signals(state: dict) -> None:
    open_by_sym = {t["symbol"] for t in state.get("open_trades", [])}
    pending_by_sym = {p["symbol"] for p in state.get("pending_limits", [])}
    last_eval = state.setdefault("last_eval_candle", {})

    for sym, sym_rest in SYMBOLS.items():
        if sym in open_by_sym or sym in pending_by_sym:
            continue
        df_5m = fetch_candles(sym_rest, "5m", HOURS_BACK_5M)
        if df_5m.empty or len(df_5m) < ROLL_LOOKBACK + 5:
            print(f"  {sym}: insufficient 5m data ({len(df_5m)} bars)")
            continue
        df_4h = fetch_candles(sym_rest, "4h", HOURS_BACK_4H)
        htf_atr_pct = get_htf_atr_pct(df_4h)

        # Don't re-eval same closed candle twice
        last_closed_ts = df_5m.index[-1].isoformat()
        if last_eval.get(sym) == last_closed_ts:
            try:
                log_eval(engine_name="liq_sweep_htf", symbol=sym, base_dir=STATE_DIR,
                         signal_fired=False,
                         state={"rejection_reason": "ALREADY_EVALUATED",
                                "htf_atr_pct": htf_atr_pct})
            except Exception:
                pass
            continue
        last_eval[sym] = last_closed_ts

        sig = detect_signal(df_5m, htf_atr_pct)

        # Telemetry
        try:
            _state = {
                "htf_atr_pct": round(htf_atr_pct, 3) if not pd.isna(htf_atr_pct) else None,
                "rejection_reason": None if sig else "NO_SWEEP_OR_REGIME_BLOCKED",
            }
            if sig:
                _state.update({k: sig[k] for k in ("side", "limit_price", "sl", "tp")})
            log_eval(engine_name="liq_sweep_htf", symbol=sym, base_dir=STATE_DIR,
                     signal_fired=(sig is not None), state=_state)
        except Exception:
            pass

        if not sig:
            continue

        pending = {
            "symbol": sym,
            "side": sig["side"],
            "limit_price": sig["limit_price"],
            "sl": sig["sl"],
            "tp": sig["tp"],
            "atr": sig["atr"],
            "htf_atr_pct": sig["htf_atr_pct"],
            "signal_ts": sig["signal_ts"],
            "placed_at": datetime.now(timezone.utc).isoformat(),
        }
        state.setdefault("pending_limits", []).append(pending)
        print(f"  {sym}: SIGNAL {sig['side']} limit={sig['limit_price']:.4f} "
              f"sl={sig['sl']:.4f} tp={sig['tp']:.4f} htf_pct={htf_atr_pct:.2f}")


def write_summary(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    n = state.get("history_n", 0)
    w = state.get("history_wins", 0)
    pnl = state.get("history_pnl", 0.0)
    sa = state.get("history_scalper_applied", 0)
    open_n = len(state.get("open_trades", []))
    pending_n = len(state.get("pending_limits", []))
    wr = (100 * w / n) if n else 0
    ev = (pnl / n) if n else 0
    sa_pct = (100 * sa / n) if n else 0
    SUMMARY_FILE.write_text(
        f"# SOL Liquidity-Sweep + HTF Paper Engine\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"## Cumulative Results\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Closed trades | {n} |\n"
        f"| Wins | {w} ({wr:.1f}%) |\n"
        f"| Net PnL | ${pnl:.2f} |\n"
        f"| EV per trade | ${ev:.3f} |\n"
        f"| Scalper offer applied | {sa}/{n} ({sa_pct:.0f}%) |\n"
        f"| Open trades | {open_n} |\n"
        f"| Pending limits | {pending_n} |\n"
    )


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    state = load_state()
    print(f"=== liq_sweep_htf paper engine — {datetime.now(timezone.utc).isoformat()} ===")

    # 1. Manage existing open trades
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

    # 2. Manage pending limits
    new_pending = []
    for pending in state.get("pending_limits", []):
        sym_rest = SYMBOLS.get(pending["symbol"])
        if sym_rest is None:
            continue
        still, opened = manage_pending_limit(pending, sym_rest)
        if opened is not None:
            state["open_trades"].append(opened)
            print(f"  FILLED {opened['symbol']} {opened['side']} @${opened['entry']:.4f}")
        elif still is not None:
            new_pending.append(still)
    state["pending_limits"] = new_pending

    # 3. Look for new signals (only if no open or pending for symbol)
    eval_signals(state)

    save_state(state)
    write_summary(state)
    print(f"=== done. open={len(state['open_trades'])} pending={len(state['pending_limits'])} ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ETH Absorption Bubble paper engine — REVERSAL setup.

W/F-validated 2026-05-02 (storage/wf_studies/absorption_bubble/):
  Variant C_maker_scalper, ETH only:
    cell `lookback_n=30 sweep_atr=0.3 vol_mult=1.5 displ_atr=0.65 body_ratio_max=0.40 wick_ratio_min=0.50 tp=1.5R`
    IS Q1+Q2:  n=72, EV/trade=$+0.174
    Q3 OOS:    EV/trade=$+0.550
    Q4 OOS:    EV/trade=$+0.640
    WR_oos:    ~58% (LONG +$0.346, SHORT +$0.391)
    trades/day: ~0.59 ETH only
  Verdict: PASS walk-forward (21 PASS cells, all ETH/Variant C).

Pattern (LONG; mirror for SHORT):
  Step 1 — LIQUIDITY SWEEP: bar i low penetrates 30-bar low by >= 0.3 × ATR
  Step 2 — ABSORPTION CANDLE: bar i shows
             - body / range <= 0.40  (small body)
             - lower_wick / range >= 0.50  (long lower wick = rejection)
             - rel_vol = volume / 20-bar mean >= 1.5  (heavy volume)
  Step 3 — RECLAIM: bar i+1 closes back above sweep low with body >= 0.65 × ATR
  Step 4 — ENTRY: at confirmation candle close (maker limit)
  STOP : absorption low − 0.3 × ATR
  TP   : 1.5R from entry
  TIME : 30 min hard stop

Symbols: ETH/USDT only (W/F-validated). Maker entry mandatory (taker = 0 PASS).

State dir: storage/absorption_bubble_paper/
Bridge: publish_signal() at trade open (Patch L bridge integration).
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

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_v2.fee_model import FeeModel  # noqa: E402

STATE_DIR = ROOT / "storage" / "absorption_bubble_paper"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
SUMMARY_FILE = STATE_DIR / "latest.md"
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"

SYMBOLS = {
    "ETH/USDT": "ETHUSD",
}

# Locked W/F params
TF = "5m"
TF_MIN = 5
LOOKBACK_N = 30
SWEEP_ATR = 0.3
VOL_MULT = 1.5
DISPL_ATR = 0.65
BODY_RATIO_MAX = 0.40
WICK_RATIO_MIN = 0.50
TP_RR = 1.0          # ABSORPTION_FAST_EXIT_5_22: was 1.5; W/F lifted EV by $+0.146/trade
SL_BUFFER_ATR = 0.3
TIME_STOP_MIN = 30
NOTIONAL_USD = 1000.0
LIMIT_EXPIRY_MIN = TF_MIN
ATR_PERIOD = 14
VOL_PERIOD = 20
HOURS_BACK_5M = 24 * 2
# ABSORPTION_FAST_EXIT_5_22 (2026-05-03) — exit W/F #1 winner
# (TP1=1.0R, vol-fade exit, max 6 bars / 30min unchanged)
VOL_FADE_RATIO = 0.7    # exit if latest-bar vol < this × 20-bar mean


# ──────────────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────────────
def fetch_candles(sym_rest: str, resolution: str, hours_back: int) -> pd.DataFrame:
    end = int(time.time())
    start = end - hours_back * 3600
    try:
        r = requests.get(
            DELTA_REST,
            params={
                "symbol": sym_rest,
                "resolution": resolution,
                "start": start,
                "end": end,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json().get("result", [])
    except Exception as e:
        print(f"  fetch_candles({sym_rest},{resolution}) failed: {e}", file=sys.stderr)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").reset_index(drop=True)
    return df.dropna()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ──────────────────────────────────────────────────────────────────────
# Pattern detection
# ──────────────────────────────────────────────────────────────────────
def detect_signal(df: pd.DataFrame) -> Optional[dict]:
    """Detect absorption bubble + reclaim. Returns pending dict or None."""
    if len(df) < LOOKBACK_N + 5:
        return None

    atr_series = add_atr(df, ATR_PERIOD)
    vol_mean = df["volume"].rolling(VOL_PERIOD).mean()

    # Inspect last 2 bars: i-1 = absorption candidate; i = reclaim/confirmation
    i = len(df) - 1
    j = i - 1  # absorption candle index
    if j < LOOKBACK_N:
        return None

    bar_j = df.iloc[j]
    bar_i = df.iloc[i]
    atr_j = float(atr_series.iloc[j])
    if atr_j <= 0 or math.isnan(atr_j):
        return None

    # Step 1: did bar j sweep below 30-bar low by >= sweep_atr × ATR?
    roll_low_30 = float(df["low"].iloc[j - LOOKBACK_N : j].min())
    roll_high_30 = float(df["high"].iloc[j - LOOKBACK_N : j].max())

    sweep_dist_low = roll_low_30 - float(bar_j["low"])
    sweep_dist_high = float(bar_j["high"]) - roll_high_30

    long_sweep = sweep_dist_low >= SWEEP_ATR * atr_j
    short_sweep = sweep_dist_high >= SWEEP_ATR * atr_j
    if not (long_sweep or short_sweep):
        return None

    # Step 2: absorption properties (small body + long opposing wick + heavy vol)
    bar_j_open = float(bar_j["open"])
    bar_j_close = float(bar_j["close"])
    bar_j_high = float(bar_j["high"])
    bar_j_low = float(bar_j["low"])
    bar_j_range = bar_j_high - bar_j_low
    if bar_j_range <= 0:
        return None
    body = abs(bar_j_close - bar_j_open)
    body_ratio = body / bar_j_range

    if body_ratio > BODY_RATIO_MAX:
        return None

    if long_sweep:
        lower_wick = min(bar_j_open, bar_j_close) - bar_j_low
        wick_ratio = lower_wick / bar_j_range
        side = "long"
        ref_extreme = bar_j_low  # for stop
        sweep_level = roll_low_30
    else:
        upper_wick = bar_j_high - max(bar_j_open, bar_j_close)
        wick_ratio = upper_wick / bar_j_range
        side = "short"
        ref_extreme = bar_j_high
        sweep_level = roll_high_30

    if wick_ratio < WICK_RATIO_MIN:
        return None

    # Volume check
    vol_j = float(bar_j["volume"])
    vol_mean_j = float(vol_mean.iloc[j]) if not math.isnan(vol_mean.iloc[j]) else 0.0
    if vol_mean_j <= 0:
        return None
    rel_vol = vol_j / vol_mean_j
    if rel_vol < VOL_MULT:
        return None

    # Step 3: bar i (current) reclaims with displacement
    bar_i_open = float(bar_i["open"])
    bar_i_close = float(bar_i["close"])
    bar_i_body = abs(bar_i_close - bar_i_open)
    if bar_i_body < DISPL_ATR * atr_j:
        return None

    if side == "long":
        # Long: bar i must close above sweep_level (back inside the range)
        if bar_i_close <= sweep_level:
            return None
        if bar_i_close <= bar_i_open:
            return None  # need bullish confirmation
        sl = ref_extreme - SL_BUFFER_ATR * atr_j
        risk = bar_i_close - sl
        if risk <= 0:
            return None
        tp = bar_i_close + risk * TP_RR
    else:
        # Short: bar i must close below sweep_level
        if bar_i_close >= sweep_level:
            return None
        if bar_i_close >= bar_i_open:
            return None  # need bearish confirmation
        sl = ref_extreme + SL_BUFFER_ATR * atr_j
        risk = sl - bar_i_close
        if risk <= 0:
            return None
        tp = bar_i_close - risk * TP_RR

    return {
        "symbol": None,  # set by caller
        "side": side,
        "limit_price": bar_i_close,
        "sl": sl,
        "tp": tp,
        "atr": atr_j,
        "absorption_low": bar_j_low,
        "absorption_high": bar_j_high,
        "rel_vol": rel_vol,
        "wick_ratio": wick_ratio,
        "body_ratio": body_ratio,
        "sweep_level": sweep_level,
        "signal_ts": bar_i["time"].isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────
# Trade lifecycle
# ──────────────────────────────────────────────────────────────────────
def maybe_fill_pending(pending: dict, sym_rest: str) -> tuple[Optional[dict], Optional[dict]]:
    """Return (pending, trade) — promote pending → trade if fill condition met."""
    px = get_current_price(sym_rest)
    if px is None:
        return pending, None

    fill = False
    if pending["side"] == "long" and px <= pending["limit_price"]:
        fill = True
    if pending["side"] == "short" and px >= pending["limit_price"]:
        fill = True
    if not fill:
        return pending, None

    entry = pending["limit_price"]
    trade = {
        "id": f"absorb_{int(time.time())}_{pending['symbol'].replace('/', '')}",
        "symbol": pending["symbol"],
        "side": pending["side"],
        "entry": entry,
        "sl": pending["sl"],
        "tp": pending["tp"],
        "atr_at_entry": pending["atr"],
        "absorption_low_at_entry": pending["absorption_low"],
        "absorption_high_at_entry": pending["absorption_high"],
        "rel_vol_at_entry": pending["rel_vol"],
        "wick_ratio_at_entry": pending["wick_ratio"],
        "notional": NOTIONAL_USD,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "candle_time": pending["signal_ts"],
        "limit_price": pending["limit_price"],
        "peak_mfe_r": 0.0,
        "status": "open",
    }
    # PATCH_L bridge — publish to shadow execution queue
    try:
        from bot.shadow_bridge import publish_signal as _bridge_pub
        _bridge_pub(
            source_engine="absorption_bubble",
            symbol=trade["symbol"],
            side=trade["side"],
            entry_price=trade["entry"],
            stop_loss=trade["sl"],
            take_profit=trade.get("tp"),
            ml_probability=0.65,   # W/F-validated proxy
            grade="A",
            setup_type="absorption_bubble",
            confidence=80.0,
            regime="reversal",
            extra_meta={
                "engine_trade_id": trade.get("id"),
                "absorption_low": trade.get("absorption_low_at_entry"),
                "absorption_high": trade.get("absorption_high_at_entry"),
                "rel_vol": trade.get("rel_vol_at_entry"),
            },
        )
    except Exception:
        pass  # fail open — paper engine continues regardless
    return None, trade


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

    # ABSORPTION_FAST_EXIT_5_22 — vol-fade exit (flow stopped confirming).
    # Check latest 5m bar's volume vs 20-bar mean. If under VOL_FADE_RATIO,
    # close at current price. We only check this AFTER 1 bar elapsed so we
    # don't insta-exit on entry.
    if age_min >= 5.0:
        try:
            df_recent = fetch_candles(sym_rest, "5m", 4)
            if not df_recent.empty and len(df_recent) >= VOL_PERIOD + 1:
                vol_mean = df_recent["volume"].astype(float).rolling(
                    VOL_PERIOD, min_periods=VOL_PERIOD).mean().iloc[-2]
                vol_now = float(df_recent["volume"].astype(float).iloc[-1])
                if vol_mean and vol_mean > 0 and vol_now < VOL_FADE_RATIO * vol_mean:
                    return close_trade(trade, px, "vol_fade")
        except Exception:
            pass  # fail-open — don't abort live trade on monitoring glitch

    # Hit TP
    if trade["side"] == "long" and px >= trade["tp"]:
        return close_trade(trade, trade["tp"], "tp_hit")
    if trade["side"] == "short" and px <= trade["tp"]:
        return close_trade(trade, trade["tp"], "tp_hit")

    # Hit SL
    if trade["side"] == "long" and px <= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")
    if trade["side"] == "short" and px >= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")

    # Time stop
    if age_min >= TIME_STOP_MIN:
        return close_trade(trade, px, f"time_stop_{TIME_STOP_MIN}min")

    return trade


def close_trade(trade: dict, exit_price: float, reason: str) -> dict:
    entry = trade["entry"]
    notional = trade["notional"]
    side = trade["side"]
    if side == "long":
        gross_pct = (exit_price - entry) / entry
    else:
        gross_pct = (entry - exit_price) / entry
    gross_usd = gross_pct * notional

    opened = datetime.fromisoformat(trade["opened_at"])
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    holding_sec = (datetime.now(timezone.utc) - opened).total_seconds()

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


def get_current_price(sym_rest: str) -> Optional[float]:
    df = fetch_candles(sym_rest, "1m", 1)
    if df.empty:
        return None
    return float(df.iloc[-1]["close"])


# ──────────────────────────────────────────────────────────────────────
# State + main eval
# ──────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_trade(trade: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with TRADES_FILE.open("a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def eval_signals(state: dict) -> None:
    open_by_sym = {t["symbol"] for t in state.get("open_trades", [])}
    pending_by_sym = {p["symbol"] for p in state.get("pending_limits", [])}
    last_eval = state.setdefault("last_eval_candle", {})

    for sym, sym_rest in SYMBOLS.items():
        if sym in open_by_sym or sym in pending_by_sym:
            continue
        df_5m = fetch_candles(sym_rest, "5m", HOURS_BACK_5M)
        if df_5m.empty or len(df_5m) < LOOKBACK_N + 5:
            print(f"  {sym}: insufficient 5m data")
            continue
        # Don't re-eval same closed candle twice
        last_ts = last_eval.get(sym)
        cur_ts = df_5m.iloc[-1]["time"].isoformat()
        if last_ts == cur_ts:
            continue
        last_eval[sym] = cur_ts

        sig = detect_signal(df_5m)
        if sig is None:
            continue
        sig["symbol"] = sym
        state.setdefault("pending_limits", []).append(sig)
        print(f"  {sym}: PENDING {sig['side']} @ {sig['limit_price']:.4f}  rel_vol={sig['rel_vol']:.1f}x")


def manage_pending_and_open(state: dict) -> None:
    new_pending = []
    new_open = list(state.get("open_trades", []))
    for pending in state.get("pending_limits", []):
        sym = pending["symbol"]
        sym_rest = SYMBOLS.get(sym)
        if not sym_rest:
            continue
        # Expire stale pending limits
        # Ages are tracked by signal_ts; safer to use re-fetch & gap
        rem_pending, trade = maybe_fill_pending(pending, sym_rest)
        if trade is not None:
            new_open.append(trade)
            print(f"  FILLED {sym} {trade['side']} @${trade['entry']:.4f}")
        elif rem_pending is not None:
            new_pending.append(rem_pending)
    state["pending_limits"] = new_pending

    still_open = []
    for trade in new_open:
        sym_rest = SYMBOLS.get(trade["symbol"])
        if not sym_rest:
            still_open.append(trade)
            continue
        managed = manage_open_trade(trade, sym_rest)
        if managed.get("status") == "closed":
            append_trade(managed)
            print(f"  CLOSED {managed['symbol']} {managed['side']} @${managed['exit_price']:.4f} {managed.get('close_reason')} net=${managed.get('net_pnl_usd',0):.4f}")
        else:
            still_open.append(managed)
    state["open_trades"] = still_open


def main() -> None:
    print(f"=== absorption_bubble paper engine — {datetime.now(timezone.utc).isoformat()} ===")
    state = load_state()
    state.setdefault("open_trades", [])
    state.setdefault("pending_limits", [])
    eval_signals(state)
    manage_pending_and_open(state)
    save_state(state)
    print(f"=== done. open={len(state['open_trades'])} pending={len(state['pending_limits'])} ===")


if __name__ == "__main__":
    main()

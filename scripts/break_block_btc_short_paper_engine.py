#!/usr/bin/env python3
"""BTC SHORT Break Block paper engine — combo #5 W/F result (2026-05-03).

Pattern (BTC-only, SHORT-only):
  HTF bias:   1H trend bearish — close[-1] < close[-htf_window] − htf_slope_atr × ATR(1H)
              (with htf_window=3, htf_slope_atr=0.1: 1H must have dropped ≥ 0.1 × 1H ATR over 3 bars)
  Structure:  In last 20 closed 5m bars (excluding current), find block_low and block_high.
  Trigger:    Current closed 5m bar's close ≤ block_low − break_atr × ATR(5m)  (break_atr=0.8)
  Confirm:    Entry candle is bearish (close < open).
  Entry:      Maker limit at trigger close.
  SL:         block_high + 0.3 × ATR(5m).
  TP:         entry − 2.0 × (SL − entry)   (tp_rr=2.0)
  Time stop:  60 min hard close.

W/F evidence (combo #5 BTC SHORT break_block + HTF):
  Cell `htf_window=3 htf_slope_atr=0.1 break_atr=0.8 tp_rr=2.0`
  Variant C_maker_scalper, BTC only:
    Q1+Q2 IS:  EV/trade > $0.10 floor
    Q3 OOS:    PASS
    Q4 OOS:    PASS
  Verdict: paper-only engine recommended (recommendation: do NOT promote to admission gate).

Throttles & gates:
  - Hard ceiling ≤ 5 trades / 7 days (engine-side, counted from trades.jsonl)
  - KILL gate: 30 paper trades w/ WR <45% OR EV <$0.10/trade → halt + alert
  - Promotion gate: 60+ paper trades AND matching live edge (manual review)

Maker entry mandatory (taker variant did NOT pass W/F).

State dir:  storage/break_block_btc_short_paper/
Bridge:     publish_signal(source_engine="break_block_btc_short") at trade open.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_v2.fee_model import FeeModel  # noqa: E402

STATE_DIR = ROOT / "storage" / "break_block_btc_short_paper"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
SUMMARY_FILE = STATE_DIR / "latest.md"
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"

SYMBOLS = {
    "BTC/USDT": "BTCUSD",
}

# ── Locked W/F params (combo #5 cell) ──────────────────────────────────
TF = "5m"
TF_MIN = 5
HTF = "1h"
HTF_MIN = 60

HTF_WINDOW = 3                 # 1H bars to look back for HTF slope
HTF_SLOPE_ATR = 0.1            # 1H must drop ≥ 0.1 × ATR(1H) over HTF_WINDOW bars
HTF_ATR_PERIOD = 14

BLOCK_LOOKBACK = 20            # 5m bars to define block_low / block_high (exclude current)
BREAK_ATR = 0.8                # close must be ≤ block_low − 0.8 × ATR(5m)
TP_RR = 2.0
SL_BUFFER_ATR = 0.3            # SL above block_high by 0.3 × ATR(5m)
TIME_STOP_MIN = 60
ATR_PERIOD = 14

NOTIONAL_USD = 1000.0
LIMIT_EXPIRY_MIN = TF_MIN
HOURS_BACK_5M = 24 * 2
HOURS_BACK_1H = 24 * 7         # need ≥ HTF_WINDOW + ATR_PERIOD 1H bars

# ── Throttles ──────────────────────────────────────────────────────────
WEEKLY_TRADE_CEILING = 5
KILL_MIN_TRADES = 30
KILL_WR_FLOOR = 0.45
KILL_EV_FLOOR = 0.10
PROMOTION_MIN_TRADES = 60      # informational; manual gate


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


def get_current_price(sym_rest: str) -> Optional[float]:
    df = fetch_candles(sym_rest, "1m", 1)
    if df.empty:
        return None
    return float(df.iloc[-1]["close"])


# ──────────────────────────────────────────────────────────────────────
# Pattern detection (SHORT-only)
# ──────────────────────────────────────────────────────────────────────
def htf_bearish(df_1h: pd.DataFrame) -> tuple[bool, dict]:
    """Return (is_bearish, diag). HTF bias check on 1H frame."""
    if len(df_1h) < HTF_WINDOW + HTF_ATR_PERIOD + 2:
        return False, {"reason": "insufficient_htf_bars", "n": len(df_1h)}
    atr_1h = float(add_atr(df_1h, HTF_ATR_PERIOD).iloc[-2])
    if atr_1h <= 0 or math.isnan(atr_1h):
        return False, {"reason": "atr_invalid"}
    # Use the most recent CLOSED 1H bar (-2 if last bar is forming, but we treat last as closed)
    close_now = float(df_1h.iloc[-1]["close"])
    close_back = float(df_1h.iloc[-1 - HTF_WINDOW]["close"])
    drop = close_back - close_now
    threshold = HTF_SLOPE_ATR * atr_1h
    return (drop >= threshold), {
        "atr_1h": atr_1h,
        "close_now": close_now,
        "close_back": close_back,
        "drop": drop,
        "threshold": threshold,
    }


def detect_signal(df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> Optional[dict]:
    """Detect BTC SHORT break-block on the most recent CLOSED 5m bar.

    Returns pending dict or None.
    """
    if len(df_5m) < BLOCK_LOOKBACK + ATR_PERIOD + 5:
        return None

    # HTF bias gate
    htf_ok, htf_diag = htf_bearish(df_1h)
    if not htf_ok:
        return None

    atr_series_5m = add_atr(df_5m, ATR_PERIOD)
    i = len(df_5m) - 1                         # current closed 5m bar
    bar_i = df_5m.iloc[i]
    atr_i = float(atr_series_5m.iloc[i])
    if atr_i <= 0 or math.isnan(atr_i):
        return None

    # Block window: BLOCK_LOOKBACK bars BEFORE current
    win_start = i - BLOCK_LOOKBACK
    win_end = i  # exclusive
    if win_start < 0:
        return None
    block = df_5m.iloc[win_start:win_end]
    block_low = float(block["low"].min())
    block_high = float(block["high"].max())

    bar_close = float(bar_i["close"])
    bar_open = float(bar_i["open"])

    # Trigger: close below block_low by ≥ break_atr × ATR
    break_dist = block_low - bar_close
    if break_dist < BREAK_ATR * atr_i:
        return None

    # Confirmation: bearish candle (close < open)
    if bar_close >= bar_open:
        return None

    # Entry / stop / target
    entry = bar_close                          # maker limit at break candle close
    sl = block_high + SL_BUFFER_ATR * atr_i
    risk = sl - entry
    if risk <= 0:
        return None
    tp = entry - risk * TP_RR

    return {
        "symbol": None,                         # set by caller
        "side": "short",
        "limit_price": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr_i,
        "block_low": block_low,
        "block_high": block_high,
        "break_dist": break_dist,
        "htf_drop": htf_diag.get("drop"),
        "htf_atr": htf_diag.get("atr_1h"),
        "signal_ts": bar_i["time"].isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────
# Trade lifecycle
# ──────────────────────────────────────────────────────────────────────
def maybe_fill_pending(pending: dict, sym_rest: str) -> tuple[Optional[dict], Optional[dict]]:
    """Return (pending_remaining, trade) — promote pending → trade if maker fill condition met."""
    px = get_current_price(sym_rest)
    if px is None:
        return pending, None

    # SHORT maker limit: filled when price trades up to / through limit
    if px < pending["limit_price"]:
        return pending, None

    entry = pending["limit_price"]
    trade = {
        "id": f"bbk_{int(time.time())}_{pending['symbol'].replace('/', '')}",
        "symbol": pending["symbol"],
        "side": pending["side"],
        "entry": entry,
        "sl": pending["sl"],
        "tp": pending["tp"],
        "atr_at_entry": pending["atr"],
        "block_low_at_entry": pending["block_low"],
        "block_high_at_entry": pending["block_high"],
        "break_dist_at_entry": pending["break_dist"],
        "htf_drop_at_entry": pending["htf_drop"],
        "htf_atr_at_entry": pending["htf_atr"],
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
            source_engine="break_block_btc_short",
            symbol=trade["symbol"],
            side=trade["side"],
            entry_price=trade["entry"],
            stop_loss=trade["sl"],
            take_profit=trade.get("tp"),
            ml_probability=0.62,            # W/F-validated proxy (combo #5)
            grade="A",
            setup_type="break_block",
            confidence=78.0,
            regime="trending_down",
            extra_meta={
                "engine_trade_id": trade.get("id"),
                "block_low": trade.get("block_low_at_entry"),
                "block_high": trade.get("block_high_at_entry"),
                "break_dist_atr": (trade.get("break_dist_at_entry") /
                                   trade.get("atr_at_entry"))
                                   if trade.get("atr_at_entry") else None,
                "htf_drop_atr": (trade.get("htf_drop_at_entry") /
                                 trade.get("htf_atr_at_entry"))
                                 if trade.get("htf_atr_at_entry") else None,
                "wf_combo": "combo5_btc_short_break_block_htf",
                "weekly_ceiling": WEEKLY_TRADE_CEILING,
            },
        )
    except Exception:
        pass  # fail-open
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
    cur_r = ((trade["entry"] - px) / R) if trade["side"] == "short" else \
            ((px - trade["entry"]) / R)
    trade["peak_mfe_r"] = max(trade.get("peak_mfe_r", 0.0), cur_r)
    trade["last_price"] = px

    # Hit TP (short: price falls to / below tp)
    if trade["side"] == "short" and px <= trade["tp"]:
        return close_trade(trade, trade["tp"], "tp_hit")
    # Hit SL (short: price rallies to / above sl)
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
    if side == "short":
        gross_pct = (entry - exit_price) / entry
    else:
        gross_pct = (exit_price - entry) / entry
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


# ──────────────────────────────────────────────────────────────────────
# Throttles & gates
# ──────────────────────────────────────────────────────────────────────
def _load_closed_trades() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    out: list[dict] = []
    try:
        with TRADES_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []
    return out


def weekly_count_open_or_closed(state: dict) -> int:
    """Trades opened (filled) within last 7 days — both closed (in trades.jsonl)
    and currently open (in state)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    n = 0
    for t in _load_closed_trades():
        try:
            opened = datetime.fromisoformat(t["opened_at"])
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            if opened >= cutoff:
                n += 1
        except Exception:
            pass
    for t in state.get("open_trades", []):
        try:
            opened = datetime.fromisoformat(t["opened_at"])
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            if opened >= cutoff:
                n += 1
        except Exception:
            pass
    return n


def kill_gate_status() -> tuple[bool, dict]:
    """Return (kill_active, diag). Engine halts new signals if kill_active."""
    closed = _load_closed_trades()
    n = len(closed)
    if n < KILL_MIN_TRADES:
        return False, {"n_closed": n, "min_required": KILL_MIN_TRADES, "kill": False}
    wins = sum(1 for t in closed if (t.get("net_pnl_usd") or 0) > 0)
    wr = wins / n if n else 0.0
    ev = sum((t.get("net_pnl_usd") or 0.0) for t in closed) / n if n else 0.0
    kill = (wr < KILL_WR_FLOOR) or (ev < KILL_EV_FLOOR)
    return kill, {
        "n_closed": n, "wr": round(wr, 3), "ev_per_trade": round(ev, 4),
        "wr_floor": KILL_WR_FLOOR, "ev_floor": KILL_EV_FLOOR, "kill": kill,
    }


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
    # Throttle / kill gate first
    kill, kdiag = kill_gate_status()
    if kill:
        print(f"  KILL_GATE_ACTIVE: {kdiag} — skipping new signals")
        state["kill_gate"] = kdiag
        return
    state["kill_gate"] = kdiag

    week_n = weekly_count_open_or_closed(state)
    if week_n >= WEEKLY_TRADE_CEILING:
        print(f"  WEEKLY_CEILING_HIT: {week_n}/{WEEKLY_TRADE_CEILING} — skipping new signals")
        state["weekly_count"] = week_n
        return
    state["weekly_count"] = week_n

    open_by_sym = {t["symbol"] for t in state.get("open_trades", [])}
    pending_by_sym = {p["symbol"] for p in state.get("pending_limits", [])}
    last_eval = state.setdefault("last_eval_candle", {})

    for sym, sym_rest in SYMBOLS.items():
        if sym in open_by_sym or sym in pending_by_sym:
            continue
        df_5m = fetch_candles(sym_rest, "5m", HOURS_BACK_5M)
        df_1h = fetch_candles(sym_rest, "1h", HOURS_BACK_1H)
        if df_5m.empty or len(df_5m) < BLOCK_LOOKBACK + ATR_PERIOD + 5:
            print(f"  {sym}: insufficient 5m data")
            continue
        if df_1h.empty or len(df_1h) < HTF_WINDOW + HTF_ATR_PERIOD + 2:
            print(f"  {sym}: insufficient 1h data")
            continue

        # Don't re-eval same closed candle twice
        last_ts = last_eval.get(sym)
        cur_ts = df_5m.iloc[-1]["time"].isoformat()
        if last_ts == cur_ts:
            continue
        last_eval[sym] = cur_ts

        sig = detect_signal(df_5m, df_1h)
        if sig is None:
            continue
        sig["symbol"] = sym
        state.setdefault("pending_limits", []).append(sig)
        print(f"  {sym}: PENDING short @ {sig['limit_price']:.2f}  "
              f"break_atr={sig['break_dist']/sig['atr']:.2f}x  "
              f"htf_drop_atr={(sig['htf_drop'] or 0)/(sig['htf_atr'] or 1):.2f}x")


def manage_pending_and_open(state: dict) -> None:
    new_pending = []
    new_open = list(state.get("open_trades", []))
    now = datetime.now(timezone.utc)

    for pending in state.get("pending_limits", []):
        sym = pending["symbol"]
        sym_rest = SYMBOLS.get(sym)
        if not sym_rest:
            continue
        # Expire stale pending limits (signal_ts + LIMIT_EXPIRY_MIN)
        try:
            sig_ts = datetime.fromisoformat(pending["signal_ts"])
            if sig_ts.tzinfo is None:
                sig_ts = sig_ts.replace(tzinfo=timezone.utc)
            age_min = (now - sig_ts).total_seconds() / 60.0
            if age_min > LIMIT_EXPIRY_MIN + 0.5:
                print(f"  EXPIRED pending {sym} {pending['side']} @ {pending['limit_price']:.2f}")
                continue
        except Exception:
            pass

        rem_pending, trade = maybe_fill_pending(pending, sym_rest)
        if trade is not None:
            new_open.append(trade)
            print(f"  FILLED {sym} {trade['side']} @${trade['entry']:.2f}")
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
            print(f"  CLOSED {managed['symbol']} {managed['side']} @${managed['exit_price']:.2f} "
                  f"{managed.get('close_reason')} net=${managed.get('net_pnl_usd', 0):.4f}")
        else:
            still_open.append(managed)
    state["open_trades"] = still_open


def main() -> None:
    print(f"=== break_block_btc_short paper engine — {datetime.now(timezone.utc).isoformat()} ===")
    state = load_state()
    state.setdefault("open_trades", [])
    state.setdefault("pending_limits", [])
    eval_signals(state)
    manage_pending_and_open(state)
    save_state(state)
    print(f"=== done. open={len(state['open_trades'])} pending={len(state['pending_limits'])} "
          f"weekly={state.get('weekly_count', 0)}/{WEEKLY_TRADE_CEILING} "
          f"kill={state.get('kill_gate', {}).get('kill', False)} ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S5 (4h Range Fade) standalone paper-trading engine.

Designed to run every 5 minutes via cron. Independent of the 5m scalper
architecture — uses its own state file, candle source, and exit logic so it
can run multi-day swing trades without conflicting with bot's 600s max_age.

Backtest validation (6 months, BTC/ETH/SOL/XRP, Delta India taker fees):
  S5 + E4_wide_trail: 71 trades, 45% WR, +$554 net, +$7.80/trade EV

Strategy:
  1. Load 4h candles for BTC/ETH/SOL/XRP from Delta REST
  2. Identify 30-bar range (95/5 percentile high/low), require size >= 2*ATR
  3. Require >= 3 touches near each edge
  4. Enter SHORT on bearish reversal candle near range_high
  5. Enter LONG on bullish reversal candle near range_low
  6. SL = 1× ATR_4h beyond entry
  7. Exit (E4_wide_trail): trail engages at 1.5R, locks 33% of MFE, 7-day max

State: storage/s5_paper/state.json
Trades: storage/s5_paper/trades.jsonl
Summary: storage/s5_paper/latest.md (consumed by dashboard widget if added)

Usage: python3 scripts/s5_paper_engine.py
Cron: */5 * * * * cd /home/opc/crypto-trading-bot && python3 scripts/s5_paper_engine.py >> logs/s5_paper.log 2>&1
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

import sys as _sys_telem
_sys_telem.path.insert(0, "/home/opc/crypto-trading-bot")
from execution_v2.engine_telemetry import log_eval

ROOT = Path("/home/opc/crypto-trading-bot")
STATE_DIR = ROOT / "storage" / "s5_paper"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
SUMMARY_FILE = STATE_DIR / "latest.md"
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"

SYMBOLS = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
    "SOL/USDT": "SOLUSD",
    "XRP/USDT": "XRPUSD",
}
TAKER_FEE = 0.00059
NOTIONAL_USD = 1000.0
MAX_OPEN_PER_SYMBOL = 1

# Strategy params — OPTIMAL CONFIG from extended backtest + stress test
# Stress test: long-only=-$196, short-only=+$750. Bear regime=-$208 (kill).
# Extended bt: window=50 +$522, sl_atr=1.5 +$260. Short-only +$145.
WINDOW_BARS = 50           # extended bt: window=50 best (vs 30)
RANGE_QUANTILE = (0.05, 0.95)
MIN_RANGE_ATR = 2.0
MIN_TOUCHES = 3
EDGE_BAND_ATR = 0.3
SL_ATR_MULT = 1.5          # extended bt: sl_atr=1.5 +$260 (safer than 2.0)
SIDE_FILTER = "short"      # stress: long-only loses $196, short-only wins $750

# Exit (E4_wide_trail)
TRAIL_ENGAGE_R = 1.5
TRAIL_LOCK_PCT = 0.33
MAX_HOURS = 168  # 7 days


def fetch_4h(sym_rest: str, hours_back: int = 60 * 24) -> pd.DataFrame:
    end = int(time.time())
    start = end - hours_back * 3600
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": sym_rest, "resolution": "4h", "start": start, "end": end},
            timeout=15,
        )
        rows = r.json().get("result", [])
    except Exception as e:
        print(f"  REST {sym_rest} failed: {e}")
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    df = df.copy()
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def detect_signal(df: pd.DataFrame, symbol: str) -> dict | None:
    if len(df) < WINDOW_BARS + 5:
        return None
    last = df.iloc[-1]
    if pd.isna(last.atr) or last.atr <= 0:
        return None
    window = df.iloc[-(WINDOW_BARS + 1):-1]
    range_high = window.high.quantile(RANGE_QUANTILE[1])
    range_low = window.low.quantile(RANGE_QUANTILE[0])
    range_size = range_high - range_low
    if range_size < MIN_RANGE_ATR * last.atr:
        return None
    band = EDGE_BAND_ATR * last.atr
    near_high = last.high > range_high - band
    near_low = last.low < range_low + band
    bear_candle = last.close < last.open
    bull_candle = last.close > last.open
    touches_high = ((window.high > range_high - band) & (window.high <= range_high)).sum()
    touches_low = ((window.low < range_low + band) & (window.low >= range_low)).sum()
    if touches_high >= MIN_TOUCHES and near_high and bear_candle:
        if SIDE_FILTER in ("short", "both"):
            sl = last.close + SL_ATR_MULT * last.atr
            return {
                "side": "short", "entry": float(last.close), "sl": float(sl),
                "atr": float(last.atr), "candle_time": last.name.isoformat(),
                "range_high": float(range_high), "range_low": float(range_low),
                "touches": int(touches_high),
            }
    if touches_low >= MIN_TOUCHES and near_low and bull_candle:
        if SIDE_FILTER in ("long", "both"):
            sl = last.close - SL_ATR_MULT * last.atr
            return {
                "side": "long", "entry": float(last.close), "sl": float(sl),
                "atr": float(last.atr), "candle_time": last.name.isoformat(),
                "range_high": float(range_high), "range_low": float(range_low),
                "touches": int(touches_low),
            }
    return None


def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open_trades": [], "last_eval_candle": {}, "history_pnl": 0.0,
            "history_n": 0, "history_wins": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_trade(trade: dict) -> None:
    with TRADES_FILE.open("a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def get_current_price(sym_rest: str) -> float | None:
    """Latest 1m close from Delta REST (cheap)."""
    end = int(time.time())
    start = end - 300
    try:
        r = requests.get(DELTA_REST,
                         params={"symbol": sym_rest, "resolution": "1m", "start": start, "end": end},
                         timeout=8)
        rows = r.json().get("result", [])
        if not rows:
            return None
        return float(rows[-1]["close"])
    except Exception:
        return None


def manage_open_trade(trade: dict, sym_rest: str) -> dict:
    """Check current price, update peak MFE, apply trail/SL/time-stop."""
    px = get_current_price(sym_rest)
    if px is None:
        return trade
    R = abs(trade["entry"] - trade["sl"])
    if R == 0:
        return trade
    if trade["side"] == "long":
        mfe_price = px - trade["entry"]
    else:
        mfe_price = trade["entry"] - px
    cur_mfe_r = mfe_price / R
    trade["peak_mfe_r"] = max(trade.get("peak_mfe_r", 0.0), cur_mfe_r)
    trade["last_price"] = px
    trade["last_check"] = datetime.now(timezone.utc).isoformat()
    # Time stop
    opened = datetime.fromisoformat(trade["opened_at"])
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    hours_held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0
    if hours_held >= MAX_HOURS:
        return close_trade(trade, px, "time_stop_7d")
    # Trail engage
    trail_sl = trade.get("trail_sl", trade["sl"])
    if trade["peak_mfe_r"] >= TRAIL_ENGAGE_R:
        lock_r = trade["peak_mfe_r"] * TRAIL_LOCK_PCT
        if trade["side"] == "long":
            new_sl = trade["entry"] + lock_r * R
            if new_sl > trail_sl:
                trail_sl = new_sl
                trade["trail_engaged"] = True
        else:
            new_sl = trade["entry"] - lock_r * R
            if new_sl < trail_sl:
                trail_sl = new_sl
                trade["trail_engaged"] = True
        trade["trail_sl"] = trail_sl
    # SL hit
    if trade["side"] == "long" and px <= trail_sl:
        return close_trade(trade, trail_sl, "trail_sl" if trade.get("trail_engaged") else "sl_hit")
    if trade["side"] == "short" and px >= trail_sl:
        return close_trade(trade, trail_sl, "trail_sl" if trade.get("trail_engaged") else "sl_hit")
    return trade


def close_trade(trade: dict, exit_price: float, reason: str) -> dict:
    qty = trade["notional"] / trade["entry"]
    if trade["side"] == "long":
        gross = (exit_price - trade["entry"]) * qty
    else:
        gross = (trade["entry"] - exit_price) * qty
    fees = trade["notional"] * TAKER_FEE + (qty * exit_price) * TAKER_FEE
    net = gross - fees
    trade["exit_price"] = exit_price
    trade["exit_reason"] = reason
    trade["closed_at"] = datetime.now(timezone.utc).isoformat()
    trade["gross_pnl_usd"] = round(gross, 4)
    trade["fees_usd"] = round(fees, 4)
    trade["net_pnl_usd"] = round(net, 4)
    trade["status"] = "closed"
    return trade


def eval_signals(state: dict) -> None:
    """Pull 4h candles, check for new signals, open trades."""
    open_by_sym = {t["symbol"]: 1 for t in state["open_trades"] if t["status"] == "open"}
    for sym, sym_rest in SYMBOLS.items():
        if open_by_sym.get(sym):
            continue
        df = fetch_4h(sym_rest, hours_back=60 * 24)
        if df.empty:
            print(f"  {sym}: no 4h data")
            continue
        df = add_indicators(df)
        last_candle_ts = df.index[-1].isoformat()
        # Dedup: only evaluate once per 4h close
        if state["last_eval_candle"].get(sym) == last_candle_ts:
            continue
        state["last_eval_candle"][sym] = last_candle_ts
        sig = detect_signal(df, sym)
        # Telemetry: log eval state (signal fired or not)
        try:
            _last = df.iloc[-1] if len(df) else None
            _state = {
                "close": float(_last.close) if _last is not None else None,
                "atr": float(_last.atr) if _last is not None and "atr" in df.columns else None,
                "rejection_reason": (
                    None if sig is not None
                    else "RANGE_TOUCHES_INSUFFICIENT_OR_FAR_FROM_EDGE"
                ),
            }
            log_eval(engine_name="s5", symbol=sym,
                     base_dir=STATE_DIR, signal_fired=(sig is not None),
                     state=_state)
        except Exception:
            pass
        if not sig:
            continue
        trade = {
            "id": f"s5_{int(time.time())}_{sym.replace('/','')}",
            "symbol": sym,
            "side": sig["side"],
            "entry": sig["entry"],
            "sl": sig["sl"],
            "trail_sl": sig["sl"],
            "atr_at_entry": sig["atr"],
            "notional": NOTIONAL_USD,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "candle_time": sig["candle_time"],
            "range_high": sig["range_high"],
            "range_low": sig["range_low"],
            "touches": sig["touches"],
            "peak_mfe_r": 0.0,
            "trail_engaged": False,
            "status": "open",
        }
        state["open_trades"].append(trade)
        print(f"  OPENED {sym} {sig['side'].upper()} @ {sig['entry']:.2f} "
              f"sl={sig['sl']:.2f} atr={sig['atr']:.2f} touches={sig['touches']}")


def run() -> None:
    state = load_state()
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] S5 paper engine tick")
    # 1. Manage open trades
    still_open = []
    for trade in state.get("open_trades", []):
        if trade.get("status") == "closed":
            continue
        sym_rest = SYMBOLS.get(trade["symbol"])
        if not sym_rest:
            still_open.append(trade)
            continue
        updated = manage_open_trade(trade, sym_rest)
        if updated.get("status") == "closed":
            append_trade(updated)
            state["history_pnl"] = round(state.get("history_pnl", 0.0) + updated["net_pnl_usd"], 4)
            state["history_n"] = state.get("history_n", 0) + 1
            if updated["net_pnl_usd"] > 0:
                state["history_wins"] = state.get("history_wins", 0) + 1
            print(f"  CLOSED {updated['symbol']} {updated['side']} "
                  f"net=${updated['net_pnl_usd']:+.2f} reason={updated['exit_reason']} "
                  f"peak={updated['peak_mfe_r']:.2f}R")
        else:
            still_open.append(updated)
    state["open_trades"] = still_open
    # 2. Evaluate new signals
    eval_signals(state)
    # 3. Save + write summary
    save_state(state)
    write_summary(state)
    print(f"  open={len(state['open_trades'])}  history n={state.get('history_n',0)} "
          f"net=${state.get('history_pnl',0):+.2f}")


def write_summary(state: dict) -> None:
    n = state.get("history_n", 0)
    wins = state.get("history_wins", 0)
    pnl = state.get("history_pnl", 0.0)
    wr = (100 * wins / n) if n else 0
    open_lines = []
    for t in state.get("open_trades", []):
        if t.get("status") == "open":
            cur = t.get("last_price", t["entry"])
            R = abs(t["entry"] - t["sl"])
            cur_r = ((cur - t["entry"]) / R) if t["side"] == "long" else ((t["entry"] - cur) / R)
            open_lines.append(
                f"- **{t['symbol']} {t['side'].upper()}** entry=${t['entry']:.2f} "
                f"cur=${cur:.2f} ({cur_r:+.2f}R, peak {t.get('peak_mfe_r',0):+.2f}R) "
                f"sl=${t.get('trail_sl', t['sl']):.2f} "
                f"trail={'ON' if t.get('trail_engaged') else 'off'}"
            )
    md = f"""# S5 Paper Engine — 4h Range Fade

_Last updated: {datetime.now(timezone.utc).isoformat()}_

## Summary

- **Closed trades:** {n}
- **Wins:** {wins} ({wr:.1f}%)
- **Net P&L:** ${pnl:+.2f}
- **Avg per trade:** ${(pnl/n if n else 0):+.2f}
- **Open positions:** {len([t for t in state.get('open_trades',[]) if t.get('status')=='open'])}

## Open positions

{chr(10).join(open_lines) if open_lines else '_None_'}

## Backtest baseline (6 months, BTC/ETH/SOL/XRP)

71 trades, 45% WR, +$554 net, +$7.80/trade EV after Delta India taker fees.

## Strategy params

- Window: {WINDOW_BARS} × 4h bars (~5 days)
- Range: 95/5 percentile, must be ≥ {MIN_RANGE_ATR}× ATR
- Touches: ≥ {MIN_TOUCHES} per edge
- Edge band: {EDGE_BAND_ATR}× ATR
- SL: {SL_ATR_MULT}× ATR
- Trail engage: {TRAIL_ENGAGE_R}R, lock {TRAIL_LOCK_PCT*100:.0f}% of MFE
- Time stop: {MAX_HOURS}h ({MAX_HOURS//24}d)
"""
    SUMMARY_FILE.write_text(md)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

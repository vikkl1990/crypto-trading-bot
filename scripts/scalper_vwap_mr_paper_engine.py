#!/usr/bin/env python3
"""Scalper-Offer VWAP Mean Reversion paper engine.

Walk-forward-validated under the Delta India Scalper-Offer fee model.
Re-tests a strategy class previously KILLED (full taker round-trip math)
under corrected fees: exit fee waived for BTC/ETH closed within 30 min.

Walk-forward result (storage/scalper_vwap_mr/walkforward.json):
  Variant C (maker entry + scalper offer):
    cell `tf=15m|k=2.0|lb=50|atr_pct<=0.4`
    IS Q1+Q2 :  n=96   EV/trade=$+0.343
    OOS Q3   :  n=17   EV/trade=$+0.910 (gap +265%)
    OOS Q4   :  n=29   EV/trade=$+0.341 (gap +99%)
    All-quarters: n=142, EV=$+0.41/trade, scalper_applied=100%
  Verdict: PASS walk-forward (both OOS positive, Q4 beats $0.10 ship gate).

Note: Variants A (full taker) and B (taker-entry + scalper offer) both
KILL — strategy only revives when paired with maker-entry economics.
This engine therefore submits LIMIT orders at the band level.

Strategy spec (mirrors backtest exactly):
  Trading TF        : 15m
  Bands             : VWAP ± 2.0 × stdev(close, 50 bars)
  Lookback          : 50 bars (rolling VWAP and stdev window)
  Regime gate       : ATR_14 percentile rank over last 100 bars ≤ 0.40
                      (RANGING regime only)
  LONG entry        : 15m close < (VWAP - 2.0×stdev) AND
                      reversal candle (close > open AND close > prev close) AND
                      ranging
                      → LIMIT BUY at lower band (maker entry)
  SHORT entry       : mirror at upper band
  Stop              : 1× ATR_14 beyond entry on the wrong side
  Target            : VWAP touch (mean revert)
  Time stop         : 28 minutes (2-min margin under 30min scalper window)
  Notional          : $1000
  Symbols           : BTC/USDT, ETH/USDT only (scalper-offer eligibility)

Paper engine architecture mirrors smc15v2_paper_engine.py:
  State dir:      storage/scalper_vwap_mr_paper/
  State file:     state.json
  Trades file:    trades.jsonl
  Summary file:   latest.md
  Cron:           */5 * * * *

This engine is independent — it does NOT touch scalp_strategy.py, the live bot,
or any signal_tracker / signal_journey / signal_learner code. Default OFF
unless explicitly enabled via cron.
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

STATE_DIR = ROOT / "storage" / "scalper_vwap_mr_paper"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
SUMMARY_FILE = STATE_DIR / "latest.md"
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"

SYMBOLS = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
}

# Strategy params (locked to walk-forward winning cell for variant C)
TF = "15m"
TF_RES = "15m"
TF_MIN = 15
K = 2.0
LOOKBACK = 50
ATR_PCT_THR = 0.40
ATR_PCT_LOOKBACK = 100
ATR_PERIOD = 14
SL_ATR_MULT = 1.0
TIME_STOP_MIN = 28
NOTIONAL_USD = 1000.0
LIMIT_EXPIRY_MIN = TF_MIN  # cancel pending limit after one bar (don't chase)
HOURS_BACK_FETCH = 24 * 7  # 7 days of 15m data plenty for 100-bar regime window


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────
def fetch_candles(sym_rest: str, hours_back: int = HOURS_BACK_FETCH) -> pd.DataFrame:
    end = int(time.time())
    start = end - hours_back * 3600
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": sym_rest, "resolution": TF_RES, "start": start, "end": end},
            timeout=15,
        )
        rows = r.json().get("result", [])
    except Exception as e:
        print(f"  REST fetch {sym_rest} failed: {e}")
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
    start = end - 300
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": sym_rest, "resolution": "1m", "start": start, "end": end},
            timeout=8,
        )
        rows = r.json().get("result", [])
        if not rows:
            return None
        return float(rows[-1]["close"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Indicators (same as backtest)
# ──────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = out["high"]; low = out["low"]; close = out["close"]
    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    out["atr_pct"] = out["atr"].rolling(ATR_PCT_LOOKBACK,
                                        min_periods=ATR_PCT_LOOKBACK).rank(pct=True)
    # VWAP & stdev bands (rolling typical-price weighted)
    typ = (high + low + close) / 3.0
    vol = out["volume"].clip(lower=1e-9)
    pv = typ * vol
    out["vwap"] = (pv.rolling(LOOKBACK, min_periods=LOOKBACK).sum()
                   / vol.rolling(LOOKBACK, min_periods=LOOKBACK).sum())
    out["stdev"] = close.rolling(LOOKBACK, min_periods=LOOKBACK).std(ddof=0)
    out["upper"] = out["vwap"] + K * out["stdev"]
    out["lower"] = out["vwap"] - K * out["stdev"]
    return out


def detect_signal(df: pd.DataFrame) -> Optional[dict]:
    """Look for a fresh signal on the LAST CLOSED bar.

    Return {side, limit_price, sl, atr, vwap, signal_ts} or None.
    """
    if len(df) < max(LOOKBACK, ATR_PCT_LOOKBACK) + 5:
        return None
    df = add_indicators(df)
    # The last fully-closed candle is index -1 (Delta returns closed bars)
    i = len(df) - 1
    row = df.iloc[i]
    if any(pd.isna(row[c]) for c in ("vwap", "stdev", "atr", "atr_pct", "upper", "lower")):
        return None
    if row["atr_pct"] > ATR_PCT_THR:
        return None  # not ranging

    prev = df.iloc[i - 1]
    bull_rev = (row["close"] > row["open"]) and (row["close"] > prev["close"])
    bear_rev = (row["close"] < row["open"]) and (row["close"] < prev["close"])

    side = None
    if (row["close"] < row["lower"]) and bull_rev:
        side = "long"
    elif (row["close"] > row["upper"]) and bear_rev:
        side = "short"
    if side is None:
        return None

    atr = float(row["atr"])
    vwap = float(row["vwap"])
    if side == "long":
        limit_price = float(row["lower"])
        sl = limit_price - SL_ATR_MULT * atr
    else:
        limit_price = float(row["upper"])
        sl = limit_price + SL_ATR_MULT * atr

    return {
        "side": side,
        "limit_price": limit_price,
        "sl": sl,
        "tp_vwap": vwap,
        "atr": atr,
        "vwap": vwap,
        "stdev": float(row["stdev"]),
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
        "last_signal_candle": {},
        "history_pnl": 0.0,
        "history_n": 0,
        "history_wins": 0,
        "history_scalper_applied": 0,
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_trade(trade: dict) -> None:
    with TRADES_FILE.open("a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Trade lifecycle
# ──────────────────────────────────────────────────────────────────────
FEE_MODEL = FeeModel()


def close_trade(trade: dict, exit_price: float, reason: str) -> dict:
    qty = trade["notional"] / trade["entry"]
    if trade["side"] == "long":
        gross = (exit_price - trade["entry"]) * qty
    else:
        gross = (trade["entry"] - exit_price) * qty

    opened = datetime.fromisoformat(trade["opened_at"])
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    closed_at = datetime.now(timezone.utc)
    holding_sec = (closed_at - opened).total_seconds()

    # Maker entry + scalper-offer-aware exit (taker if limit fails / scalper if eligible)
    fee_info = FEE_MODEL.round_trip_for_trade(
        exchange="delta", entry_type="maker", exit_type="taker",
        notional_usd=trade["notional"], symbol=trade["symbol"],
        holding_sec=holding_sec, force_no_scalper=False,
    )
    fees = fee_info["fee_usd"]
    net = gross - fees

    trade["exit_price"] = exit_price
    trade["exit_reason"] = reason
    trade["closed_at"] = closed_at.isoformat()
    trade["holding_sec"] = round(holding_sec, 1)
    trade["gross_pnl_usd"] = round(gross, 4)
    trade["fees_usd"] = round(fees, 4)
    trade["net_pnl_usd"] = round(net, 4)
    trade["scalper_applied"] = bool(fee_info["scalper_applied"])
    trade["fee_breakdown"] = {
        "entry_fee_usd": round(fee_info["entry_fee_usd"], 4),
        "exit_fee_usd": round(fee_info["exit_fee_usd"], 4),
        "entry_rate": fee_info["entry_rate"],
        "exit_rate_charged": fee_info["exit_rate_charged"],
    }
    trade["status"] = "closed"
    return trade


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
        return close_trade(trade, px, "time_stop_28min")

    # 2. SL hit
    if trade["side"] == "long" and px <= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")
    if trade["side"] == "short" and px >= trade["sl"]:
        return close_trade(trade, trade["sl"], "sl_hit")

    # 3. VWAP target (mean revert)
    tp = trade["tp_vwap"]
    if trade["side"] == "long" and px >= tp:
        return close_trade(trade, tp, "vwap_touch")
    if trade["side"] == "short" and px <= tp:
        return close_trade(trade, tp, "vwap_touch")

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
        "id": f"vwapmr_{int(time.time())}_{pending['symbol'].replace('/', '')}",
        "symbol": pending["symbol"],
        "side": pending["side"],
        "entry": entry,
        "sl": pending["sl"],
        "tp_vwap": pending["tp_vwap"],
        "atr_at_entry": pending["atr"],
        "vwap_at_entry": pending["vwap"],
        "stdev_at_entry": pending["stdev"],
        "notional": NOTIONAL_USD,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "candle_time": pending["signal_ts"],
        "limit_price": pending["limit_price"],
        "peak_mfe_r": 0.0,
        "status": "open",
    }
    return None, trade


def eval_signals(state: dict) -> None:
    open_by_sym = {t["symbol"] for t in state.get("open_trades", [])
                   if t.get("status") != "closed"}
    pending_by_sym = {p["symbol"] for p in state.get("pending_limits", [])}

    for sym, sym_rest in SYMBOLS.items():
        if sym in open_by_sym or sym in pending_by_sym:
            continue
        df = fetch_candles(sym_rest)
        if df.empty or len(df) < max(LOOKBACK, ATR_PCT_LOOKBACK) + 10:
            print(f"  {sym}: insufficient {TF} data ({len(df)} bars)")
            continue
        last_ts = df.index[-1].isoformat()
        if state.get("last_signal_candle", {}).get(sym) == last_ts:
            continue
        state.setdefault("last_signal_candle", {})[sym] = last_ts

        sig = detect_signal(df)
        # Telemetry: capture rich state from indicators
        try:
            _last = df.iloc[-1] if len(df) else None
            _state = {}
            if _last is not None:
                _state["close"] = float(_last.get("close", 0))
                _state["vwap"] = float(_last.get("vwap", 0)) if "vwap" in df.columns else None
                _state["k_dev"] = float(_last.get("k_dev", 0)) if "k_dev" in df.columns else None
                _state["atr_pct_rank"] = float(_last.get("atr_pct_rank", 0)) if "atr_pct_rank" in df.columns else None
            _state["rejection_reason"] = None if sig else "VWAP_MR_NO_SIGNAL_or_GATE_FAIL"
            log_eval(engine_name="scalper_vwap_mr", symbol=sym,
                     base_dir=STATE_DIR, signal_fired=(sig is not None), state=_state)
        except Exception:
            pass
        if not sig:
            continue
        sig["symbol"] = sym
        sig["placed_at"] = datetime.now(timezone.utc).isoformat()
        state.setdefault("pending_limits", []).append(sig)
        print(f"  PENDING {sym} {sig['side'].upper()} LIMIT @ {sig['limit_price']:.4f} "
              f"sl={sig['sl']:.4f} vwap={sig['vwap']:.4f}")


def write_summary(state: dict) -> None:
    n = state.get("history_n", 0)
    wins = state.get("history_wins", 0)
    pnl = state.get("history_pnl", 0.0)
    wr = (100 * wins / n) if n else 0
    sa = state.get("history_scalper_applied", 0)
    sa_pct = (100 * sa / n) if n else 0

    open_lines = []
    for t in state.get("open_trades", []):
        if t.get("status") == "open":
            cur = t.get("last_price", t["entry"])
            R = abs(t["entry"] - t["sl"])
            cur_r = ((cur - t["entry"]) / R) if t["side"] == "long" else \
                    ((t["entry"] - cur) / R)
            opened = datetime.fromisoformat(t["opened_at"])
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            age_min = (datetime.now(timezone.utc) - opened).total_seconds() / 60.0
            open_lines.append(
                f"- **{t['symbol']} {t['side'].upper()}** "
                f"entry=${t['entry']:.4f} cur=${cur:.4f} ({cur_r:+.2f}R, "
                f"peak {t.get('peak_mfe_r', 0):+.2f}R) sl=${t['sl']:.4f} "
                f"vwap=${t['tp_vwap']:.4f} age={age_min:.1f}m / {TIME_STOP_MIN}m"
            )
    pending_lines = []
    for p in state.get("pending_limits", []):
        cur = p.get("last_price", p["limit_price"])
        placed = datetime.fromisoformat(p["placed_at"])
        if placed.tzinfo is None:
            placed = placed.replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - placed).total_seconds() / 60.0
        pending_lines.append(
            f"- **{p['symbol']} {p['side'].upper()}** LIMIT @ ${p['limit_price']:.4f} "
            f"cur=${cur:.4f} sl=${p['sl']:.4f} vwap=${p['tp_vwap']:.4f} "
            f"age={age_min:.1f}m / {LIMIT_EXPIRY_MIN}m"
        )

    md = f"""# Scalper-Offer VWAP MR Paper Engine

_Last updated: {datetime.now(timezone.utc).isoformat()}_

## Summary
- **Closed trades:** {n}
- **Wins:** {wins} ({wr:.1f}%)
- **Net P&L:** ${pnl:+.2f}
- **Avg per trade:** ${(pnl/n if n else 0):+.3f}
- **Scalper-offer applied:** {sa}/{n} ({sa_pct:.0f}%)
- **Open positions:** {len([t for t in state.get('open_trades', []) if t.get('status') == 'open'])}
- **Pending limits:** {len(state.get('pending_limits', []))}

## Open positions
{chr(10).join(open_lines) if open_lines else '_None_'}

## Pending limits
{chr(10).join(pending_lines) if pending_lines else '_None_'}

## Walk-forward backtest baseline (variant C — maker entry + scalper offer)
Walk-forward pass on cell `tf=15m|k=2.0|lb=50|atr_pct<=0.4`:
  - IS Q1+Q2:  n=96   EV/trade=$+0.343
  - OOS Q3:    n=17   EV/trade=$+0.910 (gap +265%)
  - OOS Q4:    n=29   EV/trade=$+0.341 (gap +99%)
  - Total:     n=142  EV/trade=$+0.41/trade  scalper_applied=100%

Verdict: PASS walk-forward (both OOS positive, Q4 beats $0.10 ship gate).
Variants A (full taker) and B (taker+scalper-offer) both KILL — strategy
only revives with maker-entry economics.

## Strategy params (locked to backtest cell)
- TF                : {TF}
- Bands             : VWAP ± {K}× stdev (rolling typical-price weighted, lookback {LOOKBACK})
- Regime gate       : ATR_{ATR_PERIOD} percentile ≤ {ATR_PCT_THR} over last {ATR_PCT_LOOKBACK} bars
- Reversal candle   : close > open AND close > prev close (LONG; mirror SHORT)
- Entry             : LIMIT at band level (maker)
- Stop              : {SL_ATR_MULT}× ATR_{ATR_PERIOD} beyond entry on the wrong side
- Target            : VWAP touch
- Time stop         : {TIME_STOP_MIN} min (under 30min scalper-offer window)
- Symbols           : {', '.join(SYMBOLS.keys())} (scalper-offer eligibility)
- Notional          : ${NOTIONAL_USD}
- Limit expiry      : {LIMIT_EXPIRY_MIN} min (one bar)

## Cost model (Delta India)
- Entry leg : maker  0.024%
- Exit leg  : 0% if scalper-offer eligible (BTC/ETH ≤ 30 min), else taker 0.06%
- Expected RT under scalper offer: 0.024% (vs 0.118% old kill math)
"""
    SUMMARY_FILE.write_text(md)


# ──────────────────────────────────────────────────────────────────────
# Tick
# ──────────────────────────────────────────────────────────────────────
def run() -> None:
    state = load_state()
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] scalper-VWAP-MR tick")

    # 1) Manage open trades
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
            state["history_pnl"] = round(state.get("history_pnl", 0.0)
                                         + updated["net_pnl_usd"], 4)
            state["history_n"] = state.get("history_n", 0) + 1
            if updated["net_pnl_usd"] > 0:
                state["history_wins"] = state.get("history_wins", 0) + 1
            if updated.get("scalper_applied"):
                state["history_scalper_applied"] = state.get("history_scalper_applied", 0) + 1
            print(f"  CLOSED {updated['symbol']} {updated['side']} "
                  f"net=${updated['net_pnl_usd']:+.2f} "
                  f"reason={updated['exit_reason']} "
                  f"scalper={'Y' if updated.get('scalper_applied') else 'N'}")
        else:
            still_open.append(updated)
    state["open_trades"] = still_open

    # 2) Manage pending limits
    still_pending = []
    for pending in state.get("pending_limits", []):
        sym_rest = SYMBOLS.get(pending["symbol"])
        if not sym_rest:
            still_pending.append(pending)
            continue
        updated_pending, new_trade = manage_pending_limit(pending, sym_rest)
        if new_trade is not None:
            state["open_trades"].append(new_trade)
            print(f"  FILLED {new_trade['symbol']} {new_trade['side'].upper()} "
                  f"@ {new_trade['entry']:.4f} sl={new_trade['sl']:.4f} "
                  f"tp={new_trade['tp_vwap']:.4f}")
        elif updated_pending is not None:
            still_pending.append(updated_pending)
        else:
            print(f"  EXPIRED pending limit {pending['symbol']} "
                  f"{pending['side'].upper()}")
    state["pending_limits"] = still_pending

    # 3) Eval new signals
    eval_signals(state)

    save_state(state)
    write_summary(state)
    print(f"  open={len(state['open_trades'])}  "
          f"pending={len(state.get('pending_limits', []))}  "
          f"history n={state.get('history_n', 0)} "
          f"net=${state.get('history_pnl', 0):+.2f} "
          f"scalper_pct={(100 * state.get('history_scalper_applied', 0) / max(1, state.get('history_n', 0))):.0f}%")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

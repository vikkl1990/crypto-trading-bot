#!/usr/bin/env python3
"""SMC1 (1h Order Block + FVG retest) standalone paper-trading engine.

Designed to run every 5 minutes via cron. Independent of the 5m scalper and S5 engine.
Implements PROPER order-block and fair-value-gap detection (the backtest used an
EMA proxy that produced +$1.60 EV/trade on n=40; this implementation should give
cleaner signal quality once forward data accumulates).

Strategy:
  1. Pull 1h candles from Delta REST (BTC/ETH/SOL/XRP)
  2. Detect Order Blocks: last opposite-direction candle BEFORE a displacement
     - Displacement = 1h candle range > 1.5 × ATR_14 AND body > 60% of range
  3. Detect Fair Value Gaps: 3-candle imbalance (gap between candle[i-1] and
     candle[i+1] that candle[i] couldn't fully cover)
  4. Track ACTIVE OBs (not yet mitigated by full retrace through their zone)
  5. Entry trigger: price retraces back INTO an OB zone with a reversal candle
     in the direction of original displacement
  6. Confluence bonus when OB + FVG overlap on the same zone
  7. SL: 1 × ATR_1h beyond OB extreme (full mitigation = stopped out)
  8. Exit (E_smc1): 2.5R TP / 1R SL / 4h time stop, with optional trail at 1.5R

Backtest baseline (5.5mo 4-sym 1h, EMA approx): +$1.60 net EV/trade, 55% WR, n=40.

State: storage/smc1_paper/state.json
Trades: storage/smc1_paper/trades.jsonl
Summary: storage/smc1_paper/latest.md
Cron suggestion: */5 * * * *

Usage: python3 scripts/smc1_paper_engine.py
"""
from __future__ import annotations
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

import sys as _sys_telem
_sys_telem.path.insert(0, "/home/opc/crypto-trading-bot")
from execution_v2.engine_telemetry import log_eval

ROOT = Path("/home/opc/crypto-trading-bot")
STATE_DIR = ROOT / "storage" / "smc1_paper"
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

# Cost model
TAKER_FEE = 0.00059
NOTIONAL_USD = 1000.0

# Strategy params
ATR_LEN = 14
DISPLACEMENT_ATR_MULT = 1.5    # 1h candle range must exceed 1.5× ATR
DISPLACEMENT_BODY_FRAC = 0.60  # body must be >= 60% of range
OB_LOOKBACK_BARS = 50          # how far back to look for active OBs (1h bars)
OB_MAX_AGE_BARS = 30           # OB invalidates after 30 hours unmitigated
SL_ATR_MULT = 1.0
TP_R = 2.5
SL_R_FROM_TP = 1.0             # SL distance / TP distance ratio (we use absolute R)
TRAIL_ENGAGE_R = 1.5
TRAIL_LOCK_PCT = 0.40
MAX_HOURS = 4                  # time stop


# ──────────────────────────────────────────────────────────────────────
# Data fetch + indicators
# ──────────────────────────────────────────────────────────────────────

def fetch_1h(sym_rest: str, hours_back: int = 24 * 14) -> pd.DataFrame:
    end = int(time.time())
    start = end - hours_back * 3600
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": sym_rest, "resolution": "1h", "start": start, "end": end},
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


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < ATR_LEN + 2:
        return df
    df = df.copy()
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    return df


def get_current_price(sym_rest: str) -> float | None:
    end = int(time.time())
    start = end - 300
    try:
        r = requests.get(DELTA_REST,
                         params={"symbol": sym_rest, "resolution": "1m",
                                 "start": start, "end": end},
                         timeout=8)
        rows = r.json().get("result", [])
        if not rows:
            return None
        return float(rows[-1]["close"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Order Block + FVG detection
# ──────────────────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    """An Order Block — last opposite-direction candle before displacement."""
    direction: str          # "bull" if we expect price to bounce UP from this OB → long
                            # "bear" if we expect price to fall DOWN from OB → short
    formed_at: str          # ISO timestamp of the OB candle
    high: float
    low: float
    midpoint: float
    displacement_size_atr: float  # how big was the move that left this OB
    has_fvg: bool = False   # confluence flag

    def in_zone(self, price: float, tol: float = 0.0) -> bool:
        return (self.low - tol) <= price <= (self.high + tol)


def detect_obs_and_fvgs(df: pd.DataFrame) -> List[OrderBlock]:
    """Walk 1h candles, find recent active OBs.

    OB rules:
      - Find a displacement candle (range > 1.5×ATR, body > 60% range)
      - The LAST opposite-color candle before it = the Order Block
      - Direction of OB = direction of displacement
      - OB stays "active" until price fully passes through its zone (mitigated)

    FVG rules (for confluence):
      - For 3 consecutive bars [a, b, c]:
        - Bullish FVG if c.low > a.high (gap up; b is the displacement)
        - Bearish FVG if c.high < a.low (gap down)
      - The FVG zone is [a.high, c.low] (bull) or [c.high, a.low] (bear)
      - If an OB overlaps with an FVG, mark it has_fvg=True
    """
    if len(df) < OB_LOOKBACK_BARS or "atr" not in df.columns:
        return []

    obs: List[OrderBlock] = []
    n = len(df)
    last_price = float(df.iloc[-1].close)

    # Look at the recent OB_LOOKBACK_BARS for displacement events
    start = max(2, n - OB_LOOKBACK_BARS)
    for i in range(start, n):
        bar = df.iloc[i]
        if pd.isna(bar.atr) or bar.atr <= 0:
            continue
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            continue
        body = abs(bar.close - bar.open)
        if (bar_range < DISPLACEMENT_ATR_MULT * bar.atr or
                body / bar_range < DISPLACEMENT_BODY_FRAC):
            continue

        # Determine direction
        is_bull_disp = bar.close > bar.open
        # Find the last OPPOSITE-direction candle BEFORE this displacement
        ob_bar = None
        for j in range(i - 1, max(start - 5, 0), -1):
            prev = df.iloc[j]
            if is_bull_disp and prev.close < prev.open:  # bear candle = bull OB
                ob_bar = prev
                ob_idx = j
                break
            if (not is_bull_disp) and prev.close > prev.open:
                ob_bar = prev
                ob_idx = j
                break
        if ob_bar is None:
            continue

        # Mitigation check: has price subsequently passed THROUGH the OB zone?
        # Bull OB: mitigated if any subsequent low went BELOW ob.low
        # Bear OB: mitigated if any subsequent high went ABOVE ob.high
        post_bars = df.iloc[i:n]  # bars AFTER the displacement
        if is_bull_disp:
            if (post_bars.low < ob_bar.low).any():
                continue  # bull OB mitigated, skip
        else:
            if (post_bars.high > ob_bar.high).any():
                continue

        # Age check
        bars_since_ob = n - 1 - ob_idx
        if bars_since_ob > OB_MAX_AGE_BARS:
            continue

        ob = OrderBlock(
            direction="bull" if is_bull_disp else "bear",
            formed_at=ob_bar.name.isoformat(),
            high=float(ob_bar.high),
            low=float(ob_bar.low),
            midpoint=float((ob_bar.high + ob_bar.low) / 2),
            displacement_size_atr=float(bar_range / bar.atr),
        )

        # FVG confluence check
        if i + 1 < n:
            a = df.iloc[i - 1]
            c = df.iloc[i + 1]
            if ob.direction == "bull" and c.low > a.high:
                # bullish FVG zone [a.high, c.low]
                if not (ob.high < a.high or ob.low > c.low):
                    ob.has_fvg = True
            elif ob.direction == "bear" and c.high < a.low:
                if not (ob.low > a.low or ob.high < c.high):
                    ob.has_fvg = True

        obs.append(ob)

    # Deduplicate: keep only the most recent OB per direction
    bull_obs = [o for o in obs if o.direction == "bull"]
    bear_obs = [o for o in obs if o.direction == "bear"]
    bull_obs.sort(key=lambda o: o.formed_at, reverse=True)
    bear_obs.sort(key=lambda o: o.formed_at, reverse=True)
    return (bull_obs[:1] + bear_obs[:1])


# ──────────────────────────────────────────────────────────────────────
# Signal detection
# ──────────────────────────────────────────────────────────────────────

def detect_signal(df_1h: pd.DataFrame, symbol: str) -> dict | None:
    """Check if current 1h close is a tradeable SMC1 setup."""
    if len(df_1h) < OB_LOOKBACK_BARS + 2:
        return None
    df = add_atr(df_1h)
    if "atr" not in df.columns:
        return None
    obs = detect_obs_and_fvgs(df)
    if not obs:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(last.atr) or last.atr <= 0:
        return None

    for ob in obs:
        in_zone = ob.in_zone(float(last.close), tol=0.1 * last.atr)
        if not in_zone:
            continue
        # Reversal candle in direction of OB displacement
        if ob.direction == "bull":
            # Need bullish reversal: close > open AND close > prev.close
            if last.close > last.open and last.close > prev.close:
                sl = ob.low - SL_ATR_MULT * last.atr  # below OB
                return {
                    "side": "long",
                    "entry": float(last.close),
                    "sl": float(sl),
                    "atr": float(last.atr),
                    "ob_high": ob.high,
                    "ob_low": ob.low,
                    "ob_age_bars": (datetime.now(timezone.utc)
                                    - datetime.fromisoformat(ob.formed_at)
                                    ).total_seconds() / 3600,
                    "displacement_atr": ob.displacement_size_atr,
                    "has_fvg": ob.has_fvg,
                    "candle_time": last.name.isoformat(),
                }
        else:  # bear
            if last.close < last.open and last.close < prev.close:
                sl = ob.high + SL_ATR_MULT * last.atr  # above OB
                return {
                    "side": "short",
                    "entry": float(last.close),
                    "sl": float(sl),
                    "atr": float(last.atr),
                    "ob_high": ob.high,
                    "ob_low": ob.low,
                    "ob_age_bars": (datetime.now(timezone.utc)
                                    - datetime.fromisoformat(ob.formed_at)
                                    ).total_seconds() / 3600,
                    "displacement_atr": ob.displacement_size_atr,
                    "has_fvg": ob.has_fvg,
                    "candle_time": last.name.isoformat(),
                }
    return None


# ──────────────────────────────────────────────────────────────────────
# Trade management
# ──────────────────────────────────────────────────────────────────────

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


def manage_open_trade(trade: dict, sym_rest: str) -> dict:
    px = get_current_price(sym_rest)
    if px is None:
        return trade
    R = abs(trade["entry"] - trade["sl"])
    if R == 0:
        return trade
    if trade["side"] == "long":
        cur_r = (px - trade["entry"]) / R
        tp_price = trade["entry"] + TP_R * R
    else:
        cur_r = (trade["entry"] - px) / R
        tp_price = trade["entry"] - TP_R * R

    trade["peak_mfe_r"] = max(trade.get("peak_mfe_r", 0.0), cur_r)
    trade["last_price"] = px
    trade["last_check"] = datetime.now(timezone.utc).isoformat()

    # Time stop
    opened = datetime.fromisoformat(trade["opened_at"])
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    hours_held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0
    if hours_held >= MAX_HOURS:
        return close_trade(trade, px, "time_stop")

    # Trail engagement
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

    # TP hit
    if trade["side"] == "long" and px >= tp_price:
        return close_trade(trade, tp_price, "tp_25R")
    if trade["side"] == "short" and px <= tp_price:
        return close_trade(trade, tp_price, "tp_25R")

    # SL hit (or trail SL)
    if trade["side"] == "long" and px <= trail_sl:
        return close_trade(trade, trail_sl,
                           "trail_sl" if trade.get("trail_engaged") else "sl_hit")
    if trade["side"] == "short" and px >= trail_sl:
        return close_trade(trade, trail_sl,
                           "trail_sl" if trade.get("trail_engaged") else "sl_hit")
    return trade


def eval_signals(state: dict) -> None:
    open_by_sym = {t["symbol"]: 1 for t in state["open_trades"]
                   if t.get("status") != "closed"}
    for sym, sym_rest in SYMBOLS.items():
        if open_by_sym.get(sym):
            continue
        df_1h = fetch_1h(sym_rest, hours_back=24 * 14)
        if df_1h.empty:
            print(f"  {sym}: no 1h data")
            continue
        last_candle_ts = df_1h.index[-1].isoformat()
        # Dedup: only evaluate once per 1h close
        if state["last_eval_candle"].get(sym) == last_candle_ts:
            continue
        state["last_eval_candle"][sym] = last_candle_ts
        sig = detect_signal(df_1h, sym)
        try:
            _last = df_1h.iloc[-1] if len(df_1h) else None
            _state = {
                "close": float(_last.close) if _last is not None else None,
                "atr": float(_last.atr) if _last is not None and "atr" in df_1h.columns else None,
                "rejection_reason": None if sig is not None else "NO_OB_RETEST_OR_REVERSAL",
            }
            log_eval(engine_name="smc1", symbol=sym,
                     base_dir=STATE_DIR, signal_fired=(sig is not None), state=_state)
        except Exception:
            pass
        if not sig:
            continue
        trade = {
            "id": f"smc1_{int(time.time())}_{sym.replace('/','')}",
            "symbol": sym,
            "side": sig["side"],
            "entry": sig["entry"],
            "sl": sig["sl"],
            "trail_sl": sig["sl"],
            "atr_at_entry": sig["atr"],
            "notional": NOTIONAL_USD,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "candle_time": sig["candle_time"],
            "ob_high": sig["ob_high"],
            "ob_low": sig["ob_low"],
            "ob_age_bars": sig["ob_age_bars"],
            "displacement_atr": sig["displacement_atr"],
            "has_fvg": sig["has_fvg"],
            "peak_mfe_r": 0.0,
            "trail_engaged": False,
            "status": "open",
        }
        state["open_trades"].append(trade)
        print(f"  OPENED {sym} {sig['side'].upper()} @ {sig['entry']:.4f} "
              f"sl={sig['sl']:.4f} OB=[{sig['ob_low']:.4f},{sig['ob_high']:.4f}] "
              f"disp={sig['displacement_atr']:.2f}×ATR fvg={sig['has_fvg']}")


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
                f"- **{t['symbol']} {t['side'].upper()}** "
                f"entry=${t['entry']:.4f} cur=${cur:.4f} ({cur_r:+.2f}R, "
                f"peak {t.get('peak_mfe_r',0):+.2f}R) sl=${t.get('trail_sl', t['sl']):.4f} "
                f"OB=[{t['ob_low']:.4f},{t['ob_high']:.4f}] "
                f"disp={t['displacement_atr']:.2f}×ATR fvg={t['has_fvg']} "
                f"trail={'ON' if t.get('trail_engaged') else 'off'}"
            )
    md = f"""# SMC1 Paper Engine — 1h Order Block + FVG retest

_Last updated: {datetime.now(timezone.utc).isoformat()}_

## Summary

- **Closed trades:** {n}
- **Wins:** {wins} ({wr:.1f}%)
- **Net P&L:** ${pnl:+.2f}
- **Avg per trade:** ${(pnl/n if n else 0):+.2f}
- **Open positions:** {len([t for t in state.get('open_trades',[]) if t.get('status')=='open'])}

## Open positions

{chr(10).join(open_lines) if open_lines else '_None_'}

## Backtest baseline (5.5mo, 4 sym, EMA approx)

n=40, 55% WR, **+$1.60 net EV/trade**. This implementation uses proper OB+FVG
tracking instead of EMA proxy — forward-validate before drawing conclusions.

## Strategy params

- Displacement: 1h candle range > {DISPLACEMENT_ATR_MULT}× ATR_14, body > {DISPLACEMENT_BODY_FRAC*100:.0f}% of range
- OB lookback: {OB_LOOKBACK_BARS} bars
- OB max age: {OB_MAX_AGE_BARS} bars before invalidation
- SL: {SL_ATR_MULT}× ATR beyond OB extreme
- TP: {TP_R}R fixed
- Trail: engage at {TRAIL_ENGAGE_R}R, lock {TRAIL_LOCK_PCT*100:.0f}% of MFE peak
- Time stop: {MAX_HOURS}h
- Cost: 0.059% taker × 2 = 0.118% RT
"""
    SUMMARY_FILE.write_text(md)


def run() -> None:
    state = load_state()
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] SMC1 paper engine tick")
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
            state["history_pnl"] = round(state.get("history_pnl", 0.0)
                                         + updated["net_pnl_usd"], 4)
            state["history_n"] = state.get("history_n", 0) + 1
            if updated["net_pnl_usd"] > 0:
                state["history_wins"] = state.get("history_wins", 0) + 1
            print(f"  CLOSED {updated['symbol']} {updated['side']} "
                  f"net=${updated['net_pnl_usd']:+.2f} "
                  f"reason={updated['exit_reason']} "
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


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

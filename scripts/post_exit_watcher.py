#!/usr/bin/env python3
"""
post_exit_watcher.py

Track price action at +5 / +15 / +30 / +60 minutes AFTER a shadow trade exit.
Goal: detect whether the exit guards (especially time_decay_10m) close trades
TOO EARLY (price kept moving favorably) or correctly (price stalled / reversed).

R math
------
risk_unit = |entry_price - stop_loss|
For SHORT: post_R = (exit_price - future_price) / risk_unit  (positive = price fell further = exited too early)
For LONG : post_R = (future_price - exit_price) / risk_unit  (positive = price rose further = exited too early)

A "future_price" is taken from the 1m close of the candle whose CLOSE time is
the first one strictly >= (exit_at + checkpoint_minutes). Candles that haven't
formed yet (i.e. now() < target time) are skipped — the record is rewritten
on the next run when more checkpoints become available.

Usage
-----
Run as a cron job, e.g. every 2 minutes:

    */2 * * * * /usr/bin/python3 /home/opc/crypto-trading-bot/scripts/post_exit_watcher.py >> /home/opc/crypto-trading-bot/logs/post_exit_watcher.log 2>&1

Idempotent: a trade is reprocessed until all 4 checkpoints (5/15/30/60) are
filled, then it is moved to DONE state and not touched again.

Read-only on user_trades. No imports from the bot codebase.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path("/home/opc/crypto-trading-bot")
STORAGE_DIR = BASE_DIR / "storage" / "post_exit"
LOG_FILE = STORAGE_DIR / "post_exit_log.jsonl"
STATE_FILE = STORAGE_DIR / "state.json"
SUMMARY_FILE = STORAGE_DIR / "summary.md"

DB_DSN = "host=localhost dbname=vnedge user=vnedge password=VnEdge2026db"

DELTA_BASE = "https://api.india.delta.exchange/v2/history/candles"
HTTP_TIMEOUT = 10
INTER_CALL_SLEEP = 0.15  # ~6 req/s -> well under Delta REST limits

LOOKBACK_MINUTES = 90  # only consider trades that closed within this window
CHECKPOINTS_MIN = (5, 15, 30, 60)
SUMMARY_MIN_TRADES = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def to_delta_symbol(sym: str) -> str:
    """BTC/USDT -> BTCUSD ; ETHUSDT -> ETHUSD ; already 'BTCUSD' -> unchanged."""
    s = sym.replace("/", "").upper()
    if s.endswith("USDT"):
        s = s[:-4] + "USD"
    return s


def fetch_candles(symbol: str, start_ts: int, end_ts: int) -> list[dict]:
    """Pull 1m candles. Returns list sorted ASCENDING by time. Empty list on error."""
    params = {
        "symbol": symbol,
        "resolution": "1m",
        "start": str(int(start_ts)),
        "end": str(int(end_ts)),
    }
    url = f"{DELTA_BASE}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vnedge-post-exit-watcher/1.0"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log(f"  WARN candle fetch failed for {symbol} {start_ts}..{end_ts}: {exc}")
        return []
    candles = data.get("result") or []
    candles.sort(key=lambda c: c.get("time", 0))
    return candles


def first_close_at_or_after(candles: list[dict], target_ts: int) -> tuple[int | None, float | None]:
    """Return (candle_time, close_price) for the first 1m candle whose time >= target_ts."""
    for c in candles:
        if c.get("time", 0) >= target_ts:
            return int(c["time"]), float(c["close"])
    return None, None


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"processed": {}}  # trade_id -> {"checkpoints_filled": [...], "done": bool}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        log("WARN state file unreadable, starting fresh")
        return {"processed": {}}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def append_or_replace_record(record: dict) -> None:
    """Rewrite log file replacing any existing entry for the same trade_id."""
    trade_id = record["trade_id"]
    existing: list[dict] = []
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("trade_id") != trade_id:
                existing.append(row)
    existing.append(record)
    tmp = LOG_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, sort_keys=True) for r in existing) + "\n")
    tmp.replace(LOG_FILE)


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


def fetch_recent_shadow_exits() -> list[dict]:
    sql = """
        SELECT
            id::text                      AS trade_id,
            symbol,
            side,
            entry_price,
            exit_price,
            EXTRACT(EPOCH FROM closed_at) AS closed_at_ts,
            closed_at,
            metadata
        FROM   user_trades
        WHERE  trade_type = 'shadow'
          AND  status     = 'closed'
          AND  closed_at IS NOT NULL
          AND  closed_at >= NOW() - INTERVAL %s
          AND  exit_price IS NOT NULL
          AND  entry_price IS NOT NULL
        ORDER BY closed_at ASC
    """
    interval = f"{LOOKBACK_MINUTES} minutes"
    with psycopg2.connect(DB_DSN) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (interval,))
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def compute_record(trade: dict, candles_cache: dict) -> dict | None:
    """Compute one post-exit record. Returns None if entry data is too broken to use."""
    trade_id = trade["trade_id"]
    symbol_raw = trade["symbol"]
    side = (trade["side"] or "").lower()
    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    closed_ts = int(trade["closed_at_ts"])
    md = trade.get("metadata") or {}
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:
            md = {}
    stop_loss = md.get("stop_loss")
    exit_reason = md.get("exit_reason")

    if stop_loss is None:
        log(f"  SKIP {trade_id} no stop_loss in metadata")
        return None

    risk_unit = abs(entry_price - float(stop_loss))
    if risk_unit <= 0:
        log(f"  SKIP {trade_id} zero risk_unit (entry==stop)")
        return None

    delta_sym = to_delta_symbol(symbol_raw)

    # Fetch a window covering all checkpoints with margin. Reuse per-symbol-window cache.
    span_start = closed_ts - 60
    span_end = closed_ts + (max(CHECKPOINTS_MIN) + 5) * 60
    cache_key = (delta_sym, span_start // 60, span_end // 60)
    candles = candles_cache.get(cache_key)
    if candles is None:
        candles = fetch_candles(delta_sym, span_start, span_end)
        candles_cache[cache_key] = candles
        time.sleep(INTER_CALL_SLEEP)

    now_ts = int(time.time())
    record = {
        "trade_id": trade_id,
        "symbol": symbol_raw,
        "delta_symbol": delta_sym,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": float(stop_loss),
        "risk_unit": risk_unit,
        "exit_at": trade["closed_at"].astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exit_at_ts": closed_ts,
        "exit_reason": exit_reason,
        "regime": md.get("regime"),
        "scanner": md.get("scanner"),
        "grade": md.get("grade"),
        "ml_prob": md.get("ml_prob"),
    }

    checkpoints_filled: list[int] = []
    for mins in CHECKPOINTS_MIN:
        target_ts = closed_ts + mins * 60
        key = f"+{mins}m_R"
        price_key = f"+{mins}m_price"
        ts_key = f"+{mins}m_ts"
        if now_ts < target_ts + 60:
            # Candle for this checkpoint hasn't even started/finished — leave null.
            record[key] = None
            record[price_key] = None
            record[ts_key] = None
            continue
        c_ts, c_close = first_close_at_or_after(candles, target_ts)
        if c_close is None:
            record[key] = None
            record[price_key] = None
            record[ts_key] = None
            continue
        if side == "short":
            r_val = (exit_price - c_close) / risk_unit
        else:  # long
            r_val = (c_close - exit_price) / risk_unit
        record[key] = round(r_val, 4)
        record[price_key] = c_close
        record[ts_key] = c_ts
        checkpoints_filled.append(mins)

    record["checkpoints_filled"] = checkpoints_filled
    record["all_checkpoints_filled"] = (len(checkpoints_filled) == len(CHECKPOINTS_MIN))
    return record


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_summary() -> None:
    if not LOG_FILE.exists():
        return
    rows: list[dict] = []
    for line in LOG_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    complete = [r for r in rows if r.get("all_checkpoints_filled")]
    if len(complete) < SUMMARY_MIN_TRADES:
        return

    by_reason: dict[str, dict] = {}
    for r in complete:
        reason = r.get("exit_reason") or "unknown"
        bucket = by_reason.setdefault(
            reason,
            {"n": 0, "+5m_R": 0.0, "+15m_R": 0.0, "+30m_R": 0.0, "+60m_R": 0.0,
             "pos5": 0, "pos15": 0, "pos30": 0, "pos60": 0},
        )
        bucket["n"] += 1
        for m in CHECKPOINTS_MIN:
            v = r.get(f"+{m}m_R")
            if v is None:
                continue
            bucket[f"+{m}m_R"] += v
            if v > 0:
                bucket[f"pos{m}"] += 1

    lines = [
        "# Post-Exit Price Tracker Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total complete trades: **{len(complete)}**",
        "",
        "Positive R = price kept moving favorably AFTER exit (we likely closed too early).",
        "Negative R = price stalled/reversed after exit (the guard was correct).",
        "",
        "| exit_reason | n | avg +5m R | avg +15m R | avg +30m R | avg +60m R | %pos@60 |",
        "|---|---|---|---|---|---|---|",
    ]
    for reason in sorted(by_reason, key=lambda k: -by_reason[k]["n"]):
        b = by_reason[reason]
        n = b["n"]
        avg5 = b["+5m_R"] / n
        avg15 = b["+15m_R"] / n
        avg30 = b["+30m_R"] / n
        avg60 = b["+60m_R"] / n
        pct60 = 100.0 * b["pos60"] / n if n else 0.0
        lines.append(
            f"| {reason} | {n} | {avg5:+.3f} | {avg15:+.3f} | {avg30:+.3f} | {avg60:+.3f} | {pct60:.1f}% |"
        )
    SUMMARY_FILE.write_text("\n".join(lines) + "\n")
    log(f"summary written -> {SUMMARY_FILE} ({len(complete)} trades)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    processed: dict = state.setdefault("processed", {})

    trades = fetch_recent_shadow_exits()
    log(f"found {len(trades)} candidate shadow trades closed in last {LOOKBACK_MINUTES}m")

    candles_cache: dict = {}
    written = 0
    skipped_done = 0

    for trade in trades:
        tid = trade["trade_id"]
        prior = processed.get(tid, {})
        if prior.get("done"):
            skipped_done += 1
            continue
        record = compute_record(trade, candles_cache)
        if record is None:
            processed[tid] = {"done": True, "skipped": True}
            continue
        append_or_replace_record(record)
        written += 1
        processed[tid] = {
            "done": record["all_checkpoints_filled"],
            "checkpoints_filled": record["checkpoints_filled"],
            "exit_at_ts": record["exit_at_ts"],
        }
        log(
            f"  {tid[:8]} {record['symbol']} {record['side']} reason={record['exit_reason']} "
            f"+5={record['+5m_R']} +15={record['+15m_R']} +30={record['+30m_R']} +60={record['+60m_R']} "
            f"done={record['all_checkpoints_filled']}"
        )

    save_state(state)
    log(f"wrote {written} records, skipped {skipped_done} already-done")
    write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())

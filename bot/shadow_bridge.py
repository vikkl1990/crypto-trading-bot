"""PATCH_L_5_22 — Shadow Signal Bridge (publisher side).

Used by W/F-validated paper engines to enqueue shadow execution requests.
Main bot's shadow_signal_consumer.py polls the queue file every 5 sec.

File-queue design (v1):
  - Atomic append to storage/shadow_signal_queue.jsonl (O_APPEND on POSIX)
  - Each line is one signal request with TTL + idempotency key
  - Consumer tracks offset in storage/shadow_signal_consumed.txt

Future migration (Phase R2 per refactor doc): replace file with Postgres
table or in-process pub-sub when SignalBus refactor lands.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

QUEUE_PATH = Path("/home/opc/crypto-trading-bot/storage/shadow_signal_queue.jsonl")
SIGNAL_TTL_SEC = 60   # consumer ignores signals older than this


def publish_signal(
    source_engine: str,
    symbol: str,
    side: str,                    # "long" or "short"
    entry_price: float,
    stop_loss: float,
    take_profit: Optional[float] = None,
    ml_probability: float = 0.5,
    grade: str = "B",
    setup_type: str = "smc",
    confidence: float = 60.0,
    regime: str = "unknown",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Append a signal-request line to the shadow queue.

    Returns the signal_id assigned (UUID-derived). Engines should pass
    this back into their own bookkeeping so we can correlate paper-engine
    trades with shadow trades downstream.
    """
    if not source_engine or not symbol or not side:
        return ""
    side = side.lower()
    if side not in ("long", "short"):
        return ""

    sig_id = uuid.uuid4().hex[:16]
    record = {
        "sig_id": sig_id,
        "ts_unix": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "ttl_sec": SIGNAL_TTL_SEC,
        "source_engine": source_engine,
        "symbol": symbol,
        "side": side,
        "entry_price": float(entry_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit) if take_profit is not None else None,
        "ml_probability": float(ml_probability),
        "grade": str(grade),
        "setup_type": str(setup_type),
        "confidence": float(confidence),
        "regime": str(regime),
        "extra_meta": extra_meta or {},
    }

    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":")) + "\n"
    # POSIX guarantees O_APPEND writes are atomic for buffers < PIPE_BUF.
    # Our records are << 4KB so single-write atomicity holds.
    fd = os.open(str(QUEUE_PATH), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    # PATCH_R_5_22 (2026-05-03) — record signal in ensemble overlay for
    # cross-scanner ensemble detection. Fail-open if module unavailable.
    try:
        from bot.ensemble_overlay import get_overlay
        get_overlay().record_signal(source=source_engine, symbol=symbol, side=side)
    except Exception:
        pass
    # PARITY_WIREUP_5_22 (2026-05-03) — emit parity-audit row using bridge
    # sig_id as the unified signal_id (so bridge → consume → execute → close
    # all reference the same row).
    try:
        from bot.parity_audit import get_audit
        from datetime import datetime, timezone as _tz
        get_audit().emit_paper_signal(
            signal_id=sig_id,
            scanner=source_engine, symbol=symbol, side=side,
            signal_time_utc=datetime.now(_tz.utc),
            signal_price=float(entry_price),
            paper_entry_price=float(entry_price),
            paper_tp=float(take_profit) if take_profit is not None else None,
            paper_sl=float(stop_loss),
            meta={
                "grade": grade, "ml_probability": float(ml_probability),
                "regime": regime, "confidence": float(confidence),
                "setup_type": setup_type,
                **(extra_meta or {}),
            },
        )
    except Exception:
        pass
    return sig_id

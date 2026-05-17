"""PATCH_L_5_22 — Shadow Signal Consumer (subscriber side).

Polls storage/shadow_signal_queue.jsonl every 5 sec. For each new signal:
  1. TTL check (drop if older than ttl_sec, default 60)
  2. Idempotency check (skip if sig_id already in consumed offset)
  3. Per-engine circuit breaker check (mode A/B/C per engine)
  4. Construct a Signal-shape dict + meta
  5. Call user_real_manager._execute_shadow() for each opted-in user
  6. Mark sig_id as consumed

Per-engine breaker modes:
  A — inherit structure_bounce circuit breaker (umbrella)
  B — own per-engine WR breaker (independent)
  C — bypass breaker entirely (W/F trust)

Defaults are tunable in BRIDGE_BREAKER_MODE below or via env vars
BRIDGE_BREAKER_<engine_upper>=A|B|C
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

QUEUE_PATH = Path("/home/opc/crypto-trading-bot/storage/shadow_signal_queue.jsonl")
OFFSET_PATH = Path("/home/opc/crypto-trading-bot/storage/shadow_signal_consumed.txt")
POLL_INTERVAL_SEC = 5
MAX_LINES_PER_TICK = 50  # safety cap

# Opted-in users for v1
BRIDGED_USER_EMAILS: Set[str] = {"admin@vnedge.com"}

# Per-engine circuit breaker mode. Override via env var BRIDGE_BREAKER_<engine_upper>.
# A = inherit structure_bounce breaker
# B = own per-engine breaker (key = f"engine_{source}")
# C = bypass breaker entirely
BRIDGE_BREAKER_MODE: Dict[str, str] = {
    "liq_grab_ob_fvg":  "B",
    "liq_sweep_htf":    "B",
    "smc1":             "B",
    "smc15":            "B",
    "smc15v2":          "B",
    "scalper_vwap_mr":  "B",
    "s5":               "C",   # 4h cadence — breaker meaningless at low n
}

# Global emergency kill switch
def _global_kill() -> bool:
    return os.environ.get("BRIDGE_GLOBAL_KILL", "").lower() == "true"


def _resolve_mode(source_engine: str) -> str:
    env = os.environ.get(f"BRIDGE_BREAKER_{source_engine.upper()}", "").strip().upper()
    if env in ("A", "B", "C"):
        return env
    return BRIDGE_BREAKER_MODE.get(source_engine, "B")


def _load_consumed() -> Set[str]:
    if not OFFSET_PATH.exists():
        return set()
    try:
        with open(OFFSET_PATH) as fh:
            return {line.strip() for line in fh if line.strip()}
    except Exception:
        return set()


def _record_consumed(sig_id: str) -> None:
    try:
        with open(OFFSET_PATH, "a") as fh:
            fh.write(sig_id + "\n")
    except Exception as e:
        logger.warning("BRIDGE_CONSUME failed to record sig_id %s: %s", sig_id, e)


def _check_breaker(source_engine: str) -> Optional[str]:
    """Return veto reason if blocked, None to allow."""
    if _global_kill():
        return f"BRIDGE_GLOBAL_KILL active for {source_engine}"
    mode = _resolve_mode(source_engine)
    if mode == "C":
        return None  # bypass
    try:
        from bot.circuit_breaker import get_tracker
        tracker = get_tracker()
        if mode == "A":
            return tracker.is_blocked("structure_bounce")
        elif mode == "B":
            return tracker.is_blocked(f"engine_{source_engine}")
    except Exception as e:
        logger.warning("BRIDGE breaker check failed (fail-open): %s", e)
        return None
    return None


async def _execute_for_user(user_real_mgr, signal_record: Dict[str, Any]) -> bool:
    """Construct minimal Signal+meta and call _execute_shadow on the
    given user_real_mgr instance. Returns True on success.

    The signature of _execute_shadow is:
      _execute_shadow(self, signal, symbol, side_str, order_side,
                      entry_price, sl, tp, margin, leverage, lots,
                      product_id, tick_size, trade_contract_size, meta,
                      exit_config=None)
    """
    sym = signal_record["symbol"]
    side = signal_record["side"]
    entry = float(signal_record["entry_price"])
    sl = float(signal_record["stop_loss"])
    tp = signal_record.get("take_profit")
    if tp is not None:
        tp = float(tp)

    # Order side for ccxt-style: long=buy, short=sell
    order_side = "buy" if side == "long" else "sell"

    # Resolve Delta product spec (PRODUCT_MAP is module-level in delta_client)
    try:
        from exchange.delta_client import PRODUCT_MAP
        info = PRODUCT_MAP.get(sym) or {}
    except Exception:
        info = {}
    is_demo_user = bool(getattr(user_real_mgr, "_is_demo", False))
    # PRODUCT_MAP uses keys: demo_id, prod_id, contract_size, tick_size, tick_size_demo
    pid_key = "demo_id" if is_demo_user else "prod_id"
    tick_key = "tick_size_demo" if is_demo_user else "tick_size"
    product_id = info.get(pid_key) or info.get("prod_id") or info.get("demo_id") or 0
    if not product_id:
        logger.warning(
            "BRIDGE_CONSUME no product_id for %s (info=%s), skipping sig=%s",
            sym, list(info.keys()), signal_record.get("sig_id"),
        )
        return False
    contract_size = float(info.get("contract_size") or 1.0)
    tick_size = float(info.get(tick_key) or info.get("tick_size") or 0.01)

    # Compute lots/margin/leverage — minimal sensible defaults.
    # We use $400 notional (matches existing W/F studies) at 20x leverage.
    notional = 400.0
    leverage = 20
    margin = notional / leverage
    # Lots = notional / (price × contract_size)  [matches user_real_manager.py]
    lots = max(1, int(notional / max(entry * contract_size, 0.0001)))

    meta = {
        "source_engine": signal_record["source_engine"],
        "setup_type": signal_record.get("setup_type", "smc"),
        "scanner": signal_record["source_engine"],   # reuse scanner column
        "ml_probability": signal_record.get("ml_probability", 0.5),
        "grade": signal_record.get("grade", "B"),
        "regime": signal_record.get("regime", "unknown"),
        "confidence": signal_record.get("confidence", 60.0),
        "bridge_sig_id": signal_record.get("sig_id"),
        "bridge_publish_ts": signal_record.get("ts_iso"),
        **(signal_record.get("extra_meta") or {}),
    }

    # Build a minimal signal object — _execute_shadow uses signal mostly
    # for record-keeping. Use a SimpleNamespace shim.
    from types import SimpleNamespace
    signal = SimpleNamespace(
        symbol=sym, side=SimpleNamespace(value=side),
        entry_price=entry, stop_loss=sl, take_profit=tp,
        scanner_name=signal_record["source_engine"],
        confidence=meta["confidence"],
        grade=meta["grade"],
        ml_probability=meta["ml_probability"],
        metadata=meta,
        get=lambda k, d=None: meta.get(k, d),
    )

    try:
        result = await user_real_mgr._execute_shadow(
            signal=signal,
            symbol=sym,
            side_str=side,
            order_side=order_side,
            entry_price=entry,
            sl=sl,
            tp=tp,
            margin=margin,
            leverage=leverage,
            lots=lots,
            product_id=int(product_id),
            tick_size=tick_size,
            trade_contract_size=contract_size,
            meta=meta,
        )
        return result is not None
    except Exception as e:
        logger.warning(
            "BRIDGE_CONSUME _execute_shadow failed for %s sig=%s: %s",
            user_real_mgr.user_email if hasattr(user_real_mgr, "user_email") else "?",
            signal_record.get("sig_id"), e,
        )
        return False


async def consumer_loop(get_registry_fn) -> None:
    """Main loop. `get_registry_fn()` returns the live UserRealRegistry."""
    consumed = _load_consumed()
    last_offset = 0
    logger.warning("BRIDGE_CONSUMER started, %d already-consumed sig_ids loaded",
                   len(consumed))

    while True:
        try:
            if not QUEUE_PATH.exists():
                await asyncio.sleep(POLL_INTERVAL_SEC)
                continue

            with open(QUEUE_PATH) as fh:
                fh.seek(last_offset)
                new_lines = []
                for _ in range(MAX_LINES_PER_TICK):
                    line = fh.readline()
                    if not line:
                        break
                    new_lines.append(line)
                last_offset = fh.tell()

            if not new_lines:
                await asyncio.sleep(POLL_INTERVAL_SEC)
                continue

            now = time.time()

            # Ensure opted-in user managers exist (lazy create on first tick
            # with pending signals). Without this, signals queue up but
            # silently skip because manager hasn't been built yet.
            registry = get_registry_fn()
            user_mgrs: Dict[str, Any] = {}
            if registry is not None:
                try:
                    cache = getattr(registry, "_active_users_cache", None) or []
                    if not cache:
                        # Cache not yet populated — force refresh
                        try:
                            await registry._refresh_active_users()
                            cache = registry._active_users_cache or []
                            logger.warning(
                                "BRIDGE_CONSUME forced active_users_cache refresh, %d users loaded",
                                len(cache))
                        except Exception as e:
                            logger.warning(
                                "BRIDGE_CONSUME _refresh_active_users failed: %s", e)
                    for user_info in cache:
                        em = user_info.get("email", "")
                        if em in BRIDGED_USER_EMAILS:
                            uid = user_info.get("id")
                            if uid and uid not in registry._managers:
                                logger.warning(
                                    "BRIDGE_CONSUME ensuring manager for %s", em)
                                await registry.get_or_create_manager(user_info)
                    user_mgrs = {
                        getattr(mgr, "user_email", "") or "?": mgr
                        for mgr in registry._managers.values()
                        if hasattr(mgr, "user_email")
                    }
                except Exception as e:
                    logger.warning("BRIDGE_CONSUME registry access failed: %s", e)

            stats = {"received": 0, "stale": 0, "duplicate": 0, "blocked": 0,
                     "executed": 0, "failed": 0}

            for raw in new_lines:
                stats["received"] += 1
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue

                sig_id = rec.get("sig_id", "")
                if not sig_id:
                    continue

                if sig_id in consumed:
                    stats["duplicate"] += 1
                    continue

                ts = float(rec.get("ts_unix", 0))
                ttl = float(rec.get("ttl_sec", 60))
                if ts and (now - ts) > ttl:
                    stats["stale"] += 1
                    consumed.add(sig_id)
                    _record_consumed(sig_id)
                    continue

                src = rec.get("source_engine", "")
                veto = _check_breaker(src)
                if veto:
                    logger.info("BRIDGE_CONSUME BLOCKED sig=%s src=%s — %s",
                                sig_id, src, veto)
                    stats["blocked"] += 1
                    consumed.add(sig_id)
                    _record_consumed(sig_id)
                    continue

                # PARITY_WIREUP_5_22 — record bridge consume timestamp
                try:
                    from bot.parity_audit import get_audit
                    get_audit().update_bridge_consume(sig_id)
                except Exception:
                    pass

                # Execute for each opted-in user
                any_success = False
                for email in BRIDGED_USER_EMAILS:
                    user_mgr = user_mgrs.get(email)
                    if user_mgr is None:
                        logger.warning(
                            "BRIDGE_CONSUME no manager for %s, sig=%s skipped",
                            email, sig_id)
                        continue
                    ok = await _execute_for_user(user_mgr, rec)
                    if ok:
                        any_success = True
                        logger.warning(
                            "BRIDGE_CONSUME EXECUTED sig=%s src=%s sym=%s side=%s user=%s",
                            sig_id, src, rec.get("symbol"), rec.get("side"),
                            email,
                        )
                if any_success:
                    stats["executed"] += 1
                else:
                    stats["failed"] += 1

                consumed.add(sig_id)
                _record_consumed(sig_id)

            if stats["received"] > 0:
                logger.warning(
                    "BRIDGE_CONSUME tick received=%d executed=%d blocked=%d "
                    "stale=%d duplicate=%d failed=%d",
                    stats["received"], stats["executed"], stats["blocked"],
                    stats["stale"], stats["duplicate"], stats["failed"],
                )

        except Exception as e:
            logger.exception("BRIDGE_CONSUMER loop error (will retry): %s", e)
        await asyncio.sleep(POLL_INTERVAL_SEC)

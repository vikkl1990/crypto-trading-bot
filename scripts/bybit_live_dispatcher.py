#!/usr/bin/env python3
"""
Bybit Live Dispatcher — long-running daemon that places real Bybit orders
for users with `users.bybit_mode='live'`.

Runs as a systemd unit (NOT cron, because we maintain persistent WS connections
per user). Polls user_trades every POLL_SEC seconds for new Delta shadow signals
that need a Bybit live mirror, then dispatches via UserBybitExecutor.

Architecture:
  - One UserBybitExecutor per user (each has own WS + keys)
  - Lazy-loaded on first dispatch
  - Persistent connection (re-used across signals)
  - Idempotent: skips trades already mirrored to bybit_real

Why a separate daemon (not in bot main loop):
  - Safer: bugs here don't crash the strategy/Delta path
  - Independent restart cycle
  - Can be disabled instantly via systemctl stop without touching the bot

Run:
  sudo systemctl start bybit-live-dispatcher
  journalctl -u bybit-live-dispatcher -f
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Optional

import asyncpg

logger = logging.getLogger("bybit_live_dispatcher")

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge",
)
POLL_SEC = 5            # how often to scan for new signals
LOOKBACK_SEC = 60       # only consider signals opened in last N seconds


async def _list_live_bybit_users(pool) -> list:
    """Return list of {user_id, email} for users with bybit_mode='live' AND active Bybit live key."""
    async with pool.acquire() as con:
        rows = await con.fetch("""
            SELECT u.id, u.email
              FROM users u
              JOIN user_api_keys k ON k.user_id = u.id
             WHERE u.is_active = TRUE
               AND u.bybit_mode = 'live'
               AND k.exchange = 'bybit'
               AND k.label = 'live'
               AND k.is_active = TRUE
        """)
    return [{"user_id": str(r["id"]), "email": r["email"]} for r in rows]


async def _fetch_unmirrored_signals(pool, user_id: str, lookback_sec: int) -> list:
    """Find delta_india shadow trades opened in lookback window that haven't been
    mirrored to bybit_real for this user yet.
    """
    async with pool.acquire() as con:
        rows = await con.fetch(f"""
            SELECT d.id, d.symbol, d.side, d.entry_price, d.quantity,
                   d.opened_at, d.signal_data, d.metadata
              FROM user_trades d
             WHERE d.user_id = $1
               AND d.exchange = 'delta_india'
               AND d.trade_type = 'shadow'
               AND d.opened_at >= NOW() - INTERVAL '{int(lookback_sec)} seconds'
               AND NOT EXISTS (
                   SELECT 1 FROM user_trades b
                    WHERE b.user_id = $1
                      AND b.exchange = 'bybit'
                      AND b.trade_type = 'real'
                      AND b.symbol = d.symbol
                      AND b.opened_at >= d.opened_at - INTERVAL '5 seconds'
                      AND b.opened_at <= d.opened_at + INTERVAL '60 seconds'
               )
             ORDER BY d.opened_at DESC LIMIT 20
        """, user_id)
    return [dict(r) for r in rows]


async def main_loop():
    sys.path.insert(0, "/home/opc/crypto-trading-bot")
    from execution.user_bybit_executor import UserBybitExecutor

    logger.info("BYBIT_LIVE_DISPATCHER: starting (poll=%ss, lookback=%ss)",
                POLL_SEC, LOOKBACK_SEC)
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=4)
    executors: Dict[str, UserBybitExecutor] = {}
    try:
        while True:
            try:
                users = await _list_live_bybit_users(pool)
                if not users:
                    await asyncio.sleep(POLL_SEC)
                    continue
                # Process each user
                for u in users:
                    uid = u["user_id"]
                    email = u["email"]
                    if uid not in executors:
                        executors[uid] = UserBybitExecutor(uid, email, pool)
                        logger.info("BYBIT_LIVE_DISPATCHER: tracking %s", email)
                    sigs = await _fetch_unmirrored_signals(pool, uid, LOOKBACK_SEC)
                    for sig in sigs:
                        signal_payload = {
                            "symbol": sig["symbol"],
                            "side": sig["side"],
                            # qty already in Delta-contract units; executor converts
                            "position_size": sig["quantity"],
                        }
                        try:
                            res = await executors[uid].dispatch(signal_payload)
                            if res.get("ok"):
                                logger.warning(
                                    "BYBIT_LIVE: %s %s %s coid=%s ✓",
                                    email[:12], sig["symbol"], sig["side"],
                                    res.get("client_order_id"),
                                )
                            else:
                                logger.warning(
                                    "BYBIT_LIVE: %s %s %s FAIL: %s",
                                    email[:12], sig["symbol"], sig["side"],
                                    res.get("error"),
                                )
                        except Exception as e:
                            logger.error("BYBIT_LIVE: dispatch error: %s", e, exc_info=True)
            except Exception as e:
                logger.error("BYBIT_LIVE_DISPATCHER tick error: %s", e, exc_info=True)
            await asyncio.sleep(POLL_SEC)
    finally:
        for ex in executors.values():
            try:
                if ex._ws_client:
                    await ex._ws_client.stop()
            except Exception:
                pass
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main_loop())

"""
Phase 5.20-B1 — Global kill switch.

Singleton checked from every entry path. Backed by `bot_state` table with
in-memory 5-second cache to keep the hot path fast.

API:
    from execution.kill_switch import is_killed, get_kill_state, engage, release

    if await is_killed(db_pool):
        return False, "kill_switch_engaged"

Engagement methods:
    1. SQL:  UPDATE bot_state SET kill_switch_engaged=TRUE WHERE id=1;
    2. CLI:  python3 -m execution.kill_switch engage --reason "ops halt"
    3. Admin REST endpoint (future)

Effect when engaged:
    - All new entries blocked at admission (every user)
    - If kill_switch_close_open=TRUE: monitor closes all open positions
      at next tick (best-effort market orders)
    - Bot remains responsive — does NOT crash, just refuses to trade
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import asyncpg

logger = logging.getLogger("kill_switch")


@dataclass
class KillState:
    engaged: bool = False
    reason: str = ""
    engaged_at: Optional[str] = None
    engaged_by: str = ""
    close_open: bool = False


_CACHE: Optional[KillState] = None
_CACHE_TS: float = 0.0
_CACHE_TTL_SEC: float = 5.0


async def get_kill_state(db_pool) -> KillState:
    """Cached check — stale up to 5 seconds."""
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE is not None and (now - _CACHE_TS) < _CACHE_TTL_SEC:
        return _CACHE
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT kill_switch_engaged, kill_switch_reason, "
                "kill_switch_engaged_at, kill_switch_engaged_by, "
                "kill_switch_close_open FROM bot_state WHERE id = 1"
            )
        if row is None:
            _CACHE = KillState()
        else:
            _CACHE = KillState(
                engaged=bool(row["kill_switch_engaged"]),
                reason=row["kill_switch_reason"] or "",
                engaged_at=row["kill_switch_engaged_at"].isoformat() if row["kill_switch_engaged_at"] else None,
                engaged_by=row["kill_switch_engaged_by"] or "",
                close_open=bool(row["kill_switch_close_open"]),
            )
        _CACHE_TS = now
        if _CACHE.engaged:
            logger.warning(
                "KILL_SWITCH ENGAGED: reason=%s by=%s at=%s close_open=%s",
                _CACHE.reason, _CACHE.engaged_by, _CACHE.engaged_at, _CACHE.close_open,
            )
        return _CACHE
    except Exception as e:
        logger.warning("Kill switch check failed: %s — failing OPEN (allow trading)", e)
        return KillState()  # fail-open: don't block trading on DB hiccup


async def is_killed(db_pool) -> bool:
    """Convenience: returns True if engaged."""
    state = await get_kill_state(db_pool)
    return state.engaged


def invalidate_cache():
    """Force next call to re-fetch from DB (use when you just engaged)."""
    global _CACHE_TS
    _CACHE_TS = 0.0


# ------------------------------------------------------------------
# CLI engagement (local ops convenience)
# ------------------------------------------------------------------

def _load_env(path: str = ".env") -> None:
    try:
        with open(path) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v.strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _dsn() -> str:
    _load_env()
    if os.environ.get("DATABASE_URL", "").startswith("postgres"):
        return os.environ["DATABASE_URL"]
    return "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"


async def engage(reason: str, by: str = "cli", close_open: bool = False) -> None:
    conn = await asyncpg.connect(_dsn())
    try:
        await conn.execute("""
            UPDATE bot_state
            SET kill_switch_engaged = TRUE,
                kill_switch_reason = $1,
                kill_switch_engaged_at = NOW(),
                kill_switch_engaged_by = $2,
                kill_switch_close_open = $3,
                updated_at = NOW()
            WHERE id = 1
        """, reason, by, close_open)
        print(f"🛑 KILL SWITCH ENGAGED: reason='{reason}' by='{by}' close_open={close_open}")
        print("   Bot will stop accepting new entries within ~5 seconds.")
        if close_open:
            print("   Open positions will be emergency-closed at next monitor tick.")
    finally:
        await conn.close()


async def release(by: str = "cli") -> None:
    conn = await asyncpg.connect(_dsn())
    try:
        await conn.execute("""
            UPDATE bot_state
            SET kill_switch_engaged = FALSE,
                kill_switch_reason = '',
                kill_switch_close_open = FALSE,
                updated_at = NOW()
            WHERE id = 1
        """)
        print(f"✅ KILL SWITCH RELEASED by='{by}' — bot resuming trading on next signal.")
    finally:
        await conn.close()


async def status() -> None:
    conn = await asyncpg.connect(_dsn())
    try:
        row = await conn.fetchrow(
            "SELECT * FROM bot_state WHERE id = 1"
        )
    finally:
        await conn.close()
    if not row:
        print("(no bot_state row)")
        return
    if row["kill_switch_engaged"]:
        print(f"🛑 ENGAGED")
        print(f"   reason: {row['kill_switch_reason']}")
        print(f"   by: {row['kill_switch_engaged_by']}")
        print(f"   at: {row['kill_switch_engaged_at']}")
        print(f"   close_open: {row['kill_switch_close_open']}")
    else:
        print("🟢 RELEASED — trading enabled")


async def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engage")
    e.add_argument("--reason", required=True)
    e.add_argument("--by", default="cli")
    e.add_argument("--close-open", action="store_true",
                   help="Also emergency-close all open positions")
    sub.add_parser("release").add_argument("--by", default="cli")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "engage":
        await engage(args.reason, args.by, args.close_open)
    elif args.cmd == "release":
        await release(args.by)
    elif args.cmd == "status":
        await status()


if __name__ == "__main__":
    asyncio.run(main())

"""
Shadow-mode flag helper — checks if all active users are in shadow_live.
Used by signal_tracker to relax weak-setup filtering during shadow
validation windows (Silent Failure #14 fix, 2026-04-26).

Cached for 30 seconds to avoid hammering the DB.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import psycopg2

_CACHE: Optional[bool] = None
_CACHE_TS: float = 0.0
_CACHE_TTL = 30.0

_DB = dict(host="localhost", user="vnedge", password="VnEdge2026db", dbname="vnedge")


def is_all_users_shadow() -> bool:
    """True if EVERY active user has bot_mode='shadow_live'.
    False if any active user is on paper / demo / live, or on DB error.
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE is not None and (now - _CACHE_TS) < _CACHE_TTL:
        return _CACHE
    try:
        con = psycopg2.connect(**_DB)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT bot_mode, COUNT(*) FROM users "
                "WHERE is_active=true GROUP BY bot_mode"
            )
            modes = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
            # All users in shadow_live → True
            non_shadow = sum(c for m, c in modes.items() if m != "shadow_live")
            shadow = modes.get("shadow_live", 0)
            _CACHE = (shadow > 0 and non_shadow == 0)
            _CACHE_TS = now
            return _CACHE
        finally:
            con.close()
    except Exception:
        _CACHE = False
        _CACHE_TS = now
        return False


def invalidate_cache() -> None:
    """Force re-check on next call. Use after bot_mode changes."""
    global _CACHE_TS
    _CACHE_TS = 0.0

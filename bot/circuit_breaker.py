"""PATCH_J_5_22 — Realized-WR circuit breaker for shadow_live.

Maintains in-memory rolling window of last N closed trades per scanner.
If WR over the window drops below threshold, block new entries on that
scanner until the pause window expires.

Intended to be the LAST safety net when an established edge is collapsing
(e.g. structure_bounce in chop regime). Sizing patches (Patch H) reduce
loss severity but not loss frequency. Circuit breaker stops the bleed.

Usage:
    from bot.circuit_breaker import get_tracker

    tracker = get_tracker()
    veto = tracker.is_blocked("structure_bounce")
    if veto is not None:
        # add to vetos list, block trade

    # When a trade closes:
    tracker.record_close("structure_bounce", pnl_usd=-0.40)

Thread-safety: uses a single mutex around deque mutations. Read paths
(is_blocked) are deque slice + a divide; effectively atomic on CPython.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

# ─────────────────────────────────────────────────────────────────────
# Tunables (read once at import; safe to override before first call)
# ─────────────────────────────────────────────────────────────────────
WINDOW_N: int = 20            # last N trades considered for WR
WR_FLOOR: float = 0.40        # WR below this = circuit-break
PAUSE_SEC: int = 3600         # 60-minute pause once tripped
MIN_TRADES_FOR_DECISION: int = 20  # need full window before any blocking


class RealizedWRTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._windows: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=WINDOW_N)
        )
        self._pause_until: Dict[str, float] = {}  # scanner -> unix ts
        self._trip_count: Dict[str, int] = defaultdict(int)
        self._seeded: Dict[str, bool] = {}  # scanner -> True once lazy-seed attempted

    def _lazy_seed(self, scanner: str) -> None:
        """One-shot DB query to populate the rolling window from history.
        Called on first is_blocked() per scanner. Marks scanner as seeded
        regardless of outcome so we don't retry every signal."""
        self._seeded[scanner] = True
        try:
            import os
            import psycopg2
        except ImportError:
            return
        dsn = (
            f"host=localhost dbname=vnedge user=vnedge "
            f"password={os.environ.get('DB_PASSWORD', 'VnEdge2026db')}"
        )
        conn = psycopg2.connect(dsn, connect_timeout=3)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pnl_usd
                      FROM user_trades
                     WHERE trade_type IN ('shadow','real')
                       AND closed_at IS NOT NULL
                       AND (metadata::jsonb->>'setup_type' = %s
                            OR metadata::jsonb->>'scanner' = %s)
                     ORDER BY closed_at DESC
                     LIMIT %s
                    """,
                    (scanner, scanner, WINDOW_N),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        if rows:
            # rows is newest-first; deque appends become left-old, right-new
            self.seed_from_history(scanner, [r[0] for r in reversed(rows)])

    # ── Trade recording ──────────────────────────────────────────────
    def record_close(self, scanner: str, pnl_usd: float) -> None:
        if not scanner:
            return
        with self._lock:
            self._windows[scanner].append(float(pnl_usd or 0.0))

    def seed_from_history(self, scanner: str, pnl_list) -> None:
        """Populate the window from a DB query at bot startup."""
        if not scanner or not pnl_list:
            return
        with self._lock:
            self._windows[scanner].clear()
            for p in pnl_list[-WINDOW_N:]:
                self._windows[scanner].append(float(p or 0.0))

    # ── Decision ─────────────────────────────────────────────────────
    def is_blocked(self, scanner: str) -> Optional[str]:
        """Return veto reason string if entries should be blocked, else None.

        Veto string starts with 'CIRCUIT_BREAKER_J:' so it can be added to
        sb_hard_prefixes in scalp_strategy.py for hard kill semantics.
        """
        if not scanner:
            return None
        now = time.time()

        # Active pause?
        unpause_at = self._pause_until.get(scanner, 0.0)
        if unpause_at > now:
            mins_left = int((unpause_at - now) / 60)
            return (f"CIRCUIT_BREAKER_J: scanner={scanner} paused for "
                    f"{mins_left}m more (trip #{self._trip_count[scanner]})")

        # Lazy seed from DB on first call per scanner (avoids cold start delay)
        with self._lock:
            window = list(self._windows[scanner])
        if len(window) < MIN_TRADES_FOR_DECISION and not self._seeded.get(scanner, False):
            try:
                self._lazy_seed(scanner)
                with self._lock:
                    window = list(self._windows[scanner])
            except Exception:
                pass  # fail open — tracker warms up via record_close() instead

        if len(window) < MIN_TRADES_FOR_DECISION:
            return None

        # WR check
        wins = sum(1 for p in window if p > 0)
        wr = wins / len(window)
        if wr < WR_FLOOR:
            # TRIP — set pause
            self._pause_until[scanner] = now + PAUSE_SEC
            self._trip_count[scanner] += 1
            return (f"CIRCUIT_BREAKER_J: scanner={scanner} tripped "
                    f"WR_{len(window)}={wr:.0%} < {WR_FLOOR:.0%}, "
                    f"pausing {PAUSE_SEC//60}m (trip #{self._trip_count[scanner]})")
        return None

    # ── Diagnostics ──────────────────────────────────────────────────
    def stats(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        now = time.time()
        for scanner in list(self._windows.keys()):
            with self._lock:
                window = list(self._windows[scanner])
            n = len(window)
            wins = sum(1 for p in window if p > 0)
            wr = wins / n if n else 0.0
            unpause_at = self._pause_until.get(scanner, 0.0)
            out[scanner] = {
                "n": n,
                "wr": round(wr, 3),
                "wins": wins,
                "paused": unpause_at > now,
                "pause_remaining_sec": max(0, int(unpause_at - now)),
                "trip_count": self._trip_count[scanner],
            }
        return out


# ─────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────
_TRACKER: Optional[RealizedWRTracker] = None


def get_tracker() -> RealizedWRTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = RealizedWRTracker()
    return _TRACKER

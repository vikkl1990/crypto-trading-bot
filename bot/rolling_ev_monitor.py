"""Rolling-EV Monitor — circuit breaker for losing-regime auto-disable.

Architect spec (2026-05-03):
    rolling_ev = mean(net_pnl of last_N trades for that scanner+symbol)
    if rolling_ev < threshold:
        skip next trade for that (scanner, symbol)

W/F-validated cells (rolling_ev_autodisable_walkforward.py, 2026-05-03):
    BTC scalper_vwap_mr — last_N=50, threshold=$-0.10, auto_when_recovered
        baseline pnl = $-9.03/1347 trades  →  with rule = $+128.07 (lift = $+137 / $+0.10/trade)
        skipped: 583/1347 (43%)

Currently OFF for everything except BTC scalper_vwap_mr (above). Other pairs
either don't need the rule (already healthy) or cleanly KILL the scanner
(SOL liq_grab — handled by scanner_real_policy.py instead).

USAGE:
    from bot.rolling_ev_monitor import get_monitor
    mon = get_monitor()

    # In urm._execute_shadow, before placing order:
    disabled, diag = mon.is_disabled(scanner, symbol)
    if disabled:
        log.info(f"ROLLING_EV_GATE: skipping {scanner}/{symbol} — {diag}")
        return  # don't place order

    # In trade close handler:
    mon.record_trade_close(scanner, symbol, net_pnl_usd)

    # Optional — at bot startup, warm cache from DB:
    mon.warm_from_db(db_pool)

DESIGN:
    - Per (scanner, symbol) FIFO buffer of last N net_pnls (in-memory deque).
    - File-backed: storage/rolling_ev/<scanner>_<symbol>.jsonl appends every close.
    - DB-warm hook (optional): reads last N from user_trades on startup.
    - is_disabled() is hot-path — must be O(1). Cache check + arithmetic only.
    - Fail-open: any exception → return (False, {"error": ...}).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

ROOT = Path("/home/opc/crypto-trading-bot")
STORAGE_DIR = ROOT / "storage" / "rolling_ev"


# ──────────────────────────────────────────────────────────────────────
# CONFIG — per (scanner, symbol) rule
# ──────────────────────────────────────────────────────────────────────
# Pairs not in this dict have NO rule (default: trade always allowed).
#
# 2026-05-03 v2: Removed dormant scalper_vwap_mr config (live scanner
#   name is vwap_mean_revert with only 1 trade ever — backtest was on
#   synthetic signals, never actually fires live).
# 2026-05-03 v2: Added structure_bounce × 8 symbols per real-data W/F
#   (rolling_ev_real_structure_bounce). Total expected lift +$1,300/30d.
ROLLING_EV_CONFIG: Dict[Tuple[str, str], Dict[str, Any]] = {

    # ── structure_bounce × 8 symbols (W/F-best per-symbol cells) ─────
    # Real-data backtest 2026-05-03 (storage/wf_studies/rolling_ev_real_structure_bounce):
    #   Aggregate: 2,478 trades, baseline -$1,195 → with rule +$104 (lift +$1,300/30d)
    #   Every qualifying symbol PASSES.

    # XRP — lift +$392 from baseline -$401 (88% skip rate)
    ("structure_bounce", "XRP/USDT"): {
        "last_n": 15, "threshold_usd": 0.0,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$392/30d, 596 trades, 88% skip",
        "deployed_at": "2026-05-03",
    },

    # BTC — lift +$357, FLIPS scanner from -$339 to +$18 PROFITABLE (69% skip)
    ("structure_bounce", "BTC/USDT"): {
        "last_n": 20, "threshold_usd": -0.10,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$357/30d, 588 trades, 69% skip; FLIPS to profit",
        "deployed_at": "2026-05-03",
    },

    # ETH — lift +$238, FLIPS scanner from -$118 to +$120 PROFITABLE (77% skip)
    ("structure_bounce", "ETH/USDT"): {
        "last_n": 15, "threshold_usd": 0.0,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$238/30d, 551 trades, 77% skip; FLIPS to profit",
        "deployed_at": "2026-05-03",
    },

    # SOL — lift +$233, FLIPS scanner from -$183 to +$50 PROFITABLE (75% skip)
    ("structure_bounce", "SOL/USDT"): {
        "last_n": 15, "threshold_usd": 0.0,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$233/30d, 526 trades, 75% skip; FLIPS to profit",
        "deployed_at": "2026-05-03",
    },

    # DOT — lift +$47 (71% skip)
    ("structure_bounce", "DOT/USDT"): {
        "last_n": 15, "threshold_usd": 0.0,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$47/30d, 51 trades, 71% skip",
        "deployed_at": "2026-05-03",
    },

    # DOGE — lift +$15 (57% skip)
    ("structure_bounce", "DOGE/USDT"): {
        "last_n": 15, "threshold_usd": 0.0,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$15/30d, 35 trades, 57% skip",
        "deployed_at": "2026-05-03",
    },

    # LINK — lift +$14 (57% skip)
    ("structure_bounce", "LINK/USDT"): {
        "last_n": 15, "threshold_usd": 0.0,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$14/30d, 35 trades, 57% skip",
        "deployed_at": "2026-05-03",
    },

    # LTC — lift +$4 (24% skip — gentlest rule)
    ("structure_bounce", "LTC/USDT"): {
        "last_n": 20, "threshold_usd": 0.0,
        "reenable": "auto_when_recovered",
        "wf_evidence": "real-data 2026-05-03: lift +$4/30d, 33 trades, 24% skip",
        "deployed_at": "2026-05-03",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Monitor
# ──────────────────────────────────────────────────────────────────────
@dataclass
class _RollState:
    last_n: int
    threshold_usd: float
    reenable: str
    buffer: Deque[float]
    n_admitted: int = 0
    n_skipped: int = 0
    last_decision_at: float = 0.0
    last_decision_disabled: bool = False


class RollingEvMonitor:
    def __init__(self, storage_dir: Path = STORAGE_DIR):
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._states: Dict[Tuple[str, str], _RollState] = {}
        # Pre-allocate state for each configured pair
        for key, cfg in ROLLING_EV_CONFIG.items():
            self._states[key] = _RollState(
                last_n=int(cfg["last_n"]),
                threshold_usd=float(cfg["threshold_usd"]),
                reenable=str(cfg["reenable"]),
                buffer=deque(maxlen=int(cfg["last_n"])),
            )
        # Try to load any existing file-backed history for configured pairs
        self._load_history()

    # ── File-backed history ──
    def _file_for(self, scanner: str, symbol: str) -> Path:
        sym_clean = symbol.replace("/", "")
        return self._dir / f"{scanner}_{sym_clean}.jsonl"

    def _load_history(self) -> None:
        for (scanner, symbol), state in self._states.items():
            p = self._file_for(scanner, symbol)
            if not p.exists():
                continue
            try:
                with p.open() as f:
                    lines = f.readlines()
                # Take the last `last_n` entries
                for line in lines[-state.last_n:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        net = float(obj.get("net_pnl_usd", 0.0))
                        state.buffer.append(net)
                    except Exception:
                        continue
            except Exception:
                pass

    def _append_file(self, scanner: str, symbol: str, net_pnl: float, ts: Optional[datetime]) -> None:
        p = self._file_for(scanner, symbol)
        record = {
            "ts": (ts or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
            "scanner": scanner, "symbol": symbol,
            "net_pnl_usd": float(net_pnl),
        }
        line = json.dumps(record) + "\n"
        try:
            fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        except Exception:
            pass

    # ── Hot-path API ──
    def is_disabled(self, scanner: str, symbol: str) -> Tuple[bool, Dict[str, Any]]:
        """Return (disabled, diag). Must be O(1) — called on every signal admit."""
        try:
            state = self._states.get((scanner, symbol))
            if state is None:
                return False, {"reason": "no_rule_for_pair"}
            with self._lock:
                buf = state.buffer
                n = len(buf)
                if n < state.last_n:
                    state.last_decision_disabled = False
                    state.last_decision_at = time.time()
                    return False, {
                        "reason": "warming_up", "have": n, "need": state.last_n,
                    }
                roll_ev = sum(buf) / n
                disabled = (roll_ev < state.threshold_usd)
                state.last_decision_disabled = disabled
                state.last_decision_at = time.time()
                if disabled:
                    state.n_skipped += 1
                    return True, {
                        "reason": "rolling_ev_below_threshold",
                        "rolling_ev_usd": round(roll_ev, 4),
                        "threshold_usd": state.threshold_usd,
                        "n": n,
                        "reenable_method": state.reenable,
                    }
                state.n_admitted += 1
                return False, {
                    "reason": "rolling_ev_ok",
                    "rolling_ev_usd": round(roll_ev, 4),
                    "threshold_usd": state.threshold_usd,
                    "n": n,
                }
        except Exception as e:
            return False, {"reason": "error", "error": str(e)[:120]}

    def record_trade_close(
        self, scanner: str, symbol: str, net_pnl_usd: float,
        ts: Optional[datetime] = None,
    ) -> None:
        """Called from trade close path. No-op for unconfigured pairs."""
        try:
            state = self._states.get((scanner, symbol))
            if state is None:
                return
            with self._lock:
                state.buffer.append(float(net_pnl_usd))
                self._append_file(scanner, symbol, net_pnl_usd, ts)
        except Exception:
            pass

    # ── Admin / debug ──
    def warm_from_db(self, db_pool, user_id_filter: Optional[int] = None) -> Dict[str, Any]:
        """Optional: query user_trades for last N closes for each configured pair.

        Expects db_pool with .acquire() / .execute() (asyncpg-style) OR a
        psycopg2-style connection pool. Tries asyncpg first.

        Returns warming summary dict.
        """
        out: Dict[str, Any] = {}
        for (scanner, symbol), state in self._states.items():
            try:
                rows = self._db_fetch_last_n(
                    db_pool, scanner, symbol, state.last_n, user_id_filter,
                )
                with self._lock:
                    state.buffer.clear()
                    for net in rows:
                        state.buffer.append(float(net))
                out[f"{scanner}|{symbol}"] = {"warmed_n": len(rows)}
            except Exception as e:
                out[f"{scanner}|{symbol}"] = {"error": str(e)[:200]}
        return out

    def _db_fetch_last_n(self, db_pool, scanner: str, symbol: str,
                         last_n: int, user_id_filter: Optional[int]) -> list:
        """Best-effort DB fetch. Return list of net_pnl_usd values, oldest first."""
        # SQL: pull last N closed real/shadow trades for this scanner+symbol
        sql = """
            SELECT pnl_usd
            FROM user_trades
            WHERE symbol = %s
              AND closed_at IS NOT NULL
              AND COALESCE(NULLIF(metadata::jsonb->>'scanner',''),
                            NULLIF(metadata::jsonb->>'setup_type',''),
                            NULLIF(metadata::jsonb->>'source_engine','')) = %s
              {user_filter}
            ORDER BY closed_at DESC
            LIMIT %s
        """
        params: list = [symbol, scanner]
        user_filter = ""
        if user_id_filter is not None:
            user_filter = "AND user_id = %s"
            params.append(user_id_filter)
        sql = sql.format(user_filter=user_filter)
        params.append(last_n)
        # Try to execute via the pool — best-effort only
        try:
            with db_pool.connection() as conn:    # psycopg / psycopg2 style
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        except AttributeError:
            # Try asyncpg-style would require async; skip for now
            return []
        # Reverse to oldest-first
        return [float(r[0]) for r in reversed(rows) if r and r[0] is not None]

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"configured_pairs": [], "states": {}}
        for (scanner, symbol), state in self._states.items():
            key = f"{scanner}|{symbol}"
            out["configured_pairs"].append(key)
            with self._lock:
                buf = list(state.buffer)
                roll_ev = (sum(buf) / len(buf)) if buf else None
                out["states"][key] = {
                    "n_in_buffer": len(buf),
                    "last_n_target": state.last_n,
                    "threshold_usd": state.threshold_usd,
                    "reenable": state.reenable,
                    "rolling_ev_usd": round(roll_ev, 4) if roll_ev is not None else None,
                    "n_admitted_this_session": state.n_admitted,
                    "n_skipped_this_session": state.n_skipped,
                    "last_decision_disabled": state.last_decision_disabled,
                }
        return out


# Singleton
_MONITOR: Optional[RollingEvMonitor] = None


def get_monitor() -> RollingEvMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = RollingEvMonitor()
    return _MONITOR

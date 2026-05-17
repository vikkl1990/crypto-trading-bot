"""PATCH_K_5_22 — ATR-percentile regime gate for shadow_live.

Computes per-symbol 4h ATR percentile rank. If current ATR is in the
bottom Nth percentile (= unusually quiet = chop regime), block entries
on chop-allergic scanners (default: structure_bounce only).

Loaded from storage/candle_cache/{SYM}_USDT_4h.parquet on demand.
Result cached per-symbol for 1 hour (4h candles only update every 4h
so 1h cache is conservative).

Usage:
    from bot.regime_gate import get_regime_gate

    gate = get_regime_gate()
    veto = gate.is_blocked(symbol="BTC/USDT", scanner="structure_bounce")
    if veto:
        # add to vetos list, block trade
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────
ATR_RANK_FLOOR: float = 0.30   # below this = chop, block scanner
ATR_LOOKBACK_BARS: int = 100   # rank against last 100 4h bars
CACHE_TTL_SEC: int = 3600      # refresh once per hour
CHOP_ALLERGIC_SCANNERS = frozenset({
    "structure_bounce",
})

CACHE_DIR = Path("/home/opc/crypto-trading-bot/storage/candle_cache")


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR via simple moving average of true range."""
    n = len(c)
    if n < period + 1:
        return np.full(n, np.nan)
    tr = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(l[1:] - c[:-1]),
    ])
    out = np.full(n, np.nan)
    out[period:] = pd.Series(tr).rolling(period).mean().values[period - 1:]
    return out


class ATRPercentileGate:
    def __init__(self):
        self._lock = threading.Lock()
        # symbol -> (timestamp, current_pct_rank)
        self._cache: Dict[str, Tuple[float, Optional[float]]] = {}

    # ── Compute ──────────────────────────────────────────────────────
    def _compute_pct_rank(self, symbol: str) -> Optional[float]:
        """Load 4h candles, compute ATR(14), return current bar's pct rank
        vs last 100 bars. Returns None if data unavailable."""
        # Symbol is e.g. "BTC/USDT" → file "BTC_USDT_4h.parquet"
        sym_clean = symbol.replace("/", "_")
        path = CACHE_DIR / f"{sym_clean}_4h.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            df = df.sort_index().tail(ATR_LOOKBACK_BARS + 20)
            h = df["high"].astype(float).values
            l = df["low"].astype(float).values
            c = df["close"].astype(float).values
            atr_series = _atr(h, l, c, 14)
            atr_window = atr_series[-ATR_LOOKBACK_BARS:]
            atr_window = atr_window[~np.isnan(atr_window)]
            if len(atr_window) < 30:
                return None
            current = atr_window[-1]
            # Percentile rank: fraction of past bars ≤ current
            rank = float((atr_window <= current).sum() / len(atr_window))
            return rank
        except Exception:
            return None

    def _get_cached_or_refresh(self, symbol: str) -> Optional[float]:
        now = time.time()
        with self._lock:
            cached = self._cache.get(symbol)
            if cached and (now - cached[0]) < CACHE_TTL_SEC:
                return cached[1]
        # Compute outside lock (I/O bound)
        rank = self._compute_pct_rank(symbol)
        with self._lock:
            self._cache[symbol] = (now, rank)
        return rank

    # ── Decision ─────────────────────────────────────────────────────
    def is_blocked(self, symbol: str, scanner: str) -> Optional[str]:
        """Return veto string if scanner+symbol+regime should be blocked."""
        if not scanner or scanner not in CHOP_ALLERGIC_SCANNERS:
            return None  # only gate scanners we know are chop-allergic
        if not symbol:
            return None
        rank = self._get_cached_or_refresh(symbol)
        if rank is None:
            return None  # data unavailable, fail open
        if rank < ATR_RANK_FLOOR:
            return (f"ATR_REGIME_K: {symbol} 4h_atr_pct_rank={rank:.2f} < "
                    f"{ATR_RANK_FLOOR:.2f} (chop regime — {scanner} disabled)")
        return None

    # ── Diagnostics ──────────────────────────────────────────────────
    def stats(self) -> Dict[str, Dict]:
        out = {}
        now = time.time()
        with self._lock:
            for sym, (ts, rank) in self._cache.items():
                out[sym] = {
                    "atr_pct_rank": rank,
                    "age_sec": int(now - ts),
                    "blocked_for_chop_scanners": (rank is not None and rank < ATR_RANK_FLOOR),
                }
        return out


_GATE: Optional[ATRPercentileGate] = None


def get_regime_gate() -> ATRPercentileGate:
    global _GATE
    if _GATE is None:
        _GATE = ATRPercentileGate()
    return _GATE

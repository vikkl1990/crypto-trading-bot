"""
Phase 5.20-B3 — Passive latency measurement.

Records signal→fill latency at each stage of the entry pipeline.
Just measurement (no SLO enforcement yet — that's a Phase 5.20-C item).

Stages:
    SIGNAL_EMIT  — when scanner finished classifying
    QUALIFY_PASS — when admission gate let signal through
    ORDER_SENT   — when create_order returned
    ORDER_FILLED — when fill confirmation arrived

Storage: ring buffer in memory (no DB writes per signal). Metrics
endpoint exposes p50/p95/p99 over last 500 entries.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class LatencyRecord:
    signal_id: str
    stages_ms: Dict[str, float]  # stage_name → ms_since_signal_emit


class LatencyMeter:
    """Per-symbol rolling latency tracker."""

    def __init__(self, max_records: int = 500):
        self._records: Deque[LatencyRecord] = deque(maxlen=max_records)
        self._inflight: Dict[str, Dict[str, float]] = {}  # signal_id → {stage: timestamp_s}

    def mark(self, signal_id: str, stage: str) -> None:
        """Record a stage timestamp for this signal."""
        if not signal_id:
            return
        now = time.perf_counter()
        if signal_id not in self._inflight:
            self._inflight[signal_id] = {"_emit": now}  # implicit start
        self._inflight[signal_id][stage] = now

    def finalize(self, signal_id: str) -> Optional[LatencyRecord]:
        """Move from inflight to records. Compute deltas from emit."""
        info = self._inflight.pop(signal_id, None)
        if not info:
            return None
        emit_ts = info.pop("_emit", None)
        if emit_ts is None:
            return None
        stages_ms = {
            stage: round((ts - emit_ts) * 1000, 2)
            for stage, ts in info.items()
        }
        rec = LatencyRecord(signal_id=signal_id, stages_ms=stages_ms)
        self._records.append(rec)
        return rec

    def percentile(self, stage: str, p: float = 0.95) -> Optional[float]:
        """p50, p95, p99 etc on a stage's latency."""
        vals = [r.stages_ms[stage] for r in self._records if stage in r.stages_ms]
        if not vals:
            return None
        vals.sort()
        idx = max(0, int(len(vals) * p) - 1)
        return vals[idx]

    def stats(self) -> Dict[str, Dict[str, float]]:
        """All-stage summary."""
        stages: Dict[str, list] = {}
        for r in self._records:
            for s, v in r.stages_ms.items():
                stages.setdefault(s, []).append(v)
        out = {}
        for s, vals in stages.items():
            vals.sort()
            n = len(vals)
            if n == 0:
                continue
            out[s] = {
                "n": n,
                "p50_ms": vals[max(0, int(n * 0.50) - 1)],
                "p95_ms": vals[max(0, int(n * 0.95) - 1)],
                "p99_ms": vals[max(0, int(n * 0.99) - 1)],
                "max_ms": vals[-1],
            }
        return out

    def clear_inflight_older_than(self, sec: float = 300.0) -> int:
        """GC stale inflight signals (didn't reach finalize)."""
        cutoff = time.perf_counter() - sec
        stale = [
            sid for sid, stamps in self._inflight.items()
            if stamps.get("_emit", 0) < cutoff
        ]
        for sid in stale:
            del self._inflight[sid]
        return len(stale)


# Module singleton
_meter = LatencyMeter()


def get_meter() -> LatencyMeter:
    return _meter

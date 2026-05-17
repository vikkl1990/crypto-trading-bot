"""PATCH_R_5_22 (2026-05-03) — Ensemble Overlay (observe + tag mode).

Detects concurrent same-side signals across paper engines + main-bot scanner.
Tags resulting shadow trades with ensemble metadata. Optionally applies
size_mult / score_boost when MODE=enforce.

W/F-derived per-pair policy (combos validated 2026-05-03):
  SOL: liq_grab_ob_fvg + liquidity_sweep_htf @ window=1 → size_mult 1.5
  BTC: scalper_vwap_mr + structure_bounce @ window=3 → score_boost 0.05
  ETH: scalper_vwap_mr + structure_bounce @ window=3 → ANTI (size_mult 0.5)
  ETH: absorption_bubble + structure_bounce @ window=1 → score_boost 0.05
  ETH: absorption_bubble + scalper_vwap_mr @ window=1 → score_boost 0.05

MODES (env var ENSEMBLE_OVERLAY_MODE, default 'observe'):
  observe    — log events to journal, NO trade modification (default — Phase 1)
  shadow_tag — observe + add ensemble fields to trade metadata (no sizing)
  enforce    — observe + tag + apply size_mult / score_boost (DEFERRED until soak proves stable)

Storage:
  storage/ensemble_overlay/policy.json   — config (this module reads at startup)
  storage/ensemble_overlay/events.jsonl  — append-only journal of detected events
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

STORAGE_DIR = Path("/home/opc/crypto-trading-bot/storage/ensemble_overlay")
POLICY_PATH = STORAGE_DIR / "policy.json"
EVENTS_PATH = STORAGE_DIR / "events.jsonl"

# Lookback for storing recent signals (broad — covers any window in policy)
RECENT_SIGNALS_MAXLEN = 500
RECENT_SIGNALS_TTL_SEC = 60 * 60   # 1 hour rolling window


def _now_ts() -> float:
    return time.time()


def _mode() -> str:
    """Active mode: observe / shadow_tag / enforce. Default observe (Phase 1)."""
    return (os.environ.get("ENSEMBLE_OVERLAY_MODE") or "observe").strip().lower()


@dataclass
class _RecentSignal:
    ts: float
    source: str        # source_engine or scanner name
    symbol: str
    side: str          # "long" | "short"


@dataclass
class _EnsembleMatch:
    rule_id: str
    symbol: str
    side: str
    sources_matched: List[str]
    window_bars: int
    size_mult: float
    score_boost: float
    ev_lift_estimated: float
    deployment: str
    matched_at_ts: float


class EnsembleOverlay:
    def __init__(self):
        self._lock = threading.Lock()
        self._recent: Deque[_RecentSignal] = deque(maxlen=RECENT_SIGNALS_MAXLEN)
        self._policy: List[Dict[str, Any]] = []
        self._loaded_at: float = 0.0
        self._journal_writes: int = 0
        self._matches_seen: int = 0
        self._tags_applied: int = 0
        self._reload_policy()

    # ── Policy ──────────────────────────────────────────────────────
    def _reload_policy(self) -> None:
        try:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            if POLICY_PATH.exists():
                data = json.loads(POLICY_PATH.read_text())
                self._policy = data.get("rules", [])
                self._loaded_at = _now_ts()
            else:
                self._policy = []
        except Exception:
            self._policy = []

    def policy_summary(self) -> Dict[str, Any]:
        return {
            "rules_count": len(self._policy),
            "loaded_at": self._loaded_at,
            "mode": _mode(),
            "matches_seen": self._matches_seen,
            "tags_applied": self._tags_applied,
            "journal_writes": self._journal_writes,
            "recent_signals_n": len(self._recent),
        }

    # ── Signal ingestion ────────────────────────────────────────────
    def record_signal(self, source: str, symbol: str, side: str) -> None:
        """Called when ANY signal fires (paper engine via shadow_bridge OR
        main-bot scanner via the broadcast hook). Stores in rolling window
        for ensemble detection on subsequent same-symbol signals."""
        if not source or not symbol or not side:
            return
        side_norm = side.lower()
        if side_norm not in ("long", "short"):
            return
        with self._lock:
            # Prune stale entries (TTL guard against unbounded memory)
            cutoff = _now_ts() - RECENT_SIGNALS_TTL_SEC
            while self._recent and self._recent[0].ts < cutoff:
                self._recent.popleft()
            self._recent.append(_RecentSignal(
                ts=_now_ts(), source=source, symbol=symbol, side=side_norm,
            ))

    # ── Ensemble detection ──────────────────────────────────────────
    def _check_match(self, symbol: str, side: str, scanner: str) -> Optional[_EnsembleMatch]:
        """For a CURRENT signal (the one about to execute), look back at
        recent_signals for matching ensemble rule. Returns first match.

        scanner = the scanner that fired the CURRENT signal (e.g. structure_bounce
                  if main bot, or absorption_bubble if paper engine bridge).
        """
        if not self._policy:
            return None
        side_norm = side.lower()
        now = _now_ts()
        # Snapshot recent under lock
        with self._lock:
            recent = list(self._recent)
        # window_bars converted: 1 bar=300s on 5m, generous to allow small jitter
        # (a 1-bar window in the W/F sense ≈ ≤300s of wall time)
        for rule in self._policy:
            if rule.get("symbol") != symbol:
                continue
            req_scanners = set(rule.get("scanners", []))
            if scanner not in req_scanners:
                continue   # current signal must be one of the rule's scanners
            other_required = req_scanners - {scanner}
            if not other_required:
                continue
            window_sec = float(rule.get("window_bars", 1)) * 300.0   # 5m bars
            # Look back for any same-side signal from one of the other_required scanners
            matched_sources = {scanner}
            for r in reversed(recent):
                if (now - r.ts) > window_sec:
                    break
                if r.symbol != symbol:
                    continue
                if r.side != side_norm:
                    continue
                if r.source in other_required:
                    matched_sources.add(r.source)
                    if matched_sources >= req_scanners:
                        return _EnsembleMatch(
                            rule_id=rule.get("id", "?"),
                            symbol=symbol, side=side_norm,
                            sources_matched=sorted(matched_sources),
                            window_bars=int(rule.get("window_bars", 1)),
                            size_mult=float(rule.get("size_mult", 1.0) or 1.0),
                            score_boost=float(rule.get("score_boost", 0.0) or 0.0),
                            ev_lift_estimated=float(rule.get("ev_lift_q4", 0.0) or 0.0),
                            deployment=str(rule.get("deployment", "ml_feature_only")),
                            matched_at_ts=now,
                        )
        return None

    # ── Public hook: tag trade metadata + journal event ─────────────
    def tag_trade(self, meta: Dict[str, Any], symbol: str, side: str,
                  scanner: Optional[str] = None) -> Dict[str, Any]:
        """Called from _execute_shadow (and consumer for bridge trades).
        ALWAYS observes + journals. ONLY enriches meta if MODE != observe.

        Returns the (possibly enriched) meta dict for caller to use.
        """
        if not isinstance(meta, dict):
            return meta if meta is not None else {}
        if scanner is None:
            scanner = (meta.get("scanner") or meta.get("setup_type") or "?")
        match = self._check_match(symbol, side, scanner)
        mode = _mode()

        # Always emit journal event (observe is the floor)
        try:
            self._journal_event(symbol=symbol, side=side, scanner=scanner,
                                match=match, mode=mode)
        except Exception:
            pass

        if match is None:
            return meta

        self._matches_seen += 1

        # observe mode: no meta modification, just journal
        if mode == "observe":
            return meta

        # shadow_tag + enforce: enrich meta with overlay fields
        meta["overlay_rule_id"] = match.rule_id
        meta["overlay_sources_matched"] = match.sources_matched
        meta["overlay_window_bars"] = match.window_bars
        meta["overlay_size_mult"] = match.size_mult
        meta["overlay_score_boost"] = match.score_boost
        meta["overlay_ev_lift_estimated"] = match.ev_lift_estimated
        meta["overlay_deployment"] = match.deployment
        meta["overlay_mode"] = mode
        self._tags_applied += 1

        # enforce mode: would also modify size + score (DEFERRED in Phase 1)
        # Phase 1 stops at tagging. Phase 2 wires sizing.
        return meta

    # ── Journal ─────────────────────────────────────────────────────
    def _journal_event(self, symbol: str, side: str, scanner: str,
                       match: Optional[_EnsembleMatch], mode: str) -> None:
        record = {
            "ts": _now_ts(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "mode": mode,
            "symbol": symbol,
            "side": side.lower() if side else "?",
            "scanner": scanner,
            "matched": match is not None,
        }
        if match is not None:
            record["match"] = asdict(match)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        fd = os.open(str(EVENTS_PATH), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        self._journal_writes += 1


# ─────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────
_OVERLAY: Optional[EnsembleOverlay] = None


def get_overlay() -> EnsembleOverlay:
    global _OVERLAY
    if _OVERLAY is None:
        _OVERLAY = EnsembleOverlay()
    return _OVERLAY

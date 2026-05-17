"""DUD Advisor — wraps dud_predictor_v1.joblib for advisory-only use.

Architectural role: ADVISORY (logs predictions, does NOT block trades).
Wired into signal_tracker.track_signal() to predict P(no_follow_through)
at signal admit time. Records to storage/dud_advisor/<date>.jsonl.

After 7 days of advisory data, architect can review:
  - Did the advisor correctly predict duds?
  - Would blocking at threshold X have improved net PnL?
  - Should we promote from advisory to enforcing gate?

Per architect spec (2026-05-04 v6 system):
  TARGET: predict BEFORE entry whether trade reaches >= 0.4R within 5-10 min
  LABEL: 1 if peak_mfe_r >= 0.4R, else 0
  RULE (future): if dud_probability > threshold: reject_trade()

Currently model is dud_predictor_v1 (LightGBM, AUC 0.638 on test).
Trained on 30d shadow trades, primarily structure_bounce.

Fail-open: model load or prediction errors return None (no block, no log).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ROOT = Path("/home/opc/crypto-trading-bot")
MODEL_PATH = ROOT / "storage" / "ml_models" / "dud_predictor_v1.joblib"
META_PATH = ROOT / "storage" / "ml_models" / "dud_predictor_v1.meta.json"
LOG_DIR = ROOT / "storage" / "dud_advisor"


class DudAdvisor:
    """Singleton wrapper for dud_predictor_v1.

    Loads the model lazily on first predict() call. If model fails to load
    (e.g., joblib version mismatch, missing file), all predict() calls return
    None — system treats this as "no advice" (fail-open).
    """

    def __init__(self):
        self._model = None
        self._meta: Optional[Dict[str, Any]] = None
        self._feature_columns: Optional[list] = None
        self._cat_columns: Optional[list] = None
        self._load_attempted = False
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> bool:
        """Lazy load the model. Returns True if loaded successfully."""
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        with self._lock:
            if self._model is not None:
                return True
            self._load_attempted = True
            try:
                import joblib
                payload = joblib.load(MODEL_PATH)
                self._model = payload["model"]
                self._feature_columns = payload.get("feature_columns", [])
                self._cat_columns = payload.get("cat_columns", [])
                if META_PATH.exists():
                    self._meta = json.loads(META_PATH.read_text())
                logger.warning(
                    "DudAdvisor loaded model: AUC=%.4f, features=%d, threshold_recommended=%.2f",
                    payload.get("auc_test", 0),
                    len(self._feature_columns),
                    payload.get("best_threshold", 0.5),
                )
                return True
            except Exception as e:
                self._load_error = str(e)
                logger.warning("DudAdvisor load failed (fail-open): %s", e)
                return False

    def predict_dud_prob(self, signal_dict: Dict[str, Any]) -> Optional[float]:
        """Return P(this signal is a DUD) ∈ [0, 1] or None on failure.

        signal_dict should be the same dict passed to signal_tracker.track_signal.
        """
        if not self._load():
            return None
        try:
            import pandas as pd
            row = self._build_feature_row(signal_dict)
            df = pd.DataFrame([row], columns=self._feature_columns)
            for c in self._cat_columns:
                if c in df.columns:
                    df[c] = df[c].astype("category")
            prob = float(self._model.predict_proba(df)[0, 1])
            return prob
        except Exception as e:
            logger.debug("DudAdvisor predict failed (fail-open): %s", e)
            return None

    def _build_feature_row(self, sig: Dict[str, Any]) -> Dict[str, Any]:
        """Map signal_dict fields to model feature columns.

        Matches dud_predictor_train.py CAT_FEATS + NUM_FEATS exactly.
        Missing fields default to safe values.
        """
        from datetime import datetime as _dt
        # Parse entry_time to extract hour/dow/minute
        entry_time = sig.get("entry_time") or sig.get("opened_at")
        try:
            ts = _dt.fromisoformat(str(entry_time).replace("Z", "+00:00"))
        except Exception:
            ts = _dt.now(timezone.utc)
        meta = sig.get("metadata", {}) if isinstance(sig.get("metadata"), dict) else {}
        return {
            "scanner":     sig.get("scanner") or sig.get("setup_type") or "?",
            "symbol":      sig.get("symbol", "?"),
            "side":        sig.get("side", "?"),
            "regime":      meta.get("regime") or sig.get("regime") or "?",
            "grade":       sig.get("grade") or meta.get("grade") or "?",
            "trade_type":  sig.get("trade_type") or meta.get("trade_type") or "SCALP",
            "session":     meta.get("session") or sig.get("session") or "?",
            "hour_utc":    ts.hour,
            "dow":         ts.weekday(),
            "minute_utc":  ts.minute,
            "ml_prob":     float(sig.get("ml_probability") or meta.get("ml_prob") or 0.5),
            "atr_ratio":   float(sig.get("atr_ratio") or meta.get("atr_ratio") or 1.0),
            "confidence":  float(sig.get("confidence") or 0.0),
            "leverage":    float(sig.get("leverage") or meta.get("leverage") or 20),
            "initial_risk": float(sig.get("initial_risk") or meta.get("initial_risk") or 0),
            "entry_price": float(sig.get("entry_price") or 0),
        }

    def log_advisory(self, signal_dict: Dict[str, Any], dud_prob: float) -> None:
        """Append advisory record to today's log file."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            f = LOG_DIR / f"{today}.jsonl"
            rec = {
                "ts":          datetime.now(timezone.utc).isoformat(),
                "trade_id":    signal_dict.get("trade_id"),
                "scanner":     signal_dict.get("scanner") or signal_dict.get("setup_type"),
                "symbol":      signal_dict.get("symbol"),
                "side":        signal_dict.get("side"),
                "grade":       signal_dict.get("grade"),
                "dud_prob":    round(dud_prob, 4),
                "would_block_at_thr_0.50": dud_prob >= 0.50,
                "would_block_at_thr_0.65": dud_prob >= 0.65,
                "would_block_at_thr_0.85": dud_prob >= 0.85,
            }
            line = json.dumps(rec) + "\n"
            fd = os.open(str(f), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        except Exception as e:
            logger.debug("DudAdvisor log_advisory failed (fail-open): %s", e)

    def summary(self) -> Dict[str, Any]:
        """Stats on advisor predictions today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        f = LOG_DIR / f"{today}.jsonl"
        if not f.exists():
            return {"loaded": self._model is not None, "today_n": 0}
        n = 0; would_block_50 = 0; would_block_65 = 0; would_block_85 = 0
        try:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try:
                        r = json.loads(line)
                        n += 1
                        if r.get("would_block_at_thr_0.50"): would_block_50 += 1
                        if r.get("would_block_at_thr_0.65"): would_block_65 += 1
                        if r.get("would_block_at_thr_0.85"): would_block_85 += 1
                    except Exception: continue
        except Exception: pass
        return {
            "loaded": self._model is not None,
            "today_n": n,
            "would_block_at_0.50": would_block_50,
            "would_block_at_0.65": would_block_65,
            "would_block_at_0.85": would_block_85,
        }


# Singleton
_ADVISOR: Optional[DudAdvisor] = None


def get_advisor() -> DudAdvisor:
    global _ADVISOR
    if _ADVISOR is None:
        _ADVISOR = DudAdvisor()
    return _ADVISOR

"""
Phase 5.11 — PPP admission gate (log-only to start).

PPPGate loads a trained LR (or LightGBM) bundle and provides `.predict(features)`.
Fail-open behavior: any error returns (1.0, 'failopen_*') which admits the signal.

Usage:
    from execution.ppp_gate import get_ppp_gate
    gate = get_ppp_gate()
    score, reason = gate.predict(features_dict)
    # score ∈ [0, 1], reason ∈ {'predicted', 'failopen_disabled',
    #                          'failopen_error', 'failopen_latency', 'failopen_missing_features'}

Model bundle format (pickled by ml_training.ppp_trainer):
    {
        'model': sklearn Pipeline (scaler + LR),
        'feature_cols': [list of feature names in expected order],
        'threshold': float,
        'oof_precision': float,
        'oof_recall': float,
        'oof_roc_auc': float,
        'oof_pr_auc': float,
        'n_samples': int,
        'n_positives': int,
        'trained_at': iso-format string,
    }

Log-only vs enforce:
    This class just returns scores. The decision (admit/reject) is made by
    the caller (e.g. UserRealRegistry.broadcast_signal or a backfill job)
    based on user.ppp_mode.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ppp_gate")

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "ml_training" / "models" / "ppp_lr_v1.pkl"


class PPPGate:
    """
    Peak Probability Predictor admission gate.

    Loads a trained model bundle and predicts probability of peak_mfe_r >= 0.30
    for a given signal's feature vector. Fail-open on any error.
    """

    def __init__(self, model_path: Optional[str | Path] = None, latency_budget_ms: float = 50.0):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._bundle: Optional[dict] = None
        self._disabled = False
        self._latency_budget_ms = latency_budget_ms
        self._prediction_times_ms: List[float] = []
        self._load()

    # --------------------------------------------------------------
    # Loading
    # --------------------------------------------------------------

    def _load(self) -> None:
        """Load the pickle bundle. Sets _disabled=True if unavailable."""
        if not self._model_path.exists():
            logger.warning(
                "PPP model not found at %s — gate will fail-open",
                self._model_path,
            )
            self._disabled = True
            return

        try:
            with open(self._model_path, "rb") as f:
                self._bundle = pickle.load(f)
        except Exception as e:
            logger.error("PPP model load failed: %s — gate will fail-open", e)
            self._disabled = True
            return

        # Validate bundle shape
        required = {"model", "feature_cols", "threshold"}
        missing = required - set(self._bundle.keys())
        if missing:
            logger.error("PPP bundle missing keys: %s — disabling", missing)
            self._bundle = None
            self._disabled = True
            return

        logger.info(
            "PPP gate loaded: threshold=%.3f prec=%.3f rec=%.3f n=%d trained=%s",
            self._bundle.get("threshold", 0),
            self._bundle.get("oof_precision", 0),
            self._bundle.get("oof_recall", 0),
            self._bundle.get("n_samples", 0),
            self._bundle.get("trained_at", "?"),
        )

    def reload(self) -> bool:
        """Reload the model from disk. Returns True if success."""
        self._bundle = None
        self._disabled = False
        self._load()
        return not self._disabled

    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------

    def predict(self, features: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score a feature vector.

        Returns (score ∈ [0, 1], reason_tag).

        reason_tag values:
            'predicted'                — normal path
            'failopen_disabled'        — model not loaded
            'failopen_missing_features'— feature vector unusable
            'failopen_error'           — prediction raised exception
            'failopen_latency'         — prediction exceeded 2× budget
        """
        if self._disabled or self._bundle is None:
            return 1.0, "failopen_disabled"

        if not features:
            return 1.0, "failopen_missing_features"

        start = time.perf_counter()
        try:
            model = self._bundle["model"]
            cols = self._bundle["feature_cols"]

            # Build feature vector in model's expected column order
            x = [[float(features.get(c, 0.0) or 0.0) for c in cols]]
            score = float(model.predict_proba(x)[0][1])

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._prediction_times_ms.append(elapsed_ms)
            if len(self._prediction_times_ms) > 500:
                self._prediction_times_ms = self._prediction_times_ms[-250:]

            # Latency blow-out → fail-open to protect hot path
            if elapsed_ms > self._latency_budget_ms * 2:
                logger.warning(
                    "PPP slow prediction: %.1fms > %.1fms budget",
                    elapsed_ms, self._latency_budget_ms,
                )
                return 1.0, "failopen_latency"

            return score, "predicted"

        except Exception as e:
            logger.error("PPP predict exception: %s", e)
            return 1.0, "failopen_error"

    # --------------------------------------------------------------
    # Accessors
    # --------------------------------------------------------------

    def get_threshold(self) -> float:
        """Model's chosen decision threshold. Returns 0.0 (admit-all) if no model."""
        if self._bundle:
            return float(self._bundle.get("threshold", 0.0))
        return 0.0

    def is_loaded(self) -> bool:
        return not self._disabled and self._bundle is not None

    def get_p95_latency_ms(self) -> float:
        if not self._prediction_times_ms:
            return 0.0
        sorted_times = sorted(self._prediction_times_ms)
        idx = max(0, int(len(sorted_times) * 0.95) - 1)
        return float(sorted_times[idx])

    def get_metadata(self) -> Dict[str, Any]:
        """Diagnostic info about the loaded model."""
        if not self._bundle:
            return {"loaded": False}
        return {
            "loaded": True,
            "threshold": self._bundle.get("threshold"),
            "oof_precision": self._bundle.get("oof_precision"),
            "oof_recall": self._bundle.get("oof_recall"),
            "oof_roc_auc": self._bundle.get("oof_roc_auc"),
            "oof_pr_auc": self._bundle.get("oof_pr_auc"),
            "n_samples": self._bundle.get("n_samples"),
            "n_positives": self._bundle.get("n_positives"),
            "trained_at": self._bundle.get("trained_at"),
            "model_path": str(self._model_path),
            "p95_latency_ms": self.get_p95_latency_ms(),
        }


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------

_gate_instance: Optional[PPPGate] = None


def get_ppp_gate(model_path: Optional[str | Path] = None) -> PPPGate:
    """Lazy-loaded module-level singleton."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = PPPGate(model_path)
    return _gate_instance


def reset_ppp_gate() -> None:
    """Clear the singleton. Next get_ppp_gate() will reload from disk."""
    global _gate_instance
    _gate_instance = None

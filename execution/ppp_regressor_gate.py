"""
Phase 5.16 — PPP Regressor advisory gate.

Loads the trained LightGBM regressor (ml_training/models/ppp_regressor_v1.pkl)
and provides .predict(features) → predicted_peak_r ∈ [0, 3+].

Decision rule (log-only by default):
    if predicted_peak_r >= default_admit_threshold_r → 'admit'
    else                                              → 'reject'

Failure modes (all fail-open: predicted_peak_r=999.0, decision='admit'):
    - Model not loaded
    - Empty/invalid features
    - Prediction raises exception
    - Latency > 100ms p95

Advisory mode:
    The decision is LOGGED to signal_features.ppp_regressor_decision but
    NEVER enforced. Counterfactual analysis at end of week answers
    "what would regressor have done?" without touching live trading.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ppp_regressor_gate")

_DEFAULT_PATH = Path(__file__).parent.parent / "ml_training" / "models" / "ppp_regressor_v1.pkl"


class PPPRegressorGate:
    """Continuous-output PPP gate. Predicts peak_mfe_r in R units."""

    def __init__(self, model_path: Optional[Path] = None, latency_budget_ms: float = 50.0):
        self._model_path = Path(model_path) if model_path else _DEFAULT_PATH
        self._bundle: Optional[dict] = None
        self._disabled = False
        self._latency_budget_ms = latency_budget_ms
        self._times_ms: List[float] = []
        self._load()

    def _load(self) -> None:
        if not self._model_path.exists():
            logger.warning("Regressor not found at %s — fail-open", self._model_path)
            self._disabled = True
            return
        try:
            with open(self._model_path, "rb") as f:
                self._bundle = pickle.load(f)
        except Exception as e:
            logger.error("Regressor load failed: %s", e)
            self._disabled = True
            return
        if "model" not in self._bundle or "feature_cols" not in self._bundle:
            logger.error("Regressor bundle missing keys")
            self._bundle = None
            self._disabled = True
            return
        logger.info(
            "PPPRegressor loaded: spearman=%.3f mae=%.3f n=%d",
            self._bundle.get("oof_spearman", 0),
            self._bundle.get("oof_mae_r", 0),
            self._bundle.get("n_samples", 0),
        )

    def reload(self) -> bool:
        self._bundle = None
        self._disabled = False
        self._load()
        return not self._disabled

    def predict(self, features: Dict[str, Any]) -> Tuple[float, str]:
        """
        Returns (predicted_peak_r, reason_tag).

        reason_tag:
            'predicted'                    — normal path
            'failopen_disabled'           — model not loaded
            'failopen_missing_features'   — empty features
            'failopen_error'              — exception during predict
            'failopen_latency'            — > 2x latency budget
        """
        if self._disabled or self._bundle is None:
            return 999.0, "failopen_disabled"
        if not features:
            return 999.0, "failopen_missing_features"

        start = time.perf_counter()
        try:
            cols = self._bundle["feature_cols"]
            x = [[float(features.get(c, 0.0) or 0.0) for c in cols]]
            pred = float(self._bundle["model"].predict(x)[0])
            # Clamp to physical range
            pred = max(0.0, min(pred, 5.0))

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._times_ms.append(elapsed_ms)
            if len(self._times_ms) > 500:
                self._times_ms = self._times_ms[-250:]

            if elapsed_ms > self._latency_budget_ms * 2:
                logger.warning("Regressor slow: %.1fms", elapsed_ms)
                return 999.0, "failopen_latency"
            return pred, "predicted"
        except Exception as e:
            logger.error("Regressor predict exception: %s", e)
            return 999.0, "failopen_error"

    def get_threshold(self) -> float:
        if self._bundle:
            return float(self._bundle.get("default_admit_threshold_r", 0.20))
        return 0.0  # admit-all on failopen

    def is_loaded(self) -> bool:
        return not self._disabled and self._bundle is not None

    def get_p95_latency_ms(self) -> float:
        if not self._times_ms:
            return 0.0
        s = sorted(self._times_ms)
        return float(s[max(0, int(len(s) * 0.95) - 1)])

    def get_metadata(self) -> Dict[str, Any]:
        if not self._bundle:
            return {"loaded": False}
        return {
            "loaded": True,
            "model_type": self._bundle.get("model_type"),
            "threshold_r": self.get_threshold(),
            "spearman": self._bundle.get("oof_spearman"),
            "spearman_ci_low": self._bundle.get("oof_spearman_ci_low"),
            "spearman_ci_high": self._bundle.get("oof_spearman_ci_high"),
            "mae_r": self._bundle.get("oof_mae_r"),
            "rmse_r": self._bundle.get("oof_rmse_r"),
            "decile_lift_r": self._bundle.get("decile_lift_r"),
            "n_samples": self._bundle.get("n_samples"),
            "trained_at": self._bundle.get("trained_at"),
            "p95_latency_ms": self.get_p95_latency_ms(),
        }


_singleton: Optional[PPPRegressorGate] = None


def get_regressor_gate(model_path: Optional[Path] = None) -> PPPRegressorGate:
    global _singleton
    if _singleton is None:
        _singleton = PPPRegressorGate(model_path)
    return _singleton


def reset_regressor_gate() -> None:
    global _singleton
    _singleton = None

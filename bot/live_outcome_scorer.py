"""
Live Outcome Scorer — Local Inference Client
=============================================
Loads `storage/ml_models/model_live_shared.joblib` and scores signals
against the live-outcome model trained on REALIZED P&L (not MFE).

ARCHITECTURE
------------
Unlike the existing MLScorer (which POSTs to VM4 for per-scanner candidate
predictions), this scorer runs LOCALLY — joblib model loaded into memory,
no network round-trip. Adds ~3-5ms per signal (vs ~20-50ms for VM4 round-trip).

WHY A SEPARATE SCORER
---------------------
The candidate model and live_outcome model serve different purposes:
  - Candidate (per-scanner, 48 features, MFE-trained): predicts whether a
    candidate would have hit a 0.3R MFE peak. Trained on synthetic backtest.
  - Live_outcome (shared, 30 features, PnL-trained): predicts realized P&L > 0.
    Trained on actual `ml_live_feedback.jsonl` records.

Backtest 2026-05-02 (4,530 records) showed:
  Candidate AUC:    0.5712
  Live_outcome AUC: 0.6415  (+7.03 pp)
  Spearman ρ:       0.476   (rank-different — they're picking up complementary signal)

Top decile WR comparison (last 7d, n=87 each):
  Candidate decile 10: 92% WR, +$12.03 avg
  Live_outcome decile 10: 91% WR, +$10.94 avg
  AT >0.65 threshold:    100% WR, +$14.52 avg (n=14 — small but pristine)

USAGE
-----
At bot startup (in scalp_strategy or main.py):

    from bot.live_outcome_scorer import LiveOutcomeScorer
    self._live_outcome_scorer = LiveOutcomeScorer()

At signal time (in _build_signal or right after candidate ml_result):

    live_score = self._live_outcome_scorer.score({
        "regime":          regime,                         # str: 'sideways', 'trending_up', etc.
        "scanner":         best_sr.scanner_name,           # str
        "side":            "long" if best.side == OrderSide.LONG else "short",
        "session":         self._current_session,          # str: 'asia_late', 'asia_early', 'europe', 'us'
        "symbol":          symbol,                         # 'BTC/USDT' etc.
        "trade_type":      trade_type,                     # 'SCALP', 'INTRADAY', 'RUNNER'
        "confidence":      best.confidence,                # 0-100
        "atr_ratio":       self._atr_ratio,                # float ~0.7-1.3
        "ml_probability":  ml_result.get("probability"),   # from candidate model (cascade)
        "leverage":        leverage,                       # int (default 20)
        "position_size_usd": position_size_usd,            # estimated/actual
    })

    signal.metadata["live_outcome_prob"] = live_score["probability"]
    signal.metadata["live_outcome_model_age_h"] = live_score["model_age_hours"]

PHASE 1 (this module): MEASUREMENT ONLY — log live_outcome_prob in metadata.
                       Don't change decisions. Accumulate 24h of dual predictions.
PHASE 2 (after baseline): wire as additional gate or sizer based on observed lift.

HOT-RELOAD
----------
The model is reloaded automatically when the joblib file's mtime changes.
This means a fresh weekly retrain (which writes a new joblib) will be
picked up on the next score() call without bot restart. Failsafe to
neutral 0.5 if reload fails.

FAIL-OPEN DESIGN
----------------
Any error → returns probability=0.5 (neutral) + error string. Never raises.
Never blocks a trade. This is critical because we don't want a bug in the
scorer to take the bot offline.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("/home/opc/crypto-trading-bot/storage/ml_models/model_live_shared.joblib")
RELOAD_CHECK_INTERVAL_SEC = 300  # check mtime every 5 min, not every score

# Canonical feature set (must match live_outcome_trainer.py CANONICAL_FEATURES)
CANONICAL_FEATURES = [
    "regime_trending_up", "regime_trending_down", "regime_ranging",
    "regime_volatile", "regime_quiet", "regime_sideways",
    "confidence", "atr_ratio", "ml_probability",
    "scanner_structure_bounce", "scanner_bos_choch", "scanner_liquidity_sweep",
    "scanner_trend_continuation", "scanner_ema_momentum",
    "scanner_vwap_mean_revert", "scanner_rsi_divergence", "scanner_cvd_divergence",
    "side_long",
    "session_asia_late", "session_asia_early", "session_europe", "session_us",
    "is_scalp", "is_intraday", "is_runner",
    "is_btc", "is_eth", "is_sol",
    "leverage_norm", "position_norm",
]


# ─────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────
class LiveOutcomeScorer:
    """Local inference for the live_outcome shared model.

    Trains on realized P&L (pnl_usd > 0) — see ml_training/live_outcome_trainer.py.
    Loaded from storage/ml_models/model_live_shared.joblib.

    Returns probability in [0, 1] that a trade with the given context would
    close profitably. 0.5 = neutral (model has no opinion or failed to load).
    """

    def __init__(self, model_path: Path = MODEL_PATH, enabled: bool = True):
        self._model_path = Path(model_path)
        self._enabled = enabled
        self._model = None
        self._saved_features: List[str] = []
        self._loaded_mtime: float = 0.0
        self._last_check_ts: float = 0.0
        self._last_error: Optional[str] = None
        self._stats = {"calls": 0, "errors": 0, "model_loads": 0,
                       "trained_at": None, "n_trades": 0, "avg_auc": 0.0}
        self._reload_if_needed(force=True)

    # ─── Model loading ────────────────────────────────────────────────
    def _reload_if_needed(self, force: bool = False) -> None:
        """Hot-reload the joblib model if its mtime changed.

        Called at __init__ and (cheaply) on each score() — but actually
        stat()s the file at most once per RELOAD_CHECK_INTERVAL_SEC.
        """
        now = time.time()
        if not force and (now - self._last_check_ts) < RELOAD_CHECK_INTERVAL_SEC:
            return
        self._last_check_ts = now
        try:
            if not self._model_path.exists():
                self._last_error = f"model file missing: {self._model_path}"
                self._model = None
                return
            mtime = self._model_path.stat().st_mtime
            if mtime == self._loaded_mtime and self._model is not None:
                return  # unchanged
            import joblib  # local import to avoid module-load failure if missing
            pkg = joblib.load(self._model_path)
            if not isinstance(pkg, dict) or "model" not in pkg or "feature_names" not in pkg:
                self._last_error = "model file malformed (expected dict with 'model' + 'feature_names')"
                self._model = None
                return
            self._model = pkg["model"]
            self._saved_features = list(pkg["feature_names"])
            self._loaded_mtime = mtime
            self._stats["model_loads"] += 1
            self._stats["trained_at"] = pkg.get("trained_at")
            self._stats["n_trades"] = pkg.get("n_trades", 0)
            self._stats["avg_auc"] = pkg.get("avg_auc", 0.0)
            self._last_error = None
            logger.warning(
                "LiveOutcomeScorer: loaded model trained_at=%s n_trades=%d AUC=%.4f features=%d",
                self._stats["trained_at"], self._stats["n_trades"],
                self._stats["avg_auc"], len(self._saved_features),
            )
        except Exception as e:
            self._last_error = f"reload failed: {e}"
            logger.error("LiveOutcomeScorer reload failed: %s", e)

    # ─── Feature builder ──────────────────────────────────────────────
    @staticmethod
    def _build_features(ctx: Dict[str, Any]) -> Dict[str, float]:
        """Build a 30-feature dict from the bot's signal context.

        ctx keys (all optional, defaults assumed):
            regime, scanner, side, session, symbol, trade_type,
            confidence (0-100), atr_ratio (~1.0), ml_probability (0-1),
            leverage (default 20), position_size_usd (default 5000)
        """
        f = {k: 0.0 for k in CANONICAL_FEATURES}

        # Continuous features
        f["confidence"] = float(ctx.get("confidence", 0) or 0) / 100.0
        f["atr_ratio"] = float(ctx.get("atr_ratio", 1.0) or 1.0)
        f["ml_probability"] = float(ctx.get("ml_probability", 0.5) or 0.5)
        f["leverage_norm"] = float(ctx.get("leverage", 20) or 20) / 20.0
        f["position_norm"] = min(
            float(ctx.get("position_size_usd", 5000) or 5000) / 5000.0,
            2.0,
        )

        # One-hot regime
        rg = (ctx.get("regime") or "").lower()
        if f"regime_{rg}" in f:
            f[f"regime_{rg}"] = 1.0

        # One-hot scanner
        sc = (ctx.get("scanner") or "").lower()
        if f"scanner_{sc}" in f:
            f[f"scanner_{sc}"] = 1.0

        # Side
        if str(ctx.get("side", "")).lower() == "long":
            f["side_long"] = 1.0

        # One-hot session
        sess = (ctx.get("session") or "").lower()
        if f"session_{sess}" in f:
            f[f"session_{sess}"] = 1.0

        # One-hot trade_type
        tt = str(ctx.get("trade_type", "SCALP") or "SCALP").upper()
        if tt == "SCALP":
            f["is_scalp"] = 1.0
        elif tt == "INTRADAY":
            f["is_intraday"] = 1.0
        elif tt == "RUNNER":
            f["is_runner"] = 1.0

        # One-hot symbol (only top-3 included in model)
        sym = ctx.get("symbol", "")
        if sym == "BTC/USDT":
            f["is_btc"] = 1.0
        elif sym == "ETH/USDT":
            f["is_eth"] = 1.0
        elif sym == "SOL/USDT":
            f["is_sol"] = 1.0

        return f

    # ─── Public API ───────────────────────────────────────────────────
    def score(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Score a signal. Returns dict — never raises.

        Result dict:
            probability        — float in [0, 1]; 0.5 = neutral / unavailable
            model_age_hours    — hours since model was trained
            error              — error string if score failed (None if OK)
            verdict            — 'STRONG_TAKE' | 'TAKE' | 'WEAK' | 'AVOID' | 'NEUTRAL'
        """
        self._stats["calls"] += 1

        if not self._enabled:
            return {"probability": 0.5, "verdict": "DISABLED",
                    "model_age_hours": None, "error": "scorer disabled"}

        # Hot-reload if model file changed
        self._reload_if_needed()

        if self._model is None:
            self._stats["errors"] += 1
            return {"probability": 0.5, "verdict": "NEUTRAL",
                    "model_age_hours": None,
                    "error": self._last_error or "model not loaded"}

        try:
            features = self._build_features(ctx)
            X = pd.DataFrame([features]).reindex(columns=self._saved_features, fill_value=0.0)
            prob_arr = self._model.predict_proba(X.values)
            prob = float(prob_arr[0, 1])
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("LiveOutcomeScorer.score failed: %s", e)
            return {"probability": 0.5, "verdict": "ERROR",
                    "model_age_hours": self.model_age_hours, "error": str(e)}

        # Verdict bands (calibrated on 4,530-record backtest 2026-05-02):
        # P(>0.65) = 14 trades, 100% WR, +$14.52 avg → STRONG_TAKE
        # P(0.55-0.65) ≈ deciles 8-9, 80%+ WR → TAKE
        # P(0.50-0.55) ≈ decile 7, 80% WR → WEAK
        # P(<0.50) ≈ deciles 1-6 → AVOID
        if prob >= 0.65:
            verdict = "STRONG_TAKE"
        elif prob >= 0.55:
            verdict = "TAKE"
        elif prob >= 0.50:
            verdict = "WEAK"
        else:
            verdict = "AVOID"

        return {
            "probability": prob,
            "verdict": verdict,
            "model_age_hours": self.model_age_hours,
            "error": None,
        }

    @property
    def model_age_hours(self) -> Optional[float]:
        if not self._stats.get("trained_at"):
            return None
        try:
            return (time.time() - float(self._stats["trained_at"])) / 3600.0
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """For dashboard/monitoring."""
        return {
            **self._stats,
            "loaded_mtime": self._loaded_mtime,
            "model_age_hours": self.model_age_hours,
            "last_error": self._last_error,
        }


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    scorer = LiveOutcomeScorer()
    print("\n=== Stats ===")
    print(json.dumps(scorer.get_stats(), indent=2, default=str))

    print("\n=== Smoke test scoring ===")
    test_cases = [
        # (label, ctx)
        ("Default-ish BTC long structure_bounce", {
            "regime": "sideways", "scanner": "structure_bounce", "side": "long",
            "session": "us", "symbol": "BTC/USDT", "trade_type": "SCALP",
            "confidence": 75, "atr_ratio": 1.0, "ml_probability": 0.55,
            "leverage": 20, "position_size_usd": 5000,
        }),
        ("Strong ETH short bos_choch", {
            "regime": "trending_down", "scanner": "bos_choch", "side": "short",
            "session": "europe", "symbol": "ETH/USDT", "trade_type": "INTRADAY",
            "confidence": 90, "atr_ratio": 1.2, "ml_probability": 0.78,
            "leverage": 20, "position_size_usd": 5000,
        }),
        ("Asia-early SOL long ema_momentum (weakspot)", {
            "regime": "ranging", "scanner": "ema_momentum", "side": "long",
            "session": "asia_early", "symbol": "SOL/USDT", "trade_type": "SCALP",
            "confidence": 60, "atr_ratio": 0.8, "ml_probability": 0.45,
            "leverage": 20, "position_size_usd": 5000,
        }),
        ("Empty context (failsafe)", {}),
    ]

    for label, ctx in test_cases:
        result = scorer.score(ctx)
        print(f"\n  {label}")
        print(f"    → prob={result['probability']:.4f}  verdict={result['verdict']}  "
              f"err={result.get('error')}")

    print("\n=== Final stats ===")
    print(json.dumps(scorer.get_stats(), indent=2, default=str))

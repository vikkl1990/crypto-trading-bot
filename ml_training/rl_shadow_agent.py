"""RL Shadow Agent — PPO-inspired sizing + trail aggressiveness tuner.

Outputs:
  sizing_mult (0.5-2.0): scales position size
  trail_aggression (0.7-1.0): scales trail lock percentage

Starts in shadow mode — logs suggestions but doesn't affect trades.
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "storage" / "ml_models" / "rl_shadow_weights.npz"


class RLShadowAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.shadow_mode = True
        self._last_action = np.array([1.0, 0.85])
        self.memory = []
        self._train_count = 0

        if model_path and Path(model_path).exists():
            self._load(model_path)
            logger.info("RL Shadow Agent loaded from %s", model_path)
        elif _WEIGHTS_PATH.exists():
            self._load(str(_WEIGHTS_PATH))
            logger.info("RL Shadow Agent loaded from default path")
        else:
            self.weights = self._init_weights()
            logger.info("RL Shadow Agent initialized with random weights")

    def predict(self, signal_dict: dict) -> dict:
        obs = self._extract_obs(signal_dict)
        action = self._forward(obs)
        sizing_mult = float(np.clip(action[0], 0.5, 2.0))
        trail_aggression = float(np.clip(action[1], 0.7, 1.0))

        return {
            "sizing_mult": sizing_mult,
            "trail_aggression": trail_aggression,
            "shadow_mode": self.shadow_mode,
            "observation": obs.tolist(),
        }

    def record_outcome(self, signal_dict: dict, realized_r: float):
        obs = self._extract_obs(signal_dict)
        self.memory.append((obs, self._last_action.copy(), realized_r))
        if len(self.memory) >= 50:
            self._train_batch()

    def _extract_obs(self, signal_dict: dict) -> np.ndarray:
        meta = signal_dict.get("metadata", {})
        regime_map = {"quiet": 0, "ranging": 1, "trending_up": 2, "trending_down": 3, "breakout": 4}
        grade_map = {"REJECT": 0, "C": 1, "B": 2, "A": 3, "A+": 4}
        session_map = {"asia_early": 0, "asia_late": 1, "europe": 2, "us": 3, "nse_post_open": 4}

        return np.array([
            float(meta.get("ml_probability", meta.get("ml_prob", 0.5))),
            regime_map.get(meta.get("regime", ""), 1) / 4.0,
            min(float(meta.get("atr_ratio", 1.0)), 5.0) / 5.0,
            min(float(meta.get("rel_vol", 1.0)), 5.0) / 5.0,
            float(signal_dict.get("confidence", 70)) / 100.0,
            grade_map.get(signal_dict.get("grade", "C"), 1) / 4.0,
            1.0 if meta.get("htf_aligned", False) or meta.get("htf_bias", 0) != 0 else 0.0,
            session_map.get(meta.get("session", ""), 2) / 4.0,
            min(float(meta.get("win_streak", 0)), 10) / 10.0,
            float(meta.get("recent_wr", 0.75)),
        ], dtype=np.float32)

    def _init_weights(self):
        np.random.seed(42)
        return {
            "w1": np.random.randn(10, 32).astype(np.float32) * 0.1,
            "b1": np.zeros(32, dtype=np.float32),
            "w2": np.random.randn(32, 16).astype(np.float32) * 0.1,
            "b2": np.zeros(16, dtype=np.float32),
            "w3": np.random.randn(16, 2).astype(np.float32) * 0.1,
            "b3": np.array([1.0, 0.85], dtype=np.float32),
        }

    def _forward(self, obs):
        h1 = np.tanh(obs @ self.weights["w1"] + self.weights["b1"])
        h2 = np.tanh(h1 @ self.weights["w2"] + self.weights["b2"])
        out = h2 @ self.weights["w3"] + self.weights["b3"]
        sizing = 0.5 + 1.5 / (1 + np.exp(-out[0]))
        trail = 0.7 + 0.3 / (1 + np.exp(-out[1]))
        self._last_action = np.array([sizing, trail])
        return self._last_action

    def _train_batch(self):
        if len(self.memory) < 10:
            return
        batch = self.memory[-50:]
        lr = 0.001
        for obs, action, reward in batch:
            h1 = np.tanh(obs @ self.weights["w1"] + self.weights["b1"])
            h2 = np.tanh(h1 @ self.weights["w2"] + self.weights["b2"])
            grad = reward * 0.01
            self.weights["b3"] += lr * grad * np.sign(action - self.weights["b3"])
            self.weights["w3"] += lr * grad * np.outer(h2, np.sign(action - self.weights["b3"])) * 0.1
        self._train_count += 1
        self.memory = self.memory[-200:]
        logger.info("RL Shadow: trained batch #%d (%d samples)", self._train_count, len(batch))

    def save(self, path: str = None):
        p = Path(path or str(_WEIGHTS_PATH))
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(p), **self.weights)
        logger.info("RL Shadow: saved weights to %s", p)

    def _load(self, path: str):
        data = np.load(path)
        self.weights = {k: data[k] for k in data.files}

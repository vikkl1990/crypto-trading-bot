"""
RCA Agent — Real-time Root Cause Analyst & Parameter Tuner

Monitors ALL trading parameters, detects performance degradation,
identifies root causes, and auto-tunes parameters to maintain WR.

Runs every 30 minutes. Compares last-50 trades vs all-time baseline.
If WR drops >10% or specific exit reasons spike, it adjusts parameters.

SAFE: Only adjusts within bounded ranges. Logs every change.
Can be set to suggest-only (no auto-apply) via SUGGEST_ONLY=True.
"""
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("bot.rca_agent")

_STORAGE = Path(__file__).resolve().parent.parent / "storage"
_STATE_FILE = _STORAGE / "rca_agent_state.json"


class RCAAgent:
    """Root Cause Analyst — monitors performance and tunes parameters."""

    # Parameter bounds (never go outside these)
    BOUNDS = {
        "scalp_max_age_sec": (120, 900),       # 2-15 min
        "intraday_max_age_sec": (180, 1800),   # 3-30 min
        "early_kill_sec": (15, 120),            # 15s - 2min
        "early_kill_mfe": (0.03, 0.15),         # 0.03R - 0.15R
        "trail_lock_030": (0.50, 0.95),         # 50-95% lock at 0.3R
        "trail_lock_050": (0.60, 0.95),
        "trail_lock_075": (0.70, 0.95),
        "trail_lock_100": (0.75, 0.98),
        "trail_lock_150": (0.80, 0.98),
        "ml_threshold_btc": (0.30, 0.55),
        "ml_threshold_eth": (0.30, 0.55),
        "ml_threshold_sol": (0.30, 0.55),
        "ml_default_threshold": (0.30, 0.55),
    }

    # Baseline targets
    TARGET_WR = 75.0          # target win rate
    WR_ALERT_DROP = 10.0      # alert if WR drops this much
    TIME_STOP_MAX_PCT = 20.0  # max % of trades that should be time_stop

    def __init__(self, signal_tracker=None, suggest_only=False):
        self._tracker = signal_tracker
        self.suggest_only = suggest_only
        self._last_run = 0
        self._run_interval = 1800  # 30 min
        self._history: List[Dict] = []
        self._suggestions: List[Dict] = []
        self._applied_changes: List[Dict] = []
        self._baseline_wr = 0
        self._load_state()

    def should_run(self) -> bool:
        return time.time() - self._last_run >= self._run_interval

    def run_analysis(self, closed_trades: List[Dict] = None) -> Dict[str, Any]:
        """Run full RCA analysis. Returns report dict."""
        self._last_run = time.time()

        if closed_trades is None:
            try:
                with open(_STORAGE / "closed_signals.json") as f:
                    closed_trades = json.load(f)
            except Exception:
                return {"status": "no_data"}

        if len(closed_trades) < 50:
            return {"status": "insufficient_data", "trades": len(closed_trades)}

        # Baseline: all-time stats
        all_wins = sum(1 for t in closed_trades if t.get("pnl_pct", 0) > 0)
        all_wr = all_wins / len(closed_trades) * 100
        self._baseline_wr = all_wr

        # Recent: last 50 trades
        recent = closed_trades[-50:]
        r_wins = sum(1 for t in recent if t.get("pnl_pct", 0) > 0)
        r_wr = r_wins / len(recent) * 100
        r_pnl = sum(t.get("pnl_pct", 0) for t in recent)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_wr": round(all_wr, 1),
            "recent_wr": round(r_wr, 1),
            "wr_delta": round(r_wr - all_wr, 1),
            "recent_pnl": round(r_pnl, 2),
            "recent_trades": len(recent),
            "total_trades": len(closed_trades),
            "issues": [],
            "suggestions": [],
            "applied": [],
        }

        # Detect issues
        self._check_wr_drop(report, all_wr, r_wr)
        self._check_exit_reasons(report, recent)
        self._check_duration_profile(report, recent)
        self._check_session_performance(report, recent)
        self._check_regime_performance(report, recent)

        # Generate and optionally apply parameter adjustments
        self._generate_suggestions(report)

        if not self.suggest_only and report["suggestions"]:
            self._apply_suggestions(report)

        # Store history
        self._history.append({
            "time": report["timestamp"],
            "baseline_wr": report["baseline_wr"],
            "recent_wr": report["recent_wr"],
            "issues": len(report["issues"]),
            "applied": len(report["applied"]),
        })
        self._history = self._history[-100:]
        self._save_state()

        # Log summary
        status = "HEALTHY" if not report["issues"] else "DEGRADED"
        logger.info(
            "RCA REPORT: %s | baseline=%.1f%% recent=%.1f%% delta=%+.1f%% | issues=%d suggestions=%d applied=%d",
            status, all_wr, r_wr, r_wr - all_wr, len(report["issues"]),
            len(report["suggestions"]), len(report["applied"]),
        )

        for issue in report["issues"]:
            logger.warning("RCA ISSUE: %s", issue)
        for sug in report["suggestions"]:
            logger.info("RCA SUGGEST: %s", sug)
        for app in report["applied"]:
            logger.info("RCA APPLIED: %s", app)

        return report

    def _check_wr_drop(self, report, baseline, recent):
        delta = recent - baseline
        if delta < -self.WR_ALERT_DROP:
            report["issues"].append(
                "WR dropped %.1f%% (%.1f%% → %.1f%%) — significant degradation" % (abs(delta), baseline, recent)
            )

    def _check_exit_reasons(self, report, recent):
        reasons = {}
        for t in recent:
            r = t.get("exit_reason", "?")
            if r not in reasons:
                reasons[r] = {"count": 0, "wins": 0, "pnl": 0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t.get("pnl_pct", 0)
            if t.get("pnl_pct", 0) > 0:
                reasons[r]["wins"] += 1

        # Check time_stop / max_age dominance
        time_exits = sum(d["count"] for r, d in reasons.items()
                        if "time_stop" in r or "max_age" in r or "momentum_kill" in r or "early_kill" in r)
        time_pct = time_exits / len(recent) * 100
        if time_pct > self.TIME_STOP_MAX_PCT:
            report["issues"].append(
                "Time-based exits at %.0f%% (%d/%d) — max_age may be too tight" % (time_pct, time_exits, len(recent))
            )
            report["_time_exit_pct"] = time_pct

        # Check if trail_profit is still dominant winner
        trail = reasons.get("trail_profit", {})
        if trail.get("count", 0) > 0:
            trail_wr = trail["wins"] / trail["count"] * 100
            report["_trail_wr"] = trail_wr
            report["_trail_count"] = trail["count"]

        report["_exit_reasons"] = reasons

    def _check_duration_profile(self, report, recent):
        fast = [t for t in recent if t.get("trade_duration_sec", 0) < 30]
        slow = [t for t in recent if t.get("trade_duration_sec", 0) >= 180]

        if fast:
            fast_wr = sum(1 for t in fast if t.get("pnl_pct", 0) > 0) / len(fast) * 100
        else:
            fast_wr = 0
        if slow:
            slow_wr = sum(1 for t in slow if t.get("pnl_pct", 0) > 0) / len(slow) * 100
        else:
            slow_wr = 0

        report["_fast_wr"] = fast_wr
        report["_slow_wr"] = slow_wr

        if slow_wr < 30 and len(slow) >= 5:
            report["issues"].append(
                "Slow trades (3m+) at %.0f%% WR (%d trades) — consider widening max_age or tightening entry" % (slow_wr, len(slow))
            )

    def _check_session_performance(self, report, recent):
        sessions = {}
        for t in recent:
            s = t.get("metadata", {}).get("session", "?")
            if s not in sessions:
                sessions[s] = {"count": 0, "wins": 0, "pnl": 0}
            sessions[s]["count"] += 1
            sessions[s]["pnl"] += t.get("pnl_pct", 0)
            if t.get("pnl_pct", 0) > 0:
                sessions[s]["wins"] += 1

        for s, d in sessions.items():
            if d["count"] >= 5:
                wr = d["wins"] / d["count"] * 100
                if wr < 40:
                    report["issues"].append(
                        "Session '%s' at %.0f%% WR (%d trades) — consider blocking" % (s, wr, d["count"])
                    )
        report["_sessions"] = sessions

    def _check_regime_performance(self, report, recent):
        regimes = {}
        for t in recent:
            r = t.get("metadata", {}).get("regime", "?")
            if r not in regimes:
                regimes[r] = {"count": 0, "wins": 0, "pnl": 0}
            regimes[r]["count"] += 1
            regimes[r]["pnl"] += t.get("pnl_pct", 0)
            if t.get("pnl_pct", 0) > 0:
                regimes[r]["wins"] += 1

        for r, d in regimes.items():
            if d["count"] >= 5:
                wr = d["wins"] / d["count"] * 100
                if wr < 40:
                    report["issues"].append(
                        "Regime '%s' at %.0f%% WR (%d trades) — weak conditions" % (r, wr, d["count"])
                    )
        report["_regimes"] = regimes

    def _generate_suggestions(self, report):
        suggestions = []

        # If time-based exits are too high → widen max_age
        if report.get("_time_exit_pct", 0) > self.TIME_STOP_MAX_PCT:
            current_scalp = 180  # 3 min
            new_scalp = min(current_scalp + 120, self.BOUNDS["scalp_max_age_sec"][1])
            suggestions.append({
                "param": "scalp_max_age_sec",
                "current": current_scalp,
                "suggested": new_scalp,
                "reason": "time_exits at %.0f%% — widen to give trades more room" % report["_time_exit_pct"],
            })

            current_intra = 300  # 5 min
            new_intra = min(current_intra + 180, self.BOUNDS["intraday_max_age_sec"][1])
            suggestions.append({
                "param": "intraday_max_age_sec",
                "current": current_intra,
                "suggested": new_intra,
                "reason": "time_exits at %.0f%% — widen INTRADAY too" % report["_time_exit_pct"],
            })

        # If slow trades have very low WR → tighten max_age instead
        if report.get("_slow_wr", 100) < 20 and report.get("_time_exit_pct", 0) <= self.TIME_STOP_MAX_PCT:
            suggestions.append({
                "param": "scalp_max_age_sec",
                "current": 180,
                "suggested": 120,
                "reason": "slow trades at %.0f%% WR — tighten to cut losses faster" % report["_slow_wr"],
            })

        # If WR dropped significantly → tighten trail to lock more profit
        if report.get("wr_delta", 0) < -10:
            suggestions.append({
                "param": "trail_lock_030",
                "current": 0.85,
                "suggested": min(0.92, self.BOUNDS["trail_lock_030"][1]),
                "reason": "WR dropped %.1f%% — tighten trail to lock more profit" % abs(report["wr_delta"]),
            })

        report["suggestions"] = suggestions
        self._suggestions = suggestions

    def _apply_suggestions(self, report):
        """DISABLED: RCA was overriding manual config. Log only, no mutations."""
        for sug in report.get("suggestions", []):
            logger.info("RCA SUGGEST (NOT APPLIED): %s", sug)
        return

    def _apply_suggestions_DISABLED(self, report):
        """Apply parameter changes to signal_tracker config."""
        applied = []

        for sug in report["suggestions"]:
            param = sug["param"]
            new_val = sug["suggested"]

            # Validate within bounds
            lo, hi = self.BOUNDS.get(param, (0, 999))
            new_val = max(lo, min(hi, new_val))

            # Apply to signal tracker if available
            if self._tracker:
                try:
                    if param == "scalp_max_age_sec":
                        from bot.signal_tracker import TRADE_TYPE_CONFIG, TRADE_TYPE_SCALP
                        old = TRADE_TYPE_CONFIG[TRADE_TYPE_SCALP]["max_age_sec"]
                        TRADE_TYPE_CONFIG[TRADE_TYPE_SCALP]["max_age_sec"] = int(new_val)
                        applied.append("%s: %s → %s (reason: %s)" % (param, old, new_val, sug["reason"]))

                    elif param == "intraday_max_age_sec":
                        from bot.signal_tracker import TRADE_TYPE_CONFIG, TRADE_TYPE_INTRADAY
                        old = TRADE_TYPE_CONFIG[TRADE_TYPE_INTRADAY]["max_age_sec"]
                        TRADE_TYPE_CONFIG[TRADE_TYPE_INTRADAY]["max_age_sec"] = int(new_val)
                        applied.append("%s: %s → %s (reason: %s)" % (param, old, new_val, sug["reason"]))

                    elif param == "early_kill_sec":
                        from bot.signal_tracker import TRADE_TYPE_CONFIG, TRADE_TYPE_SCALP
                        old = TRADE_TYPE_CONFIG[TRADE_TYPE_SCALP]["early_kill_sec"]
                        TRADE_TYPE_CONFIG[TRADE_TYPE_SCALP]["early_kill_sec"] = int(new_val)
                        applied.append("%s: %s → %s" % (param, old, new_val))

                except Exception as e:
                    logger.warning("RCA: Failed to apply %s: %s", param, e)

            self._applied_changes.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "param": param,
                "old": sug["current"],
                "new": new_val,
                "reason": sug["reason"],
            })

        report["applied"] = applied
        self._applied_changes = self._applied_changes[-50:]

    def get_report(self) -> Dict:
        """Get the latest analysis for dashboard display."""
        return {
            "last_run": self._last_run,
            "baseline_wr": self._baseline_wr,
            "history": self._history[-10:],
            "pending_suggestions": self._suggestions,
            "applied_changes": self._applied_changes[-10:],
            "suggest_only": self.suggest_only,
        }

    def _save_state(self):
        try:
            state = {
                "history": self._history,
                "applied_changes": self._applied_changes,
                "last_run": self._last_run,
                "baseline_wr": self._baseline_wr,
            }
            _STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception:
            pass

    def _load_state(self):
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text())
                self._history = data.get("history", [])
                self._applied_changes = data.get("applied_changes", [])
                self._last_run = data.get("last_run", 0)
                self._baseline_wr = data.get("baseline_wr", 0)
        except Exception:
            pass

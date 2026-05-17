"""
Phase 5.17 — PPP Dashboard API.

Provides /api/ppp endpoint with:
  - Model metadata (binary classifier + regressor)
  - Recent decision counts (admit/reject/failopen)
  - Top features by importance
  - Counterfactual P&L if PPP enforced

Read-only — no write operations.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg
from aiohttp import web

logger = logging.getLogger("ppp_api")


async def _model_metadata() -> Dict[str, Any]:
    """Load metadata for both binary and regressor models."""
    out = {"binary": None, "regressor": None}
    try:
        from execution.ppp_gate import get_ppp_gate
        out["binary"] = get_ppp_gate().get_metadata()
    except Exception as e:
        out["binary"] = {"error": str(e)}
    try:
        from execution.ppp_regressor_gate import get_regressor_gate
        out["regressor"] = get_regressor_gate().get_metadata()
    except Exception as e:
        out["regressor"] = {"error": str(e)}
    return out


async def _recent_decisions(conn: asyncpg.Connection, hours: int = 24) -> Dict[str, Any]:
    """Counts of recent advisory decisions per model."""
    row = await conn.fetchrow(f"""
        SELECT
          COUNT(*) AS n_total,
          COUNT(*) FILTER (WHERE ppp_decision = 'admit') AS n_admit,
          COUNT(*) FILTER (WHERE ppp_decision = 'reject') AS n_reject,
          COUNT(*) FILTER (WHERE ppp_reason LIKE 'failopen%') AS n_failopen,
          COUNT(*) FILTER (WHERE ppp_regressor_decision = 'admit') AS reg_n_admit,
          COUNT(*) FILTER (WHERE ppp_regressor_decision = 'reject') AS reg_n_reject,
          AVG(ppp_lr_score) AS avg_binary_score,
          AVG(ppp_regressor_score) AS avg_regressor_score
        FROM signal_features
        WHERE emitted_at > NOW() - INTERVAL '{int(hours)} hours'
    """)
    if not row:
        return {}
    return {
        "window_hours": hours,
        "total_signals": row["n_total"] or 0,
        "binary": {
            "admit": row["n_admit"] or 0,
            "reject": row["n_reject"] or 0,
            "failopen": row["n_failopen"] or 0,
            "avg_score": float(row["avg_binary_score"] or 0),
        },
        "regressor": {
            "admit": row["reg_n_admit"] or 0,
            "reject": row["reg_n_reject"] or 0,
            "avg_score_r": float(row["avg_regressor_score"] or 0),
        },
    }


async def _counterfactual_pnl(conn: asyncpg.Connection) -> Dict[str, Any]:
    """If PPP enforced (binary + regressor separately), what would P&L be?"""
    rows = await conn.fetch("""
        SELECT
          sf.ppp_decision,
          sf.ppp_regressor_decision,
          sf.will_peak_30r,
          ut.pnl_usd
        FROM signal_features sf
        LEFT JOIN user_trades ut ON ut.id::text = sf.label_trade_id
        WHERE sf.label_captured = TRUE
          AND ut.pnl_usd IS NOT NULL
    """)
    if not rows:
        return {"sample_size": 0}

    bin_admit_pnl = sum(float(r["pnl_usd"] or 0) for r in rows if r["ppp_decision"] == "admit")
    bin_reject_pnl = sum(float(r["pnl_usd"] or 0) for r in rows if r["ppp_decision"] == "reject")
    reg_admit_pnl = sum(float(r["pnl_usd"] or 0) for r in rows if r["ppp_regressor_decision"] == "admit")
    reg_reject_pnl = sum(float(r["pnl_usd"] or 0) for r in rows if r["ppp_regressor_decision"] == "reject")
    return {
        "sample_size": len(rows),
        "binary": {
            "admit_cohort_pnl": round(bin_admit_pnl, 2),
            "reject_cohort_pnl": round(bin_reject_pnl, 2),
            "savings_if_enforced": round(-bin_reject_pnl, 2),
        },
        "regressor": {
            "admit_cohort_pnl": round(reg_admit_pnl, 2),
            "reject_cohort_pnl": round(reg_reject_pnl, 2),
            "savings_if_enforced": round(-reg_reject_pnl, 2),
        },
    }


async def _maker_calibration(conn: asyncpg.Connection, days: int = 7) -> Dict[str, Any]:
    """Real maker fill rate from user_trades.metadata.fee_type."""
    rows = await conn.fetch(f"""
        SELECT
          COALESCE(NULLIF(metadata::jsonb->>'fee_type', ''), 'unknown') AS fill_type,
          COUNT(*) AS n
        FROM user_trades
        WHERE trade_type = 'real'
          AND opened_at >= NOW() - INTERVAL '{int(days)} days'
        GROUP BY fill_type
    """)
    counts = {r["fill_type"]: r["n"] for r in rows}
    total = sum(counts.values())
    maker = counts.get("maker", 0)
    taker = counts.get("taker", 0)
    other = total - maker - taker
    return {
        "window_days": days,
        "total_real_trades": total,
        "maker_fills": maker,
        "taker_fills": taker,
        "other": other,
        "maker_fill_rate_pct": round(100 * maker / max(maker + taker, 1), 1),
    }


def make_ppp_handler(db_pool):
    """Factory for the /api/ppp handler bound to a DB pool."""
    async def _handle_ppp(request: web.Request) -> web.Response:
        try:
            payload = {"models": await _model_metadata()}
            if db_pool:
                async with db_pool.acquire() as conn:
                    payload["recent_24h"] = await _recent_decisions(conn, 24)
                    payload["counterfactual"] = await _counterfactual_pnl(conn)
                    payload["maker_calibration_7d"] = await _maker_calibration(conn, 7)
            else:
                payload["error"] = "no_db_pool"
            return web.json_response(payload)
        except Exception as e:
            logger.exception("ppp api failed")
            return web.json_response({"error": str(e)}, status=500)
    return _handle_ppp

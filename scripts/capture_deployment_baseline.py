#!/usr/bin/env python3
"""
Phase 5.20-D — Capture deployment baseline.

Run on every bot start. Snapshots current performance metrics so the
auto-revert detector has a "before deploy" reference to compare against.

Logic:
  1. Compute pre-restart performance from last 30 closed trades
  2. Mark previous baseline as superseded
  3. Insert new baseline as is_active=TRUE
  4. Capture git_sha + active phases (best-effort)

Cron / startup hook:
    # Add to systemd ExecStartPre or call manually after restart:
    python3 scripts/capture_deployment_baseline.py
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


def _load_env(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v.strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _dsn():
    _load_env()
    if os.environ.get("DATABASE_URL", "").startswith("postgres"):
        return os.environ["DATABASE_URL"]
    return "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"


def _git_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out[:40]
    except Exception:
        return ""


def _bot_pid():
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "orchestrator"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0


async def fetch_pre_baseline_metrics(conn, n: int = 30):
    rows = await conn.fetch(f"""
        SELECT pnl_usd, closed_at
        FROM user_trades
        WHERE pnl_usd IS NOT NULL
          AND trade_type IN ('real', 'shadow')
        ORDER BY closed_at DESC
        LIMIT {int(n)}
    """)
    if not rows:
        return None
    pnls = [float(r["pnl_usd"] or 0) for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": len(pnls),
        "wr_pct": (wins / len(pnls)) * 100,
        "avg_pnl": sum(pnls) / len(pnls),
    }


async def fetch_active_phases() -> dict:
    """Best-effort: list of active phase markers from recent trades."""
    return {
        "marker": "phase_5.20_audit_safe_batch",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _config_hash() -> str:
    """Hash of relevant config files."""
    h = hashlib.sha256()
    for f in [".env", "config/settings.yaml"]:
        p = Path(__file__).parent.parent / f
        if p.exists():
            try:
                h.update(p.read_bytes())
            except Exception:
                pass
    return h.hexdigest()[:16]


async def main():
    conn = await asyncpg.connect(_dsn())
    try:
        # Mark all prior baselines as inactive
        await conn.execute("UPDATE deployment_baseline SET is_active = FALSE WHERE is_active = TRUE")

        # Compute pre-baseline metrics
        metrics = await fetch_pre_baseline_metrics(conn) or {}
        phases = await fetch_active_phases()

        new_baseline = await conn.fetchrow("""
            INSERT INTO deployment_baseline
              (bot_pid, git_sha, config_hash, active_phases,
               baseline_n_trades, baseline_wr_pct, baseline_avg_pnl_usd,
               baseline_sharpe_7d, baseline_latency_p95_ms, baseline_sl_revert_per_hr,
               is_active, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NULL, NULL, NULL, TRUE, $8)
            RETURNING id, captured_at
        """,
            _bot_pid(),
            _git_sha(),
            _config_hash(),
            json.dumps(phases),
            metrics.get("n", 0),
            metrics.get("wr_pct", 0),
            metrics.get("avg_pnl", 0),
            f"phase_5.20_audit batch ship",
        )

        print(f"\n=== Deployment baseline captured ===")
        print(f"  baseline_id: {new_baseline['id']}")
        print(f"  captured_at: {new_baseline['captured_at']}")
        print(f"  bot_pid:     {_bot_pid()}")
        print(f"  git_sha:     {_git_sha()[:12]}")
        print(f"  Pre-deploy metrics:")
        print(f"    n_trades: {metrics.get('n', 0)}")
        print(f"    WR:       {metrics.get('wr_pct', 0):.1f}%")
        print(f"    avg_pnl:  ${metrics.get('avg_pnl', 0):+.3f}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())

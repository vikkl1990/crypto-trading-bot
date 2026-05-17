#!/usr/bin/env python3
"""
Backfill `signal_features` from historical `user_trades` rows.

Per PPP_V2_IMPLEMENTATION_PLAN.md §5: accelerate Phase B training data by
reconstructing Phase 1 features from candle history on already-closed trades.

Usage:
    python3 scripts/backfill_ppp_features.py --days 30
    python3 scripts/backfill_ppp_features.py --days 7 --dry-run
    python3 scripts/backfill_ppp_features.py --symbol BTC/USDT --days 14

What it does:
    1. Query user_trades for closed trades in the lookback window.
    2. For each trade: load cached 5m candles for its symbol.
    3. Window the candles to the 25 bars preceding trade.opened_at.
    4. Build Phase 1 features via ml_training.ppp_features.build_phase1_features.
    5. INSERT a row into signal_features with the label (peak_mfe_r) captured.
    6. Skip trades with insufficient candle history or missing peak_mfe_r.

Output:
    Reports: total eligible / attempted / inserted / skipped (with reasons).
    Verifies final count in signal_features table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make sibling packages importable
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import asyncpg
import pandas as pd

from ml_training.ppp_features import build_phase1_features, feature_names

# Optional PPP gate — scored per row so we can A/B "what PPP would have decided"
try:
    from execution.ppp_gate import get_ppp_gate
    _PPP_AVAILABLE = True
except Exception:
    _PPP_AVAILABLE = False

# Candle cache lives at storage/candle_cache/{safe_symbol}_{tf}.parquet
# We read parquet directly to avoid needing an exchange_client instance.
_CANDLE_CACHE_DIR = Path(__file__).parent.parent / "storage" / "candle_cache"


def _load_cached_candles(symbol: str, timeframe: str = "5m") -> pd.DataFrame | None:
    """Read cached parquet candles for a symbol. Returns None if missing."""
    safe = symbol.replace("/", "_").replace(":", "_")
    pq_path = _CANDLE_CACHE_DIR / f"{safe}_{timeframe}.parquet"
    csv_path = _CANDLE_CACHE_DIR / f"{safe}_{timeframe}.csv"

    if pq_path.exists():
        try:
            df = pd.read_parquet(pq_path)
            if not df.empty:
                return df
        except Exception:
            pass
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
            if "datetime" in df.columns:
                df.set_index("datetime", inplace=True)
            if not df.empty:
                return df
        except Exception:
            pass
    return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_ppp")


def load_env_dict(env_path: str = ".env") -> dict:
    """Minimal .env reader — avoids dependency on python-dotenv."""
    env = {}
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    # Apply to os.environ so other modules pick up
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


def _get_pg_dsn(env: dict) -> str:
    """Build asyncpg DSN from env fallbacks."""
    if env.get("DATABASE_URL", "").startswith("postgres"):
        return env["DATABASE_URL"]
    # Fallback: individual vars
    user = env.get("PGUSER") or env.get("POSTGRES_USER") or "vnedge"
    pw   = env.get("PGPASSWORD") or env.get("POSTGRES_PASSWORD") or "VnEdge2026db"
    host = env.get("PGHOST") or env.get("POSTGRES_HOST") or "localhost"
    port = env.get("PGPORT") or env.get("POSTGRES_PORT") or "5432"
    db   = env.get("PGDATABASE") or env.get("POSTGRES_DB") or "vnedge"
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


async def fetch_eligible_trades(
    conn: asyncpg.Connection, days: int, symbol_filter: str | None
) -> list:
    """
    Return closed trades with a captured peak_mfe_r in the lookback window.
    Excludes trades already in signal_features (idempotent re-runs).
    """
    q = """
    SELECT
        ut.id,
        ut.symbol,
        ut.side,
        ut.opened_at,
        ut.entry_price,
        ut.trade_type,
        ut.metadata::jsonb->>'peak_mfe_r'  AS peak_r,
        ut.metadata::jsonb->>'grade'       AS grade,
        ut.metadata::jsonb->>'ml_prob'     AS ml_prob,
        ut.metadata::jsonb->>'regime'      AS regime,
        ut.metadata::jsonb->>'scanner'     AS scanner_type,
        ut.metadata::jsonb->>'confidence'  AS confidence
    FROM user_trades ut
    LEFT JOIN signal_features sf
        ON sf.label_trade_id = ut.id::text
    WHERE ut.opened_at >= NOW() - INTERVAL '%s days'
      AND ut.closed_at IS NOT NULL
      AND ut.pnl_usd IS NOT NULL
      AND ut.metadata::jsonb->>'peak_mfe_r' IS NOT NULL
      AND sf.signal_id IS NULL
      {symbol_clause}
    ORDER BY ut.opened_at ASC
    """ % days
    params: list = []
    if symbol_filter:
        q = q.format(symbol_clause="AND ut.symbol = $1")
        params.append(symbol_filter)
    else:
        q = q.format(symbol_clause="")

    rows = await conn.fetch(q, *params)
    return [dict(r) for r in rows]


def build_candle_window(
    collector: CandleCollector,
    symbol: str,
    opened_at: datetime,
    lookback_bars: int = 25,
) -> pd.DataFrame | None:
    """
    Load cached 5m candles and return the `lookback_bars` bars preceding opened_at.
    Returns None if insufficient history.
    """
    df = collector.load_cached(symbol, "5m")
    if df is None or len(df) == 0:
        return None

    # Ensure timestamp column is datetime
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif df.index.name == "timestamp" or pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.reset_index()
    else:
        logger.warning("Candle df for %s has no timestamp column — skipping", symbol)
        return None

    # Strip timezone from opened_at for safe comparison (both to UTC)
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)

    df = df[df["timestamp"] <= opened_at].tail(lookback_bars).reset_index(drop=True)
    if len(df) < 20:
        return None
    return df


async def ensure_table_exists(conn: asyncpg.Connection) -> None:
    """Verify signal_features table is present. Raises if migration not applied."""
    exists = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'signal_features')"
    )
    if not exists:
        raise RuntimeError(
            "signal_features table not found. Run migration 006_ppp_schema.sql first."
        )


async def insert_backfilled_row(
    conn: asyncpg.Connection,
    trade: dict,
    features: dict,
    ppp_score: float | None = None,
    ppp_threshold: float | None = None,
    ppp_reason: str | None = None,
) -> None:
    """INSERT one backfilled row. ON CONFLICT DO NOTHING for idempotency.

    If ppp_score provided, also populates the advisory decision (what PPP
    WOULD have done — log-only, never enforced by backfill).
    """
    peak_r = float(trade["peak_r"] or 0)
    will_peak = peak_r >= 0.30
    signal_id = f"hist_{trade['id']}"

    # Advisory decision: what would PPP have done if active in enforce mode?
    ppp_decision = None
    if ppp_score is not None and ppp_threshold is not None:
        ppp_decision = "admit" if ppp_score >= ppp_threshold else "reject"

    await conn.execute(
        """
        INSERT INTO signal_features (
            signal_id, emitted_at, symbol, side,
            scanner_type, grade, confidence, regime, ml_probability,
            features,
            peak_mfe_r, will_peak_30r, label_captured, label_trade_id, label_captured_at,
            ppp_lr_score, ppp_threshold, ppp_decision, ppp_reason, ppp_model_type
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,TRUE,$13, NOW(),
                $14, $15, $16, $17, 'lr')
        ON CONFLICT (signal_id) DO UPDATE SET
            ppp_lr_score = EXCLUDED.ppp_lr_score,
            ppp_threshold = EXCLUDED.ppp_threshold,
            ppp_decision = EXCLUDED.ppp_decision,
            ppp_reason = EXCLUDED.ppp_reason,
            ppp_model_type = EXCLUDED.ppp_model_type,
            updated_at = NOW()
        """,
        signal_id,
        trade["opened_at"],
        trade["symbol"],
        trade["side"],
        trade.get("scanner_type") or "",
        trade.get("grade") or "",
        int(trade.get("confidence") or 0) if (trade.get("confidence") or "").isdigit() else 0,
        trade.get("regime") or "",
        float(trade.get("ml_prob") or 0) if trade.get("ml_prob") else 0.0,
        json.dumps(features),
        peak_r,
        will_peak,
        str(trade["id"]),
        ppp_score,
        ppp_threshold,
        ppp_decision,
        ppp_reason,
    )


async def main(days: int, symbol: str | None, dry_run: bool):
    env = load_env_dict()
    dsn = _get_pg_dsn(env)
    logger.info("Connecting to PostgreSQL at %s...", dsn.split("@")[-1])
    conn = await asyncpg.connect(dsn)

    try:
        await ensure_table_exists(conn)
        trades = await fetch_eligible_trades(conn, days=days, symbol_filter=symbol)
        logger.info("Eligible trades (last %d days): %d", days, len(trades))

        if not trades:
            logger.warning("No eligible trades — nothing to backfill.")
            return

        # Candles read via parquet files directly (see _load_cached_candles above)

        counters: Counter = Counter()
        candle_cache_by_symbol: dict = {}  # avoid re-loading same symbol's parquet

        # Optional PPP scoring (log-only — never enforces rejection from backfill)
        ppp_gate = None
        ppp_threshold = None
        if _PPP_AVAILABLE:
            try:
                ppp_gate = get_ppp_gate()
                if ppp_gate.is_loaded():
                    ppp_threshold = ppp_gate.get_threshold()
                    logger.info(
                        "PPP gate loaded (threshold=%.3f) — scoring each row advisory-only",
                        ppp_threshold,
                    )
                else:
                    ppp_gate = None
                    logger.info("PPP gate not loaded (no model) — skipping scoring")
            except Exception as e:
                logger.warning("PPP gate init failed: %s — skipping scoring", e)
                ppp_gate = None

        for t in trades:
            counters["attempted"] += 1
            try:
                sym = t["symbol"]
                # Reuse loaded df across trades with same symbol
                if sym not in candle_cache_by_symbol:
                    candle_cache_by_symbol[sym] = _load_cached_candles(sym, "5m")
                df_full = candle_cache_by_symbol[sym]
                if df_full is None or len(df_full) == 0:
                    counters["skip_no_candles"] += 1
                    continue

                # Normalize: ensure we have a "timestamp" column (not index)
                if "timestamp" not in df_full.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_full.index):
                        df_full = df_full.reset_index().rename(
                            columns={df_full.index.name or "index": "timestamp"}
                        )
                    else:
                        # Try 'datetime' column fallback
                        if "datetime" in df_full.columns:
                            df_full = df_full.rename(columns={"datetime": "timestamp"})
                        else:
                            counters["skip_no_timestamp_col"] += 1
                            continue

                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(df_full["timestamp"]):
                    df_full["timestamp"] = pd.to_datetime(
                        df_full["timestamp"], utc=True, errors="coerce"
                    )
                # Make timezone-aware (UTC) if naive
                if df_full["timestamp"].dt.tz is None:
                    df_full["timestamp"] = df_full["timestamp"].dt.tz_localize("UTC")

                # Update cache with the normalized df so next trade reuses the fix
                candle_cache_by_symbol[sym] = df_full

                opened = t["opened_at"]
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=timezone.utc)

                df_window = df_full[df_full["timestamp"] <= opened].tail(25).reset_index(drop=True)
                if len(df_window) < 20:
                    counters["skip_insufficient_history"] += 1
                    continue

                # Parse confidence robustly
                conf_raw = t.get("confidence") or "0"
                try:
                    conf_int = int(float(conf_raw))
                except (TypeError, ValueError):
                    conf_int = 0

                ml_prob_raw = t.get("ml_prob")
                try:
                    ml_prob = float(ml_prob_raw) if ml_prob_raw else 0.0
                except (TypeError, ValueError):
                    ml_prob = 0.0

                features = build_phase1_features(
                    candles_df=df_window,
                    symbol=sym,
                    side=t["side"],
                    emitted_at=opened,
                    grade=t.get("grade"),
                    ml_prob=ml_prob,
                    confidence=conf_int,
                    regime=t.get("regime"),
                    scanner_type=t.get("scanner_type"),
                )

                # Sanity check: features contain expected keys
                if set(features.keys()) != set(feature_names()):
                    counters["skip_bad_features"] += 1
                    continue

                # Score with PPP if available
                ppp_score = None
                ppp_reason = None
                if ppp_gate is not None:
                    ppp_score, ppp_reason = ppp_gate.predict(features)

                if not dry_run:
                    t_with_parsed = {**t, "confidence": str(conf_int), "ml_prob": str(ml_prob)}
                    await insert_backfilled_row(
                        conn, t_with_parsed, features,
                        ppp_score=ppp_score,
                        ppp_threshold=ppp_threshold,
                        ppp_reason=ppp_reason,
                    )
                counters["inserted"] += 1
                if ppp_score is not None:
                    counters[f"ppp_{'admit' if ppp_score >= (ppp_threshold or 0.5) else 'reject'}"] += 1

            except Exception as e:
                counters[f"err_{type(e).__name__}"] += 1
                logger.debug("trade %s error: %s", t["id"], e)

        # Report
        logger.info("=" * 60)
        logger.info("BACKFILL RESULT (%s):", "DRY-RUN" if dry_run else "COMMITTED")
        for k, v in sorted(counters.items(), key=lambda kv: -kv[1]):
            logger.info("  %-30s %d", k, v)

        # Verify
        if not dry_run:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM signal_features WHERE label_captured = TRUE"
            )
            positives = await conn.fetchval(
                "SELECT COUNT(*) FROM signal_features WHERE label_captured = TRUE AND will_peak_30r = TRUE"
            )
            logger.info("=" * 60)
            logger.info("Total labeled rows in signal_features: %d", total)
            logger.info("  of which positives (peak ≥ 0.30R):   %d (%.1f%%)",
                        positives, 100 * positives / max(total, 1))
    finally:
        await conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30)")
    ap.add_argument("--symbol", type=str, default=None, help="Filter to a single symbol (e.g. BTC/USDT)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be inserted without writing")
    args = ap.parse_args()
    asyncio.run(main(days=args.days, symbol=args.symbol, dry_run=args.dry_run))

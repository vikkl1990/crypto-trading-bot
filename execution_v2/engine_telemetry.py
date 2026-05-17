"""Per-tick evaluation telemetry — gives forward visibility into engine state.

Each paper engine tick should call `log_eval()` once per symbol with the
current detector state — even when no signal fires. This creates a per-engine
eval_trace.jsonl that lets us see HOW CLOSE engines are to firing.

Without this, an engine that returns None on every tick gives zero visibility
into whether it's broken (always None) or correctly waiting (None because
conditions don't match).

Usage in an engine's eval_signals() loop:

    from execution_v2.engine_telemetry import log_eval

    for sym in SYMBOLS:
        df = fetch_candles(sym)
        if df.empty:
            continue
        # ... compute indicators ...

        sig = detect_signal(df, sym)

        log_eval(
            engine_name="scalper_vwap_mr",
            symbol=sym,
            base_dir=Path("/home/opc/crypto-trading-bot/storage/scalper_vwap_mr_paper"),
            signal_fired=(sig is not None),
            state={
                "close": float(last.close),
                "vwap": float(last.vwap),
                "k_dev": float(last.k_dev),
                "atr_pct_rank": float(last.atr_rank),
                "bull_candle": bool(...),
                "bear_candle": bool(...),
                "rejection_reason": "atr_pct_rank>0.4_NOT_RANGING" if not sig else None,
            },
        )

The rejection_reason is the single most important field — it tells us why
the signal didn't fire. Use a short ALL_CAPS_TAG format for grep-ability.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def log_eval(engine_name: str, symbol: str, base_dir: Path,
             signal_fired: bool, state: Dict[str, Any]) -> None:
    """Append a single eval record to base_dir/eval_trace.jsonl.

    File grows ~1 line per (engine × symbol × tick) — at 5min cron and 4 symbols
    that's 1152 lines/day per engine. Keeps last 14 days = ~16k lines.
    Pruning is the operator's responsibility (dashboard / housekeeping cron).
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "engine": engine_name,
        "symbol": symbol,
        "signal_fired": bool(signal_fired),
        **state,
    }
    try:
        with (base / "eval_trace.jsonl").open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Telemetry is best-effort — never crash an engine over logging
        pass


def latest_state(base_dir: Path, symbol: str = None) -> Dict[str, Any] | None:
    """Read most recent eval record for an engine (optionally per-symbol).

    Used by dashboard to show the current near-miss state of each engine.
    """
    base = Path(base_dir)
    f = base / "eval_trace.jsonl"
    if not f.exists():
        return None
    last = None
    try:
        with f.open() as h:
            for line in h:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if symbol and rec.get("symbol") != symbol:
                    continue
                last = rec
    except Exception:
        return None
    return last


def per_symbol_latest(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Most recent eval state per symbol — for dashboard display."""
    base = Path(base_dir)
    f = base / "eval_trace.jsonl"
    if not f.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with f.open() as h:
            for line in h:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sym = rec.get("symbol")
                if sym:
                    out[sym] = rec
    except Exception:
        return out
    return out


def rejection_summary(base_dir: Path, last_n_records: int = 100) -> Dict[str, int]:
    """Count rejection_reason occurrences in the last N tick records.

    Useful for understanding which gate is firing most often.
    """
    base = Path(base_dir)
    f = base / "eval_trace.jsonl"
    if not f.exists():
        return {}
    counts: Dict[str, int] = {}
    try:
        with f.open() as h:
            lines = h.readlines()[-last_n_records:]
            for line in lines:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("signal_fired"):
                    counts["__FIRED__"] = counts.get("__FIRED__", 0) + 1
                else:
                    reason = str(rec.get("rejection_reason") or "UNSPECIFIED")
                    counts[reason] = counts.get(reason, 0) + 1
    except Exception:
        return counts
    return counts


if __name__ == "__main__":
    # Smoke test
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Log a few eval ticks
        log_eval("test_engine", "BTC/USDT", base, signal_fired=False,
                 state={"k_dev": -3.13, "atr_pct_rank": 1.00,
                        "rejection_reason": "ATR_PCT_HIGH_NOT_RANGING"})
        log_eval("test_engine", "BTC/USDT", base, signal_fired=False,
                 state={"k_dev": -2.50, "atr_pct_rank": 0.80,
                        "rejection_reason": "ATR_PCT_HIGH_NOT_RANGING"})
        log_eval("test_engine", "ETH/USDT", base, signal_fired=False,
                 state={"k_dev": -1.20, "atr_pct_rank": 0.30,
                        "rejection_reason": "K_DEV_INSUFFICIENT"})
        log_eval("test_engine", "ETH/USDT", base, signal_fired=True,
                 state={"k_dev": -2.30, "atr_pct_rank": 0.35})

        print("Latest BTC/USDT state:")
        print(f"  {latest_state(base, 'BTC/USDT')}")

        print("\nPer-symbol latest:")
        for sym, rec in per_symbol_latest(base).items():
            print(f"  {sym}: signal_fired={rec['signal_fired']} reason={rec.get('rejection_reason')}")

        print("\nRejection summary:")
        for reason, n in rejection_summary(base).items():
            print(f"  {n:4d} × {reason}")

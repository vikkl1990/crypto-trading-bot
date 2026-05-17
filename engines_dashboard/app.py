"""
Engines Dashboard — VM2 (158.101.112.94:8082)

Read-only Flask app that monitors the new engines (S5, SMC1, SMC1.5, SMC1.5v2)
running paper-shadow on VM1. State files are pulled by sync.sh every minute.

This app NEVER modifies VM1 — it only reads local files in ./data/ that were
rsync'd in.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import markdown
import pandas as pd
from flask import Flask, abort, render_template, url_for

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

app = Flask(__name__, static_folder=str(BASE / "static"), template_folder=str(BASE / "templates"))

# ---------------------------------------------------------------------------
# Engine registry — strategy names + 1-line descriptions + backtest baselines
# ---------------------------------------------------------------------------

ENGINES: dict[str, dict[str, Any]] = {
    "s5": {
        "name": "S5",
        "full_name": "4h Range Fade",
        "tag": "range_fade_4h",
        "desc": "Fade 4h range edges after 3+ touches; structural mean-reversion play.",
        "backtest": {
            "n": 71,
            "wr": 0.45,
            "ev_per_trade": 7.80,
            "net": 554.0,
            "summary": "71 trades, 45% WR, +$554 net, +$7.80/trade EV after taker fees.",
            "wf_status": "NO_WF",
        },
        "paper_dir": "s5_paper",
        "color": "info",
    },
    "smc1": {
        "name": "SMC1",
        "full_name": "1h Order Block + FVG retest",
        "tag": "smc1_ob_fvg",
        "desc": "1h OB + FVG retest after displacement; original SMC implementation.",
        "backtest": {
            "n": 40,
            "wr": 0.55,
            "ev_per_trade": 1.60,
            "net": 64.0,
            "summary": "n=40, 55% WR, +$1.60 net EV/trade. Forward-validate before drawing conclusions.",
            "wf_status": "NO_WF",
        },
        "paper_dir": "smc1_paper",
        "color": "purple",
    },
    "smc15": {
        "name": "SMC1.5",
        "full_name": "Fullstack Sweep+BOS+CHoCH+OB",
        "tag": "smc15_fullstack",
        "desc": "Sweep → BOS → CHoCH → OB/BRK/RB; 3.2x EV vs SMC1 by stricter gating.",
        "backtest": {
            "n": 89,
            "wr": 0.607,
            "ev_per_trade": 5.15,
            "net": 458.64,
            "summary": "Best cell EB: n=89, 60.7% WR, +$5.15 EV/trade. All exit cfgs SHIP.",
            "wf_status": "NO_WF",
        },
        "paper_dir": "smc15_paper",
        "color": "success",
    },
    "smc15v2": {
        "name": "SMC1.5v2",
        "full_name": "Wick-entry LIMIT variant",
        "tag": "smc15_wick_limit",
        "desc": "Limit order at OB extreme (maker-fee entry); +20% EV vs SMC1.5 in OOS Q4.",
        "backtest": {
            "n": 64,
            "wr": 0.859,
            "ev_per_trade": 9.88,
            "net": 632.4,
            "summary": "Walk-forward PASS. n=64, 85.9% WR, +$632 net. OOS Q4 +20% over SMC1.5.",
            "wf_status": "PASSED",
            "wf_detail": {
                "IS_Q1Q2": {"n": 36, "wr": 0.861, "ev": 11.34},
                "OOS_Q3":  {"n": 3,  "wr": 1.000, "ev": 23.21},
                "OOS_Q4":  {"n": 25, "wr": 0.840, "ev": 6.18},
            },
        },
        "paper_dir": "smc15v2_paper",
        "color": "warning",
    },
}

# Queued / dark code (W/F passed but holding for live promotion)
QUEUED: list[dict[str, Any]] = [
    {
        "name": "HTF_HARD_VETO_AGRADE",
        "status": "WF_PASSED_DARK",
        "desc": "HTF hard-veto for A-grade signals. Walk-forward passed; not yet wired live.",
        "report_dir": "scanner_refinements",
    },
    {
        "name": "MACRO_EMA200_VETO",
        "status": "HOLD_DARK",
        "desc": "EMA200 daily macro filter — held in shadow pending more data.",
        "report_dir": "ema200_data",
    },
    {
        "name": "MACD_DIV_VETO",
        "status": "PENDING_WF",
        "desc": "MACD divergence veto — backtest done, walk-forward pending.",
        "report_dir": "macd_div",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _safe_read_text(p: Path) -> str | None:
    try:
        return p.read_text()
    except Exception:
        return None


def _read_trades(p: Path, limit: int | None = None) -> list[dict]:
    trades: list[dict] = []
    if not p.exists():
        return trades
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return trades
    if limit:
        trades = trades[-limit:]
    return trades


def _trade_stats(trades: list[dict]) -> dict[str, Any]:
    closed = [t for t in trades if t.get("status") == "closed"]
    n = len(closed)
    wins = sum(1 for t in closed if (t.get("net_pnl_usd") or 0) > 0)
    net = sum((t.get("net_pnl_usd") or 0) for t in closed)
    avg = (net / n) if n else 0.0
    wr = (wins / n) if n else 0.0
    return {
        "n_closed": n,
        "wins": wins,
        "wr": wr,
        "net_pnl": net,
        "avg_per_trade": avg,
    }


def _days_running(trades: list[dict]) -> float | None:
    if not trades:
        return None
    timestamps = []
    for t in trades:
        ts = t.get("opened_at") or t.get("candle_time")
        if not ts:
            continue
        try:
            timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        except Exception:
            continue
    if not timestamps:
        return None
    oldest = min(timestamps)
    now = datetime.now(timezone.utc)
    return (now - oldest).total_seconds() / 86400.0


def get_engine_state(engine_key: str) -> dict[str, Any]:
    """Build full engine view: state.json + trades.jsonl + backtest baseline + comparison."""
    cfg = ENGINES[engine_key]
    paper = DATA / cfg["paper_dir"]

    state = _safe_read_json(paper / "state.json") or {}
    trades = _read_trades(paper / "trades.jsonl")
    latest_md = _safe_read_text(paper / "latest.md")

    stats = _trade_stats(trades)

    # Fall back to state.json fields when trades.jsonl missing
    if stats["n_closed"] == 0 and "history_n" in state:
        stats["n_closed"] = state.get("history_n", 0) or 0
        stats["wins"] = state.get("history_wins", 0) or 0
        stats["net_pnl"] = state.get("history_pnl", 0.0) or 0.0
        if stats["n_closed"]:
            stats["wr"] = stats["wins"] / stats["n_closed"]
            stats["avg_per_trade"] = stats["net_pnl"] / stats["n_closed"]

    open_positions = state.get("open_trades", []) or []
    days = _days_running(trades) if trades else None

    # Compare forward EV/trade vs backtest EV/trade
    bt_ev = cfg["backtest"]["ev_per_trade"]
    fwd_ev = stats["avg_per_trade"]
    gap_pct = None
    if stats["n_closed"] >= 1 and bt_ev:
        gap_pct = (fwd_ev / bt_ev) * 100.0

    return {
        "key": engine_key,
        "cfg": cfg,
        "state": state,
        "stats": stats,
        "open_positions": open_positions,
        "trades": trades,
        "n_open": len(open_positions),
        "days_running": days,
        "latest_md_html": markdown.markdown(latest_md, extensions=["tables", "fenced_code"]) if latest_md else None,
        "fwd_vs_bt_pct": gap_pct,
    }


def list_backtest_reports() -> list[dict[str, Any]]:
    """Discover all .md / .csv / .json reports under each backtest dir."""
    out: list[dict[str, Any]] = []
    dirs = ["smc15", "smc15_variants", "scalp_research", "scanner_refinements", "ema200_data", "macd_div", "post_exit"]
    for d in dirs:
        root = DATA / d
        if not root.exists():
            out.append({"dir": d, "files": [], "missing": True})
            continue
        files: list[dict[str, Any]] = []
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix in (".md", ".csv", ".json"):
                files.append({
                    "name": p.name,
                    "path": f"{d}/{p.name}",
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "ext": p.suffix.lstrip("."),
                })
        out.append({"dir": d, "files": files, "missing": False})
    return out


def last_sync_str() -> str:
    p = DATA / ".last_sync"
    if not p.exists():
        return "never"
    return p.read_text().strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def overview():
    cards = [get_engine_state(k) for k in ENGINES]
    return render_template(
        "index.html",
        engines=cards,
        queued=QUEUED,
        last_sync=last_sync_str(),
    )


@app.route("/engine/<name>")
def engine_detail(name: str):
    if name not in ENGINES:
        abort(404)
    view = get_engine_state(name)
    # Last 50 trades, newest first
    last50 = sorted(view["trades"], key=lambda t: t.get("opened_at") or "", reverse=True)[:50]
    view["last50"] = last50
    return render_template("engine_detail.html", view=view, last_sync=last_sync_str())


@app.route("/backtest")
def backtest_browser():
    reports = list_backtest_reports()
    return render_template("backtest.html", reports=reports, last_sync=last_sync_str())


@app.route("/backtest/<path:relpath>")
def backtest_file(relpath: str):
    # Security: only allow paths under DATA, no traversal
    safe = (DATA / relpath).resolve()
    try:
        safe.relative_to(DATA.resolve())
    except ValueError:
        abort(403)
    if not safe.exists() or not safe.is_file():
        abort(404)

    ext = safe.suffix.lower()
    raw = _safe_read_text(safe) or ""

    if ext == ".md":
        body_html = markdown.markdown(raw, extensions=["tables", "fenced_code"])
        return render_template("report_view.html", title=safe.name, body_html=body_html, raw=None, kind="md", last_sync=last_sync_str())
    elif ext == ".csv":
        try:
            df = pd.read_csv(safe, nrows=500)
            table_html = df.to_html(classes="data-table", index=False, border=0)
        except Exception as e:
            table_html = f"<pre>error parsing csv: {e}</pre>"
        return render_template("report_view.html", title=safe.name, body_html=table_html, raw=None, kind="csv", last_sync=last_sync_str())
    elif ext == ".json":
        try:
            obj = json.loads(raw)
            pretty = json.dumps(obj, indent=2)
        except Exception:
            pretty = raw
        return render_template("report_view.html", title=safe.name, body_html=None, raw=pretty, kind="json", last_sync=last_sync_str())

    abort(415)




# ---------------------------------------------------------------------------
# Shadow Engines (execution_v2 — fill-aware engines)
# ---------------------------------------------------------------------------

def _read_jsonl(p: Path, limit: int = 5000) -> list:
    if not p.exists():
        return []
    rows = []
    try:
        with p.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
                if len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


def _shadow_engine_summary(engine_dir: Path) -> dict:
    """Compute execution_v2 summary for a single engine directory."""
    candidates = _read_jsonl(engine_dir / "candidates.jsonl")
    trades = _read_jsonl(engine_dir / "trades.jsonl")
    name = engine_dir.name.replace("_paper", "")

    n_candidates = len(candidates)
    n_filled = sum(1 for c in candidates if c.get("filled"))
    n_submitted = sum(1 for c in candidates if c.get("submitted"))
    n_ev_rejected = sum(1 for c in candidates if not c.get("ev_gate_passed", True))
    fill_rate = (n_filled / n_submitted) if n_submitted > 0 else 0.0

    n_trades = len(trades)
    wins = sum(1 for t in trades if (t.get("net_pnl_usd") or 0) > 0)
    net_pnl = sum(t.get("net_pnl_usd") or 0 for t in trades)
    gross_pnl = sum(t.get("gross_pnl_usd") or 0 for t in trades)
    fees = sum(t.get("fees_usd_total") or 0 for t in trades)
    maker_n = sum(1 for t in trades if t.get("entry_fee_type") == "maker")
    maker_pct = (maker_n / n_trades) if n_trades > 0 else 0.0
    fill_times = [t.get("time_to_fill_sec") for t in trades if t.get("time_to_fill_sec")]
    avg_ttf = sum(fill_times) / len(fill_times) if fill_times else 0.0
    win_rate = (wins / n_trades) if n_trades > 0 else 0.0
    ev_per_trade = (net_pnl / n_trades) if n_trades > 0 else 0.0

    # Pass / Hold / Kill verdict per spec section 10
    if n_trades >= 50 and ev_per_trade > 0.10 and maker_pct > 0.40:
        verdict = "PASS"
    elif n_trades >= 30 and (ev_per_trade < 0 or maker_pct < 0.20):
        verdict = "KILL"
    elif n_trades < 50:
        verdict = "HOLD"
    else:
        verdict = "HOLD"

    return {
        "name": name,
        "candidates_seen": n_candidates,
        "submitted": n_submitted,
        "filled": n_filled,
        "fill_rate": fill_rate,
        "ev_rejected": n_ev_rejected,
        "closed_trades": n_trades,
        "wins": wins,
        "win_rate": win_rate,
        "net_pnl_usd": net_pnl,
        "gross_pnl_usd": gross_pnl,
        "fees_usd_total": fees,
        "ev_per_trade": ev_per_trade,
        "maker_pct": maker_pct,
        "avg_time_to_fill_sec": avg_ttf,
        "verdict": verdict,
    }


def list_shadow_engines() -> list:
    """Auto-discover engines that have candidates.jsonl under data/."""
    out = []
    if not DATA.exists():
        return out
    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "candidates.jsonl").exists():
            out.append(_shadow_engine_summary(sub))
    return out


@app.route("/shadow-engines")
def shadow_engines_view():
    engines = list_shadow_engines()
    return render_template(
        "shadow_engines.html",
        engines=engines,
        last_sync=last_sync_str(),
    )


@app.errorhandler(404)
def err_404(_e):
    return render_template("error.html", code=404, msg="Not found", last_sync=last_sync_str()), 404


@app.errorhandler(500)
def err_500(_e):
    return render_template("error.html", code=500, msg="Internal error", last_sync=last_sync_str()), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8083))
    app.run(host="0.0.0.0", port=port, debug=False)

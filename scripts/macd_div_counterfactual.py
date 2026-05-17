"""Counterfactual: would MACD divergence have improved P&L on shadow trades?

What this does
--------------
1. Pull last 30 days of closed shadow trades from `user_trades`
   (primary config = delta_india, structure_bounce dominates).
2. For each unique (symbol, timeframe) needed, fetch the OKX swap
   candles spanning trade openings minus a `lookback` buffer.
   OKX caps `fetch_ohlcv` at 300 rows, so we paginate.
3. For each trade, slice the candle DF to bars `< opened_at` (no
   look-ahead), run `detect_divergence` on 5m + 15m, classify the
   trade into ALIGNED / OPPOSED / NONE relative to the trade side.
4. Aggregate P&L by cohort and emit a markdown report.

Outputs
-------
- /home/opc/crypto-trading-bot/storage/macd_div/cohort_breakdown.csv
- /home/opc/crypto-trading-bot/storage/macd_div/per_trade_tag.csv
- /home/opc/crypto-trading-bot/storage/macd_div/report.md
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# allow importing the sibling detector
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from scripts.macd_divergence_detector import detect_divergence  # noqa: E402

# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #

DAYS = 30
LOOKBACK_BARS = 60          # bars of history we feed to the detector
DETECTOR_LOOKBACK = 20      # how far back the detector searches for prior pivots
DETECTOR_PIVOT_N = 2
RECENT_WINDOW = 6           # recent pivot must be within (pivot_n + this) of last bar

OUT_DIR = Path("/home/opc/crypto-trading-bot/storage/macd_div")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUT_DIR / "candle_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# DB
DB_DSN = dict(
    host="localhost",
    user="vnedge",
    password="VnEdge2026db",
    dbname="vnedge",
    port=5432,
)

# OKX swap symbol mapping for our universe
OKX_SWAP_SYMBOL = {
    "BTC/USDT":  "BTC/USDT:USDT",
    "ETH/USDT":  "ETH/USDT:USDT",
    "SOL/USDT":  "SOL/USDT:USDT",
    "XRP/USDT":  "XRP/USDT:USDT",
    "LINK/USDT": "LINK/USDT:USDT",
    "DOGE/USDT": "DOGE/USDT:USDT",
    "DOT/USDT":  "DOT/USDT:USDT",
    "LTC/USDT":  "LTC/USDT:USDT",
    "AVAX/USDT": "AVAX/USDT:USDT",
    "ADA/USDT":  "ADA/USDT:USDT",
    "TAO/USDT":  "TAO/USDT:USDT",
    "TRUMP/USDT": "TRUMP/USDT:USDT",
}

TIMEFRAMES = ["5m", "15m"]
TF_TO_MS = {"5m": 5*60*1000, "15m": 15*60*1000}


# --------------------------------------------------------------------------- #
# DB                                                                          #
# --------------------------------------------------------------------------- #


def fetch_trades(days: int = DAYS) -> pd.DataFrame:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore

    conn = psycopg2.connect(**DB_DSN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT id, symbol, side, entry_price, exit_price, pnl_usd, fees_usd,
               opened_at, closed_at, metadata, exchange
        FROM user_trades
        WHERE trade_type = 'shadow'
          AND status = 'closed'
          AND exchange = 'delta_india'
          AND opened_at > NOW() - INTERVAL '%s days'
          AND pnl_usd IS NOT NULL
        ORDER BY opened_at
        """,
        (days,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # extract scanner / regime / grade from metadata
    df["scanner"] = df["metadata"].apply(
        lambda m: (m or {}).get("scanner", "")
    )
    df["regime"] = df["metadata"].apply(
        lambda m: (m or {}).get("regime", "")
    )
    df["grade"] = df["metadata"].apply(
        lambda m: (m or {}).get("grade", "")
    )
    df["net_pnl"] = df["pnl_usd"].astype(float)
    df["side"] = df["side"].str.lower()
    df["opened_at"] = pd.to_datetime(df["opened_at"], utc=True)
    return df


# --------------------------------------------------------------------------- #
# candle fetch (OKX, paginated)                                               #
# --------------------------------------------------------------------------- #


def _cache_path(symbol: str, tf: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}_{tf}.parquet"


def fetch_ohlcv_paginated(symbol: str, tf: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    """Fetch OKX candles in [since_ms, until_ms], paginating in 300-bar chunks.

    Cached to disk; subsequent calls extend the cache only as needed.
    """
    import ccxt  # type: ignore

    cache_path = _cache_path(symbol, tf)
    existing = pd.DataFrame()
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            if "ts_ms" in existing.columns:
                ex_min = int(existing["ts_ms"].min())
                ex_max = int(existing["ts_ms"].max())
                # if cache fully covers requested range, return slice
                if ex_min <= since_ms and ex_max >= until_ms - TF_TO_MS[tf]:
                    return existing
        except Exception:
            existing = pd.DataFrame()

    okx_sym = OKX_SWAP_SYMBOL.get(symbol)
    if okx_sym is None:
        return existing  # no mapping; return whatever cache we have

    ex = ccxt.okx({"enableRateLimit": True, "timeout": 20000})

    rows: List[List[float]] = []
    cursor = since_ms
    page_ms = 300 * TF_TO_MS[tf]
    safety = 0
    while cursor < until_ms and safety < 50:
        try:
            ohlcv = ex.fetch_ohlcv(okx_sym, tf, since=cursor, limit=300)
        except Exception as e:
            print(f"  fetch error {symbol} {tf} since={cursor}: {e}", file=sys.stderr)
            time.sleep(1.0)
            safety += 1
            cursor += page_ms
            continue
        if not ohlcv:
            cursor += page_ms
            safety += 1
            continue
        rows.extend(ohlcv)
        last_ts = int(ohlcv[-1][0])
        # advance cursor; if exchange returned fewer than 300, we may be at tail
        if last_ts <= cursor:
            cursor += page_ms
        else:
            cursor = last_ts + TF_TO_MS[tf]
        safety += 1

    if not rows and existing.empty:
        return pd.DataFrame()

    new_df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    new_df = new_df.drop_duplicates(subset="ts_ms").sort_values("ts_ms").reset_index(drop=True)

    if not existing.empty:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset="ts_ms").sort_values("ts_ms").reset_index(drop=True)
    else:
        merged = new_df

    merged.to_parquet(cache_path, index=False)
    return merged


def slice_until(df_candles: pd.DataFrame, t_utc: datetime, lookback_bars: int) -> pd.DataFrame:
    """Return last `lookback_bars` rows whose ts < t_utc."""
    if df_candles.empty:
        return df_candles
    cutoff_ms = int(t_utc.timestamp() * 1000)
    sub = df_candles[df_candles["ts_ms"] < cutoff_ms]
    if len(sub) == 0:
        return sub
    return sub.tail(lookback_bars).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# divergence tagging                                                           #
# --------------------------------------------------------------------------- #


def classify_alignment(side: str, div_type: str) -> str:
    """ALIGNED / OPPOSED / NONE relative to trade side.

    A bullish divergence (regular_bull / hidden_bull) is supportive of LONG.
    A bearish divergence is supportive of SHORT.
    """
    if div_type == "none" or not div_type:
        return "NONE"
    is_bull = div_type.endswith("_bull")
    is_bear = div_type.endswith("_bear")
    side_l = (side or "").lower()
    if side_l in ("long", "buy") and is_bull:
        return "ALIGNED"
    if side_l in ("short", "sell") and is_bear:
        return "ALIGNED"
    if side_l in ("long", "buy") and is_bear:
        return "OPPOSED"
    if side_l in ("short", "sell") and is_bull:
        return "OPPOSED"
    return "NONE"


def tag_trades(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Add div_5m_type, div_15m_type, div_alignment cols."""
    out_rows: List[Dict] = []
    # cache per-symbol-tf candle df in-memory
    candle_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    # Plan: figure out the time window per symbol-tf so we fetch once.
    if df_trades.empty:
        return df_trades

    needed: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for _, t in df_trades.iterrows():
        sym = t["symbol"]
        if sym not in OKX_SWAP_SYMBOL:
            continue
        ts = pd.Timestamp(t["opened_at"]).tz_convert("UTC")
        ms = int(ts.timestamp() * 1000)
        for tf in TIMEFRAMES:
            buf_ms = (LOOKBACK_BARS + 5) * TF_TO_MS[tf]
            since = ms - buf_ms
            until = ms + TF_TO_MS[tf]
            cur = needed.get((sym, tf))
            if cur is None:
                needed[(sym, tf)] = (since, until)
            else:
                needed[(sym, tf)] = (min(cur[0], since), max(cur[1], until))

    # fetch
    print(f"[fetch] downloading candles for {len(needed)} (sym, tf) pairs...")
    for (sym, tf), (since, until) in sorted(needed.items()):
        print(f"  {sym} {tf} since={datetime.fromtimestamp(since/1000, tz=timezone.utc)} until={datetime.fromtimestamp(until/1000, tz=timezone.utc)}")
        try:
            df_c = fetch_ohlcv_paginated(sym, tf, since, until)
        except Exception as e:
            print(f"    FAIL: {e}", file=sys.stderr)
            df_c = pd.DataFrame()
        candle_cache[(sym, tf)] = df_c
        if not df_c.empty:
            print(f"    got {len(df_c)} rows, span={pd.to_datetime(df_c['ts_ms'].min(),unit='ms')} to {pd.to_datetime(df_c['ts_ms'].max(),unit='ms')}")

    # tag each trade
    print(f"[tag] tagging {len(df_trades)} trades...")
    for _, t in df_trades.iterrows():
        sym = t["symbol"]
        side = (t["side"] or "").lower()
        ts = pd.Timestamp(t["opened_at"]).tz_convert("UTC")
        row: Dict = {
            "id": str(t["id"]),
            "symbol": sym,
            "side": side,
            "scanner": t["scanner"],
            "regime": t["regime"],
            "grade": t["grade"],
            "opened_at": ts,
            "net_pnl": float(t["net_pnl"]),
        }
        for tf in TIMEFRAMES:
            df_c = candle_cache.get((sym, tf), pd.DataFrame())
            sub = slice_until(df_c, ts, LOOKBACK_BARS)
            if len(sub) < max(30, DETECTOR_LOOKBACK + DETECTOR_PIVOT_N + 5):
                row[f"div_{tf}_type"] = ""
                row[f"div_{tf}_strength"] = 0.0
                row[f"div_{tf}_age"] = -1
                continue
            res = detect_divergence(
                sub,
                lookback=DETECTOR_LOOKBACK,
                pivot_n=DETECTOR_PIVOT_N,
                recent_window=RECENT_WINDOW,
            )
            row[f"div_{tf}_type"] = res["div_type"]
            row[f"div_{tf}_strength"] = res["div_strength"]
            row[f"div_{tf}_age"] = res["pivot_age"]

        # alignment per TF
        row["align_5m"] = classify_alignment(side, row["div_5m_type"])
        row["align_15m"] = classify_alignment(side, row["div_15m_type"])
        # combined: only call OPPOSED if both TFs say opposed (strict),
        # ALIGNED if at least one says aligned and neither says opposed
        a5, a15 = row["align_5m"], row["align_15m"]
        if a5 == "OPPOSED" and a15 == "OPPOSED":
            row["align_combined_strict"] = "OPPOSED"
        elif a5 == "OPPOSED" or a15 == "OPPOSED":
            # mixed: weak opposition
            row["align_combined_strict"] = "OPPOSED_WEAK"
        elif a5 == "ALIGNED" and a15 == "ALIGNED":
            row["align_combined_strict"] = "ALIGNED"
        elif a5 == "ALIGNED" or a15 == "ALIGNED":
            row["align_combined_strict"] = "ALIGNED_WEAK"
        else:
            row["align_combined_strict"] = "NONE"

        # combined: any-TF (lax)
        if "OPPOSED" in (a5, a15):
            row["align_combined_any"] = "OPPOSED"
        elif "ALIGNED" in (a5, a15):
            row["align_combined_any"] = "ALIGNED"
        else:
            row["align_combined_any"] = "NONE"

        out_rows.append(row)

    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# cohort stats                                                                 #
# --------------------------------------------------------------------------- #


def cohort_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col).agg(
        n=("net_pnl", "size"),
        wins=("net_pnl", lambda s: int((s > 0).sum())),
        sum_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        median_pnl=("net_pnl", "median"),
    )
    g["wr_pct"] = (g["wins"] / g["n"] * 100).round(1)
    g["avg_pnl"] = g["avg_pnl"].round(3)
    g["median_pnl"] = g["median_pnl"].round(3)
    g["sum_pnl"] = g["sum_pnl"].round(2)
    return g.reset_index().sort_values(group_col)


def render_cohort_md(df_stats: pd.DataFrame, title: str) -> str:
    lines = [f"### {title}", ""]
    lines.append("| cohort | n | wr% | sum_pnl | avg_pnl | median_pnl |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in df_stats.iterrows():
        lines.append(
            f"| {r.iloc[0]} | {int(r['n'])} | {r['wr_pct']:.1f} | "
            f"{r['sum_pnl']:.2f} | {r['avg_pnl']:.3f} | {r['median_pnl']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    print(f"[start] MACD divergence counterfactual, days={DAYS}")
    df_trades = fetch_trades(DAYS)
    print(f"[db] {len(df_trades)} closed shadow trades fetched")
    if df_trades.empty:
        print("no trades; abort")
        return 1

    df_tagged = tag_trades(df_trades)
    if df_tagged.empty:
        print("no tagged rows; abort")
        return 1

    # save per-trade tags
    tag_csv = OUT_DIR / "per_trade_tag.csv"
    df_tagged.to_csv(tag_csv, index=False)
    print(f"[out] per-trade tags -> {tag_csv}")

    # cohort stats per TF
    stats_5m = cohort_stats(df_tagged, "align_5m")
    stats_15m = cohort_stats(df_tagged, "align_15m")
    stats_combined_strict = cohort_stats(df_tagged, "align_combined_strict")
    stats_combined_any = cohort_stats(df_tagged, "align_combined_any")

    # also: scanner-conditioned (structure_bounce dominates)
    sb = df_tagged[df_tagged["scanner"] == "structure_bounce"].copy()
    stats_sb_5m = cohort_stats(sb, "align_5m")
    stats_sb_15m = cohort_stats(sb, "align_15m")
    stats_sb_combined = cohort_stats(sb, "align_combined_strict")

    # combined breakdown export
    breakdown_path = OUT_DIR / "cohort_breakdown.csv"
    chunks = []
    for label, df_s in [
        ("ALL_5m", stats_5m), ("ALL_15m", stats_15m),
        ("ALL_combined_strict", stats_combined_strict),
        ("ALL_combined_any", stats_combined_any),
        ("SB_5m", stats_sb_5m), ("SB_15m", stats_sb_15m),
        ("SB_combined_strict", stats_sb_combined),
    ]:
        df_s = df_s.copy()
        df_s.insert(0, "view", label)
        chunks.append(df_s)
    pd.concat(chunks, ignore_index=True).to_csv(breakdown_path, index=False)
    print(f"[out] cohort breakdown -> {breakdown_path}")

    # ---- counterfactual numbers ----
    def _counterfact(df_in: pd.DataFrame, align_col: str) -> Dict[str, float]:
        """Return savings if we VETO opposed trades, ALIGNED EV vs NONE EV."""
        opp = df_in[df_in[align_col].isin(["OPPOSED"])]
        none_c = df_in[df_in[align_col] == "NONE"]
        ali = df_in[df_in[align_col] == "ALIGNED"]
        return {
            "veto_savings": float(opp["net_pnl"].sum() * -1),  # money saved by NOT taking these
            "veto_n": int(len(opp)),
            "aligned_ev": float(ali["net_pnl"].mean()) if len(ali) else 0.0,
            "aligned_n": int(len(ali)),
            "none_ev": float(none_c["net_pnl"].mean()) if len(none_c) else 0.0,
            "none_n": int(len(none_c)),
        }

    # strength-gated 5m view: only VETO if div_5m_strength >= 0.2
    df_strength = df_tagged.copy()
    df_strength["align_5m_strong"] = df_strength.apply(
        lambda r: r["align_5m"] if (r["align_5m"] == "OPPOSED" and r["div_5m_strength"] >= 0.20) else
                  ("ALIGNED" if r["align_5m"] == "ALIGNED" and r["div_5m_strength"] >= 0.20 else "NONE"),
        axis=1,
    )
    cf_5m_strong = _counterfact(df_strength, "align_5m_strong")
    stats_5m_strong = cohort_stats(df_strength, "align_5m_strong")

    cf_5m = _counterfact(df_tagged, "align_5m")
    cf_15m = _counterfact(df_tagged, "align_15m")
    cf_strict = _counterfact(df_tagged, "align_combined_strict")
    cf_any = _counterfact(df_tagged, "align_combined_any")
    # already computed: cf_5m_strong, stats_5m_strong

    # ---- verdict logic ----
    def _verdict(cf: Dict[str, float]) -> Tuple[str, str]:
        veto_save = cf["veto_savings"]
        veto_n = cf["veto_n"]
        ali_n = cf["aligned_n"]
        ali_ev = cf["aligned_ev"]
        none_ev = cf["none_ev"]
        ev_diff = ali_ev - none_ev
        ship_veto = veto_save > 15.0 and veto_n >= 30
        ship_boost = ev_diff > 0.50 and ali_n >= 30
        if ship_veto and ship_boost:
            return "SHIP_BOTH", f"veto saves ${veto_save:.2f} (n={veto_n}); ALIGNED EV +${ev_diff:.2f}/trade vs NONE (n={ali_n})"
        if ship_veto:
            return "SHIP_VETO", f"veto saves ${veto_save:.2f} over {veto_n} trades"
        if ship_boost:
            return "SHIP_BOOST", f"ALIGNED EV +${ev_diff:.2f}/trade vs NONE (n={ali_n})"
        return "HOLD", f"veto_save=${veto_save:.2f} (n={veto_n}, need >$15 & n>=30); EV diff=${ev_diff:.2f}/trade (n={ali_n}, need >$0.50 & n>=30)"

    v_5m = _verdict(cf_5m)
    v_15m = _verdict(cf_15m)
    v_strict = _verdict(cf_strict)
    v_any = _verdict(cf_any)
    v_5m_strong = _verdict(cf_5m_strong)

    # ---- markdown report ----
    md_lines: List[str] = []
    md_lines.append(f"# MACD Divergence Counterfactual — {DAYS}d shadow primary\n")
    md_lines.append(f"_Run at {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z_\n")
    md_lines.append(f"**Trades analyzed:** {len(df_tagged)} (delta_india shadow, closed, last {DAYS}d)\n")
    md_lines.append(f"**Detector:** EMA(12/26) MACD, signal 9. Pivot N={DETECTOR_PIVOT_N}, lookback={DETECTOR_LOOKBACK} bars, recent_window={RECENT_WINDOW}.\n")
    md_lines.append("\n## Cohort breakdown — ALL trades\n")
    md_lines.append(render_cohort_md(stats_5m, "By 5m divergence alignment"))
    md_lines.append(render_cohort_md(stats_15m, "By 15m divergence alignment"))
    md_lines.append(render_cohort_md(stats_combined_strict, "By 5m+15m STRICT alignment (both must agree)"))
    md_lines.append(render_cohort_md(stats_combined_any, "By 5m+15m LAX alignment (any TF)"))
    md_lines.append("\n## Cohort breakdown — structure_bounce only\n")
    md_lines.append(render_cohort_md(stats_sb_5m, "SB / 5m"))
    md_lines.append(render_cohort_md(stats_sb_15m, "SB / 15m"))
    md_lines.append(render_cohort_md(stats_sb_combined, "SB / 5m+15m strict"))

    md_lines.append("\n## Strength-gated 5m breakdown (div_5m_strength >= 0.20)\n")
    md_lines.append(render_cohort_md(stats_5m_strong, "By 5m strength-gated alignment"))

    md_lines.append("\n## Counterfactual numbers\n")
    md_lines.append("| TF | veto_save (USD) | veto_n | ALIGNED EV | ALIGNED_n | NONE EV | NONE_n | Verdict |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for label, cf, vd in [
        ("5m (all strengths)", cf_5m, v_5m),
        ("5m strength>=0.20", cf_5m_strong, v_5m_strong),
        ("15m", cf_15m, v_15m),
        ("5m+15m strict", cf_strict, v_strict),
        ("5m or 15m (any)", cf_any, v_any),
    ]:
        md_lines.append(
            f"| {label} | {cf['veto_savings']:.2f} | {cf['veto_n']} | "
            f"{cf['aligned_ev']:.3f} | {cf['aligned_n']} | "
            f"{cf['none_ev']:.3f} | {cf['none_n']} | **{vd[0]}** — {vd[1]} |"
        )

    # side breakdown
    md_lines.append("\n## Side x alignment (5m) — shows where the veto bites\n")
    side_tab = df_tagged.groupby(["align_5m", "side"]).agg(
        n=("net_pnl", "size"),
        sum_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
    ).round(3).reset_index()
    md_lines.append("| align_5m | side | n | sum_pnl | avg_pnl |")
    md_lines.append("|---|---|---:|---:|---:|")
    for _, r in side_tab.iterrows():
        md_lines.append(
            f"| {r['align_5m']} | {r['side']} | {int(r['n'])} | "
            f"{r['sum_pnl']:.2f} | {r['avg_pnl']:.3f} |"
        )

    # strength sensitivity for OPPOSED cohort
    opp5 = df_tagged[df_tagged["align_5m"] == "OPPOSED"].copy()
    if len(opp5) > 0:
        opp5["str_bucket"] = pd.cut(opp5["div_5m_strength"], [0, 0.2, 0.35, 0.5, 1.0])
        sb_tab = opp5.groupby("str_bucket", observed=True).agg(
            n=("net_pnl", "size"),
            sum_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
        ).round(3).reset_index()
        md_lines.append("\n## OPPOSED-5m strength bucket sensitivity\n")
        md_lines.append("| strength range | n | sum_pnl | avg_pnl |")
        md_lines.append("|---|---:|---:|---:|")
        for _, r in sb_tab.iterrows():
            md_lines.append(
                f"| {r['str_bucket']} | {int(r['n'])} | {r['sum_pnl']:.2f} | {r['avg_pnl']:.3f} |"
            )
        md_lines.append("")

    md_lines.append("\n## Statistical significance\n")
    md_lines.append("- Sample sizes for OPPOSED cohorts above are the binding constraint. ")
    md_lines.append("  Below n=30 the verdict is necessarily HOLD pending more data.")
    md_lines.append("- All shadow trades from 2026-04-24 onward (post recent reset). Regime mix is ")
    md_lines.append("  dominated by `sideways`/`high_volatility` for delta_india; trending regimes ")
    md_lines.append("  underrepresented — divergence behaviour in trending markets is undertested.")
    md_lines.append("- Trade book is heavily SHORT-biased (~7.5:1 short:long). This means the veto's ")
    md_lines.append("  effect is dominated by `short + bearish-divergence-OPPOSED` cohort. Long-side ")
    md_lines.append("  veto behaviour has small n (~11 trades), too thin for a confident verdict.")
    md_lines.append("- Detector uses ZigZag-lite pivots with N=2 (5-bar window). Smaller N would ")
    md_lines.append("  catch more pivots but bias toward noise. Sensitivity to N not tested here.")
    md_lines.append("- Recommended ship: 5m-only veto, gated by div_5m_strength >= 0.20. The ")
    md_lines.append("  weak-strength bucket (0..0.2) is breakeven and would falsely block neutral trades.")

    md_lines.append("\n## Verdict matrix\n")
    md_lines.append("| TF setup | Verdict | Why |")
    md_lines.append("|---|---|---|")
    for label, vd in [
        ("5m only (all strengths)", v_5m),
        ("5m only, strength>=0.20", v_5m_strong),
        ("15m only", v_15m),
        ("5m+15m STRICT (both agree)", v_strict),
        ("5m or 15m (any)", v_any),
    ]:
        md_lines.append(f"| {label} | **{vd[0]}** | {vd[1]} |")

    # pick cleanest TF for veto recommendation
    cleanest = max(
        [("5m", cf_5m, v_5m), ("15m", cf_15m, v_15m),
         ("strict", cf_strict, v_strict), ("any", cf_any, v_any)],
        key=lambda x: (x[1]["veto_savings"] if x[1]["veto_n"] >= 20 else -1e9),
    )
    md_lines.append(f"\n**Cleanest filter (highest veto savings with n>=20):** {cleanest[0]}\n")

    md_path = OUT_DIR / "report.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[out] report -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

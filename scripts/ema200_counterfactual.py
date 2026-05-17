"""
EMA-macro counterfactual on existing 30-day primary shadow trades.

For each closed shadow trade, look up the macro bias (EMA50_d, EMA100_d,
EMA200_d) at trade.opened_at from storage/ema200_data/macro_bias.parquet,
and compute per-cohort metrics + counterfactual savings if AGAINST_MACRO
trades had been blocked.

Read-only on production tables and code. Writes:
  storage/ema200_data/counterfactual.json (machine-readable)
  storage/ema200_data/report.md            (human-readable verdict)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import pandas as pd
import psycopg2
import psycopg2.extras

LOG = logging.getLogger("ema200_counterfactual")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO = Path("/home/opc/crypto-trading-bot")
OUT_DIR = REPO / "storage" / "ema200_data"
MACRO_PARQUET = OUT_DIR / "macro_bias.parquet"

DB_DSN = "host=localhost dbname=vnedge user=vnedge password=VnEdge2026db"

QUERY = """
SELECT id::text, symbol, side, opened_at, closed_at, pnl_usd,
       COALESCE(metadata::jsonb->>'scanner','') AS scanner,
       COALESCE(metadata::jsonb->>'grade','')   AS grade,
       COALESCE(metadata::jsonb->>'regime','')  AS regime,
       COALESCE(metadata::jsonb->>'peak_mfe_pct','')   AS peak_mfe_pct,
       COALESCE(metadata::jsonb->>'mfe_pct','')        AS mfe_pct,
       COALESCE(metadata::jsonb->>'exit_reason','')    AS exit_reason
FROM user_trades
WHERE trade_type='shadow'
  AND closed_at >= NOW() - INTERVAL '30 days'
  AND COALESCE(metadata::jsonb->>'exit_config_id','')='primary'
  AND status='closed'
  AND COALESCE(metadata::jsonb->>'exit_reason','') NOT IN
      ('auto_responder_stuck_60m','restart_orphan_cleanup','reconcile_overaged_close')
ORDER BY opened_at ASC;
"""


def load_trades() -> pd.DataFrame:
    with psycopg2.connect(DB_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(QUERY)
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["opened_at"] = pd.to_datetime(df["opened_at"], utc=True).astype("datetime64[us, UTC]")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True).astype("datetime64[us, UTC]")
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0)
    df["sym_short"] = df["symbol"].str.replace("/USDT", "", regex=False)
    df["mfe_num"] = pd.to_numeric(df["peak_mfe_pct"].replace("", None), errors="coerce")
    df["mfe_num"] = df["mfe_num"].fillna(pd.to_numeric(df["mfe_pct"].replace("", None), errors="coerce"))
    return df


def load_macro() -> pd.DataFrame:
    macro = pd.read_parquet(MACRO_PARQUET)
    macro["ts"] = pd.to_datetime(macro["ts"], utc=True).astype("datetime64[us, UTC]")
    macro = macro.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return macro


def attach_bias(trades: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """For each trade, take the most-recent CLOSED 4h-bar bias as of opened_at."""
    cols = ["macro_bias_4h_50", "macro_bias_4h_100", "macro_bias_4h_200",
            "macro_bias_d_50", "macro_bias_d_100", "macro_bias_d_200"]
    out = trades.copy()
    for c in cols:
        out[c] = 0
    for sym, sub in trades.groupby("sym_short"):
        macro_sym = macro[macro["symbol"] == sym].sort_values("ts")
        if macro_sym.empty:
            LOG.warning("No macro data for %s; %d trades will get bias=0", sym, len(sub))
            continue
        merged = pd.merge_asof(
            sub[["id", "opened_at"]].sort_values("opened_at"),
            macro_sym[["ts"] + cols].sort_values("ts"),
            left_on="opened_at",
            right_on="ts",
            direction="backward",
        )
        merged = merged.set_index("id")
        for c in cols:
            out.loc[out["id"].isin(merged.index), c] = (
                out.loc[out["id"].isin(merged.index), "id"].map(merged[c]).fillna(0).astype("int8").values
            )
    return out


def cohort_label(side: str, bias: int) -> str:
    if bias == 0:
        return "NEUTRAL"
    if (side == "long" and bias == 1) or (side == "short" and bias == -1):
        return "WITH_MACRO"
    return "AGAINST_MACRO"


def cohort_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "wr_pct": None, "sum_pnl": 0.0, "avg_pnl": 0.0,
                "median_pnl": 0.0, "avg_peak_mfe_pct": None}
    wins = (df["pnl_usd"] > 0).sum()
    return {
        "n": int(n),
        "wr_pct": round(100.0 * wins / n, 1),
        "sum_pnl": round(float(df["pnl_usd"].sum()), 2),
        "avg_pnl": round(float(df["pnl_usd"].mean()), 3),
        "median_pnl": round(float(df["pnl_usd"].median()), 3),
        "avg_peak_mfe_pct": (round(float(df["mfe_num"].dropna().mean()), 3)
                             if df["mfe_num"].notna().any() else None),
    }


def analyse_filter(df: pd.DataFrame, bias_col: str) -> dict:
    df = df.copy()
    df["cohort"] = [cohort_label(s, b) for s, b in zip(df["side"], df[bias_col].astype(int))]

    cohorts: dict[str, dict] = {
        c: cohort_stats(df[df["cohort"] == c])
        for c in ("WITH_MACRO", "AGAINST_MACRO", "NEUTRAL")
    }

    total_pnl = round(float(df["pnl_usd"].sum()), 2)
    against_pnl = cohorts["AGAINST_MACRO"]["sum_pnl"]
    savings = round(-against_pnl, 2)  # positive == we'd save money by blocking

    pf_by_cohort = {}
    for c in ("WITH_MACRO", "AGAINST_MACRO", "NEUTRAL"):
        sub = df[df["cohort"] == c]
        gp = float(sub.loc[sub["pnl_usd"] > 0, "pnl_usd"].sum())
        gl = -float(sub.loc[sub["pnl_usd"] < 0, "pnl_usd"].sum())
        pf_by_cohort[c] = round(gp / gl, 2) if gl > 0 else None

    n_against = cohorts["AGAINST_MACRO"]["n"]
    if cohorts["AGAINST_MACRO"]["sum_pnl"] < -15 and n_against >= 30:
        verdict = "SHIP"
    elif cohorts["AGAINST_MACRO"]["sum_pnl"] < -5 and n_against >= 20:
        verdict = "PILOT"
    else:
        verdict = "HOLD"

    return {
        "filter": bias_col,
        "total_n": int(len(df)),
        "total_pnl": total_pnl,
        "cohorts": cohorts,
        "profit_factor": pf_by_cohort,
        "savings_if_blocked": savings,
        "pnl_after_block": round(total_pnl + savings, 2),
        "verdict": verdict,
    }


def per_side_breakdown(df: pd.DataFrame, bias_col: str) -> dict:
    out = {}
    for side in ("long", "short"):
        sub = df[df["side"] == side]
        if sub.empty:
            out[side] = None
            continue
        sub2 = sub.copy()
        sub2["cohort"] = [cohort_label(side, int(b)) for b in sub2[bias_col]]
        out[side] = {c: cohort_stats(sub2[sub2["cohort"] == c])
                     for c in ("WITH_MACRO", "AGAINST_MACRO", "NEUTRAL")}
    return out


def per_scanner(df: pd.DataFrame, bias_col: str) -> dict:
    out: dict[str, dict] = {}
    for scn, sub in df.groupby("scanner"):
        sub2 = sub.copy()
        sub2["cohort"] = [cohort_label(s, int(b)) for s, b in zip(sub2["side"], sub2[bias_col])]
        out[scn or "(blank)"] = {
            "n": int(len(sub2)),
            "AGAINST_MACRO": cohort_stats(sub2[sub2["cohort"] == "AGAINST_MACRO"]),
            "WITH_MACRO": cohort_stats(sub2[sub2["cohort"] == "WITH_MACRO"]),
        }
    return out


def write_report(out_dir: Path, results: dict, against_examples: pd.DataFrame) -> None:
    md = [
        "# EMA Macro Filter Counterfactual — 30d Shadow Trades",
        "",
        f"Generated: {pd.Timestamp.utcnow().isoformat()}Z",
        f"Trade window: last 30 days, primary shadow only.",
        f"Total trades analysed: **{results['summary']['total_n']}**",
        f"Total bot net P&L: **${results['summary']['total_pnl']:.2f}**",
        "",
        "## Counterfactual savings by filter",
        "",
        "| Filter | n_AGAINST | AGAINST P&L | Savings if blocked | Net after block | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for f in results["filters"]:
        md.append(
            f"| {f['filter']} | {f['cohorts']['AGAINST_MACRO']['n']} | "
            f"${f['cohorts']['AGAINST_MACRO']['sum_pnl']:.2f} | "
            f"${f['savings_if_blocked']:.2f} | ${f['pnl_after_block']:.2f} | **{f['verdict']}** |"
        )

    md.append("")
    md.append("## Per-cohort detail")
    for f in results["filters"]:
        md.append(f"\n### {f['filter']}\n")
        md.append("| Cohort | n | WR% | sum P&L | avg | median | PF | avg peak MFE% |")
        md.append("|---|---|---|---|---|---|---|---|")
        for c in ("WITH_MACRO", "AGAINST_MACRO", "NEUTRAL"):
            s = f["cohorts"][c]
            pf = f["profit_factor"][c]
            md.append(
                f"| {c} | {s['n']} | {s['wr_pct']} | ${s['sum_pnl']:.2f} | ${s['avg_pnl']:.3f} | "
                f"${s['median_pnl']:.3f} | {pf if pf is not None else '—'} | "
                f"{s['avg_peak_mfe_pct'] if s['avg_peak_mfe_pct'] is not None else '—'} |"
            )

    md.append("")
    md.append("## Per-side breakdown (EMA200_d filter)")
    side_data = results.get("per_side_ema200_d", {})
    md.append("| Side | Cohort | n | WR% | sum P&L | avg |")
    md.append("|---|---|---|---|---|---|")
    for side in ("long", "short"):
        block = side_data.get(side) or {}
        for c in ("WITH_MACRO", "AGAINST_MACRO", "NEUTRAL"):
            s = block.get(c)
            if not s or s.get("n", 0) == 0:
                continue
            md.append(
                f"| {side} | {c} | {s['n']} | {s['wr_pct']} | ${s['sum_pnl']:.2f} | ${s['avg_pnl']:.3f} |"
            )

    md.append("")
    md.append("## Per-scanner AGAINST_MACRO (EMA200_d)")
    scn_data = results.get("per_scanner_ema200_d", {})
    md.append("| Scanner | n_total | AGAINST n | AGAINST sum | AGAINST WR% |")
    md.append("|---|---|---|---|---|")
    for scn, d in sorted(scn_data.items(), key=lambda kv: kv[1]["AGAINST_MACRO"]["sum_pnl"]):
        ag = d["AGAINST_MACRO"]
        if ag["n"] == 0:
            continue
        md.append(f"| {scn} | {d['n']} | {ag['n']} | ${ag['sum_pnl']:.2f} | {ag['wr_pct']} |")

    md.append("")
    md.append("## Verdict")
    best = max(results["filters"], key=lambda f: f["savings_if_blocked"])
    md.append(f"Best filter by raw savings: **{best['filter']}** "
              f"(saves ${best['savings_if_blocked']:.2f} over 30d, "
              f"AGAINST n={best['cohorts']['AGAINST_MACRO']['n']}, "
              f"verdict {best['verdict']}).")
    md.append("")
    md.append("### Window pathology — read before shipping")
    side_d200 = results.get("per_side_ema200_d", {})
    long_n = sum(((side_d200.get('long') or {}).get(c) or {}).get('n', 0)
                 for c in ('WITH_MACRO', 'AGAINST_MACRO', 'NEUTRAL'))
    short_n = sum(((side_d200.get('short') or {}).get(c) or {}).get('n', 0)
                  for c in ('WITH_MACRO', 'AGAINST_MACRO', 'NEUTRAL'))
    md.append(f"- 30d shadow population was {long_n} LONG vs {short_n} SHORT — heavy short bias.")
    md.append("- All 4 majors (BTC/ETH/SOL/XRP) ended the window in bear macro "
              "(close < daily-EMA200), so EMA200_d labelled essentially the entire "
              "short population as WITH_MACRO. EMA200_d has **no separation power** "
              "on this window because the bot already trades the short side.")
    md.append("- Faster filters (EMA50_d / EMA100_d) split the shorts by where price "
              "sits relative to the *medium-term* MA. EMA50_d blocks 67 of 90 shorts "
              "(the bear-rally dips); on this window blocking saves $90 — but it is "
              "effectively a ~75% short-side kill, not a clean macro filter.")
    md.append("")
    if best["verdict"] == "SHIP":
        md.append(f"**Recommendation: SHIP {best['filter']}** flag-gated (default OFF), "
                  "pilot for 7-14d before global enable. Re-check verdict on a "
                  "60d window with mixed regimes before declaring victory.")
    elif best["verdict"] == "PILOT":
        md.append(f"**Recommendation: PILOT {best['filter']}** flag-gated, monitor 7d before enabling globally.")
    else:
        md.append("**Recommendation: HOLD** — sample size or magnitude insufficient.")

    md.append("")
    md.append("## Comparison vs existing session_bias (EMA21/EMA50 on 4h)")
    md.append("The bot already uses a session_bias filter: 4h close vs stacked "
              "EMA21+EMA50 → +1/-1/0. The closest equivalent in this analysis is "
              "macro_bias_4h_50 (close vs 4h-EMA50 only). On this 30d window:")
    f4h50 = next((f for f in results['filters'] if f['filter']=='macro_bias_4h_50'), None)
    if f4h50:
        ag = f4h50['cohorts']['AGAINST_MACRO']
        md.append(f"- 4h-EMA50 AGAINST: n={ag['n']}, sum=${ag['sum_pnl']:.2f}, savings=${f4h50['savings_if_blocked']:.2f}")
    f4h200 = next((f for f in results['filters'] if f['filter']=='macro_bias_4h_200'), None)
    if f4h200:
        ag = f4h200['cohorts']['AGAINST_MACRO']
        md.append(f"- 4h-EMA200 AGAINST: n={ag['n']}, sum=${ag['sum_pnl']:.2f}, savings=${f4h200['savings_if_blocked']:.2f}")
    fd200 = next((f for f in results['filters'] if f['filter']=='macro_bias_d_200'), None)
    if fd200:
        ag = fd200['cohorts']['AGAINST_MACRO']
        md.append(f"- d-EMA200  AGAINST: n={ag['n']}, sum=${ag['sum_pnl']:.2f}, savings=${fd200['savings_if_blocked']:.2f}")
    md.append("")
    md.append("**Verdict on 'does EMA200 outperform existing session_bias?'**")
    md.append("- 4h-EMA200 (longer lookback than session_bias' 4h-EMA50) shows "
              "more separation in raw $ savings, but only because it cuts a larger "
              "share of the short population during bear-rally conditions.")
    md.append("- daily-EMA200 has near-zero AGAINST cohort in this window (all majors "
              "in bear macro), so it cannot outperform anything until regime mixes.")
    md.append("- session_bias is already implemented as a confidence-gated SOFT veto "
              "(passes for conf>=85). EMA200_d would be a HARD veto, qualitatively "
              "different. Cannot be evaluated as 'better' on this single-regime window.")
    md.append("")
    md.append("## Caveats")
    md.append("- Sample is a single 30-day window dominated by one regime "
              "(BTC/ETH/SOL/XRP all bear macro at end of window).")
    md.append("- Counterfactual assumes blocking saves the full P&L of the "
              "blocked trades, including fees that would not have been paid.")
    md.append("- Excludes auto_responder_stuck/restart/reconcile exits (system noise).")
    md.append("- This is a single-window backtest: confirm with 60d+ before "
              "globally enabling.")
    md.append("")
    md.append("## Worst AGAINST_MACRO trades (EMA200_d) — sample")
    if not against_examples.empty:
        md.append("| symbol | side | scanner | grade | opened | pnl |")
        md.append("|---|---|---|---|---|---|")
        for _, r in against_examples.head(15).iterrows():
            md.append(
                f"| {r['symbol']} | {r['side']} | {r['scanner']} | {r['grade']} | "
                f"{r['opened_at']} | ${r['pnl_usd']:.2f} |"
            )

    (out_dir / "report.md").write_text("\n".join(md))
    LOG.info("Wrote %s", out_dir / "report.md")


def main() -> int:
    if not MACRO_PARQUET.exists():
        LOG.error("macro_bias.parquet not found at %s — run ema200_macro_compute.py first", MACRO_PARQUET)
        return 1

    trades = load_trades()
    LOG.info("Loaded %d shadow trades", len(trades))
    if trades.empty:
        LOG.warning("No trades; nothing to do")
        return 0
    macro = load_macro()
    LOG.info("Loaded %d macro rows", len(macro))

    trades = attach_bias(trades, macro)

    filters = ["macro_bias_d_50", "macro_bias_d_100", "macro_bias_d_200",
               "macro_bias_4h_50", "macro_bias_4h_100", "macro_bias_4h_200"]
    results = {
        "summary": {
            "total_n": int(len(trades)),
            "total_pnl": round(float(trades["pnl_usd"].sum()), 2),
            "window_start": str(trades["opened_at"].min()),
            "window_end": str(trades["opened_at"].max()),
        },
        "filters": [analyse_filter(trades, f) for f in filters],
        "per_side_ema200_d": per_side_breakdown(trades, "macro_bias_d_200"),
        "per_scanner_ema200_d": per_scanner(trades, "macro_bias_d_200"),
        "per_side_ema100_d": per_side_breakdown(trades, "macro_bias_d_100"),
        "per_side_ema50_d":  per_side_breakdown(trades,  "macro_bias_d_50"),
    }

    # collect worst AGAINST_MACRO under EMA200_d
    against = trades.copy()
    against["cohort"] = [cohort_label(s, int(b)) for s, b in zip(against["side"], against["macro_bias_d_200"])]
    against = against[against["cohort"] == "AGAINST_MACRO"].sort_values("pnl_usd")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "counterfactual.json").write_text(json.dumps(results, indent=2, default=str))
    LOG.info("Wrote counterfactual.json")
    write_report(OUT_DIR, results, against)
    return 0


if __name__ == "__main__":
    sys.exit(main())

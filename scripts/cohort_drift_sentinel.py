#!/usr/bin/env python3
"""
Cohort Drift Sentinel — Agent 3 (2026-04-26 v1, Batch B).

Detects when a (regime, scanner, side, session) cohort's win rate or PF has
drifted significantly from its 30-day baseline. Emits structured findings to
stdout (cron piped to logger -t cohort_drift) and to
storage/sentinel/cohort_drift_<date>.txt.

Trigger thresholds (v1, conservative):
  - Cohort needs >= 20 trades in last 7d AND >= 60 in baseline (30d)
  - WR drift >= 10pp absolute → FLAG
  - PF drop >= 50% → FLAG
  - Both → CRITICAL

Cron: 0 */1 * * * /home/opc/crypto-trading-bot/scripts/cohort_drift_sentinel.py
"""
import os
import sys
import time
import json
import datetime
import psycopg2
from collections import defaultdict

DB_CFG = dict(host="localhost", user="vnedge", password="VnEdge2026db", dbname="vnedge")
OUT_DIR = "/home/opc/crypto-trading-bot/storage/sentinel"
RECENT_DAYS = 7
BASELINE_DAYS = 30
MIN_RECENT_N = 20
MIN_BASELINE_N = 60
WR_DRIFT_PP_THRESHOLD = 10.0   # absolute percentage points
PF_DROP_PCT_THRESHOLD = 50.0   # %


def fetch_cohort_stats(con, days_recent: int, days_baseline: int):
    """Returns {cohort_key: {recent: {n,wr,pf}, baseline: {n,wr,pf}}}."""
    sql = """
        SELECT
          COALESCE(metadata::jsonb->>'regime','?') AS regime,
          COALESCE(metadata::jsonb->>'scanner', metadata::jsonb->>'setup_type','?') AS scanner,
          side,
          COALESCE(metadata::jsonb->>'session','?') AS session,
          CASE WHEN closed_at >= NOW() - INTERVAL '%s days' THEN 'recent' ELSE 'baseline' END AS bucket,
          COUNT(*) AS n,
          SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END)::float AS gw,
          SUM(CASE WHEN pnl_usd < 0 THEN -pnl_usd ELSE 0 END)::float AS gl
        FROM user_trades
        WHERE closed_at >= NOW() - INTERVAL '%s days'
          AND closed_at IS NOT NULL
          AND trade_type IN ('paper','shadow','real')
        GROUP BY regime, scanner, side, session, bucket
    """
    cur = con.cursor()
    cur.execute(sql, (days_recent, days_baseline))
    rows = cur.fetchall()
    cur.close()
    out = defaultdict(lambda: {"recent": None, "baseline": None})
    for regime, scanner, side, session, bucket, n, wins, gw, gl in rows:
        key = (regime, scanner, side, session)
        wr = (wins / n * 100.0) if n else 0.0
        pf = (gw / gl) if gl and gl > 0 else (float('inf') if gw > 0 else 0.0)
        out[key][bucket] = {"n": n, "wr": wr, "pf": pf, "wins": wins, "gw": gw, "gl": gl}
    return out


def analyze(stats):
    findings = []
    for key, buckets in stats.items():
        rec = buckets.get("recent"); base = buckets.get("baseline")
        if not rec or not base:
            continue
        if rec["n"] < MIN_RECENT_N or base["n"] < MIN_BASELINE_N:
            continue
        wr_delta = rec["wr"] - base["wr"]
        pf_drop_pct = ((base["pf"] - rec["pf"]) / base["pf"] * 100.0) if base["pf"] and base["pf"] > 0 and base["pf"] != float('inf') else 0.0
        flags = []
        if abs(wr_delta) >= WR_DRIFT_PP_THRESHOLD:
            flags.append(f"WR_drift={wr_delta:+.1f}pp")
        if pf_drop_pct >= PF_DROP_PCT_THRESHOLD:
            flags.append(f"PF_drop={pf_drop_pct:.0f}%")
        if not flags:
            continue
        severity = "CRITICAL" if len(flags) >= 2 else "FLAG"
        findings.append({
            "severity": severity,
            "cohort": "/".join(key),
            "recent_n": rec["n"], "baseline_n": base["n"],
            "recent_wr": round(rec["wr"], 1), "baseline_wr": round(base["wr"], 1),
            "recent_pf": round(rec["pf"], 2) if rec["pf"] != float('inf') else 'inf',
            "baseline_pf": round(base["pf"], 2) if base["pf"] != float('inf') else 'inf',
            "flags": flags,
        })
    findings.sort(key=lambda x: (x["severity"] != "CRITICAL", -x["recent_n"]))
    return findings


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_file = os.path.join(OUT_DIR, f"cohort_drift_{ts}.txt")
    try:
        con = psycopg2.connect(**DB_CFG)
    except Exception as e:
        print(f"COHORT_DRIFT: ERROR db_connect: {e}")
        return 1
    try:
        stats = fetch_cohort_stats(con, RECENT_DAYS, BASELINE_DAYS)
        findings = analyze(stats)
    finally:
        con.close()

    header = f"# Cohort Drift Sentinel — {ts} UTC | recent={RECENT_DAYS}d baseline={BASELINE_DAYS}d"
    if not findings:
        msg = f"{header}\nNo drift detected ({len(stats)} cohorts evaluated)."
        print("COHORT_DRIFT: OK no_drift cohorts=" + str(len(stats)))
        with open(out_file, "w") as f:
            f.write(msg + "\n")
        return 0

    # Stdout summary (cron pipe to logger)
    print(f"COHORT_DRIFT: findings={len(findings)} CRITICAL={sum(1 for f in findings if f['severity']=='CRITICAL')}")
    for f in findings[:10]:
        print(f"  {f['severity']} {f['cohort']} rec_n={f['recent_n']} wr={f['recent_wr']}vs{f['baseline_wr']} pf={f['recent_pf']}vs{f['baseline_pf']} {','.join(f['flags'])}")

    # File: full detail
    with open(out_file, "w") as fp:
        fp.write(header + "\n\n")
        for f in findings:
            fp.write(json.dumps(f, default=str) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

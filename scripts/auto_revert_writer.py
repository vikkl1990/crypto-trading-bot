#!/usr/bin/env python3
"""
auto_revert_writer.py — emits auto_revert_events rows.

Standalone helper called from cron-driven detectors (auto_revert_detector,
cohort_drift_sentinel) AND from runtime code (orchestrator on cb_trip).
Idempotent — same event won't double-write within `dedupe_window_min`.

Usage (CLI):
  python3 auto_revert_writer.py --type auto_revert \\
      --severity warn --title "ML high_beta scanner reverted" \\
      --detail "Cohort WR dropped from 64% to 41% over last 50 trades" \\
      --cohort "trending_up/high_beta/long/asia" \\
      --metric wr_pct --before 64 --after 41

Usage (Python import):
  from scripts.auto_revert_writer import emit_event
  emit_event(event_type="cohort_drift", severity="critical",
             title="Cohort SIDEWAYS/structure_bounce dropped 18pp",
             cohort="sideways/structure_bounce/short/london",
             metric_key="wr_pct", before=58.0, after=40.0)
"""
import argparse
import sys
import datetime
import psycopg2

DB_CFG = dict(host="localhost", user="vnedge", password="VnEdge2026db", dbname="vnedge")
DEDUPE_WINDOW_MIN = 30


def emit_event(event_type, severity="warn", title="", detail="",
               cohort=None, metric_key=None, before=None, after=None,
               user_email=None, dedupe_window_min=DEDUPE_WINDOW_MIN):
    if not title:
        raise ValueError("title is required")
    con = psycopg2.connect(**DB_CFG)
    try:
        cur = con.cursor()
        # Dedupe: check for similar event in last N min
        cur.execute(
            """SELECT id FROM auto_revert_events
                WHERE event_type = %s AND title = %s
                  AND COALESCE(cohort,'') = COALESCE(%s,'')
                  AND event_at >= NOW() - INTERVAL '%s minutes'
                LIMIT 1""",
            (event_type, title, cohort, dedupe_window_min)
        )
        existing = cur.fetchone()
        if existing:
            cur.close()
            return existing[0]   # already emitted; return existing id
        cur.execute(
            """INSERT INTO auto_revert_events
                  (event_type, severity, title, detail, cohort,
                   metric_key, metric_before, metric_after, user_email)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (event_type, severity, title, detail or None, cohort,
             metric_key, before, after, user_email)
        )
        new_id = cur.fetchone()[0]
        con.commit()
        cur.close()
        return new_id
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, dest="event_type")
    ap.add_argument("--severity", default="warn",
                    choices=["info", "ok", "warn", "error", "critical"])
    ap.add_argument("--title", required=True)
    ap.add_argument("--detail", default="")
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--metric", dest="metric_key", default=None)
    ap.add_argument("--before", type=float, default=None)
    ap.add_argument("--after", type=float, default=None)
    ap.add_argument("--user", dest="user_email", default=None)
    args = ap.parse_args()
    eid = emit_event(
        event_type=args.event_type, severity=args.severity, title=args.title,
        detail=args.detail, cohort=args.cohort, metric_key=args.metric_key,
        before=args.before, after=args.after, user_email=args.user_email,
    )
    print(f"AUTO_REVERT_WRITER: emitted event_id={eid} ({args.event_type}/{args.severity})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

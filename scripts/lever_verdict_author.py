#!/usr/bin/env python3
"""
Lever Verdict Author — Agent 6 (2026-04-26 v1, Batch B).

For each ACTIVE experiment lever, emits a structured verdict comparing the
treatment cohort vs control cohort. Levers are detected from
metadata::jsonb->>'lever' or 'experiment_id'.

Output: storage/verdicts/lever_verdict_<date>_<lever>.md per lever.

v1 covers known levers from Wave 6.C era:
  - cohort_filter (Lever 2)
  - sizing_override (Lever 1)
  - exit_policy (Lever 3)
  - mark_alignment

A lever is "active" if it has >= 30 trades in last 7 days.
Verdict format mirrors backtest verdicts so they're comparable.

Cron: 0 5 * * * /home/opc/crypto-trading-bot/scripts/lever_verdict_author.py
"""
import os
import sys
import json
import datetime
import psycopg2

DB_CFG = dict(host="localhost", user="vnedge", password="VnEdge2026db", dbname="vnedge")
OUT_DIR = "/home/opc/crypto-trading-bot/storage/verdicts"
WINDOW_DAYS = 7
MIN_N_PER_ARM = 15
KNOWN_LEVER_KEYS = ["cohort_filter", "sizing_override", "exit_policy", "mark_alignment", "lever"]


def fetch_lever_arms(con, lever_key: str, days: int):
    """Returns {arm_value: {n, wr, pf, pnl_total}} for a given lever_key."""
    sql = """
        SELECT
          COALESCE(metadata::jsonb->>%s, 'control') AS arm,
          COUNT(*) AS n,
          SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END)::float AS gw,
          SUM(CASE WHEN pnl_usd < 0 THEN -pnl_usd ELSE 0 END)::float AS gl,
          SUM(pnl_usd)::float AS pnl_total,
          AVG(pnl_usd)::float AS pnl_avg
        FROM user_trades
        WHERE closed_at >= NOW() - INTERVAL '%s days'
          AND closed_at IS NOT NULL
          AND trade_type IN ('paper','shadow','real')
        GROUP BY arm
    """ % ("%s", days)
    cur = con.cursor()
    cur.execute(sql, (lever_key,))
    rows = cur.fetchall()
    cur.close()
    out = {}
    for arm, n, wins, gw, gl, pnl_total, pnl_avg in rows:
        if not n:
            continue
        wr = wins / n * 100.0 if n else 0.0
        pf = (gw / gl) if gl and gl > 0 else (float('inf') if gw > 0 else 0.0)
        out[str(arm)] = dict(n=n, wins=wins, wr=wr, pf=pf, pnl_total=pnl_total, pnl_avg=pnl_avg)
    return out


def render_verdict(lever_key: str, arms: dict) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Lever Verdict — `{lever_key}` ({ts})", ""]
    if not arms or len(arms) < 2:
        lines.append("⚠ INSUFFICIENT DATA — need ≥2 arms with trades.")
        return "\n".join(lines)
    # Identify control = arm named 'control' or first arm by name
    control_name = "control" if "control" in arms else next(iter(sorted(arms.keys())))
    control = arms[control_name]
    lines.append(f"**Window:** last {WINDOW_DAYS} days · **Control arm:** `{control_name}` (n={control['n']})")
    lines.append("")
    lines.append("| Arm | N | WR | PF | NET | Avg | ΔWR vs ctl | ΔPF vs ctl |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm_name, m in sorted(arms.items()):
        if m["n"] < MIN_N_PER_ARM:
            continue
        d_wr = m["wr"] - control["wr"]
        d_pf = (m["pf"] - control["pf"]) if (m["pf"] != float('inf') and control["pf"] != float('inf')) else None
        pf_str = f"{m['pf']:.2f}" if m["pf"] != float('inf') else "∞"
        d_pf_str = f"{d_pf:+.2f}" if d_pf is not None else "n/a"
        lines.append(
            f"| `{arm_name}` | {m['n']} | {m['wr']:.1f}% | {pf_str} | "
            f"${m['pnl_total']:+.2f} | ${m['pnl_avg']:+.3f} | "
            f"{d_wr:+.1f}pp | {d_pf_str} |"
        )

    # Verdict line
    lines.append("")
    best_arm = max((a for a in arms if arms[a]["n"] >= MIN_N_PER_ARM and a != control_name),
                   key=lambda a: arms[a]["pnl_total"], default=None)
    if best_arm is None:
        lines.append("**Verdict:** ⚠ no treatment arm with sufficient sample size.")
    else:
        m = arms[best_arm]
        lift = m["pnl_total"] - control["pnl_total"] if control["pnl_total"] else 0
        if lift > 0 and (m["wr"] - control["wr"]) >= 5 and m["pf"] > control["pf"]:
            lines.append(f"**Verdict:** 🟢 `{best_arm}` outperforms control by ${lift:+.2f} NET — recommend keep")
        elif lift > 0:
            lines.append(f"**Verdict:** 🟡 `{best_arm}` marginal lift ${lift:+.2f} — needs more data or tighter criteria")
        else:
            lines.append(f"**Verdict:** 🔴 `{best_arm}` underperforms control by ${lift:+.2f} — recommend revert")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
    try:
        con = psycopg2.connect(**DB_CFG)
    except Exception as e:
        print(f"LEVER_VERDICT: ERROR db_connect: {e}")
        return 1
    try:
        emitted = 0
        for lever_key in KNOWN_LEVER_KEYS:
            arms = fetch_lever_arms(con, lever_key, WINDOW_DAYS)
            if not arms or sum(a["n"] for a in arms.values()) < 30:
                continue
            verdict = render_verdict(lever_key, arms)
            out_file = os.path.join(OUT_DIR, f"lever_verdict_{ts}_{lever_key}.md")
            with open(out_file, "w") as f:
                f.write(verdict + "\n")
            print(f"LEVER_VERDICT: emitted {lever_key} -> {out_file}")
            emitted += 1
        if emitted == 0:
            print(f"LEVER_VERDICT: no_active_levers (need >=30 trades in {WINDOW_DAYS}d per lever)")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

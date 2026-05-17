#!/usr/bin/env python3
"""
Dashboard Whitelist CLI — manage IP/CIDR allowlist for port 8080.

Source of truth: `dashboard_whitelist` DB table (migration 022).
Sync to iptables: `dashboard_whitelist_sync.sh` (cron every 5 min).

Usage:
    python3 -m scripts.dashboard_whitelist add 1.2.3.4 'admin-mobile' [--expires 2026-12-31] [--by architect]
    python3 -m scripts.dashboard_whitelist remove 1.2.3.4
    python3 -m scripts.dashboard_whitelist list [--include-disabled]
    python3 -m scripts.dashboard_whitelist disable 1.2.3.4
    python3 -m scripts.dashboard_whitelist enable 1.2.3.4
    python3 -m scripts.dashboard_whitelist sync     # generate iptables rules (dry-run prints)
    python3 -m scripts.dashboard_whitelist apply    # generate + apply (requires sudo)
"""
import argparse
import os
import sys
import subprocess
import datetime
from typing import Optional

import psycopg2

DB_CFG = dict(host="localhost", user="vnedge", password="VnEdge2026db", dbname="vnedge")
RULE_COMMENT = "dashboard-whitelist"   # iptables comment marker
PORT = 8080


def db_connect():
    con = psycopg2.connect(**DB_CFG)
    con.autocommit = True
    return con


def cmd_add(ip: str, label: str, granted_by: str = "cli",
            expires: Optional[str] = None, notes: str = "") -> int:
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """INSERT INTO dashboard_whitelist
                  (ip_cidr, label, granted_by, expires_at, notes)
               VALUES (%s::inet, %s, %s, %s, %s)
               ON CONFLICT (ip_cidr) DO UPDATE SET
                  label = EXCLUDED.label,
                  enabled = TRUE,
                  granted_by = EXCLUDED.granted_by,
                  granted_at = NOW(),
                  expires_at = EXCLUDED.expires_at,
                  notes = COALESCE(EXCLUDED.notes, dashboard_whitelist.notes)
               RETURNING id, ip_cidr, label""",
            (ip, label, granted_by, expires, notes or None)
        )
        row = cur.fetchone()
        cur.close()
        print(f"WHITELIST: added/updated id={row[0]} {row[1]} '{row[2]}'")
        print(f"           Run 'apply' (or wait ≤5 min for cron) to push to iptables.")
        return 0
    finally:
        con.close()


def cmd_remove(ip: str) -> int:
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute("DELETE FROM dashboard_whitelist WHERE ip_cidr = %s::inet RETURNING id", (ip,))
        row = cur.fetchone()
        cur.close()
        if row:
            print(f"WHITELIST: removed id={row[0]} {ip}")
            return 0
        print(f"WHITELIST: no entry found for {ip}")
        return 1
    finally:
        con.close()


def cmd_set_enabled(ip: str, enabled: bool) -> int:
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute("UPDATE dashboard_whitelist SET enabled = %s WHERE ip_cidr = %s::inet RETURNING id", (enabled, ip))
        row = cur.fetchone()
        cur.close()
        if row:
            print(f"WHITELIST: id={row[0]} {ip} enabled={enabled}")
            return 0
        print(f"WHITELIST: no entry found for {ip}")
        return 1
    finally:
        con.close()


def cmd_list(include_disabled: bool = False) -> int:
    con = db_connect()
    try:
        cur = con.cursor()
        where = "" if include_disabled else "WHERE enabled = TRUE"
        cur.execute(f"""
            SELECT id, ip_cidr, label, granted_by, granted_at, expires_at, enabled, notes
              FROM dashboard_whitelist {where}
             ORDER BY enabled DESC, ip_cidr
        """)
        rows = cur.fetchall()
        cur.close()
        if not rows:
            print("WHITELIST: empty")
            return 0
        print(f"{'ID':>4}  {'IP/CIDR':<22}  {'LABEL':<22}  {'BY':<14}  {'EXPIRES':<22}  {'EN':<3}  NOTES")
        print("-" * 120)
        for r in rows:
            (rid, ip, lbl, by, gat, eat, en, notes) = r
            ex_str = eat.strftime("%Y-%m-%d %H:%M") if eat else "(permanent)"
            en_str = "Y" if en else "n"
            print(f"{rid:>4}  {str(ip):<22}  {lbl:<22.22}  {by:<14.14}  {ex_str:<22}  {en_str:<3}  {(notes or '')[:50]}")
        return 0
    finally:
        con.close()


def fetch_active_ips() -> list:
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT ip_cidr, label
              FROM dashboard_whitelist
             WHERE enabled = TRUE
               AND (expires_at IS NULL OR expires_at > NOW())
             ORDER BY ip_cidr
        """)
        rows = cur.fetchall()
        cur.close()
        return [(str(r[0]), r[1]) for r in rows]
    finally:
        con.close()


def get_existing_whitelist_rules() -> list:
    """Return iptables rule numbers (descending) that have RULE_COMMENT."""
    try:
        out = subprocess.check_output(
            ["sudo", "iptables", "-L", "INPUT", "-n", "--line-numbers"],
            text=True, stderr=subprocess.DEVNULL
        )
        nums = []
        for line in out.splitlines():
            if RULE_COMMENT in line:
                parts = line.split()
                if parts and parts[0].isdigit():
                    nums.append(int(parts[0]))
        return sorted(nums, reverse=True)   # delete from high to low
    except subprocess.CalledProcessError:
        return []


def cmd_sync(apply: bool = False) -> int:
    """Generate iptables rules from current DB state. Optionally apply via sudo iptables."""
    ips = fetch_active_ips()
    print(f"WHITELIST_SYNC: {len(ips)} active entries in DB")

    # Build commands
    delete_cmds = []
    for n in get_existing_whitelist_rules():
        delete_cmds.append(["sudo", "iptables", "-D", "INPUT", str(n)])

    insert_cmds = []
    # Insert at position 2 (after loopback rule but before drops)
    for ip, label in reversed(ips):   # reversed because each insert pushes the previous down
        insert_cmds.append([
            "sudo", "iptables", "-I", "INPUT", "2",
            "-p", "tcp", "--dport", str(PORT),
            "-s", ip,
            "-m", "comment", "--comment", f"{RULE_COMMENT}:{label[:40]}",
            "-j", "ACCEPT"
        ])

    print(f"  → delete {len(delete_cmds)} stale rules, insert {len(insert_cmds)} fresh")
    if not apply:
        print("DRY RUN — pass 'apply' to execute. Sample commands:")
        for c in (delete_cmds + insert_cmds)[:5]:
            print("    " + " ".join(c))
        return 0

    # Apply
    for c in delete_cmds:
        try:
            subprocess.run(c, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  delete failed: {' '.join(c)} → {e.stderr.decode()[:100]}")
    for c in insert_cmds:
        try:
            subprocess.run(c, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  insert failed: {' '.join(c)} → {e.stderr.decode()[:100]}")

    # Persist
    try:
        save = subprocess.run(["sudo", "iptables-save"], capture_output=True, check=True)
        with open("/tmp/iptables.save.tmp", "wb") as f:
            f.write(save.stdout)
        subprocess.run(["sudo", "mv", "/tmp/iptables.save.tmp", "/etc/sysconfig/iptables"], check=True)
        print(f"WHITELIST_SYNC: APPLIED — {len(insert_cmds)} ACCEPT rules + persisted to /etc/sysconfig/iptables")
    except Exception as e:
        print(f"WHITELIST_SYNC: persist warning: {e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add"); a.add_argument("ip"); a.add_argument("label")
    a.add_argument("--by", default="cli"); a.add_argument("--expires", default=None)
    a.add_argument("--notes", default="")

    r = sub.add_parser("remove"); r.add_argument("ip")
    d = sub.add_parser("disable"); d.add_argument("ip")
    e = sub.add_parser("enable"); e.add_argument("ip")
    sub.add_parser("list").add_argument("--include-disabled", action="store_true")
    sub.add_parser("sync")
    sub.add_parser("apply")

    args = ap.parse_args()
    if args.cmd == "add":
        return cmd_add(args.ip, args.label, args.by, args.expires, args.notes)
    if args.cmd == "remove":
        return cmd_remove(args.ip)
    if args.cmd == "disable":
        return cmd_set_enabled(args.ip, False)
    if args.cmd == "enable":
        return cmd_set_enabled(args.ip, True)
    if args.cmd == "list":
        return cmd_list(getattr(args, "include_disabled", False))
    if args.cmd == "sync":
        return cmd_sync(apply=False)
    if args.cmd == "apply":
        return cmd_sync(apply=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 4.x canary v5 — tracks 4.0 through 4.5 cohorts.
Guardrails: G1-G6 vs baseline (pre-4.x). G7 NO-DEGRADE vs best prior cohort."""
import asyncio, os, sys, statistics
from collections import Counter
sys.path.insert(0, '/home/opc/crypto-trading-bot')
from dotenv import load_dotenv
load_dotenv('/home/opc/crypto-trading-bot/.env')
from db import init_db, get_pool

BASELINE = {"n": 16, "wr_pct": 25.0, "avg_pnl": 0.0262, "avg_mfe": 0.080}

async def cohort_stats(conn, phase):
    rows = await conn.fetch(
        """SELECT pnl_usd, metadata->>'exit_reason' AS reason,
                  metadata->>'peak_mfe_r' AS mfe, metadata->>'grade' AS grade
           FROM user_trades WHERE trade_type='real' AND status='closed'
             AND pnl_usd IS NOT NULL
             AND COALESCE(metadata->>'exit_reason','') != 'orphan_reconciled'
             AND metadata->>'phase' = $1
           ORDER BY closed_at DESC""", phase)
    if not rows: return None
    pnls = [float(r["pnl_usd"]) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mfes = [float(r["mfe"] or 0) for r in rows]
    return {
        "n": len(rows), "wr_pct": round(100*len(wins)/len(rows),1),
        "net": round(sum(pnls),3), "avg_pnl": round(sum(pnls)/len(rows),4),
        "avg_win": round(sum(wins)/len(wins),3) if wins else 0.0,
        "avg_loss": round(sum(losses)/len(losses),3) if losses else 0.0,
        "avg_mfe": round(statistics.mean(mfes),3) if mfes else 0.0,
        "reasons": dict(Counter(r["reason"] or "?" for r in rows).most_common(6)),
    }

async def main():
    await init_db(os.getenv("DATABASE_URL"))
    p = get_pool()
    async with p.acquire() as conn:
        phases = {t: await cohort_stats(conn, t) for t in ("4.0","4.1","4.2","4.3","4.4","4.5")}

    print(f"\n{'Metric':<10} {'Base':<7} {'4.0':<8} {'4.1':<10} {'4.2':<10} {'4.3':<8} {'4.4':<10} {'4.5':<10}")
    print("-" * 78)
    def cell(d, k):
        if not d or d.get(k) is None: return "—"
        return str(d.get(k))
    for k in ("n","wr_pct","avg_pnl","avg_win","avg_loss","avg_mfe"):
        print(f"  {k:<8} {BASELINE.get(k,'—')!s:<7} {cell(phases['4.0'],k):<8} {cell(phases['4.1'],k):<10} {cell(phases['4.2'],k):<10} {cell(phases['4.3'],k):<8} {cell(phases['4.4'],k):<10} {cell(phases['4.5'],k):<10}")

    for tag, d in phases.items():
        if d:
            print(f"\n[{tag}] reasons: {d['reasons']}")

    # Active phase = highest with data
    active_tag = None
    for t in ("4.5","4.4","4.3","4.2","4.1","4.0"):
        if phases.get(t):
            active_tag = t; break
    if not active_tag:
        return
    active = phases[active_tag]
    if active["n"] < 6:
        print(f"\n[holding] Phase {active_tag} n={active['n']} — need ≥6 for guardrails")
        return

    alerts = []
    n = active["n"]
    if active["wr_pct"] - BASELINE["wr_pct"] < -5.0:
        alerts.append(f"G1: WR {active['wr_pct']}% vs baseline {BASELINE['wr_pct']}%")
    if active["avg_pnl"] < BASELINE["avg_pnl"] * 0.5 and active["avg_pnl"] < 0.01:
        alerts.append(f"G2: avg_pnl {active['avg_pnl']} below baseline")
    ek = 100*active["reasons"].get("early_kill",0)/n
    if ek >= 70: alerts.append(f"G3: early_kill {ek:.0f}% >=70%")
    prof = active["reasons"].get("trail_profit",0) + active["reasons"].get("mfe_pullback",0)
    if 100*prof/n < 15: alerts.append(f"G4: profit exits {100*prof/n:.0f}% <15%")
    if active["avg_mfe"] < 0.05: alerts.append(f"G6: avg_mfe {active['avg_mfe']} < 0.05R")

    # G7 no-degrade vs Phase 4.1 (known-good)
    ref = phases.get("4.1")
    if ref and ref["n"] >= 4:
        wr_gap = active["wr_pct"] - ref["wr_pct"]
        pnl_gap = 100*(active["avg_pnl"] - ref["avg_pnl"])/max(abs(ref["avg_pnl"]), 0.001)
        if wr_gap < -10.0:
            alerts.append(f"G7 DEGRADE: WR {active['wr_pct']}% vs 4.1 {ref['wr_pct']}% ({wr_gap:+.1f}pp)")
        if active["avg_pnl"] < 0.05 and pnl_gap < -50.0:
            alerts.append(f"G7 DEGRADE: avg_pnl {active['avg_pnl']} vs 4.1 {ref['avg_pnl']} ({pnl_gap:+.0f}%)")

    print("\n" + "="*78)
    if alerts:
        print(f"🚨 Phase {active_tag} GUARDRAIL ALERTS:")
        for a in alerts: print(f"  [!] {a}")
        print("\n→ Rollback: bash /home/opc/crypto-trading-bot/scripts/rollback_phase4.sh")
        sys.exit(2)
    else:
        print(f"✅ Phase {active_tag} within guardrails ({n} trades)")
        if ref:
            print(f"   vs 4.1: WR Δ={active['wr_pct']-ref['wr_pct']:+.1f}pp | avg_pnl Δ=${active['avg_pnl']-ref['avg_pnl']:+.4f}")

asyncio.run(main())

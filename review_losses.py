import json
from collections import defaultdict

signals = json.load(open("storage/closed_signals.json"))
paper = signals[-50:]

print("=" * 100)
print("PATTERN ANALYSIS: WHAT IS CAUSING LOSSES")
print("=" * 100)

# 1. trail_profit LOSSES
trail_losses = [t for t in paper if "trail_profit" in (t.get("exit_reason","") or t.get("reason","")) and (t.get("pnl_usd",0) or 0) <= 0]
tl_pnl = sum(t.get("pnl_usd",0) for t in trail_losses)
print(f"\n1. TRAIL_PROFIT labeled as LOSS: {len(trail_losses)} trades (${tl_pnl:+.2f})")
for t in trail_losses:
    mfe = t.get("mfe_r",0) or t.get("peak_mfe_r",0) or 0
    print(f"   {t.get('symbol','?'):10} {t.get('side','?'):5} pnl=${t.get('pnl_usd',0):+.2f} mfe={mfe:.2f}R conf={t.get('confidence',0)} grade={t.get('grade','?')}")

# 2. stop_loss hits
sl_hits = [t for t in paper if "stop_loss" in (t.get("exit_reason","") or t.get("reason",""))]
sl_pnl = sum(t.get("pnl_usd",0) for t in sl_hits)
print(f"\n2. STOP_LOSS HITS: {len(sl_hits)} trades (${sl_pnl:+.2f})")
for t in sl_hits:
    mfe = t.get("mfe_r",0) or t.get("peak_mfe_r",0) or 0
    regime = t.get("regime","") or (t.get("metadata",{}) or {}).get("regime","?")
    print(f"   {t.get('symbol','?'):10} {t.get('side','?'):5} ${t.get('pnl_usd',0):+.2f} mfe={mfe:.2f}R conf={t.get('confidence',0)} grade={t.get('grade','?')} {regime}")

# 3. early_kill
ek = [t for t in paper if "early_kill" in (t.get("exit_reason","") or t.get("reason",""))]
ek_pnl = sum(t.get("pnl_usd",0) for t in ek)
print(f"\n3. EARLY KILLS: {len(ek)} trades (${ek_pnl:+.2f})")
for t in ek:
    regime = t.get("regime","") or (t.get("metadata",{}) or {}).get("regime","?")
    print(f"   {t.get('symbol','?'):10} {t.get('side','?'):5} ${t.get('pnl_usd',0):+.2f} conf={t.get('confidence',0)} grade={t.get('grade','?')} {regime}")

# 4. Chandelier
ch = [t for t in paper if "chandelier" in (t.get("exit_reason","") or t.get("reason",""))]
ch_pnl = sum(t.get("pnl_usd",0) for t in ch)
print(f"\n4. CHANDELIER EXITS: {len(ch)} trades (${ch_pnl:+.2f})")
for t in ch:
    mfe = t.get("mfe_r",0) or t.get("peak_mfe_r",0) or 0
    print(f"   {t.get('symbol','?'):10} {t.get('side','?'):5} ${t.get('pnl_usd',0):+.2f} mfe={mfe:.2f}R")

# 5. Grade vs outcome
print(f"\n5. GRADE vs OUTCOME (last 50):")
for g in ["A+", "A", "B", "C", "REJECT"]:
    bucket = [t for t in paper if t.get("grade") == g]
    if bucket:
        bp = sum(t.get("pnl_usd",0) or 0 for t in bucket)
        bw = sum(1 for t in bucket if (t.get("pnl_usd",0) or 0) > 0)
        wr = bw/len(bucket)*100
        print(f"   {g:8} {len(bucket):3d} trades | WR: {bw}/{len(bucket)} ({wr:4.0f}%) | PnL: ${bp:+8.2f}")

# 6. Big losses (> $3)
big = [t for t in paper if (t.get("pnl_usd",0) or 0) < -3]
big_pnl = sum(t.get("pnl_usd",0) for t in big)
print(f"\n6. BIG LOSSES (> $3): {len(big)} trades (${big_pnl:+.2f})")
for t in big:
    reason = t.get("exit_reason","") or t.get("reason","")
    mfe = t.get("mfe_r",0) or t.get("peak_mfe_r",0) or 0
    print(f"   {t.get('symbol','?'):10} {t.get('side','?'):5} ${t.get('pnl_usd',0):+.2f} | {reason} | conf={t.get('confidence',0)} grade={t.get('grade','?')} mfe={mfe:.2f}R")

# 7. WR trend
first25 = paper[:25]
last25 = paper[25:]
fw = sum(1 for t in first25 if (t.get("pnl_usd",0) or 0) > 0)
lw = sum(1 for t in last25 if (t.get("pnl_usd",0) or 0) > 0)
fp = sum(t.get("pnl_usd",0) or 0 for t in first25)
lp = sum(t.get("pnl_usd",0) or 0 for t in last25)
print(f"\n7. WR TREND: First 25: {fw}/25 ({fw/25*100:.0f}%) ${fp:+.2f} | Last 25: {lw}/25 ({lw/25*100:.0f}%) ${lp:+.2f}")

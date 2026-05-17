#!/usr/bin/env python3
"""STRICT_4H_TREND_VETO counterfactual backtest.

For every shadow trade in the last N days, reconstruct what `session_bias` was
at signal time using historical 4h candles, then ask: if the new veto were ON,
which trades would have been blocked, and what would the net P&L delta be?

Replays exact bot logic:
  session_bias = +1 if (close > EMA21_4h > EMA50_4h)
  session_bias = -1 if (close < EMA21_4h < EMA50_4h)
  session_bias =  0 otherwise

Blocked trades = (session_bias > 0 AND side=short) OR (session_bias < 0 AND side=long).

Run: python3 scripts/backtest_strict_4h_veto.py [days]
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
import pandas as pd
import requests

CACHE_DIR = Path("/home/opc/crypto-trading-bot/storage/candle_cache")
DELTA_REST = "https://api.india.delta.exchange/v2/history/candles"
SYMBOLS_4H = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"}  # LINK not cached
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7

DB_DSN = "postgresql://vnedge:VnEdge2026db@localhost:5432/vnedge"


def calc_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def fetch_4h_recent(sym: str, days_back: int = 60) -> pd.DataFrame:
    pname = sym.replace("/", "").replace("USDT", "USD")
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - days_back * 86400
    try:
        r = requests.get(
            DELTA_REST,
            params={"symbol": pname, "resolution": "4h", "start": start, "end": end},
            timeout=15,
        )
        rows = r.json().get("result", [])
    except Exception as e:
        print(f"  REST fetch {sym} ({pname}) failed: {e}")
        return pd.DataFrame()
    if not rows:
        print(f"  REST fetch {sym} ({pname}): empty result")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    df["close"] = df["close"].astype(float)
    return df


def load_4h(sym: str) -> pd.DataFrame:
    fname = sym.replace("/", "_") + "_4h.parquet"
    cached = pd.DataFrame()
    if (CACHE_DIR / fname).exists():
        try:
            cached = pd.read_parquet(CACHE_DIR / fname)
            if cached.index.tz is None:
                cached.index = cached.index.tz_localize("UTC")
        except Exception as e:
            print(f"  cache read {sym}: {e}")
    fresh = fetch_4h_recent(sym, days_back=60)
    if cached.empty and fresh.empty:
        return pd.DataFrame()
    if cached.empty:
        combined = fresh
    elif fresh.empty:
        combined = cached
    else:
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    if "close" in combined.columns:
        combined["ema21"] = calc_ema(combined["close"], 21)
        combined["ema50"] = calc_ema(combined["close"], 50)
    return combined


def session_bias_at(df_4h: pd.DataFrame, ts: pd.Timestamp) -> tuple[int, bool]:
    """Return (session_bias, data_available). bias=0 + avail=False = no data, treat as no-veto."""
    if df_4h.empty or "ema21" not in df_4h.columns:
        return 0, False
    candles_before = df_4h[df_4h.index <= ts]
    if len(candles_before) < 50:
        return 0, False
    # FRESHNESS GUARD: stale-cache fallback would corrupt counterfactual.
    # Require at least one candle within 8h of ts (one 4h bar tolerance).
    last = candles_before.iloc[-1]
    age_hr = (ts - last.name).total_seconds() / 3600.0
    if age_hr > 8:
        return 0, False
    close = float(last["close"])
    e21 = float(last["ema21"])
    e50 = float(last["ema50"])
    if pd.isna(e21) or pd.isna(e50):
        return 0, False
    if close > e21 > e50:
        return 1, True
    if close < e21 < e50:
        return -1, True
    return 0, True


async def main():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(f"""
        SELECT id, symbol, side, opened_at, closed_at, pnl_usd,
               metadata::jsonb->>'exit_config_id' AS exit_config_id,
               metadata::jsonb->>'exit_reason' AS exit_reason,
               COALESCE((metadata::jsonb->>'peak_mfe_r')::numeric, 0) AS peak_mfe_r
        FROM user_trades
        WHERE trade_type='shadow'
          AND closed_at >= NOW() - INTERVAL '{DAYS} days'
          AND status='closed'
          AND COALESCE(metadata::jsonb->>'exit_config_id','')='primary'
          AND COALESCE(metadata::jsonb->>'exit_reason','') NOT IN
              ('auto_responder_stuck_60m','restart_orphan_cleanup','reconcile_overaged_close')
        ORDER BY opened_at;
    """)
    await conn.close()
    print(f"Loaded {len(rows)} primary shadow trades over last {DAYS} days\n")
    if not rows:
        print("No data — abort.")
        return

    candle_cache = {}
    syms_in_data = sorted(set(r["symbol"] for r in rows))
    for sym in syms_in_data:
        if sym not in SYMBOLS_4H:
            print(f"  {sym}: NO 4h cache — skipping (treated as session_bias=0)")
            candle_cache[sym] = pd.DataFrame()
            continue
        df = load_4h(sym)
        candle_cache[sym] = df
        latest = df.index[-1] if len(df) else "EMPTY"
        print(f"  {sym}: {len(df)} 4h bars, latest={latest}")
    print()

    blocked = []
    allowed = []
    no_data = []
    for r in rows:
        ts = r["opened_at"]
        if ts.tzinfo is None:
            ts = pd.Timestamp(ts, tz="UTC")
        sb, avail = session_bias_at(candle_cache.get(r["symbol"], pd.DataFrame()), ts)
        if not avail:
            no_data.append({"symbol": r["symbol"], "opened_at": ts, "pnl": float(r["pnl_usd"] or 0)})
            continue
        opposes = (sb < 0 and r["side"] == "long") or (sb > 0 and r["side"] == "short")
        record = {
            "id": str(r["id"]),
            "symbol": r["symbol"],
            "side": r["side"],
            "opened_at": ts,
            "pnl": float(r["pnl_usd"] or 0),
            "peak_r": float(r["peak_mfe_r"] or 0),
            "exit_reason": r["exit_reason"] or "",
            "session_bias": sb,
        }
        (blocked if opposes else allowed).append(record)

    total = len(blocked) + len(allowed)
    nd_pnl = sum(t["pnl"] for t in no_data)
    if no_data:
        print(f"⚠ {len(no_data)} trades had NO 4h data within 8h tolerance (excluded from cohort math). "
              f"Net of excluded cohort: ${nd_pnl:+.2f}")
        print()
    bl_pnl = sum(t["pnl"] for t in blocked)
    al_pnl = sum(t["pnl"] for t in allowed)
    bl_wins = sum(1 for t in blocked if t["pnl"] > 0)
    al_wins = sum(1 for t in allowed if t["pnl"] > 0)
    bl_losers = [t for t in blocked if t["pnl"] < 0]
    bl_winners = [t for t in blocked if t["pnl"] > 0]

    print("=" * 70)
    print(f"STRICT 4H VETO BACKTEST — last {DAYS}d, primary shadow only")
    print("=" * 70)
    print(f"Total trades evaluated:    {total}")
    print(f"  Would be VETOED:         {len(blocked):4d}  ({100*len(blocked)/total:.1f}%)")
    print(f"  Would stay OPEN:         {len(allowed):4d}  ({100*len(allowed)/total:.1f}%)")
    print()
    print(f"VETOED cohort net P&L:     ${bl_pnl:+8.2f}  ({len(blocked)} trades, {bl_wins} wins, "
          f"WR={100*bl_wins/max(len(blocked),1):.1f}%)")
    print(f"  ├─ losers:               ${sum(t['pnl'] for t in bl_losers):+8.2f}  ({len(bl_losers)} trades)")
    print(f"  └─ winners:              ${sum(t['pnl'] for t in bl_winners):+8.2f}  ({len(bl_winners)} trades)")
    print()
    print(f"ALLOWED cohort net P&L:    ${al_pnl:+8.2f}  ({len(allowed)} trades, {al_wins} wins, "
          f"WR={100*al_wins/max(len(allowed),1):.1f}%)")
    print()
    print(f"NET DELTA if VETO ENABLED: ${-bl_pnl:+8.2f}  (savings = -1 × vetoed cohort net)")
    print()

    # Side breakdown
    print("VETOED by side:")
    for sd in ("long", "short"):
        s_block = [t for t in blocked if t["side"] == sd]
        s_allow = [t for t in allowed if t["side"] == sd]
        if s_block or s_allow:
            print(f"  {sd:5s}  vetoed={len(s_block):3d} (${sum(t['pnl'] for t in s_block):+7.2f}, "
                  f"WR={100*sum(1 for t in s_block if t['pnl']>0)/max(len(s_block),1):.0f}%)   "
                  f"kept={len(s_allow):3d} (${sum(t['pnl'] for t in s_allow):+7.2f})")

    # Symbol breakdown
    print("\nVETOED by symbol:")
    for sym in syms_in_data:
        s_block = [t for t in blocked if t["symbol"] == sym]
        s_allow = [t for t in allowed if t["symbol"] == sym]
        if s_block or s_allow:
            print(f"  {sym:10s}  vetoed={len(s_block):3d} (${sum(t['pnl'] for t in s_block):+7.2f})   "
                  f"kept={len(s_allow):3d} (${sum(t['pnl'] for t in s_allow):+7.2f})")

    # Statistical significance — bootstrap or just label sample size
    print("\nSAMPLE SIZE / SIGNIFICANCE:")
    if len(blocked) >= 30 and bl_pnl < 0:
        print(f"  ✅ Veto cohort net=${bl_pnl:+.2f} over n={len(blocked)} — material savings, "
              f"sample meets minimum n=30 threshold")
    elif len(blocked) >= 30:
        print(f"  ⚠ Veto cohort net=${bl_pnl:+.2f} over n={len(blocked)} — sample is large but "
              f"NOT clearly losing; veto would block break-even cohort")
    else:
        print(f"  ⚠ Only n={len(blocked)} blocked trades — below n=30 threshold; "
              f"verdict not statistically reliable, extend window")

    # Verdict
    print("\nVERDICT:")
    if bl_pnl < -5 and len(blocked) >= 30:
        print(f"  🟢 SHIP: blocking {len(blocked)} trades saves ${-bl_pnl:.2f} with "
              f"strong sample. Enable STRICT_4H_TREND_VETO=1.")
    elif bl_pnl < 0 and len(blocked) >= 15:
        print(f"  🟡 PILOT: vetoed cohort negative (${bl_pnl:+.2f}, n={len(blocked)}), "
              f"sample weak. Enable on shadow only, re-evaluate after 7 days.")
    else:
        print(f"  🔴 HOLD: vetoed cohort not clearly losing (${bl_pnl:+.2f}, "
              f"n={len(blocked)}). Veto would block break-even or winning trades. "
              f"Do NOT enable; investigate alternative filters.")


if __name__ == "__main__":
    asyncio.run(main())

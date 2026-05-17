"""Phase 4.6 backfill — recompute pnl_usd and fees_usd for all closed
real trades using:
  notional = position_size * contract_size * entry_price (or lookup from PRODUCT_MAP)
  gross = (exit - entry) * position_size * contract_size
  fees  = notional_entry + notional_exit both × 0.0005 (0.05% taker estimate)
  net   = gross - fees

Writes net into pnl_usd column, fees into fees_usd column, preserves
a `_pnl_v1_gross` field in metadata so we can audit the old vs new.
"""
import asyncio, os, sys
sys.path.insert(0,'/home/opc/crypto-trading-bot')
from dotenv import load_dotenv
load_dotenv('/home/opc/crypto-trading-bot/.env')
from db import init_db, get_pool

CONTRACT_SIZES_DEMO = {
    "BTC/USDT": 0.001, "ETH/USDT": 0.01, "SOL/USDT": 1.0,
    "XRP/USDT": 1.0, "ADA/USDT": 1.0, "DOGE/USDT": 100.0,
    "SHIB/USDT": 1000.0,
}

async def main():
    await init_db(os.getenv("DATABASE_URL"))
    p = get_pool()
    print("Backfilling pnl_usd (net) and fees_usd on closed real trades...")
    async with p.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, symbol, side, entry_price, exit_price, quantity,
                      pnl_usd AS old_pnl, fees_usd, metadata
               FROM user_trades
               WHERE trade_type='real' AND status='closed'
                 AND entry_price > 0 AND exit_price > 0
                 AND COALESCE(metadata->>'exit_reason','') != 'orphan_reconciled'
            """)
        updated = 0; skipped = 0
        for r in rows:
            m = r["metadata"] or {}
            if isinstance(m, str):
                import json
                try: m = json.loads(m)
                except: m = {}
            # If already v4.6-tagged with gross+net, skip
            if m.get("gross_pnl_usd") is not None and m.get("fees_usd") is not None:
                skipped += 1; continue

            sym = r["symbol"]
            side = r["side"]
            entry = float(r["entry_price"])
            exit_  = float(r["exit_price"])
            qty   = float(r["quantity"] or 0)
            cs    = float(m.get("contract_size") or CONTRACT_SIZES_DEMO.get(sym, 1.0))
            # gross
            if side == "long":
                gross = (exit_ - entry) * qty * cs
            else:
                gross = (entry - exit_) * qty * cs
            # fees: 0.05% × notional × 2 legs
            fees = (entry + exit_) * qty * cs * 0.0005  # sum both sides
            net = gross - fees

            # Preserve original pnl_usd as audit
            import json as _j
            patch = _j.dumps({
                "gross_pnl_usd": round(gross, 4),
                "net_pnl_usd": round(net, 4),
                "fees_usd": round(fees, 4),
                "contract_size": cs,
                "_backfilled": True,
                "_pnl_v1_gross_overstatement": round(float(r["old_pnl"] or 0) - gross, 4),
            })
            await conn.execute(
                """UPDATE user_trades SET pnl_usd=$1, fees_usd=$2,
                       metadata = metadata || $3::jsonb
                   WHERE id=$4""",
                net, fees, patch, r["id"])
            updated += 1

        print(f"\nBackfilled {updated} trades. Skipped {skipped} already-tagged.")
        # Show summary
        stats = await conn.fetchrow(
            """SELECT COUNT(*), ROUND(SUM(pnl_usd)::numeric,2) AS sum_net,
                      ROUND(SUM(fees_usd)::numeric,2) AS sum_fees,
                      ROUND(SUM((metadata->>'gross_pnl_usd')::numeric)::numeric,2) AS sum_gross
               FROM user_trades
               WHERE trade_type='real' AND status='closed'
                 AND COALESCE(metadata->>'exit_reason','') != 'orphan_reconciled'""")
        print(f"\nAggregate real trades: n={stats['count']} | gross=${stats['sum_gross']} | fees=${stats['sum_fees']} | NET=${stats['sum_net']}")

asyncio.run(main())

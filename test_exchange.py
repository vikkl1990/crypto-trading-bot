import asyncio
import yaml

with open("config/settings.yaml") as f:
    cfg = yaml.safe_load(f)

async def test():
    from exchange.ccxt_client import CcxtExchangeClient
    client = CcxtExchangeClient(cfg)
    await client.connect()
    exchange = client._exchange
    
    print("=== 1. CONNECTION ===")
    print("Exchange:", exchange.id)
    
    print("\n=== 2. BALANCE ===")
    try:
        bal = await client.fetch_balance()
        if isinstance(bal, dict):
            free = bal.get("free", {})
            if isinstance(free, dict):
                usdt = free.get("USDT", free.get("USD", 0))
            else:
                usdt = 0
            print("USDT Free: $%.2f" % float(usdt or 0))
        else:
            print("Type:", type(bal).__name__)
    except Exception as e:
        print("ERROR:", e)
    
    print("\n=== 3. MARKETS ===")
    try:
        for sym in ["BTC/USDT", "BTC/USDT:USDT", "BTCUSDT"]:
            m = exchange.markets.get(sym)
            if m:
                lim = m.get("limits", {})
                prec = m.get("precision", {})
                print("Symbol:", m.get("symbol"))
                print("  Type:", m.get("type"))
                print("  Contract:", m.get("contractSize"))
                print("  Min amount:", lim.get("amount", {}).get("min"))
                print("  Min cost:", lim.get("cost", {}).get("min"))
                print("  Precision price:", prec.get("price"))
                print("  Precision amount:", prec.get("amount"))
                break
    except Exception as e:
        print("ERROR:", e)
    
    print("\n=== 4. TICKER ===")
    try:
        t = await client.fetch_ticker("BTC/USDT")
        if isinstance(t, dict):
            print("Last: $%.2f" % (t.get("last", 0) or 0))
            print("Bid: $%.2f" % (t.get("bid", 0) or 0))
            print("Ask: $%.2f" % (t.get("ask", 0) or 0))
    except Exception as e:
        print("ERROR:", e)
    
    print("\n=== 5. POSITIONS ===")
    try:
        pos = await exchange.fetch_positions()
        open_pos = [p for p in pos if abs(float(p.get("contracts", 0) or 0)) > 0]
        print("Open:", len(open_pos))
        for p in open_pos:
            print("  %s %s contracts=%s notional=%s" % (
                p.get("symbol"), p.get("side"), p.get("contracts"), p.get("notional")))
    except Exception as e:
        print("ERROR:", e)
    
    print("\n=== 6. CAPABILITIES ===")
    print("create_order:", hasattr(exchange, "create_order"))
    print("set_leverage:", hasattr(exchange, "set_leverage"))
    print("set_margin_mode:", hasattr(exchange, "set_margin_mode"))
    print("cancel_order:", hasattr(exchange, "cancel_order"))
    print("fetch_my_trades:", hasattr(exchange, "fetch_my_trades"))
    print("fetch_open_orders:", hasattr(exchange, "fetch_open_orders"))
    
    print("\n=== 7. EXECUTION ENGINE ===")
    try:
        from execution.engine import ExecutionEngine
        e = ExecutionEngine(client, cfg, None)
        print("Created:", e is not None)
        print("Paper mode:", getattr(e, "_paper_mode", "not found"))
        print("Mode from config:", cfg.get("bot", {}).get("operating_mode", "?"))
    except Exception as e:
        print("ERROR:", e)
    
    print("\n=== 8. GAPS FOR REAL TRADING ===")
    gaps = []
    mode = cfg.get("bot", {}).get("operating_mode", "?")
    if "paper" in str(mode).lower():
        gaps.append("operating_mode=%s — ExecutionEngine will simulate, not place real orders" % mode)
    
    if not hasattr(exchange, "set_leverage"):
        gaps.append("No set_leverage — cannot set leverage before order")
    
    rt = cfg.get("real_trading", {})
    if not rt.get("enabled"):
        gaps.append("real_trading.enabled=False in config")
    
    if rt.get("dry_run", True):
        gaps.append("real_trading.dry_run=True in config (overridden by state file)")
    
    for i, g in enumerate(gaps):
        print("  GAP %d: %s" % (i+1, g))
    
    if not gaps:
        print("  No gaps — ready for real trading")
    
    await client.close()

asyncio.run(test())

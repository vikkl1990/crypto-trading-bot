#!/usr/bin/env python3
"""
PROD L2 Maker Probe — Test 3 of EXECUTION_INVESTIGATION_RESULTS.md

Passively observes Delta India PROD L2 + recent trades.
NO orders placed. Read-only against PROD.

For each sample:
  1. Snapshot bid/ask of PROD L2 for symbol
  2. Compute hypothetical maker BUY price using bot's exact logic
     (Phase 5.5.2 sym_bp + anti-cross clamp)
  3. Sleep probe_window_ms (500ms standard, 2500ms patient)
  4. Pull trades that occurred during the window
  5. Did any TAKER-SELL cross at or below our maker_px? → YES = we would have filled

Output: per-symbol fill rate. Compare to testnet 0% baseline to settle H1'
(testnet vs PROD liquidity hypothesis).

Usage:
  python3 prod_l2_maker_probe.py --samples 30 --window-ms 500
  python3 prod_l2_maker_probe.py --smoke   # 5 samples, fast confidence check
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

PROD_BASE = "https://api.india.delta.exchange"

# Symbol -> (bp_offset, tick_size)
# Mirrors sym_bp_map in execution/user_real_manager.py (Phase 5.5.2)
SYMBOLS = {
    "BTCUSD": (1.0, 0.5),
    "ETHUSD": (1.5, 0.05),
    "SOLUSD": (2.0, 0.001),
}


def http_get(url: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VNEdge-Probe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  [WARN] HTTP {url} -> {e}", file=sys.stderr)
        return None


def get_l2(symbol: str) -> Optional[Tuple[float, float]]:
    d = http_get(f"{PROD_BASE}/v2/l2orderbook/{symbol}")
    if not d:
        return None
    r = d.get("result", {})
    buys = r.get("buy") or []
    sells = r.get("sell") or []
    if not buys or not sells:
        return None
    try:
        return float(buys[0]["price"]), float(sells[0]["price"])
    except (KeyError, ValueError, TypeError):
        return None


def get_trades_since(symbol: str, since_us: int) -> list:
    d = http_get(f"{PROD_BASE}/v2/trades/{symbol}?page_size=500")
    if not d:
        return []
    trades = d.get("result", []) or []
    return [t for t in trades if int(t.get("timestamp", 0)) >= since_us]


def round_to_tick(price: float, tick: float) -> float:
    return round(price / tick) * tick


def compute_maker_buy_px(bid: float, ask: float, bp: float, tick: float) -> Tuple[float, bool]:
    """Mirror of user_real_manager.py BUY-side maker placement.
    Returns (final_px, was_clamped_to_below_ask).
    """
    target = bid * (1.0 + bp / 10000.0)
    clamp_px = ask - tick
    final = min(target, clamp_px)
    final_rounded = round_to_tick(final, tick)
    was_clamped = (target > clamp_px)
    return final_rounded, was_clamped


def probe_sample(symbol: str, bp: float, tick: float, window_ms: int) -> dict:
    snap = get_l2(symbol)
    if not snap:
        return {"ok": False, "symbol": symbol, "reason": "l2_fetch_failed"}
    bid, ask = snap
    maker_px, was_clamped = compute_maker_buy_px(bid, ask, bp, tick)
    spread_ticks = max(0, round((ask - bid) / tick))
    is_at_bid_or_below = (maker_px <= bid)  # H2' clamp-collapse marker

    t_start_us = int(time.time() * 1_000_000)
    time.sleep(window_ms / 1000.0)
    t_end_us = int(time.time() * 1_000_000)

    trades = get_trades_since(symbol, t_start_us)
    taker_sells = [t for t in trades if t.get("seller_role") == "taker"]

    would_fill = False
    fill_px = None
    for t in taker_sells:
        try:
            tp = float(t["price"])
            if tp <= maker_px:
                would_fill = True
                fill_px = tp
                break
        except (KeyError, ValueError, TypeError):
            continue

    return {
        "ok": True,
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "spread_ticks": spread_ticks,
        "maker_px": maker_px,
        "was_clamped": was_clamped,
        "is_at_bid_or_below": is_at_bid_or_below,
        "n_trades_in_window": len(trades),
        "n_taker_sells_in_window": len(taker_sells),
        "would_fill": would_fill,
        "fill_px": fill_px,
        "window_us": t_end_us - t_start_us,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=30, help="samples per symbol")
    ap.add_argument("--window-ms", type=int, default=500, help="probe window in ms")
    ap.add_argument("--inter-sample-s", type=int, default=12, help="seconds between samples")
    ap.add_argument("--smoke", action="store_true", help="5 samples, fast diagnostic")
    args = ap.parse_args()

    if args.smoke:
        args.samples = 5
        args.inter_sample_s = 4

    print(f"# PROD L2 Maker Probe — Test 3 (read-only)")
    print(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"# Window: {args.window_ms}ms | Samples/sym: {args.samples} | Gap: {args.inter_sample_s}s")
    print(f"# Symbols: {list(SYMBOLS.keys())}")
    print()

    results = {sym: [] for sym in SYMBOLS}

    for i in range(args.samples):
        for symbol, (bp, tick) in SYMBOLS.items():
            res = probe_sample(symbol, bp, tick, args.window_ms)
            results[symbol].append(res)
            if res.get("ok"):
                fill_marker = "FILL" if res["would_fill"] else "miss"
                clamp_marker = " [clamp@bid]" if res["is_at_bid_or_below"] else ""
                print(
                    f"  s={i+1:>3}  {symbol}  "
                    f"bid={res['bid']:>9}  ask={res['ask']:>9}  "
                    f"maker={res['maker_px']:>9}  "
                    f"sp={res['spread_ticks']}tk  "
                    f"sells={res['n_taker_sells_in_window']:>3}  "
                    f"-> {fill_marker}{clamp_marker}"
                )
            else:
                print(f"  s={i+1:>3}  {symbol}  ERR: {res.get('reason')}")
        if i < args.samples - 1:
            time.sleep(args.inter_sample_s)

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    grand_n = grand_fill = 0
    for symbol, rows in results.items():
        ok = [r for r in rows if r.get("ok")]
        if not ok:
            print(f"  {symbol}: no successful samples")
            continue
        n = len(ok)
        wf = sum(1 for r in ok if r["would_fill"])
        clamp = sum(1 for r in ok if r["is_at_bid_or_below"])
        avg_spread = sum(r["spread_ticks"] for r in ok) / n
        avg_sells = sum(r["n_taker_sells_in_window"] for r in ok) / n
        rate = (wf / n * 100.0) if n else 0.0
        grand_n += n
        grand_fill += wf
        print(
            f"  {symbol:<8}  fill_rate={rate:>5.1f}%  ({wf}/{n})  "
            f"clamp@bid={clamp}/{n}  "
            f"avg_spread={avg_spread:.1f}tk  "
            f"avg_taker_sells/win={avg_sells:.1f}"
        )
    if grand_n:
        agg = grand_fill / grand_n * 100.0
        print(f"  {'AGGREGATE':<8}  fill_rate={agg:>5.1f}%  ({grand_fill}/{grand_n})")
        print()
        if agg >= 30.0:
            print(f"  GATE: PASS — PROD maker rate {agg:.1f}% >= 30% target")
            print("  Next step: surgical fixes only (telemetry + clamp-collapse fix)")
        elif agg >= 5.0:
            print(f"  GATE: PARTIAL — PROD maker rate {agg:.1f}% (target 30%)")
            print("  Next step: full surgical refactor + per-symbol bp recalibration")
        else:
            print(f"  GATE: FAIL — PROD maker rate {agg:.1f}% (well below 5% even)")
            print("  Next step: this isn't a testnet artifact. H2' clamp-collapse or H4 staleness")
            print("             likely dominant. Refactor scope as planned.")

    print()
    print(f"# Finished: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Bybit Shadow Simulator — Architect's "Exchange Compare" tab data source.

Runs as cron every 1 minute. For each Delta India shadow/paper trade
opened in the last 60 seconds, generates a parallel `exchange='bybit'`
shadow trade row using:
  - Live Bybit L2 (via WebSocket — REST is geo-blocked from VM)
  - Same maker simulation logic as Delta shadow path
  - Bybit India fee schedule (0.01% maker, 0.06% taker, +1.18× GST)

Output: rows in user_trades with exchange='bybit', trade_type='shadow'.
Comparison computed by /api/exchange-comparison endpoint.

Cron: * * * * * /home/opc/miniconda3/bin/python3.13 /home/opc/crypto-trading-bot/scripts/bybit_shadow_simulator.py 2>&1 | logger -t bybit_shadow

Bybit symbol mapping:
    BTC/USDT  →  BTCUSDT  (perpetual linear)
    ETH/USDT  →  ETHUSDT
    SOL/USDT  →  SOLUSDT
    XRP/USDT  →  XRPUSDT

Bybit India fees (VIP0):
    Maker: 0.01% (1 bp)
    Taker: 0.06% (6 bp)
    GST:   1.18× multiplier on India accounts
"""
import json
import os
import sys
import time
import uuid
import socket
import datetime
from typing import Dict, Optional, Tuple

import psycopg2
import websocket  # type: ignore[import]

DB_CFG = dict(host="localhost", user="vnedge", password="VnEdge2026db", dbname="vnedge")

BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"
SYMBOL_MAP = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USDT": "XRPUSDT",
}
REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# Bybit India effective fees including 18% GST
BYBIT_MAKER_FEE_PCT = 0.0001 * 1.18   # 0.0118%
BYBIT_TAKER_FEE_PCT = 0.0006 * 1.18   # 0.0708%

# Maker simulation parameters (mirrors Delta sym_bp_map shape)
SYM_BP = {
    "BTCUSDT": 1.0,
    "ETHUSDT": 1.5,
    "SOLUSDT": 3.0,
    "XRPUSDT": 1.5,
}

import os as _os

WS_SAMPLE_SEC = int(_os.environ.get("WS_SAMPLE_SEC", "3"))    # was 6 — tightened for daemon mode
LOOKBACK_SEC = int(_os.environ.get("LOOKBACK_SEC", "120"))    # was 600 — daemon catches new opens fast
PROBE_WINDOW_MS = 500                  # maker probe window (matches Delta standard mode)
ONLY_MIRROR_CLOSED = _os.environ.get("ONLY_MIRROR_CLOSED", "false").lower() == "true"
                                       # 2026-04-26 v2: default FALSE (daemon mirrors opens AND closes
                                       # in real-time for parallel comparison; was true under cron).
DAEMON_MODE = _os.environ.get("DAEMON_MODE", "true").lower() == "true"
DAEMON_INTERVAL_SEC = int(_os.environ.get("DAEMON_INTERVAL_SEC", "3"))


# ─────────────────────────────────────────────────────────────────
# 1. Bybit L2 collector via WebSocket
# ─────────────────────────────────────────────────────────────────
def collect_bybit_l2(seconds: int = WS_SAMPLE_SEC) -> Dict[str, list]:
    """Subscribe to Bybit orderbook stream for SECONDS, return per-symbol
    list of (ts_ms, bid_px, bid_sz, ask_px, ask_sz) snapshots.
    """
    out: Dict[str, list] = {sym: [] for sym in SYMBOL_MAP.values()}
    try:
        ws = websocket.create_connection(BYBIT_WS, timeout=8)
        ws.send(json.dumps({
            "op": "subscribe",
            "args": [f"orderbook.50.{sym}" for sym in SYMBOL_MAP.values()],
        }))
        start = time.time()
        ws.settimeout(2.0)
        while time.time() - start < seconds:
            try:
                msg = ws.recv()
                d = json.loads(msg)
                if d.get("op") == "subscribe":
                    continue
                topic = d.get("topic", "")
                if "orderbook" not in topic:
                    continue
                sym = topic.split(".")[-1]
                if sym not in out:
                    continue
                data = d.get("data", {})
                bids = data.get("b", [])
                asks = data.get("a", [])
                if not bids or not asks:
                    continue
                ts = int(d.get("ts", time.time() * 1000))
                try:
                    out[sym].append((
                        ts,
                        float(bids[0][0]), float(bids[0][1]),
                        float(asks[0][0]), float(asks[0][1]),
                    ))
                except (ValueError, IndexError, TypeError):
                    continue
            except socket.timeout:
                continue
            except websocket.WebSocketException:
                break
        ws.close()
    except Exception as e:
        print(f"BYBIT_SHADOW: WS collect error: {e}", file=sys.stderr)
    return out


# ─────────────────────────────────────────────────────────────────
# 2. Maker fill simulation (mirrors execution/maker_modes.simulate_maker_fill)
# ─────────────────────────────────────────────────────────────────
def round_to_tick(price: float, tick: float) -> float:
    return round(price / tick) * tick if tick > 0 else price


def simulate_bybit_fill(side: str, snapshots: list, bp: float) -> Tuple[bool, float, float, float, float]:
    """Returns (filled_maker, fill_price, signal_bid, signal_ask, fee_pct).
    side: 'long' or 'short'
    snapshots: list of (ts_ms, bid_px, bid_sz, ask_px, ask_sz) over WS_SAMPLE_SEC
    bp: maker offset basis points

    Logic:
      - Take FIRST snapshot as signal moment
      - Compute hypothetical maker price (bid * (1 + bp/10000) for buy)
      - Apply anti-cross clamp
      - Walk subsequent snapshots in PROBE_WINDOW_MS
      - If a counterparty crossed our maker price during probe: filled=True, fee=maker
      - Else: simulate Tier 3 taker fill at top of opposite book, fee=taker
    """
    if not snapshots:
        return False, 0.0, 0.0, 0.0, BYBIT_TAKER_FEE_PCT
    ts0, bid0, bidsz0, ask0, asksz0 = snapshots[0]
    if side.lower() == "long":
        target = bid0 * (1.0 + bp / 10000.0)
        clamp_px = ask0 - 0.01
        maker_px = min(target, clamp_px)
        # Filled if subsequent ASK crosses down to maker_px during probe window
        deadline_ms = ts0 + PROBE_WINDOW_MS
        for ts, bid, _, ask, _ in snapshots[1:]:
            if ts > deadline_ms:
                break
            if ask <= maker_px:
                return True, maker_px, bid0, ask0, BYBIT_MAKER_FEE_PCT
        # No fill within probe — Tier 3 taker at current ask (last snapshot)
        last_ask = snapshots[-1][3]
        return False, last_ask, bid0, ask0, BYBIT_TAKER_FEE_PCT
    else:  # short
        target = ask0 * (1.0 - bp / 10000.0)
        clamp_px = bid0 + 0.01
        maker_px = max(target, clamp_px)
        deadline_ms = ts0 + PROBE_WINDOW_MS
        for ts, bid, _, _, _ in snapshots[1:]:
            if ts > deadline_ms:
                break
            if bid >= maker_px:
                return True, maker_px, bid0, ask0, BYBIT_MAKER_FEE_PCT
        last_bid = snapshots[-1][1]
        return False, last_bid, bid0, ask0, BYBIT_TAKER_FEE_PCT


# ─────────────────────────────────────────────────────────────────
# 3. Find delta shadow trades to mirror, generate Bybit shadow rows
# ─────────────────────────────────────────────────────────────────
def fetch_orphaned_open_mirrors(con, lookback_sec: int) -> list:
    """v2 (2026-04-26 evening): find bybit shadow trades that are still OPEN
    but whose delta_india source has CLOSED. Need to UPDATE the open mirror
    with delta's close data + Bybit-side fill price (using current Bybit L2).
    """
    cur = con.cursor()
    cur.execute(f"""
        SELECT b.id::text AS mirror_id, b.user_id::text, b.symbol, b.side,
               b.entry_price, b.quantity, b.fees_usd,
               d.id::text AS delta_id, d.exit_price AS d_exit, d.pnl_usd AS d_pnl,
               d.closed_at AS d_closed_at,
               b.metadata::jsonb->>'mirror_of_delta_trade_id' AS linked
          FROM user_trades b
          JOIN user_trades d
            ON d.id::text = (b.metadata::jsonb->>'mirror_of_delta_trade_id')
         WHERE b.exchange = 'bybit' AND b.trade_type = 'shadow'
           AND b.closed_at IS NULL
           AND d.closed_at IS NOT NULL
           AND d.closed_at >= NOW() - INTERVAL '{int(lookback_sec)} seconds'
         ORDER BY d.closed_at DESC LIMIT 100
    """)
    rows = cur.fetchall()
    cur.close()
    out = []
    for r in rows:
        out.append({
            "mirror_id": r[0], "user_id": r[1], "symbol": r[2], "side": r[3],
            "b_entry": float(r[4] or 0), "b_qty": float(r[5] or 0),
            "b_open_fees": float(r[6] or 0),
            "delta_id": r[7], "d_exit": float(r[8] or 0), "d_pnl": float(r[9] or 0),
            "d_closed_at": r[10],
        })
    return out


def update_mirror_with_close(con, mirror, snapshots_by_sym):
    """For an orphaned open bybit mirror, simulate Bybit close at current L2 +
    UPDATE the row with exit + pnl.
    """
    by_sym = SYMBOL_MAP.get(mirror["symbol"])
    if not by_sym:
        return False
    snaps = snapshots_by_sym.get(by_sym, [])
    if not snaps:
        # Fallback: use delta exit price as bybit exit (conservative)
        exit_px = mirror["d_exit"]
        fee_pct = BYBIT_TAKER_FEE_PCT
    else:
        # Use last snapshot's mid-price for the close
        last = snaps[-1]
        bid, ask = last[1], last[3]
        # Close = OPPOSITE side; long closes at bid, short at ask
        if str(mirror["side"]).lower() == "long":
            exit_px = bid
        else:
            exit_px = ask
        fee_pct = BYBIT_TAKER_FEE_PCT

    # Recompute Bybit-side close PnL
    notional_in = mirror["b_entry"] * mirror["b_qty"]
    notional_out = exit_px * mirror["b_qty"]
    if str(mirror["side"]).lower() == "long":
        gross = notional_out - notional_in
    else:
        gross = notional_in - notional_out
    close_fee = notional_out * fee_pct
    new_total_fees = mirror["b_open_fees"] + close_fee
    new_pnl = gross - new_total_fees
    cur = con.cursor()
    cur.execute("""
        UPDATE user_trades
           SET status = 'closed',
               closed_at = NOW(),
               exit_price = %s,
               pnl_usd = %s,
               fees_usd = %s,
               metadata = metadata || jsonb_build_object(
                              'close_via', 'bybit_shadow_update',
                              'close_fee_added', %s
                          )
         WHERE id = %s
        RETURNING id::text
    """, (exit_px, new_pnl, new_total_fees, close_fee, mirror["mirror_id"]))
    new_id = cur.fetchone()
    cur.close()
    return new_id is not None


def fetch_unmirrored_delta_trades(con, lookback_sec: int) -> list:
    """Return delta_india shadow/paper trades that don't have a bybit mirror yet.
    2026-04-26 fix: include trades that CLOSED in the lookback window even if
    opened earlier, AND default to only-closed mode for meaningful PnL.
    """
    cur = con.cursor()
    closed_filter = "AND d.closed_at IS NOT NULL" if ONLY_MIRROR_CLOSED else ""
    cur.execute(f"""
        SELECT d.id, d.user_id, d.symbol, d.side, d.entry_price,
               d.exit_price, d.quantity, d.opened_at, d.closed_at,
               d.pnl_usd, d.fees_usd, d.metadata, d.signal_data,
               d.trade_type, d.status
          FROM user_trades d
         WHERE d.exchange = 'delta_india'
           AND d.trade_type IN ('shadow', 'paper')
           {closed_filter}
           AND (d.opened_at >= NOW() - INTERVAL '{int(lookback_sec)} seconds'
                OR d.closed_at >= NOW() - INTERVAL '{int(lookback_sec)} seconds')
           AND NOT EXISTS (
               SELECT 1 FROM user_trades b
                WHERE b.exchange = 'bybit'
                  AND b.user_id = d.user_id
                  AND b.symbol = d.symbol
                  AND b.opened_at = d.opened_at
                  AND b.side = d.side
           )
         ORDER BY d.closed_at DESC NULLS LAST, d.opened_at DESC
         LIMIT 200
    """)
    rows = cur.fetchall()
    cur.close()
    return rows


def insert_bybit_shadow(con, src: tuple, by_fill_price: float, by_filled_maker: bool,
                       by_fee_pct: float, by_signal_bid: float, by_signal_ask: float) -> str:
    """Insert mirror Bybit shadow trade row.

    UNIT-CONVERSION FIX (2026-04-26):
    Delta India stores `quantity` in CONTRACTS where each contract = `contract_size`
    of base currency (ETH 0.01, BTC 0.001, SOL 0.1, XRP/USDT/ADA 1.0).
    Bybit linear futures use BASE CURRENCY directly (BTCUSDT qty in BTC).
    So we MUST multiply by contract_size to get the equivalent Bybit qty.
    Without this: 28 ETH-contracts (=0.28 ETH) was treated as 28 ETH → 100x notional.
    """
    (src_id, user_id, symbol, side, src_entry, src_exit, qty, opened_at,
     closed_at, src_pnl, src_fees, src_meta, src_sigdata, trade_type, status) = src

    # Convert Delta contracts → Bybit base-currency qty
    src_meta_dict = src_meta if isinstance(src_meta, dict) else (json.loads(src_meta) if src_meta else {})
    contract_size = float(src_meta_dict.get("contract_size", 1.0) or 1.0)
    by_qty = float(qty) * contract_size  # equivalent base-currency size on Bybit

    # Recompute PnL using Bybit fill prices.
    # - Entry: by_fill_price (Bybit's fill)
    # - Exit:  src_exit (use Delta's exit since strategy decides when to exit; assumes parity)
    new_pnl = 0.0
    new_fees = 0.0
    if src_exit is not None:
        notional_in = by_fill_price * by_qty
        notional_out = float(src_exit) * by_qty
        if str(side).lower() == "long":
            gross = notional_out - notional_in
        else:
            gross = notional_in - notional_out
        # Fees on entry + exit (assume same fee tier for both)
        fee_in = notional_in * by_fee_pct
        fee_out = notional_out * by_fee_pct
        new_fees = fee_in + fee_out
        new_pnl = gross - new_fees

    # METADATA_SCANNER_NORM_5_22 (2026-05-03) — extract scanner once, write both fields.
    _norm_scanner = (src_sigdata or {}).get("scanner") if isinstance(src_sigdata, dict) else None
    new_meta = {
        "exchange": "bybit",
        "mirror_of_delta_trade_id": str(src_id),
        "fee_type": "maker" if by_filled_maker else "taker",
        "by_signal_bid": by_signal_bid,
        "by_signal_ask": by_signal_ask,
        "by_fill_price": by_fill_price,
        "by_fee_pct": by_fee_pct,
        "by_qty_base": by_qty,
        "src_entry_price": float(src_entry) if src_entry else None,
        "src_qty_contracts": float(qty),
        "src_contract_size": contract_size,
        "src_pnl_usd": float(src_pnl) if src_pnl else None,
        "exit_reason": "mirrored_from_delta",
        "scanner": _norm_scanner,    # canonical write path (matches standard urm)
    }
    new_signal_data = {
        "signal_price": float(src_entry) if src_entry else by_fill_price,
        "entry_price": by_fill_price,
        "scanner": _norm_scanner,    # METADATA_SCANNER_NORM_5_22: same value as metadata.scanner
        "side": side,
    }

    cur = con.cursor()
    new_id = str(uuid.uuid4())
    # Store quantity in BASE CURRENCY (Bybit convention) — already converted via by_qty
    if closed_at is not None:
        cur.execute("""
            INSERT INTO user_trades (id, user_id, exchange, trade_type, symbol, side,
                entry_price, exit_price, quantity, status, pnl_usd, fees_usd,
                signal_data, metadata, opened_at, closed_at)
            VALUES (%s, %s, 'bybit', %s, %s, %s,
                    %s, %s, %s, 'closed', %s, %s,
                    %s::jsonb, %s::jsonb, %s, %s)
        """, (
            new_id, user_id, trade_type, symbol, side,
            by_fill_price, float(src_exit), by_qty, new_pnl, new_fees,
            json.dumps(new_signal_data), json.dumps(new_meta),
            opened_at, closed_at,
        ))
    else:
        cur.execute("""
            INSERT INTO user_trades (id, user_id, exchange, trade_type, symbol, side,
                entry_price, quantity, status, signal_data, metadata, opened_at)
            VALUES (%s, %s, 'bybit', %s, %s, %s,
                    %s, %s, 'open', %s::jsonb, %s::jsonb, %s)
        """, (
            new_id, user_id, trade_type, symbol, side,
            by_fill_price, by_qty,
            json.dumps(new_signal_data), json.dumps(new_meta), opened_at,
        ))
    cur.close()
    return new_id


# ─────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────
def main() -> int:
    t_start = time.time()
    # 1. Collect Bybit L2 snapshots over WS_SAMPLE_SEC
    snapshots_by_sym = collect_bybit_l2(WS_SAMPLE_SEC)
    n_snap = sum(len(v) for v in snapshots_by_sym.values())
    if n_snap == 0:
        print(f"BYBIT_SHADOW: no WS snapshots received — skipping cycle")
        return 1

    # 2. Open DB
    try:
        con = psycopg2.connect(**DB_CFG)
        con.autocommit = True
    except Exception as e:
        print(f"BYBIT_SHADOW: DB connect error: {e}", file=sys.stderr)
        return 2

    try:
        # 3a. NEW: Update orphaned OPEN bybit mirrors whose delta source CLOSED
        orphans = fetch_orphaned_open_mirrors(con, LOOKBACK_SEC)
        if orphans:
            print(f"BYBIT_SHADOW: found {len(orphans)} orphaned open mirrors to close")
            updated = 0
            for o in orphans:
                try:
                    if update_mirror_with_close(con, o, snapshots_by_sym):
                        updated += 1
                except Exception as e:
                    print(f"BYBIT_SHADOW: orphan close error for {o['mirror_id']}: {e}", file=sys.stderr)
            print(f"BYBIT_SHADOW: closed {updated}/{len(orphans)} orphan mirrors")

        # 3b. Find unmirrored delta shadow trades (creates new mirrors)
        rows = fetch_unmirrored_delta_trades(con, LOOKBACK_SEC)
        if not rows:
            print(f"BYBIT_SHADOW: no new delta trades to mirror (snap={n_snap}, took {time.time()-t_start:.1f}s)")
            return 0

        # 4. For each, simulate Bybit shadow + insert
        inserted = 0
        skipped = 0
        for r in rows:
            (src_id, _user_id, symbol, side, _entry, _exit,
             _qty, _open_at, _close_at, _pnl, _fees, _meta, _sigd,
             _ttype, _st) = r
            by_sym = SYMBOL_MAP.get(symbol)
            if not by_sym:
                skipped += 1
                continue
            snaps = snapshots_by_sym.get(by_sym, [])
            if len(snaps) < 2:
                skipped += 1
                continue
            bp = SYM_BP.get(by_sym, 1.5)
            filled_maker, fill_px, sig_bid, sig_ask, fee_pct = simulate_bybit_fill(
                str(side), snaps, bp
            )
            try:
                insert_bybit_shadow(con, r, fill_px, filled_maker, fee_pct, sig_bid, sig_ask)
                inserted += 1
            except Exception as e:
                print(f"BYBIT_SHADOW: insert error for src={src_id}: {e}", file=sys.stderr)
                skipped += 1

        elapsed = time.time() - t_start
        print(f"BYBIT_SHADOW: inserted={inserted} skipped={skipped} src_rows={len(rows)} snaps={n_snap} elapsed={elapsed:.1f}s")
        return 0
    finally:
        con.close()


def daemon_loop():
    """Long-running mode — same as main() but loops at DAEMON_INTERVAL_SEC.
    Used by the systemd-managed daemon for near-real-time mirroring.
    Reuses one DB connection to avoid reconnect churn.
    """
    print(f"BYBIT_SHADOW_DAEMON: starting (interval={DAEMON_INTERVAL_SEC}s, "
          f"sample={WS_SAMPLE_SEC}s, lookback={LOOKBACK_SEC}s, "
          f"only_closed={ONLY_MIRROR_CLOSED})")
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("BYBIT_SHADOW_DAEMON: interrupted")
            return 0
        except Exception as e:
            print(f"BYBIT_SHADOW_DAEMON: cycle error: {e}", file=sys.stderr)
        time.sleep(DAEMON_INTERVAL_SEC)


if __name__ == "__main__":
    if DAEMON_MODE:
        sys.exit(daemon_loop())
    sys.exit(main())

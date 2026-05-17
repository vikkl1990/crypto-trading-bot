"""
Direct WebSocket client for Delta Exchange India.

Connects to wss://socket.india.delta.exchange for real-time price feeds
and authenticated private channels (orders, positions).

Public channels: v2/ticker (price feeds)
Private channels: orders (fill events), positions (position changes)

Usage:
    ws = DeltaWebSocket(symbols=["BTC/USDT", "ETH/USDT"],
                        api_key="...", api_secret="...")
    ws.on_price = my_callback
    ws.on_order_fill = my_fill_callback
    await ws.connect()
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("bot.delta_ws")  # use bot.* namespace for visibility

# Delta India WebSocket endpoints
WS_URL_PROD = "wss://socket.india.delta.exchange"
WS_URL_DEMO = "wss://socket.testnet.delta.exchange"
WS_URL = WS_URL_PROD  # Default to production

# Symbol mapping: our format → Delta WS format
# FIX C1: Expanded to all active trading pairs so update_real_trades() sees price ticks
# for every pair that can have a real position (previously missing: XRP/LTC/ADA/LINK/DOT/TAO)
# FIX 2026-04-19: Added meme coins with 1000x multiplier prefix (matches
# _DELTA_BASE_OVERRIDE in exchange/ccxt_client.py and data/feed.py).
# Without these, DeltaWS silently drops meme subscriptions → no live price feed
# → scanners can't evaluate memes in real-time → zero meme signals.
SYMBOL_MAP = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
    "AVAX/USDT": "AVAXUSD",
    "SOL/USDT": "SOLUSD",
    "DOGE/USDT": "DOGEUSD",
    "XRP/USDT": "XRPUSD",
    "LTC/USDT": "LTCUSD",
    "ADA/USDT": "ADAUSD",
    "LINK/USDT": "LINKUSD",
    "DOT/USDT": "DOTUSD",
    "TAO/USDT": "TAOUSD",
    # Memes with 1000x multiplier (thin-price contracts on Delta India)
    "PEPE/USDT": "1000PEPEUSD",
    "SHIB/USDT": "1000SHIBUSD",
    "BONK/USDT": "1000BONKUSD",
    "FLOKI/USDT": "1000FLOKIUSD",
    # 1:1 memes (normal contract size)
    "WIF/USDT": "WIFUSD",
    "SUI/USDT": "SUIUSD",
    "NEAR/USDT": "NEARUSD",
    "TRUMP/USDT": "TRUMPUSD",
    "POPCAT/USDT": "POPCATUSD",
    "MEME/USDT": "MEMEUSD",
}

REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}


class DeltaWebSocket:
    """Real-time price feed via Delta Exchange WebSocket.

    Automatically reconnects on disconnect with exponential backoff.
    Falls back gracefully — if WS fails, the REST polling continues.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        on_price: Optional[Callable] = None,
        on_order_fill: Optional[Callable] = None,
        on_position_update: Optional[Callable] = None,
        api_key: str = "",
        api_secret: str = "",
        mode: str = "live",
        ping_interval: int = 25,
        max_reconnects: int = 50,
    ):
        self._symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self._mode = mode  # "demo" or "live"
        # Always use production WS for price feeds (testnet has no real prices)
        # Private channels only work in live mode (prod WS + prod API keys)
        # In demo mode: prices from prod WS, order/position sync via REST polling
        self._ws_url = WS_URL_PROD
        self.on_price = on_price  # callback(symbol, last, bid, ask, mark)
        self.on_order_fill = on_order_fill  # callback(symbol, order_id, client_order_id, fill_price, side, size)
        self.on_position_update = on_position_update  # callback(symbol, size, entry_price, pnl)
        self._api_key = api_key
        self._api_secret = api_secret
        self._ping_interval = ping_interval
        self._max_reconnects = max_reconnects

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._connected = False
        self._authenticated = False
        self._reconnect_count = 0
        self._last_msg_time: float = 0

        # Latest prices (thread-safe via asyncio single-thread)
        self.prices: Dict[str, float] = {}
        self.bids: Dict[str, float] = {}
        self.asks: Dict[str, float] = {}
        self.marks: Dict[str, float] = {}

        # Phase 4.3 — live-forming bar cache. Each symbol has a dict per
        # resolution holding the MOST RECENT bar's OHLCV + high-water-marks.
        # Consumers read:
        #   candles_1m[symbol] → {open, high, low, close, volume, ts, bar_start}
        #   candles_5m[symbol] → same schema
        # We keep only the latest bar per resolution (monitor only needs
        # highs/lows for MFE capture, not historical series).
        self.candles_1m: Dict[str, Dict] = {}
        self.candles_5m: Dict[str, Dict] = {}

        # Phase 5.2 (2026-04-23) — TICK-LEVEL rolling high/low.
        # Delta candle updates arrive at ~1 Hz; ticker updates at ~10 Hz.
        # A spike that rises 0.20R and reverts in 300ms NEVER appears in
        # the candle.high before the bar closes (some candle updates lag).
        # Monitor was missing those spikes → demo peak_mfe_r capped ~0.25R
        # while paper saw 0.35R+ on identical setups.
        # Schema:
        #   tick_highs[symbol] = highest `last` seen since window_start
        #   tick_lows[symbol]  = lowest `last` seen since window_start
        #   tick_window_start[symbol] = unix-seconds when window opened
        # Window resets every TICK_WINDOW_SEC so monitors can detect
        # post-entry-only highs (only trust if window_start >= trade.opened_at).
        self.tick_highs: Dict[str, float] = {}
        self.tick_lows: Dict[str, float] = {}
        self.tick_window_start: Dict[str, float] = {}
        self.TICK_WINDOW_SEC: float = 60.0

        # Phase 5.6-ComponentA (2026-04-24) — PRODUCTION L2 ORDERBOOK.
        # Read-only subscription to l2_orderbook channel. Does NOT affect
        # execution path. Purpose: diagnostic data showing whether
        # PRODUCTION orderbook depth supports our intended maker offsets.
        # Because our WS_URL is already production (see line ~95-97), L2
        # updates come from the REAL market, not testnet. Whether our
        # demo orders on testnet get filled is still bound by testnet
        # books — but we now KNOW what they would do on live.
        # Schema:
        #   l2_orderbook[symbol] = {
        #     "bids": [(price, size), ...],   # descending price
        #     "asks": [(price, size), ...],   # ascending price
        #     "ts":   unix_seconds_updated
        #   }
        # Top ~20 levels per side typically.
        self.l2_orderbook: Dict[str, Dict] = {}
        self.l2_msg_count: int = 0
        self._l2_first_snapshot_logged: Dict[str, bool] = {}

        # Stats
        self.msg_count: int = 0
        self.candle_msg_count: int = 0
        self.private_msg_count: int = 0
        self.connect_time: float = 0
        self.avg_latency_ms: float = 0
        self._latencies: List[float] = []

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed

    async def connect(self) -> None:
        """Start the WebSocket connection loop."""
        self._running = True
        self._session = aiohttp.ClientSession()
        asyncio.create_task(self._connection_loop(), name="delta-ws")
        logger.info("DeltaWebSocket starting for %s", self._symbols)

    async def close(self) -> None:
        """Gracefully shut down."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._connected = False
        logger.info("DeltaWebSocket closed")

    async def _connection_loop(self) -> None:
        """Main loop: connect, subscribe, read messages, reconnect on failure."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self._reconnect_count += 1

                if self._reconnect_count > self._max_reconnects:
                    logger.error(
                        "DeltaWS: exceeded %d reconnect attempts. Giving up.",
                        self._max_reconnects,
                    )
                    break

                delay = min(2 ** min(self._reconnect_count, 6), 60)
                logger.warning(
                    "DeltaWS disconnected (%s). Reconnect %d/%d in %ds...",
                    exc, self._reconnect_count, self._max_reconnects, delay,
                )
                await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        """Single connection lifecycle: connect → auth → subscribe → read."""
        logger.info("DeltaWS connecting to %s (mode=%s) ...", self._ws_url, self._mode)

        async with self._session.ws_connect(
            self._ws_url,
            heartbeat=self._ping_interval,
            timeout=30,
        ) as ws:
            self._ws = ws
            self._connected = True
            self._authenticated = False
            self._reconnect_count = 0
            self.connect_time = time.time()
            logger.info("DeltaWS connected!")

            # Authenticate for private channels (orders, positions)
            # Only in LIVE mode — demo keys don't work on production WS
            if self._api_key and self._api_secret and self._mode == "live":
                await self._authenticate(ws)
            elif self._mode == "demo":
                logger.info("DeltaWS: skipping auth (demo mode — private channels via REST)")
                self._authenticated = False

            # Subscribe to ticker channels + private channels
            await self._subscribe(ws)

            # Read messages
            async for msg in ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._last_msg_time = time.time()
                    await self._handle_message(msg.data)

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("DeltaWS: connection closed/error: %s", msg.data)
                    break

        self._connected = False
        # Phase 5.20-B4 (2026-04-25) — fire on_disconnect callback so subscribers
        # (e.g. risk manager) can react: cancel resting orders, pause new entries,
        # alert on degraded mode. Failsafe: callback errors don't crash WS loop.
        cb = getattr(self, "on_disconnect", None)
        if cb is not None:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb({"reason": "ws_closed_or_error", "ts": time.time()})
                else:
                    cb({"reason": "ws_closed_or_error", "ts": time.time()})
            except Exception as _cb_e:
                logger.error("DeltaWS on_disconnect callback failed: %s", _cb_e)

    async def _authenticate(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Authenticate for private channels using HMAC-SHA256.

        Phase 5.9-D (2026-04-24) — MIGRATED FROM "auth" TO "key-auth".
        Delta deprecated the old {"type":"auth"} method on 2025-12-31.
        New method uses {"type":"key-auth"} with identical signature
        formula: HMAC-SHA256(secret, "GET" + timestamp + "/live") → hex.
        Timestamp is Unix SECONDS stringified (not ms).
        Source: https://community.delta.exchange/t/api-changelog-08-october-2025/2171
        """
        try:
            timestamp = str(int(time.time()))
            signature_data = f"GET{timestamp}/live"
            signature = hmac.new(
                self._api_secret.encode("utf-8"),
                signature_data.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            auth_msg = {
                "type": "key-auth",  # Phase 5.9-D: was "auth" pre-2026-04-24
                "payload": {
                    "api-key": self._api_key,
                    "signature": signature,
                    "timestamp": timestamp,
                }
            }
            await ws.send_json(auth_msg)
            self._authenticated = True
            logger.info("DeltaWS: key-auth message sent (awaiting confirmation)")
        except Exception as e:
            logger.error("DeltaWS: authentication failed: %s", e)
            self._authenticated = False

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Subscribe to public ticker + candlestick + private channels.

        Phase 4.3 (2026-04-22) — added candlestick_1m + candlestick_5m
        so `self.candles` tracks live-forming bar highs/lows. This feeds
        UserRealManager._monitor_trade's candle-high MFE tracker, which
        on REST-only was missing 16-second intraband spikes.
        """
        delta_symbols = []
        for sym in self._symbols:
            ds = SYMBOL_MAP.get(sym)
            if ds:
                delta_symbols.append(ds)
            else:
                logger.warning("DeltaWS: unknown symbol mapping for %s", sym)

        if not delta_symbols:
            return

        # Public channels: ticker + candlesticks + L2 orderbook (Phase 5.6-A)
        channels = [
            {"name": "v2/ticker", "symbols": delta_symbols},
            {"name": "candlestick_1m", "symbols": delta_symbols},
            {"name": "candlestick_5m", "symbols": delta_symbols},
            # Phase 5.6-A (2026-04-24) — PRODUCTION L2 orderbook (read-only).
            # This subscribes on the production WS URL (always prod per
            # line ~95-97), giving us REAL-market depth data regardless
            # of demo/live mode. Used for diagnostics (maker-fill
            # probability analysis) — does NOT affect order placement yet.
            {"name": "l2_orderbook", "symbols": delta_symbols},
        ]

        # Private channels (requires auth) — need symbol arrays
        # Phase 5.9-D (2026-04-24) — expanded private channel set:
        #   orders         — order status transitions (placed/filled/cancelled)
        #   positions      — position size/entry/liquidation updates
        #   v2/user_trades — fill events (preferred over legacy user_trades)
        #   margins        — available margin/wallet balance deltas
        # These replace REST polling for state (get_position, balance fetch)
        # in live mode, cutting pre-close latency from 200-500ms to ~0ms.
        if self._authenticated:
            channels.append({"name": "orders", "symbols": delta_symbols})
            channels.append({"name": "positions", "symbols": delta_symbols})
            channels.append({"name": "v2/user_trades", "symbols": delta_symbols})
            channels.append({"name": "margins", "symbols": ["all"]})
            logger.info("DeltaWS: subscribing to private channels (orders, positions, v2/user_trades, margins)")

        subscribe_msg = {
            "type": "subscribe",
            "payload": {"channels": channels}
        }

        await ws.send_json(subscribe_msg)
        logger.info("DeltaWS subscribed: ticker+candles for %s | private=%s",
                    delta_symbols, self._authenticated)

    async def _handle_message(self, raw: str) -> None:
        """Parse incoming WebSocket message and update prices / handle private events."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", "")

        # Subscription confirmation
        if msg_type == "subscriptions":
            logger.info("DeltaWS subscription confirmed: %s", data)
            return

        # Auth confirmation — accept both "auth" (legacy) and "key-auth" (5.9-D)
        # to be robust during migration period.
        if msg_type in ("auth", "key-auth"):
            if data.get("success"):
                logger.info("DeltaWS: authenticated successfully (%s)", msg_type)
            else:
                logger.error("DeltaWS: auth failed (%s): %s", msg_type, data)
                self._authenticated = False
            return

        # ── PUBLIC: Ticker update ──
        if msg_type == "v2/ticker":
            await self._handle_ticker(data)
            return

        # ── PUBLIC: L2 Orderbook update (Phase 5.6-A) ──
        # Delta's l2_orderbook channel sends full book snapshots (usually
        # 20+ levels per side) on each update. Read-only for now — populates
        # self.l2_orderbook cache for diagnostic consumers.
        if msg_type == "l2_orderbook":
            await self._handle_l2(data)
            return

        # ── PUBLIC: Candlestick update (Phase 4.3) ──
        if msg_type == "candlestick_1m":
            await self._handle_candle(data, resolution="1m")
            return
        if msg_type == "candlestick_5m":
            await self._handle_candle(data, resolution="5m")
            return

        # ── PRIVATE: Order events ──
        if msg_type == "orders":
            await self._handle_order_event(data)
            return

        # ── PRIVATE: Position events ──
        if msg_type == "positions":
            await self._handle_position_event(data)
            return

        # ── PRIVATE: User-trade fills (Phase 5.9-D) ──
        # v2/user_trades is the preferred fill stream (vs legacy user_trades).
        # Gives us immediate fill confirmation with exact price/fee/taker flag.
        # Currently logged only — future session will wire callbacks to
        # update UserTradeRecord.entry_fill_price / fee_usd from here instead
        # of parsing REST order responses.
        if msg_type == "v2/user_trades":
            await self._handle_user_trade_event(data)
            return

        # ── PRIVATE: Margin/balance deltas (Phase 5.9-D) ──
        # Fires on any wallet balance change (fills, funding, withdrawals).
        # Replaces 60s-cached REST balance fetch — real-time margin awareness
        # for live-mode pre-trade sizing decisions.
        if msg_type == "margins":
            await self._handle_margins_event(data)
            return

    async def _handle_ticker(self, data: dict) -> None:
        """Process ticker price update."""
        delta_symbol = data.get("symbol", "")
        symbol = REVERSE_MAP.get(delta_symbol, "")

        if not symbol:
            return

        try:
            last = float(data.get("close", 0))
            mark = float(data.get("mark_price", last))
            quotes = data.get("quotes", {})
            bid = float(quotes.get("best_bid", last))
            ask = float(quotes.get("best_ask", last))
        except (ValueError, TypeError):
            return

        if last <= 0:
            return

        # Update price cache
        self.prices[symbol] = last
        self.bids[symbol] = bid
        self.asks[symbol] = ask
        self.marks[symbol] = mark
        self.msg_count += 1

        # Phase 5.2 — tick-level rolling high/low (windowed reset).
        # See __init__ for the rationale. Window resets every TICK_WINDOW_SEC
        # so consumers can detect post-entry-only highs by comparing
        # tick_window_start[sym] >= trade.opened_at.
        _now = time.time()
        _ws = self.tick_window_start.get(symbol, 0.0)
        if _ws == 0.0 or (_now - _ws) > self.TICK_WINDOW_SEC:
            self.tick_highs[symbol] = last
            self.tick_lows[symbol] = last
            self.tick_window_start[symbol] = _now
        else:
            _h = self.tick_highs.get(symbol, last)
            _l = self.tick_lows.get(symbol, last)
            if last > _h:
                self.tick_highs[symbol] = last
            if last < _l:
                self.tick_lows[symbol] = last

        # Track latency
        ts = data.get("timestamp")
        if ts:
            try:
                latency = (time.time() * 1000) - float(ts)
                self._latencies.append(latency)
                if len(self._latencies) > 100:
                    self._latencies = self._latencies[-100:]
                self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
            except Exception:
                pass

        # Fire callback
        if self.on_price:
            try:
                await self.on_price(symbol, last, bid, ask, mark)
            except Exception as exc:
                logger.debug("DeltaWS price callback error: %s", exc)

    async def _handle_candle(self, data: dict, resolution: str = "1m") -> None:
        """Phase 4.3 — parse live candlestick update.

        Delta's candlestick channel sends the in-forming bar with OHLCV
        fields as strings. We keep only the latest bar per symbol/res and
        let consumers (UserRealManager._monitor_trade) read highs/lows
        for accurate peak-MFE tracking without waiting for REST polling.

        Payload fields (Delta v2 spec):
          symbol, resolution, candle_start_time, open, high, low, close,
          volume, timestamp
        """
        delta_symbol = data.get("symbol", "")
        symbol = REVERSE_MAP.get(delta_symbol, "")
        if not symbol:
            return
        try:
            bar = {
                "open":       float(data.get("open", 0) or 0),
                "high":       float(data.get("high", 0) or 0),
                "low":        float(data.get("low", 0) or 0),
                "close":      float(data.get("close", 0) or 0),
                "volume":     float(data.get("volume", 0) or 0),
                "bar_start":  int(data.get("candle_start_time", 0) or 0),
                "ts":         int(data.get("timestamp", time.time() * 1e6)),
            }
        except (ValueError, TypeError):
            return
        if bar["high"] <= 0:
            return
        target = self.candles_1m if resolution == "1m" else self.candles_5m
        target[symbol] = bar
        self.candle_msg_count += 1

    async def _handle_l2(self, data: dict) -> None:
        """Phase 5.6-A (2026-04-24) — parse production L2 orderbook update.

        Delta's l2_orderbook payload shape (per Delta India docs):
          {
            "type": "l2_orderbook",
            "symbol": "BTCUSD",
            "buy":  [["78000.5", "42"], ["78000.0", "128"], ...],   # descending
            "sell": [["78001.0", "35"], ["78001.5", "94"],  ...],   # ascending
            "timestamp": 1729800000123456,
            "sequence_no": 12345,
          }
        Some deployments use "bids"/"asks" instead of "buy"/"sell" — handle both.

        Parses into self.l2_orderbook[symbol] = {bids, asks, ts}.
        Bids are (price, size) tuples sorted descending by price.
        Asks are ascending. Consumers query via get_cumulative_depth().

        Fail-safe: if payload structure differs, log once per symbol and skip.
        """
        delta_symbol = data.get("symbol", "")
        symbol = REVERSE_MAP.get(delta_symbol, "")
        if not symbol:
            return
        try:
            # Delta India actual format: list of dicts with limit_price + size + depth
            # Example: [{"depth": "1584", "limit_price": "77885.5", "size": 1584}, ...]
            raw_bids = data.get("buy") or data.get("bids") or []
            raw_asks = data.get("sell") or data.get("asks") or []
            bids = [
                (float(lvl.get("limit_price", 0)), float(lvl.get("size", 0)))
                for lvl in raw_bids[:20]
                if isinstance(lvl, dict) and lvl.get("limit_price")
            ]
            asks = [
                (float(lvl.get("limit_price", 0)), float(lvl.get("size", 0)))
                for lvl in raw_asks[:20]
                if isinstance(lvl, dict) and lvl.get("limit_price")
            ]
            # Guard: bids should be descending, asks ascending
            bids.sort(key=lambda x: -x[0])
            asks.sort(key=lambda x: x[0])
            self.l2_orderbook[symbol] = {
                "bids": bids,
                "asks": asks,
                "ts": time.time(),
            }
            self.l2_msg_count += 1
            # Log first snapshot per symbol at WARNING for startup visibility
            if not self._l2_first_snapshot_logged.get(symbol):
                self._l2_first_snapshot_logged[symbol] = True
                logger.warning(
                    "L2_INIT %s | bids_top=%s asks_top=%s depth=%d/%d",
                    symbol,
                    f"{bids[0][0]:.4f}@{bids[0][1]:.0f}" if bids else "none",
                    f"{asks[0][0]:.4f}@{asks[0][1]:.0f}" if asks else "none",
                    len(bids), len(asks),
                )
        except (ValueError, TypeError, IndexError) as e:
            # Unexpected payload shape — log once then stop noise
            if not self._l2_first_snapshot_logged.get(symbol):
                self._l2_first_snapshot_logged[symbol] = True
                logger.warning("L2_INIT %s | parse FAIL (structure unexpected): %s | sample=%s",
                               symbol, e, str(data)[:200])

    def cumulative_depth(self, symbol: str, side: str, price_through: float, n_levels: int = 10) -> float:
        """Phase 5.6-A — sum of liquidity from top of book through `price_through`.

        Args:
            symbol: internal symbol (e.g. "BTC/USDT")
            side:   "bid" (buy-side, for shorting) or "ask" (sell-side, for longing)
            price_through: the offset price we're interested in. For a LONG
                           maker entry, this is our proposed bid. We want to
                           know cumulative BID depth above this price (how much
                           liquidity we'd be queued behind).
            n_levels: max levels to look at (defaults 10)

        Returns:
            Cumulative size (lots) from top of book down to price_through.
            Returns 0 if no L2 data cached yet.

        Usage:
            # Before placing post_only buy at 78245:
            depth = delta_ws.cumulative_depth("BTC/USDT", "bid", 78245)
            if depth < 50:  # not enough liquidity ahead of us
                # Skip maker, go market
        """
        book = self.l2_orderbook.get(symbol)
        if not book:
            return 0.0
        if side == "bid":
            # Want cumulative bid volume at or above price_through
            return sum(q for p, q in book["bids"][:n_levels] if p >= price_through)
        else:
            # "ask" — cumulative ask volume at or below price_through
            return sum(q for p, q in book["asks"][:n_levels] if p <= price_through)

    async def _handle_order_event(self, data: dict) -> None:
        """Process private order channel events (SL/TP fills).

        Fires on_order_fill callback when a reduce_only order is filled,
        which means SL or TP was hit on the exchange.
        """
        self.private_msg_count += 1

        # Order state: open, pending, closed, cancelled
        state = data.get("state", "")
        if state not in ("closed",):
            return  # Only care about filled orders

        # Only process reduce_only fills (SL/TP exits, not entries)
        if data.get("reduce_only") != True and str(data.get("reduce_only", "")).lower() != "true":
            return

        product_symbol = data.get("product_symbol", "")
        symbol = REVERSE_MAP.get(product_symbol, "")
        order_id = str(data.get("id", ""))
        client_order_id = data.get("client_order_id", "")
        side = data.get("side", "")
        size = int(data.get("size", 0) or 0)

        # Get fill price
        fill_price = float(data.get("average_fill_price", 0) or data.get("limit_price", 0) or 0)

        logger.info(
            "DeltaWS ORDER FILL: %s %s %d lots @ %.4f | order=%s | coid=%s",
            symbol or product_symbol, side, size, fill_price,
            order_id[:12], client_order_id[:12] if client_order_id else "-",
        )

        if self.on_order_fill and fill_price > 0:
            try:
                await self.on_order_fill(
                    symbol=symbol or product_symbol,
                    order_id=order_id,
                    client_order_id=client_order_id,
                    fill_price=fill_price,
                    side=side,
                    size=size,
                )
            except Exception as exc:
                logger.error("DeltaWS order fill callback error: %s", exc)

    async def _handle_position_event(self, data: dict) -> None:
        """Process private position channel events.

        Fires on_position_update callback when a position closes (size=0),
        which catches liquidations and exchange-side closes.
        """
        self.private_msg_count += 1

        product_symbol = data.get("product_symbol", data.get("symbol", ""))
        symbol = REVERSE_MAP.get(product_symbol, "")
        size = int(data.get("size", 0) or 0)
        entry_price = float(data.get("entry_price", 0) or 0)
        pnl = float(data.get("realized_pnl", 0) or data.get("pnl", 0) or 0)

        logger.info(
            "DeltaWS POSITION UPDATE: %s | size=%d entry=%.4f pnl=%.4f",
            symbol or product_symbol, size, entry_price, pnl,
        )

        if self.on_position_update:
            try:
                await self.on_position_update(
                    symbol=symbol or product_symbol,
                    size=size,
                    entry_price=entry_price,
                    pnl=pnl,
                )
            except Exception as exc:
                logger.error("DeltaWS position update callback error: %s", exc)

    async def _handle_user_trade_event(self, data: dict) -> None:
        """Process v2/user_trades fill events (Phase 5.9-D).

        Fires for every fill on the authenticated account. Data shape:
          {"symbol": "BTCUSD", "price": "78123.5", "size": 3,
           "role": "maker"|"taker", "commission": "0.0189", "side": "buy",
           "order_id": 12345, "client_order_id": "vn_..."}
        Currently log-only; callback wiring deferred to avoid racing with
        existing REST-based fill parsing in user_real_manager.
        """
        self.private_msg_count += 1
        try:
            product_symbol = data.get("product_symbol", data.get("symbol", ""))
            role = data.get("role", "?")
            price = float(data.get("price", 0) or 0)
            size = int(data.get("size", 0) or 0)
            commission = float(data.get("commission", 0) or 0)
            coid = str(data.get("client_order_id", "") or "")[:16]
            logger.info(
                "DeltaWS USER_TRADE: %s %s %d @ %.4f | role=%s fee=%.4f coid=%s",
                product_symbol, data.get("side", "?"), size, price,
                role, commission, coid or "-",
            )
        except Exception as e:
            logger.warning("DeltaWS user_trade handler error: %s", e)

    async def _handle_margins_event(self, data: dict) -> None:
        """Process margins/wallet balance events (Phase 5.9-D).

        Fires on any balance delta. Data shape varies but typically has
        available_balance, order_margin, position_margin fields.
        Currently log-only; future session will cache values to replace
        60s REST balance polling in user_real_manager.py.
        """
        self.private_msg_count += 1
        try:
            avail = float(data.get("available_balance", 0) or 0)
            pos_m = float(data.get("position_margin", 0) or 0)
            ord_m = float(data.get("order_margin", 0) or 0)
            logger.info(
                "DeltaWS MARGINS: avail=%.4f pos_margin=%.4f order_margin=%.4f",
                avail, pos_m, ord_m,
            )
        except Exception as e:
            logger.warning("DeltaWS margins handler error: %s", e)

    def get_status(self) -> Dict:
        """Return WebSocket status for dashboard."""
        return {
            "connected": self.is_connected,
            "authenticated": self._authenticated,
            "url": WS_URL,
            "symbols": self._symbols,
            "msg_count": self.msg_count,
            "candle_msg_count": self.candle_msg_count,
            "private_msg_count": self.private_msg_count,
            "reconnects": self._reconnect_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_msg_ago": round(time.time() - self._last_msg_time, 1) if self._last_msg_time else None,
            "uptime_sec": round(time.time() - self.connect_time) if self.connect_time else 0,
            "prices": dict(self.prices),
            "candles_1m_symbols": list(self.candles_1m.keys()),
            "candles_5m_symbols": list(self.candles_5m.keys()),
        }

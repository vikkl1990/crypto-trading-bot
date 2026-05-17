"""
Bybit V5 Private WebSocket Client — order placement + position/wallet feed.

Bypasses the CloudFront REST geo-block by going through stream.bybit.com (verified
reachable from Oracle Cloud India). Supports:
  - HMAC-SHA256 auth on connect
  - Order submit / cancel via WS messages
  - Subscriptions: order, position, wallet, execution
  - Async (uses 'websockets' library, not 'websocket-client')
  - Reconnect with exp backoff

Bybit V5 docs:
  - Auth:  https://bybit-exchange.github.io/docs/v5/ws/connect
  - Trade: https://bybit-exchange.github.io/docs/v5/order/create-order  (WS variant)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:  # pragma: no cover
    websockets = None
    ConnectionClosed = WebSocketException = Exception

logger = logging.getLogger("bybit_private_ws")

WS_PRIVATE_LIVE = "wss://stream.bybit.com/v5/private"
WS_TRADE_LIVE = "wss://stream.bybit.com/v5/trade"   # dedicated trade-only WS (faster)


@dataclass
class BybitFillEvent:
    symbol: str
    side: str
    qty: float
    price: float
    order_id: str
    client_order_id: str
    fee: float
    fee_currency: str
    ts_ms: int
    raw: Dict[str, Any] = field(default_factory=dict)


class BybitPrivateWS:
    """One instance per user (each user has own auth + own WS connection)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        on_fill: Optional[Callable[[BybitFillEvent], Awaitable[None]]] = None,
        on_position: Optional[Callable[[Dict], Awaitable[None]]] = None,
        on_wallet: Optional[Callable[[Dict], Awaitable[None]]] = None,
        ping_interval: int = 20,
        max_reconnects: int = 50,
    ):
        if not websockets:
            raise RuntimeError("`websockets` package required (pip install websockets)")
        self._api_key = api_key
        self._api_secret = api_secret
        self.on_fill = on_fill
        self.on_position = on_position
        self.on_wallet = on_wallet
        self._ping_interval = ping_interval
        self._max_reconnects = max_reconnects

        self._ws: Optional[Any] = None
        self._connected = False
        self._authenticated = False
        self._reconnects = 0
        self._stop = asyncio.Event()
        self._req_futures: Dict[str, asyncio.Future] = {}   # req_id → Future

    # ── Auth helpers ───────────────────────────────────────────────
    def _sign(self, expires_ms: int) -> str:
        """HMAC-SHA256(secret, 'GET/realtime' + expires)."""
        msg = f"GET/realtime{expires_ms}"
        return hmac.new(
            self._api_secret.encode(), msg.encode(), hashlib.sha256
        ).hexdigest()

    async def _auth(self) -> bool:
        expires = int((time.time() + 60) * 1000)
        sig = self._sign(expires)
        msg = {"op": "auth", "args": [self._api_key, expires, sig]}
        await self._ws.send(json.dumps(msg))
        # Read auth response
        for _ in range(5):
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            d = json.loads(raw)
            if d.get("op") == "auth":
                if d.get("success"):
                    self._authenticated = True
                    logger.info("BYBIT_PRIV_WS: authenticated as %s...", self._api_key[:8])
                    return True
                logger.error("BYBIT_PRIV_WS: auth FAILED: %s", d.get("ret_msg"))
                return False
        logger.error("BYBIT_PRIV_WS: no auth response")
        return False

    async def _subscribe(self, topics: List[str]) -> None:
        msg = {"op": "subscribe", "args": topics}
        await self._ws.send(json.dumps(msg))

    # ── Trade ops ──────────────────────────────────────────────────
    async def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",   # "Market" | "Limit"
        price: Optional[float] = None,
        time_in_force: str = "IOC",   # "GTC" | "IOC" | "FOK" | "PostOnly"
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
        category: str = "linear",
    ) -> Dict[str, Any]:
        """Submit order via WS. Returns Bybit response dict.

        symbol: Bybit format e.g. 'BTCUSDT'
        side:   'Buy' | 'Sell' (Bybit casing)
        qty:    base currency (BTC, ETH, etc.)
        """
        if not self._connected or not self._authenticated:
            return {"success": False, "error": "not_connected_or_auth"}
        coid = client_order_id or f"vne_{uuid.uuid4().hex[:16]}"
        req_id = uuid.uuid4().hex
        args = [{
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force,
            "orderLinkId": coid,
            "reduceOnly": reduce_only,
        }]
        if price is not None and order_type == "Limit":
            args[0]["price"] = str(price)
        msg = {"reqId": req_id, "header": {"X-BAPI-TIMESTAMP": str(int(time.time() * 1000))},
               "op": "order.create", "args": args}
        fut = asyncio.get_event_loop().create_future()
        self._req_futures[req_id] = fut
        await self._ws.send(json.dumps(msg))
        try:
            resp = await asyncio.wait_for(fut, timeout=8)
            return resp
        except asyncio.TimeoutError:
            return {"success": False, "error": "ws_response_timeout", "client_order_id": coid}
        finally:
            self._req_futures.pop(req_id, None)

    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        category: str = "linear",
    ) -> Dict[str, Any]:
        if not self._connected or not self._authenticated:
            return {"success": False, "error": "not_connected_or_auth"}
        req_id = uuid.uuid4().hex
        args = [{"category": category, "symbol": symbol}]
        if order_id:
            args[0]["orderId"] = order_id
        if client_order_id:
            args[0]["orderLinkId"] = client_order_id
        msg = {"reqId": req_id, "op": "order.cancel", "args": args}
        fut = asyncio.get_event_loop().create_future()
        self._req_futures[req_id] = fut
        await self._ws.send(json.dumps(msg))
        try:
            return await asyncio.wait_for(fut, timeout=8)
        except asyncio.TimeoutError:
            return {"success": False, "error": "ws_response_timeout"}
        finally:
            self._req_futures.pop(req_id, None)

    # ── Lifecycle ──────────────────────────────────────────────────
    async def connect_and_run(self) -> None:
        """Long-running: maintain auth + subscriptions + dispatch events."""
        while not self._stop.is_set() and self._reconnects < self._max_reconnects:
            try:
                async with websockets.connect(
                    WS_TRADE_LIVE,        # dedicated trade WS for order ops
                    ping_interval=self._ping_interval,
                    open_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnects = 0
                    logger.info("BYBIT_PRIV_WS: connected, authenticating...")
                    if not await self._auth():
                        await asyncio.sleep(5)
                        continue
                    # Subscribe to fills/positions/wallet on a SECOND connection
                    # (Bybit splits trade WS from stream WS in V5)
                    asyncio.create_task(self._stream_run())
                    # Read loop on trade WS
                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            d = json.loads(raw)
                            req_id = d.get("reqId")
                            if req_id and req_id in self._req_futures:
                                self._req_futures[req_id].set_result(d)
                        except asyncio.TimeoutError:
                            continue
                        except (ConnectionClosed, WebSocketException):
                            break
            except Exception as e:
                logger.warning("BYBIT_PRIV_WS: connect error: %s", e)
            self._connected = False
            self._authenticated = False
            self._reconnects += 1
            backoff = min(60, 2 ** self._reconnects)
            logger.info("BYBIT_PRIV_WS: reconnecting in %ds (attempt %d)", backoff, self._reconnects)
            await asyncio.sleep(backoff)

    async def _stream_run(self) -> None:
        """Separate WS for stream subscriptions (order/position/wallet/execution)."""
        while not self._stop.is_set():
            try:
                async with websockets.connect(WS_PRIVATE_LIVE, ping_interval=self._ping_interval, open_timeout=10) as ws:
                    # Auth
                    expires = int((time.time() + 60) * 1000)
                    sig = self._sign(expires)
                    await ws.send(json.dumps({"op": "auth", "args": [self._api_key, expires, sig]}))
                    auth_ok = False
                    for _ in range(5):
                        raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        d = json.loads(raw)
                        if d.get("op") == "auth":
                            auth_ok = bool(d.get("success"))
                            break
                    if not auth_ok:
                        logger.error("BYBIT_PRIV_WS stream: auth failed")
                        return
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": ["execution.linear", "order.linear", "position.linear", "wallet"],
                    }))
                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                            d = json.loads(raw)
                            await self._dispatch_stream_msg(d)
                        except asyncio.TimeoutError:
                            continue
                        except (ConnectionClosed, WebSocketException):
                            break
            except Exception as e:
                logger.warning("BYBIT_PRIV_WS stream: %s", e)
            await asyncio.sleep(5)

    async def _dispatch_stream_msg(self, d: Dict) -> None:
        topic = d.get("topic", "")
        data = d.get("data", [])
        try:
            if topic.startswith("execution") and self.on_fill:
                for item in data:
                    fill = BybitFillEvent(
                        symbol=item.get("symbol", ""),
                        side=item.get("side", ""),
                        qty=float(item.get("execQty", 0) or 0),
                        price=float(item.get("execPrice", 0) or 0),
                        order_id=item.get("orderId", ""),
                        client_order_id=item.get("orderLinkId", ""),
                        fee=float(item.get("execFee", 0) or 0),
                        fee_currency=item.get("feeCurrency", ""),
                        ts_ms=int(item.get("execTime", 0) or 0),
                        raw=item,
                    )
                    await self.on_fill(fill)
            elif topic.startswith("position") and self.on_position:
                for item in data:
                    await self.on_position(item)
            elif topic.startswith("wallet") and self.on_wallet:
                for item in data:
                    await self.on_wallet(item)
        except Exception as e:
            logger.error("BYBIT_PRIV_WS dispatch: %s", e)

    async def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass


# ── Standalone smoke test (no real keys → demonstrates auth flow) ─
async def _smoke_test() -> None:
    """Without keys: just verify connection. With keys (env vars): test auth."""
    import os
    key = os.getenv("BYBIT_API_KEY", "")
    secret = os.getenv("BYBIT_API_SECRET", "")
    if not key or not secret:
        print("SMOKE: no BYBIT_API_KEY env — connection-only test")
        try:
            async with websockets.connect(WS_TRADE_LIVE, open_timeout=10) as ws:
                print(f"SMOKE: ✓ connected to {WS_TRADE_LIVE} (auth would follow)")
        except Exception as e:
            print(f"SMOKE: ✗ {e}")
        return
    print(f"SMOKE: with key {key[:8]}... — full auth test")
    cli = BybitPrivateWS(api_key=key, api_secret=secret)

    async def on_fill(f):
        print(f"  FILL: {f.symbol} {f.side} qty={f.qty} px={f.price} fee={f.fee}")

    cli.on_fill = on_fill
    runner = asyncio.create_task(cli.connect_and_run())
    await asyncio.sleep(8)   # let auth + subscribe complete
    print("SMOKE: connected:", cli._connected, "auth:", cli._authenticated)
    await cli.stop()
    runner.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke_test())

"""
UserBybitExecutor — places real orders on Bybit when the user's
`users.bybit_mode = 'live'` and they have an active Bybit live key.

Independent of the existing Delta execution path. Called from the orchestrator's
signal-dispatch flow PARALLEL to the Delta UserRealManager.

Flow per signal:
  1. Look up user's Bybit credentials from user_api_keys (bybit, label='live')
  2. Lazily start a private WS connection (one per user, persistent)
  3. Submit market order: side, qty (converted to base currency)
  4. Wait for execution event → record in user_trades (exchange='bybit', trade_type='real')
  5. SL/TP managed app-side (poll position via stream subscription)

For MVP: market orders only. Maker/limit logic added in a follow-up.

Symbol mapping:
   BTC/USDT → BTCUSDT  (Bybit linear perp)
   ETH/USDT → ETHUSDT
   SOL/USDT → SOLUSDT
   XRP/USDT → XRPUSDT
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger("user_bybit_executor")

SYMBOL_MAP = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USDT": "XRPUSDT",
    "ADA/USDT": "ADAUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "DOT/USDT": "DOTUSDT",
    "LTC/USDT": "LTCUSDT",
    "AVAX/USDT": "AVAXUSDT",
    "LINK/USDT": "LINKUSDT",
    "TAO/USDT": "TAOUSDT",
    "NEAR/USDT": "NEARUSDT",
    "SUI/USDT": "SUIUSDT",
    "WIF/USDT": "WIFUSDT",
    "TRUMP/USDT": "TRUMPUSDT",
    "PEPE/USDT": "1000PEPEUSDT",   # Bybit lists PEPE as 1000PEPEUSDT
    "SHIB/USDT": "1000SHIBUSDT",
    "BONK/USDT": "1000BONKUSDT",
    "POPCAT/USDT": "POPCATUSDT",
    "MEME/USDT": "MEMEUSDT",
}


class UserBybitExecutor:
    """One per user. Lazy-starts WS on first dispatch."""

    def __init__(self, user_id: str, user_email: str, db_pool):
        self.user_id = user_id
        self.user_email = user_email
        self._db_pool = db_pool
        self._ws_client = None      # type: Any
        self._ws_task: Optional[asyncio.Task] = None
        self._api_key = ""
        self._api_secret = ""
        self._loaded = False

    async def _load_keys(self) -> bool:
        """Load + decrypt user's Bybit live API keys from user_api_keys."""
        try:
            async with self._db_pool.acquire() as con:
                row = await con.fetchrow(
                    """SELECT api_key_enc, api_secret_enc FROM user_api_keys
                       WHERE user_id=$1 AND exchange='bybit' AND label='live' AND is_active=TRUE
                       LIMIT 1""",
                    self.user_id
                )
            if not row:
                logger.info("BYBIT_EXEC: user %s has no active Bybit live key", self.user_email)
                return False
            from cryptography.fernet import Fernet
            import os
            key_str = os.environ.get("FERNET_KEY", "")
            if not key_str:
                try:
                    for line in open("/home/opc/crypto-trading-bot/.env"):
                        if line.startswith("FERNET_KEY="):
                            key_str = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                except Exception:
                    pass
            if not key_str:
                logger.error("BYBIT_EXEC: no FERNET_KEY available — can't decrypt")
                return False
            f = Fernet(key_str.encode() if isinstance(key_str, str) else key_str)
            self._api_key = f.decrypt(row["api_key_enc"].encode()).decode()
            self._api_secret = f.decrypt(row["api_secret_enc"].encode()).decode()
            self._loaded = True
            return True
        except Exception as e:
            logger.error("BYBIT_EXEC: key load failed for %s: %s", self.user_email, e)
            return False

    async def _ensure_ws(self) -> bool:
        """Lazy-start the WS client + reader task."""
        if self._ws_client is not None:
            return self._ws_client._connected and self._ws_client._authenticated
        if not self._loaded and not await self._load_keys():
            return False
        try:
            from exchange.bybit_private_ws import BybitPrivateWS
        except ImportError as e:
            logger.error("BYBIT_EXEC: import bybit_private_ws failed: %s", e)
            return False

        async def _on_fill(fill):
            try:
                await self._record_fill(fill)
            except Exception as ex:
                logger.error("BYBIT_EXEC: _record_fill error: %s", ex)

        self._ws_client = BybitPrivateWS(
            api_key=self._api_key,
            api_secret=self._api_secret,
            on_fill=_on_fill,
        )
        self._ws_task = asyncio.create_task(self._ws_client.connect_and_run())
        # Wait up to 8s for auth
        for _ in range(40):
            if self._ws_client._authenticated:
                logger.info("BYBIT_EXEC: %s ready", self.user_email)
                return True
            await asyncio.sleep(0.2)
        logger.warning("BYBIT_EXEC: %s WS not auth'd after 8s", self.user_email)
        return False

    async def dispatch(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point — called when a signal fires for this user.
        Returns dict with at least {"ok": bool, ...}.
        """
        out = {"ok": False, "exchange": "bybit", "user": self.user_email}
        symbol = signal.get("symbol", "")
        side_raw = str(signal.get("side", "")).lower()
        if side_raw not in ("long", "short"):
            out["error"] = f"unknown_side:{side_raw}"
            return out
        bybit_sym = SYMBOL_MAP.get(symbol)
        if not bybit_sym:
            out["error"] = f"unmapped_symbol:{symbol}"
            return out

        # Ensure WS up + authed
        if not await self._ensure_ws():
            out["error"] = "ws_not_ready"
            return out

        # Bybit side casing
        bybit_side = "Buy" if side_raw == "long" else "Sell"

        # Quantity: use signal.position_size as Delta contracts → convert to base currency
        # using Delta-style contract_size mapping
        from execution.user_bybit_executor import _delta_contract_size_for
        delta_qty = float(signal.get("position_size", 0) or 0)
        contract_size = _delta_contract_size_for(symbol)
        bybit_qty = delta_qty * contract_size
        if bybit_qty <= 0:
            out["error"] = f"zero_qty (delta_qty={delta_qty} contract_size={contract_size})"
            return out

        # Round to symbol's qty precision (Bybit BTC=0.001 step, ETH=0.01, SOL=0.1)
        qty_step = _bybit_qty_step(bybit_sym)
        bybit_qty = round(bybit_qty / qty_step) * qty_step
        if bybit_qty < qty_step:
            bybit_qty = qty_step

        coid = f"vne_{uuid.uuid4().hex[:14]}"
        logger.warning(
            "BYBIT_EXEC: submit %s %s %s qty=%s coid=%s",
            self.user_email[:12], bybit_sym, bybit_side, bybit_qty, coid
        )
        resp = await self._ws_client.submit_order(
            symbol=bybit_sym,
            side=bybit_side,
            qty=bybit_qty,
            order_type="Market",
            time_in_force="IOC",
            client_order_id=coid,
        )
        out["bybit_resp"] = resp
        retCode = (resp.get("data", {}) or {}).get("retCode", resp.get("retCode"))
        if retCode in (0, "0"):
            out["ok"] = True
            out["client_order_id"] = coid
            # Fill event will arrive via WS stream → _record_fill → user_trades insert
        else:
            out["error"] = f"bybit_reject:{resp.get('retMsg', resp.get('ret_msg', 'unknown'))}"
        return out

    async def _record_fill(self, fill) -> None:
        """When fill event arrives via stream WS, persist to user_trades."""
        if not self._db_pool:
            return
        try:
            # Convert Bybit symbol back to internal
            inv = {v: k for k, v in SYMBOL_MAP.items()}
            internal_sym = inv.get(fill.symbol, fill.symbol)
            side = "long" if fill.side == "Buy" else "short"
            async with self._db_pool.acquire() as con:
                await con.execute(
                    """INSERT INTO user_trades
                          (id, user_id, exchange, trade_type, symbol, side,
                           entry_price, quantity, status, fees_usd,
                           signal_data, metadata, opened_at)
                       VALUES (gen_random_uuid(), $1, 'bybit', 'real', $2, $3,
                               $4, $5, 'open', $6,
                               $7::jsonb, $8::jsonb, NOW())""",
                    self.user_id, internal_sym, side,
                    float(fill.price), float(fill.qty), float(fill.fee),
                    json.dumps({"entry_price": float(fill.price), "side": side, "from_fill_event": True}),
                    json.dumps({
                        "exchange": "bybit",
                        "fee_type": "taker",   # market order = taker
                        "fee_currency": fill.fee_currency,
                        "client_order_id": fill.client_order_id,
                        "bybit_order_id": fill.order_id,
                        "fill_ts_ms": fill.ts_ms,
                    }),
                )
            logger.info("BYBIT_EXEC: recorded fill %s %s qty=%s px=%s",
                        internal_sym, side, fill.qty, fill.price)
        except Exception as e:
            logger.error("BYBIT_EXEC: record_fill DB error: %s", e)


# ── Helpers ──────────────────────────────────────────────────────
def _delta_contract_size_for(symbol: str) -> float:
    """Mirror of Delta India contract sizes (for converting qty units)."""
    s = symbol.upper()
    if "BTC" in s: return 0.001
    if "ETH" in s: return 0.01
    if "SOL" in s: return 0.1
    if "AVAX" in s: return 0.1
    if "DOGE" in s: return 1.0
    return 1.0


def _bybit_qty_step(bybit_sym: str) -> float:
    """Bybit V5 linear perp qty step size. Source: bybit instruments-info endpoint."""
    s = bybit_sym.upper()
    if s.startswith("BTC"): return 0.001
    if s.startswith("ETH"): return 0.01
    if s.startswith("SOL"): return 0.1
    if s.startswith("XRP"): return 1.0
    if s.startswith("DOGE"): return 1.0
    if s.startswith("LTC"): return 0.1
    if s.startswith("ADA"): return 1.0
    if s.startswith("AVAX"): return 0.1
    if s.startswith("LINK"): return 0.1
    if s.startswith("TAO"): return 0.01
    if s.startswith("NEAR"): return 1.0
    if s.startswith("SUI"): return 1.0
    if s.startswith("WIF"): return 1.0
    if s.startswith("TRUMP"): return 1.0
    if s.startswith("1000"): return 1.0   # PEPE/SHIB/BONK lots
    return 1.0

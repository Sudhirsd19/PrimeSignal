import asyncio
import json
import time
import math
import hmac
import hashlib
import aiohttp
from typing import Any
from execution.execution_result import ExecutionResult, ExecutionState, ExecutionIntentJournal, new_intent_id

class CoinDCXClient:
    intent_journal: ExecutionIntentJournal

    def __init__(self, api_key: str, secret_key: str, intent_journal_path=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.coindcx.com"
        self.markets_info = {}
        self.initialized = False
        # FIX G: Reuse a persistent session to avoid per-request TCP overhead
        self._session = None
        self._usdt_inr_cache: tuple[float, dict[str, float]] | None = None
        self.intent_journal = ExecutionIntentJournal(intent_journal_path)

    async def _get_session(self):
        """Lazily create and reuse a single aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        """Close the persistent session on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def initialize(self):
        """Loads market details from CoinDCX to extract precisions and minimum limits."""
        if self.initialized:
            return True
        try:
            url = f"{self.base_url}/exchange/v1/markets_details"
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    markets = await response.json()
                    for m in markets:
                        name = m.get('coindcx_name')
                        pair = m.get('pair')
                        details = {
                            'min_quantity': float(m.get('min_quantity') or 0.0),
                            'min_notional': float(m.get('min_notional') or 0.0),
                            'precision': int(m.get('target_currency_precision') or 6),
                            'pair': pair
                        }
                        if name:
                            self.markets_info[name] = details
                            norm_name = name.replace('-', '').replace('_', '').upper()
                            self.markets_info[norm_name] = details
                        if pair:
                            self.markets_info[pair] = details
                            norm_pair = pair.replace('-', '').replace('_', '').replace('/', '').upper()
                            self.markets_info[norm_pair] = details
                    self.initialized = True
                    print(f"[CoinDCX] Loaded market details for {len(self.markets_info)} pair mappings.")
                    return True
                else:
                    print(f"[CoinDCX] ERROR: Failed to load market details. Status: {response.status}")
                    return False
        except Exception as e:
            print(f"[CoinDCX] ERROR initializing market details: {e}")
            return False

    def _sign(self, payload: dict):
        """Generate headers and HMAC signature for request authentication."""
        payload_str = json.dumps(payload, separators=(',', ':'))
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': signature
        }
        return payload_str, headers

    async def execute_with_retry(self, func, *args, retries: int = 3, delay: float = 1.0, **kwargs):
        """Exponential backoff retry handler for CoinDCX API calls."""
        for attempt in range(retries):
            try:
                res = await func(*args, **kwargs)
                if res is not None:
                    return res
            except Exception as e:
                print(f"[CoinDCX Retry] Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))
        return None

    async def fetch_balance(self):
        """Fetches account balances from CoinDCX and formats them into a CCXT-compatible dict."""
        if not self.initialized:
            await self.initialize()

        url = f"{self.base_url}/exchange/v1/users/balances"
        payload = {"timestamp": int(time.time() * 1000)}
        payload_str, headers = self._sign(payload)

        for attempt in range(3):
            try:
                session = await self._get_session()
                async with session.post(url, data=payload_str, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                    if response.status == 200:
                        balances = await response.json()
                        formatted_balances = {'total': {}, 'free': {}, 'used': {}}
                        for item in balances:
                            curr = item.get('currency', '').upper()
                            balance = float(item.get('balance') or 0.0)
                            locked = float(item.get('locked_balance') or 0.0)
                            formatted_balances['total'][curr] = balance
                            formatted_balances['free'][curr] = balance - locked
                            formatted_balances['used'][curr] = locked
                        return formatted_balances
                    elif response.status == 429:
                        print(f"[CoinDCX] Rate limit hit (429). Retrying in {attempt+1}s...")
                        await asyncio.sleep(1.0 * (2 ** attempt))
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX] ERROR fetching balances: {response.status} - {err_text}")
                        return None
            except Exception as e:
                print(f"[CoinDCX] Balance fetch error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1.0 * (2 ** attempt))
        return None

    async def fetch_ticker_data(self, coindcx_symbol: str):
        """Fetches public ticker data for a single symbol to get bid, ask, and index price."""
        url = f"{self.base_url}/exchange/ticker"
        for attempt in range(3):
            try:
                session = await self._get_session()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                    if response.status == 200:
                        tickers = await response.json()
                        target = next((t for t in tickers if t.get('market') == coindcx_symbol or t.get('pair') == coindcx_symbol), None)
                        if target:
                            return {
                                'last': float(target.get('last_price') or 0.0),
                                'bid': float(target.get('bid') or 0.0),
                                'ask': float(target.get('ask') or 0.0),
                                'volume': float(target.get('volume') or 0.0),
                            }
                        else:
                            return None
                    elif response.status == 429:
                        await asyncio.sleep(1.0 * (2 ** attempt))
                    else:
                        return None
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.0 * (2 ** attempt))
        return None

    async def fetch_usdt_inr_rate(self, side: str = "BUY") -> float | None:
        """
        Dynamically fetches live executable USDT/INR rate from CoinDCX ticker.
        Applies conservative worst-side pricing:
        - For BUY/LONG: uses 'ask' (higher cost in INR per USDT).
        - For SELL/SHORT: uses 'bid' (lower realized INR per USDT).
        Enforces strict sanity bounds [70.0, 120.0] and 30s cache TTL.
        Returns None if rate is unavailable, invalid, or out of bounds (fail-closed).
        """
        now = time.time()
        if hasattr(self, '_usdt_inr_cache') and self._usdt_inr_cache:
            cache_ts, cache_data = self._usdt_inr_cache
            if (now - cache_ts) < 30.0:
                return self._select_side_rate(cache_data, side)

        url = f"{self.base_url}/exchange/ticker"
        for attempt in range(2):
            try:
                session = await self._get_session()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status == 200:
                        tickers = await response.json()
                        target = next(
                            (t for t in tickers if t.get('market') in ('USDTINR', 'I-USDT_INR', 'B-USDT_INR') or t.get('pair') in ('USDTINR', 'I-USDT_INR', 'B-USDT_INR', 'USDT/INR')),
                            None
                        )
                        if target:
                            last_p = float(target.get('last_price') or 0.0)
                            bid_p = float(target.get('bid') or 0.0)
                            ask_p = float(target.get('ask') or 0.0)
                            data = {'last': last_p, 'bid': bid_p, 'ask': ask_p}
                            rate = self._select_side_rate(data, side)
                            if rate is not None and 70.0 <= rate <= 120.0:
                                self._usdt_inr_cache = (now, data)
                                return rate
                    elif response.status == 429:
                        await asyncio.sleep(0.5 * (2 ** attempt))
            except Exception as e:
                print(f"[CoinDCX FX] Ticker fetch error (attempt {attempt+1}): {e}")

        # Grace period for temporary network glitch: allow cache up to 60s
        if hasattr(self, '_usdt_inr_cache') and self._usdt_inr_cache:
            cache_ts, cache_data = self._usdt_inr_cache
            if (now - cache_ts) < 60.0:
                return self._select_side_rate(cache_data, side)

        return None

    def _select_side_rate(self, data: dict, side: str) -> float | None:
        last_p = float(data.get('last') or 0.0)
        bid_p = float(data.get('bid') or 0.0)
        ask_p = float(data.get('ask') or 0.0)
        if side.upper() in ('BUY', 'LONG'):
            rate = ask_p if ask_p > 0 else last_p
        else:
            rate = bid_p if bid_p > 0 else last_p
        if rate is not None and 70.0 <= rate <= 120.0:
            return rate
        return None

    async def place_order(
        self,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        symbol: str | None = None,
        client_order_id: str | None = None,
        intent_id: str | None = None,
    ) -> ExecutionResult:
        """Places one idempotent CoinDCX spot order.

        The client identity is created once, before the first POST, and reused
        for every retry of this economic intent.  An unresolved request never
        becomes a rejection merely because the lookup endpoint was unavailable.
        """
        if not self.initialized:
            await self.initialize()

        if not symbol:
            print("[CoinDCX] Error: symbol required for order placement.")
            return ExecutionResult(state=ExecutionState.NOT_SUBMITTED, requested_qty=amount or 0.0, error="symbol required", venue="COINDCX")

        # CoinDCX expected market code (e.g. BTCINR)
        market_name = symbol.replace('/', '').upper()
        
        # Apply strict precision truncation (floor) to prevent insufficient balance errors
        m_info = self.markets_info.get(market_name)
        if m_info:
            precision = m_info['precision']
            multiplier = 10 ** precision
            amount = math.floor(amount * multiplier) / multiplier
            min_q = m_info['min_quantity']
            if amount < min_q:
                print(f"[CoinDCX] Order rejected: Amount {amount} is below CoinDCX minimum {min_q} for {market_name}")
                return ExecutionResult(state=ExecutionState.REJECTED, requested_qty=amount, venue="COINDCX", error="below CoinDCX minimum")
        else:
            amount = math.floor(amount * 1000000.0) / 1000000.0

        intent_id = intent_id or new_intent_id()
        client_order_id = client_order_id or f"PS_DCX_{intent_id[:20].upper()}"

        url = f"{self.base_url}/exchange/v1/orders/create"
        payload = {
            "side": side.lower(),
            "order_type": "market_order" if order_type.lower() == "market" else "limit_order",
            "market": market_name,
            "total_quantity": amount,
            "timestamp": int(time.time() * 1000),
            "client_order_id": client_order_id,
        }
        
        if order_type.lower() == "limit" and price is not None:
            payload["price_per_unit"] = price

        try:
            self.intent_journal.create(
                intent_id=intent_id,
                client_order_id=client_order_id,
                venue="COINDCX",
                account_mode="spot",
                symbol=market_name,
                side=side.lower(),
                requested_qty=amount,
                order_role="ENTRY",
                price=price,
            )
        except Exception as journal_error:
            return ExecutionResult(
                state=ExecutionState.NOT_SUBMITTED,
                requested_qty=amount,
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue="COINDCX",
                error=f"intent journal failed: {journal_error}",
            )

        payload_str, headers = self._sign(payload)

        for attempt in range(3):
            try:
                print(f"[CoinDCX] Sending spot order (attempt {attempt+1}): {payload}")
                session = await self._get_session()
                async with session.post(url, data=payload_str, headers=headers, timeout=aiohttp.ClientTimeout(total=12.0)) as response:
                    if response.status == 200:
                        res = await response.json()
                        print(f"[CoinDCX] Order placed successfully! ID: {res.get('id')}")
                        result = ExecutionResult.from_exchange(
                            res,
                            requested_qty=amount,
                            client_order_id=client_order_id,
                            intent_id=intent_id,
                            venue="COINDCX",
                        )
                        if not result.exchange_order_id:
                            result.state = ExecutionState.SUBMISSION_UNKNOWN
                            result.error = "successful response did not contain exchange order id"
                        self.intent_journal.result(result)
                        return result
                    elif response.status == 429:
                        print(f"[CoinDCX] Order rate-limited (429). Retrying in {attempt+1}s...")
                        await asyncio.sleep(1.0 * (2 ** attempt))
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX] ERROR placing order: {response.status} - {err_text}")
                        if any(token in err_text.lower() for token in ("duplicate", "client_order_id", "already exists")):
                            existing_order = await self._reconcile_ambiguous_order(
                                market_name, side, amount, payload["timestamp"] - 5000,
                                client_order_id=client_order_id, intent_id=intent_id,
                            )
                            if existing_order is not None and not existing_order.is_unknown:
                                self.intent_journal.result(existing_order)
                                return existing_order
                            result = ExecutionResult(
                                state=ExecutionState.EXECUTION_UNKNOWN,
                                requested_qty=amount,
                                client_order_id=client_order_id,
                                intent_id=intent_id,
                                venue="COINDCX",
                                error=f"duplicate client order id unresolved: {err_text}",
                            )
                            self.intent_journal.result(result)
                            return result
                        result = ExecutionResult(
                            state=ExecutionState.REJECTED,
                            requested_qty=amount,
                            client_order_id=client_order_id,
                            intent_id=intent_id,
                            venue="COINDCX",
                            error=err_text,
                        )
                        self.intent_journal.result(result)
                        return result
            except Exception as e:
                print(f"[CoinDCX] Order execution exception on attempt {attempt+1}: {e}")
                # GAP-01 FIX: Reconcile exchange order state before blindly retrying to prevent duplicate fills
                print(f"[CoinDCX TIMEOUT] Verifying exchange state for {market_name} {side} ({amount}) before retry...")
                existing_order = await self._reconcile_ambiguous_order(
                    market_name, side, amount, created_after_ts=payload["timestamp"] - 5000,
                    client_order_id=client_order_id, intent_id=intent_id,
                )
                if existing_order is not None and not existing_order.is_unknown:
                    print(f"[CoinDCX IDEMPOTENCY] Discovered existing order {existing_order.exchange_order_id} on exchange after timeout; adopting without duplicate submission.")
                    self.intent_journal.result(existing_order)
                    return existing_order

                # A POST whose response was lost is an ambiguous economic
                # mutation.  Never issue a second POST merely because reads
                # are unavailable; quarantine the original intent instead.
                result = ExecutionResult(
                    state=ExecutionState.EXECUTION_UNKNOWN,
                    requested_qty=amount,
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue="COINDCX",
                    error="submission response lost and authoritative lookup failed",
                )
                self.intent_journal.result(result)
                return result
        result = ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            requested_qty=amount,
            client_order_id=client_order_id,
            intent_id=intent_id,
            venue="COINDCX",
            error="all submission attempts and authoritative lookups failed",
        )
        self.intent_journal.result(result)
        return result

    async def fetch_active_orders(self, market: str | None = None) -> list | None:
        """Fetches list of active/open orders from CoinDCX."""
        url = f"{self.base_url}/exchange/v1/orders/active_orders"
        payload: dict[str, Any] = {"timestamp": int(time.time() * 1000)}
        if market:
            payload["market"] = market
        payload_str, headers = self._sign(payload)
        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('orders', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                return None
        except Exception as e:
            print(f"[CoinDCX] Error fetching active orders: {e}")
            return None

    async def fetch_recent_trades(self, market: str | None = None) -> list | None:
        """Fetches recent trade execution history from CoinDCX."""
        url = f"{self.base_url}/exchange/v1/orders/trade_history"
        payload: dict[str, Any] = {"timestamp": int(time.time() * 1000)}
        if market:
            payload["market"] = market
        payload_str, headers = self._sign(payload)
        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else data.get('trades', [])
                return None
        except Exception as e:
            print(f"[CoinDCX] Error fetching trade history: {e}")
            return None

    async def _reconcile_ambiguous_order(
        self,
        market: str,
        side: str,
        amount: float,
        created_after_ts: int,
        client_order_id: str | None = None,
        intent_id: str | None = None,
    ) -> ExecutionResult | None:
        """
        Reconciles ambiguous order state after network timeout.
        Queries active orders and recent trade history to find if an order matching
        the market, side, and quantity was executed around the timestamp.
        """
        try:
            # 1. Check active orders. Prefer exact client identity. A heuristic
            # fallback is accepted only when exactly one candidate exists.
            active_orders = await self.fetch_active_orders(market=market)
            candidates = []
            for ord in (active_orders or []):
                ord_client_id = ord.get('client_order_id') or ord.get('clientOrderId')
                if client_order_id and ord_client_id and str(ord_client_id) == client_order_id:
                    return ExecutionResult.from_exchange(
                        ord, requested_qty=amount, client_order_id=client_order_id,
                        intent_id=intent_id, venue="COINDCX",
                    )
                ord_side = (ord.get('side') or '').lower()
                ord_qty = float(ord.get('total_quantity') or ord.get('quantity') or 0.0)
                ord_ts = int(ord.get('created_at') or ord.get('timestamp') or 0)
                if ord_side == side.lower() and abs(ord_qty - amount) <= 1e-5:
                    if ord_ts >= (created_after_ts - 10000):
                        candidates.append(ord)
            if len(candidates) == 1:
                return ExecutionResult.from_exchange(
                    candidates[0], requested_qty=amount,
                    client_order_id=client_order_id, intent_id=intent_id, venue="COINDCX",
                )
            
            # 2. Check recent trade history (for immediate market fills)
            recent_trades = await self.fetch_recent_trades(market=market)
            candidates = []
            for tr in (recent_trades or []):
                tr_client_id = tr.get('client_order_id') or tr.get('clientOrderId')
                if client_order_id and tr_client_id and str(tr_client_id) == client_order_id:
                    result = ExecutionResult.from_exchange(
                        tr, requested_qty=amount, client_order_id=client_order_id,
                        intent_id=intent_id, venue="COINDCX",
                    )
                    result.state = ExecutionState.FILLED if result.filled_qty > 0 else ExecutionState.EXECUTION_UNKNOWN
                    result.exchange_order_id = str(tr.get('order_id') or tr.get('id')) if (tr.get('order_id') or tr.get('id')) else None
                    return result
                tr_side = (tr.get('side') or '').lower()
                tr_qty = float(tr.get('quantity') or tr.get('total_quantity') or 0.0)
                tr_ts = int(tr.get('created_at') or tr.get('timestamp') or 0)
                if tr_side == side.lower() and abs(tr_qty - amount) <= 1e-5:
                    if tr_ts >= (created_after_ts - 10000):
                        candidates.append(tr)
            if len(candidates) == 1:
                result = ExecutionResult.from_exchange(
                    candidates[0], requested_qty=amount,
                    client_order_id=client_order_id, intent_id=intent_id, venue="COINDCX",
                )
                result.exchange_order_id = str(candidates[0].get('order_id') or candidates[0].get('id')) if (candidates[0].get('order_id') or candidates[0].get('id')) else None
                result.state = ExecutionState.FILLED if result.filled_qty > 0 else ExecutionState.EXECUTION_UNKNOWN
                return result
        except Exception as e:
            print(f"[CoinDCX RECONCILE] Error checking ambiguous order state: {e}")
        return None

    async def fetch_user_info(self):
        """Fetches user profile information from CoinDCX."""
        url = f"{self.base_url}/exchange/v1/users/info"
        payload = {"timestamp": int(time.time() * 1000)}
        payload_str, headers = self._sign(payload)

        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers) as response:
                    if response.status == 200:
                        res_data = await response.json()
                        if isinstance(res_data, list) and len(res_data) > 0:
                            return res_data[0]
                        elif isinstance(res_data, dict):
                            return res_data
                        return None
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX] ERROR fetching user info: {response.status} - {err_text}")
                        return None
        except Exception as e:
            print(f"[CoinDCX] ERROR calling user info endpoint: {e}")
            return None

    async def fetch_order_status(self, order_id: str | None = None, client_order_id: str | None = None) -> dict | None:
        """Fetches order status by exchange ID or exact client identity."""
        url = f"{self.base_url}/exchange/v1/orders/status"
        payload: dict[str, Any] = {
            "timestamp": int(time.time() * 1000),
        }
        if order_id:
            payload["id"] = order_id
        if client_order_id:
            payload["client_order_id"] = client_order_id
        payload_str, headers = self._sign(payload)
        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'id': data.get('id'),
                        'status': (data.get('status') or '').lower(),
                        'price': float(data.get('avg_price') or data.get('price_per_unit') or 0),
                        'amount': float(data.get('total_quantity') or 0),
                        'filled': float(data.get('filled_quantity') or 0),
                        'remaining': float(data.get('remaining_quantity') or max(0.0, float(data.get('total_quantity') or 0.0) - float(data.get('filled_quantity') or 0.0))),
                        'client_order_id': data.get('client_order_id') or data.get('clientOrderId') or client_order_id,
                    }
                else:
                    return None
        except Exception as e:
            print(f"[CoinDCX] Error fetching order status: {e}")
            return None

    async def place_stop_loss(
        self,
        side: str,
        amount: float,
        stop_price: float,
        symbol: str,
        client_order_id: str | None = None,
        intent_id: str | None = None,
    ) -> ExecutionResult:
        """Places a conditional stop-loss / trigger order on CoinDCX."""
        if not self.initialized:
            await self.initialize()
        
        market_name = symbol.replace('/', '').upper()
        m_info = self.markets_info.get(market_name)
        if m_info:
            precision = m_info['precision']
            multiplier = 10 ** precision
            amount = math.floor(amount * multiplier) / multiplier
        else:
            amount = math.floor(amount * 1000000.0) / 1000000.0

        intent_id = intent_id or new_intent_id()
        client_order_id = client_order_id or f"PS_DCX_SL_{intent_id[:18].upper()}"

        url = f"{self.base_url}/exchange/v1/orders/create"
        # CoinDCX conditional / stop limit order format
        # Directional buffer for stop-limit execution (LOGIC-007 fix):
        # Long exit (sell): Limit price placed 0.5% below stop trigger
        # Short exit (buy): Limit price placed 0.5% above stop trigger
        sl_buffer_pct = 0.005
        if side.lower() == 'sell':
            limit_price = stop_price * (1.0 - sl_buffer_pct)
        else:
            limit_price = stop_price * (1.0 + sl_buffer_pct)
        
        # Round limit price to appropriate precision based on price magnitude
        price_dec = 2 if stop_price >= 100 else (4 if stop_price >= 1 else 8)
        limit_price = round(limit_price, price_dec)

        payload = {
            "side": side.lower(),
            "order_type": "stop_limit",
            "market": market_name,
            "total_quantity": amount,
            "price_per_unit": limit_price,
            "stop_price": stop_price,
            "timestamp": int(time.time() * 1000),
            "client_order_id": client_order_id,
        }
        payload_str, headers = self._sign(payload)

        try:
            self.intent_journal.create(
                intent_id=intent_id,
                client_order_id=client_order_id,
                venue="COINDCX",
                account_mode="spot",
                symbol=market_name,
                side=side.lower(),
                requested_qty=amount,
                order_role="SL",
                price=stop_price,
            )
        except Exception as journal_error:
            return ExecutionResult(
                state=ExecutionState.NOT_SUBMITTED,
                requested_qty=amount,
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue="COINDCX",
                error=f"intent journal failed: {journal_error}",
            )

        for attempt in range(3):
            try:
                print(f"[CoinDCX Native SL] Placing Stop Loss @ {stop_price} ({payload})")
                session = await self._get_session()
                async with session.post(url, data=payload_str, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                    if response.status == 200:
                        res = await response.json()
                        print(f"[CoinDCX Native SL] ✅ SL Order Placed Successfully! ID: {res.get('id')}")
                        result = ExecutionResult.from_exchange(
                            {**res, 'stop_price': stop_price},
                            requested_qty=amount,
                            client_order_id=client_order_id,
                            intent_id=intent_id,
                            venue="COINDCX",
                        )
                        if not result.exchange_order_id:
                            result.state = ExecutionState.SUBMISSION_UNKNOWN
                            result.error = "successful response did not contain exchange order id"
                        self.intent_journal.result(result)
                        return result
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX Native SL] Error: {response.status} - {err_text}")
                        if any(token in err_text.lower() for token in ("duplicate", "client_order_id", "already exists")):
                            existing_sl = await self._reconcile_ambiguous_order(
                                market_name, side, amount, payload["timestamp"] - 5000,
                                client_order_id=client_order_id, intent_id=intent_id,
                            )
                            if existing_sl:
                                self.intent_journal.result(existing_sl)
                                return existing_sl
                            result = ExecutionResult(
                                state=ExecutionState.EXECUTION_UNKNOWN,
                                requested_qty=amount,
                                client_order_id=client_order_id,
                                intent_id=intent_id,
                                venue="COINDCX",
                                error=f"duplicate client order id unresolved: {err_text}",
                            )
                            self.intent_journal.result(result)
                            return result
                        if attempt < 2:
                            await asyncio.sleep(1.0 * (2 ** attempt))
            except Exception as e:
                print(f"[CoinDCX Native SL] Exception on attempt {attempt+1}: {e}")
                print(f"[CoinDCX Native SL TIMEOUT] Verifying exchange state for {market_name} Stop Loss before retry...")
                existing_sl = await self._reconcile_ambiguous_order(
                    market_name, side, amount, created_after_ts=payload["timestamp"] - 5000,
                    client_order_id=client_order_id, intent_id=intent_id,
                )
                if existing_sl:
                    print(f"[CoinDCX Native SL] Discovered existing Stop Loss order {existing_sl['id']} on exchange after timeout; adopting.")
                    self.intent_journal.result(existing_sl)
                    return existing_sl
                result = ExecutionResult(
                    state=ExecutionState.EXECUTION_UNKNOWN,
                    requested_qty=amount,
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue="COINDCX",
                    error="stop-loss response lost and authoritative lookup failed",
                )
                self.intent_journal.result(result)
                return result
        result = ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            requested_qty=amount,
            client_order_id=client_order_id,
            intent_id=intent_id,
            venue="COINDCX",
            error="all stop-loss submission attempts and lookups failed",
        )
        self.intent_journal.result(result)
        return result

    async def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancels an active order on CoinDCX."""
        url = f"{self.base_url}/exchange/v1/orders/cancel"
        payload = {
            "id": order_id,
            "timestamp": int(time.time() * 1000)
        }
        payload_str, headers = self._sign(payload)
        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                if response.status == 200:
                    print(f"[CoinDCX] Cancelled order {order_id}")
                    return ExecutionResult(
                        state=ExecutionState.CANCELLED,
                        exchange_order_id=order_id,
                        venue="COINDCX",
                    )
                else:
                    err_text = await response.text()
                    print(f"[CoinDCX] Error cancelling order {order_id}: {err_text}")
                    return ExecutionResult(
                        state=ExecutionState.CANCEL_UNKNOWN if response.status >= 500 else ExecutionState.REJECTED,
                        exchange_order_id=order_id,
                        venue="COINDCX",
                        error=err_text,
                    )
        except Exception as e:
            print(f"[CoinDCX] Exception cancelling order: {e}")
            return ExecutionResult(
                state=ExecutionState.CANCEL_UNKNOWN,
                exchange_order_id=order_id,
                venue="COINDCX",
                error="cancel request outcome unknown",
            )

    async def wait_for_fill(self, order_id: str, timeout: float = 30.0, requested_qty: float = 0.0, client_order_id: str | None = None, intent_id: str | None = None) -> ExecutionResult:
        """Polls CoinDCX order status until filled, cancelled, or timeout."""
        start = time.time()
        poll_interval = 1.0
        while time.time() - start < timeout:
            status = await self.fetch_order_status(order_id=order_id, client_order_id=client_order_id)
            if status:
                result = ExecutionResult.from_exchange(
                    status,
                    requested_qty=requested_qty,
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue="COINDCX",
                )
                s = status.get('status', '')
                if result.state in (ExecutionState.FILLED, ExecutionState.PARTIALLY_FILLED):
                    elapsed = int((time.time() - start) * 1000)
                    print(f"[CoinDCX FILL] Order {order_id} FILLED in {elapsed}ms")
                    return result
                elif s in ('cancelled', 'rejected'):
                    print(f"[CoinDCX FILL] Order {order_id} ended: {s}")
                    return result
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 3.0)
        print(f"[CoinDCX FILL] Order {order_id} TIMED OUT after {timeout}s")
        return ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            requested_qty=requested_qty or 0.0,
            client_order_id=client_order_id,
            intent_id=intent_id,
            exchange_order_id=order_id,
            venue="COINDCX",
            error="fill polling timed out or status remained unavailable",
        )

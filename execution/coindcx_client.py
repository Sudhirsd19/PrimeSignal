import asyncio
import json
import time
import math
import hmac
import hashlib
import aiohttp

class CoinDCXClient:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.coindcx.com"
        self.markets_info = {}
        self.initialized = False
        # FIX G: Reuse a persistent session to avoid per-request TCP overhead
        self._session = None

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
                async with session.post(url, data=payload_str, headers=headers, timeout=10.0) as response:
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
                async with session.get(url, timeout=10.0) as response:
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

    async def place_order(self, side: str, order_type: str, amount: float, price: float | None = None, symbol: str | None = None):
        """Places a spot order on CoinDCX with automatic retry and precision truncation."""
        if not self.initialized:
            await self.initialize()

        if not symbol:
            print("[CoinDCX] Error: symbol required for order placement.")
            return None

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
                return None
        else:
            amount = math.floor(amount * 1000000.0) / 1000000.0

        url = f"{self.base_url}/exchange/v1/orders/create"
        payload = {
            "side": side.lower(),
            "order_type": "market_order" if order_type.lower() == "market" else "limit_order",
            "market": market_name,
            "total_quantity": amount,
            "timestamp": int(time.time() * 1000)
        }
        
        if order_type.lower() == "limit" and price is not None:
            payload["price_per_unit"] = price

        payload_str, headers = self._sign(payload)

        for attempt in range(3):
            try:
                print(f"[CoinDCX] Sending spot order (attempt {attempt+1}): {payload}")
                session = await self._get_session()
                async with session.post(url, data=payload_str, headers=headers, timeout=12.0) as response:
                    if response.status == 200:
                        res = await response.json()
                        print(f"[CoinDCX] Order placed successfully! ID: {res.get('id')}")
                        return {
                            'id': res.get('id'),
                            'price': float(res.get('avg_price') or res.get('price_per_unit') or price or 0.0),
                            'status': res.get('status', '').lower(),
                            'amount': float(res.get('total_quantity') or amount)
                        }
                    elif response.status == 429:
                        print(f"[CoinDCX] Order rate-limited (429). Retrying in {attempt+1}s...")
                        await asyncio.sleep(1.0 * (2 ** attempt))
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX] ERROR placing order: {response.status} - {err_text}")
                        return None
            except Exception as e:
                print(f"[CoinDCX] Order execution exception on attempt {attempt+1}: {e}")
                # GAP-01 FIX: Reconcile exchange order state before blindly retrying to prevent duplicate fills
                print(f"[CoinDCX TIMEOUT] Verifying exchange state for {market_name} {side} ({amount}) before retry...")
                existing_order = await self._reconcile_ambiguous_order(market_name, side, amount, created_after_ts=payload["timestamp"] - 5000)
                if existing_order:
                    print(f"[CoinDCX IDEMPOTENCY] ✅ Discovered existing order {existing_order['id']} on exchange after timeout! Adopting without duplicate submission.")
                    return existing_order
                    
                if attempt < 2:
                    await asyncio.sleep(1.0 * (2 ** attempt))
        return None

    async def fetch_active_orders(self, market: str | None = None) -> list | None:
        """Fetches list of active/open orders from CoinDCX."""
        url = f"{self.base_url}/exchange/v1/orders/active_orders"
        payload = {"timestamp": int(time.time() * 1000)}
        if market:
            payload["market"] = market
        payload_str, headers = self._sign(payload)
        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers, timeout=10.0) as response:
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
        payload = {"timestamp": int(time.time() * 1000)}
        if market:
            payload["market"] = market
        payload_str, headers = self._sign(payload)
        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers, timeout=10.0) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else data.get('trades', [])
                return None
        except Exception as e:
            print(f"[CoinDCX] Error fetching trade history: {e}")
            return None

    async def _reconcile_ambiguous_order(self, market: str, side: str, amount: float, created_after_ts: int) -> dict | None:
        """
        Reconciles ambiguous order state after network timeout.
        Queries active orders and recent trade history to find if an order matching
        the market, side, and quantity was executed around the timestamp.
        """
        try:
            # 1. Check active orders
            active_orders = await self.fetch_active_orders(market=market)
            for ord in (active_orders or []):
                ord_side = (ord.get('side') or '').lower()
                ord_qty = float(ord.get('total_quantity') or ord.get('quantity') or 0.0)
                ord_ts = int(ord.get('created_at') or ord.get('timestamp') or 0)
                if ord_side == side.lower() and abs(ord_qty - amount) <= 1e-5:
                    if ord_ts >= (created_after_ts - 10000):
                        return {
                            'id': ord.get('id'),
                            'price': float(ord.get('avg_price') or ord.get('price_per_unit') or 0.0),
                            'status': (ord.get('status') or 'open').lower(),
                            'amount': ord_qty
                        }
            
            # 2. Check recent trade history (for immediate market fills)
            recent_trades = await self.fetch_recent_trades(market=market)
            for tr in (recent_trades or []):
                tr_side = (tr.get('side') or '').lower()
                tr_qty = float(tr.get('quantity') or tr.get('total_quantity') or 0.0)
                tr_ts = int(tr.get('created_at') or tr.get('timestamp') or 0)
                if tr_side == side.lower() and abs(tr_qty - amount) <= 1e-5:
                    if tr_ts >= (created_after_ts - 10000):
                        return {
                            'id': tr.get('order_id') or tr.get('id'),
                            'price': float(tr.get('price') or tr.get('avg_price') or 0.0),
                            'status': 'filled',
                            'amount': tr_qty
                        }
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

    async def fetch_order_status(self, order_id: str) -> dict | None:
        """Fetches the status of a specific order by ID."""
        url = f"{self.base_url}/exchange/v1/orders/status"
        payload = {
            "id": order_id,
            "timestamp": int(time.time() * 1000),
        }
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
                    }
                else:
                    return None
        except Exception as e:
            print(f"[CoinDCX] Error fetching order status: {e}")
            return None

    async def place_stop_loss(self, side: str, amount: float, stop_price: float, symbol: str) -> dict | None:
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
            "timestamp": int(time.time() * 1000)
        }
        payload_str, headers = self._sign(payload)

        for attempt in range(3):
            try:
                print(f"[CoinDCX Native SL] Placing Stop Loss @ {stop_price} ({payload})")
                session = await self._get_session()
                async with session.post(url, data=payload_str, headers=headers, timeout=10.0) as response:
                    if response.status == 200:
                        res = await response.json()
                        print(f"[CoinDCX Native SL] ✅ SL Order Placed Successfully! ID: {res.get('id')}")
                        return {
                            'id': res.get('id'),
                            'stop_price': stop_price,
                            'status': res.get('status', 'open').lower(),
                            'amount': amount
                        }
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX Native SL] Error: {response.status} - {err_text}")
                        if attempt < 2:
                            await asyncio.sleep(1.0 * (2 ** attempt))
            except Exception as e:
                print(f"[CoinDCX Native SL] Exception on attempt {attempt+1}: {e}")
                print(f"[CoinDCX Native SL TIMEOUT] Verifying exchange state for {market_name} Stop Loss before retry...")
                existing_sl = await self._reconcile_ambiguous_order(market_name, side, amount, created_after_ts=payload["timestamp"] - 5000)
                if existing_sl:
                    print(f"[CoinDCX Native SL] ✅ Discovered existing Stop Loss order {existing_sl['id']} on exchange after timeout! Adopting.")
                    return existing_sl
                if attempt < 2:
                    await asyncio.sleep(1.0 * (2 ** attempt))
        return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancels an active order on CoinDCX."""
        url = f"{self.base_url}/exchange/v1/orders/cancel"
        payload = {
            "id": order_id,
            "timestamp": int(time.time() * 1000)
        }
        payload_str, headers = self._sign(payload)
        try:
            session = await self._get_session()
            async with session.post(url, data=payload_str, headers=headers, timeout=10.0) as response:
                if response.status == 200:
                    print(f"[CoinDCX] Cancelled order {order_id}")
                    return True
                else:
                    err_text = await response.text()
                    print(f"[CoinDCX] Error cancelling order {order_id}: {err_text}")
                    return False
        except Exception as e:
            print(f"[CoinDCX] Exception cancelling order: {e}")
            return False

    async def wait_for_fill(self, order_id: str, timeout: float = 30.0) -> dict | None:
        """Polls CoinDCX order status until filled, cancelled, or timeout."""
        start = time.time()
        poll_interval = 1.0
        while time.time() - start < timeout:
            status = await self.fetch_order_status(order_id)
            if status:
                s = status.get('status', '')
                if s in ('filled', 'completed', 'closed'):
                    elapsed = int((time.time() - start) * 1000)
                    print(f"[CoinDCX FILL] Order {order_id} FILLED in {elapsed}ms")
                    return status
                elif s in ('cancelled', 'rejected'):
                    print(f"[CoinDCX FILL] Order {order_id} ended: {s}")
                    return None
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 3.0)
        print(f"[CoinDCX FILL] Order {order_id} TIMED OUT after {timeout}s")
        return None


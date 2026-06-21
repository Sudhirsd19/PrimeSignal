import asyncio
import json
import time
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

    async def initialize(self):
        """Loads market details from CoinDCX to extract precisions and minimum limits."""
        if self.initialized:
            return True
        try:
            url = f"{self.base_url}/exchange/v1/markets_details"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        markets = await response.json()
                        for m in markets:
                            name = m.get('coindcx_name')
                            if name:
                                self.markets_info[name] = {
                                    'min_quantity': float(m.get('min_quantity') or 0.0),
                                    'min_notional': float(m.get('min_notional') or 0.0),
                                    'precision': int(m.get('target_currency_precision') or 6),
                                    'pair': m.get('pair')
                                }
                        self.initialized = True
                        print(f"[CoinDCX] Loaded market details for {len(self.markets_info)} pairs.")
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

    async def fetch_balance(self):
        """Fetches account balances from CoinDCX and formats them into a CCXT-compatible dict."""
        if not self.initialized:
            await self.initialize()

        url = f"{self.base_url}/exchange/v1/users/balances"
        payload = {"timestamp": int(time.time() * 1000)}
        payload_str, headers = self._sign(payload)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload_str, headers=headers) as response:
                    if response.status == 200:
                        balances = await response.json()
                        
                        # Format to CCXT style: {'total': {currency: amount}, 'free': {currency: amount}, ...}
                        formatted_balances = {'total': {}, 'free': {}, 'used': {}}
                        for item in balances:
                            curr = item.get('currency', '').upper()
                            balance = float(item.get('balance') or 0.0)
                            locked = float(item.get('locked_balance') or 0.0)
                            
                            formatted_balances['total'][curr] = balance
                            formatted_balances['free'][curr] = balance - locked
                            formatted_balances['used'][curr] = locked
                            
                        return formatted_balances
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX] ERROR fetching balances: {response.status} - {err_text}")
                        return None
        except Exception as e:
            print(f"[CoinDCX] ERROR calling balances endpoint: {e}")
            return None

    async def fetch_ticker_data(self, coindcx_symbol: str):
        """Fetches public ticker data for a single symbol to get bid, ask, and index price."""
        url = f"{self.base_url}/exchange/ticker"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        tickers = await response.json()
                        # Tickers is a list of dicts. Find the one matching coindcx_symbol
                        target = next((t for t in tickers if t.get('market') == coindcx_symbol or t.get('pair') == coindcx_symbol), None)
                        if target:
                            return {
                                'last': float(target.get('last_price') or 0.0),
                                'bid': float(target.get('bid') or 0.0),
                                'ask': float(target.get('ask') or 0.0),
                                'volume': float(target.get('volume') or 0.0),
                            }
                        else:
                            print(f"[CoinDCX] Ticker symbol {coindcx_symbol} not found in public feed.")
                            return None
                    else:
                        print(f"[CoinDCX] ERROR fetching ticker data: {response.status}")
                        return None
        except Exception as e:
            print(f"[CoinDCX] ERROR calling public ticker: {e}")
            return None

    async def place_order(self, side: str, order_type: str, amount: float, price: float = None, symbol: str = None):
        """Places a spot order on CoinDCX."""
        if not self.initialized:
            await self.initialize()

        if not symbol:
            print("[CoinDCX] Error: symbol required for order placement.")
            return None

        # CoinDCX expected market code (e.g. BTCINR)
        market_name = symbol.replace('/', '').upper()
        
        # Apply precision rounding using CoinDCX markets_details info
        m_info = self.markets_info.get(market_name)
        if m_info:
            precision = m_info['precision']
            amount = round(amount, precision)
            min_q = m_info['min_quantity']
            if amount < min_q:
                print(f"[CoinDCX] Order rejected: Amount {amount} is below CoinDCX minimum {min_q} for {market_name}")
                return None
        else:
            # Fallback to standard spot rounding
            amount = round(amount, 6)

        url = f"{self.base_url}/exchange/v1/orders/create"
        
        # Build payload
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

        try:
            print(f"[CoinDCX] Sending spot order: {payload}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload_str, headers=headers) as response:
                    if response.status == 200:
                        res = await response.json()
                        print(f"[CoinDCX] Order placed successfully! ID: {res.get('id')}")
                        return {
                            'id': res.get('id'),
                            'price': float(res.get('avg_price') or res.get('price_per_unit') or price or 0.0),
                            'status': res.get('status', '').lower(),
                            'amount': float(res.get('total_quantity') or amount)
                        }
                    else:
                        err_text = await response.text()
                        print(f"[CoinDCX] ERROR placing order: {response.status} - {err_text}")
                        return None
        except Exception as e:
            print(f"[CoinDCX] ERROR executing place_order call: {e}")
            return None

    async def fetch_user_info(self):
        """Fetches user profile information from CoinDCX."""
        url = f"{self.base_url}/exchange/v1/users/info"
        payload = {"timestamp": int(time.time() * 1000)}
        payload_str, headers = self._sign(payload)

        try:
            async with aiohttp.ClientSession() as session:
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

import ccxt.async_support as ccxt
import asyncio
import time
from config import Config


class ExecutionEngine:
    def __init__(self):
        is_futures = Config.EXCHANGE_TYPE == 'futures'

        # 1. Public client (for data — always spot for Binance public streams)
        self.public_client = ccxt.binance()

        # 2. Trading client (for private actions)
        options = {}
        if Config.API_KEY and Config.API_KEY != "your_api_key_here":
            options['apiKey'] = Config.API_KEY
        if Config.SECRET_KEY and Config.SECRET_KEY != "your_api_secret_here":
            options['secret'] = Config.SECRET_KEY

        # Feature 2: Futures support — set defaultType to 'future' for USDT-M
        if is_futures:
            options['options'] = {'defaultType': 'future'}
            self.trade_client = ccxt.binance(options)
            print(f"[EXECUTION] Binance USDT-M Futures mode enabled (Leverage: {Config.FUTURES_LEVERAGE}x, Margin: {Config.FUTURES_MARGIN_MODE})")
        else:
            self.trade_client = ccxt.binance(options)

        # 3. CoinDCX Integration (spot only — CoinDCX has no standard futures API)
        from execution.coindcx_client import CoinDCXClient
        self.coindcx_client = None
        if Config.COINDCX_API_KEY and Config.COINDCX_API_KEY != "your_coindcx_key_here":
            if is_futures:
                print("[EXECUTION] CoinDCX disabled in futures mode (CoinDCX does not support futures via API).")
            else:
                self.coindcx_client = CoinDCXClient(Config.COINDCX_API_KEY, Config.COINDCX_SECRET_KEY)
                print("[EXECUTION] CoinDCX client integrated successfully.")

        if Config.USE_TESTNET:
            self.trade_client.set_sandbox_mode(True)
            print("[EXECUTION] Sandbox mode enabled (Binance Testnet)")
        else:
            print("[EXECUTION] WARNING: Live mainnet enabled. Operating with real funds.")

        self._tickers_cache = {}
        self._tickers_cache_time = 0
        self._futures_initialized = False

    async def _init_futures(self, symbol=None):
        """One-time setup for futures: load markets, set leverage and margin mode."""
        if self._futures_initialized or Config.EXCHANGE_TYPE != 'futures':
            return
        try:
            await self.trade_client.load_markets()
            sym = symbol or Config.SYMBOL
            # Set margin mode (isolated / cross)
            try:
                await self.trade_client.set_margin_mode(Config.FUTURES_MARGIN_MODE, sym)
                print(f"[FUTURES] Margin mode set to {Config.FUTURES_MARGIN_MODE.upper()} for {sym}")
            except Exception as e:
                print(f"[FUTURES] Margin mode already set or not changeable: {e}")
            # Set leverage
            try:
                await self.trade_client.set_leverage(Config.FUTURES_LEVERAGE, sym)
                print(f"[FUTURES] Leverage set to {Config.FUTURES_LEVERAGE}x for {sym}")
            except Exception as e:
                print(f"[FUTURES] Leverage already set or not changeable: {e}")
            self._futures_initialized = True
        except Exception as e:
            print(f"[FUTURES] ERROR initializing futures settings: {e}")

    async def close(self):
        await self.public_client.close()
        await self.trade_client.close()
        if self.coindcx_client:
            await self.coindcx_client.close()

    async def fetch_balance(self):
        """Fetch balances with automatic retry."""
        if self.coindcx_client:
            return await self.coindcx_client.fetch_balance()
        return await self.execute_with_retry(self.trade_client.fetch_balance)

    async def fetch_current_price(self, symbol=None):
        """Fetch last price from public ticker."""
        if symbol is None:
            symbol = Config.SYMBOL
        ticker = await self.execute_with_retry(self.public_client.fetch_ticker, symbol)
        if ticker:
            return ticker['last']
        return None

    async def fetch_ticker_data(self, symbol=None):
        """Fetch full ticker data including bid, ask, and quoteVolume."""
        if symbol is None:
            symbol = Config.SYMBOL
        return await self.execute_with_retry(self.public_client.fetch_ticker, symbol)

    async def fetch_all_tickers(self):
        """Fetch all tickers with a 60-second cache to find top volume symbols."""
        now = time.time()
        if now - self._tickers_cache_time > 60 or not self._tickers_cache:
            tickers = await self.execute_with_retry(self.public_client.fetch_tickers)
            if tickers:
                self._tickers_cache = tickers
                self._tickers_cache_time = now
        return self._tickers_cache

    async def fetch_ohlcv(self, symbol=None, timeframe=None, limit=100):
        """Fetch historical candlestick data (OHLCV) from public client."""
        if symbol is None:
            symbol = Config.SYMBOL
        if timeframe is None:
            timeframe = Config.TIMEFRAME
        return await self.execute_with_retry(self.public_client.fetch_ohlcv, symbol, timeframe, None, limit)

    # ── Feature 1: Order fill confirmation ───────────────────────────────
    async def wait_for_fill(self, order_id: str, symbol: str, timeout: float = 30.0) -> dict | None:
        """
        Polls order status until filled, cancelled, or timeout.
        Returns the final order dict or None on timeout/cancellation.
        """
        start = time.time()
        poll_interval = 0.5  # start fast, then slow down
        while time.time() - start < timeout:
            try:
                order = await self.trade_client.fetch_order(order_id, symbol)
                status = (order.get('status') or '').lower()
                if status in ('closed', 'filled'):
                    elapsed = int((time.time() - start) * 1000)
                    print(f"[FILL] Order {order_id} FILLED in {elapsed}ms. Avg price: {order.get('average', order.get('price'))}")
                    return order
                elif status in ('canceled', 'cancelled', 'rejected', 'expired'):
                    print(f"[FILL] Order {order_id} ended with status: {status}")
                    return None
            except Exception as e:
                print(f"[FILL] Error polling order {order_id}: {e}")
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 3.0)  # gradual backoff

        print(f"[FILL] Order {order_id} TIMED OUT after {timeout}s — status unknown")
        return None

    async def place_order(self, side, order_type, amount, price=None,
                          max_slippage_pct=0.005, symbol=None,
                          is_exit_order=False, confirm_fill=True):
        """
        Routes orders with slippage checks, retry logic, and fill confirmation.

        is_exit_order (bool): If True, bypasses slippage guard. Exit orders
            MUST always execute regardless of slippage.
        confirm_fill (bool): If True, polls order status until filled/cancelled.
        """
        if symbol is None:
            symbol = Config.SYMBOL

        # Initialize futures settings on first order if applicable
        if Config.EXCHANGE_TYPE == 'futures':
            await self._init_futures(symbol)

        if self.coindcx_client:
            coindcx_symbol = symbol
            if Config.COINDCX_TRADE_INR:
                target = symbol.split('/')[0]
                coindcx_symbol = f"{target}/INR"
            order = await self.coindcx_client.place_order(side, order_type, amount, price, symbol=coindcx_symbol)
            # CoinDCX fill confirmation via their order status endpoint
            if order and confirm_fill and order.get('id'):
                confirmed = await self.coindcx_client.wait_for_fill(order['id'])
                if confirmed:
                    order.update(confirmed)
            return order

        # 0. Enforce exchange LOT_SIZE precision
        try:
            if not self.trade_client.markets:
                await self.trade_client.load_markets()
            precise_amount = float(self.trade_client.amount_to_precision(symbol, amount))
            if precise_amount != amount:
                print(f"[EXECUTION] Quantity rounded for exchange precision: {amount} → {precise_amount}")
            amount = precise_amount
        except Exception as e:
            print(f"[EXECUTION] WARNING: Could not apply precision rounding ({e}). Using raw amount.")

        # Check minimum order size
        try:
            markets = self.trade_client.markets
            if markets and symbol in markets:
                min_amount = markets[symbol].get('limits', {}).get('amount', {}).get('min', 0) or 0
                if amount < min_amount:
                    print(f"[EXECUTION] Order rejected: Amount {amount:.8f} is below minimum {min_amount} for {symbol}")
                    return None
        except Exception as e:
            print(f"[EXECUTION] WARNING: Could not check minimum order size ({e}). Proceeding anyway.")

        # 1. Slippage check for market ENTRY orders only
        if order_type.upper() == "MARKET" and not is_exit_order:
            ticker = await self.execute_with_retry(self.public_client.fetch_ticker, symbol)
            if not ticker:
                print("[EXECUTION] Order aborted: Unable to fetch live price ticker for slippage check.")
                return None

            current_price = ticker['last']
            if price is not None:
                if side.upper() == "BUY":
                    slippage = (current_price - price) / price
                    if slippage > max_slippage_pct:
                        print(f"[EXECUTION] Order aborted: Slippage ({slippage*100:.2f}%) exceeds max ({max_slippage_pct*100:.2f}%).")
                        return None
                elif side.upper() == "SELL":
                    slippage = (price - current_price) / price
                    if slippage > max_slippage_pct:
                        print(f"[EXECUTION] Order aborted: Slippage ({slippage*100:.2f}%) exceeds max ({max_slippage_pct*100:.2f}%).")
                        return None

        # 2. Place order with retries
        fn = None
        args = [symbol, amount]

        if order_type.upper() == "MARKET":
            fn = self.trade_client.create_market_order
            args = [symbol, side.lower(), amount]
        elif order_type.upper() == "LIMIT":
            if price is None:
                print("[EXECUTION] Order error: Limit orders require a price.")
                return None
            args = [symbol, side.lower(), amount, price]
            fn = self.trade_client.create_order

        if fn is None:
            fn = self.trade_client.create_order
            args = [symbol, order_type.lower(), side.lower(), amount, price]

        print(f"[EXECUTION] Sending {order_type.upper()} {side.upper()} order for {amount} {symbol}...")
        order = await self.execute_with_retry(fn, *args)
        if order:
            print(f"[EXECUTION] Order submitted! ID: {order['id']}, Status: {order['status']}")
            # Feature 1: Poll for fill confirmation
            if confirm_fill and order.get('id') and order.get('status') != 'closed':
                confirmed = await self.wait_for_fill(order['id'], symbol, timeout=30.0)
                if confirmed:
                    order = confirmed
                else:
                    print(f"[EXECUTION] WARNING: Order {order['id']} may not have filled. Check manually.")
        return order

    async def execute_with_retry(self, func, *args, retries=3, delay=1.0):
        """
        Executes a CCXT call with exponential backoff retry logic.
        """
        for attempt in range(1, retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args)
                else:
                    return func(*args)
            except ccxt.InsufficientFunds as e:
                print(f"[EXECUTION] Trade failed (Insufficient Funds): {e}")
                break
            except ccxt.InvalidOrder as e:
                print(f"[EXECUTION] Trade failed (Invalid Order): {e}")
                break
            except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e:
                if attempt == retries:
                    print(f"[EXECUTION] API failed after {retries} attempts. Final Error: {e}")
                    raise e
                sleep_time = delay * (2 ** (attempt - 1))
                print(f"[EXECUTION] API error ({e}). Retrying in {sleep_time:.1f}s (Attempt {attempt}/{retries})...")
                await asyncio.sleep(sleep_time)
            except ccxt.BaseError as e:
                print(f"[EXECUTION] CCXT Error (not retryable): {type(e).__name__}: {e}")
                break
            except Exception as e:
                import traceback
                print(f"[EXECUTION] CRITICAL: Unexpected exception in {func.__name__}:")
                print(f"[EXECUTION]   Error type: {type(e).__name__}")
                print(f"[EXECUTION]   Error message: {e}")
                traceback.print_exc()
                break
        return None

    async def fetch_coindcx_user_info(self):
        """Fetch CoinDCX user profile information."""
        if self.coindcx_client:
            return await self.coindcx_client.fetch_user_info()
        return None

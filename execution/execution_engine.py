import ccxt.async_support as ccxt
import asyncio
from config import Config

class ExecutionEngine:
    def __init__(self):
        # 1. Public client (for data)
        self.public_client = ccxt.binance()
        
        # 2. Trading client (for private actions)
        options = {}
        if Config.API_KEY and Config.API_KEY != "your_api_key_here":
            options['apiKey'] = Config.API_KEY
        if Config.SECRET_KEY and Config.SECRET_KEY != "your_api_secret_here":
            options['secret'] = Config.SECRET_KEY
            
        self.trade_client = ccxt.binance(options)
        
        # 3. CoinDCX Integration
        from execution.coindcx_client import CoinDCXClient
        self.coindcx_client = None
        if Config.COINDCX_API_KEY and Config.COINDCX_API_KEY != "your_coindcx_key_here":
            self.coindcx_client = CoinDCXClient(Config.COINDCX_API_KEY, Config.COINDCX_SECRET_KEY)
            print("[EXECUTION] CoinDCX client integrated successfully.")
            
        if Config.USE_TESTNET:
            self.trade_client.set_sandbox_mode(True)
            print("[EXECUTION] Sandbox mode enabled (Binance Testnet)")
        else:
            print("[EXECUTION] WARNING: Live mainnet enabled. Operating with real funds.")
            
        self._tickers_cache = {}
        self._tickers_cache_time = 0

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
        import time
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


    async def place_order(self, side, order_type, amount, price=None, max_slippage_pct=0.005, symbol=None, is_exit_order=False):
        """
        Routes orders with slippage checks and retry logic.

        is_exit_order (bool): If True, bypasses slippage guard. Exit orders
            MUST always execute regardless of slippage — blocking an exit during
            a flash crash leaves the position completely unprotected.
        """
        if symbol is None:
            symbol = Config.SYMBOL
            
        if self.coindcx_client:
            coindcx_symbol = symbol
            if Config.COINDCX_TRADE_INR:
                target = symbol.split('/')[0]
                coindcx_symbol = f"{target}/INR"
            return await self.coindcx_client.place_order(side, order_type, amount, price, symbol=coindcx_symbol)

        # 0. Enforce exchange LOT_SIZE precision (CRITICAL-4 fix)
        # Binance rejects orders where amount does not match step size filter.
        try:
            # Load markets data if not already loaded (needed for precision info)
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
        # ATTACK-5 FIX: Exit orders bypass slippage guard entirely. We must close
        # the position at any available price — slippage is acceptable on exit.
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
                        print(f"[EXECUTION] Order aborted: Slippage ({slippage*100:.2f}%) exceeds max threshold ({max_slippage_pct*100:.2f}%). Expected: {price:.2f}, Live: {current_price:.2f}")
                        return None
                elif side.upper() == "SELL":
                    slippage = (price - current_price) / price
                    if slippage > max_slippage_pct:
                        print(f"[EXECUTION] Order aborted: Slippage ({slippage*100:.2f}%) exceeds max threshold ({max_slippage_pct*100:.2f}%). Expected: {price:.2f}, Live: {current_price:.2f}")
                        return None
                        
        # 2. Place order with retries
        fn = None
        args = [symbol, amount]
        
        if order_type.upper() == "MARKET":
            fn = self.trade_client.create_market_order
            # CCXT create_market_order signature: (symbol, side, amount, params={})
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
            print(f"[EXECUTION] Order executed! ID: {order['id']}, Avg Price: {order.get('price', price)}, Status: {order['status']}")
        return order

    async def execute_with_retry(self, func, *args, retries=3, delay=1.0):
        """
        Executes a CCXT call with exponential backoff retry logic.
        """
        for attempt in range(1, retries + 1):
            try:
                # Call the API function
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
                # All CCXT-specific errors inherit from BaseError (not retryable)
                print(f"[EXECUTION] CCXT Error (not retryable): {type(e).__name__}: {e}")
                break
            except Exception as e:
                # Unexpected exception - log fully for debugging
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

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
        
        if Config.USE_TESTNET:
            self.trade_client.set_sandbox_mode(True)
            print("[EXECUTION] Sandbox mode enabled (Binance Testnet)")
        else:
            print("[EXECUTION] WARNING: Live mainnet enabled. Operating with real funds.")

    async def close(self):
        await self.public_client.close()
        await self.trade_client.close()

    async def fetch_balance(self):
        """Fetch balances with automatic retry."""
        return await self.execute_with_retry(self.trade_client.fetch_balance)

    async def fetch_current_price(self, symbol=None):
        """Fetch last price from public ticker."""
        if symbol is None:
            symbol = Config.SYMBOL
        ticker = await self.execute_with_retry(self.public_client.fetch_ticker, symbol)
        if ticker:
            return ticker['last']
        return None

    async def fetch_ohlcv(self, symbol=None, timeframe=None, limit=100):
        """Fetch historical candlestick data (OHLCV) from public client."""
        if symbol is None:
            symbol = Config.SYMBOL
        if timeframe is None:
            timeframe = Config.TIMEFRAME
        return await self.execute_with_retry(self.public_client.fetch_ohlcv, symbol, timeframe, None, limit)


    async def place_order(self, side, order_type, amount, price=None, max_slippage_pct=0.005, symbol=None):
        """
        Routes orders with slippage checks and retry logic.
        """
        if symbol is None:
            symbol = Config.SYMBOL
            
        # 1. Slippage check for market orders
        if order_type.upper() == "MARKET":
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
            except Exception as e:
                print(f"[EXECUTION] Unexpected execution engine error: {e}")
                break
        return None

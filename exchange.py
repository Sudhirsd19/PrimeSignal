import ccxt.async_support as ccxt
from config import Config

class ExchangeConnector:
    def __init__(self):
        # 1. Initialize public client (always Mainnet, for public data like price and candles)
        self.public_client = ccxt.binance()
        print("Connected to Binance SPOT MAINNET for market data feeds")

        # 2. Initialize trading client (Mainnet or Testnet, for private data like balance, orders)
        options = {}
        if Config.API_KEY and Config.API_KEY != "your_api_key_here":
            options['apiKey'] = Config.API_KEY
        if Config.SECRET_KEY and Config.SECRET_KEY != "your_api_secret_here":
            options['secret'] = Config.SECRET_KEY

        self.trade_client = ccxt.binance(options)
        if Config.USE_TESTNET:
            self.trade_client.set_sandbox_mode(True)
            print("Connected to Binance SPOT TESTNET (Sandbox Mode) for trading actions")
        else:
            print("WARNING: Connected to Binance SPOT MAINNET (Real Funds Mode) for trading actions")

    async def close(self):
        """Close exchange sessions."""
        await self.public_client.close()
        await self.trade_client.close()

    async def fetch_balance(self):
        """Fetch account balance from trading client."""
        try:
            balance = await self.trade_client.fetch_balance()
            return balance
        except ccxt.AuthenticationError:
            print("ERROR: Authentication Error: Invalid API Key or Secret.")
            return None
        except Exception as e:
            print(f"ERROR fetching balance: {e}")
            return None

    async def fetch_current_price(self, symbol=None):
        """Fetch current ticker price from public client."""
        if symbol is None:
            symbol = Config.SYMBOL
        try:
            ticker = await self.public_client.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"ERROR fetching current price for {symbol}: {e}")
            return None

    async def fetch_ohlcv(self, symbol=None, timeframe=None, limit=100):
        """Fetch historical candlestick data (OHLCV) from public client."""
        if symbol is None:
            symbol = Config.SYMBOL
        if timeframe is None:
            timeframe = Config.LTF_TIMEFRAME  # default to lower timeframe
        try:
            ohlcv = await self.public_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            print(f"ERROR fetching historical OHLCV data: {e}")
            return None

    async def place_market_order(self, side, amount, symbol=None):
        """Place a market order on trading client."""
        if symbol is None:
            symbol = Config.SYMBOL
        try:
            print(f"Placing Market {side.upper()} order for {amount} {symbol}...")
            order = await self.trade_client.create_market_order(symbol, side, amount)
            print(f"Order Placed Successfully! Order ID: {order['id']}, Status: {order['status']}")
            return order
        except ccxt.InsufficientFunds as e:
            print(f"ERROR: Insufficient Funds to execute {side} order: {e}")
            return None
        except ccxt.InvalidOrder as e:
            print(f"ERROR: Invalid Order error: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Failed to place market order: {e}")
            return None

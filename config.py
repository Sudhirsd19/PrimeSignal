import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Exchange API settings
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "yes")
    
    # Product Settings
    SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
    SUPPORTED_SYMBOLS = os.getenv("SUPPORTED_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,AVAX/USDT,DOT/USDT,LTC/USDT,TRX/USDT,LINK/USDT,ATOM/USDT,ETC/USDT,FIL/USDT,APT/USDT,NEAR/USDT,ARB/USDT,OP/USDT,POL/USDT").split(",")
    TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "0.001"))
    
    # Risk parameters
    RISK_PCT = float(os.getenv("RISK_PCT", "1.0"))
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))
    MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))
    TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.015")) # Deprecated in favor of ATR
    TRAILING_ATR_MULT = float(os.getenv("TRAILING_ATR_MULT", "2.0"))
    MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", "2.0"))
    
    # Strategy settings
    HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "1h")
    LTF_TIMEFRAME = os.getenv("LTF_TIMEFRAME", "5m")
    SHORT_EMA = int(os.getenv("SHORT_EMA", "9"))
    LONG_EMA = int(os.getenv("LONG_EMA", "21"))
    TREND_EMA = int(os.getenv("TREND_EMA", "200"))
    
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
    RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", "30"))
    RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", "70"))
    ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
    MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0005"))
    MAX_ZONE_AGE_CANDLES = int(os.getenv("MAX_ZONE_AGE_CANDLES", "50"))
    ZONE_BUFFER_PCT = float(os.getenv("ZONE_BUFFER_PCT", "0.001"))
    VWAP_TOLERANCE = float(os.getenv("VWAP_TOLERANCE", "0.002"))
    MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.002"))
    MIN_24H_VOL_USDT = float(os.getenv("MIN_24H_VOL_USDT", "50000000"))
    MAX_CANDLE_MOVE_PCT = float(os.getenv("MAX_CANDLE_MOVE_PCT", "0.015"))
    VOLATILITY_PAUSE_CANDLES = int(os.getenv("VOLATILITY_PAUSE_CANDLES", "2"))
    COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))
    MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))
    FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))
    
    # Machine Learning configurations
    ML_CONFIRMATION_THRESHOLD = float(os.getenv("ML_CONFIRMATION_THRESHOLD", "0.50"))
    ML_TRAIN_BARS = int(os.getenv("ML_TRAIN_BARS", "2000"))
    
    # Test Mode config
    TEST_MODE = os.getenv("TEST_MODE", "False").lower() in ("true", "1", "yes")
    PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "yes")
    
    # Telegram notifier settings
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # CoinDCX settings
    COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
    COINDCX_SECRET_KEY = os.getenv("COINDCX_SECRET_KEY", "")
    COINDCX_TRADE_INR = os.getenv("COINDCX_TRADE_INR", "True").lower() in ("true", "1", "yes")

    # Exchange type: 'spot' or 'futures' (Binance USDT-M Futures)
    EXCHANGE_TYPE = os.getenv("EXCHANGE_TYPE", "spot").lower()

    # Futures-specific settings (only used when EXCHANGE_TYPE='futures')
    FUTURES_LEVERAGE = int(os.getenv("FUTURES_LEVERAGE", "1"))
    FUTURES_MARGIN_MODE = os.getenv("FUTURES_MARGIN_MODE", "isolated").lower()  # 'isolated' or 'cross'

    @classmethod
    def validate(cls):
        has_keys = True
        
        # Check Binance Keys
        has_binance = True
        if not cls.API_KEY or cls.API_KEY == "your_api_key_here":
            has_binance = False
        if not cls.SECRET_KEY or cls.SECRET_KEY == "your_api_secret_here":
            has_binance = False
            
        # Check CoinDCX Keys
        has_coindcx = True
        if not cls.COINDCX_API_KEY or cls.COINDCX_API_KEY == "your_coindcx_key_here":
            has_coindcx = False
        if not cls.COINDCX_SECRET_KEY or cls.COINDCX_SECRET_KEY == "your_coindcx_secret_here":
            has_coindcx = False

        if not has_binance and not has_coindcx:
            print("WARNING: Neither Binance nor CoinDCX credentials found. Trading engine will run in DRY-RUN mode.")
            has_keys = False
        elif has_coindcx:
            print(f"[INIT] CoinDCX integration active. Mode: {'PAPER TRADING (Demo)' if cls.PAPER_TRADING else 'LIVE TRADING'}")
        elif has_binance:
            print(f"[INIT] Binance integration active. Mode: {'PAPER TRADING (Demo)' if cls.PAPER_TRADING else 'LIVE TRADING'}")

        # Validate critical numeric ranges to prevent account-wipe settings
        if cls.RISK_PCT > 5.0:
            print(f"⚠️  WARNING: RISK_PCT={cls.RISK_PCT}% is dangerously high! Recommended: 0.5–2%. Capping at 5%.")
            cls.RISK_PCT = 5.0
        if cls.MIN_RISK_REWARD_RATIO < 1.0:
            print(f"⚠️  WARNING: MIN_RISK_REWARD_RATIO={cls.MIN_RISK_REWARD_RATIO} is below 1.0. This means losses > gains by design. Minimum set to 1.5.")
            cls.MIN_RISK_REWARD_RATIO = 1.5
            
        print("--- PrimeSignal Institutional Settings Loaded ---")
        print(f"  Target Symbol       : {cls.SYMBOL}")
        print(f"  Execution Frame     : {cls.LTF_TIMEFRAME} | Trend Frame: {cls.HTF_TIMEFRAME}")
        print(f"  Account Risk Limit  : {cls.RISK_PCT}% | Max Daily Drawdown: {cls.MAX_DAILY_LOSS_PCT}%")
        print(f"  SMC Indicators      : RSI ({cls.RSI_PERIOD}), ATR ({cls.ATR_PERIOD}), EMA ({cls.TREND_EMA})")
        print(f"  ML Confidence Min   : {cls.ML_CONFIRMATION_THRESHOLD * 100:.1f}%")
        if has_coindcx:
            print(f"  Exchange Routing    : CoinDCX (INR Pairs: {cls.COINDCX_TRADE_INR})")
        else:
            print(f"  Exchange Routing    : Binance (Sandbox: {cls.USE_TESTNET})")
        print("-------------------------------------------------")
        return has_keys

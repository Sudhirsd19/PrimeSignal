# PrimeSignal v2.6.0 - Institutional Strategy Engine (1.0R TP1 / 2.0R TP2, 25.0 ADX, EMA 21 Pullback, Max 6 Daily Trades)
import os
import sys

# Reconfigure stdout/stderr to utf-8 on Windows to prevent UnicodeEncodeError
if sys.platform == 'win32':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
        getattr(sys.stderr, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except (AttributeError, Exception):
        pass

import aiohttp.connector
import aiohttp.resolver
aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver

# C-06 FIX: dotenv is optional. A missing dev dependency must not make the whole
# engine un-importable (it prevented the test-suite and CLI tools from running).
try:  # pragma: no cover - trivial import shim
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class Config:
    # Exchange API settings
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "yes")
    
    # Product Settings (Top 20 High-Liquidity Institutional USDT Pairs)
    SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
    SUPPORTED_SYMBOLS = os.getenv("SUPPORTED_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,DOGE/USDT,ADA/USDT,AVAX/USDT,SUI/USDT,LINK/USDT,DOT/USDT,NEAR/USDT,LTC/USDT,BCH/USDT,UNI/USDT,APT/USDT,ICP/USDT,TRX/USDT,ATOM/USDT,OP/USDT").split(",")
    TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "0.001"))
    ENABLE_DYNAMIC_SCANNER = os.getenv("ENABLE_DYNAMIC_SCANNER", "False").lower() in ("true", "1", "yes")
    
    # ─── Per-Trade Risk Ladder ───
    # RISK_PCT is the BASELINE per-trade risk (%). The strategy scales it by the
    # setup-quality score, so the effective per-trade risk is always
    #   RISK_PCT x {LOW_MULT, MID_MULT, HIGH_MULT}
    # With the default RISK_PCT=1.0 the live ladder is 0.75% / 1.00% / 1.25%.
    # Lower RISK_PCT to trade smaller; raise it to trade larger. This value now
    # genuinely drives position sizing (it previously had no effect at all).
    RISK_PCT = float(os.getenv("RISK_PCT", "1.0"))
    RISK_TIER_LOW_MULT = float(os.getenv("RISK_TIER_LOW_MULT", "0.75"))
    RISK_TIER_MID_MULT = float(os.getenv("RISK_TIER_MID_MULT", "1.0"))
    RISK_TIER_HIGH_MULT = float(os.getenv("RISK_TIER_HIGH_MULT", "1.25"))
    # Score thresholds that select each ladder tier
    RISK_TIER_MID_SCORE = float(os.getenv("RISK_TIER_MID_SCORE", "3.5"))
    RISK_TIER_HIGH_SCORE = float(os.getenv("RISK_TIER_HIGH_SCORE", "4.5"))
    MAX_TRADE_ALLOCATION_PCT = float(os.getenv("MAX_TRADE_ALLOCATION_PCT", "0.35")) # Max 35% of total wallet equity per trade
    DYNAMIC_BE_BUFFER_PCT = float(os.getenv("DYNAMIC_BE_BUFFER_PCT", "0.0030")) # Dynamic Roundtrip Fee (0.20%) + Slippage Buffer (0.10%)
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0"))
    # Daily Profit Lock: Auto-pause after reaching daily profit target to protect gains
    MAX_DAILY_PROFIT_PCT = float(os.getenv("MAX_DAILY_PROFIT_PCT", "10.0"))
    ENABLE_DAILY_PROFIT_LOCK = os.getenv("ENABLE_DAILY_PROFIT_LOCK", "True").lower() in ("true", "1", "yes")
    # Dynamic Kelly Criterion Position Sizing
    ENABLE_KELLY_SIZING = os.getenv("ENABLE_KELLY_SIZING", "False").lower() in ("true", "1", "yes")
    KELLY_LOOKBACK_TRADES = int(os.getenv("KELLY_LOOKBACK_TRADES", "20"))
    CONSECUTIVE_LOSS_LIMIT = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "2"))
    MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "2"))
    MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "6"))
    TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.015")) # Deprecated in favor of ATR
    TRAILING_ATR_MULT = float(os.getenv("TRAILING_ATR_MULT", "1.5"))
    TSL_ACTIVATION_R = float(os.getenv("TSL_ACTIVATION_R", "0.55")) # Early Breakeven Lock activation at +0.55R
    MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.0")) # TP1 Target: 1.0R
    RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "2.0")) # TP2 Target: 2.0R
    TP1_SCALE_OUT_PCT = float(os.getenv("TP1_SCALE_OUT_PCT", "0.65")) # 65% profit booking at TP1 (1.0R) target
    # Fraction of the REMAINING position booked at TP2. Was read by main.py but
    # missing from Config, so it silently always used 0.65 (P2-3 FIX).
    TP2_REMAINING_SCALE_PCT = float(os.getenv("TP2_REMAINING_SCALE_PCT", "0.65")) # 65% of remainder at TP2
    
    # Triple-Barrier Label constants (used in ml/confirmation.py FIX-B)
    # ML_LABEL_SL_PCT mirrors the strategy's hard 0.5% minimum stop distance.
    # When ML_LABEL_TP_AUTO is True the TP barrier is DERIVED from the live trade
    # geometry (SL x MIN_RISK_REWARD_RATIO) so training labels and execution
    # targets can never drift apart again (H-03 FIX).
    ML_LABEL_TP_AUTO     = os.getenv("ML_LABEL_TP_AUTO", "True").lower() in ("true", "1", "yes")
    ML_LABEL_TP_PCT      = float(os.getenv("ML_LABEL_TP_PCT",      "0.006"))
    ML_LABEL_SL_PCT      = float(os.getenv("ML_LABEL_SL_PCT",      "-0.005"))
    ML_LABEL_LOOKAHEAD   = int(os.getenv(  "ML_LABEL_LOOKAHEAD",   "20"))
    
    # ─── ULTIMATE SHIELD: 5 Advanced Institutional Protection Parameters ───
    # 1. Stagnation Killer (Time-based exit)
    ENABLE_TIME_STOP = os.getenv("ENABLE_TIME_STOP", "True").lower() in ("true", "1", "yes")
    MAX_STAGNANT_CANDLES = int(os.getenv("MAX_STAGNANT_CANDLES", "24")) # 6 hours on 15m
    STAGNANT_MAX_R_DISTANCE = float(os.getenv("STAGNANT_MAX_R_DISTANCE", "0.25")) # exit if within ±0.25R
    
    # 2. Early Structural Invalidation Exit
    ENABLE_STRUCTURAL_EXIT = os.getenv("ENABLE_STRUCTURAL_EXIT", "False").lower() in ("true", "1", "yes") # Disabled by default to avoid stop-hunt panic exits
    EARLY_EXIT_MAX_LOSS_R = float(os.getenv("EARLY_EXIT_MAX_LOSS_R", "0.45")) # Cut trade at max -0.45R instead of full -1.0R
    
    # 3. Funding Rate & Crowded Sentiment Filter
    ENABLE_FUNDING_RATE_FILTER = os.getenv("ENABLE_FUNDING_RATE_FILTER", "True").lower() in ("true", "1", "yes")
    MAX_FUNDING_RATE_PCT = float(os.getenv("MAX_FUNDING_RATE_PCT", "0.035")) # 0.035% per 8h
    
    # 4. Volatility Compression / Bollinger Band Squeeze Filter
    ENABLE_BB_SQUEEZE_FILTER = os.getenv("ENABLE_BB_SQUEEZE_FILTER", "True").lower() in ("true", "1", "yes")
    BB_SQUEEZE_PERCENTILE = float(os.getenv("BB_SQUEEZE_PERCENTILE", "12.0")) # lowest 12% bandwidth
    
    # 5. Macro Economic News Blackout Window Filter (CPI/FOMC auto-pause)
    # Backed by a REAL economic calendar feed (see core/macro_calendar.py).
    # Default feed is the free ForexFactory weekly calendar (no API key required).
    ENABLE_MACRO_NEWS_FILTER = os.getenv("ENABLE_MACRO_NEWS_FILTER", "True").lower() in ("true", "1", "yes")
    ECONOMIC_CALENDAR_URL = os.getenv(
        "ECONOMIC_CALENDAR_URL",
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    )
    ECONOMIC_CALENDAR_CACHE_MINS = int(os.getenv("ECONOMIC_CALENDAR_CACHE_MINS", "30"))
    NEWS_BLACKOUT_BEFORE_MIN = int(os.getenv("NEWS_BLACKOUT_BEFORE_MIN", "15"))
    NEWS_BLACKOUT_AFTER_MIN = int(os.getenv("NEWS_BLACKOUT_AFTER_MIN", "20"))
    # Minimum calendar impact that triggers a blackout: 'low' | 'medium' | 'high'.
    NEWS_MIN_IMPACT = os.getenv("NEWS_MIN_IMPACT", "high").strip().lower()
    # Opt-in legacy behaviour: block fixed recurring clock windows every weekday.
    # OFF by default because the previous implementation blocked ~90 minutes of
    # every single weekday regardless of whether any data was actually due,
    # and it was presented to users as a news-calendar filter (H-06 FIX).
    ENABLE_RECURRING_NEWS_WINDOWS = os.getenv("ENABLE_RECURRING_NEWS_WINDOWS", "False").lower() in ("true", "1", "yes")
    
    # ─── NEXT-GEN PROPRIETARY QUANT INNOVATIONS ───
    # 1. Liquidation Magnetic Heatmap & Hunt Engine
    ENABLE_LIQUIDATION_MAGNET = os.getenv("ENABLE_LIQUIDATION_MAGNET", "True").lower() in ("true", "1", "yes")
    LIQUIDATION_PROXIMITY_PCT = float(os.getenv("LIQUIDATION_PROXIMITY_PCT", "0.003")) # 0.3% pool proximity
    
    # 2. Real-Time CVD Absorption & Footprint Divergence Engine
    ENABLE_CVD_ABSORPTION = os.getenv("ENABLE_CVD_ABSORPTION", "True").lower() in ("true", "1", "yes")
    CVD_DIVERGENCE_LOOKBACK = int(os.getenv("CVD_DIVERGENCE_LOOKBACK", "20"))
    
    # 3. Cross-Asset Lead-Lag Latency Momentum Arbitrage (BTC Velocity Propagation)
    ENABLE_LEAD_LAG_ARBITRAGE = os.getenv("ENABLE_LEAD_LAG_ARBITRAGE", "True").lower() in ("true", "1", "yes")
    BTC_IMPULSE_VELOCITY_PCT = float(os.getenv("BTC_IMPULSE_VELOCITY_PCT", "0.0030")) # 0.30% spike in 30s
    ALT_LAG_MAX_REACTION_PCT = float(os.getenv("ALT_LAG_MAX_REACTION_PCT", "0.0008")) # alt moved < 0.08%
    
    # 4. Dual-Brain Adversarial AI Debate Courtroom
    ENABLE_ADVERSARIAL_DEBATE = os.getenv("ENABLE_ADVERSARIAL_DEBATE", "True").lower() in ("true", "1", "yes")
    MIN_AI_CONVICTION_PCT = float(os.getenv("MIN_AI_CONVICTION_PCT", "0.70")) # 70% net dominance required
    
    # Session & Timing Filters
    ENABLE_SESSION_FILTER = os.getenv("ENABLE_SESSION_FILTER", "True").lower() in ("true", "1", "yes")
    ENABLE_WEEKEND_FILTER = os.getenv("ENABLE_WEEKEND_FILTER", "True").lower() in ("true", "1", "yes")
    
    # Strategy settings
    HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "1h")
    HTF_MAX_STALENESS_MULT = float(os.getenv("HTF_MAX_STALENESS_MULT", "2.0"))
    LTF_TIMEFRAME = os.getenv("LTF_TIMEFRAME", "15m")
    ADX_MIN_THRESHOLD = float(os.getenv("ADX_MIN_THRESHOLD", "20.0"))
    SHORT_EMA = int(os.getenv("SHORT_EMA", "9"))
    LONG_EMA = int(os.getenv("LONG_EMA", "21"))
    TREND_EMA = int(os.getenv("TREND_EMA", "200"))
    
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
    RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", "35"))
    RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", "65"))
    ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
    MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0005"))
    MAX_ZONE_AGE_CANDLES = int(os.getenv("MAX_ZONE_AGE_CANDLES", "100"))
    ZONE_BUFFER_PCT = float(os.getenv("ZONE_BUFFER_PCT", "0.005"))
    VWAP_TOLERANCE = float(os.getenv("VWAP_TOLERANCE", "0.002"))
    MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.002"))
    MIN_24H_VOL_USDT = float(os.getenv("MIN_24H_VOL_USDT", "50000000"))
    MAX_CANDLE_MOVE_PCT = float(os.getenv("MAX_CANDLE_MOVE_PCT", "0.015"))
    VOLATILITY_PAUSE_CANDLES = int(os.getenv("VOLATILITY_PAUSE_CANDLES", "2"))
    COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "20"))
    TP_EXIT_COOLDOWN_MINUTES = int(os.getenv("TP_EXIT_COOLDOWN_MINUTES", "25"))   # M-04 FIX: Was hardcoded in main.py
    POST_EXIT_COOLDOWN_MINUTES = int(os.getenv("POST_EXIT_COOLDOWN_MINUTES", "15"))  # M-05 FIX: Was hardcoded in main.py
    MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))
    FEE_RATE = float(os.getenv("FEE_RATE", "0.00075"))
    MAX_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "6.0"))    # Total portfolio risk cap (%)
    
    # Structure & Divergence settings
    STRUCTURE_LOOKBACK = int(os.getenv("STRUCTURE_LOOKBACK", "30"))
    RSI_DIVERGENCE_LOOKBACK = int(os.getenv("RSI_DIVERGENCE_LOOKBACK", "20"))
    
    # Machine Learning configurations
    ML_CONFIRMATION_THRESHOLD = float(os.getenv("ML_CONFIRMATION_THRESHOLD", "0.60"))
    ML_TRAIN_BARS = int(float(os.getenv("ML_TRAIN_BARS", "25")))  # C-04 FIX: int(float()) to handle decimal env strings
    # ML_GATE_MODE controls whether the model actually filters entries:
    #   'auto' (default) -> gate entries only when the model proved real edge in
    #                       time-series cross-validation, otherwise only scale risk
    #   'gate'           -> always require ML confirmation to enter
    #   'risk'           -> never block, only adjust risk/Targets (legacy behaviour)
    #   'off'            -> ignore the model entirely
    ML_GATE_MODE = os.getenv("ML_GATE_MODE", "auto").strip().lower()
    ML_MIN_CV_ACCURACY = float(os.getenv("ML_MIN_CV_ACCURACY", "0.55"))
    # LTF bars fetched at warm-up. The ML model needs 2000+ bars for a stable fit;
    # the pipeline paginates past Binance's 1000-bar per-request limit.
    LTF_HISTORY_BARS = int(os.getenv("LTF_HISTORY_BARS", "2000"))
    
    # Test Mode config
    TEST_MODE = os.getenv("TEST_MODE", "False").lower() in ("true", "1", "yes")
    PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "yes")
    PAPER_CURRENCY = os.getenv("PAPER_CURRENCY", "INR" if os.getenv("COINDCX_TRADE_INR", "True").lower() in ("true", "1", "yes") else "USDT").upper()
    PAPER_STARTING_BALANCE = float(os.getenv("PAPER_STARTING_BALANCE", "2000.0" if os.getenv("PAPER_CURRENCY", "INR").upper() == "INR" else "10000.0"))
    
    # Telegram notifier settings
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # CoinDCX settings
    COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
    COINDCX_SECRET_KEY = os.getenv("COINDCX_SECRET_KEY", "")
    COINDCX_TRADE_INR = os.getenv("COINDCX_TRADE_INR", "True").lower() in ("true", "1", "yes")
    USDT_INR_RATE = float(os.getenv("USDT_INR_RATE", "85.0"))

    # Exchange Routing Venue: 'BINANCE' (default) or 'COINDCX'
    TRADING_VENUE = os.getenv("TRADING_VENUE", "BINANCE").strip().upper()

    # Exchange type: 'spot' or 'futures' (Binance USDT-M Futures)
    EXCHANGE_TYPE = os.getenv("EXCHANGE_TYPE", "spot").lower()

    # Futures-specific settings (only used when EXCHANGE_TYPE='futures')
    FUTURES_LEVERAGE = int(os.getenv("FUTURES_LEVERAGE", "1"))
    FUTURES_MARGIN_MODE = os.getenv("FUTURES_MARGIN_MODE", "isolated").lower()  # 'isolated' or 'cross'

    # ─── Venue / Instrument Capabilities ───
    @classmethod
    def venue_supports_short(cls) -> bool:
        """True only when the configured venue can actually hold a short.

        Spot venues (Binance spot, CoinDCX spot) cannot short. Emitting SELL
        signals on those venues produced bare market sells of an asset the
        account may not own, and a BUY stop-loss that opens a NEW long instead
        of protecting the position (C-01 FIX).
        """
        return str(getattr(cls, 'EXCHANGE_TYPE', 'spot')).lower() == 'futures'

    # ─── Canonical Risk Units ───
    @classmethod
    def get_max_portfolio_risk_fraction(cls) -> float:
        """Total portfolio risk cap as a FRACTION (e.g. 6.0% -> 0.06).

        Single source of truth so the main loop and RiskManager can never drift
        into mixing percent and fraction units again (C-02 FIX).
        """
        raw = float(getattr(cls, 'MAX_PORTFOLIO_RISK_PCT', 6.0))
        return raw / 100.0 if raw > 0.2 else raw

    @classmethod
    def risk_pct_for_score(cls, score: float) -> float:
        """Per-trade risk FRACTION for a setup-quality score.

        This is the function the live loop uses, so the value printed by
        validate() and the value actually risked are guaranteed to agree.
        """
        base = float(getattr(cls, 'RISK_PCT', 1.0)) / 100.0
        score = float(score or 0.0)
        if score >= float(getattr(cls, 'RISK_TIER_HIGH_SCORE', 4.5)):
            mult = float(getattr(cls, 'RISK_TIER_HIGH_MULT', 1.25))
        elif score >= float(getattr(cls, 'RISK_TIER_MID_SCORE', 3.5)):
            mult = float(getattr(cls, 'RISK_TIER_MID_MULT', 1.0))
        else:
            mult = float(getattr(cls, 'RISK_TIER_LOW_MULT', 0.75))
        return base * mult

    @classmethod
    def get_risk_config_snapshot(cls) -> dict:
        """Returns a canonical serializable dictionary of all governing risk parameters."""
        return {
            "risk_pct": cls.RISK_PCT,
            "risk_tier_low_pct": round(cls.risk_pct_for_score(0.0) * 100.0, 4),
            "risk_tier_mid_pct": round(cls.risk_pct_for_score(cls.RISK_TIER_MID_SCORE) * 100.0, 4),
            "risk_tier_high_pct": round(cls.risk_pct_for_score(cls.RISK_TIER_HIGH_SCORE) * 100.0, 4),
            "max_portfolio_risk_fraction": cls.get_max_portfolio_risk_fraction(),
            "tp2_remaining_scale_pct": cls.TP2_REMAINING_SCALE_PCT,
            "ml_gate_mode": cls.ML_GATE_MODE,
            "ml_label_tp_auto": cls.ML_LABEL_TP_AUTO,
            "max_daily_loss_pct": cls.MAX_DAILY_LOSS_PCT,
            "max_daily_profit_pct": cls.MAX_DAILY_PROFIT_PCT,
            "max_daily_trades": cls.MAX_DAILY_TRADES,
            "max_trade_allocation_pct": cls.MAX_TRADE_ALLOCATION_PCT,
            "trailing_atr_mult": cls.TRAILING_ATR_MULT,
            "min_risk_reward_ratio": cls.MIN_RISK_REWARD_RATIO,
            "risk_reward_ratio": cls.RISK_REWARD_RATIO,
            "tp1_scale_out_pct": cls.TP1_SCALE_OUT_PCT,
            "enable_funding_rate_filter": cls.ENABLE_FUNDING_RATE_FILTER,
            "max_funding_rate_pct": cls.MAX_FUNDING_RATE_PCT,
            "exchange_type": cls.EXCHANGE_TYPE.lower(),
            "futures_leverage": cls.FUTURES_LEVERAGE,
            "futures_margin_mode": cls.FUTURES_MARGIN_MODE.lower(),
            "trading_venue": cls.TRADING_VENUE.upper(),
        }

    @classmethod
    def get_risk_config_hash(cls) -> str:
        """Computes a deterministic SHA-256 hash of the canonical risk configuration."""
        import json
        import hashlib
        canonical_str = json.dumps(cls.get_risk_config_snapshot(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

    @classmethod
    async def update_dynamic_symbols(cls, limit: int = 15):
        """Dynamically fetches the top trending USDT pairs from Binance based on volume and momentum."""
        if not getattr(cls, 'ENABLE_DYNAMIC_SCANNER', False):
            print(f"[DYNAMIC SCANNER] Dynamic hot-swap is disabled. Retaining Top 20 Institutional list: {len(cls.SUPPORTED_SYMBOLS)} pairs.")
            return
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ticker/24hr') as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter USDT pairs, exclude stablecoins, fiats & leveraged tokens
                        exclude = ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'EURUSDT', 'BUSDUSDT', 'USDPUSDT', 'WBTCUSDT', 'AUDUSDT', 'GBPUSDT', 'JPYUSDT', 'TRYUSDT', 'ZARUSDT']
                        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and d['symbol'] not in exclude and not d['symbol'].endswith('UPUSDT') and not d['symbol'].endswith('DOWNUSDT')]
                        
                        # Sort by Quote Volume (USDT) descending to ensure deep liquidity
                        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                        
                        # Take Top 60 most liquid coins
                        top_liquid = usdt_pairs[:60]
                        
                        # Sort these Top 60 by absolute 24h price change (High Momentum / Trending)
                        top_liquid.sort(key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)
                        
                        # Select Top N
                        top_n = top_liquid[:limit]
                        
                        dynamic_symbols = [f"{d['symbol'].replace('USDT', '')}/USDT" for d in top_n]
                        
                        if dynamic_symbols:
                            cls.SUPPORTED_SYMBOLS = dynamic_symbols
                            # Update default UI chart symbol to the #1 trending coin
                            if cls.SYMBOL not in dynamic_symbols:
                                cls.SYMBOL = dynamic_symbols[0]
                                
                            print(f"\n[DYNAMIC SCANNER] Loaded Top {limit} Trending Coins!")
                            print(f"-> {', '.join(dynamic_symbols)}\n")
        except Exception as e:
            print(f"[WARNING] Failed to update dynamic symbols (falling back to static list): {e}")

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

        if cls.TRADING_VENUE not in ("BINANCE", "COINDCX"):
            print(f"WARNING: Invalid TRADING_VENUE='{cls.TRADING_VENUE}'. Defaulting to 'BINANCE'.")
            cls.TRADING_VENUE = "BINANCE"

        if cls.TRADING_VENUE == "COINDCX":
            if cls.EXCHANGE_TYPE == "futures":
                raise ValueError("CRITICAL CONFIG ERROR: CoinDCX does not support futures trading via API. Set EXCHANGE_TYPE=spot or TRADING_VENUE=BINANCE.")
            if not has_coindcx and not cls.PAPER_TRADING:
                raise ValueError("CRITICAL CONFIG ERROR: TRADING_VENUE is set to COINDCX for live trading, but valid CoinDCX credentials were not found.")

        if not has_binance and not has_coindcx:
            print("WARNING: Neither Binance nor CoinDCX credentials found. Trading engine will run in DRY-RUN mode.")
            has_keys = False
        elif cls.TRADING_VENUE == "COINDCX" and has_coindcx:
            print(f"[INIT] CoinDCX integration active. Mode: {'PAPER TRADING (Demo)' if cls.PAPER_TRADING else 'LIVE TRADING'}")
        elif cls.TRADING_VENUE == "BINANCE" and has_binance:
            print(f"[INIT] Binance integration active. Mode: {'PAPER TRADING (Demo)' if cls.PAPER_TRADING else 'LIVE TRADING'}")

        # Validate critical numeric ranges to prevent account-wipe settings.
        # Cap the HIGHEST rung of the ladder (not just the baseline) so a large
        # RISK_PCT cannot silently push real per-trade risk past 5%.
        high_tier_pct = cls.risk_pct_for_score(cls.RISK_TIER_HIGH_SCORE) * 100.0
        if high_tier_pct > 5.0:
            print(f"⚠️  WARNING: RISK_PCT={cls.RISK_PCT}% puts the top risk tier at {high_tier_pct:.2f}% per trade — dangerously high! Recommended baseline: 0.5–1.5%. Capping the ladder at 5%.")
            cls.RISK_PCT = 5.0 / max(cls.RISK_TIER_HIGH_MULT, 1e-9)
        if cls.MIN_RISK_REWARD_RATIO < 1.0:
            print(f"⚠️  WARNING: MIN_RISK_REWARD_RATIO={cls.MIN_RISK_REWARD_RATIO} is below 1.0. This means losses > gains by design. Minimum set to 1.0.")
            cls.MIN_RISK_REWARD_RATIO = 1.0

        # Venue capability guard (C-01 FIX): fail loudly rather than emitting
        # un-executable SELL signals on a venue that cannot short.
        supports_short = cls.venue_supports_short()
        if not supports_short:
            print(f"ℹ️  DIRECTION MODE: {cls.TRADING_VENUE} is a SPOT venue — SHORT signals are disabled automatically.")
            print("     -> Set EXCHANGE_TYPE=futures (Binance USDT-M) to enable short setups.")
        print("--- PrimeSignal Institutional Settings Loaded ---")
        print(f"  Target Symbol       : {cls.SYMBOL}")
        print(f"  Execution Frame     : {cls.LTF_TIMEFRAME} | Trend Frame: {cls.HTF_TIMEFRAME}")
        print(f"  Live Risk Ladder    : {cls.risk_pct_for_score(0.0)*100:.2f}% (<= {cls.RISK_TIER_MID_SCORE}) | "
              f"{cls.risk_pct_for_score(cls.RISK_TIER_MID_SCORE)*100:.2f}% (>= {cls.RISK_TIER_MID_SCORE}) | "
              f"{cls.risk_pct_for_score(cls.RISK_TIER_HIGH_SCORE)*100:.2f}% (>= {cls.RISK_TIER_HIGH_SCORE})")
        print(f"  Portfolio Risk Cap  : {cls.get_max_portfolio_risk_fraction()*100:.2f}% | Max Daily Drawdown: {cls.MAX_DAILY_LOSS_PCT}%")
        print(f"  SMC Indicators      : RSI ({cls.RSI_PERIOD}), ATR ({cls.ATR_PERIOD}), EMA ({cls.TREND_EMA})")
        print(f"  ML Confirmation Min : {cls.ML_CONFIRMATION_THRESHOLD * 100:.1f}% | Gate Mode: {cls.ML_GATE_MODE}")
        print(f"  ML Label Barriers   : SL {abs(cls.ML_LABEL_SL_PCT)*100:.2f}% | TP "
              f"{'auto (' + format(abs(cls.ML_LABEL_SL_PCT) * cls.MIN_RISK_REWARD_RATIO * 100, '.2f') + '%)' if cls.ML_LABEL_TP_AUTO else format(abs(cls.ML_LABEL_TP_PCT)*100, '.2f') + '%'}")
        print(f"  Trading Venue       : {cls.TRADING_VENUE} ({'Futures' if cls.EXCHANGE_TYPE == 'futures' else 'Spot'}) | Shorts: {'ENABLED' if supports_short else 'DISABLED'}")
        print(f"  Macro News Window   : {cls.NEWS_MIN_IMPACT}-impact events, {cls.NEWS_BLACKOUT_BEFORE_MIN}m before / {cls.NEWS_BLACKOUT_AFTER_MIN}m after" +
              (" + legacy recurring windows" if cls.ENABLE_RECURRING_NEWS_WINDOWS else ""))
        print("-------------------------------------------------")
        return has_keys



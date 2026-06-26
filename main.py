import asyncio
import sys
import uvicorn
import time
import datetime
import json
from pathlib import Path

# Reconfigure stdout/stderr to utf-8 on Windows to prevent UnicodeEncodeError
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from config import Config
from execution.execution_engine import ExecutionEngine
from core.data_pipeline import RealTimeDataPipeline
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
from strategies.indicators import prepare_dataframe, calculate_atr
from ml.confirmation import MLSignalConfirmator
from risk.risk_manager import RiskManager
from alerts.notifier import TelegramNotifier
from dashboard.app import app, DashboardState, add_log_message

class PrimeSignalBot:
    def __init__(self):
        self.has_keys = Config.validate()
        
        # Initialize Core Modules
        self.execution = ExecutionEngine()
        self.pipeline = RealTimeDataPipeline(self.execution)
        self.strategy = MultiTimeframeSMCStrategy()
        self.risk = RiskManager()
        self.notifier = TelegramNotifier()
        
        self.ml_models = {sym: MLSignalConfirmator() for sym in Config.SUPPORTED_SYMBOLS}
        
        # Internal State tracking (Per Symbol)
        self.in_position = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self.position_side = {sym: "HOLD" for sym in Config.SUPPORTED_SYMBOLS}
        self.entry_price = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.entry_price_usdt = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.stop_loss = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.take_profit = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.highest_price_reached = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.lowest_price_reached = {sym: 999999.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.position_size = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.position_mode = {sym: "STRICT" for sym in Config.SUPPORTED_SYMBOLS}
        self.entry_time = {sym: 0 for sym in Config.SUPPORTED_SYMBOLS}
        self.last_trade_time = {sym: 0 for sym in Config.SUPPORTED_SYMBOLS}
        self.last_zone_traded = {sym: None for sym in Config.SUPPORTED_SYMBOLS}
        self.volatility_pause_until = {sym: 0 for sym in Config.SUPPORTED_SYMBOLS}
        self.partial_tp_taken = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self.take_profit_1r = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.tp2_taken = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self.take_profit_2r = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.consecutive_losses = 0
        self.global_pause_until = 0
        self.relaxed_losses = 0
        self.relaxed_disabled_until = 0
        self.relaxed_trades_today = 0
        self.trades_today = 0
        self.last_trade_day = datetime.datetime.now(datetime.timezone.utc).date()
        self.trade_history = []
        self.cluster_loss_pause_until = 0
        self.cluster_risk_penalty = False
        self.global_last_trade_time = 0
        self.traded_zones_cache = {}

        # Dry-run virtual balance (used when no API keys are set)
        self._dry_run_balance_usdt = 10000.0   # starting paper balance
        
        if not self.has_keys:
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            DashboardState.balance_base = 0.0
            print("[INIT] ✅ Dry-run mode: Virtual balance initialized to $10,000 USDT")

        # Per-symbol locks to prevent concurrent candle processing on the same symbol
        self._candle_locks = {sym: asyncio.Lock() for sym in Config.SUPPORTED_SYMBOLS}
        self._pending_candle_evaluations = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self._last_reset_date = datetime.datetime.now(datetime.timezone.utc).date()
        
        # Link callbacks
        self.pipeline.on_candle_close_callback = self.on_candle_close

    _STATE_FILE = Path("bot_state.json")

    def save_state(self):
        """Persist current position state to disk for crash recovery."""
        state = {
            'in_position': self.in_position,
            'position_side': self.position_side,
            'entry_price': self.entry_price,
            'entry_price_usdt': self.entry_price_usdt,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'entry_time': self.entry_time,
            'highest_price_reached': self.highest_price_reached,
            'lowest_price_reached': self.lowest_price_reached,
            '_dry_run_balance_usdt': self._dry_run_balance_usdt,
            # FIX F: Persist partial TP and cooldown state for crash recovery
            'partial_tp_taken': self.partial_tp_taken,
            'tp2_taken': self.tp2_taken,
            'take_profit_1r': self.take_profit_1r,
            'take_profit_2r': self.take_profit_2r,
            'last_trade_time': self.last_trade_time,
            'trades_today': self.trades_today,
            'trade_history': self.trade_history,
            'cluster_loss_pause_until': self.cluster_loss_pause_until,
            'global_pause_until': self.global_pause_until,
        }
        try:
            self._STATE_FILE.write_text(json.dumps(state))
        except Exception as e:
            print(f"[STATE] Failed to save state: {e}")

    def is_live_trading(self):
        """Returns True only if we should send real API orders."""
        return self.has_keys and not Config.PAPER_TRADING

    def load_state(self):
        """Restore position state from disk after a restart."""
        if not self._STATE_FILE.exists():
            return
        try:
            state = json.loads(self._STATE_FILE.read_text())
            
            # Helper to safely load dict state, falling back to default if new symbols were added
            def safe_load(key, default_val):
                loaded_dict = state.get(key, {})
                return {sym: loaded_dict.get(sym, default_val) for sym in Config.SUPPORTED_SYMBOLS}

            self.in_position = safe_load('in_position', False)
            self.position_side = safe_load('position_side', 'HOLD')
            self.entry_price = safe_load('entry_price', 0.0)
            self.entry_price_usdt = safe_load('entry_price_usdt', 0.0)
            self.stop_loss = safe_load('stop_loss', 0.0)
            self.take_profit = safe_load('take_profit', 0.0)
            self.position_size = safe_load('position_size', 0.0)
            self.entry_time = safe_load('entry_time', 0)
            self.highest_price_reached = safe_load('highest_price_reached', 0.0)
            self.lowest_price_reached = safe_load('lowest_price_reached', 999999.0)
            self._dry_run_balance_usdt = state.get('_dry_run_balance_usdt', 10000.0)

            # FIX F: Restore partial TP and cooldown state
            self.partial_tp_taken = safe_load('partial_tp_taken', False)
            self.tp2_taken = safe_load('tp2_taken', False)
            self.take_profit_1r = safe_load('take_profit_1r', 0.0)
            self.take_profit_2r = safe_load('take_profit_2r', 0.0)
            self.last_trade_time = safe_load('last_trade_time', 0)
            self.trades_today = state.get('trades_today', 0)
            self.trade_history = state.get('trade_history', [])
            self.cluster_loss_pause_until = state.get('cluster_loss_pause_until', 0)
            self.global_pause_until = state.get('global_pause_until', 0)
            
            # FIX #7: Ghost position detection — clear positions with invalid data
            for sym in Config.SUPPORTED_SYMBOLS:
                if self.in_position[sym]:
                    if self.position_size[sym] <= 0 or self.entry_price[sym] <= 0:
                        print(f"[STATE] ⚠️ Ghost position detected for {sym} (size={self.position_size[sym]}, entry={self.entry_price[sym]}). Resetting.")
                        self.in_position[sym] = False
                        self.position_side[sym] = "HOLD"
                        self.entry_price[sym] = 0.0
                        self.position_size[sym] = 0.0
                        self.stop_loss[sym] = 0.0
                        self.take_profit[sym] = 0.0

            # Sync to dashboard for active UI symbol
            sym = Config.SYMBOL
            DashboardState.in_position = self.in_position[sym]
            DashboardState.position_side = self.position_side[sym]
            DashboardState.entry_price = self.entry_price[sym]
            DashboardState.stop_loss = self.stop_loss[sym]
            DashboardState.take_profit = self.take_profit[sym]
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            
            open_positions = sum(1 for s in Config.SUPPORTED_SYMBOLS if self.in_position[s])
            if open_positions > 0:
                add_log_message(f"[STATE] Recovered {open_positions} open positions from disk.")
            else:
                add_log_message("[STATE] State file loaded — no open position to recover.")
        except Exception as e:
            print(f"[STATE] Failed to load state: {e}")

    async def initialize(self):
        add_log_message("Starting system initialization for all supported symbols...")

        self.load_state()
        await self.pipeline.start()
        await asyncio.sleep(3)
        
        # Initial Balance load
        if self.has_keys and not Config.PAPER_TRADING:
            balance = await self.execution.fetch_balance()
            if balance:
                if Config.COINDCX_TRADE_INR:
                    inr_balance = balance.get('total', {}).get('INR', None)
                    if inr_balance is not None:
                        DashboardState.balance_usdt = inr_balance
                    else:
                        add_log_message("[WARNING] CoinDCX INR Balance fetch returned None. Keeping last known value.")
                else:
                    usdt_balance = balance.get('total', {}).get('USDT', None)
                    if usdt_balance and usdt_balance > 0:
                        DashboardState.balance_usdt = usdt_balance
                    else:
                        add_log_message(f"[WARNING] Balance fetch returned {usdt_balance}. Check account type. Keeping last known value.")
                DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
        else:
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            DashboardState.balance_base = 0.0

        # Initial CoinDCX Profile Load
        if self.execution.coindcx_client:
            try:
                profile = await self.execution.fetch_coindcx_user_info()
                if profile:
                    DashboardState.coindcx_profile = {
                        "name": f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip() or "CoinDCX User",
                        "email": profile.get("email", "unknown"),
                        "id": profile.get("coindcx_id", "unknown"),
                        "status": "Connected"
                    }
                else:
                    DashboardState.coindcx_profile = {
                        "name": "CoinDCX User",
                        "email": "unknown",
                        "id": "unknown",
                        "status": "Auth Error"
                    }
            except Exception as e:
                print(f"[BOT] CoinDCX profile fetch failed: {e}")
                DashboardState.coindcx_profile = {
                    "name": "CoinDCX User",
                    "email": "unknown",
                    "id": "unknown",
                    "status": f"Error: {str(e)}"
                }
        else:
            DashboardState.coindcx_profile = {
                "name": "Dry Run Mode",
                "email": "demo@coindcx.com",
                "id": "DEMO12345",
                "status": "Demo Mode (Keys Missing)"
            }
        
        # Train ML Models on historical candles for each symbol
        for sym in Config.SUPPORTED_SYMBOLS:
            ltf_history = self.pipeline.ltf_candles[sym]
            if ltf_history:
                df = prepare_dataframe(ltf_history)
                trained = self.ml_models[sym].train(df)
                if not trained:
                    self.ml_models[sym] = None
        
        add_log_message("ML Models initialized (optional filtering mode).")

        DashboardState.latest_price = self.pipeline.latest_prices.get(Config.SYMBOL, 0.0)
        DashboardState.chart_history = self.pipeline.ltf_candles[Config.SYMBOL][-100:] if self.pipeline.ltf_candles[Config.SYMBOL] else []
        add_log_message(f"System ready. Multi-symbol watch active. UI viewing {Config.SYMBOL}")

    async def on_candle_close(self, symbol):
        if self._candle_locks[symbol].locked():
            if not self._pending_candle_evaluations[symbol]:
                self._pending_candle_evaluations[symbol] = True
            return

        async with self._candle_locks[symbol]:
            await self._on_candle_close_impl(symbol)
            
            while self._pending_candle_evaluations[symbol]:
                self._pending_candle_evaluations[symbol] = False
                await self._on_candle_close_impl(symbol)

    async def get_open_positions_info(self):
        count = 0
        total_risk_pct = 0.0
        longs_count = 0
        shorts_count = 0
        current_eq = self._dry_run_balance_usdt if not self.has_keys else DashboardState.balance_usdt

        for sym in Config.SUPPORTED_SYMBOLS:
            if self.in_position[sym]:
                count += 1
                if self.position_side[sym] == "LONG":
                    longs_count += 1
                elif self.position_side[sym] == "SHORT":
                    shorts_count += 1
                risk_usdt = self.position_size[sym] * abs(self.entry_price[sym] - self.stop_loss[sym])
                if current_eq > 0:
                    total_risk_pct += (risk_usdt / current_eq)
                else:
                    total_risk_pct += getattr(Config, 'RISK_PCT', 0.01)
        return count, total_risk_pct, longs_count, shorts_count

    def calculate_total_equity(self):
        current_equity = self._dry_run_balance_usdt
        for sym in Config.SUPPORTED_SYMBOLS:
            if self.in_position[sym]:
                live_price = self.pipeline.latest_prices.get(sym, self.entry_price[sym])
                if self.position_side[sym] == "LONG":
                    current_equity += self.position_size[sym] * live_price
                elif self.position_side[sym] == "SHORT":
                    unrealized_pnl = self.position_size[sym] * (self.entry_price[sym] - live_price)
                    current_equity += (self.position_size[sym] * self.entry_price[sym]) + unrealized_pnl
        return current_equity

    async def _on_candle_close_impl(self, symbol):
        print(f"[DEBUG] [{symbol}] === CANDLE CLOSE EVALUATION START ===")
        print(f"[DEBUG] [{symbol}] has_keys={self.has_keys}, PAPER_TRADING={Config.PAPER_TRADING}, is_live={self.is_live_trading()}")
        print(f"[DEBUG] [{symbol}] in_position={self.in_position[symbol]}, position_side={self.position_side[symbol]}")

        if time.time() < self.global_pause_until:
            print(f"[DEBUG] [{symbol}] BLOCKED by global_pause_until ({self.global_pause_until - time.time():.0f}s remaining)")
            return
            
        # Update balance via API if live
        if self.has_keys and not Config.PAPER_TRADING:
            balance = await self.execution.fetch_balance()
            if balance:
                if Config.COINDCX_TRADE_INR:
                    inr_balance = balance.get('total', {}).get('INR', None)
                    if inr_balance is not None:
                        DashboardState.balance_usdt = inr_balance
                else:
                    usdt_balance = balance.get('total', {}).get('USDT', None)
                    if usdt_balance and usdt_balance > 0:
                        DashboardState.balance_usdt = usdt_balance
                DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
                
        # Check drawdown circuit breakers
        current_equity = DashboardState.balance_usdt if (self.has_keys and not Config.PAPER_TRADING) else self.calculate_total_equity()
            
        if not self.risk.check_circuit_breaker(current_equity):
            add_log_message("Trading halted: Daily drawdown limit reached.")
            await self.notifier.send_message("❌ TRADING HALTED: Daily loss circuit breaker triggered.")
            return

        DashboardState.daily_drawdown_pct = self.risk.current_drawdown_pct

        ltf_df = prepare_dataframe(self.pipeline.ltf_candles[symbol])
        htf_df = prepare_dataframe(self.pipeline.htf_candles[symbol])
        
        # Check high volatility kill switch
        if not ltf_df.empty:
            last_candle = ltf_df.iloc[-1]
            move_pct = abs(last_candle['close'] - last_candle['open']) / last_candle['open']
            if move_pct > getattr(Config, 'MAX_CANDLE_MOVE_PCT', 0.015):
                avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) > 14 else 0.0
                if last_candle['volume'] < 1.5 * avg_vol:
                    self.volatility_pause_until[symbol] = len(ltf_df) + getattr(Config, 'VOLATILITY_PAUSE_CANDLES', 2)
                    add_log_message(f"[{symbol}] Trading paused: High volatility detected ({move_pct*100:.2f}% move) on LOW volume.")
                else:
                    add_log_message(f"[{symbol}] High volatility ({move_pct*100:.2f}%) on HIGH volume. Institutional move allowed.")

        if len(ltf_df) < self.volatility_pause_until.get(symbol, 0):
            return

        
        # Session and Execution Delay Filters
        import datetime
        current_hour = datetime.datetime.now(datetime.timezone.utc).hour
        is_low_volume_session = not (12 <= current_hour <= 21)
        
        if self.has_keys:
            open_time = ltf_df.iloc[-1]['timestamp'] / 1000.0 if 'timestamp' in ltf_df.columns else ltf_df.index[-1].timestamp()
            # FIX E: Parse LTF_TIMEFRAME to seconds dynamically instead of hardcoding 5 min
            _tf_seconds = {'1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400}
            close_time = open_time + _tf_seconds.get(Config.LTF_TIMEFRAME, 300)
            delay = time.time() - close_time
            if delay > 10:
                add_log_message(f"[{symbol}] Trade skipped: Execution delay ({delay:.1f}s) > 10s. Stale signal protection.")
                return
            
        signal, metadata = self.strategy.generate_signal(
            htf_df,
            ltf_df,
            relaxed=False
        )
        print(f"[DEBUG] [{symbol}] Strict signal={signal}, debug_checks={metadata.get('debug_checks')}, score={metadata.get('score')}")
        relaxed_used = False
        
        # Dual-Pass Execution
        if signal == "HOLD":
            open_count, _, _, _ = await self.get_open_positions_info()
            
            # Reset daily trades
            # FIX H: Replace deprecated utcnow() with timezone-aware call
            current_date = datetime.datetime.now(datetime.timezone.utc).date()
            if current_date != self.last_trade_day:
                self.trades_today = 0
                self.last_trade_day = current_date
                
            if open_count < 2 and (time.time() - self.global_last_trade_time) >= 20 * 60 and time.time() > self.global_pause_until:
                if self.relaxed_trades_today < 2 and time.time() > self.relaxed_disabled_until:
                    super_relaxed = False
                    if (time.time() - self.global_last_trade_time) >= 30 * 60 and metadata.get('market_regime') != 'HIGH_VOL':
                        super_relaxed = True
                        
                    signal, metadata = self.strategy.generate_signal(
                        htf_df,
                        ltf_df,
                        relaxed=True,
                        super_relaxed=super_relaxed
                    )
                    if signal != "HOLD":
                        relaxed_used = True


        if symbol == Config.SYMBOL:
            DashboardState.active_ob = metadata.get('reason', 'No OB/FVG')
            DashboardState.active_ob_level = metadata.get('active_ob_level', 0.0)
            DashboardState.active_ob_type = metadata.get('active_ob_type', 'NONE')
            DashboardState.active_bullish_ob_level = metadata.get('active_bullish_ob_level', 0.0)
            DashboardState.active_bearish_ob_level = metadata.get('active_bearish_ob_level', 0.0)
            if self.ml_models[symbol] is not None:
                DashboardState.ml_confidence = self.ml_models[symbol].predict_bias(ltf_df)
            else:
                DashboardState.ml_confidence = 0.5
            DashboardState.chart_history = self.pipeline.ltf_candles[symbol][-100:]
        
        if signal == "HOLD":
            # Log debug checks for rejection reason
            debug = metadata.get('debug_checks', {})
            reason_str = f"Trend: {debug.get('trend', 'FAIL')}, Zone: {debug.get('zone', 'FAIL')}, Trigger: {debug.get('trigger', 'FAIL')}, VWAP: {debug.get('vwap', 'FAIL')}, Vol: {debug.get('volatility', 'FAIL')}"
            print(f"[NO TRADE] [{symbol}] Reason: {metadata.get('reason')} | {reason_str}")
            return
            
        # Session Volume Block
        if is_low_volume_session:
            avg_vol = ltf_df['volume'].rolling(20).mean().iloc[-2] if len(ltf_df) > 20 else 0.0
            if ltf_df['volume'].iloc[-1] < 1.2 * avg_vol:
                add_log_message(f"[{symbol}] Trade skipped: Outside 12-22 UTC and volume not > 1.2x average.")
                return
                
        # 4H Bias logic
        htf_4h_df = self.pipeline.htf_4h_candles.get(symbol)
        if htf_4h_df is not None and len(htf_4h_df) > 50:
            import pandas as pd
            if isinstance(htf_4h_df, list): htf_4h_df = pd.DataFrame(htf_4h_df, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            ema_4h = htf_4h_df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            if signal == "BUY" and htf_4h_df['close'].iloc[-1] < ema_4h:
                metadata['score'] = metadata.get('score', 3) - 0.5
            elif signal == "SELL" and htf_4h_df['close'].iloc[-1] > ema_4h:
                metadata['score'] = metadata.get('score', 3) - 0.5
                
        add_log_message(f"[{symbol}] Raw strategy signal: {signal} ({metadata.get('reason')})")
        
        # Position Reversal logic
        if self.in_position[symbol]:
            print(f"[DEBUG] [{symbol}] Already in position ({self.position_side[symbol]}). Signal={signal}. Checking reversal.")
            if self.position_side[symbol] == "LONG" and signal == "SELL":
                add_log_message(f"[{symbol}] Trend reversal: Closing LONG position.")
                await self.exit_position(symbol, "SIGNAL_REVERSAL")
            elif self.position_side[symbol] == "SHORT" and signal == "BUY":
                add_log_message(f"[{symbol}] Trend reversal: Closing SHORT position.")
                await self.exit_position(symbol, "SIGNAL_REVERSAL")
            return

        # BTC Correlation Filter
        # FIX D: BTC correlation filter — ltf_candles stores lists of lists, not DataFrames
        if signal == "BUY" and symbol != "BTC/USDT":
            btc_candles = self.pipeline.ltf_candles.get("BTC/USDT")
            if btc_candles and len(btc_candles) > 0:
                btc_last = btc_candles[-1]  # [timestamp, open, high, low, close, volume]
                btc_drop = (btc_last[1] - btc_last[4]) / btc_last[1]  # (open - close) / open
                if btc_drop > 0.01:
                    add_log_message(f"[{symbol}] Trade blocked: BTC dropped > 1% in last 5m. Blocking altcoin longs.")
                    return
        
        # Daily Trade Limit
        if self.trades_today >= 6:
            add_log_message(f"[{symbol}] Trade skipped: Max 6 trades per day reached.")
            return

        # Cluster Loss Cooldown
        if time.time() < getattr(self, 'cluster_loss_pause_until', 0):
            add_log_message(f"[{symbol}] Trade skipped: Cluster loss cooldown active.")
            return

        # Cooldown Check
        if time.time() - self.last_trade_time.get(symbol, 0) < getattr(Config, 'COOLDOWN_MINUTES', 15) * 60:
            add_log_message(f"[{symbol}] Trade skipped due to cooldown.")
            return

        # Same Zone Check with Traded Zones Cache
        zone_id = metadata.get('zone_id')
        cache_key = f"{symbol}_{zone_id}"
        if zone_id and self.traded_zones_cache.get(cache_key):
            add_log_message(f"[{symbol}] Trade skipped: already traded in this zone ({zone_id}).")
            return
            
        # Clear out old cache (basic cleanup - ideally based on candle count but here based on simple dict size)
        if len(self.traded_zones_cache) > 1000:
            self.traded_zones_cache.clear()
            
        # FIX #2: Define entry_price BEFORE ML block uses it
        entry_price = ltf_df['close'].iloc[-1]
        add_log_message(f"[{symbol}] Entry price set: {entry_price:.4f}")

        # ML Confidence Scaler & Soft Session Filter
        prob = 1.0
        ml_confidence_weight = 0
        if self.ml_models[symbol] is not None:
            prob = self.ml_models[symbol].predict_bias(ltf_df)
            if symbol == Config.SYMBOL:
                DashboardState.ml_confidence = prob
            add_log_message(f"[{symbol}] ML confidence score: {prob:.2f}")

            # Task 7: ML TP Logic - now entry_price is defined
            risk_usdt = abs(metadata.get('stop_loss', entry_price) - entry_price)
            if prob > 0.65:
                metadata['tp2'] = entry_price + (2.5 * risk_usdt) if signal == "BUY" else entry_price - (2.5 * risk_usdt)
                ml_confidence_weight = 1
            elif prob < 0.55:
                metadata['tp2'] = entry_price + (1.5 * risk_usdt) if signal == "BUY" else entry_price - (1.5 * risk_usdt)
                ml_confidence_weight = -1
            else:
                metadata['tp2'] = entry_price + (2.0 * risk_usdt) if signal == "BUY" else entry_price - (2.0 * risk_usdt)
        if is_low_volume_session:
            avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) > 14 else 0.0
            if ltf_df['volume'].iloc[-1] < 0.6 * avg_vol:
                prob *= 0.5
                add_log_message(f"[{symbol}] Low volume session filter triggered, confidence reduced to {prob:.2f}")
                
                # Override TP2 to 1.5R instead of 2R
                risk_usdt = abs(metadata.get('stop_loss', entry_price) - entry_price)
                if signal == "BUY":
                    metadata['take_profit'] = entry_price + (1.5 * risk_usdt)
                    metadata['tp2'] = metadata['take_profit']
                elif signal == "SELL":
                    metadata['take_profit'] = entry_price - (1.5 * risk_usdt)
                    metadata['tp2'] = metadata['take_profit']
            
        # Task 5: Smart Risk Allocation (Final Edge)
        score = metadata.get('score', 3)
        if score >= 4.5: trade_risk_pct = 0.0125
        elif score >= 3.5: trade_risk_pct = 0.01
        else: trade_risk_pct = 0.0075
        
        if getattr(self, 'cluster_risk_penalty', False):
            trade_risk_pct *= 0.5
            add_log_message(f"[{symbol}] Cluster Loss Penalty: Risk slashed by 50%.")
            
        # Runner Logic Metadata
        metadata['tp1_size'] = 0.50
        metadata['tp2_size'] = 0.30
        metadata['runner_size'] = 0.20
        
        # Task 10: Equity Protection
        if not hasattr(self, 'hourly_peak_equity'):
            self.hourly_peak_equity = current_equity
            self.last_hour_ts = time.time()
        
        if time.time() - self.last_hour_ts > 3600:
            self.hourly_peak_equity = current_equity
            self.last_hour_ts = time.time()
            self.hourly_dd_penalty = False
            
        if current_equity > self.hourly_peak_equity:
            self.hourly_peak_equity = current_equity
            
        hourly_dd_pct = (self.hourly_peak_equity - current_equity) / self.hourly_peak_equity
        if hourly_dd_pct > 0.03:
            self.hourly_dd_penalty = True
        if hourly_dd_pct < 0.01:
            self.hourly_dd_penalty = False
            
        if getattr(self, 'hourly_dd_penalty', False):
            trade_risk_pct *= 0.5
            add_log_message(f"[{symbol}] Equity Protection: Hourly DD > 3%. Risk slashed by 50%.")

        open_count, total_risk, longs_count, shorts_count = await self.get_open_positions_info()
        max_risk_cap = getattr(Config, 'MAX_PORTFOLIO_RISK_PCT', 0.06)
        
        if signal == "BUY" and longs_count >= 2:
            add_log_message(f"[{symbol}] Trade skipped: Max 2 LONG positions already open.")
            return
        if signal == "SELL" and shorts_count >= 2:
            add_log_message(f"[{symbol}] Trade skipped: Max 2 SHORT positions already open.")
            return
        
        # Task 10: Priority Ranking
        priority_score = (score * 0.7) + (prob * 0.3)
        
        if priority_score < 3.5 and total_risk + trade_risk_pct > max_risk_cap - 0.04:
            add_log_message(f"[{symbol}] Trade skipped: Priority score {priority_score:.1f} < 3.5. Reserving cap space.")
            return
        if priority_score < 4.5 and total_risk + trade_risk_pct > max_risk_cap - 0.02:
            add_log_message(f"[{symbol}] Trade skipped: Priority score {priority_score:.1f} < 4.5. Reserving cap space.")
            return
        if total_risk + trade_risk_pct > max_risk_cap:
            add_log_message(f"[{symbol}] Trade blocked: Absolute exposure limit reached.")
            return
        
        
        # Liquidity & Spread Filter
        ticker = await self.execution.fetch_ticker_data(symbol)
        if not ticker:
            return
            
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        vol = ticker.get('quoteVolume', 0)
        
        if bid and ask and bid > 0 and ask > 0:
            spread = (ask - bid) / ((ask + bid) / 2)
            max_spread = 0.0015
            if spread > max_spread:
                add_log_message(f"[{symbol}] Rejected: High spread ({spread*100:.3f}%)")
                return
                
        min_vol = 30000000 if relaxed_used else getattr(Config, 'MIN_24H_VOL_USDT', 50000000)
        if vol < min_vol:
            all_tickers = await self.execution.fetch_all_tickers()
            is_top_20 = False
            if all_tickers:
                sorted_tickers = sorted([t for t in all_tickers.values() if t.get('quoteVolume')], key=lambda x: x.get('quoteVolume', 0), reverse=True)
                top_20 = [t['symbol'] for t in sorted_tickers[:20]]
                if symbol in top_20:
                    is_top_20 = True
            
            if not is_top_20:
                add_log_message(f"[{symbol}] Rejected: Low volume ({vol:,.0f} USDT) and not in top 20.")
                return
        
        # Slippage Check
        live_price = ticker.get('last', entry_price)
        if abs(live_price - entry_price) / entry_price > getattr(Config, 'MAX_SLIPPAGE_PCT', 0.002):
            add_log_message(f"[{symbol}] Trade skipped: Slippage too high. Signal: {entry_price}, Live: {live_price}")
            return
        entry_price = live_price  # Execute at live price
        self.entry_price_usdt[symbol] = entry_price  # Save reference USDT price
        
        sl = metadata['stop_loss']
        tp = metadata['take_profit']
        
        # Translate to INR if using CoinDCX INR trading
        if self.execution.coindcx_client and Config.COINDCX_TRADE_INR:
            coindcx_symbol = f"{symbol.split('/')[0]}INR"
            coindcx_ticker = await self.execution.coindcx_client.fetch_ticker_data(coindcx_symbol)
            if coindcx_ticker:
                entry_price_inr = coindcx_ticker['last']
                sl_pct = abs(entry_price - sl) / entry_price
                tp_pct = abs(entry_price - tp) / entry_price
                
                entry_price = entry_price_inr
                if signal == "BUY":
                    sl = entry_price * (1 - sl_pct)
                    tp = entry_price * (1 + tp_pct)
                else:
                    sl = entry_price * (1 + sl_pct)
                    tp = entry_price * (1 - tp_pct)
                add_log_message(f"[{symbol}] CoinDCX INR price translated: Entry {entry_price:.2f} INR | SL {sl:.2f} INR | TP {tp:.2f} INR")
            else:
                add_log_message(f"[{symbol}] WARNING: Failed to fetch CoinDCX INR price. Sizing in USDT.")
        
        pos_size = self.risk.calculate_position_size(current_equity, entry_price, sl)
        print(f"[DEBUG] [{symbol}] Raw pos_size from risk_mgr: {pos_size:.8f}")
        
        # Scale pos_size by the dynamic trade_risk_pct (default calculate_position_size uses Config.RISK_PCT)
        # FIX #2: RISK_PCT is stored as percentage (e.g. 1.0 = 1%). Convert to decimal for correct scaling.
        risk_pct_decimal = getattr(Config, 'RISK_PCT', 2.0) / 100.0
        pos_size = pos_size * (trade_risk_pct / risk_pct_decimal)
        print(f"[DEBUG] [{symbol}] Scaled pos_size: {pos_size:.8f} (trade_risk_pct={trade_risk_pct}, risk_pct_decimal={risk_pct_decimal})")
        if pos_size <= 0.0:
            print(f"[DEBUG] [{symbol}] BLOCKED: pos_size <= 0 after scaling")
            return

        # FIX A (CRITICAL-1): CoinDCX is a SPOT exchange — cannot short-sell assets you don't own
        if signal == "SELL" and self.execution.coindcx_client and Config.COINDCX_TRADE_INR:
            add_log_message(f"[{symbol}] ⚠️ SELL signal blocked: CoinDCX spot exchange does not support short selling.")
            return
            
        if signal == "BUY":
            add_log_message(f"[{symbol}] Executing BUY (LONG). Size: {pos_size:.6f} | SL: {sl:.2f} | TP: {tp:.2f}")
            order = None
            # FIX #8: Use centralized live trading check
            if self.is_live_trading():
                print(f"[DEBUG] [{symbol}] BUY → Routing to LIVE exchange API")
                order = await self.execution.place_order('buy', 'market', pos_size, price=entry_price, symbol=symbol)
            else:
                print(f"[DEBUG] [{symbol}] BUY → Dry-run mock order")
                position_cost = pos_size * entry_price
                if position_cost <= self._dry_run_balance_usdt:
                    self._dry_run_balance_usdt -= position_cost
                    order = {'id': 'MOCK_BUY_ORDER_ID', 'price': entry_price, 'status': 'filled'}

            if order:
                self.in_position[symbol] = True
                self.position_side[symbol] = "LONG"
                self.entry_price[symbol] = entry_price
                self.stop_loss[symbol] = sl
                
                # Initialize TP levels for LONG
                self.partial_tp_taken[symbol] = False
                self.tp2_taken[symbol] = False
                r_amount = abs(sl - entry_price)
                self.take_profit_1r[symbol] = entry_price + r_amount
                self.take_profit_2r[symbol] = metadata.get('tp2', entry_price + (2 * r_amount))
                self.take_profit[symbol] = entry_price + (10 * r_amount)
                
                self.highest_price_reached[symbol] = entry_price
                self.position_size[symbol] = pos_size
                self.entry_time[symbol] = int(time.time() * 1000)
                self.last_trade_time[symbol] = time.time()
                self.position_mode[symbol] = metadata.get('mode', 'STRICT')
                self.last_zone_traded[symbol] = metadata.get('zone_id')
                self.trades_today += 1
                self.global_last_trade_time = time.time()
                if metadata.get('mode') == 'RELAXED':
                    self.relaxed_trades_today += 1

                if symbol == Config.SYMBOL:
                    DashboardState.in_position = True
                    DashboardState.position_side = "LONG"
                    DashboardState.entry_price = entry_price
                    DashboardState.stop_loss = sl
                    DashboardState.take_profit = tp

                self.save_state()
                # FIX #1: Define msg_str in BUY branch before sending notification
                msg_str = (
                    f"🟢 *BUY (LONG) {symbol}*\\n"
                    f"Mode: {metadata.get('mode', 'STRICT')}\\n"
                    f"Setup Type: {metadata.get('setup_type', 'NONE')}\\n"
                    f"Entry: {entry_price:.4f}\\n"
                    f"Stop Loss: {sl:.4f}\\n"
                    f"TP1 (1R): {metadata.get('tp1', 0.0):.4f}\\n"
                    f"TP2 (2R): {metadata.get('tp2', 0.0):.4f}\\n"
                    f"Position Size: {pos_size:.6f}\\n"
                    f"Confidence: {prob:.2f}\\n"
                    f"Reason: {metadata.get('reason', 'N/A')}"
                )
                add_log_message(f"[{symbol}] " + msg_str.replace('\\n', ' | '))
                await self.notifier.send_message(msg_str)
            else:
                # FIX #4: Log order rejection with reason
                add_log_message(f"[{symbol}] ❌ BUY order REJECTED (check execution logs for reason: slippage/min-amount/liquidity)")
                await self.notifier.send_message(f"⚠️ BUY REJECTED {symbol}: Order failed to execute. Check bot logs.")
                
        elif signal == "SELL":
            msg_str = (
                f"🔴 *SELL (SHORT) {symbol}*\\n"
                f"Mode: {metadata.get('mode', 'STRICT')}\\n"
                f"Setup Type: {metadata.get('setup_type', 'NONE')}\\n"
                f"Entry: {entry_price:.4f}\\n"
                f"Stop Loss: {sl:.4f}\\n"
                f"TP1 (1R): {metadata.get('tp1', 0.0):.4f}\\n"
                f"TP2 (2R): {metadata.get('tp2', 0.0):.4f}\\n"
                f"Position Size: {pos_size:.6f}\\n"
                f"Confidence: {prob:.2f}\\n"
                f"Reason: {metadata.get('reason', 'N/A')}"
            )
            add_log_message(f"[{symbol}] " + msg_str.replace('\\n', ' | '))
            order = None
            # FIX #1: Added `not Config.PAPER_TRADING` — previously SELL orders leaked to live API in paper mode
            if self.is_live_trading():
                print(f"[DEBUG] [{symbol}] SELL → Routing to LIVE exchange API")
                order = await self.execution.place_order('sell', 'market', pos_size, price=entry_price, symbol=symbol)
            else:
                print(f"[DEBUG] [{symbol}] SELL → Dry-run mock order")
                collateral = pos_size * entry_price
                if collateral <= self._dry_run_balance_usdt:
                    self._dry_run_balance_usdt -= collateral
                    order = {'id': 'MOCK_SELL_ORDER_ID', 'price': entry_price, 'status': 'filled'}
                
            if order:
                self.in_position[symbol] = True
                self.position_side[symbol] = "SHORT"
                self.entry_price[symbol] = entry_price
                self.stop_loss[symbol] = sl
                
                # Initialize TP levels for SHORT
                self.partial_tp_taken[symbol] = False
                self.tp2_taken[symbol] = False
                r_amount = abs(sl - entry_price)
                self.take_profit_1r[symbol] = entry_price - r_amount
                self.take_profit_2r[symbol] = metadata.get('tp2', entry_price - (2 * r_amount))
                self.take_profit[symbol] = entry_price - (10 * r_amount)
                
                self.lowest_price_reached[symbol] = entry_price
                self.position_size[symbol] = pos_size
                self.entry_time[symbol] = int(time.time() * 1000)
                self.last_trade_time[symbol] = time.time()
                self.position_mode[symbol] = metadata.get('mode', 'STRICT')
                self.last_zone_traded[symbol] = metadata.get('zone_id')
                self.trades_today += 1
                self.global_last_trade_time = time.time()
                if metadata.get('mode') == 'RELAXED':
                    self.relaxed_trades_today += 1
                
                if symbol == Config.SYMBOL:
                    DashboardState.in_position = True
                    DashboardState.position_side = "SHORT"
                    DashboardState.entry_price = entry_price
                    DashboardState.stop_loss = sl
                    DashboardState.take_profit = tp

                self.save_state()
                await self.notifier.send_message(msg_str)
            else:
                add_log_message(f"[{symbol}] ❌ SELL order REJECTED (check execution logs)")
                await self.notifier.send_message(f"⚠️ SELL REJECTED {symbol}: Order failed to execute. Check logs.")

    async def run_live_risk_monitor(self):
        while True:
            try:
                if DashboardState.symbol_change_requested:
                    new_symbol = DashboardState.symbol_change_requested
                    DashboardState.symbol_change_requested = None
                    await self.change_bot_symbol(new_symbol)

                now_utc = datetime.datetime.now(datetime.timezone.utc)
                if now_utc.date() != self._last_reset_date:
                    current_eq = DashboardState.balance_usdt if self.has_keys else self.calculate_total_equity()
                    self.risk.reset_daily_equity(current_eq)
                    self._last_reset_date = now_utc.date()
                    add_log_message(f"[RISK] Daily equity checkpoint reset at UTC midnight.")

                for symbol in Config.SUPPORTED_SYMBOLS:
                    if self.in_position[symbol] and self.pipeline.latest_prices.get(symbol, 0.0) > 0:
                        curr_price = self.pipeline.latest_prices[symbol]
                        
                        ltf_df = prepare_dataframe(self.pipeline.ltf_candles[symbol])
                        curr_atr = calculate_atr(ltf_df, Config.ATR_PERIOD).iloc[-1] if not ltf_df.empty else 0.001
                        
                        # FIX C (CRITICAL-3): Fetch LIVE INR price from CoinDCX instead of using frozen scale factor
                        if self.execution.coindcx_client and Config.COINDCX_TRADE_INR and self.entry_price_usdt[symbol] > 0:
                            coindcx_sym = f"{symbol.split('/')[0]}INR"
                            coindcx_tick = await self.execution.coindcx_client.fetch_ticker_data(coindcx_sym)
                            if coindcx_tick and coindcx_tick['last'] > 0:
                                live_inr_price = coindcx_tick['last']
                                live_usdt_price = self.pipeline.latest_prices.get(symbol, 1.0)
                                inr_usdt_ratio = live_inr_price / live_usdt_price if live_usdt_price > 0 else 1.0
                                curr_price = live_inr_price
                                curr_atr = curr_atr * inr_usdt_ratio
                            else:
                                # Fallback to frozen scale if CoinDCX ticker unavailable
                                scale_factor = self.entry_price[symbol] / self.entry_price_usdt[symbol]
                                curr_price = curr_price * scale_factor
                                curr_atr = curr_atr * scale_factor
                        
                        if self.position_side[symbol] == "LONG":
                            self.highest_price_reached[symbol] = max(self.highest_price_reached[symbol], curr_price)
                            
                            # TP1 (50%)
                            if not self.partial_tp_taken[symbol] and curr_price >= self.take_profit_1r[symbol]:
                                add_log_message(f"[{symbol}] TP1 (1R) hit. Booking 50% profit.")
                                tp1_size = self.position_size[symbol] * (0.50 / 1.0)
                                if self.is_live_trading():
                                    await self.execution.place_order('sell', 'market', tp1_size, symbol=symbol, is_exit_order=True)
                                else:
                                    self._dry_run_balance_usdt += tp1_size * curr_price
                                self.position_size[symbol] -= tp1_size
                                self.partial_tp_taken[symbol] = True
                                
                                if self.entry_price[symbol] > self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = self.entry_price[symbol]
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = self.entry_price[symbol]
                                    add_log_message(f"[{symbol}] Stop Loss moved to breakeven.")
                                    
                            # TP2 (30%)
                            if self.partial_tp_taken[symbol] and not self.tp2_taken[symbol] and curr_price >= self.take_profit_2r[symbol]:
                                add_log_message(f"[{symbol}] TP2 hit. Booking 30% profit. Runner trails.")
                                tp2_size = self.position_size[symbol] * (0.30 / 0.50)
                                if self.is_live_trading():
                                    await self.execution.place_order('sell', 'market', tp2_size, symbol=symbol, is_exit_order=True)
                                else:
                                    self._dry_run_balance_usdt += tp2_size * curr_price
                                self.position_size[symbol] -= tp2_size
                                self.tp2_taken[symbol] = True

                            if self.partial_tp_taken[symbol]:
                                new_sl = self.risk.update_trailing_stop(self.entry_price[symbol], self.highest_price_reached[symbol], self.stop_loss[symbol], curr_atr, "LONG")
                                if new_sl > self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = new_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = new_sl
                                
                            if curr_price >= self.take_profit[symbol]:
                                await self.exit_position(symbol, "TAKE_PROFIT")
                            elif curr_price <= self.stop_loss[symbol]:
                                await self.exit_position(symbol, "TRAILING_STOP")
                                
                        elif self.position_side[symbol] == "SHORT":
                            self.lowest_price_reached[symbol] = min(self.lowest_price_reached[symbol], curr_price)
                            
                            # TP1 (50%)
                            if not self.partial_tp_taken[symbol] and curr_price <= self.take_profit_1r[symbol]:
                                add_log_message(f"[{symbol}] TP1 (1R) hit. Booking 50% profit.")
                                tp1_size = self.position_size[symbol] * (0.50 / 1.0)
                                if self.is_live_trading():
                                    await self.execution.place_order('buy', 'market', tp1_size, symbol=symbol, is_exit_order=True)
                                else:
                                    self._dry_run_balance_usdt += tp1_size * (self.entry_price[symbol] - curr_price) + (tp1_size * self.entry_price[symbol])
                                self.position_size[symbol] -= tp1_size
                                self.partial_tp_taken[symbol] = True
                                
                                if self.entry_price[symbol] < self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = self.entry_price[symbol]
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = self.entry_price[symbol]
                                    add_log_message(f"[{symbol}] Stop Loss moved to breakeven.")
                                    
                            # TP2 (30%)
                            if self.partial_tp_taken[symbol] and not self.tp2_taken[symbol] and curr_price <= self.take_profit_2r[symbol]:
                                add_log_message(f"[{symbol}] TP2 hit. Booking 30% profit. Runner trails.")
                                tp2_size = self.position_size[symbol] * (0.30 / 0.50)
                                if self.is_live_trading():
                                    await self.execution.place_order('buy', 'market', tp2_size, symbol=symbol, is_exit_order=True)
                                else:
                                    self._dry_run_balance_usdt += tp2_size * (self.entry_price[symbol] - curr_price) + (tp2_size * self.entry_price[symbol])
                                self.position_size[symbol] -= tp2_size
                                self.tp2_taken[symbol] = True

                            if self.partial_tp_taken[symbol]:
                                new_sl = self.risk.update_trailing_stop(self.entry_price[symbol], self.lowest_price_reached[symbol], self.stop_loss[symbol], curr_atr, "SHORT")
                                if new_sl < self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = new_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = new_sl
                                
                            if curr_price <= self.take_profit[symbol]:
                                await self.exit_position(symbol, "TAKE_PROFIT")
                            elif curr_price >= self.stop_loss[symbol]:
                                await self.exit_position(symbol, "TRAILING_STOP")

                # Update UI for selected Config.SYMBOL
                sym = Config.SYMBOL
                latest_price = self.pipeline.latest_prices.get(sym, 0.0)
                
                # Fetch live INR price from CoinDCX if in real INR trading mode
                if self.execution.coindcx_client and Config.COINDCX_TRADE_INR and not Config.PAPER_TRADING:
                    coindcx_symbol = f"{sym.split('/')[0]}INR"
                    coindcx_ticker = await self.execution.coindcx_client.fetch_ticker_data(coindcx_symbol)
                    if coindcx_ticker:
                        latest_price = coindcx_ticker['last']
                        
                DashboardState.latest_price = latest_price
                if not self.has_keys:
                    DashboardState.balance_usdt = self.calculate_total_equity()
                    
                if self.pipeline.ltf_candles[sym]:
                    DashboardState.chart_history = self.pipeline.ltf_candles[sym][-100:]
                
                if self.in_position[sym] and self.pipeline.latest_prices.get(sym, 0.0) > 0:
                    curr_price = self.pipeline.latest_prices[sym]
                    if self.position_side[sym] == "LONG":
                        pnl_pct = (curr_price - self.entry_price[sym]) / self.entry_price[sym] * 100.0
                        pnl_usdt = self.position_size[sym] * (curr_price - self.entry_price[sym])
                    else:
                        pnl_pct = (self.entry_price[sym] - curr_price) / self.entry_price[sym] * 100.0
                        pnl_usdt = self.position_size[sym] * (self.entry_price[sym] - curr_price)
                    DashboardState.current_pnl_pct = pnl_pct
                    DashboardState.current_pnl_usdt = pnl_usdt
                else:
                    DashboardState.current_pnl_pct = 0.0
                    DashboardState.current_pnl_usdt = 0.0

                # Update CoinDCX balances
                if self.execution.coindcx_client:
                    try:
                        balance = await self.execution.fetch_balance()
                        if balance:
                            updated_bals = []
                            total_bals = balance.get('total', {})
                            free_bals = balance.get('free', {})
                            used_bals = balance.get('used', {})
                            for curr, total in total_bals.items():
                                if total > 0.0:
                                    updated_bals.append({
                                        "currency": curr,
                                        "balance": total,
                                        "available": free_bals.get(curr, total),
                                        "locked": used_bals.get(curr, 0.0)
                                    })
                            # Sort to put INR and USDT first, then others by balance descending
                            def sort_key(x):
                                c = x['currency']
                                if c == 'INR': return (0, -x['balance'])
                                if c == 'USDT': return (1, -x['balance'])
                                return (2, -x['balance'])
                            updated_bals.sort(key=sort_key)
                            DashboardState.coindcx_balances = updated_bals
                    except Exception as e:
                        print(f"[RISK MONITOR] Error fetching CoinDCX balances: {e}")
                else:
                    # In dry run mode, put some mock balances
                    DashboardState.coindcx_balances = [
                        {"currency": "INR", "balance": 50000.0, "available": 50000.0, "locked": 0.0},
                        {"currency": "USDT", "balance": 1000.0, "available": 1000.0, "locked": 0.0},
                        {"currency": "BTC", "balance": 0.05, "available": 0.05, "locked": 0.0},
                        {"currency": "ETH", "balance": 0.5, "available": 0.5, "locked": 0.0}
                    ]

            except Exception as e:
                import traceback
                print(f"[RISK MONITOR] Error: {e}")
                traceback.print_exc()
            await asyncio.sleep(5.0)

    async def exit_position(self, symbol, reason):
        # FIX #5: Better fallback for exit_price to avoid 0.0 values
        exit_price = self.pipeline.latest_prices.get(symbol) or self.entry_price[symbol]
        if exit_price <= 0 or not exit_price:
            exit_price = self.entry_price[symbol]
        add_log_message(f"[{symbol}] Exiting at price: {exit_price:.4f} (reason: {reason})")
        order = None
        if self.is_live_trading():
            side = 'buy' if self.position_side[symbol] == 'SHORT' else 'sell'
            order = await self.execution.place_order(side, 'market', self.position_size[symbol], price=exit_price, is_exit_order=True, symbol=symbol)
        else:
            order = {'id': 'MOCK_EXIT_ORDER_ID', 'price': exit_price, 'status': 'filled'}
            
        if order:
            if self.position_side[symbol] == "LONG":
                pnl_pct = (exit_price - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0
                pnl_usdt = self.position_size[symbol] * (exit_price - self.entry_price[symbol])
                if not self.has_keys or Config.PAPER_TRADING:
                    self._dry_run_balance_usdt += self.position_size[symbol] * exit_price
            else:
                pnl_pct = (self.entry_price[symbol] - exit_price) / self.entry_price[symbol] * 100.0
                pnl_usdt = self.position_size[symbol] * (self.entry_price[symbol] - exit_price)
                if not self.has_keys or Config.PAPER_TRADING:
                    self._dry_run_balance_usdt += (self.position_size[symbol] * self.entry_price[symbol]) + pnl_usdt
                
            trade_record = {
                'symbol': symbol,
                'side': self.position_side[symbol],
                'entry_price': self.entry_price[symbol],
                'exit_price': exit_price,
                'pnl_usdt': pnl_usdt,
                'pnl_pct': pnl_pct,
                'entry_time': self.entry_time[symbol],
                'exit_time': int(time.time() * 1000)
            }
            DashboardState.trades.append(trade_record)
            if len(DashboardState.trades) > 500:
                DashboardState.trades = DashboardState.trades[-500:]

            # Task 5: Cluster Loss Tracking
            is_loss = pnl_usdt < 0
            self.trade_history.append(is_loss)
            if len(self.trade_history) > 6:
                self.trade_history.pop(0)
                
            if len(self.trade_history) >= 2 and all(self.trade_history[-2:]):
                cooldown_time = time.time() + (2 * 3600)
                self.cluster_loss_pause_until = cooldown_time
                self.global_pause_until = cooldown_time  # Update global pause
                add_log_message("🚨 [SAFETY] 2 consecutive losses. Trading paused globally for 2 hours.")
                self.trade_history.clear()
            elif len(self.trade_history) >= 6 and sum(self.trade_history) >= 3:
                self.cluster_risk_penalty = True
                add_log_message("🚨 [SAFETY] 3 losses in last 6 trades. Global risk slashed by 50%.")
            else:
                self.cluster_risk_penalty = False

            # Update relaxed cooldowns
            position_mode = self.position_mode.get(symbol, 'STRICT')
            if position_mode == 'RELAXED' and is_loss:
                self.relaxed_losses += 1
                if self.relaxed_losses >= 2:
                    self.relaxed_disabled_until = time.time() + 7200
                    add_log_message("🚨 [SAFETY] 2 relaxed losses. Relaxed mode disabled for 2 hours.")
                    self.relaxed_losses = 0
            elif not is_loss and position_mode == 'RELAXED':
                self.relaxed_losses = 0

            self.in_position[symbol] = False
            self.position_side[symbol] = "HOLD"
            self.position_size[symbol] = 0.0
            
            if symbol == Config.SYMBOL:
                DashboardState.in_position = False
                DashboardState.position_side = "HOLD"

            self.save_state()
            await self.notifier.send_message(
                f"🚨 *{symbol} LIQUIDATED ({reason})*\nExit Price: {exit_price:.2f}\nPnL: {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT)"
            )

    async def change_bot_symbol(self, new_symbol):
        if new_symbol not in Config.SUPPORTED_SYMBOLS:
            add_log_message(f"Symbol {new_symbol} is not tracked by the background pipeline.")
            return

        add_log_message(f"Dashboard view switched to {new_symbol}.")
        Config.SYMBOL = new_symbol
        
        DashboardState.in_position = self.in_position[new_symbol]
        DashboardState.position_side = self.position_side[new_symbol]
        DashboardState.entry_price = self.entry_price[new_symbol]
        DashboardState.stop_loss = self.stop_loss[new_symbol]
        DashboardState.take_profit = self.take_profit[new_symbol]
        DashboardState.latest_price = self.pipeline.latest_prices.get(new_symbol, 0.0)
        DashboardState.chart_history = self.pipeline.ltf_candles[new_symbol][-100:] if self.pipeline.ltf_candles[new_symbol] else []
        
        if self.ml_models[new_symbol] is not None and self.pipeline.ltf_candles[new_symbol]:
            df = prepare_dataframe(self.pipeline.ltf_candles[new_symbol])
            DashboardState.ml_confidence = self.ml_models[new_symbol].predict_bias(df)
        else:
            DashboardState.ml_confidence = 0.5

    async def shutdown(self):
        add_log_message("Shutting down exchange sessions gracefully...")
        await self.execution.close()
        self.pipeline.stop()

async def start_all():
    import dashboard.app as dashboard_module
    bot = PrimeSignalBot()
    dashboard_module.bot_instance = bot

    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    if sys.platform == 'win32' and sys.version_info < (3, 12):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(start_all())
    except KeyboardInterrupt:
        print("\nStopping bot...")

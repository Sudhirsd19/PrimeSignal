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
from ml.adversarial_debate import AdversarialDebateCourtroom
from core.lead_lag_arbitrage import LeadLagArbitrageEngine
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
        self.lead_lag = LeadLagArbitrageEngine()
        self.courtroom = AdversarialDebateCourtroom()
        
        self.ml_models = {sym: MLSignalConfirmator() for sym in Config.SUPPORTED_SYMBOLS}
        
        # Internal State tracking (Per Symbol)
        self.in_position = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self.position_side = {sym: "HOLD" for sym in Config.SUPPORTED_SYMBOLS}
        self.entry_price = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
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
        """Persist current position state to disk and Firebase Cloud DB for crash recovery."""
        state = {
            'in_position': self.in_position,
            'position_side': self.position_side,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'entry_time': self.entry_time,
            'highest_price_reached': self.highest_price_reached,
            'lowest_price_reached': self.lowest_price_reached,
            '_dry_run_balance_usdt': self._dry_run_balance_usdt,
            'take_profit_1r': self.take_profit_1r,
            'take_profit_2r': self.take_profit_2r,
            'partial_tp_taken': self.partial_tp_taken,
            'tp2_taken': self.tp2_taken,
            'position_mode': self.position_mode,
            'last_trade_time': self.last_trade_time,
            'last_zone_traded': self.last_zone_traded,
            'closed_trades': list(DashboardState.trades[-100:]),
        }
        try:
            self._STATE_FILE.write_text(json.dumps(state))
        except Exception as e:
            print(f"[STATE] Failed to save state to local file: {e}")
            
        try:
            from core.firebase_manager import FirebaseManager
            firebase = FirebaseManager()
            if firebase.is_connected:
                firebase.db.collection("bot_state").document("current").set(state)
        except Exception as e:
            print(f"[STATE] Firebase state save note: {e}")

    def load_state(self):
        """Restore position state from Firebase Cloud DB or local disk after a restart."""
        state = None
        
        # 1. Try Firebase Cloud DB first (survives Render container restarts)
        try:
            from core.firebase_manager import FirebaseManager
            firebase = FirebaseManager()
            if firebase.is_connected:
                doc = firebase.db.collection("bot_state").document("current").get()
                if doc.exists:
                    state = doc.to_dict()
                    add_log_message("[STATE] Recovered position state from Firebase Cloud DB.")
        except Exception as e:
            print(f"[STATE] Firebase load check: {e}")

        # 2. Fallback to local file if not restored from Cloud DB
        if not state and self._STATE_FILE.exists():
            try:
                state = json.loads(self._STATE_FILE.read_text())
            except Exception as e:
                print(f"[STATE] Failed to load local state: {e}")
                
        if not state:
            # Check if historical trades exist in data/trade_logs.jsonl
            self._load_trade_logs()
            return

        try:
            # Helper to safely load dict state, falling back to default if new symbols were added
            def safe_load(key, default_val):
                loaded_dict = state.get(key, {})
                return {sym: loaded_dict.get(sym, default_val) for sym in Config.SUPPORTED_SYMBOLS}

            self.in_position = safe_load('in_position', False)
            self.position_side = safe_load('position_side', 'HOLD')
            self.entry_price = safe_load('entry_price', 0.0)
            self.stop_loss = safe_load('stop_loss', 0.0)
            self.take_profit = safe_load('take_profit', 0.0)
            self.position_size = safe_load('position_size', 0.0)
            self.entry_time = safe_load('entry_time', 0)
            self.highest_price_reached = safe_load('highest_price_reached', 0.0)
            self.lowest_price_reached = safe_load('lowest_price_reached', 999999.0)
            self._dry_run_balance_usdt = state.get('_dry_run_balance_usdt', 10000.0)
            self.take_profit_1r = safe_load('take_profit_1r', 0.0)
            self.take_profit_2r = safe_load('take_profit_2r', 0.0)
            self.partial_tp_taken = safe_load('partial_tp_taken', False)
            self.tp2_taken = safe_load('tp2_taken', False)
            self.position_mode = safe_load('position_mode', 'STRICT')
            self.last_trade_time = safe_load('last_trade_time', 0)
            self.last_zone_traded = safe_load('last_zone_traded', None)
            
            # Restore closed trades history
            saved_trades = state.get('closed_trades', [])
            if saved_trades:
                DashboardState.trades = list(saved_trades)
            
            # Sync any additional logs from data/trade_logs.jsonl
            self._load_trade_logs()

            # Sync to dashboard for active UI symbol
            sym = Config.SYMBOL
            DashboardState.in_position = self.in_position[sym]
            DashboardState.position_side = self.position_side[sym]
            DashboardState.entry_price = self.entry_price[sym]
            DashboardState.stop_loss = self.stop_loss[sym]
            DashboardState.take_profit = self.take_profit[sym]
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            
            # Immediately populate active_positions map for UI refresh recovery
            active_pos_map = {}
            for s in Config.SUPPORTED_SYMBOLS:
                if self.in_position[s]:
                    active_pos_map[s] = {
                        'side': self.position_side[s],
                        'entry_price': self.entry_price[s],
                        'stop_loss': self.stop_loss[s],
                        'take_profit': self.take_profit[s],
                        'position_size': self.position_size[s],
                        'current_pnl_usdt': 0.0,
                        'current_pnl_pct': 0.0
                    }
            DashboardState.active_positions = active_pos_map

            open_positions = sum(1 for s in Config.SUPPORTED_SYMBOLS if self.in_position[s])
            if open_positions > 0:
                add_log_message(f"[STATE] Recovered {open_positions} open positions into Dashboard state.")
            else:
                add_log_message("[STATE] State loaded — no open positions to recover.")
        except Exception as e:
            print(f"[STATE] Failed to process state: {e}")

    def _load_trade_logs(self):
        """Read historical closed trades from data/trade_logs.jsonl into DashboardState.trades."""
        try:
            log_file = Path("data") / "trade_logs.jsonl"
            if log_file.exists():
                existing_keys = {
                    f"{t.get('symbol')}_{t.get('exit_time') or t.get('time') or 0}_{round(float(t.get('pnl_usdt', 0) or 0), 4)}"
                    for t in DashboardState.trades
                }
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                tr = json.loads(line)
                                k = f"{tr.get('symbol')}_{tr.get('exit_time') or tr.get('time') or 0}_{round(float(tr.get('pnl_usdt', 0) or 0), 4)}"
                                if k not in existing_keys:
                                    DashboardState.trades.append(tr)
                                    existing_keys.add(k)
                            except Exception:
                                pass
                if len(DashboardState.trades) > 500:
                    DashboardState.trades = DashboardState.trades[-500:]
        except Exception as e:
            print(f"[STATE] Failed to read data/trade_logs.jsonl: {e}")

    def reset_account_state(self, target_balance: float = 10000.0):
        """Resets virtual paper balance to target amount and closes all simulated positions for fresh trading."""
        self._dry_run_balance_usdt = float(target_balance)
        for sym in Config.SUPPORTED_SYMBOLS:
            self.in_position[sym] = False
            self.position_side[sym] = "HOLD"
            self.entry_price[sym] = 0.0
            self.stop_loss[sym] = 0.0
            self.take_profit[sym] = 0.0
            self.take_profit_1r[sym] = 0.0
            self.take_profit_2r[sym] = 0.0
            self.position_size[sym] = 0.0
            self.entry_time[sym] = 0
            self.highest_price_reached[sym] = 0.0
            self.lowest_price_reached[sym] = 999999.0
            self.partial_tp_taken[sym] = False
            self.tp2_taken[sym] = False

        self.traded_zones_cache.clear()
        DashboardState.balance_usdt = float(target_balance)
        DashboardState.balance_base = 0.0
        DashboardState.active_positions.clear()
        DashboardState.in_position = False
        DashboardState.position_side = "HOLD"
        DashboardState.entry_price = 0.0
        DashboardState.stop_loss = 0.0
        DashboardState.take_profit = 0.0
        DashboardState.current_pnl_pct = 0.0
        DashboardState.current_pnl_usdt = 0.0
        self.save_state()
        add_log_message(f"🔄 [ACCOUNT RESET] Virtual paper balance reset to ${target_balance:,.2f} USDT. All positions cleared for fresh trading.")

    async def initialize(self):
        add_log_message("Starting system initialization for all supported symbols...")

        self.load_state()
        await self.pipeline.start()
        await asyncio.sleep(3)
        
        # Initial Balance load
        if self.has_keys:
            balance = await self.execution.fetch_balance()
            if balance:
                usdt_balance = balance.get('total', {}).get('USDT', None)
                if usdt_balance and usdt_balance > 0:
                    DashboardState.balance_usdt = usdt_balance
                else:
                    add_log_message(f"[WARNING] Balance fetch returned {usdt_balance}. Check account type. Keeping last known value.")
                DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
        else:
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            DashboardState.balance_base = 0.0
        
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
        current_eq = self.calculate_total_equity() if not self.has_keys else DashboardState.balance_usdt

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
            if self.in_position[sym] and self.position_size[sym] and self.position_size[sym] > 1e-7:
                live_price = self.pipeline.latest_prices.get(sym, self.entry_price[sym])
                if self.position_side[sym] == "LONG":
                    current_equity += self.position_size[sym] * live_price
                elif self.position_side[sym] == "SHORT":
                    unrealized_pnl = self.position_size[sym] * (self.entry_price[sym] - live_price)
                    current_equity += (self.position_size[sym] * self.entry_price[sym]) + unrealized_pnl
        return current_equity

    def is_macro_news_blackout(self):
        """
        Checks if current UTC time falls in high-impact economic news windows
        (US CPI / PPI / NFP / FOMC release times e.g. 12:20-12:45 UTC, 13:20-13:45 UTC, 18:00-19:30 UTC).
        """
        if not getattr(Config, 'ENABLE_MACRO_NEWS_FILTER', True):
            return False, ""
            
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        hour = now_utc.hour
        minute = now_utc.minute
        weekday = now_utc.weekday() # 0=Mon, 4=Fri
        
        # Only weekdays have high-impact macro economic data releases
        if weekday >= 5:
            return False, ""
            
        # Window 1: 12:20 - 12:45 UTC (US 8:30 AM EDT daylight savings / 8:30 AM EST releases)
        if hour == 12 and (20 <= minute <= 45):
            return True, "US Morning Macro Data Window (12:20-12:45 UTC)"
            
        # Window 2: 13:20 - 13:45 UTC (US 8:30 AM EST standard winter releases)
        if hour == 13 and (20 <= minute <= 45):
            return True, "US Main Macro Data Window (13:20-13:45 UTC)"
            
        # Window 3: 18:00 - 19:30 UTC (Fed FOMC Rate Decision & Press Conf, typically Wednesdays)
        if (weekday == 2) and (18 <= hour <= 19):
            return True, "Fed FOMC Rate Decision Window (18:00-19:30 UTC)"
            
        return False, ""

    async def _on_candle_close_impl(self, symbol):
        if time.time() < self.global_pause_until:
            return
            
        # Macro Economic News Blackout Window Filter (CPI/FOMC Volatility Protection)
        is_news, news_reason = self.is_macro_news_blackout()
        if is_news:
            add_log_message(f"[{symbol}] Trade skipped: {news_reason}. Macro volatility blackout active.")
            return

        # Update balance via API if live
        if self.has_keys:
            balance = await self.execution.fetch_balance()
            if balance:
                usdt_balance = balance.get('total', {}).get('USDT', None)
                if usdt_balance and usdt_balance > 0:
                    DashboardState.balance_usdt = usdt_balance
                DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
                
        # Check drawdown circuit breakers
        current_equity = DashboardState.balance_usdt if self.has_keys else self.calculate_total_equity()
            
        if not self.risk.check_circuit_breaker(current_equity):
            DashboardState.signal_light = "RED"
            DashboardState.signal_light_reason = f"🚨 SLEEP MODE ACTIVE: Daily loss limit hit ({self.risk.current_drawdown_pct:.2f}%). Trading suspended until 00:00 UTC."
            add_log_message(f"🚨 SLEEP MODE / CIRCUIT BREAKER TRIGGERED: Daily loss limit reached ({self.risk.current_drawdown_pct:.2f}%). All entries suspended.")
            await self.notifier.send_message(f"🚨 *SLEEP MODE ACTIVATED*\\nDaily loss circuit breaker triggered ({self.risk.current_drawdown_pct:.2f}%). All new trades suspended until 00:00 UTC.")
            return

        DashboardState.daily_drawdown_pct = self.risk.current_drawdown_pct

        ltf_df = prepare_dataframe(self.pipeline.ltf_candles[symbol])
        htf_df = prepare_dataframe(self.pipeline.htf_candles[symbol])
        
        # Check high volatility kill switch
        if not ltf_df.empty:
            last_candle = ltf_df.iloc[-1]
            move_pct = abs(last_candle['close'] - last_candle['open']) / max(last_candle['open'], 1e-8)
            if move_pct > getattr(Config, 'MAX_CANDLE_MOVE_PCT', 0.015):
                avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) > 14 else 0.0
                if last_candle['volume'] < 1.5 * avg_vol:
                    pause_candles = getattr(Config, 'VOLATILITY_PAUSE_CANDLES', 2)
                    tf_minutes = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
                    self.volatility_pause_until[symbol] = time.time() + (pause_candles * tf_minutes * 60)
                    add_log_message(f"[{symbol}] Trading paused: High volatility detected ({move_pct*100:.2f}% move) on LOW volume.")
                else:
                    add_log_message(f"[{symbol}] High volatility ({move_pct*100:.2f}%) on HIGH volume. Institutional move allowed.")

        if time.time() < self.volatility_pause_until.get(symbol, 0):
            return

        
        # Session and Execution Delay Filters
        import datetime
        current_hour = datetime.datetime.now(datetime.timezone.utc).hour
        is_low_volume_session = not (12 <= current_hour <= 21)
        
        if self.has_keys:
            open_time = ltf_df.iloc[-1]['time'] / 1000.0 if 'time' in ltf_df.columns else ltf_df.index[-1].timestamp()
            tf_mins = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
            close_time = open_time + (tf_mins * 60)
            delay = time.time() - close_time
            if delay > 10:
                add_log_message(f"[{symbol}] Trade skipped: Execution delay ({delay:.1f}s) > 10s. Stale signal protection.")
                return
            
        signal, metadata = self.strategy.generate_signal(
            htf_df,
            ltf_df,
            relaxed=False
        )
        relaxed_used = False
        
        # Dual-Pass Execution
        if signal == "HOLD":
            open_count, _, _, _ = await self.get_open_positions_info()
            
            # Reset daily trades
            current_date = datetime.datetime.now(datetime.timezone.utc).date()
            if current_date != self.last_trade_day:
                self.trades_today = 0
                self.last_trade_day = current_date
                
            if open_count < 2 and (time.time() - self.global_last_trade_time) >= 20 * 60 and time.time() > self.global_pause_until:
                if self.relaxed_trades_today < 2 and time.time() > self.relaxed_disabled_until:
                    signal, metadata = self.strategy.generate_signal(
                        htf_df,
                        ltf_df,
                        relaxed=True
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
                nc_pred = self.ml_models[symbol].predict_next_candle(ltf_df)
                DashboardState.next_candle_color = nc_pred['color']
                DashboardState.next_candle_prob = nc_pred['confidence_pct']
            else:
                DashboardState.ml_confidence = 0.5
                DashboardState.next_candle_color = "GREEN"
                DashboardState.next_candle_prob = 50.0
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
            vol_mult = 1.0 if metadata.get('score', 0) >= 3.5 else 1.2
            if ltf_df['volume'].iloc[-1] < vol_mult * avg_vol:
                add_log_message(f"[{symbol}] Trade skipped: Outside 12-22 UTC and volume not > {vol_mult}x average.")
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
            if self.position_side[symbol] == "LONG" and signal == "SELL":
                add_log_message(f"[{symbol}] Trend reversal: Closing LONG position.")
                await self.exit_position(symbol, "SIGNAL_REVERSAL")
            elif self.position_side[symbol] == "SHORT" and signal == "BUY":
                add_log_message(f"[{symbol}] Trend reversal: Closing SHORT position.")
                await self.exit_position(symbol, "SIGNAL_REVERSAL")
            return

        # BTC Correlation Filter
        if signal == "BUY" and symbol != "BTC/USDT":
            btc_raw = self.pipeline.ltf_candles.get("BTC/USDT")
            if btc_raw:
                btc_df = prepare_dataframe(btc_raw) if isinstance(btc_raw, list) else btc_raw
                if not btc_df.empty:
                    btc_last = btc_df.iloc[-1]
                    btc_drop = (btc_last['open'] - btc_last['close']) / btc_last['open']
                    if btc_drop > 0.014:
                        add_log_message(f"[{symbol}] Trade blocked: BTC dropped > 1.4% in last 5m. Blocking altcoin longs.")
                        return

        # Funding Rate & Crowded Trade Sentiment Filter
        if getattr(Config, 'ENABLE_FUNDING_RATE_FILTER', True):
            try:
                fr = await self.execution.fetch_funding_rate(symbol)
                max_fr = getattr(Config, 'MAX_FUNDING_RATE_PCT', 0.035) / 100.0 # e.g. 0.035% = 0.00035
                if signal == "BUY" and fr > max_fr:
                    add_log_message(f"[{symbol}] Trade blocked: Extreme Bullish Funding Rate ({fr*100:+.4f}% > +{max_fr*100:.3f}%). Long liquidation trap protection.")
                    return
                elif signal == "SELL" and fr < -max_fr:
                    add_log_message(f"[{symbol}] Trade blocked: Extreme Bearish Funding Rate ({fr*100:+.4f}% < -{max_fr*100:.3f}%). Short squeeze trap protection.")
                    return
            except Exception:
                pass
        
        # Daily Trade Limit
        max_daily_trades = getattr(Config, 'MAX_DAILY_TRADES', 6)
        if self.trades_today >= max_daily_trades:
            add_log_message(f"[{symbol}] Trade skipped: Max {max_daily_trades} trades per day reached ({self.trades_today}/{max_daily_trades}).")
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
        
        sl = metadata['stop_loss']
        tp = metadata['take_profit']
        
        pos_size = self.risk.calculate_position_size(current_equity, entry_price, sl)
        
        # Scale pos_size by the dynamic trade_risk_pct (default calculate_position_size uses Config.RISK_PCT)
        # So we adjust it relative to default RISK_PCT
        pos_size = pos_size * (trade_risk_pct / (getattr(Config, 'RISK_PCT', 0.8) / 100.0))
        if pos_size <= 0.0:
            return

        # ── Next-Gen Proprietary Edge: Dual-Brain Adversarial AI Debate Courtroom ──
        market_context = {
            'cvd': metadata.get('cvd', {}),
            'liquidation': metadata.get('liquidation', {}),
            'ml_confidence': prob,
            'funding_rate': fr if 'fr' in locals() else 0.0,
            'bb_squeeze': False,
            'spread_pct': spread if 'spread' in locals() else 0.0005
        }
        debate_result = self.courtroom.conduct_debate(signal, metadata, market_context)
        if debate_result.get('verdict') != 'APPROVED':
            add_log_message(f"[{symbol}] ⚖️ Trade REJECTED by Dual-Brain AI Courtroom ({debate_result['conviction_pct']}% conviction): {', '.join(debate_result['prosecutor_objections'])}")
            return
        else:
            add_log_message(f"[{symbol}] ⚖️ Trade APPROVED by Dual-Brain AI Courtroom ({debate_result['conviction_pct']}% conviction)!")
            
        if signal == "BUY":
            add_log_message(f"[{symbol}] Executing BUY (LONG). Size: {pos_size:.6f} | SL: {sl:.2f} | TP: {tp:.2f}")
            order = None
            if self.has_keys:
                order = await self.execution.place_order('buy', 'market', pos_size, price=entry_price, symbol=symbol)
            else:
                position_cost = pos_size * entry_price
                if position_cost <= self._dry_run_balance_usdt:
                    self._dry_run_balance_usdt -= position_cost
                    order = {'id': 'MOCK_BUY_ORDER_ID', 'price': entry_price, 'status': 'filled'}

            if order:
                self.in_position[symbol] = True
                self.position_side[symbol] = "LONG"
                self.entry_price[symbol] = entry_price
                self.stop_loss[symbol] = sl
                
                # Initialize TP levels for LONG (3-Stage: TP1 50%, TP2 30%, Runner 20%)
                self.partial_tp_taken[symbol] = False
                self.tp2_taken[symbol] = False
                r_amount = abs(sl - entry_price)
                self.take_profit_1r[symbol] = metadata.get('tp1', entry_price + (1.2 * r_amount))
                self.take_profit_2r[symbol] = metadata.get('tp2', entry_price + (2.2 * r_amount))
                # Extended Runner Target at 3.5R (so runner target stays above SL & TP2)
                self.take_profit[symbol] = metadata.get('tp3', entry_price + (3.5 * r_amount))
                
                self.highest_price_reached[symbol] = entry_price
                self.position_size[symbol] = pos_size
                self.entry_time[symbol] = int(time.time() * 1000)
                self.last_trade_time[symbol] = time.time()
                self.position_mode[symbol] = metadata.get('mode', 'STRICT')
                zone_id = metadata.get('zone_id')
                self.last_zone_traded[symbol] = zone_id
                if zone_id:
                    self.traded_zones_cache[f"{symbol}_{zone_id}"] = True
                self.trades_today += 1
                self.global_last_trade_time = time.time()
                if metadata.get('mode') == 'RELAXED':
                    self.relaxed_trades_today += 1

                if symbol == Config.SYMBOL:
                    DashboardState.in_position = True
                    DashboardState.position_side = "LONG"
                    DashboardState.entry_price = entry_price
                    DashboardState.stop_loss = sl
                    DashboardState.take_profit = self.take_profit[symbol]

                self.save_state()
                # Telegram Notification with 3-Stage Target Levels
                msg_str = (
                    f"🟢 *BUY (LONG) {symbol}*\\n"
                    f"Mode: {metadata.get('mode', 'STRICT')}\\n"
                    f"Setup Type: {metadata.get('setup_type', 'NONE')}\\n"
                    f"Entry: {entry_price:.4f}\\n"
                    f"Stop Loss: {sl:.4f}\\n"
                    f"TP1 (1.2R - 50%): {self.take_profit_1r[symbol]:.4f}\\n"
                    f"TP2 (2.2R - 30%): {self.take_profit_2r[symbol]:.4f}\\n"
                    f"Runner (3.5R - 20%): {self.take_profit[symbol]:.4f}\\n"
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
            order = None
            if self.has_keys:
                order = await self.execution.place_order('sell', 'market', pos_size, price=entry_price, symbol=symbol)
            else:
                collateral = pos_size * entry_price
                if collateral <= self._dry_run_balance_usdt:
                    self._dry_run_balance_usdt -= collateral
                    order = {'id': 'MOCK_SELL_ORDER_ID', 'price': entry_price, 'status': 'filled'}
                
            if order:
                self.in_position[symbol] = True
                self.position_side[symbol] = "SHORT"
                self.entry_price[symbol] = entry_price
                self.stop_loss[symbol] = sl
                
                # Initialize TP levels for SHORT (3-Stage: TP1 50%, TP2 30%, Runner 20%)
                self.partial_tp_taken[symbol] = False
                self.tp2_taken[symbol] = False
                r_amount = abs(sl - entry_price)
                self.take_profit_1r[symbol] = metadata.get('tp1', entry_price - (1.2 * r_amount))
                self.take_profit_2r[symbol] = metadata.get('tp2', entry_price - (2.2 * r_amount))
                # Extended Runner Target at 3.5R (so runner target stays below SL & TP2)
                self.take_profit[symbol] = metadata.get('tp3', entry_price - (3.5 * r_amount))
                
                self.lowest_price_reached[symbol] = entry_price
                self.position_size[symbol] = pos_size
                self.entry_time[symbol] = int(time.time() * 1000)
                self.last_trade_time[symbol] = time.time()
                self.position_mode[symbol] = metadata.get('mode', 'STRICT')
                zone_id = metadata.get('zone_id')
                self.last_zone_traded[symbol] = zone_id
                if zone_id:
                    self.traded_zones_cache[f"{symbol}_{zone_id}"] = True
                self.trades_today += 1
                self.global_last_trade_time = time.time()
                if metadata.get('mode') == 'RELAXED':
                    self.relaxed_trades_today += 1
                
                if symbol == Config.SYMBOL:
                    DashboardState.in_position = True
                    DashboardState.position_side = "SHORT"
                    DashboardState.entry_price = entry_price
                    DashboardState.stop_loss = sl
                    DashboardState.take_profit = self.take_profit[symbol]

                self.save_state()
                msg_str = (
                    f"🔴 *SELL (SHORT) {symbol}*\\n"
                    f"Mode: {metadata.get('mode', 'STRICT')}\\n"
                    f"Setup Type: {metadata.get('setup_type', 'NONE')}\\n"
                    f"Entry: {entry_price:.4f}\\n"
                    f"Stop Loss: {sl:.4f}\\n"
                    f"TP1 (1.2R - 50%): {self.take_profit_1r[symbol]:.4f}\\n"
                    f"TP2 (2.2R - 30%): {self.take_profit_2r[symbol]:.4f}\\n"
                    f"Runner (3.5R - 20%): {self.take_profit[symbol]:.4f}\\n"
                    f"Position Size: {pos_size:.6f}\\n"
                    f"Confidence: {prob:.2f}\\n"
                    f"Reason: {metadata.get('reason', 'N/A')}"
                )
                add_log_message(f"[{symbol}] " + msg_str.replace('\\n', ' | '))
                await self.notifier.send_message(msg_str)
            else:
                add_log_message(f"[{symbol}] ❌ SELL order REJECTED (check execution logs)")
                await self.notifier.send_message(f"⚠️ SELL REJECTED {symbol}: Order failed to execute. Check logs.")

    async def change_bot_symbol(self, new_symbol: str):
        """Switches the primary active symbol in the dashboard dynamically."""
        if new_symbol and new_symbol in Config.SUPPORTED_SYMBOLS:
            Config.SYMBOL = new_symbol
            DashboardState.symbol = new_symbol
            DashboardState.in_position = self.in_position.get(new_symbol, False)
            DashboardState.position_side = self.position_side.get(new_symbol, "HOLD")
            DashboardState.entry_price = self.entry_price.get(new_symbol, 0.0)
            DashboardState.stop_loss = self.stop_loss.get(new_symbol, 0.0)
            DashboardState.take_profit = self.take_profit.get(new_symbol, 0.0)
            add_log_message(f"🔄 Active dashboard symbol switched to: {new_symbol}")
            print(f"[BOT] Switched active symbol to {new_symbol}")

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
                    sym_price = self.pipeline.latest_prices.get(symbol, 0.0)
                    if sym_price > 0:
                        self.lead_lag.record_tick(symbol, sym_price)
                        if symbol != "BTC/USDT":
                            lead_lag_opp = self.lead_lag.evaluate_lead_lag(symbol)
                            if lead_lag_opp.get('signal') != 'NONE':
                                add_log_message(f"[{symbol}] ⚡ Cross-Asset Lead-Lag Impulse: {lead_lag_opp['signal']} (BTC Velocity: {lead_lag_opp['btc_velocity_pct']}%, Lag Edge: +{lead_lag_opp['edge_pct']}%)")

                    if self.in_position[symbol] and self.pipeline.latest_prices.get(symbol, 0.0) > 0:
                        curr_price = self.pipeline.latest_prices[symbol]
                        
                        ltf_df = prepare_dataframe(self.pipeline.ltf_candles[symbol])
                        curr_atr = calculate_atr(ltf_df, Config.ATR_PERIOD).iloc[-1] if not ltf_df.empty else 0.001
                        
                        if self.position_side[symbol] == "LONG":
                            self.highest_price_reached[symbol] = max(self.highest_price_reached[symbol], curr_price)
                            r_dist = abs(self.entry_price[symbol] - self.stop_loss[symbol])
                            
                            # ZERO-RISK FREE-TRADE LOCK: Move SL to Breakeven at +0.50R Profit
                            if self.highest_price_reached[symbol] >= self.entry_price[symbol] + (0.50 * r_dist):
                                fee_offset = min(self.entry_price[symbol] * 0.0015, 0.15 * r_dist)
                                be_sl = min(curr_price * 0.9995, self.entry_price[symbol] + fee_offset)
                                if be_sl > self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = be_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = be_sl
                                    add_log_message(f"[{symbol}] 🛡️ ZERO-RISK FREE-TRADE ACTIVATED: SL moved to Breakeven ({be_sl:.4f})")

                            # 1. ⏱️ STAGNATION KILLER: Time-based exit if trade loses momentum near entry
                            if getattr(Config, 'ENABLE_TIME_STOP', True) and not self.partial_tp_taken[symbol]:
                                entry_ts = self.entry_time.get(symbol, 0)
                                if entry_ts > 0:
                                    tf_mins = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
                                    time_elapsed_secs = time.time() - (entry_ts / 1000.0)
                                    candles_open = time_elapsed_secs / (tf_mins * 60)
                                    max_stagnant_candles = getattr(Config, 'MAX_STAGNANT_CANDLES', 16)
                                    if candles_open >= max_stagnant_candles:
                                        stagnant_max_dist = getattr(Config, 'STAGNANT_MAX_R_DISTANCE', 0.25) * r_dist
                                        if abs(curr_price - self.entry_price[symbol]) <= stagnant_max_dist:
                                            add_log_message(f"[{symbol}] ⏱️ STAGNATION KILLER: Position open for {candles_open:.1f} candles ({time_elapsed_secs/3600:.1f}h) with no momentum. Auto-exiting at scratch.")
                                            await self.notifier.send_message(f"⏱️ *STAGNATION AUTO-EXIT ({symbol})*\nTrade open for {candles_open:.1f} candles without momentum. Scratch-closed to reclaim capital.")
                                            await self.exit_position(symbol, "STAGNANT_TIME_EXIT")
                                            continue

                            # 2. ⚡ EARLY STRUCTURAL EXIT: Cut loss early if 15m structure breaks against LONG
                            if getattr(Config, 'ENABLE_STRUCTURAL_EXIT', True) and not self.partial_tp_taken[symbol] and len(ltf_df) >= 3:
                                last_c = ltf_df.iloc[-1]
                                prev_c = ltf_df.iloc[-2]
                                ema_20 = calculate_ema(ltf_df, 20).iloc[-1] if len(ltf_df) >= 20 else 0.0
                                avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) >= 14 else 1.0
                                unrealized_loss_r = (self.entry_price[symbol] - curr_price) / r_dist if r_dist > 0 else 0.0
                                early_max_r = getattr(Config, 'EARLY_EXIT_MAX_LOSS_R', 0.45)
                                if 0.15 <= unrealized_loss_r <= early_max_r and ema_20 > 0:
                                    is_bearish_engulf = (last_c['close'] < last_c['open']) and (prev_c['close'] > prev_c['open']) and (last_c['close'] < prev_c['open'])
                                    is_heavy_vol = last_c['volume'] > 1.8 * avg_vol
                                    is_below_ema = last_c['close'] < ema_20
                                    if (is_bearish_engulf or is_below_ema) and is_heavy_vol:
                                        add_log_message(f"[{symbol}] ⚡ EARLY STRUCTURAL EXIT: Heavy bearish breakdown below EMA20. Cutting loss early at -{unrealized_loss_r:.2f}R.")
                                        await self.notifier.send_message(f"⚡ *EARLY STRUCTURAL EXIT ({symbol})*\nMarket structure broke against LONG before full SL. Loss cut early at -{unrealized_loss_r:.2f}R (Saved ~{(1.0-unrealized_loss_r)*100:.0f}% loss).")
                                        await self.exit_position(symbol, "EARLY_STRUCTURAL_INVALIDATION")
                                        continue
                            
                            # TP1 (50% Scale-Out at 1.2R)
                            if not self.partial_tp_taken[symbol] and curr_price >= self.take_profit_1r[symbol]:
                                add_log_message(f"[{symbol}] 🎯 Target 1 hit! Booking 50% profit.")
                                tp1_size = self.position_size[symbol] * 0.50
                                tp1_success = False
                                if self.has_keys:
                                    tp1_order = await self.execution.place_order('sell', 'market', tp1_size, symbol=symbol, is_exit_order=True)
                                    tp1_success = bool(tp1_order)
                                else:
                                    self._dry_run_balance_usdt += tp1_size * curr_price
                                    tp1_success = True
                                if tp1_success:
                                    self.position_size[symbol] -= tp1_size
                                    self.partial_tp_taken[symbol] = True
                                    
                                    # Log partial TP1 trade record for accurate PnL tracking
                                    tp1_pnl = tp1_size * (curr_price - self.entry_price[symbol])
                                    DashboardState.trades.append({
                                        'symbol': symbol, 'side': 'LONG', 'type': 'TP1_PARTIAL',
                                        'entry': self.entry_price[symbol], 'exit': curr_price,
                                        'size': tp1_size, 'pnl': round(tp1_pnl, 4),
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    
                                    # PROFIT LOCK: Guarantee profit by setting Stop Loss to Breakeven (+0.15%)
                                    profit_lock_sl = self.entry_price[symbol] * 1.0015
                                    if profit_lock_sl > self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = profit_lock_sl
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = profit_lock_sl
                                        guar_pnl_pct = max(0.0, (profit_lock_sl - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0)
                                        guar_pnl_usdt = max(0.0, self.position_size[symbol] * (profit_lock_sl - self.entry_price[symbol]))
                                        add_log_message(f"[{symbol}] 🔒 PROFIT LOCKED at Stop Loss: {profit_lock_sl:.4f} (+{guar_pnl_usdt:.2f} USDT). New Target: TP2 @ {self.take_profit_2r[symbol]:.4f}")
                                        await self.notifier.send_message(
                                            f"🔒 *PROFIT LOCKED ({symbol})*\n"
                                            f"🎯 TP1 Hit! 50% profit booked.\n"
                                            f"🔒 Guaranteed Locked Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                            f"🎯 New Active Target: TP2 (2.2R) @ {self.take_profit_2r[symbol]:.4f}"
                                        )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP1 order REJECTED by exchange. State NOT updated.")
                                    
                            # TP2 (30% Scale-Out at 2.2R -> Leaves 20% Runner)
                            if self.partial_tp_taken[symbol] and not self.tp2_taken[symbol] and curr_price >= self.take_profit_2r[symbol]:
                                add_log_message(f"[{symbol}] 🎯 Target 2 (2.2R) hit! Booking 30% profit. 20% Runner active.")
                                tp2_size = self.position_size[symbol] * 0.60  # 60% of remaining 50% = 30% of original
                                tp2_success = False
                                if self.has_keys:
                                    tp2_order = await self.execution.place_order('sell', 'market', tp2_size, symbol=symbol, is_exit_order=True)
                                    tp2_success = bool(tp2_order)
                                else:
                                    self._dry_run_balance_usdt += tp2_size * curr_price
                                    tp2_success = True
                                if tp2_success:
                                    self.position_size[symbol] -= tp2_size
                                    self.tp2_taken[symbol] = True
                                    
                                    # Log partial TP2 trade record for accurate PnL tracking
                                    tp2_pnl = tp2_size * (curr_price - self.entry_price[symbol])
                                    DashboardState.trades.append({
                                        'symbol': symbol, 'side': 'LONG', 'type': 'TP2_PARTIAL',
                                        'entry': self.entry_price[symbol], 'exit': curr_price,
                                        'size': tp2_size, 'pnl': round(tp2_pnl, 4),
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    # Lock SL at TP1 level (Guaranteed deep profit lock)
                                    if self.take_profit_1r[symbol] > self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = self.take_profit_1r[symbol]
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = self.take_profit_1r[symbol]
                                    guar_pnl_pct = max(0.0, (self.stop_loss[symbol] - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0)
                                    guar_pnl_usdt = max(0.0, self.position_size[symbol] * (self.stop_loss[symbol] - self.entry_price[symbol]))
                                    add_log_message(f"[{symbol}] 🚀 TP2 Hit! SL locked at TP1 level ({self.stop_loss[symbol]:.4f}). Trailing Runner active.")
                                    await self.notifier.send_message(
                                        f"🚀 *DEEP PROFIT LOCKED ({symbol})*\n"
                                        f"🎯 TP2 Hit! 30% profit booked.\n"
                                        f"🔒 Guaranteed Deep Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                        f"🎯 New Active Target: Runner Target (3.5R) @ {self.take_profit[symbol]:.4f}"
                                    )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP2 order REJECTED by exchange. State NOT updated.")

                            if self.partial_tp_taken[symbol]:
                                new_sl = self.risk.update_trailing_stop(self.entry_price[symbol], self.highest_price_reached[symbol], self.stop_loss[symbol], curr_atr, "LONG")
                                if new_sl > self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = new_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = new_sl
                                
                            if curr_price >= self.take_profit[symbol]:
                                await self.exit_position(symbol, "TAKE_PROFIT_RUNNER")
                            elif curr_price <= self.stop_loss[symbol]:
                                await self.exit_position(symbol, "TRAILING_STOP")
                                
                        elif self.position_side[symbol] == "SHORT":
                            self.lowest_price_reached[symbol] = min(self.lowest_price_reached[symbol], curr_price)
                            r_dist = abs(self.entry_price[symbol] - self.stop_loss[symbol])
                            
                            # ZERO-RISK FREE-TRADE LOCK: Move SL to Breakeven at +0.50R Profit
                            if self.lowest_price_reached[symbol] <= self.entry_price[symbol] - (0.50 * r_dist):
                                fee_offset = min(self.entry_price[symbol] * 0.0015, 0.15 * r_dist)
                                be_sl = max(curr_price * 1.0005, self.entry_price[symbol] - fee_offset)
                                if be_sl < self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = be_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = be_sl
                                    add_log_message(f"[{symbol}] 🛡️ ZERO-RISK FREE-TRADE ACTIVATED: SL moved to Breakeven ({be_sl:.4f})")

                            # 1. ⏱️ STAGNATION KILLER: Time-based exit if trade loses momentum near entry
                            if getattr(Config, 'ENABLE_TIME_STOP', True) and not self.partial_tp_taken[symbol]:
                                entry_ts = self.entry_time.get(symbol, 0)
                                if entry_ts > 0:
                                    tf_mins = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
                                    time_elapsed_secs = time.time() - (entry_ts / 1000.0)
                                    candles_open = time_elapsed_secs / (tf_mins * 60)
                                    max_stagnant_candles = getattr(Config, 'MAX_STAGNANT_CANDLES', 16)
                                    if candles_open >= max_stagnant_candles:
                                        stagnant_max_dist = getattr(Config, 'STAGNANT_MAX_R_DISTANCE', 0.25) * r_dist
                                        if abs(curr_price - self.entry_price[symbol]) <= stagnant_max_dist:
                                            add_log_message(f"[{symbol}] ⏱️ STAGNATION KILLER: Position open for {candles_open:.1f} candles ({time_elapsed_secs/3600:.1f}h) with no momentum. Auto-exiting at scratch.")
                                            await self.notifier.send_message(f"⏱️ *STAGNATION AUTO-EXIT ({symbol})*\nTrade open for {candles_open:.1f} candles without momentum. Scratch-closed to reclaim capital.")
                                            await self.exit_position(symbol, "STAGNANT_TIME_EXIT")
                                            continue

                            # 2. ⚡ EARLY STRUCTURAL EXIT: Cut loss early if 15m structure breaks against SHORT
                            if getattr(Config, 'ENABLE_STRUCTURAL_EXIT', True) and not self.partial_tp_taken[symbol] and len(ltf_df) >= 3:
                                last_c = ltf_df.iloc[-1]
                                prev_c = ltf_df.iloc[-2]
                                ema_20 = calculate_ema(ltf_df, 20).iloc[-1] if len(ltf_df) >= 20 else 0.0
                                avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) >= 14 else 1.0
                                unrealized_loss_r = (curr_price - self.entry_price[symbol]) / r_dist if r_dist > 0 else 0.0
                                early_max_r = getattr(Config, 'EARLY_EXIT_MAX_LOSS_R', 0.45)
                                if 0.15 <= unrealized_loss_r <= early_max_r and ema_20 > 0:
                                    is_bullish_engulf = (last_c['close'] > last_c['open']) and (prev_c['close'] < prev_c['open']) and (last_c['close'] > prev_c['open'])
                                    is_heavy_vol = last_c['volume'] > 1.8 * avg_vol
                                    is_above_ema = last_c['close'] > ema_20
                                    if (is_bullish_engulf or is_above_ema) and is_heavy_vol:
                                        add_log_message(f"[{symbol}] ⚡ EARLY STRUCTURAL EXIT: Heavy bullish breakout above EMA20. Cutting loss early at -{unrealized_loss_r:.2f}R.")
                                        await self.notifier.send_message(f"⚡ *EARLY STRUCTURAL EXIT ({symbol})*\nMarket structure broke against SHORT before full SL. Loss cut early at -{unrealized_loss_r:.2f}R (Saved ~{(1.0-unrealized_loss_r)*100:.0f}% loss).")
                                        await self.exit_position(symbol, "EARLY_STRUCTURAL_INVALIDATION")
                                        continue
                            
                            # TP1 (50% Scale-Out at 1.2R)
                            if not self.partial_tp_taken[symbol] and curr_price <= self.take_profit_1r[symbol]:
                                add_log_message(f"[{symbol}] 🎯 Target 1 hit! Booking 50% profit.")
                                tp1_size = self.position_size[symbol] * 0.50
                                tp1_success = False
                                if self.has_keys:
                                    tp1_order = await self.execution.place_order('buy', 'market', tp1_size, symbol=symbol, is_exit_order=True)
                                    tp1_success = bool(tp1_order)
                                else:
                                    self._dry_run_balance_usdt += tp1_size * (self.entry_price[symbol] - curr_price) + (tp1_size * self.entry_price[symbol])
                                    tp1_success = True
                                if tp1_success:
                                    self.position_size[symbol] -= tp1_size
                                    self.partial_tp_taken[symbol] = True
                                    
                                    # Log partial TP1 trade record for accurate PnL tracking
                                    tp1_pnl = tp1_size * (self.entry_price[symbol] - curr_price)
                                    DashboardState.trades.append({
                                        'symbol': symbol, 'side': 'SHORT', 'type': 'TP1_PARTIAL',
                                        'entry': self.entry_price[symbol], 'exit': curr_price,
                                        'size': tp1_size, 'pnl': round(tp1_pnl, 4),
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    
                                    # PROFIT LOCK: Guarantee profit by setting Stop Loss to Breakeven (-0.15%)
                                    profit_lock_sl = self.entry_price[symbol] * 0.9985
                                    if profit_lock_sl < self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = profit_lock_sl
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = profit_lock_sl
                                        guar_pnl_pct = max(0.0, (self.entry_price[symbol] - profit_lock_sl) / self.entry_price[symbol] * 100.0)
                                        guar_pnl_usdt = max(0.0, self.position_size[symbol] * (self.entry_price[symbol] - profit_lock_sl))
                                        add_log_message(f"[{symbol}] 🔒 PROFIT LOCKED at Stop Loss: {profit_lock_sl:.4f} (+{guar_pnl_usdt:.2f} USDT). New Target: TP2 @ {self.take_profit_2r[symbol]:.4f}")
                                        await self.notifier.send_message(
                                            f"🔒 *PROFIT LOCKED ({symbol})*\n"
                                            f"🎯 TP1 Hit! 50% profit booked.\n"
                                            f"🔒 Guaranteed Locked Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                            f"🎯 New Active Target: TP2 (2.2R) @ {self.take_profit_2r[symbol]:.4f}"
                                        )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP1 order REJECTED by exchange. State NOT updated.")
                                    
                            # TP2 (30% Scale-Out at 2.2R -> Leaves 20% Runner)
                            if self.partial_tp_taken[symbol] and not self.tp2_taken[symbol] and curr_price <= self.take_profit_2r[symbol]:
                                add_log_message(f"[{symbol}] 🎯 Target 2 (2.2R) hit! Booking 30% profit. 20% Runner active.")
                                tp2_size = self.position_size[symbol] * 0.60  # 60% of remaining 50% = 30% of original
                                tp2_success = False
                                if self.has_keys:
                                    tp2_order = await self.execution.place_order('buy', 'market', tp2_size, symbol=symbol, is_exit_order=True)
                                    tp2_success = bool(tp2_order)
                                else:
                                    self._dry_run_balance_usdt += tp2_size * (self.entry_price[symbol] - curr_price) + (tp2_size * self.entry_price[symbol])
                                    tp2_success = True
                                if tp2_success:
                                    self.position_size[symbol] -= tp2_size
                                    self.tp2_taken[symbol] = True
                                    
                                    # Log partial TP2 trade record for accurate PnL tracking
                                    tp2_pnl = tp2_size * (self.entry_price[symbol] - curr_price)
                                    DashboardState.trades.append({
                                        'symbol': symbol, 'side': 'SHORT', 'type': 'TP2_PARTIAL',
                                        'entry': self.entry_price[symbol], 'exit': curr_price,
                                        'size': tp2_size, 'pnl': round(tp2_pnl, 4),
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    # Lock SL at TP1 level (Guaranteed deep profit lock)
                                    if self.take_profit_1r[symbol] < self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = self.take_profit_1r[symbol]
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = self.take_profit_1r[symbol]
                                    guar_pnl_pct = max(0.0, (self.entry_price[symbol] - self.stop_loss[symbol]) / self.entry_price[symbol] * 100.0)
                                    guar_pnl_usdt = max(0.0, self.position_size[symbol] * (self.entry_price[symbol] - self.stop_loss[symbol]))
                                    add_log_message(f"[{symbol}] 🚀 TP2 Hit! SL locked at TP1 level ({self.stop_loss[symbol]:.4f}). Trailing Runner active.")
                                    await self.notifier.send_message(
                                        f"🚀 *DEEP PROFIT LOCKED ({symbol})*\n"
                                        f"🎯 TP2 Hit! 30% profit booked.\n"
                                        f"🔒 Guaranteed Deep Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                        f"🎯 New Active Target: Runner Target (3.5R) @ {self.take_profit[symbol]:.4f}"
                                    )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP2 order REJECTED by exchange. State NOT updated.")

                            if self.partial_tp_taken[symbol]:
                                new_sl = self.risk.update_trailing_stop(self.entry_price[symbol], self.lowest_price_reached[symbol], self.stop_loss[symbol], curr_atr, "SHORT")
                                if new_sl < self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = new_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = new_sl
                                
                            if curr_price <= self.take_profit[symbol]:
                                await self.exit_position(symbol, "TAKE_PROFIT_RUNNER")
                            elif curr_price >= self.stop_loss[symbol]:
                                await self.exit_position(symbol, "TRAILING_STOP")

                # Update UI for selected Config.SYMBOL
                sym = Config.SYMBOL
                DashboardState.latest_price = self.pipeline.latest_prices.get(sym, 0.0)
                if not self.has_keys:
                    DashboardState.balance_usdt = self.calculate_total_equity()
                    
                if self.pipeline.ltf_candles[sym]:
                    DashboardState.chart_history = self.pipeline.ltf_candles[sym][-100:]
                
                # Sync all active positions to DashboardState for multi-symbol UI rendering
                active_pos_map = {}
                for s in Config.SUPPORTED_SYMBOLS:
                    if self.in_position[s]:
                        # Clean up ghost positions where size is zero or depleted
                        if self.position_size[s] is None or self.position_size[s] <= 1e-7:
                            self.in_position[s] = False
                            self.position_side[s] = "HOLD"
                            self.position_size[s] = 0.0
                            continue

                        live_p = self.pipeline.latest_prices.get(s, self.entry_price[s])
                        entry_val = self.entry_price[s]
                        pos_sz = self.position_size[s]
                        if live_p > 0 and entry_val > 0:
                            if self.position_side[s] == "LONG":
                                p_pct = (live_p - entry_val) / entry_val * 100.0
                                p_usdt = pos_sz * (live_p - entry_val)
                            else:
                                p_pct = (entry_val - live_p) / entry_val * 100.0
                                p_usdt = pos_sz * (entry_val - live_p)
                        else:
                            p_pct = 0.0
                            p_usdt = 0.0
                        
                        is_long = (self.position_side[s] == "LONG")
                        sl_val = self.stop_loss[s]
                        entry_val = self.entry_price[s]
                        is_profit_locked = (is_long and sl_val >= entry_val) or (not is_long and sl_val <= entry_val and sl_val > 0)
                        
                        r_dist = abs(entry_val - sl_val) if abs(entry_val - sl_val) > 0 else (entry_val * 0.01)
                        target_1r = self.take_profit_1r[s] if self.take_profit_1r[s] > 0 else (entry_val + 1.2 * r_dist if is_long else entry_val - 1.2 * r_dist)
                        target_2r = self.take_profit_2r[s] if self.take_profit_2r[s] > 0 else (entry_val + 2.2 * r_dist if is_long else entry_val - 2.2 * r_dist)
                        final_tp = self.take_profit[s] if self.take_profit[s] > 0 else (entry_val + 3.5 * r_dist if is_long else entry_val - 3.5 * r_dist)
                        
                        # Guarantee final_tp is beyond target_2r
                        if is_long and final_tp <= target_2r:
                            final_tp = entry_val + 3.5 * r_dist
                        elif not is_long and final_tp >= target_2r:
                            final_tp = entry_val - 3.5 * r_dist

                        tp1_hit = self.partial_tp_taken[s]
                        tp2_hit = self.tp2_taken[s]
                        
                        # Dynamic target escalation upon hitting targets
                        if not tp1_hit:
                            active_target = target_1r
                            active_target_name = "TP1 (1.2R)"
                            target_stage = 1
                        elif not tp2_hit:
                            active_target = target_2r
                            active_target_name = "TP2 (2.2R)"
                            target_stage = 2
                        else:
                            # Runner stage: Ensure target is strictly ahead of trailing SL
                            if is_long:
                                active_target = max(final_tp, sl_val + r_dist)
                            else:
                                active_target = min(final_tp, sl_val - r_dist)
                            active_target_name = "Runner Target"
                            target_stage = 3

                        # Calculate guaranteed locked profit in USDT and %
                        guaranteed_pnl_usdt = 0.0
                        guaranteed_pnl_pct = 0.0
                        if is_profit_locked and entry_val > 0:
                            if is_long:
                                guaranteed_pnl_pct = max(0.0, (sl_val - entry_val) / entry_val * 100.0)
                                guaranteed_pnl_usdt = max(0.0, pos_sz * (sl_val - entry_val))
                            else:
                                guaranteed_pnl_pct = max(0.0, (entry_val - sl_val) / entry_val * 100.0)
                                guaranteed_pnl_usdt = max(0.0, pos_sz * (entry_val - sl_val))

                        active_pos_map[s] = {
                            'side': self.position_side[s],
                            'entry_price': entry_val,
                            'stop_loss': sl_val,
                            'take_profit': active_target,
                            'target_1r': target_1r,
                            'target_2r': target_2r,
                            'final_target': final_tp,
                            'active_target': active_target,
                            'active_target_name': active_target_name,
                            'target_stage': target_stage,
                            'position_size': self.position_size[s],
                            'current_pnl_usdt': p_usdt,
                            'current_pnl_pct': p_pct,
                            'guaranteed_pnl_usdt': guaranteed_pnl_usdt,
                            'guaranteed_pnl_pct': guaranteed_pnl_pct,
                            'tp1_hit': tp1_hit,
                            'tp2_hit': tp2_hit,
                            'profit_locked': is_profit_locked,
                            'live_price': live_p
                        }
                DashboardState.active_positions = active_pos_map

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

            except Exception as e:
                import traceback
                print(f"[RISK MONITOR] Error: {e}")
                traceback.print_exc()
            await asyncio.sleep(1.0)

    async def emergency_close_all(self):
        """Instantly close all open positions across all supported symbols."""
        closed_symbols = []
        failed_symbols = []
        add_log_message("🚨 EMERGENCY CLOSE ALL TRIGGERED: Exiting all active positions immediately...")
        for sym in Config.SUPPORTED_SYMBOLS:
            if self.in_position[sym]:
                try:
                    await self.exit_position(sym, "EMERGENCY_CLOSE_ALL")
                    # Verify it was actually closed
                    if not self.in_position[sym]:
                        closed_symbols.append(sym)
                    else:
                        failed_symbols.append(sym)
                        add_log_message(f"[{sym}] ⚠️ Exit order may have been rejected. Force-clearing local state.")
                        # Force-clear local state even if exchange order failed
                        self.in_position[sym] = False
                        self.position_side[sym] = "HOLD"
                        self.position_size[sym] = 0.0
                except Exception as e:
                    import traceback
                    add_log_message(f"[{sym}] Error closing position during emergency stop: {e}")
                    traceback.print_exc()
                    failed_symbols.append(sym)
                    # Force-clear local state on error
                    self.in_position[sym] = False
                    self.position_side[sym] = "HOLD"
                    self.position_size[sym] = 0.0
        
        DashboardState.active_positions = {}
        DashboardState.in_position = False
        DashboardState.position_side = "HOLD"
        self.save_state()

        count = len(closed_symbols)
        fail_count = len(failed_symbols)
        msg = f"Emergency stop executed: Closed {count} positions."
        if fail_count > 0:
            msg += f" WARNING: {fail_count} positions force-cleared ({', '.join(failed_symbols)})."
        add_log_message(f"🚨 {msg}")
        await self.notifier.send_message(f"🚨 *EMERGENCY CLOSE ALL EXECUTED*\nClosed {count} positions. Force-cleared {fail_count}.")
        return count + fail_count, msg

    async def exit_position(self, symbol, reason):
        # FIX #5: Better fallback for exit_price to avoid 0.0 values
        exit_price = self.pipeline.latest_prices.get(symbol) or self.entry_price[symbol]
        if exit_price <= 0 or not exit_price:
            exit_price = self.entry_price[symbol]
        add_log_message(f"[{symbol}] Exiting at price: {exit_price:.4f} (reason: {reason})")
        order = None
        if self.has_keys:
            side = 'buy' if self.position_side[symbol] == 'SHORT' else 'sell'
            order = await self.execution.place_order(side, 'market', self.position_size[symbol], price=exit_price, is_exit_order=True, symbol=symbol)
        else:
            order = {'id': 'MOCK_EXIT_ORDER_ID', 'price': exit_price, 'status': 'filled'}
            
        if order:
            if self.position_side[symbol] == "LONG":
                pnl_pct = (exit_price - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0
                pnl_usdt = self.position_size[symbol] * (exit_price - self.entry_price[symbol])
                if not self.has_keys:
                    self._dry_run_balance_usdt += self.position_size[symbol] * exit_price
            else:
                pnl_pct = (self.entry_price[symbol] - exit_price) / self.entry_price[symbol] * 100.0
                pnl_usdt = self.position_size[symbol] * (self.entry_price[symbol] - exit_price)
                if not self.has_keys:
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

            # Persist to data/trade_logs.jsonl on disk
            try:
                log_dir = Path("data")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "trade_logs.jsonl"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trade_record) + "\n")
            except Exception as e:
                print(f"[LOG] Failed to write trade log: {e}")

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
            pos_mode = self.position_mode.get(symbol, 'STRICT')
            if pos_mode == 'RELAXED' and is_loss:
                self.relaxed_losses += 1
                if self.relaxed_losses >= 2:
                    self.relaxed_disabled_until = time.time() + 7200
                    add_log_message("🚨 [SAFETY] 2 relaxed losses. Relaxed mode disabled for 2 hours.")
                    self.relaxed_losses = 0
            elif not is_loss and pos_mode == 'RELAXED':
                self.relaxed_losses = 0

            self.in_position[symbol] = False
            self.position_side[symbol] = "HOLD"
            self.position_size[symbol] = 0.0
            self.entry_price[symbol] = 0.0
            self.stop_loss[symbol] = 0.0
            self.take_profit[symbol] = 0.0
            self.take_profit_1r[symbol] = 0.0
            self.take_profit_2r[symbol] = 0.0
            self.partial_tp_taken[symbol] = False
            self.tp2_taken[symbol] = False

            if symbol in DashboardState.active_positions:
                new_positions = DashboardState.active_positions.copy()
                del new_positions[symbol]
                DashboardState.active_positions = new_positions

            if symbol == Config.SYMBOL:
                DashboardState.in_position = False
                DashboardState.position_side = "HOLD"
                DashboardState.entry_price = 0.0
                DashboardState.stop_loss = 0.0
                DashboardState.take_profit = 0.0
                DashboardState.current_pnl_pct = 0.0
                DashboardState.current_pnl_usdt = 0.0

            self.save_state()
            await self.notifier.send_message(
                f"🚨 *{symbol} CLOSED ({reason})*\nExit Price: {exit_price:.2f}\nPnL: {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT)"
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

    async def change_execution_timeframe(self, new_tf):
        new_tf = new_tf.lower()
        if new_tf not in ['1m', '5m']:
            return
        add_log_message(f"⏱️ Switching execution timeframe to {new_tf.upper()}...")
        Config.LTF_TIMEFRAME = new_tf
        
        # Warm up LTF historical candles for the new timeframe across all symbols
        for sym in Config.SUPPORTED_SYMBOLS:
            ltf_ohlcv = await self.execution.fetch_ohlcv(
                symbol=sym,
                timeframe=new_tf,
                limit=500
            )
            if ltf_ohlcv:
                self.pipeline.ltf_candles[sym] = ltf_ohlcv
                if self.ml_models[sym] is not None:
                    df = prepare_dataframe(ltf_ohlcv)
                    self.ml_models[sym].train(df)
        
        await self.pipeline.restart_streams()
        sym = Config.SYMBOL
        DashboardState.ltf_timeframe = new_tf
        DashboardState.chart_history = self.pipeline.ltf_candles[sym][-100:] if self.pipeline.ltf_candles[sym] else []
        add_log_message(f"✅ Execution timeframe switched to {new_tf.upper()}. Chart & signals active.")

    def lock_position_profit(self, symbol):
        """Manually lock profit for an active position by adjusting Stop Loss."""
        if not self.in_position[symbol]:
            return False, f"No active position open for {symbol}."
            
        live_p = self.pipeline.latest_prices.get(symbol, self.entry_price[symbol])
        entry_p = self.entry_price[symbol]
        sl_p = self.stop_loss[symbol]
        is_long = (self.position_side[symbol] == "LONG")
        r_dist = abs(entry_p - sl_p)
        
        # Guard: Reject profit lock if position is currently in a loss
        if is_long and live_p < entry_p:
            return False, f"Cannot lock profit for {symbol} — position is currently in loss (Live: {live_p:.4f} < Entry: {entry_p:.4f})."
        if not is_long and live_p > entry_p:
            return False, f"Cannot lock profit for {symbol} — position is currently in loss (Live: {live_p:.4f} > Entry: {entry_p:.4f})."
        
        if is_long:
            # Lock at least breakeven + 0.15R or 50% of current unrealized profit
            lock_delta = max(0.15 * r_dist, (live_p - entry_p) * 0.5)
            new_sl = max(sl_p, entry_p + lock_delta)
            self.stop_loss[symbol] = new_sl
            if symbol == Config.SYMBOL:
                DashboardState.stop_loss = new_sl
        else:
            lock_delta = max(0.15 * r_dist, (entry_p - live_p) * 0.5)
            new_sl = min(sl_p, entry_p - lock_delta)
            self.stop_loss[symbol] = new_sl
            if symbol == Config.SYMBOL:
                DashboardState.stop_loss = new_sl
                
        self.save_state()
        msg = f"🔒 Profit locked for {symbol}! New Stop Loss set to {self.stop_loss[symbol]:.4f}"
        add_log_message(f"[{symbol}] {msg}")
        return True, msg

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

import asyncio
import sys
import uvicorn
import time

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
from strategies.indicators import prepare_dataframe
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
        self.ml = MLSignalConfirmator()
        self.risk = RiskManager()
        self.notifier = TelegramNotifier()
        
        # Internal State tracking
        self.in_position = False
        self.position_side = "HOLD"
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.highest_price_reached = 0.0
        self.lowest_price_reached = 999999.0
        self.position_size = 0.0
        self.entry_time = 0

        # Dry-run virtual balance (used when no API keys are set)
        self._dry_run_balance_usdt = 10000.0   # starting paper balance
        
        # CRITICAL FIX: Initialize DashboardState with dry-run balance
        # This ensures position sizing always has a valid equity value
        if not self.has_keys:
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            DashboardState.balance_base = 0.0
            print("[INIT] ✅ Dry-run mode: Virtual balance initialized to $10,000 USDT")

        # CRITICAL-3 FIX: Lock to prevent concurrent candle processing.
        # Without this, two candle closes firing while a previous evaluation is
        # still running (e.g. awaiting balance fetch) would both see
        # in_position=False and place duplicate orders.
        self._candle_lock = asyncio.Lock()
        
        # Link callbacks
        self.pipeline.on_candle_close_callback = self.on_candle_close

    # ─── CRITICAL-5 FIX: State Persistence ─────────────────────────────────
    # Saves bot position state to disk so a server restart (common on Render
    # free tier) can recover the open position instead of leaving it unprotected.
    import json as _json
    from pathlib import Path as _Path
    _STATE_FILE = _Path("bot_state.json")

    def save_state(self):
        """Persist current position state to disk for crash recovery."""
        import json
        from pathlib import Path
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
        }
        try:
            Path("bot_state.json").write_text(json.dumps(state))
        except Exception as e:
            print(f"[STATE] Failed to save state: {e}")

    def load_state(self):
        """Restore position state from disk after a restart."""
        import json
        from pathlib import Path
        state_file = Path("bot_state.json")
        if not state_file.exists():
            return
        try:
            state = json.loads(state_file.read_text())
            self.in_position = state.get('in_position', False)
            self.position_side = state.get('position_side', 'HOLD')
            self.entry_price = state.get('entry_price', 0.0)
            self.stop_loss = state.get('stop_loss', 0.0)
            self.take_profit = state.get('take_profit', 0.0)
            self.position_size = state.get('position_size', 0.0)
            self.entry_time = state.get('entry_time', 0)
            self.highest_price_reached = state.get('highest_price_reached', 0.0)
            self.lowest_price_reached = state.get('lowest_price_reached', 999999.0)
            self._dry_run_balance_usdt = state.get('_dry_run_balance_usdt', 10000.0)
            # Sync to dashboard
            DashboardState.in_position = self.in_position
            DashboardState.position_side = self.position_side
            DashboardState.entry_price = self.entry_price
            DashboardState.stop_loss = self.stop_loss
            DashboardState.take_profit = self.take_profit
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            if self.in_position:
                add_log_message(f"[STATE] Recovered {self.position_side} position from disk: entry={self.entry_price:.2f}, SL={self.stop_loss:.2f}, TP={self.take_profit:.2f}")
            else:
                add_log_message("[STATE] State file loaded — no open position to recover.")
        except Exception as e:
            print(f"[STATE] Failed to load state: {e}")

    async def initialize(self):
        """
        Initializes websocket connections, loads history, and trains the ML classifier.
        Also restores any previously saved position state from disk.
        """
        add_log_message("Starting system initialization...")

        # CRITICAL-5 FIX: Restore persisted state before starting
        # This recovers any open position after a server restart
        self.load_state()

        await self.pipeline.start()
        
        # Wait a moment for websocket connection to load initial ticks
        await asyncio.sleep(3)
        
        # Initial Balance load
        if self.has_keys:
            balance = await self.execution.fetch_balance()
            if balance:
                # MINOR-4 FIX: Validate balance before assigning — a zero/None
                # result (wrong account type, API error) would silently set an
                # incorrect equity and allow oversized position sizing.
                usdt_balance = balance.get('total', {}).get('USDT', None)
                if usdt_balance and usdt_balance > 0:
                    DashboardState.balance_usdt = usdt_balance
                else:
                    add_log_message(f"[WARNING] Balance fetch returned {usdt_balance}. Check account type. Keeping last known value.")
                DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
        else:
            # Ensure dry-run balance is synced to DashboardState
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            DashboardState.balance_base = 0.0
        
        # Train ML Model on historical candles
        ltf_history = self.pipeline.ltf_candles
        if ltf_history:
            df = prepare_dataframe(ltf_history)
            add_log_message("Training ML confirmation model on historical price ticks...")
            trained = self.ml.train(df)
            if trained:
                add_log_message("ML Model training completed successfully.")
            else:
                add_log_message("ML Model training skipped (insufficient warm-up history).")

        # Initial price check
        DashboardState.latest_price = self.pipeline.latest_price
        DashboardState.chart_history = self.pipeline.ltf_candles[-100:]
        if ltf_history:
            DashboardState.ml_confidence = self.ml.predict_bias(df)
        add_log_message(f"System ready. Watching {Config.SYMBOL} at {DashboardState.latest_price} USDT")

    async def on_candle_close(self):
        """
        Callback executed every time a lower-timeframe (LTF) candle closes.
        Runs strategy generation, ML validation, risk checking, and execution.

        CRITICAL-3 FIX: Uses an asyncio.Lock to prevent concurrent execution.
        If a previous candle evaluation is still running (e.g. awaiting an API
        call), the new candle is skipped instead of creating a duplicate trade.
        """
        if self._candle_lock.locked():
            add_log_message("[LOCK] Candle fired while previous evaluation still running — skipping to prevent duplicate orders.")
            return

        async with self._candle_lock:
            await self._on_candle_close_impl()

    async def _on_candle_close_impl(self):
        """Internal candle close handler — called exclusively inside _candle_lock."""
        add_log_message("LTF Candle closed. Running strategy check...")
        
        # 1. Update balances on candle close
        if self.has_keys:
            balance = await self.execution.fetch_balance()
            if balance:
                # MINOR-4 FIX: Validate balance before assigning
                usdt_balance = balance.get('total', {}).get('USDT', None)
                if usdt_balance and usdt_balance > 0:
                    DashboardState.balance_usdt = usdt_balance
                else:
                    add_log_message(f"[WARNING] Candle balance refresh returned {usdt_balance}. Keeping last known value.")
                DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
                
        # 2. Check drawdown circuit breakers
        if not self.has_keys:
            current_equity = self._dry_run_balance_usdt
            if self.in_position:
                if self.position_side == "LONG":
                    current_equity += self.position_size * self.pipeline.latest_price
                elif self.position_side == "SHORT":
                    unrealized_pnl = self.position_size * (self.entry_price - self.pipeline.latest_price)
                    current_equity += (self.position_size * self.entry_price) + unrealized_pnl
        else:
            current_equity = DashboardState.balance_usdt + (DashboardState.balance_base * self.pipeline.latest_price)
            
        if not self.risk.check_circuit_breaker(current_equity):
            add_log_message("Trading halted: Daily drawdown limit reached.")
            await self.notifier.send_message("❌ TRADING HALTED: Daily loss circuit breaker triggered.")
            return

        # Sync daily drawdown percentage to dashboard
        DashboardState.daily_drawdown_pct = self.risk.current_drawdown_pct

        # 3. Build dataframes for strategy evaluation
        htf_df = prepare_dataframe(self.pipeline.htf_candles)
        ltf_df = prepare_dataframe(self.pipeline.ltf_candles)
        
        # Update ML confidence bias metric for the dashboard
        DashboardState.ml_confidence = self.ml.predict_bias(ltf_df)
        
        # Update chart history
        DashboardState.chart_history = self.pipeline.ltf_candles[-100:]
        
        # 4. Generate Signal
        signal, metadata = self.strategy.generate_signal(htf_df, ltf_df)
        
        # Update dashboard state indicators
        DashboardState.active_ob = metadata.get('reason', 'No OB/FVG')
        DashboardState.active_ob_level = metadata.get('active_ob_level', 0.0)
        DashboardState.active_ob_type = metadata.get('active_ob_type', 'NONE')
        DashboardState.active_bullish_ob_level = metadata.get('active_bullish_ob_level', 0.0)
        DashboardState.active_bearish_ob_level = metadata.get('active_bearish_ob_level', 0.0)
        
        if signal == "HOLD":
            return
            
        add_log_message(f"Raw strategy signal generated: {signal} ({metadata.get('reason')})")
        
        # 5. Check if we already have an active position matching the signal
        if self.in_position:
            if self.position_side == "LONG" and signal == "SELL":
                add_log_message("Trend reversal: Closing LONG position.")
                await self.exit_position("SIGNAL_REVERSAL")
            elif self.position_side == "SHORT" and signal == "BUY":
                add_log_message("Trend reversal: Closing SHORT position.")
                await self.exit_position("SIGNAL_REVERSAL")
            else:
                add_log_message(f"Ignoring {signal} signal: Already holding a {self.position_side} position.")
            return
            
        # 6. ML Confirmation Filter
        confirmed, prob = self.ml.confirm_signal(ltf_df, signal)
        DashboardState.ml_confidence = prob
        
        if not confirmed:
            add_log_message(f"Trade filtered by ML confirmation filter. Bias score: {prob:.2f} (Required: {Config.ML_CONFIRMATION_THRESHOLD:.2f})")
            return
            
        add_log_message(f"Trade confirmed by ML filter. Bias score: {prob:.2f}. Proceeding to risk checks...")
        
        # 7. Execute orders based on signal
        # BUG FIX #4: Use the last CLOSED candle close price for entry.
        # Strategy computed SL/TP relative to ltf_df['close'].iloc[-2].
        # Using live ticker (latest_price) creates SL distance mismatch.
        entry_price = prepare_dataframe(self.pipeline.ltf_candles)['close'].iloc[-2]
        sl = metadata['stop_loss']
        tp = metadata['take_profit']
        
        # Determine dynamic size
        print(f"[DEBUG] Position size calculation:")
        print(f"[DEBUG]   Balance: {DashboardState.balance_usdt}")
        print(f"[DEBUG]   Entry: {entry_price:.2f}, SL: {sl:.2f}")
        pos_size = self.risk.calculate_position_size(DashboardState.balance_usdt, entry_price, sl)
        print(f"[DEBUG]   Result: {pos_size:.6f}")
        if pos_size <= 0.0:
            add_log_message("❌ Trade aborted: Risk manager returned zero position size.")
            return
            
        if signal == "BUY":
            add_log_message(f"Executing BUY (LONG) entry order. Size: {pos_size:.6f} | SL: {sl:.2f} | TP: {tp:.2f}")
            
            order = None
            if self.has_keys:
                order = await self.execution.place_order('buy', 'market', pos_size, price=entry_price)
            else:
                # BUG FIX #8: Dry-run — simulate balance deduction
                position_cost = pos_size * entry_price
                if position_cost <= self._dry_run_balance_usdt:
                    self._dry_run_balance_usdt -= position_cost
                    DashboardState.balance_usdt = self._dry_run_balance_usdt
                    order = {'id': 'MOCK_BUY_ORDER_ID', 'price': entry_price, 'status': 'filled'}
                else:
                    add_log_message(f"[DRY-RUN] Insufficient simulated balance ({self._dry_run_balance_usdt:.2f} USDT) for this trade.")
                    order = None
                
            if order:
                self.in_position = True
                self.position_side = "LONG"
                self.entry_price = entry_price
                self.stop_loss = sl
                self.take_profit = tp
                self.highest_price_reached = entry_price
                self.position_size = pos_size
                self.entry_time = int(time.time() * 1000)
                
                # Sync dashboard state
                DashboardState.in_position = True
                DashboardState.position_side = "LONG"
                DashboardState.entry_price = entry_price
                DashboardState.stop_loss = sl
                DashboardState.take_profit = tp

                # CRITICAL-5 FIX: Persist state so restart can recover position
                self.save_state()
                
                await self.notifier.send_message(
                    f"🟢 *BUY (LONG) Order Executed*\n"
                    f"Price: {entry_price:.2f} USDT\n"
                    f"Size: {pos_size:.6f}\n"
                    f"Stop Loss: {sl:.2f}\n"
                    f"Take Profit: {tp:.2f}\n"
                    f"Reason: {metadata.get('reason')}"
                )
                
        elif signal == "SELL":
            add_log_message(f"Executing SELL (SHORT) entry order. Size: {pos_size:.6f} | SL: {sl:.2f} | TP: {tp:.2f}")
            
            order = None
            if self.has_keys:
                order = await self.execution.place_order('sell', 'market', pos_size, price=entry_price)
            else:
                # BUG FIX #8: Dry-run short — simulate margin hold (use balance as collateral)
                collateral = pos_size * entry_price
                if collateral <= self._dry_run_balance_usdt:
                    self._dry_run_balance_usdt -= collateral
                    DashboardState.balance_usdt = self._dry_run_balance_usdt
                    order = {'id': 'MOCK_SELL_ORDER_ID', 'price': entry_price, 'status': 'filled'}
                else:
                    add_log_message(f"[DRY-RUN] Insufficient simulated balance for SHORT collateral.")
                    order = None
                
            if order:
                self.in_position = True
                self.position_side = "SHORT"
                self.entry_price = entry_price
                self.stop_loss = sl
                self.take_profit = tp
                self.lowest_price_reached = entry_price
                self.position_size = pos_size
                self.entry_time = int(time.time() * 1000)
                
                # Sync dashboard state
                DashboardState.in_position = True
                DashboardState.position_side = "SHORT"
                DashboardState.entry_price = entry_price
                DashboardState.stop_loss = sl
                DashboardState.take_profit = tp

                # CRITICAL-5 FIX: Persist state so restart can recover position
                self.save_state()
                
                await self.notifier.send_message(
                    f"🔴 *SELL (SHORT) Order Executed*\n"
                    f"Price: {entry_price:.2f} USDT\n"
                    f"Size: {pos_size:.6f}\n"
                    f"Stop Loss: {sl:.2f}\n"
                    f"Take Profit: {tp:.2f}\n"
                    f"Reason: {metadata.get('reason')}"
                )

    async def run_live_risk_monitor(self):
        """
        Periodic task running every second to check trailing stops, 
        take profits, and manage live position state updates.
        """
        while True:
            try:
                # Check if symbol change was requested via UI
                if DashboardState.symbol_change_requested:
                    new_symbol = DashboardState.symbol_change_requested
                    DashboardState.symbol_change_requested = None
                    await self.change_bot_symbol(new_symbol)

                # Midnight daily equity reset (UTC)
                import datetime
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                if now_utc.hour == 0 and now_utc.minute == 0 and now_utc.second < 2:
                    if not self.has_keys:
                        current_eq = self._dry_run_balance_usdt
                        if self.in_position:
                            if self.position_side == "LONG":
                                current_eq += self.position_size * self.pipeline.latest_price
                            elif self.position_side == "SHORT":
                                unrealized_pnl = self.position_size * (self.entry_price - self.pipeline.latest_price)
                                current_eq += (self.position_size * self.entry_price) + unrealized_pnl
                    else:
                        current_eq = DashboardState.balance_usdt + (DashboardState.balance_base * self.pipeline.latest_price)
                    self.risk.reset_daily_equity(current_eq)
                    add_log_message("[RISK] Daily equity checkpoint reset at UTC midnight.")

                # Update latest price to dashboard
                DashboardState.latest_price = self.pipeline.latest_price
                
                # Update simulated balance_usdt to represent total equity (cash + position value)
                if not self.has_keys:
                    eq = self._dry_run_balance_usdt
                    if self.in_position:
                        if self.position_side == "LONG":
                            eq += self.position_size * self.pipeline.latest_price
                            DashboardState.balance_base = self.position_size
                        elif self.position_side == "SHORT":
                            unrealized_pnl = self.position_size * (self.entry_price - self.pipeline.latest_price)
                            eq += (self.position_size * self.entry_price) + unrealized_pnl
                            DashboardState.balance_base = 0.0
                    else:
                        DashboardState.balance_base = 0.0
                    DashboardState.balance_usdt = eq
                
                # Update chart history in real-time
                if self.pipeline.ltf_candles:
                    DashboardState.chart_history = self.pipeline.ltf_candles[-100:]
                
                if self.in_position and self.pipeline.latest_price > 0:
                    curr_price = self.pipeline.latest_price
                    
                    # Compute running unrealized PnL
                    if self.position_side == "LONG":
                        self.highest_price_reached = max(self.highest_price_reached, curr_price)
                        # Check trailing stop-loss updates
                        new_sl = self.risk.update_trailing_stop(self.entry_price, self.highest_price_reached, self.stop_loss, "LONG")
                        if new_sl > self.stop_loss:
                            self.stop_loss = new_sl
                            DashboardState.stop_loss = new_sl
                            add_log_message(f"[RISK] Trailing stop updated to {new_sl:.2f}")
                            
                        # Check stop hit
                        if curr_price <= self.stop_loss:
                            add_log_message(f"🚨 Trailing Stop hit at {curr_price:.2f}. Liquidating position.")
                            await self.exit_position("TRAILING_STOP")
                        # Check profit target hit
                        elif curr_price >= self.take_profit:
                            add_log_message(f"🎯 Take profit target hit at {curr_price:.2f}. Liquidating position.")
                            await self.exit_position("TAKE_PROFIT")
                            
                        # Update unrealized PnL
                        pnl_pct = (curr_price - self.entry_price) / self.entry_price * 100.0
                        pnl_usdt = self.position_size * (curr_price - self.entry_price)
                        DashboardState.current_pnl_pct = pnl_pct
                        DashboardState.current_pnl_usdt = pnl_usdt
                    elif self.position_side == "SHORT":
                        self.lowest_price_reached = min(self.lowest_price_reached, curr_price)
                        # Check trailing stop-loss updates
                        new_sl = self.risk.update_trailing_stop(self.entry_price, self.lowest_price_reached, self.stop_loss, "SHORT")
                        if new_sl < self.stop_loss:
                            self.stop_loss = new_sl
                            DashboardState.stop_loss = new_sl
                            add_log_message(f"[RISK] Trailing stop updated to {new_sl:.2f}")
                            
                        # Check stop hit (price goes ABOVE stop loss)
                        if curr_price >= self.stop_loss:
                            add_log_message(f"🚨 Trailing Stop hit at {curr_price:.2f}. Liquidating position.")
                            await self.exit_position("TRAILING_STOP")
                        # Check profit target hit (price goes BELOW take profit)
                        elif curr_price <= self.take_profit:
                            add_log_message(f"🎯 Take profit target hit at {curr_price:.2f}. Liquidating position.")
                            await self.exit_position("TAKE_PROFIT")
                            
                        # Update unrealized PnL
                        pnl_pct = (self.entry_price - curr_price) / self.entry_price * 100.0
                        pnl_usdt = self.position_size * (self.entry_price - curr_price)
                        DashboardState.current_pnl_pct = pnl_pct
                        DashboardState.current_pnl_usdt = pnl_usdt
            except Exception as e:
                # MINOR-7 FIX: Print full traceback so bugs in SL/TP monitoring
                # are never silently swallowed
                import traceback
                print(f"[RISK MONITOR] Error: {e}")
                traceback.print_exc()
            await asyncio.sleep(1.0)

    async def exit_position(self, reason):
        """Helper to force exit current position due to stop/limit triggers."""
        exit_price = self.pipeline.latest_price
        order = None
        if self.has_keys:
            side = 'buy' if self.position_side == 'SHORT' else 'sell'
            # ATTACK-5 FIX: Pass is_exit_order=True to bypass slippage guard.
            # Exit orders MUST execute regardless of slippage — blocking an exit
            # during a flash crash leaves the position unprotected indefinitely.
            order = await self.execution.place_order(side, 'market', self.position_size, price=exit_price, is_exit_order=True)
        else:
            order = {'id': 'MOCK_EXIT_ORDER_ID', 'price': exit_price, 'status': 'filled'}
            
        if order:
            if self.position_side == "LONG":
                pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100.0
                pnl_usdt = self.position_size * (exit_price - self.entry_price)
            else: # SHORT
                pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100.0
                pnl_usdt = self.position_size * (self.entry_price - exit_price)
                
            trade_record = {
                'side': self.position_side,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'pnl_usdt': pnl_usdt,
                'pnl_pct': pnl_pct,
                'entry_time': self.entry_time,
                'exit_time': int(time.time() * 1000)
            }
            DashboardState.trades.append(trade_record)

            # ATTACK-2 FIX: Bound trades list at 500 entries to prevent unbounded
            # memory growth in long-running paper trading sessions.
            if len(DashboardState.trades) > 500:
                DashboardState.trades = DashboardState.trades[-500:]

            # BUG FIX #8: Dry-run — credit virtual balance with exit proceeds
            if not self.has_keys:
                if self.position_side == "LONG":
                    # Return: cash from selling position at exit price
                    self._dry_run_balance_usdt += self.position_size * exit_price
                else:
                    # Return: collateral + short profit (or minus loss)
                    self._dry_run_balance_usdt += (self.position_size * self.entry_price) + pnl_usdt
                DashboardState.balance_usdt = self._dry_run_balance_usdt
                add_log_message(f"[DRY-RUN] Virtual balance after exit: {self._dry_run_balance_usdt:.2f} USDT")
            
            self.in_position = False
            self.position_side = "HOLD"
            self.position_size = 0.0
            DashboardState.in_position = False
            DashboardState.position_side = "HOLD"

            # CRITICAL-5 FIX: Clear saved state after position is closed
            self.save_state()
            
            await self.notifier.send_message(
                f"🚨 *POSITION LIQUIDATED ({reason})*\n"
                f"Exit Price: {exit_price:.2f} USDT\n"
                f"PnL: {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT)"
            )

    async def change_bot_symbol(self, new_symbol):
        """
        Dynamically changes the bot's trading asset:
        1. Stops the current pipeline.
        2. Resets open position and dashboard states.
        3. Updates symbol config.
        4. Re-initializes pipeline caches and starts WebSocket feed.
        5. Retrains the Machine Learning confirmation model on the new coin's history.
        """
        # ATTACK-7 FIX: Guard against concurrent symbol changes.
        # If the previous WebSocket loop is still running (not yet cancelled),
        # deferring is safer than starting a second pipeline that will mix
        # candle data from two different symbols into the same cache.
        ws_task = getattr(self.pipeline, 'websocket_task', None)
        if ws_task and not ws_task.done() and not self.pipeline.websocket_active:
            add_log_message(f"[SYMBOL] Previous pipeline still shutting down. Deferring change to {new_symbol}...")
            DashboardState.symbol_change_requested = new_symbol  # re-queue for next monitor tick
            return

        if self.in_position:
            add_log_message(f"Force-closing active {self.position_side} position on {Config.SYMBOL} before switching to {new_symbol}...")
            await self.exit_position("SYMBOL_CHANGE")
            
        add_log_message(f"Initiating symbol change request to {new_symbol}...")
        
        # 1. Stop current websocket pipeline
        self.pipeline.stop()
        await asyncio.sleep(0.1) # tiny sleep for WS task shutdown buffer
        
        # 2. Reset bot state
        self.in_position = False
        self.position_side = "HOLD"
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Reset Dashboard state
        DashboardState.in_position = False
        DashboardState.position_side = "HOLD"
        DashboardState.entry_price = 0.0
        DashboardState.stop_loss = 0.0
        DashboardState.take_profit = 0.0
        DashboardState.active_ob = "No OB"
        DashboardState.active_fvg = "No FVG"
        DashboardState.active_bullish_ob_level = 0.0
        DashboardState.active_bearish_ob_level = 0.0
        DashboardState.chart_history = []
        DashboardState.trades = []
        
        # 3. Update symbol config
        Config.SYMBOL = new_symbol
        # Adjust default dry-run trade amount for safety
        if "BTC" in new_symbol:
            Config.TRADE_AMOUNT = 0.001
        elif "ETH" in new_symbol:
            Config.TRADE_AMOUNT = 0.02
        else:
            Config.TRADE_AMOUNT = 1.0 # default fallback for other altcoins
            
        # 4. Restart pipeline
        self.pipeline.ltf_candles = []
        self.pipeline.htf_candles = []
        self.pipeline.latest_price = 0.0
        
        # Re-start pipeline (will fetch history and connect websocket)
        await self.pipeline.start()
        
        # Immediately set latest_price from the last candle close in history to avoid 0.0 lag state
        if self.pipeline.ltf_candles:
            self.pipeline.latest_price = self.pipeline.ltf_candles[-1][4]
        
        # 6. Retrain ML Model on the new coin's historical data
        ltf_history = self.pipeline.ltf_candles
        if ltf_history:
            df = prepare_dataframe(ltf_history)
            add_log_message(f"Retraining ML confirmation model on historical {new_symbol} ticks...")
            trained = self.ml.train(df)
            if trained:
                add_log_message("ML Model retrained successfully.")
                DashboardState.ml_confidence = self.ml.predict_bias(df)
            else:
                add_log_message("ML Model retraining skipped (insufficient warm-up history).")
                
        # Update dashboard state indicators
        DashboardState.latest_price = self.pipeline.latest_price
        DashboardState.chart_history = self.pipeline.ltf_candles[-100:]
        add_log_message(f"Symbol successfully changed. Watching {Config.SYMBOL} at {DashboardState.latest_price} USDT")

    async def shutdown(self):
        add_log_message("Shutting down exchange sessions gracefully...")
        await self.execution.close()
        self.pipeline.stop()

# CRITICAL-2 FIX: Removed dead main() and run_bot_loops() functions.
# main() was never called and referenced a non-existent method
# (bot.run_risk_monitor_task instead of bot.run_live_risk_monitor).
# run_bot_loops() was also never called. Both are replaced by start_all().

async def start_all():
    import dashboard.app as dashboard_module

    bot = PrimeSignalBot()

    # Register bot with dashboard so startup event can launch it
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
    # MINOR-6 FIX: WindowsSelectorEventLoopPolicy deprecated in Python 3.12+
    if sys.platform == 'win32' and sys.version_info < (3, 12):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(start_all())
    except KeyboardInterrupt:
        print("\nStopping bot...")

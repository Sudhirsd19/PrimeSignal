import pandas as pd
import numpy as np
import datetime
from config import Config
from strategies.indicators import prepare_dataframe
from risk.risk_manager import RiskManager

class BacktestEngine:
    def __init__(self, strategy, risk_manager=None, ml_confirmator=None):
        self.strategy = strategy
        self.risk = risk_manager if risk_manager else RiskManager()
        self.ml = ml_confirmator
        
        # Performance Tracking
        self.trades = []
        self.equity_curve = []
        self.balance = 0.0

    def run(self, htf_candles, ltf_candles, initial_balance=10000.0):
        print(f"[BACKTEST] Initializing backtest simulation with {initial_balance:.2f} USDT...")
        
        # Clear metrics
        self.trades = []
        self.equity_curve = []
        self.balance = initial_balance
        
        self.relaxed_enabled = True # Control flag for backtest auto-disable
        
        # 1. Prepare DataFrames
        htf_df = prepare_dataframe(htf_candles)
        ltf_df = prepare_dataframe(ltf_candles)
        from strategies.indicators import calculate_atr
        ltf_atr = calculate_atr(ltf_df, Config.ATR_PERIOD)
        
        if len(htf_df) < Config.TREND_EMA or len(ltf_df) < Config.LONG_EMA + 10:
            print("ERROR: Not enough historical candles to run backtest.")
            return None
            
        print(f"[BACKTEST] Aligned data frames: HTF ({len(htf_df)} bars) | LTF ({len(ltf_df)} bars)")
        
        # Track state
        in_position = False
        position_side = None
        entry_price = 0.0
        position_size = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        highest_price = 0.0
        lowest_price = 999999.0
        entry_time = None
        last_trade_time = 0
        last_zone_traded = None
        volatility_pause_until = 0
        partial_tp_taken = False
        take_profit_1r = 0.0
        setup_mode = 'STRICT'
        
        consecutive_losses = 0
        global_pause_until = 0
        relaxed_losses = 0
        relaxed_disabled_until = 0
        relaxed_trades_today = 0
        fee_rate = getattr(Config, 'FEE_RATE', 0.001)
        slippage_pct = getattr(Config, 'MAX_SLIPPAGE_PCT', 0.002)

        # We start loop from index where indicators are warmed up on both sides
        start_idx = max(Config.TREND_EMA * 12, 100) # Ensure HTF EMA is warm (1h EMA 200 = 200 hours, so at 5m we need at least 2400 bars)
        if start_idx >= len(ltf_df) - 50:
            start_idx = Config.LONG_EMA + 20
            
        last_trade_day = ltf_df.index[start_idx].date()
        print(f"[BACKTEST] Running simulation loop from bar {start_idx} to {len(ltf_df)}...")
        
        for i in range(start_idx, len(ltf_df)):
            ltf_time = ltf_df.index[i]
            curr_candle = ltf_df.iloc[i]
            
            # Slice historical data to prevent look-ahead bias
            # For LTF, we use index up to i (meaning candle i is the current live candle, i-1 is the last completed)
            sub_ltf = ltf_df.iloc[:i+1]
            
            # For HTF, we can only see candles that closed BEFORE the current LTF timestamp
            sub_htf = htf_df[htf_df.index < ltf_time]
            
            current_close = curr_candle['close']
            current_high = curr_candle['high']
            current_low = curr_candle['low']
            curr_atr = ltf_atr.iloc[i]
            
            if ltf_time.date() != last_trade_day:
                relaxed_trades_today = 0
                last_trade_day = ltf_time.date()

            if ltf_time.timestamp() < global_pause_until:
                continue
            
            # Record equity curve
            current_equity = self.balance
            if in_position:
                if position_side == "LONG":
                    # Total equity = remaining cash + current value of open position
                    current_equity = self.balance + (position_size * current_close)
                else:
                    current_equity = self.balance + (position_size * (entry_price - current_close))
            self.equity_curve.append({'timestamp': ltf_time, 'equity': current_equity})
            
            # --- MANAGE OPEN POSITION ---
            if in_position:
                if position_side == "LONG":
                    # Update highest price for trailing stop
                    highest_price = max(highest_price, current_high)
                    
                    # Partial TP Check BEFORE SL to ensure it correctly triggers if both hit
                    if not partial_tp_taken and current_high >= take_profit_1r:
                        partial_exit_price = max(take_profit_1r, curr_candle['open'])
                        half_size = position_size * 0.5
                        
                        gross_val = half_size * partial_exit_price
                        fee = gross_val * fee_rate
                        self.balance += gross_val - fee
                        
                        pnl_usdt = (half_size * partial_exit_price) - (half_size * entry_price) - fee
                        pnl_pct = (partial_exit_price - entry_price) / entry_price * 100
                        
                        self.trades.append({
                            'side': 'LONG',
                            'entry_time': entry_time,
                            'exit_time': ltf_time,
                            'entry_price': entry_price,
                            'exit_price': partial_exit_price,
                            'pnl_usdt': pnl_usdt,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'PARTIAL_TP_1R',
                            'setup_mode': setup_mode
                        })
                        position_size -= half_size
                        partial_tp_taken = True
                        
                        if entry_price > stop_loss:
                            stop_loss = entry_price

                    if partial_tp_taken:
                        stop_loss = self.risk.update_trailing_stop(entry_price, highest_price, stop_loss, curr_atr, "LONG")
                    
                    # PRIORITY EXIT CHECK: Take Profit checked FIRST
                    if current_high >= take_profit:
                        exit_price = max(take_profit, curr_candle['open'])
                        gross_val = position_size * exit_price
                        fee = gross_val * fee_rate
                        self.balance = gross_val - fee
                        
                        pnl_usdt = self.balance - (position_size * entry_price)
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        
                        self.trades.append({
                            'side': 'LONG',
                            'entry_time': entry_time,
                            'exit_time': ltf_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_usdt': pnl_usdt,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'TAKE_PROFIT',
                            'setup_mode': setup_mode
                        })
                        if pnl_usdt < 0:
                            consecutive_losses += 1
                            if setup_mode == 'RELAXED': relaxed_losses += 1
                        else:
                            consecutive_losses = 0
                            if setup_mode == 'RELAXED': relaxed_losses = 0
                            
                        if consecutive_losses >= 3:
                            global_pause_until = ltf_time.timestamp() + 3600
                            consecutive_losses = 0
                        if relaxed_losses >= 2:
                            relaxed_disabled_until = ltf_time.timestamp() + 7200
                            relaxed_losses = 0
                            
                        in_position = False
                        position_size = 0.0

                    # Check Stop Loss
                    elif current_low <= stop_loss:
                        # Exit at stop loss level or open if gap down
                        exit_price = min(stop_loss, curr_candle['open'])
                        gross_val = position_size * exit_price
                        fee = gross_val * fee_rate
                        self.balance = gross_val - fee
                        
                        pnl_usdt = self.balance - (position_size * entry_price) # Close value minus initial cost
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        
                        self.trades.append({
                            'side': 'LONG',
                            'entry_time': entry_time,
                            'exit_time': ltf_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_usdt': pnl_usdt,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'STOP_LOSS',
                            'setup_mode': setup_mode
                        })
                        if pnl_usdt < 0:
                            consecutive_losses += 1
                            if setup_mode == 'RELAXED': relaxed_losses += 1
                        else:
                            consecutive_losses = 0
                            if setup_mode == 'RELAXED': relaxed_losses = 0
                            
                        if consecutive_losses >= 3:
                            global_pause_until = ltf_time.timestamp() + 3600
                            consecutive_losses = 0
                        if relaxed_losses >= 2:
                            relaxed_disabled_until = ltf_time.timestamp() + 7200
                            relaxed_losses = 0
                            
                        in_position = False
                        position_size = 0.0
                        

                        
                elif position_side == "SHORT":
                    # For short, we track lowest price reached
                    lowest_price = min(lowest_price, current_low)
                    
                    # Partial TP Check
                    if not partial_tp_taken and current_low <= take_profit_1r:
                        partial_exit_price = min(take_profit_1r, curr_candle['open'])
                        half_size = position_size * 0.5
                        
                        pnl_usdt = half_size * (entry_price - partial_exit_price)
                        fee = half_size * partial_exit_price * fee_rate
                        self.balance += pnl_usdt - fee
                        
                        pnl_pct = (entry_price - partial_exit_price) / entry_price * 100
                        
                        self.trades.append({
                            'side': 'SHORT',
                            'entry_time': entry_time,
                            'exit_time': ltf_time,
                            'entry_price': entry_price,
                            'exit_price': partial_exit_price,
                            'pnl_usdt': pnl_usdt - fee,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'PARTIAL_TP_1R',
                            'setup_mode': setup_mode
                        })
                        position_size -= half_size
                        partial_tp_taken = True
                        
                        if entry_price < stop_loss:
                            stop_loss = entry_price

                    if partial_tp_taken:
                        stop_loss = self.risk.update_trailing_stop(entry_price, lowest_price, stop_loss, curr_atr, "SHORT")
                    
                    # PRIORITY EXIT CHECK: Take Profit checked FIRST
                    if current_low <= take_profit:
                        exit_price = min(take_profit, curr_candle['open'])
                        pnl_usdt = position_size * (entry_price - exit_price)
                        self.balance = self.balance + pnl_usdt - (position_size * exit_price * fee_rate)
                        
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                        
                        self.trades.append({
                            'side': 'SHORT',
                            'entry_time': entry_time,
                            'exit_time': ltf_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_usdt': pnl_usdt,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'TAKE_PROFIT',
                            'setup_mode': setup_mode
                        })
                        if pnl_usdt < 0:
                            consecutive_losses += 1
                            if setup_mode == 'RELAXED': relaxed_losses += 1
                        else:
                            consecutive_losses = 0
                            if setup_mode == 'RELAXED': relaxed_losses = 0
                            
                        if consecutive_losses >= 3:
                            global_pause_until = ltf_time.timestamp() + 3600
                            consecutive_losses = 0
                        if relaxed_losses >= 2:
                            relaxed_disabled_until = ltf_time.timestamp() + 7200
                            relaxed_losses = 0
                            
                        in_position = False
                        position_size = 0.0

                    # Check Stop Loss
                    elif current_high >= stop_loss:
                        exit_price = max(stop_loss, curr_candle['open'])
                        # PnL for short = size * (entry - exit)
                        pnl_usdt = position_size * (entry_price - exit_price)
                        self.balance = self.balance + pnl_usdt - (position_size * exit_price * fee_rate)
                        
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                        
                        self.trades.append({
                            'side': 'SHORT',
                            'entry_time': entry_time,
                            'exit_time': ltf_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_usdt': pnl_usdt,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'STOP_LOSS',
                            'setup_mode': setup_mode
                        })
                        if pnl_usdt < 0:
                            consecutive_losses += 1
                            if setup_mode == 'RELAXED': relaxed_losses += 1
                        else:
                            consecutive_losses = 0
                            if setup_mode == 'RELAXED': relaxed_losses = 0
                            
                        if consecutive_losses >= 3:
                            global_pause_until = ltf_time.timestamp() + 3600
                            consecutive_losses = 0
                        if relaxed_losses >= 2:
                            relaxed_disabled_until = ltf_time.timestamp() + 7200
                            relaxed_losses = 0
                            
                        in_position = False
                        position_size = 0.0
                        

                        
            # --- EVALUATE NEW ENTRIES ---
            else:
                # Volatility Kill Switch
                last_candle = sub_ltf.iloc[-1]
                move_pct = abs(last_candle['close'] - last_candle['open']) / last_candle['open']
                if move_pct > getattr(Config, 'MAX_CANDLE_MOVE_PCT', 0.015):
                    volatility_pause_until = i + getattr(Config, 'VOLATILITY_PAUSE_CANDLES', 2)
                    continue
                    
                if i < volatility_pause_until:
                    continue

                signal, signal_meta = self.strategy.generate_signal(sub_htf, sub_ltf, relaxed=False)
                setup_mode = 'STRICT'
                
                if signal == "HOLD" and self.relaxed_enabled:
                    if ltf_time.timestamp() - last_trade_time >= 30 * 60:
                        if relaxed_trades_today < 2 and ltf_time.timestamp() > relaxed_disabled_until:
                            signal, signal_meta = self.strategy.generate_signal(sub_htf, sub_ltf, relaxed=True)
                            setup_mode = 'RELAXED' 
                
                if signal in ("BUY", "SELL"):
                    # ML Confidence Scaler
                    ml_prob = 1.0
                    if self.ml:
                        ml_prob = self.ml.predict_bias(sub_ltf)
                        
                    trade_risk_pct = getattr(Config, 'RISK_PCT', 0.02)
                    if ml_prob < getattr(Config, 'ML_CONFIRMATION_THRESHOLD', 0.60):
                        trade_risk_pct *= 0.5
                        
                    # Calculate Stop Loss and Take Profit
                    sl = signal_meta['stop_loss']
                    tp = signal_meta['take_profit']
                    
                    if sl is None or tp is None:
                        continue
                        
                    # Cooldown check
                    if ltf_time.timestamp() - last_trade_time < getattr(Config, 'COOLDOWN_MINUTES', 15) * 60:
                        continue
                        
                    # Zone check
                    zone_id = signal_meta.get('zone_id')
                    if zone_id and zone_id == last_zone_traded:
                        continue
                        
                    # Slippage simulation
                    entry_price = current_close * (1 + slippage_pct) if signal == "BUY" else current_close * (1 - slippage_pct)
                    
                    entry_time = ltf_time
                    stop_loss = sl
                    take_profit = tp
                    highest_price = entry_price
                    lowest_price = entry_price
                    last_trade_time = ltf_time.timestamp()
                    last_zone_traded = zone_id
                    
                    # Calculate dynamic position size based on risk percent
                    position_size = self.risk.calculate_position_size(self.balance, entry_price, stop_loss)
                    position_size = position_size * (trade_risk_pct / getattr(Config, 'RISK_PCT', 0.02))
                    
                    if position_size <= 0.0:
                        continue
                        
                    if setup_mode == "RELAXED":
                        relaxed_trades_today += 1
                        
                    if signal == "BUY":
                        position_side = "LONG"
                        fee = position_size * entry_price * fee_rate
                        self.balance -= (position_size * entry_price) + fee
                        partial_tp_taken = False
                        take_profit_1r = entry_price + (entry_price - stop_loss)
                        in_position = True
                    elif signal == "SELL":
                        position_side = "SHORT"
                        # For short spot simulation, we assume margin-trading/contracts or simply track quote balance
                        fee = position_size * entry_price * fee_rate
                        self.balance -= fee
                        partial_tp_taken = False
                        take_profit_1r = entry_price - (stop_loss - entry_price)
                        in_position = True

        # Liquidate open position at the end of backtest
        if in_position:
            exit_price = ltf_df['close'].iloc[-1]
            exit_time = ltf_df.index[-1]
            if position_side == "LONG":
                gross_val = position_size * exit_price
                fee = gross_val * fee_rate
                self.balance = gross_val - fee
                pnl_usdt = self.balance - (position_size * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_usdt = position_size * (entry_price - exit_price)
                self.balance = self.balance + pnl_usdt - (position_size * exit_price * fee_rate)
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                
            self.trades.append({
                'side': position_side,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_usdt': pnl_usdt,
                'pnl_pct': pnl_pct,
                'exit_reason': 'FORCE_CLOSE_END',
                'setup_mode': setup_mode
            })
            
        # 3. Calculate Performance Metrics
        return self.calculate_metrics(initial_balance)

    def calculate_metrics(self, initial_balance):
        base_metrics = {
            'initial_balance': initial_balance,
            'final_balance': self.balance,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'profit_factor': 0.0,
            'strict_win_rate': 0.0,
            'strict_pf': 0.0,
            'strict_dd': 0.0,
            'relaxed_win_rate': 0.0,
            'relaxed_pf': 0.0,
            'relaxed_dd': 0.0
        }
        if not self.trades:
            return base_metrics
            
        trade_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        def get_subset_metrics(df_sub):
            if len(df_sub) == 0: return 0.0, 0.0, 0.0
            w = df_sub[df_sub['pnl_usdt'] > 0]
            l = df_sub[df_sub['pnl_usdt'] <= 0]
            wr = (len(w) / len(df_sub)) * 100.0
            tw = w['pnl_usdt'].sum()
            tl = abs(l['pnl_usdt'].sum())
            pf = tw / tl if tl > 0 else float('inf')
            # approx DD just for this subset based on PNL cumulative
            cum = df_sub['pnl_usdt'].cumsum()
            peak = cum.cummax()
            dd = ((cum - peak) / initial_balance * 100.0).min() if len(cum) > 0 else 0.0
            return wr, pf, dd
            
        total_trades = len(trade_df)
        wins = trade_df[trade_df['pnl_usdt'] > 0]
        losses = trade_df[trade_df['pnl_usdt'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100.0
        
        total_win = wins['pnl_usdt'].sum()
        total_loss = abs(losses['pnl_usdt'].sum())
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # Max Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['dd'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100.0
        max_drawdown = equity_df['dd'].min()
        
        # Sharpe Ratio (daily/candle returns based)
        equity_df['returns'] = equity_df['equity'].pct_change()
        mean_return = equity_df['returns'].mean()
        std_return = equity_df['returns'].std()
        
        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(288 * 365)
        else:
            sharpe_ratio = 0.0
            
        final_return_pct = ((self.balance - initial_balance) / initial_balance) * 100.0
        
        strict_df = trade_df[trade_df.get('setup_mode', 'STRICT') == 'STRICT']
        relaxed_df = trade_df[trade_df.get('setup_mode', 'STRICT') == 'RELAXED']
        
        swr, spf, sdd = get_subset_metrics(strict_df)
        rwr, rpf, rdd = get_subset_metrics(relaxed_df)
        
        metrics = {
            'initial_balance': initial_balance,
            'final_balance': self.balance,
            'total_return_pct': final_return_pct,
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor,
            'strict_win_rate': swr,
            'strict_pf': spf,
            'strict_dd': sdd,
            'relaxed_win_rate': rwr,
            'relaxed_pf': rpf,
            'relaxed_dd': rdd
        }
        
        if rpf < spf and rpf > 0:
            print(f"[BACKTEST] WARNING: Relaxed mode PF ({rpf:.2f}) < Strict PF ({spf:.2f}). Consider disabling.")
            
        return metrics

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
        
        # 1. Prepare DataFrames
        htf_df = prepare_dataframe(htf_candles)
        ltf_df = prepare_dataframe(ltf_candles)
        
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
        
        fee_rate = 0.001  # 0.1% spot fee
        
        # We start loop from index where indicators are warmed up on both sides
        start_idx = max(Config.TREND_EMA * 12, 100) # Ensure HTF EMA is warm (1h EMA 200 = 200 hours, so at 5m we need at least 2400 bars)
        if start_idx >= len(ltf_df) - 50:
            start_idx = Config.LONG_EMA + 20
            
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
                    stop_loss = self.risk.update_trailing_stop(entry_price, highest_price, stop_loss, "LONG")
                    
                    # Check Stop Loss
                    if current_low <= stop_loss:
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
                            'exit_reason': 'STOP_LOSS'
                        })
                        in_position = False
                        position_size = 0.0
                        
                    # Check Take Profit
                    elif current_high >= take_profit:
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
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        in_position = False
                        position_size = 0.0
                        
                elif position_side == "SHORT":
                    # For short, we track lowest price reached
                    lowest_price = min(lowest_price, current_low)
                    stop_loss = self.risk.update_trailing_stop(entry_price, lowest_price, stop_loss, "SHORT")
                    
                    # Check Stop Loss
                    if current_high >= stop_loss:
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
                            'exit_reason': 'STOP_LOSS'
                        })
                        in_position = False
                        position_size = 0.0
                        
                    # Check Take Profit
                    elif current_low <= take_profit:
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
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        in_position = False
                        position_size = 0.0
                        
            # --- EVALUATE NEW ENTRIES ---
            else:
                signal, signal_meta = self.strategy.generate_signal(sub_htf, sub_ltf)
                
                if signal in ("BUY", "SELL"):
                    # Check ML Confirmation Layer if enabled
                    is_confirmed = True
                    ml_prob = 1.0
                    if self.ml:
                        is_confirmed, ml_prob = self.ml.confirm_signal(sub_ltf, signal)
                        
                    if not is_confirmed:
                        # ML filtered the trade
                        continue
                        
                    # Calculate Stop Loss and Take Profit
                    sl = signal_meta['stop_loss']
                    tp = signal_meta['take_profit']
                    
                    if sl is None or tp is None:
                        continue
                        
                    entry_price = current_close
                    entry_time = ltf_time
                    stop_loss = sl
                    take_profit = tp
                    highest_price = entry_price
                    lowest_price = entry_price
                    
                    # Calculate dynamic position size based on risk percent
                    position_size = self.risk.calculate_position_size(self.balance, entry_price, stop_loss)
                    
                    if position_size <= 0.0:
                        continue
                        
                    if signal == "BUY":
                        position_side = "LONG"
                        fee = position_size * entry_price * fee_rate
                        self.balance -= (position_size * entry_price) + fee
                        in_position = True
                    elif signal == "SELL":
                        position_side = "SHORT"
                        # For short spot simulation, we assume margin-trading/contracts or simply track quote balance
                        fee = position_size * entry_price * fee_rate
                        self.balance -= fee  # Deduct fee, balance changes on exit
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
                'exit_reason': 'FORCE_CLOSE_END'
            })
            
        # 3. Calculate Performance Metrics
        return self.calculate_metrics(initial_balance)

    def calculate_metrics(self, initial_balance):
        if not self.trades:
            return {
                'initial_balance': initial_balance,
                'final_balance': self.balance,
                'total_return_pct': 0.0,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'profit_factor': 0.0
            }
            
        trade_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
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
        
        # Annualized Sharpe (simplified for active periods)
        if std_return > 0:
            # Assuming 5-minute candles, 288 candles per day, 365 days a year
            sharpe_ratio = (mean_return / std_return) * np.sqrt(288 * 365)
        else:
            sharpe_ratio = 0.0
            
        final_return_pct = ((self.balance - initial_balance) / initial_balance) * 100.0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': self.balance,
            'total_return_pct': final_return_pct,
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor
        }

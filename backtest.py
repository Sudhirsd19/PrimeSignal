import asyncio
import sys
from config import Config
from execution.execution_engine import ExecutionEngine
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
from strategies.indicators import prepare_dataframe
from ml.confirmation import MLSignalConfirmator
from backtester.backtester import BacktestEngine

async def main():
    print("====================================================")
    print("STARTING PRIMESIGNAL INSTITUTIONAL BACKTEST RUN")
    print("====================================================")
    
    # 1. Connect to exchange to fetch historical data
    execution = ExecutionEngine()
    
    import os
    import json
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    htf_file = os.path.join(_base_dir, "htf_data.json")
    ltf_file = os.path.join(_base_dir, "ltf_data.json")
    
    if os.path.exists(htf_file) and os.path.exists(ltf_file):
        print("[DATA] Loading historical data from local cache...")
        with open(htf_file, 'r') as f:
            htf_ohlcv = json.load(f)
        with open(ltf_file, 'r') as f:
            ltf_ohlcv = json.load(f)
        await execution.close()
    else:
        # Fetch HTF (1h) history (Binance max limit is 1000)
        htf_limit = 1000
        print(f"[DATA] Fetching last {htf_limit} candles for HTF ({Config.SYMBOL} @ {Config.HTF_TIMEFRAME})...")
        htf_ohlcv = await execution.fetch_ohlcv(
            symbol=Config.SYMBOL, 
            timeframe=Config.HTF_TIMEFRAME, 
            limit=htf_limit
        )
        
        # Fetch LTF (5m) history (1000 limit)
        ltf_limit = 1000
        print(f"[DATA] Fetching last {ltf_limit} candles for LTF ({Config.SYMBOL} @ {Config.LTF_TIMEFRAME})...")
        ltf_ohlcv = await execution.fetch_ohlcv(
            symbol=Config.SYMBOL, 
            timeframe=Config.LTF_TIMEFRAME, 
            limit=ltf_limit
        )
        await execution.close()
    
    if not htf_ohlcv or not ltf_ohlcv:
        print("ERROR: Failed to fetch historical data from Binance.")
        return
        
    print(f"[DATA] Received {len(htf_ohlcv)} HTF candles and {len(ltf_ohlcv)} LTF candles.")

    # 2. Setup strategy and ML components
    strategy = MultiTimeframeSMCStrategy()
    ml_confirmator = MLSignalConfirmator()
    
    # Split: Use first 30% of candles for training, remaining 70% for backtesting
    split_idx = 300
    
    # Warm up and train ML model on the training segment
    print(f"\n[ML] Training confirmation classifier on warm-up data (0 to {split_idx})...")
    warmup_ltf_candles = ltf_ohlcv[:split_idx]
    warmup_df = prepare_dataframe(warmup_ltf_candles)
    
    trained = ml_confirmator.train(warmup_df)
    if trained:
        print("[ML] Confirmation model trained and active.")
    else:
        print("[ML] WARNING: Confirmation model training failed. Proceeding without ML validation.")
        ml_confirmator = None

    # 3. Setup Backtest Engine
    # Backtest on the remaining out-of-sample candles
    test_ltf_candles = ltf_ohlcv[split_idx:]
    
    test_ltf_df = prepare_dataframe(test_ltf_candles)
    if len(test_ltf_df) == 0:
        print("ERROR: No test candles found after splitting.")
        return
        
    initial_capital = 10000.0
    
    # 1. Backtest WITH ML Confirmation
    print("\n[BACKTEST] Running simulation WITH ML Confirmation Filter...")
    backtester_ml = BacktestEngine(strategy, ml_confirmator=ml_confirmator)
    metrics_ml = backtester_ml.run(htf_ohlcv, test_ltf_candles, initial_balance=initial_capital)
    
    # 2. Backtest WITHOUT ML Confirmation (Raw SMC Only)
    print("\n[BACKTEST] Running simulation WITHOUT ML Confirmation (Raw SMC Only)...")
    backtester_raw = BacktestEngine(strategy, ml_confirmator=None)
    metrics_raw = backtester_raw.run(htf_ohlcv, test_ltf_candles, initial_balance=initial_capital)
    
    if not metrics_ml or not metrics_raw:
        return
        
    # 4. Print Comparative Summary Report
    print("\n====================================================")
    print("PRIMESIGNAL COMPARATIVE PERFORMANCE REPORT")
    print("====================================================")
    print(f"Strategy Name        : {strategy.name}")
    print(f"Trading Symbol       : {Config.SYMBOL}")
    print(f"Timeframes           : LTF: {Config.LTF_TIMEFRAME} | HTF: {Config.HTF_TIMEFRAME}")
    print("----------------------------------------------------")
    print(f"Metric               | SMC + ML Filter  | Raw SMC Only")
    print("----------------------------------------------------")
    print(f"Initial Capital      | {metrics_ml['initial_balance']:.2f} USDT     | {metrics_raw['initial_balance']:.2f} USDT")
    print(f"Final Account Value  | {metrics_ml['final_balance']:.2f} USDT     | {metrics_raw['final_balance']:.2f} USDT")
    print(f"Total Return         | {metrics_ml['total_return_pct']:+.2f}%           | {metrics_raw['total_return_pct']:+.2f}%")
    print(f"Total Trades         | {metrics_ml['total_trades']}                | {metrics_raw['total_trades']}")
    print(f"Wins / Losses        | {metrics_ml['wins']}W / {metrics_ml['losses']}L          | {metrics_raw['wins']}W / {metrics_raw['losses']}L")
    print(f"Win Rate             | {metrics_ml['win_rate']:.2f}%           | {metrics_raw['win_rate']:.2f}%")
    print(f"Profit Factor        | {metrics_ml['profit_factor']:.2f}             | {metrics_raw['profit_factor']:.2f}")
    print(f"Max Peak Drawdown    | {metrics_ml['max_drawdown_pct']:.2f}%           | {metrics_raw['max_drawdown_pct']:.2f}%")
    print(f"Annualized Sharpe    | {metrics_ml['sharpe_ratio']:.2f}            | {metrics_raw['sharpe_ratio']:.2f}")
    print("====================================================")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

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
        print("[DATA] Loading 30-day historical data from local cache...")
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
        if htf_ohlcv is None:
            print("ERROR: Failed to fetch historical data")
            await execution.close()
            return
        
        # Fetch LTF (5m) history (1000 limit)
        ltf_limit = 1000
        print(f"[DATA] Fetching last {ltf_limit} candles for LTF ({Config.SYMBOL} @ {Config.LTF_TIMEFRAME})...")
        ltf_ohlcv = await execution.fetch_ohlcv(
            symbol=Config.SYMBOL, 
            timeframe=Config.LTF_TIMEFRAME, 
            limit=ltf_limit
        )
        if ltf_ohlcv is None:
            print("ERROR: Failed to fetch historical data")
            await execution.close()
            return
        await execution.close()
    
    if not htf_ohlcv or not ltf_ohlcv:
        print("ERROR: Failed to fetch historical data from Binance.")
        return

    # BUG-10 FIX: Only resample when LTF data is 5m (300,000 ms interval).
    # If data is already 15m (900,000 ms), skip — resampling identical data
    # creates a timestamp mismatch between LTF and HTF alignment in the backtest loop.
    if Config.LTF_TIMEFRAME == "15m":
        is_5m_data = len(ltf_ohlcv) >= 2 and (ltf_ohlcv[1][0] - ltf_ohlcv[0][0]) == 300_000
        if is_5m_data:
            df = prepare_dataframe(ltf_ohlcv)
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            df_15m['timestamp'] = df_15m.index.astype('int64') // 1000000
            ltf_ohlcv = df_15m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
            print(f"[DATA] Resampled 5m → 15m: {len(ltf_ohlcv)} candles")
        else:
            print(f"[DATA] LTF data already at 15m — skipping resample.")
        
    print(f"[DATA] Prepared {len(htf_ohlcv)} HTF ({Config.HTF_TIMEFRAME}) candles and {len(ltf_ohlcv)} LTF ({Config.LTF_TIMEFRAME}) candles.")

    # 2. Setup strategy and ML components
    strategy = MultiTimeframeSMCStrategy()
    ml_confirmator = MLSignalConfirmator()
    
    # BUG-03 FIX: Increased from 10% to 30% for ML warmup.
    # With 1000 LTF bars: 10% ≈ 25h of training; 30% ≈ 75h — much more representative.
    # For production use, fetch 90+ days of data to get 864+ warmup bars (≈9 days).
    split_idx = int(len(ltf_ohlcv) * 0.30)
    
    # Warm up and train ML model on the training segment
    print(f"\n[ML] Training confirmation classifier on warm-up data (0 to {split_idx})...")
    warmup_ltf_candles = ltf_ohlcv[:split_idx]
    warmup_df = prepare_dataframe(warmup_ltf_candles)
    
    trained = ml_confirmator.train(warmup_df)
    if trained:
        print("[ML] Confirmation model trained and active.")
    else:
        print("[ML] WARNING: ML confirmation model training failed — trading without ML filter.")
        ml_confirmator = None

    # 3. Setup Backtest Engine
    # Backtest on the remaining out-of-sample candles (Full 30-day period)
    test_ltf_candles = ltf_ohlcv[split_idx:]
    
    test_ltf_df = prepare_dataframe(test_ltf_candles)
    if len(test_ltf_df) == 0:
        print("ERROR: No test candles found after splitting.")
        return
        
    bars_per_day = 96.0 if Config.LTF_TIMEFRAME == "15m" else 288.0
    duration_days = len(test_ltf_df) / bars_per_day
    print(f"\n[BACKTEST] Testing out-of-sample period: {len(test_ltf_df)} candles ({duration_days:.1f} Days @ {Config.LTF_TIMEFRAME})...")
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
    print("----------------------------------------------------")
    print(f"Strict Win Rate      | {metrics_ml['strict_win_rate']:.2f}%           | {metrics_raw['strict_win_rate']:.2f}%")
    print(f"Strict Profit Factor | {metrics_ml['strict_pf']:.2f}             | {metrics_raw['strict_pf']:.2f}")
    print(f"Relaxed Win Rate     | {metrics_ml['relaxed_win_rate']:.2f}%           | {metrics_raw['relaxed_win_rate']:.2f}%")
    print(f"Relaxed Profit Factor| {metrics_ml['relaxed_pf']:.2f}             | {metrics_raw['relaxed_pf']:.2f}")
    print("====================================================")

    # FIX-E: Walk-Forward Validation — split test period into First Half vs Second Half.
    # If both halves are profitable → strategy is robust across different market regimes.
    # If only the first half passes → strategy may be overfit to a single regime.
    print("\n====================================================")
    print("WALK-FORWARD VALIDATION (Out-of-Sample Stability)")
    print("====================================================")
    mid_idx = len(test_ltf_candles) // 2
    wf_half1 = test_ltf_candles[:mid_idx]
    wf_half2 = test_ltf_candles[mid_idx:]

    bars_per_day = 96.0 if Config.LTF_TIMEFRAME == "15m" else 288.0
    h1_days = len(wf_half1) / bars_per_day
    h2_days = len(wf_half2) / bars_per_day
    print(f"First Half:  {len(wf_half1)} bars ({h1_days:.1f} days)")
    print(f"Second Half: {len(wf_half2)} bars ({h2_days:.1f} days)")
    print("----------------------------------------------------")

    wf_results = {}
    for label, half_candles in [("First Half ", wf_half1), ("Second Half", wf_half2)]:
        if len(half_candles) < 50:
            print(f"[WF] {label}: Too few candles to backtest.")
            continue
        wf_engine = BacktestEngine(strategy, ml_confirmator=ml_confirmator)
        wf_metrics = wf_engine.run(htf_ohlcv, half_candles, initial_balance=initial_capital)
        if wf_metrics:
            wf_results[label] = wf_metrics

    if len(wf_results) == 2:
        h1 = wf_results["First Half "]
        h2 = wf_results["Second Half"]

        def _pf_str(pf):
            return f"{pf:.2f}" if pf != float('inf') else "∞"

        print(f"{'Metric':<22} | {'First Half':>12} | {'Second Half':>12}")
        print("-" * 52)
        print(f"{'Total Return':<22} | {h1['total_return_pct']:>+11.2f}% | {h2['total_return_pct']:>+11.2f}%")
        print(f"{'Total Trades':<22} | {h1['total_trades']:>12} | {h2['total_trades']:>12}")
        print(f"{'Win Rate':<22} | {h1['win_rate']:>11.2f}% | {h2['win_rate']:>11.2f}%")
        print(f"{'Profit Factor':<22} | {_pf_str(h1['profit_factor']):>12} | {_pf_str(h2['profit_factor']):>12}")
        print(f"{'Max Drawdown':<22} | {h1['max_drawdown_pct']:>11.2f}% | {h2['max_drawdown_pct']:>11.2f}%")
        print(f"{'Sharpe Ratio':<22} | {h1['sharpe_ratio']:>12.2f} | {h2['sharpe_ratio']:>12.2f}")
        print("-" * 52)
        # Stability verdict
        both_profitable = h1['total_return_pct'] > 0 and h2['total_return_pct'] > 0
        both_pf_above_1 = h1['profit_factor'] > 1.0 and h2['profit_factor'] > 1.0
        if both_profitable and both_pf_above_1:
            print("✅ WALK-FORWARD PASS: Strategy profitable in BOTH halves — consistent edge.")
        elif both_profitable:
            print("⚠️  PARTIAL PASS: Both halves profitable but Profit Factor < 1 in one half.")
        else:
            print("❌ WALK-FORWARD FAIL: Strategy not profitable in both halves — possible overfit.")
            print("   → Increase training data, tighten filters, or reduce relaxed mode entries.")
    print("====================================================")

if __name__ == "__main__":
    if sys.platform == 'win32' and sys.version_info < (3, 12):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())


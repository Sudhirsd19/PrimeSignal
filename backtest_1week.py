import json
import os
import datetime
import pandas as pd
from config import Config
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
from strategies.indicators import prepare_dataframe
from ml.confirmation import MLSignalConfirmator
from backtester.backtester import BacktestEngine

def run_1week_backtest():
    print("====================================================")
    print("      PRIMESIGNAL 1-WEEK (7-DAY) BACKTEST RUN       ")
    print("====================================================")
    
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    htf_file = os.path.join(_base_dir, "htf_data.json")
    ltf_file = os.path.join(_base_dir, "ltf_data.json")
    
    if not os.path.exists(htf_file) or not os.path.exists(ltf_file):
        print(f"ERROR: Historical data files not found at:\n  - {htf_file}\n  - {ltf_file}")
        print("Run `python fetch_30d.py` first to download historical candle data.")
        return

    with open(htf_file, 'r') as f:
        htf_ohlcv = json.load(f)
    with open(ltf_file, 'r') as f:
        ltf_ohlcv = json.load(f)

    if not htf_ohlcv or not ltf_ohlcv:
        print("ERROR: Historical data files are empty.")
        return

    # If LTF is 15m and source data is 5m (300000ms delta), resample accurately in UTC
    if Config.LTF_TIMEFRAME == "15m" and len(ltf_ohlcv) >= 2:
        if (ltf_ohlcv[1][0] - ltf_ohlcv[0][0]) == 300000:
            df = prepare_dataframe(ltf_ohlcv)
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            ltf_ohlcv = [
                [
                    int(pd.to_datetime(str(ts), utc=True).timestamp() * 1000),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ]
                for ts, row in df_15m.iterrows()
            ]

    # Calculate bars per day dynamically from LTF timeframe setting
    tf = Config.LTF_TIMEFRAME.lower()
    if tf.endswith('m'):
        tf_minutes = max(1, int(tf[:-1]))
    elif tf.endswith('h'):
        tf_minutes = max(1, int(tf[:-1]) * 60)
    elif tf.endswith('d'):
        tf_minutes = max(1, int(tf[:-1]) * 1440)
    else:
        tf_minutes = 15

    bars_per_day = max(1, 1440 // tf_minutes)
    one_week_bars = bars_per_day * 7 # e.g. 672 bars for 15m
    total_ltf = len(ltf_ohlcv)

    if total_ltf < 50:
        print(f"ERROR: Insufficient LTF data ({total_ltf} bars). Need at least 50 bars.")
        return

    # Account for BacktestEngine warmup requirement (at least 100 bars)
    warmup_buffer = min(100, max(0, total_ltf - one_week_bars))
    test_slice_len = min(total_ltf, one_week_bars + warmup_buffer)
    
    test_ltf_candles = ltf_ohlcv[-test_slice_len:]
    warmup_ltf_candles = ltf_ohlcv[:-one_week_bars] if total_ltf > one_week_bars else ltf_ohlcv[:max(30, int(total_ltf * 0.3))]

    start_ts = test_ltf_candles[warmup_buffer][0] / 1000 if len(test_ltf_candles) > warmup_buffer else test_ltf_candles[0][0] / 1000
    end_ts = test_ltf_candles[-1][0] / 1000
    start_dt = datetime.datetime.fromtimestamp(start_ts, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    end_dt = datetime.datetime.fromtimestamp(end_ts, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    print(f"\n[TIMEFRAME] Testing 1 Week (7 Days):")
    print(f"  • Start: {start_dt}")
    print(f"  • End:   {end_dt}")
    print(f"  • Total Evaluation Bars: {len(test_ltf_candles)} (incl. {warmup_buffer} warmup bars)")
    print(f"  • ML Warm-up Training Bars: {len(warmup_ltf_candles)}")

    # 1. Train ML Model on Warm-up Data
    strategy = MultiTimeframeSMCStrategy()
    ml_confirmator = MLSignalConfirmator()
    if len(warmup_ltf_candles) >= 30:
        warmup_df = prepare_dataframe(warmup_ltf_candles)
        trained = ml_confirmator.train(warmup_df)
        if trained:
            print("[ML] Confirmation classifier trained on warm-up data.")
        else:
            print("[ML] Confirmation model training skipped/failed.")
            ml_confirmator = None
    else:
        print("[ML] Insufficient data for ML warm-up training — running without ML filter.")
        ml_confirmator = None

    initial_capital = 10000.0

    # 2. Run simulation with ML Filter
    print("\n[RUNNING] 1-Week Backtest with SMC + ML Filter...")
    backtester_ml = BacktestEngine(strategy, ml_confirmator=ml_confirmator)
    metrics_ml = backtester_ml.run(htf_ohlcv, test_ltf_candles, initial_balance=initial_capital)

    # 3. Run simulation with Raw SMC Only
    print("\n[RUNNING] 1-Week Backtest with Raw SMC Only...")
    backtester_raw = BacktestEngine(strategy, ml_confirmator=None)
    metrics_raw = backtester_raw.run(htf_ohlcv, test_ltf_candles, initial_balance=initial_capital)

    if not metrics_ml or not metrics_raw:
        print("\nERROR: Backtest failed to compute metrics due to insufficient aligned data.")
        return

    # 4. Display Formatted Output
    wl_ml_str = f"{metrics_ml['wins']}W / {metrics_ml['losses']}L"
    wl_raw_str = f"{metrics_raw['wins']}W / {metrics_raw['losses']}L"

    print("\n" + "="*60)
    print("          PRIMESIGNAL 1-WEEK PERFORMANCE REPORT         ")
    print("="*60)
    print(f"Asset / Pair         : {Config.SYMBOL}")
    print(f"Test Duration        : 7 Days ({start_dt} to {end_dt})")
    print(f"Timeframes           : LTF: {Config.LTF_TIMEFRAME} | HTF: {Config.HTF_TIMEFRAME}")
    print("-"*60)
    print(f"{'Metric':<22} | {'SMC + ML Filter':<16} | {'Raw SMC Only':<16}")
    print("-"*60)
    print(f"{'Initial Capital':<22} | {metrics_ml['initial_balance']:>11.2f} USDT | {metrics_raw['initial_balance']:>11.2f} USDT")
    print(f"{'Final Balance':<22} | {metrics_ml['final_balance']:>11.2f} USDT | {metrics_raw['final_balance']:>11.2f} USDT")
    print(f"{'Total Return':<22} | {metrics_ml['total_return_pct']:>+11.2f}% | {metrics_raw['total_return_pct']:>+11.2f}%")
    print(f"{'Total Trades':<22} | {metrics_ml['total_trades']:>16} | {metrics_raw['total_trades']:>16}")
    print(f"{'Wins / Losses':<22} | {wl_ml_str:>16} | {wl_raw_str:>16}")
    print(f"{'Win Rate':<22} | {metrics_ml['win_rate']:>11.2f}% | {metrics_raw['win_rate']:>11.2f}%")
    print(f"{'Profit Factor':<22} | {metrics_ml['profit_factor']:>16.2f} | {metrics_raw['profit_factor']:>16.2f}")
    print(f"{'Max Peak Drawdown':<22} | {metrics_ml['max_drawdown_pct']:>11.2f}% | {metrics_raw['max_drawdown_pct']:>11.2f}%")
    print(f"{'Sharpe Ratio':<22} | {metrics_ml['sharpe_ratio']:>16.2f} | {metrics_raw['sharpe_ratio']:>16.2f}")
    print("-"*60)

    # Detailed Trade Log
    print("\n" + "="*60)
    print("              EXECUTED TRADES LOG (1 WEEK)              ")
    print("="*60)
    trades_to_show = backtester_ml.trades if backtester_ml.trades else backtester_raw.trades
    tag = "(SMC + ML)" if backtester_ml.trades else "(Raw SMC)"
    if not trades_to_show:
        print("No trades triggered during this 1-week period under tested market conditions.")
    else:
        print(f"Showing executed trades {tag}:")
        for idx, t in enumerate(trades_to_show, 1):
            e_time = t.get('entry_time', '')
            x_time = t.get('exit_time', '')
            side = t.get('side', '')
            entry = t.get('entry_price', 0.0)
            exit_p = t.get('exit_price', 0.0)
            pnl = t.get('pnl', 0.0)
            pnl_pct = t.get('pnl_pct', 0.0)
            reason = t.get('exit_reason', '')
            mode = t.get('mode', '')
            print(f"Trade #{idx} [{side}] ({mode}) | In: {e_time} @ {entry:.2f} -> Out: {x_time} @ {exit_p:.2f}")
            print(f"   PnL: {pnl:+.2f} USDT ({pnl_pct:+.2f}%) | Exit: {reason}")
            print("-" * 60)
    print("="*60)

if __name__ == "__main__":
    run_1week_backtest()


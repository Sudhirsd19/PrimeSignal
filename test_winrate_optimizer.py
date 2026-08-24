import json
import os
import pandas as pd
import numpy as np
from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks, detect_structure
from ml.confirmation import MLSignalConfirmator

def run_simulation(tsl_activation_r=0.6, rr_ratio=1.5, ml_threshold=0.60, min_adx=20.0):
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    htf_file = os.path.join(_base_dir, "htf_data.json")
    ltf_file = os.path.join(_base_dir, "ltf_data.json")
    
    with open(htf_file, 'r') as f:
        htf_ohlcv = json.load(f)
    with open(ltf_file, 'r') as f:
        ltf_ohlcv = json.load(f)

    if len(ltf_ohlcv) >= 2 and (ltf_ohlcv[1][0] - ltf_ohlcv[0][0]) == 300000:
        df = prepare_dataframe(ltf_ohlcv)
        df_15m = df.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        ltf_ohlcv = [
            [int(ts.timestamp() * 1000), float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume'])]
            for ts, row in df_15m.iterrows()
        ]

    htf_df = prepare_dataframe(htf_ohlcv)
    ltf_df = prepare_dataframe(ltf_ohlcv)

    # Train ML
    split_idx = int(len(ltf_ohlcv) * 0.15)
    warmup_df = prepare_dataframe(ltf_ohlcv[:split_idx])
    ml = MLSignalConfirmator()
    ml.train(warmup_df)

    test_ltf_df = prepare_dataframe(ltf_ohlcv[split_idx:])
    ltf_atr = calculate_atr(test_ltf_df, 14)

    # Simulation state
    balance = 10000.0
    in_position = False
    position_side = None
    entry_price = 0.0
    stop_loss = 0.0
    initial_sl = 0.0
    take_profit = 0.0
    tp1 = 0.0
    tp1_taken = False
    highest_price = 0.0
    lowest_price = 999999.0
    position_size = 0.0
    trades = []
    fee_rate = 0.00075

    from strategies.multi_timeframe import MultiTimeframeSMCStrategy
    strategy = MultiTimeframeSMCStrategy()

    for i in range(100, len(test_ltf_df)):
        ltf_time = test_ltf_df.index[i]
        curr_candle = test_ltf_df.iloc[i]
        sub_ltf = test_ltf_df.iloc[max(0, i-250):i+1]
        sub_htf = htf_df[htf_df.index < ltf_time].iloc[-250:]
        
        curr_close = curr_candle['close']
        curr_high = curr_candle['high']
        curr_low = curr_candle['low']
        curr_atr = ltf_atr.iloc[i]

        if in_position:
            if position_side == "LONG":
                highest_price = max(highest_price, curr_high)
                risk_dist = abs(entry_price - initial_sl)
                
                # Fast Break-even move at tsl_activation_r (e.g. 0.6R)
                if highest_price >= entry_price + (risk_dist * tsl_activation_r):
                    be_level = entry_price * 1.0015
                    stop_loss = max(stop_loss, be_level)

                # TP1 scale out or full TP
                if curr_high >= take_profit:
                    pnl = position_size * (take_profit - entry_price) - (position_size * take_profit * fee_rate)
                    balance += position_size * take_profit - (position_size * take_profit * fee_rate)
                    trades.append({'pnl': pnl, 'side': 'LONG', 'reason': 'TP'})
                    in_position = False
                elif curr_low <= stop_loss:
                    exit_p = min(stop_loss, curr_candle['open'])
                    pnl = position_size * (exit_p - entry_price) - (position_size * exit_p * fee_rate)
                    balance += position_size * exit_p - (position_size * exit_p * fee_rate)
                    trades.append({'pnl': pnl, 'side': 'LONG', 'reason': 'SL/BE'})
                    in_position = False

            elif position_side == "SHORT":
                lowest_price = min(lowest_price, curr_low)
                risk_dist = abs(entry_price - initial_sl)
                
                if lowest_price <= entry_price - (risk_dist * tsl_activation_r):
                    be_level = entry_price * 0.9985
                    stop_loss = min(stop_loss, be_level)

                if curr_low <= take_profit:
                    pnl = position_size * (entry_price - take_profit) - (position_size * take_profit * fee_rate)
                    balance += pnl
                    trades.append({'pnl': pnl, 'side': 'SHORT', 'reason': 'TP'})
                    in_position = False
                elif curr_high >= stop_loss:
                    exit_p = max(stop_loss, curr_candle['open'])
                    pnl = position_size * (entry_price - exit_p) - (position_size * exit_p * fee_rate)
                    balance += pnl
                    trades.append({'pnl': pnl, 'side': 'SHORT', 'reason': 'SL/BE'})
                    in_position = False

        else:
            signal, meta = strategy.generate_signal(sub_htf, sub_ltf, relaxed=False)
            if signal in ("BUY", "SELL"):
                # ML filter
                prob = ml.predict_bias(sub_ltf)
                if signal == "BUY" and prob < ml_threshold:
                    continue
                if signal == "SELL" and (1.0 - prob) < ml_threshold:
                    continue

                sl = meta.get('stop_loss')
                tp = meta.get('take_profit')
                if not sl or not tp:
                    continue

                entry_price = curr_close
                stop_loss = sl
                initial_sl = sl
                take_profit = tp
                highest_price = entry_price
                lowest_price = entry_price
                
                # Dynamic risk sizing
                risk_per_trade = balance * 0.015
                dist = abs(entry_price - stop_loss)
                position_size = min(risk_per_trade / dist, balance / entry_price)
                in_position = True
                position_side = "LONG" if signal == "BUY" else "SHORT"

    if not trades:
        return 0, 0, 0.0, balance

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / len(trades) * 100
    print(f"Params: TSL_R={tsl_activation_r} | ML_TH={ml_threshold} => Trades: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)} | WinRate: {wr:.2f}% | Final: {balance:.2f}")
    return len(trades), len(wins), wr, balance

if __name__ == "__main__":
    for tsl in [0.4, 0.5, 0.6, 0.8]:
        for ml_th in [0.55, 0.60, 0.65]:
            run_simulation(tsl_activation_r=tsl, ml_threshold=ml_th)

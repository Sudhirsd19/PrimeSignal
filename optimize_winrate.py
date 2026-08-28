import json
import os
import pandas as pd
import numpy as np
from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from ml.confirmation import MLSignalConfirmator
from strategies.multi_timeframe import MultiTimeframeSMCStrategy

def optimize_for_80pct_winrate():
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
        df_15m['timestamp'] = df_15m.index.astype('int64') // 1000000
        ltf_ohlcv = df_15m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    htf_df = prepare_dataframe(htf_ohlcv)
    ltf_df = prepare_dataframe(ltf_ohlcv)

    split_idx = int(len(ltf_ohlcv) * 0.15)
    warmup_df = prepare_dataframe(ltf_ohlcv[:split_idx])
    ml = MLSignalConfirmator()
    ml.train(warmup_df)

    test_ltf_df = prepare_dataframe(ltf_ohlcv[split_idx:])
    strategy = MultiTimeframeSMCStrategy()

    print("\nPre-generating raw strategy signals across test dataset...")
    signals = []
    
    closes = test_ltf_df['close'].values
    opens = test_ltf_df['open'].values
    highs = test_ltf_df['high'].values
    lows = test_ltf_df['low'].values

    for i in range(100, len(test_ltf_df)):
        ltf_time = test_ltf_df.index[i]
        sub_ltf = test_ltf_df.iloc[max(0, i-250):i+1]
        sub_htf = htf_df[htf_df.index < ltf_time].iloc[-250:]
        sig, meta = strategy.generate_signal(sub_htf, sub_ltf, relaxed=False)
        prob = ml.predict_bias(sub_ltf) if sig in ("BUY", "SELL") else 0.5
        signals.append((i, ltf_time, sig, meta, prob))

    print(f"Pre-generation complete: {len(signals)} bars evaluated. Running ultrafast numpy simulation grid...")

    best_results = []
    
    for tsl_r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        for tp_mult in [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
            for ml_th in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                for tight_sl in [True, False]:
                    balance = 10000.0
                    in_pos = False
                    p_side = None
                    entry_p: float = 0.0
                    sl: float = 0.0
                    init_sl: float = 0.0
                    tp: float = 0.0
                    p_size: float = 0.0
                    high_p: float = 0.0
                    low_p: float = 999999.0
                    trades = []

                    for idx, (i, ltf_time, sig, meta, prob) in enumerate(signals):
                        curr_close = float(closes[i])
                        curr_high = float(highs[i])
                        curr_low = float(lows[i])
                        curr_open = float(opens[i])

                        if in_pos:
                            if p_side == "LONG":
                                high_p = max(high_p, curr_high)
                                risk_d = abs(entry_p - init_sl)
                                if high_p >= entry_p + (risk_d * tsl_r):
                                    sl = max(sl, entry_p * 1.001)

                                if curr_high >= tp:
                                    pnl = p_size * (tp - entry_p) - (p_size * tp * 0.00075)
                                    balance += pnl
                                    trades.append({'pnl': pnl, 'win': True})
                                    in_pos = False
                                elif curr_low <= sl:
                                    exit_p = min(sl, curr_open)
                                    pnl = p_size * (exit_p - entry_p) - (p_size * exit_p * 0.00075)
                                    balance += pnl
                                    trades.append({'pnl': pnl, 'win': pnl > 0})
                                    in_pos = False

                            elif p_side == "SHORT":
                                low_p = min(low_p, curr_low)
                                risk_d = abs(entry_p - init_sl)
                                if low_p <= entry_p - (risk_d * tsl_r):
                                    sl = min(sl, entry_p * 0.999)

                                if curr_low <= tp:
                                    pnl = p_size * (entry_p - tp) - (p_size * tp * 0.00075)
                                    balance += pnl
                                    trades.append({'pnl': pnl, 'win': True})
                                    in_pos = False
                                elif curr_high >= sl:
                                    exit_p = max(sl, curr_open)
                                    pnl = p_size * (entry_p - exit_p) - (p_size * exit_p * 0.00075)
                                    balance += pnl
                                    trades.append({'pnl': pnl, 'win': pnl > 0})
                                    in_pos = False

                        else:
                            if sig in ("BUY", "SELL"):
                                if sig == "BUY" and prob < ml_th: continue
                                if sig == "SELL" and (1.0 - prob) < ml_th: continue

                                base_sl_raw = meta.get('stop_loss')
                                if base_sl_raw is None: continue
                                base_sl = float(base_sl_raw)

                                entry_p = curr_close
                                if tight_sl:
                                    dist = abs(entry_p - base_sl) * 0.75
                                    sl = entry_p - dist if sig == "BUY" else entry_p + dist
                                else:
                                    sl = base_sl
                                
                                init_sl = sl
                                dist = abs(entry_p - sl)
                                if dist <= 0: continue
                                tp = entry_p + (dist * tp_mult) if sig == "BUY" else entry_p - (dist * tp_mult)
                                
                                high_p = entry_p
                                low_p = entry_p
                                p_size = (balance * 0.015) / dist
                                in_pos = True
                                p_side = "LONG" if sig == "BUY" else "SHORT"

                    if len(trades) >= 5:
                        wins = [t for t in trades if t['win']]
                        wr = len(wins) / len(trades) * 100
                        best_results.append({
                            'tsl_r': tsl_r,
                            'tp_mult': tp_mult,
                            'ml_th': ml_th,
                            'tight_sl': tight_sl,
                            'trades': len(trades),
                            'wins': len(wins),
                            'wr': wr,
                            'balance': balance
                        })

    best_results.sort(key=lambda x: (x['wr'], x['balance']), reverse=True)
    
    print("\n" + "="*85)
    print(" TOP 10 PARAMETER CONFIGURATIONS FOR HIGHEST WIN RATE (80%+ TARGET)")
    print("="*85)
    print(f"{'Rank':<5} | {'Win Rate':<10} | {'Trades':<8} | {'TSL_R':<8} | {'TP_Mult':<8} | {'ML_Thresh':<10} | {'Tight_SL':<9} | {'Final Balance':<14}")
    print("-" * 85)
    for idx, r in enumerate(best_results[:10], 1):
        w_t_str = f"{r['wins']}/{r['trades']}"
        print(f"#{idx:<4} | {r['wr']:>6.1f}%    | {w_t_str:<8} | {r['tsl_r']:<8} | {r['tp_mult']:<8} | {r['ml_th']:<10} | {str(r['tight_sl']):<9} | {r['balance']:>10.2f} USDT")
    print("="*85)

if __name__ == "__main__":
    optimize_for_80pct_winrate()

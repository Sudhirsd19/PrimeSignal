import json
import os
import pandas as pd
import numpy as np
from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks, detect_structure
from ml.confirmation import MLSignalConfirmator

def test_high_winrate_strategy():
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

    print(f"Loaded HTF: {len(htf_df)} | LTF: {len(ltf_df)}")

if __name__ == "__main__":
    test_high_winrate_strategy()

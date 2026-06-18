import pandas as pd
import numpy as np

def calculate_ema(df, period, column='close'):
    """
    Calculates Exponential Moving Average (EMA).
    """
    if len(df) < period:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period=14, column='close'):
    """
    Calculates Relative Strength Index (RSI) using Wilder's smoothing.
    """
    if len(df) < period:
        return pd.Series([50.0] * len(df), index=df.index)
    
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    """
    Calculates Average True Range (ATR).
    """
    if len(df) < period:
        return pd.Series([0.0] * len(df), index=df.index)
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_vwap(df):
    """
    Calculates Volume Weighted Average Price (VWAP) with daily session reset.
    Resets at UTC midnight to match institutional VWAP calculation.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    volume = df['volume']

    # Group by UTC date for daily session reset
    date_groups = df.index.date

    cum_tp_vol = pd.Series(0.0, index=df.index)
    cum_vol = pd.Series(0.0, index=df.index)

    for date in set(date_groups):
        mask = date_groups == date
        cum_tp_vol[mask] = (typical_price[mask] * volume[mask]).cumsum()
        cum_vol[mask] = volume[mask].cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, 1e-9)
    return vwap

def prepare_dataframe(ohlcv_data):
    """
    Converts list of list OHLCV candles to a pandas DataFrame.
    Format: [[timestamp, open, high, low, close, volume], ...]
    """
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

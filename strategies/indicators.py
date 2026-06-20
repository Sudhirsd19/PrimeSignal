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

    for date in sorted(set(date_groups)):  # sorted() ensures chronological order for correct cumsum
        mask = date_groups == date
        cum_tp_vol[mask] = (typical_price[mask] * volume[mask]).cumsum()
        cum_vol[mask] = volume[mask].cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, 1e-9)
    return vwap

def calculate_adx(df, period=14):
    """
    Calculates Average Directional Index (ADX) and DMI using Wilder's smoothing.
    Returns DataFrame with columns: ['plus_di', 'minus_di', 'adx']
    """
    if len(df) < period:
        return pd.DataFrame({'plus_di': [0.0]*len(df), 'minus_di': [0.0]*len(df), 'adx': [0.0]*len(df)}, index=df.index)
        
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder's Smoothing: EMA with alpha=1/period
    smoothed_plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    smoothed_minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    smoothed_tr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr.replace(0, 1e-9))
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr.replace(0, 1e-9))
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return pd.DataFrame({'plus_di': plus_di, 'minus_di': minus_di, 'adx': adx})

def prepare_dataframe(ohlcv_data):
    """
    Converts list of list OHLCV candles to a pandas DataFrame.
    Format: [[timestamp, open, high, low, close, volume], ...]
    """
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

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

def detect_rsi_divergence(df, rsi_series, lookback=20):
    """
    Detects RSI divergence by comparing price swing points with RSI swing points.
    
    Bullish Divergence: Price makes a LOWER low, but RSI makes a HIGHER low.
    Bearish Divergence: Price makes a HIGHER high, but RSI makes a LOWER high.
    
    Args:
        df: DataFrame with OHLCV data
        rsi_series: pandas Series of RSI values (same index as df)
        lookback: number of bars to scan for swing points
        
    Returns:
        str or None: 'BULLISH', 'BEARISH', or None
    """
    if len(df) < lookback + 5 or len(rsi_series) < lookback + 5:
        return None
    
    # Use the completed candle window (exclude live candle at -1)
    end = len(df) - 1  # last completed = -2, but we use iloc indices
    start = max(0, end - lookback)
    
    price_lows = df['low'].iloc[start:end]
    price_highs = df['high'].iloc[start:end]
    rsi_window = rsi_series.iloc[start:end]
    
    if len(price_lows) < 10:
        return None
    
    # Find swing lows (local minima with 3-bar lookback/forward)
    swing_lows = []
    swing_highs = []
    for i in range(3, len(price_lows) - 3):
        # Swing low: lower than 3 bars on each side
        if price_lows.iloc[i] == price_lows.iloc[i-3:i+4].min():
            swing_lows.append(i)
        # Swing high: higher than 3 bars on each side
        if price_highs.iloc[i] == price_highs.iloc[i-3:i+4].max():
            swing_highs.append(i)
    
    # Check for Bullish Divergence (need at least 2 swing lows)
    if len(swing_lows) >= 2:
        latest_sw = swing_lows[-1]
        prev_sw = swing_lows[-2]
        # Price made lower low
        if price_lows.iloc[latest_sw] < price_lows.iloc[prev_sw]:
            # But RSI made higher low
            if rsi_window.iloc[latest_sw] > rsi_window.iloc[prev_sw]:
                return 'BULLISH'
    
    # Check for Bearish Divergence (need at least 2 swing highs)
    if len(swing_highs) >= 2:
        latest_sw = swing_highs[-1]
        prev_sw = swing_highs[-2]
        # Price made higher high
        if price_highs.iloc[latest_sw] > price_highs.iloc[prev_sw]:
            # But RSI made lower high
            if rsi_window.iloc[latest_sw] < rsi_window.iloc[prev_sw]:
                return 'BEARISH'
    
    return None


def prepare_dataframe(ohlcv_data):
    """
    Converts list of list OHLCV candles to a pandas DataFrame.
    Format: [[timestamp, open, high, low, close, volume], ...]
    """
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

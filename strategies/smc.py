import pandas as pd
import numpy as np

def detect_fvgs(df):
    """
    Detects Fair Value Gaps (FVGs) in the DataFrame.
    Returns:
        fvg_series: pandas Series of dicts containing FVG details or None.
    """
    fvg_list = [None] * len(df)
    
    # Needs at least 3 candles
    for i in range(2, len(df)):
        c_prev2 = df.iloc[i-2]
        c_prev = df.iloc[i-1]
        c_curr = df.iloc[i]
        
        # Bullish FVG: Low of candle i is greater than High of candle i-2
        if c_curr['low'] > c_prev2['high']:
            # Verify it's an impulsive expansion candle
            if c_prev['close'] > c_prev['open']:
                fvg_list[i] = {
                    'type': 'BULLISH',
                    'top': c_curr['low'],
                    'bottom': c_prev2['high'],
                    'mitigated': False,
                    'timestamp': df.index[i-1]
                }
                
        # Bearish FVG: High of candle i is less than Low of candle i-2
        elif c_curr['high'] < c_prev2['low']:
            if c_prev['close'] < c_prev['open']:
                fvg_list[i] = {
                    'type': 'BEARISH',
                    'top': c_prev2['low'],
                    'bottom': c_curr['high'],
                    'mitigated': False,
                    'timestamp': df.index[i-1]
                }
                
    return pd.Series(fvg_list, index=df.index)


def detect_order_blocks(df, lookback=50):
    """
    Identifies bullish and bearish order blocks.
    Bullish OB: Last bearish candle before a strong bullish impulsive move.
    Bearish OB: Last bullish candle before a strong bearish impulsive move.
    After detection, a mitigation pass checks if price has since traded
    through the OB zone, marking it as mitigated so stale OBs are filtered out.
    """
    ob_list = [None] * len(df)
    
    # Precompute rolling average body size to avoid recalculating inside the loop (massive speedup!)
    avg_bodies = abs(df['close'] - df['open']).rolling(14).mean()
    
    for i in range(5, len(df)):
        # Calculate displacement / momentum: candle size relative to Average True Range
        candle_body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        avg_body = avg_bodies.iloc[i]
        
        # If we have a strong bullish move
        if df.iloc[i]['close'] > df.iloc[i]['open'] and candle_body > 1.5 * avg_body:
            # Look back to find the last bearish candle
            for j in range(i-1, i-5, -1):
                if df.iloc[j]['close'] < df.iloc[j]['open']:
                    ob_list[i] = {
                        'type': 'BULLISH',
                        'top': max(df.iloc[j]['open'], df.iloc[j]['high']),
                        'bottom': df.iloc[j]['low'],
                        'mitigated': False,
                        'timestamp': df.index[j]
                    }
                    break
                    
        # If we have a strong bearish move
        elif df.iloc[i]['close'] < df.iloc[i]['open'] and candle_body > 1.5 * avg_body:
            # Look back to find the last bullish candle
            for j in range(i-1, i-5, -1):
                if df.iloc[j]['close'] > df.iloc[j]['open']:
                    ob_list[i] = {
                        'type': 'BEARISH',
                        'top': df.iloc[j]['high'],
                        'bottom': min(df.iloc[j]['open'], df.iloc[j]['low']),
                        'mitigated': False,
                        'timestamp': df.index[j]
                    }
                    break

    # --- MITIGATION PASS ---
    # An OB is 'mitigated' only when price has FULLY traded through the zone
    # (closed beyond the OB's far edge), meaning the imbalance is consumed.
    #
    # Bullish OB  : mitigated when a candle closes BELOW the OB bottom
    #               (price punched through the entire demand zone)
    # Bearish OB  : mitigated when a candle closes ABOVE the OB top
    #               (price punched through the entire supply zone)
    #
    # BUG FIX: Previously used ob['top']/ob['bottom'] as thresholds which
    # fired on the very NEXT candle, marking every OB immediately stale.
    for i in range(len(ob_list)):
        ob = ob_list[i]
        if ob is None:
            continue
        for k in range(i + 1, len(df)):
            candle = df.iloc[k]
            if ob['type'] == 'BULLISH':
                # Mitigated when price closes BELOW OB bottom (zone fully consumed)
                if candle['close'] < ob['bottom']:
                    ob_list[i] = dict(ob, mitigated=True)
                    break
            elif ob['type'] == 'BEARISH':
                # Mitigated when price closes ABOVE OB top (zone fully consumed)
                if candle['close'] > ob['top']:
                    ob_list[i] = dict(ob, mitigated=True)
                    break
                    
    return pd.Series(ob_list, index=df.index)


def detect_structure(df, period=5):
    """
    Identifies market structure changes (BOS / CHOCH) by finding swing points.
    A swing high has period lower highs on left and right.
    A swing low has period higher lows on left and right.
    """
    bos_list = [None] * len(df)
    choch_list = [None] * len(df)
    
    swing_highs = []
    swing_lows = []
    
    current_trend = 1  # 1 for bullish, -1 for bearish
    
    for i in range(period, len(df) - period):
        idx = df.index[i]
        highs = df['high'].iloc[i-period:i+period+1]
        lows = df['low'].iloc[i-period:i+period+1]
        
        is_swing_high = (highs.max() == df['high'].iloc[i])
        is_swing_low = (lows.min() == df['low'].iloc[i])
        
        if is_swing_high:
            swing_highs.append((idx, df['high'].iloc[i]))
        if is_swing_low:
            swing_lows.append((idx, df['low'].iloc[i]))
            
        # Check for Breaks of Structure (BOS)
        close_price = df['close'].iloc[i]
        
        # Bullish BOS: Price closes above previous swing high in an uptrend
        if current_trend == 1 and swing_highs:
            prev_high = swing_highs[-2][1] if len(swing_highs) > 1 else swing_highs[-1][1]
            if close_price > prev_high:
                bos_list[i] = {'type': 'BULLISH', 'level': prev_high}
                # If was in downtrend, this would be a CHOCH
                
        # Bearish BOS: Price closes below previous swing low in a downtrend
        elif current_trend == -1 and swing_lows:
            prev_low = swing_lows[-2][1] if len(swing_lows) > 1 else swing_lows[-1][1]
            if close_price < prev_low:
                bos_list[i] = {'type': 'BEARISH', 'level': prev_low}
                
        # Trend Reversals (CHOCH - Change of Character)
        if current_trend == 1 and swing_lows:
            last_low = swing_lows[-1][1]
            if close_price < last_low:
                choch_list[i] = {'type': 'BEARISH', 'level': last_low}
                current_trend = -1
        elif current_trend == -1 and swing_highs:
            last_high = swing_highs[-1][1]
            if close_price > last_high:
                choch_list[i] = {'type': 'BULLISH', 'level': last_high}
                current_trend = 1
                
    return pd.Series(bos_list, index=df.index), pd.Series(choch_list, index=df.index)

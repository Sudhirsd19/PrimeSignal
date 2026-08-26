import pandas as pd
import numpy as np

def detect_fvgs(df):
    """
    Detects Fair Value Gaps (FVGs) in the DataFrame using vectorized numpy arrays.
    Returns:
        fvg_series: pandas Series of dicts containing FVG details or None.
    """
    n = len(df)
    fvg_list = [None] * n
    if n < 3:
        return pd.Series(fvg_list, index=df.index)

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    timestamps = df.index

    # Precalculate rolling volume average using pandas
    avg_vols = df['volume'].rolling(14, min_periods=1).mean().values

    # Step 1: Detect raw FVGs
    for i in range(2, n):
        # Bullish FVG: Low of candle i is greater than High of candle i-2
        if lows[i] > highs[i-2]:
            if closes[i-1] > opens[i-1]:
                fvg_list[i] = {
                    'type': 'BULLISH',
                    'top': lows[i],
                    'bottom': highs[i-2],
                    'mitigated': False,
                    'partially_mitigated': False,
                    'timestamp': timestamps[i-1]
                }
        # Bearish FVG: High of candle i is less than Low of candle i-2
        elif highs[i] < lows[i-2]:
            if closes[i-1] < opens[i-1]:
                fvg_list[i] = {
                    'type': 'BEARISH',
                    'top': lows[i-2],
                    'bottom': highs[i],
                    'mitigated': False,
                    'partially_mitigated': False,
                    'timestamp': timestamps[i-1]
                }

    # Step 2: Mitigation pass (vectorized lookahead)
    for i in range(n):
        fvg = fvg_list[i]
        if fvg is None:
            continue

        fvg_top = fvg['top']
        fvg_bottom = fvg['bottom']
        fvg_type = fvg['type']
        end_k = min(n, i + 150)

        for k in range(i + 1, end_k):
            k_low = lows[k]
            k_high = highs[k]

            if k_low <= fvg_top and k_high >= fvg_bottom:
                fvg['mitigated'] = True
                candle_range = k_high - k_low
                is_partial = False

                if candle_range > 0:
                    avg_v = avg_vols[k]
                    k_open = opens[k]
                    k_close = closes[k]
                    k_vol = volumes[k]

                    if fvg_type == 'BULLISH':
                        lower_body = min(k_open, k_close)
                        wick_len = lower_body - k_low
                        if k_close > fvg_bottom and (wick_len / candle_range) >= 0.30 and k_vol >= 1.2 * avg_v:
                            is_partial = True
                    elif fvg_type == 'BEARISH':
                        upper_body = max(k_open, k_close)
                        wick_len = k_high - upper_body
                        if k_close < fvg_top and (wick_len / candle_range) >= 0.30 and k_vol >= 1.2 * avg_v:
                            is_partial = True

                fvg['partially_mitigated'] = is_partial
                if not is_partial:
                    break

        fvg_list[i] = fvg

    return pd.Series(fvg_list, index=df.index)


def detect_order_blocks(df, lookback=50):
    """
    Identifies bullish and bearish order blocks using high-performance numpy arrays.
    """
    n = len(df)
    ob_list = [None] * n
    if n < 6:
        return pd.Series(ob_list, index=df.index)

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    timestamps = df.index

    # Rolling averages
    body_diff = np.abs(closes - opens)
    avg_bodies = pd.Series(body_diff).rolling(14, min_periods=1).mean().values
    avg_vols = pd.Series(volumes).rolling(14, min_periods=1).mean().values

    # Step 1: Detect OB formation
    for i in range(5, n):
        candle_body = body_diff[i]
        avg_body = avg_bodies[i]
        candle_vol = volumes[i]
        avg_vol = avg_vols[i]

        if candle_body > 1.5 * avg_body and candle_vol >= 1.5 * avg_vol:
            # Bullish move
            if closes[i] > opens[i]:
                for j in range(i-1, max(-1, i-5), -1):
                    if closes[j] < opens[j]:
                        ob_list[i] = {
                            'type': 'BULLISH',
                            'top': max(opens[j], highs[j]),
                            'bottom': lows[j],
                            'mitigated': False,
                            'partially_mitigated': False,
                            'timestamp': timestamps[j]
                        }
                        break
            # Bearish move
            elif closes[i] < opens[i]:
                for j in range(i-1, max(-1, i-5), -1):
                    if closes[j] > opens[j]:
                        ob_list[i] = {
                            'type': 'BEARISH',
                            'top': highs[j],
                            'bottom': min(opens[j], lows[j]),
                            'mitigated': False,
                            'partially_mitigated': False,
                            'timestamp': timestamps[j]
                        }
                        break

    # Step 2: Mitigation pass
    for i in range(n):
        ob = ob_list[i]
        if ob is None:
            continue

        ob_top = ob['top']
        ob_bottom = ob['bottom']
        ob_type = ob['type']
        end_k = min(n, i + 150)

        ob_height = max(1e-6, ob_top - ob_bottom)

        for k in range(i + 1, end_k):
            k_low = lows[k]
            k_high = highs[k]
            k_close = closes[k]
            
            # P1 Geometric Rule: 25% penetration from proximal boundary
            if ob_type == 'BULLISH':
                # Proximal = ob_top, Distal = ob_bottom
                penetration = (ob_top - k_low) / ob_height
                if penetration >= 0.25:
                    ob['mitigated'] = True
                if k_close < ob_bottom:
                    ob['invalidated'] = True
                    break
            elif ob_type == 'BEARISH':
                # Proximal = ob_bottom, Distal = ob_top
                penetration = (k_high - ob_bottom) / ob_height
                if penetration >= 0.25:
                    ob['mitigated'] = True
                if k_close > ob_top:
                    ob['invalidated'] = True
                    break

            if ob.get('mitigated', False):
                candle_range = k_high - k_low
                is_partial = False

                if candle_range > 0:
                    avg_v = avg_vols[k]
                    k_open = opens[k]
                    k_vol = volumes[k]

                    if ob_type == 'BULLISH':
                        lower_body = min(k_open, k_close)
                        wick_len = lower_body - k_low
                        if k_close > ob_bottom and (wick_len / candle_range) >= 0.30 and k_vol >= 1.2 * avg_v:
                            is_partial = True
                    elif ob_type == 'BEARISH':
                        upper_body = max(k_open, k_close)
                        wick_len = k_high - upper_body
                        if k_close < ob_top and (wick_len / candle_range) >= 0.30 and k_vol >= 1.2 * avg_v:
                            is_partial = True

                ob['partially_mitigated'] = is_partial
                if not is_partial:
                    break

        ob_list[i] = ob

    return pd.Series(ob_list, index=df.index)


def detect_structure(df, period=5):
    """
    Identifies market structure changes (BOS / CHOCH) using numpy arrays.
    """
    n = len(df)
    bos_list = [None] * n
    choch_list = [None] * n
    if n < period * 2 + 1:
        return pd.Series(bos_list, index=df.index), pd.Series(choch_list, index=df.index)

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    timestamps = df.index

    swing_highs = []
    swing_lows = []
    current_trend = 1

    for i in range(period, n):
        # Swing detection requires a look-ahead window of 'period' bars
        if i <= n - period - 1:
            window_highs = highs[i-period:i+period+1]
            window_lows = lows[i-period:i+period+1]

            val_h = highs[i]
            val_l = lows[i]

            is_swing_high = (window_highs.max() == val_h)
            is_swing_low = (window_lows.min() == val_l)

            if is_swing_high:
                swing_highs.append((timestamps[i], val_h))
            if is_swing_low:
                swing_lows.append((timestamps[i], val_l))

        # BOS/CHoCH detection only needs previously confirmed swings — runs to live edge
        close_price = closes[i]

        if current_trend == 1 and swing_highs:
            prev_high = swing_highs[-1][1]
            if close_price > prev_high:
                bos_list[i] = {'type': 'BULLISH', 'level': prev_high}
        elif current_trend == -1 and swing_lows:
            prev_low = swing_lows[-1][1]
            if close_price < prev_low:
                bos_list[i] = {'type': 'BEARISH', 'level': prev_low}

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

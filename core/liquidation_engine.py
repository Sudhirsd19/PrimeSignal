import numpy as np
import pandas as pd
from config import Config

class LiquidationEngine:
    """
    Proprietary Liquidation Magnetic Heatmap & Stop-Hunt Engine.
    Estimates 25x, 50x, and 100x leveraged liquidation clusters and detects stop hunts.
    """
    def __init__(self):
        pass

    def calculate_liquidation_pools(self, df: pd.DataFrame, lookback: int = 60) -> dict:
        """
        Estimates major liquidation density clusters from swing structure.
        """
        if len(df) < 15:
            return {
                'nearest_short_liq': 0.0,
                'nearest_long_liq': 0.0,
                'short_liq_density': 0.0,
                'long_liq_density': 0.0,
                'status': 'INSUFFICIENT_DATA'
            }

        recent_df = df.iloc[-min(len(df), lookback):]
        curr_price = max(float(df['close'].iloc[-1]), 1e-6)
        
        # Find swing highs and lows
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        swing_highs = []
        swing_lows = []
        for i in range(2, len(recent_df) - 2):
            if highs[i] == max(highs[max(0, i-3):min(len(highs), i+4)]):
                swing_highs.append(highs[i])
            if lows[i] == min(lows[max(0, i-3):min(len(lows), i+4)]):
                swing_lows.append(lows[i])

        if not swing_highs:
            swing_highs = [float(recent_df['high'].max())]
        if not swing_lows:
            swing_lows = [float(recent_df['low'].min())]

        # Calculate estimated liquidation price clusters for 25x (4%), 50x (2%), 100x (1%)
        short_liq_pools = []
        for sh in swing_highs:
            short_liq_pools.extend([sh * 1.01, sh * 1.02, sh * 1.04])

        long_liq_pools = []
        for sl in swing_lows:
            long_liq_pools.extend([sl * 0.99, sl * 0.98, sl * 0.96])

        # Filter pools above and below current price
        upper_pools = [p for p in short_liq_pools if p > curr_price]
        lower_pools = [p for p in long_liq_pools if p < curr_price]

        nearest_short_liq = min(upper_pools) if upper_pools else (curr_price * 1.02)
        nearest_long_liq = max(lower_pools) if lower_pools else (curr_price * 0.98)

        # Proximity ratio
        dist_to_short_liq = (nearest_short_liq - curr_price) / curr_price
        dist_to_long_liq = (curr_price - nearest_long_liq) / curr_price

        # Check for active liquidation hunt / sweep on recent candle
        last_c = recent_df.iloc[-1]
        prev_c = recent_df.iloc[-2] if len(recent_df) >= 2 else last_c
        
        hunt_signal = 'NONE'
        proximity_thresh = getattr(Config, 'LIQUIDATION_PROXIMITY_PCT', 0.003)

        # Bullish Liquidation Hunt: Price swept long liquidations and snapped back up with a wick
        if (last_c['low'] <= nearest_long_liq * 1.001) and (last_c['close'] > nearest_long_liq):
            candle_range = last_c['high'] - last_c['low']
            lower_wick = min(last_c['open'], last_c['close']) - last_c['low']
            if candle_range > 0 and (lower_wick / candle_range) >= 0.35:
                hunt_signal = 'BULLISH_LIQUIDATION_HUNT'

        # Bearish Liquidation Hunt: Price swept short liquidations and rejected back down with a wick
        elif (last_c['high'] >= nearest_short_liq * 0.999) and (last_c['close'] < nearest_short_liq):
            candle_range = last_c['high'] - last_c['low']
            upper_wick = last_c['high'] - max(last_c['open'], last_c['close'])
            if candle_range > 0 and (upper_wick / candle_range) >= 0.35:
                hunt_signal = 'BEARISH_LIQUIDATION_HUNT'

        return {
            'nearest_short_liq': round(nearest_short_liq, 4),
            'nearest_long_liq': round(nearest_long_liq, 4),
            'dist_to_short_pct': round(dist_to_short_liq * 100, 2),
            'dist_to_long_pct': round(dist_to_long_liq * 100, 2),
            'hunt_signal': hunt_signal
        }

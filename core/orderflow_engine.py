import numpy as np
import pandas as pd
from config import Config

class OrderFlowEngine:
    """
    Proprietary Order Flow Footprint & Cumulative Volume Delta (CVD) Absorption Engine.
    Detects institutional passive limit order absorption and footprint divergences.
    """
    def __init__(self):
        pass

    def calculate_cvd_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates Cumulative Volume Delta (CVD) series from OHLCV.
        """
        if len(df) < 5:
            return pd.Series([0.0] * len(df), index=df.index)

        # Intra-bar volume delta estimation based on close position within the bar range
        high_low = (df['high'] - df['low']).replace(0, 1e-9)
        close_pos = (df['close'] - df['low']) / high_low
        
        # Intra-bar delta: close in top 50% = net buy, bottom 50% = net sell
        bar_delta = df['volume'] * (2.0 * close_pos - 1.0)
        cvd_series = bar_delta.cumsum()
        return cvd_series

    def detect_absorption_divergence(self, df: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Detects Bullish or Bearish Institutional Absorption Divergences.
        """
        lookback = getattr(Config, 'CVD_DIVERGENCE_LOOKBACK', lookback)
        if len(df) < lookback:
            return {
                'absorption': 'NEUTRAL',
                'delta_ratio': 0.0,
                'cvd_trend': 'NEUTRAL',
                'confidence': 0.5
            }

        sub_df = df.iloc[-lookback:]
        cvd = self.calculate_cvd_series(sub_df)
        prices = sub_df['close'].values
        cvd_vals = cvd.values

        # Recent 5-bar vs older 15-bar comparison
        recent_p = prices[-1]
        prev_min_p = np.min(prices[:-3])
        prev_max_p = np.max(prices[:-3])

        recent_cvd = cvd_vals[-1]
        prev_min_cvd = np.min(cvd_vals[:-3])
        prev_max_cvd = np.max(cvd_vals[:-3])

        absorption = 'NEUTRAL'
        confidence = 0.5

        # 1. Bullish Absorption: Price made a Lower Low or equal low, but CVD made a Higher Low
        if (recent_p <= prev_min_p * 1.002) and (recent_cvd > prev_min_cvd):
            absorption = 'BULLISH_ABSORPTION'
            confidence = 0.85

        # 2. Bearish Absorption: Price made a Higher High or equal high, but CVD made a Lower High
        elif (recent_p >= prev_max_p * 0.998) and (recent_cvd < prev_max_cvd):
            absorption = 'BEARISH_ABSORPTION'
            confidence = 0.85

        # Recent 3-bar delta ratio
        recent_delta = cvd_vals[-1] - cvd_vals[-4] if len(cvd_vals) >= 4 else 0.0
        recent_vol = sub_df['volume'].iloc[-3:].sum()
        delta_ratio = recent_delta / (recent_vol + 1e-9)

        cvd_trend = 'BULLISH' if recent_delta > 0 else 'BEARISH'

        return {
            'absorption': absorption,
            'delta_ratio': round(float(delta_ratio), 4),
            'cvd_trend': cvd_trend,
            'confidence': confidence
        }

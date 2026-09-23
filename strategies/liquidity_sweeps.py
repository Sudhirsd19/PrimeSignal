"""
Institutional Liquidity Sweep Engine
====================================
Detects institutional liquidity sweeps ('Turtle Soup' / Judas Swings) around:
1. Previous Day High (PDH) & Previous Day Low (PDL)
2. Asian Session Range (00:00 - 08:00 UTC) High (ASH) & Low (ASL)

When price sweeps a key liquidity pool and immediately closes back inside
the range with a strong rejection wick, it confirms institutional absorption.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from strategies.indicators import prepare_dataframe


class LiquiditySweepEngine:
    """Detects liquidity sweeps on key institutional levels."""

    def __init__(
        self,
        min_wick_ratio: float = 0.50,      # Rejection wick must be >= 50% of candle range (hammer/pinbar)
        buffer_pct: float = 0.0015,         # 0.15% buffer for SL placement
    ):
        self.min_wick_ratio = min_wick_ratio
        self.buffer_pct = buffer_pct

    def compute_session_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes Previous Day High/Low and Asian Session High/Low from 15m/1h candle history.
        """
        levels = {
            'pdh': 0.0,
            'pdl': 0.0,
            'asian_high': 0.0,
            'asian_low': 0.0,
        }
        if df.empty or len(df) < 20:
            return levels

        # Ensure datetime index
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.index = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
            elif 'time' in data.columns:
                data.index = pd.to_datetime(data['time'], unit='ms', utc=True)
            else:
                return levels

        # 1. Previous Day High / Low (PDH / PDL)
        try:
            daily = data.resample('1D').agg({'high': 'max', 'low': 'min'}).dropna()
            if len(daily) >= 2:
                # Previous completed day is iloc[-2]
                levels['pdh'] = float(daily['high'].iloc[-2])
                levels['pdl'] = float(daily['low'].iloc[-2])
            elif len(daily) == 1:
                levels['pdh'] = float(daily['high'].iloc[-1])
                levels['pdl'] = float(daily['low'].iloc[-1])
        except Exception:
            pass

        # 2. Asian Session High / Low (00:00 to 08:00 UTC)
        try:
            # Filter candles between 00:00 and 08:00 UTC of current day
            current_day = data.index[-1].date()
            asian_candles = data[(data.index.date == current_day) & (data.index.hour >= 0) & (data.index.hour < 8)]
            if len(asian_candles) >= 4:
                levels['asian_high'] = float(asian_candles['high'].max())
                levels['asian_low'] = float(asian_candles['low'].min())
        except Exception:
            pass

        return levels

    def detect_sweep_setup(self, df: Any) -> Dict[str, Any]:
        """
        Scans the latest closed candle (iloc[-2]) and current candle (iloc[-1])
        for institutional liquidity sweeps.

        Returns dict with:
            is_setup: bool
            signal: 'BUY', 'SELL', or 'NONE'
            sweep_level_type: 'PDH', 'PDL', 'ASIAN_HIGH', 'ASIAN_LOW', or 'NONE'
            sweep_level: float
            stop_loss: float
            suggested_tp: float
            rejection_wick_ratio: float
            reason: str
        """
        empty_res = {
            'is_setup': False,
            'signal': 'NONE',
            'sweep_level_type': 'NONE',
            'sweep_level': 0.0,
            'stop_loss': 0.0,
            'suggested_tp': 0.0,
            'rejection_wick_ratio': 0.0,
            'reason': 'No liquidity sweep'
        }

        data = prepare_dataframe(df) if isinstance(df, list) else df
        if not isinstance(data, pd.DataFrame) or len(data) < 30:
            return empty_res

        levels = self.compute_session_levels(data)
        pdh = levels['pdh']
        pdl = levels['pdl']
        ash = levels['asian_high']
        asl = levels['asian_low']

        # Check candidate candles (last closed bar iloc[-1] or prior closed bar iloc[-2])
        candidate_candles = [data.iloc[-1]]
        if len(data) >= 2:
            candidate_candles.append(data.iloc[-2])

        for candle in candidate_candles:
            c_open = float(candle['open'])
            c_high = float(candle['high'])
            c_low = float(candle['low'])
            c_close = float(candle['close'])
            c_range = max(1e-9, c_high - c_low)

            upper_wick = c_high - max(c_open, c_close)
            lower_wick = min(c_open, c_close) - c_low

            upper_wick_ratio = upper_wick / c_range
            lower_wick_ratio = lower_wick / c_range

            # ── 1. Bullish Liquidity Sweep (Sweep below PDL or Asian Low + Rejection) ──
            target_low_level = None
            level_type = None

            if pdl > 0 and c_low < pdl and c_close > pdl:
                target_low_level = pdl
                level_type = "PDL"
            elif asl > 0 and c_low < asl and c_close > asl:
                target_low_level = asl
                level_type = "ASIAN_LOW"

            # Require true rejection: strong lower wick (>= min_wick_ratio) AND candle is not a pure dumping red bar
            is_bullish_sweep_candle = (c_close >= c_open) or (lower_wick_ratio >= 0.60)
            if target_low_level and lower_wick_ratio >= self.min_wick_ratio and is_bullish_sweep_candle:
                sl_price = c_low * (1.0 - self.buffer_pct)
                risk = abs(c_close - sl_price)
                # Target range midpoint or PDH
                tp_target = max(c_close + (2.5 * risk), pdh if pdh > c_close else c_close + (2.5 * risk))
                return {
                    'is_setup': True,
                    'signal': 'BUY',
                    'sweep_level_type': level_type,
                    'sweep_level': target_low_level,
                    'stop_loss': round(sl_price, 4),
                    'suggested_tp': round(tp_target, 4),
                    'rejection_wick_ratio': round(lower_wick_ratio, 3),
                    'reason': f"Bullish Liquidity Purge: {level_type} swept @ {target_low_level:.2f} with {lower_wick_ratio*100:.1f}% lower rejection wick"
                }

            # ── 2. Bearish Liquidity Sweep (Sweep above PDH or Asian High + Rejection) ──
            target_high_level = None
            high_level_type = None

            if pdh > 0 and c_high > pdh and c_close < pdh:
                target_high_level = pdh
                high_level_type = "PDH"
            elif ash > 0 and c_high > ash and c_close < ash:
                target_high_level = ash
                high_level_type = "ASIAN_HIGH"

            is_bearish_sweep_candle = (c_close <= c_open) or (upper_wick_ratio >= 0.60)
            if target_high_level and upper_wick_ratio >= self.min_wick_ratio and is_bearish_sweep_candle:
                sl_price = c_high * (1.0 + self.buffer_pct)
                risk = abs(sl_price - c_close)
                # Target range midpoint or PDL
                tp_target = min(c_close - (2.5 * risk), pdl if (pdl > 0 and pdl < c_close) else c_close - (2.5 * risk))
                return {
                    'is_setup': True,
                    'signal': 'SELL',
                    'sweep_level_type': high_level_type,
                    'sweep_level': target_high_level,
                    'stop_loss': round(sl_price, 4),
                    'suggested_tp': round(tp_target, 4),
                    'rejection_wick_ratio': round(upper_wick_ratio, 3),
                    'reason': f"Bearish Liquidity Purge: {high_level_type} swept @ {target_high_level:.2f} with {upper_wick_ratio*100:.1f}% upper rejection wick"
                }
        return empty_res

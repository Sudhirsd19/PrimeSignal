"""
Market Regime Classification Engine
===================================
Institutional adaptive trading framework that detects market regime:
1. TRENDING_BULL / TRENDING_BEAR: Strong momentum, high directional conviction.
2. CHOP_COMPRESSION: Low ADX, range consolidation.
3. VOLATILITY_SHOCK: Extreme ATR expansion / news whipsaw.
4. NEUTRAL_RANGE: Mean-reverting balanced auction.

Adjusts trade targets, runner allowance, and risk exposure adaptively.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict
import pandas as pd
import numpy as np

from strategies.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_bollinger_bands,
    prepare_dataframe,
)


class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    CHOP_COMPRESSION = "CHOP_COMPRESSION"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    NEUTRAL_RANGE = "NEUTRAL_RANGE"


class MarketRegimeClassifier:
    """Classifies asset market regime to dynamically adapt strategy execution parameters."""

    def __init__(
        self,
        adx_trend_threshold: float = 22.0,
        adx_chop_threshold: float = 19.0,
        shock_atr_multiplier: float = 2.2,
    ):
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_chop_threshold = adx_chop_threshold
        self.shock_atr_multiplier = shock_atr_multiplier

    def classify_regime(self, df: Any) -> Dict[str, Any]:
        """
        Analyzes OHLCV dataframe and returns market regime diagnostics and adaptive execution parameters.
        """
        data = prepare_dataframe(df) if isinstance(df, list) else df
        if not isinstance(data, pd.DataFrame) or len(data) < 35:
            return {
                'regime': MarketRegime.NEUTRAL_RANGE.value,
                'adx': 20.0,
                'atr_ratio': 1.0,
                'bb_squeeze': False,
                'allow_breakout': True,
                'allow_mean_reversion': True,
                'tp1_mult': 1.5,
                'tp2_mult': 2.2,
                'runner_allowed': True,
                'risk_mult': 1.0,
                'description': 'Insufficient data for regime classification'
            }

        # 1. Trend Strength (ADX)
        adx_df = calculate_adx(data, period=14)
        curr_adx = float(adx_df['adx'].iloc[-1]) if not adx_df.empty and not math.isnan(adx_df['adx'].iloc[-1]) else 20.0

        # 2. Volatility State (ATR & Shock Ratio)
        atr_series = calculate_atr(data, period=14)
        curr_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not math.isnan(atr_series.iloc[-1]) else 1.0
        rolling_atr_mean = float(atr_series.rolling(50, min_periods=10).mean().iloc[-1])
        atr_ratio = (curr_atr / rolling_atr_mean) if (rolling_atr_mean and rolling_atr_mean > 0) else 1.0

        # 3. Trend Direction & Structure
        close_p = float(data['close'].iloc[-1])
        ema20 = float(calculate_ema(data, 20).iloc[-1])
        ema50 = float(calculate_ema(data, 50).iloc[-1])
        ema200 = float(calculate_ema(data, min(200, len(data))).iloc[-1]) if len(data) >= 80 else ema50

        bullish_alignment = close_p > ema20 >= ema50 >= ema200
        bearish_alignment = close_p < ema20 <= ema50 <= ema200

        # 4. Bollinger Band Width (Compression Detection)
        bb = calculate_bollinger_bands(data, period=20, std_dev=2.0)
        curr_bbw = float(bb['bandwidth'].iloc[-1]) if 'bandwidth' in bb.columns and not math.isnan(bb['bandwidth'].iloc[-1]) else 0.04
        bbw_rolling_min = float(bb['bandwidth'].rolling(50, min_periods=10).quantile(0.25).iloc[-1]) if 'bandwidth' in bb.columns else 0.03
        is_compressed = curr_bbw <= bbw_rolling_min

        # ─── Regime Decision Matrix ───
        if atr_ratio >= self.shock_atr_multiplier:
            regime = MarketRegime.VOLATILITY_SHOCK
            desc = f"Extreme Volatility Shock (ATR {atr_ratio:.2f}x average)"
            tp1_mult = 1.3
            tp2_mult = 1.8
            allow_breakout = False
            allow_mean_reversion = False
            runner_allowed = False
            risk_mult = 0.50   # Half risk during wild price spikes

        elif curr_adx >= self.adx_trend_threshold and bullish_alignment:
            regime = MarketRegime.TRENDING_BULL
            desc = f"Strong Bullish Trend (ADX: {curr_adx:.1f}, EMAs aligned)"
            tp1_mult = 1.5
            tp2_mult = 2.6
            allow_breakout = True
            allow_mean_reversion = False
            runner_allowed = True
            risk_mult = 1.00

        elif curr_adx >= self.adx_trend_threshold and bearish_alignment:
            regime = MarketRegime.TRENDING_BEAR
            desc = f"Strong Bearish Trend (ADX: {curr_adx:.1f}, EMAs aligned)"
            tp1_mult = 1.5
            tp2_mult = 2.6
            allow_breakout = True
            allow_mean_reversion = False
            runner_allowed = True
            risk_mult = 1.00

        elif curr_adx < self.adx_chop_threshold or is_compressed:
            regime = MarketRegime.CHOP_COMPRESSION
            desc = f"Choppy Range Compression (ADX: {curr_adx:.1f}, Squeeze: {is_compressed})"
            tp1_mult = 1.2
            tp2_mult = 1.6
            allow_breakout = False   # Breakouts fail in compression; fakeouts dominate!
            allow_mean_reversion = True
            runner_allowed = False
            risk_mult = 0.75

        else:
            regime = MarketRegime.NEUTRAL_RANGE
            desc = f"Balanced Auction Range (ADX: {curr_adx:.1f})"
            tp1_mult = 1.4
            tp2_mult = 2.0
            allow_breakout = True
            allow_mean_reversion = True
            runner_allowed = True
            risk_mult = 0.90

        return {
            'regime': regime.value,
            'adx': round(curr_adx, 2),
            'atr_ratio': round(atr_ratio, 2),
            'bb_squeeze': bool(is_compressed),
            'allow_breakout': allow_breakout,
            'allow_mean_reversion': allow_mean_reversion,
            'tp1_mult': tp1_mult,
            'tp2_mult': tp2_mult,
            'runner_allowed': runner_allowed,
            'risk_mult': risk_mult,
            'description': desc
        }

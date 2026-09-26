"""
HUMAN-MIND TREND-PULLBACK SCALPER STRATEGY
==========================================
Discretionary Human Trader Mimicry Engine:
- 1-Hour HTF Dominant Trend Alignment (No counter-trend chop).
- 15-Minute Value-Zone Pullback (Never chase green/red breakout tops).
- Candlestick Reversal Trigger (Hammer, Shooting Star, Engulfing).
- Volume Expansion Confirmation (> 1.20x 20-period MA).
- Scalper Profit Booking: TP1 at 1.3R (50%), Breakeven Lock, TP2 at 2.2R (50%).
"""

import time
import math
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from strategies.base import BaseStrategy
from config import Config

class HumanMindScalperStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="HumanMindScalper")

    def generate_signal(self, htf_df, ltf_df, relaxed=False, allow_short=True) -> Tuple[str, Dict[str, Any]]:
        metadata: Dict[str, Any] = {
            'htf_trend': 'NEUTRAL',
            'ltf_rsi': 50.0,
            'stop_loss': None,
            'take_profit': None,
            'take_profit_1r': None,
            'tp1': None,
            'tp2': None,
            'tp3': None,
            'reason': 'No setup',
            'active_ob_level': 0.0,
            'active_ob_type': 'NONE',
            'active_bullish_ob_level': 0.0,
            'active_bearish_ob_level': 0.0,
            'zone_id': None,
            'setup_type': 'HUMAN_PULLBACK',
            'mode': 'STRICT',
            'score': 4.0,
            'regime_diag': {'regime': 'TREND', 'risk_mult': 1.0, 'bb_squeeze': False},
            'debug_checks': {'trend': 'FAIL', 'pullback': 'FAIL', 'trigger': 'FAIL', 'volume': 'FAIL'}
        }

        if htf_df is None or ltf_df is None or len(htf_df) < 55 or len(ltf_df) < 55:
            metadata['reason'] = "Insufficient candle history"
            return "HOLD", metadata

        # Convert to DataFrame if list
        if not isinstance(htf_df, pd.DataFrame):
            htf_df = pd.DataFrame(htf_df)
        if not isinstance(ltf_df, pd.DataFrame):
            ltf_df = pd.DataFrame(ltf_df)

        # ── 1. HTF 1-Hour Trend ──
        htf_ema20 = htf_df['close'].ewm(span=20, adjust=False).mean()
        htf_ema50 = htf_df['close'].ewm(span=50, adjust=False).mean()
        htf_last_close = htf_df['close'].iloc[-1]

        if htf_ema20.iloc[-1] > htf_ema50.iloc[-1]:
            htf_trend = 'BULLISH'
        elif htf_ema20.iloc[-1] < htf_ema50.iloc[-1]:
            htf_trend = 'BEARISH'
        else:
            htf_trend = 'NEUTRAL'

        metadata['htf_trend'] = htf_trend
        if htf_trend == 'NEUTRAL':
            metadata['reason'] = "Neutral 1H HTF Trend"
            return "HOLD", metadata

        metadata['debug_checks']['trend'] = 'PASS'

        # Check venue capability
        if htf_trend == 'BEARISH' and not allow_short:
            metadata['reason'] = "Shorts disabled by venue capabilities"
            return "HOLD", metadata

        # ── 2. LTF 15m Indicators ──
        ltf_close = ltf_df['close']
        ltf_open = ltf_df['open']
        ltf_high = ltf_df['high']
        ltf_low = ltf_df['low']
        ltf_vol = ltf_df['volume']

        ema20_series = ltf_close.ewm(span=20, adjust=False).mean()
        ema50_series = ltf_close.ewm(span=50, adjust=False).mean()
        vol_ma_series = ltf_vol.rolling(20).mean()

        # RSI 14
        delta = ltf_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))

        # Evaluate last closed candle
        curr_c = ltf_close.iloc[-1]
        curr_o = ltf_open.iloc[-1]
        curr_h = ltf_high.iloc[-1]
        curr_l = ltf_low.iloc[-1]
        curr_v = ltf_vol.iloc[-1]

        prev_c = ltf_close.iloc[-2]
        prev_o = ltf_open.iloc[-2]
        prev_h = ltf_high.iloc[-2]
        prev_l = ltf_low.iloc[-2]

        curr_ema20 = ema20_series.iloc[-1]
        curr_ema50 = ema50_series.iloc[-1]
        curr_vol_ma = vol_ma_series.iloc[-1] if not math.isnan(vol_ma_series.iloc[-1]) else 1.0
        curr_rsi = rsi_series.iloc[-1] if not math.isnan(rsi_series.iloc[-1]) else 50.0
        metadata['ltf_rsi'] = curr_rsi

        body = abs(curr_c - curr_o)
        lower_wick = min(curr_c, curr_o) - curr_l
        upper_wick = curr_h - max(curr_c, curr_o)
        is_volume_confirmed = curr_v >= (curr_vol_ma * 1.20)

        # ── 3. LONG SETUP ──
        if htf_trend == 'BULLISH':
            # Trend stack: 15m EMA 20 > EMA 50
            if not (curr_ema20 > curr_ema50):
                metadata['reason'] = f"LTF not aligned with 1H: EMA20 ({curr_ema20:.4f}) <= EMA50 ({curr_ema50:.4f})"
                return "HOLD", metadata

            # Value-zone pullback: Low touched or probed near EMA 20/50
            pullback_ok = (curr_l <= curr_ema20 * 1.002) and (curr_l >= curr_ema50 * 0.990)
            if not pullback_ok:
                metadata['reason'] = "Price not in value-zone pullback (EMA 20/50)"
                return "HOLD", metadata
            metadata['debug_checks']['pullback'] = 'PASS'

            # RSI check (40 - 62)
            if not (38.0 <= curr_rsi <= 63.0):
                metadata['reason'] = f"RSI out of bounds for Long entry ({curr_rsi:.1f})"
                return "HOLD", metadata

            # Trigger candle: Hammer (rejection) or Bullish Engulfing
            is_hammer = (curr_c > curr_o) and (lower_wick >= 1.2 * max(body, 1e-6))
            is_engulfing = (curr_c > prev_h) and (prev_c < prev_o)

            if not (is_hammer or is_engulfing):
                metadata['reason'] = "Waiting for Bullish Trigger Candle (Hammer or Engulfing)"
                return "HOLD", metadata
            metadata['debug_checks']['trigger'] = 'PASS'

            if not is_volume_confirmed:
                metadata['reason'] = f"Insufficient volume confirmation ({curr_v:.1f} < {curr_vol_ma * 1.20:.1f})"
                return "HOLD", metadata
            metadata['debug_checks']['volume'] = 'PASS'

            # Valid Long setup!
            entry_p = curr_c
            # Stop loss just below pullback low with 0.3% buffer
            sl_p = min(curr_l, prev_l) * 0.997
            risk_d = entry_p - sl_p

            # Clamp risk distance between 0.6% and 2.0%
            if risk_d < (entry_p * 0.006):
                sl_p = entry_p * 0.994
                risk_d = entry_p - sl_p
            elif risk_d > (entry_p * 0.020):
                sl_p = entry_p * 0.980
                risk_d = entry_p - sl_p

            tp1_mult = getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.3)
            tp2_mult = getattr(Config, 'RISK_REWARD_RATIO', 2.2)

            tp1_p = entry_p + (tp1_mult * risk_d)
            tp2_p = entry_p + (tp2_mult * risk_d)

            metadata['stop_loss'] = sl_p
            metadata['take_profit'] = tp1_p
            metadata['take_profit_1r'] = tp1_p
            metadata['tp1'] = tp1_p
            metadata['tp2'] = tp2_p
            metadata['score'] = 4.5
            metadata['zone_id'] = f"HUMAN_LONG_{int(time.time())}"
            metadata['reason'] = f"1H Bullish Trend + 15m Value Pullback ({'Hammer' if is_hammer else 'Engulfing'})"
            return "BUY", metadata

        # ── 4. SHORT SETUP ──
        elif htf_trend == 'BEARISH' and allow_short:
            if not (curr_ema20 < curr_ema50):
                metadata['reason'] = f"LTF not aligned with 1H: EMA20 ({curr_ema20:.4f}) >= EMA50 ({curr_ema50:.4f})"
                return "HOLD", metadata

            # Value-zone pullback up to EMA 20/50
            pullback_ok = (curr_h >= curr_ema20 * 0.998) and (curr_h <= curr_ema50 * 1.010)
            if not pullback_ok:
                metadata['reason'] = "Price not in value-zone pullback (EMA 20/50)"
                return "HOLD", metadata
            metadata['debug_checks']['pullback'] = 'PASS'

            # RSI check (38 - 60)
            if not (37.0 <= curr_rsi <= 62.0):
                metadata['reason'] = f"RSI out of bounds for Short entry ({curr_rsi:.1f})"
                return "HOLD", metadata

            # Trigger candle: Shooting star (upper rejection) or Bearish Engulfing
            is_shooting_star = (curr_c < curr_o) and (upper_wick >= 1.2 * max(body, 1e-6))
            is_bearish_engulfing = (curr_c < prev_l) and (prev_c > prev_o)

            if not (is_shooting_star or is_bearish_engulfing):
                metadata['reason'] = "Waiting for Bearish Trigger Candle (Shooting Star or Engulfing)"
                return "HOLD", metadata
            metadata['debug_checks']['trigger'] = 'PASS'

            if not is_volume_confirmed:
                metadata['reason'] = f"Insufficient volume confirmation ({curr_v:.1f} < {curr_vol_ma * 1.20:.1f})"
                return "HOLD", metadata
            metadata['debug_checks']['volume'] = 'PASS'

            # Valid Short setup!
            entry_p = curr_c
            sl_p = max(curr_h, prev_h) * 1.003
            risk_d = sl_p - entry_p

            if risk_d < (entry_p * 0.006):
                sl_p = entry_p * 1.006
                risk_d = sl_p - entry_p
            elif risk_d > (entry_p * 0.020):
                sl_p = entry_p * 1.020
                risk_d = sl_p - entry_p

            tp1_mult = getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.3)
            tp2_mult = getattr(Config, 'RISK_REWARD_RATIO', 2.2)

            tp1_p = entry_p - (tp1_mult * risk_d)
            tp2_p = entry_p - (tp2_mult * risk_d)

            metadata['stop_loss'] = sl_p
            metadata['take_profit'] = tp1_p
            metadata['take_profit_1r'] = tp1_p
            metadata['tp1'] = tp1_p
            metadata['tp2'] = tp2_p
            metadata['score'] = 4.5
            metadata['zone_id'] = f"HUMAN_SHORT_{int(time.time())}"
            metadata['reason'] = f"1H Bearish Trend + 15m Value Pullback ({'ShootingStar' if is_shooting_star else 'BearishEngulfing'})"
            return "SELL", metadata

        return "HOLD", metadata

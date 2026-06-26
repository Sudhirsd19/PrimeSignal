from strategies.base import BaseStrategy
from strategies.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap, calculate_adx
from strategies.smc import detect_fvgs, detect_order_blocks, detect_structure
from config import Config
import numpy as np

class MultiTimeframeSMCStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MultiTimeframeSMC")

    def generate_signal(self, htf_df, ltf_df, relaxed=False, super_relaxed=False):
        """
        Executes Multi-Timeframe Smart Money Concepts strategy.
        """
        metadata = {
            'htf_trend': 'NEUTRAL',
            'strong_trend': False,
            'ltf_rsi': 50.0,
            'stop_loss': None,
            'take_profit': None,
            'take_profit_1r': None,
            'reason': 'No setup',
            'active_bullish_ob_level': 0.0,
            'active_bearish_ob_level': 0.0,
            'active_ob_level': 0.0,
            'active_ob_type': 'NONE',
            'zone_id': None,
            'setup_type': 'NONE',
            'debug_checks': {
                'trend': 'FAIL',
                'zone': 'FAIL',
                'trigger': 'FAIL',
                'vwap': 'FAIL',
                'volatility': 'FAIL'
            }
        }

        if len(htf_df) < Config.TREND_EMA or len(ltf_df) < Config.LONG_EMA + 10:
            metadata['reason'] = "Insufficient data"
            return "HOLD", metadata

        htf_ema_50 = calculate_ema(htf_df, 50)
        htf_ema_200 = calculate_ema(htf_df, 200)
        
        latest_htf_close = htf_df['close'].iloc[-1]
        latest_htf_ema_50 = htf_ema_50.iloc[-1]
        latest_htf_ema_200 = htf_ema_200.iloc[-1]

        if latest_htf_close > latest_htf_ema_50 > latest_htf_ema_200:
            htf_trend = 'BULLISH'
        elif latest_htf_close < latest_htf_ema_50 < latest_htf_ema_200:
            htf_trend = 'BEARISH'
        else:
            htf_trend = 'NEUTRAL'

        metadata['htf_trend'] = htf_trend
        if htf_trend == 'NEUTRAL':
            metadata['reason'] = "Neutral HTF Trend"
            return "HOLD", metadata
            
        metadata['debug_checks']['trend'] = 'PASS'

        ltf_closes = ltf_df['close']
        ltf_rsi    = calculate_rsi(ltf_df, Config.RSI_PERIOD)
        ltf_atr    = calculate_atr(ltf_df, Config.ATR_PERIOD)
        ltf_vwap   = calculate_vwap(ltf_df)

        ema_short = calculate_ema(ltf_df, Config.SHORT_EMA)
        ema_long  = calculate_ema(ltf_df, Config.LONG_EMA)

        fvgs       = detect_fvgs(ltf_df)
        obs        = detect_order_blocks(ltf_df)
        
        strong_trend = False
        ema_dist = abs(latest_htf_ema_50 - latest_htf_ema_200) / latest_htf_ema_200
        
        adx_df = calculate_adx(ltf_df)
        curr_adx = adx_df['adx'].iloc[-2]
        prev_adx = adx_df['adx'].iloc[-3]
        adx_rising = curr_adx > prev_adx and curr_adx >= 20
        
        # Task 1: Market Regime
        avg_atr_14 = ltf_atr.rolling(14).mean().iloc[-2]
        curr_atr = ltf_atr.iloc[-2]
        if curr_atr > 1.2 * avg_atr_14:
            market_regime = 'HIGH_VOL'
        else:
            if curr_adx > 25: market_regime = 'TREND'
            elif curr_adx >= 15: market_regime = 'MIXED'
            else: market_regime = 'RANGE'
        metadata['market_regime'] = market_regime

        if curr_adx < 15 and ema_dist < 0.002:
            metadata['reason'] = "Chop Market Filter (ADX<15 & EMA diff<0.2%)"
            return "HOLD", metadata
            
        if ema_dist >= 0.005 or adx_rising:
            mom_count = 0
            avg_body = abs(ltf_df['close'] - ltf_df['open']).rolling(14).mean().iloc[-2]
            directional_closes = 0
            for i in range(1, 4):
                idx = -1 - i
                c_close = ltf_df.iloc[idx]['close']
                c_open = ltf_df.iloc[idx]['open']
                if htf_trend == 'BULLISH':
                    if c_close > c_open:
                        directional_closes += 1
                        if (c_close - c_open) > 1.2 * avg_body: mom_count += 1
                    else:
                        directional_closes = 0
                else:
                    if c_open > c_close:
                        directional_closes += 1
                        if (c_open - c_close) > 1.2 * avg_body: mom_count += 1
                    else:
                        directional_closes = 0
            if mom_count >= 1 or directional_closes >= 2:
                strong_trend = True
        metadata['strong_trend'] = strong_trend

        import datetime
        current_hour = datetime.datetime.utcnow().hour
        session_name = 'OTHER'
        if 0 <= current_hour < 8: session_name = 'ASIA'
        elif 8 <= current_hour < 13: session_name = 'LONDON'
        elif 13 <= current_hour < 22: session_name = 'NY'
        metadata['session'] = session_name
        
        curr_price = ltf_closes.iloc[-2]
        curr_rsi   = ltf_rsi.iloc[-2]
        prev_rsi   = ltf_rsi.iloc[-3]
        curr_atr   = ltf_atr.iloc[-2]
        curr_vwap  = ltf_vwap.iloc[-2]
        prev_vwap  = ltf_vwap.iloc[-3]
        
        curr_ema_50 = calculate_ema(ltf_df, 50).iloc[-2]

        prev_short = ema_short.iloc[-3]
        prev_long  = ema_long.iloc[-3]
        curr_short = ema_short.iloc[-2]
        curr_long  = ema_long.iloc[-2]

        metadata['ltf_rsi']  = curr_rsi
        metadata['ltf_vwap'] = curr_vwap
        metadata['curr_atr'] = curr_atr
        metadata['curr_adx'] = curr_adx
        metadata['ema_short'] = curr_short
        metadata['ema_long'] = curr_long
        metadata['curr_ema_50'] = curr_ema_50
        metadata['htf_ema_50'] = latest_htf_ema_50
        metadata['htf_ema_200'] = latest_htf_ema_200
        metadata['candle_volume'] = float(ltf_df['volume'].iloc[-2])

        vol_pass = (curr_atr / curr_price) >= Config.MIN_ATR_PCT
        metadata['debug_checks']['volatility'] = 'PASS' if vol_pass else 'FAIL'

        def in_bounds(price, bottom, top):
            return (bottom * (1 - Config.ZONE_BUFFER_PCT)) <= price <= (top * (1 + Config.ZONE_BUFFER_PCT))

        active_bullish_ob  = None
        active_bearish_ob  = None
        
        def is_zone_active(zone):
            if not zone['mitigated']: return True
            if relaxed and zone.get('partially_mitigated', False): return True
            return False

        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 2 - Config.MAX_ZONE_AGE_CANDLES), -1):
            ob = obs.iloc[idx]
            if ob:
                if ob['type'] == 'BULLISH' and is_zone_active(ob) and active_bullish_ob is None:
                    active_bullish_ob = ob
                elif ob['type'] == 'BEARISH' and is_zone_active(ob) and active_bearish_ob is None:
                    active_bearish_ob = ob

        active_bullish_fvg = None
        active_bearish_fvg = None

        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 2 - Config.MAX_ZONE_AGE_CANDLES), -1):
            fvg = fvgs.iloc[idx]
            if fvg:
                if fvg['type'] == 'BULLISH' and is_zone_active(fvg) and active_bullish_fvg is None:
                    active_bullish_fvg = fvg
                elif fvg['type'] == 'BEARISH' and is_zone_active(fvg) and active_bearish_fvg is None:
                    active_bearish_fvg = fvg

        vwap_tol = Config.VWAP_TOLERANCE * 2 if relaxed else Config.VWAP_TOLERANCE

        metadata['active_ob_details'] = {
            'bullish': {'bottom': active_bullish_ob['bottom'], 'top': active_bullish_ob['top']} if active_bullish_ob else None,
            'bearish': {'bottom': active_bearish_ob['bottom'], 'top': active_bearish_ob['top']} if active_bearish_ob else None
        }
        metadata['active_fvg_details'] = {
            'bullish': {'bottom': active_bullish_fvg['bottom'], 'top': active_bullish_fvg['top']} if active_bullish_fvg else None,
            'bearish': {'bottom': active_bearish_fvg['bottom'], 'top': active_bearish_fvg['top']} if active_bearish_fvg else None
        }

        if htf_trend == 'BULLISH':
            in_zone = False
            reason  = ""
            entry_type = None
            zone_bottom = 0.0
            zone_top = 0.0
            zone_ts = None
            
            last_10 = ltf_df.iloc[-12:-2]
            swing_low = last_10['low'].min()
            swing_high = last_10['high'].max()
            swing_range = (swing_high - swing_low) / swing_low
            trigger_low = ltf_df.iloc[-2]['low']
            trigger_close = ltf_df.iloc[-2]['close']
            trigger_open = ltf_df.iloc[-2]['open']
            trigger_high = ltf_df.iloc[-2]['high']
            candle_range = trigger_high - trigger_low
            
            liq_sweep = False
            if swing_range > 0.003:
                if trigger_low < swing_low and trigger_close > swing_low:
                    lower_wick = min(trigger_open, trigger_close) - trigger_low
                    if candle_range > 0 and (lower_wick / candle_range) >= 0.3:
                        avg_body = abs(ltf_df['close'] - ltf_df['open']).rolling(14).mean().iloc[-2]
                        if abs(trigger_close - trigger_open) > 1.2 * avg_body:
                            liq_sweep = True

            if active_bullish_ob and in_bounds(curr_price, active_bullish_ob['bottom'], active_bullish_ob['top']):
                in_zone = True
                entry_type = "OB"
                zone_bottom = active_bullish_ob['bottom']
                zone_top = active_bullish_ob['top']
                zone_ts = active_bullish_ob['timestamp']
                reason  = f"Price inside Bullish OB [{zone_bottom:.2f}-{zone_top:.2f}]"

            elif not in_zone and active_bullish_fvg and in_bounds(curr_price, active_bullish_fvg['bottom'], active_bullish_fvg['top']):
                in_zone = True
                entry_type = "FVG"
                zone_bottom = active_bullish_fvg['bottom']
                zone_top = active_bullish_fvg['top']
                zone_ts = active_bullish_fvg['timestamp']
                reason  = f"Price inside Bullish FVG [{zone_bottom:.2f}-{zone_top:.2f}]"
            
            elif not in_zone and liq_sweep:
                in_zone = True
                entry_type = "SWEEP"
                zone_bottom = trigger_low
                zone_top = swing_low
                zone_ts = ltf_df.index[-2]
                reason = f"Liquidity Sweep of Swing Low [{swing_low:.2f}]"
                
            # Secondary Setups
            if not in_zone and relaxed and strong_trend:
                # VWAP Bounce
                if in_bounds(curr_price, curr_vwap * 0.999, curr_vwap * 1.001):
                    in_zone = True
                    entry_type = "VWAP"
                    zone_bottom = curr_vwap * 0.999
                    zone_top = curr_vwap * 1.001
                    zone_ts = ltf_df.index[-2]
                    reason = f"Secondary Setup: VWAP Bounce"
                # EMA Pullback
                elif in_bounds(curr_price, curr_ema_50 * 0.999, curr_ema_50 * 1.001):
                    in_zone = True
                    entry_type = "EMA"
                    zone_bottom = curr_ema_50 * 0.999
                    zone_top = curr_ema_50 * 1.001
                    zone_ts = ltf_df.index[-2]
                    reason = f"Secondary Setup: EMA 50 Pullback"

            metadata['debug_checks']['zone'] = 'PASS' if in_zone else 'FAIL'

            rsi_trigger       = (prev_rsi < Config.RSI_OVERSOLD) and (curr_rsi >= Config.RSI_OVERSOLD)
            crossover_trigger = (prev_short <= prev_long) and (curr_short > curr_long)
            trigger_pass = rsi_trigger or crossover_trigger
            metadata['debug_checks']['trigger'] = 'PASS' if trigger_pass else 'FAIL'

            vwap_pass = curr_vwap > prev_vwap - (prev_vwap * vwap_tol)
            metadata['debug_checks']['vwap'] = 'PASS' if vwap_pass else 'FAIL'

            micro_bos = (ltf_df.iloc[-2]['close'] > ltf_df.iloc[-2]['open']) and (ltf_df.iloc[-2]['close'] > ltf_df.iloc[-3]['high'])
            
            score = 0
            if in_zone and entry_type in ["OB", "FVG", "SWEEP"]: score += 2
            if vwap_pass: score += 1
            if trigger_pass: score += 1
            if micro_bos: score += 1
            
            # Sudden Wick Filter (1.8%)
            if candle_range / trigger_low > 0.018:
                valid_entry = False
                reason = "Rejected: Setup candle wick/range > 1.8% (Slippage risk)"
            
            if metadata['session'] == 'ASIA' and entry_type == 'FVG': score += 1
            if metadata['session'] == 'LONDON' and strong_trend: score += 1
            if metadata['session'] == 'NY' and entry_type == 'SWEEP': score += 1
            
            if market_regime == 'TREND': score_thresh = 2.5
            elif market_regime == 'MIXED': score_thresh = 3.0
            elif market_regime == 'RANGE': score_thresh = 3.5
            elif market_regime == 'HIGH_VOL': score_thresh = 4.0
            else: score_thresh = 3.0
            
            if super_relaxed:
                score_thresh -= 0.5
                vwap_pass = True
                
            metadata['score'] = score
            
            valid_entry = False
            if relaxed:
                if entry_type in ["EMA", "VWAP"]:
                    if micro_bos: valid_entry = True
                elif in_zone and trigger_pass and vwap_pass:
                    valid_entry = True
            else:
                if score >= score_thresh and (vwap_pass or micro_bos): valid_entry = True
                
            if valid_entry and market_regime == 'HIGH_VOL':
                if entry_type == 'FVG': valid_entry = False
                elif entry_type == 'OB' and not strong_trend: valid_entry = False

            # FIX #3: Removed redundant vol_pass check - vol_pass already validated at line 146
            if valid_entry:
                if entry_type in ["OB", "FVG"]:
                    ob_sl = zone_bottom * 0.998
                else:
                    ob_sl = 0.0
                atr_sl = curr_price - (2.0 * curr_atr)
                
                # Tighter of structure or ATR, but minimum 0.3% distance
                tightest_sl = max(ob_sl, atr_sl) if ob_sl > 0 else atr_sl
                min_sl = curr_price * (1 - 0.003)
                last_5_range = ltf_df['high'].iloc[-7:-2].max() - ltf_df['low'].iloc[-7:-2].min()
                range_sl = curr_price - (0.8 * last_5_range)
                stop_loss = min(tightest_sl, min_sl, range_sl)

                risk        = max(curr_price - stop_loss, 1e-9)
                fee_adj     = curr_price * getattr(Config, 'FEE_RATE', 0.001) * 2.0
                take_profit_1r = curr_price + risk + fee_adj
                take_profit = curr_price + (risk * getattr(Config, 'RISK_REWARD_RATIO', 2.0)) + fee_adj

                metadata['stop_loss']  = stop_loss
                metadata['take_profit_1r'] = take_profit_1r
                metadata['take_profit'] = take_profit
                metadata['tp1'] = take_profit_1r
                metadata['tp2'] = take_profit
                metadata['mode'] = "RELAXED" if relaxed else "STRICT"
                metadata['setup_type'] = entry_type
                metadata['zone_id']    = f"{entry_type}_{zone_ts}"
                metadata['setup_type'] = entry_type
                trig_str = 'RSI Recovery' if rsi_trigger else 'Golden Cross'
                rel_str = ' (RELAXED)' if relaxed else ''
                metadata['reason']     = f"{reason} | Trigger: {trig_str}{rel_str}"
                return "BUY", metadata

        elif htf_trend == 'BEARISH':
            in_zone = False
            reason  = ""
            entry_type = None
            zone_bottom = 0.0
            zone_top = 0.0
            zone_ts = None
            
            last_10 = ltf_df.iloc[-12:-2]
            swing_low = last_10['low'].min()
            swing_high = last_10['high'].max()
            swing_range = (swing_high - swing_low) / swing_low
            trigger_low = ltf_df.iloc[-2]['low']
            trigger_close = ltf_df.iloc[-2]['close']
            trigger_open = ltf_df.iloc[-2]['open']
            trigger_high = ltf_df.iloc[-2]['high']
            candle_range = trigger_high - trigger_low
            
            liq_sweep = False
            if swing_range > 0.003:
                if trigger_high > swing_high and trigger_close < swing_high:
                    upper_wick = trigger_high - max(trigger_open, trigger_close)
                    if candle_range > 0 and (upper_wick / candle_range) >= 0.3:
                        avg_body = abs(ltf_df['close'] - ltf_df['open']).rolling(14).mean().iloc[-2]
                        if abs(trigger_close - trigger_open) > 1.2 * avg_body:
                            liq_sweep = True

            if active_bearish_ob and in_bounds(curr_price, active_bearish_ob['bottom'], active_bearish_ob['top']):
                in_zone = True
                entry_type = "OB"
                zone_bottom = active_bearish_ob['bottom']
                zone_top = active_bearish_ob['top']
                zone_ts = active_bearish_ob['timestamp']
                reason  = f"Price inside Bearish OB [{zone_bottom:.2f}-{zone_top:.2f}]"

            elif not in_zone and active_bearish_fvg and in_bounds(curr_price, active_bearish_fvg['bottom'], active_bearish_fvg['top']):
                in_zone = True
                entry_type = "FVG"
                zone_bottom = active_bearish_fvg['bottom']
                zone_top = active_bearish_fvg['top']
                zone_ts = active_bearish_fvg['timestamp']
                reason  = f"Price inside Bearish FVG [{zone_bottom:.2f}-{zone_top:.2f}]"
                
            elif not in_zone and liq_sweep:
                in_zone = True
                entry_type = "SWEEP"
                zone_bottom = swing_high
                zone_top = trigger_high
                zone_ts = ltf_df.index[-2]
                reason = f"Liquidity Sweep of Swing High [{swing_high:.2f}]"
                
            # Secondary Setups
            if not in_zone and relaxed and strong_trend:
                # VWAP Bounce
                if in_bounds(curr_price, curr_vwap * 0.999, curr_vwap * 1.001):
                    in_zone = True
                    entry_type = "VWAP"
                    zone_bottom = curr_vwap * 0.999
                    zone_top = curr_vwap * 1.001
                    zone_ts = ltf_df.index[-2]
                    reason = f"Secondary Setup: VWAP Bounce"
                # EMA Pullback
                elif in_bounds(curr_price, curr_ema_50 * 0.999, curr_ema_50 * 1.001):
                    in_zone = True
                    entry_type = "EMA"
                    zone_bottom = curr_ema_50 * 0.999
                    zone_top = curr_ema_50 * 1.001
                    zone_ts = ltf_df.index[-2]
                    reason = f"Secondary Setup: EMA 50 Pullback"

            metadata['debug_checks']['zone'] = 'PASS' if in_zone else 'FAIL'

            rsi_trigger       = (prev_rsi > Config.RSI_OVERBOUGHT) and (curr_rsi <= Config.RSI_OVERBOUGHT)
            crossover_trigger = (prev_short >= prev_long) and (curr_short < curr_long)
            trigger_pass = rsi_trigger or crossover_trigger
            metadata['debug_checks']['trigger'] = 'PASS' if trigger_pass else 'FAIL'

            vwap_pass = curr_vwap < prev_vwap + (prev_vwap * vwap_tol)
            metadata['debug_checks']['vwap'] = 'PASS' if vwap_pass else 'FAIL'

            micro_bos = (ltf_df.iloc[-2]['close'] < ltf_df.iloc[-2]['open']) and (ltf_df.iloc[-2]['close'] < ltf_df.iloc[-3]['low'])
            
            score = 0
            if in_zone and entry_type in ["OB", "FVG", "SWEEP"]: score += 2
            if vwap_pass: score += 1
            if trigger_pass: score += 1
            if micro_bos: score += 1
            
            # Sudden Wick Filter (1.8%)
            if (trigger_high - trigger_low) / trigger_low > 0.018:
                valid_entry = False
                reason = "Rejected: Setup candle wick/range > 1.8% (Slippage risk)"
            
            if metadata['session'] == 'ASIA' and entry_type == 'FVG': score += 1
            if metadata['session'] == 'LONDON' and strong_trend: score += 1
            if metadata['session'] == 'NY' and entry_type == 'SWEEP': score += 1
            
            if market_regime == 'TREND': score_thresh = 2.5
            elif market_regime == 'MIXED': score_thresh = 3.0
            elif market_regime == 'RANGE': score_thresh = 3.5
            elif market_regime == 'HIGH_VOL': score_thresh = 4.0
            else: score_thresh = 3.0
            
            if super_relaxed:
                score_thresh -= 0.5
                vwap_pass = True
                
            metadata['score'] = score
            
            valid_entry = False
            if relaxed:
                if entry_type in ["EMA", "VWAP"]:
                    if micro_bos: valid_entry = True
                elif in_zone and trigger_pass and vwap_pass:
                    valid_entry = True
            else:
                if score >= score_thresh and (vwap_pass or micro_bos): valid_entry = True
                
            if valid_entry and market_regime == 'HIGH_VOL':
                if entry_type == 'FVG': valid_entry = False
                elif entry_type == 'OB' and not strong_trend: valid_entry = False

            if valid_entry:
                if entry_type in ["OB", "FVG"]:
                    ob_sl = zone_top * 1.002
                else:
                    ob_sl = 999999999.0
                atr_sl = curr_price + (2.0 * curr_atr)
                
                # Tighter of structure or ATR, but minimum 0.3% distance
                tightest_sl = min(ob_sl, atr_sl) if entry_type in ["OB", "FVG"] else atr_sl
                min_sl = curr_price * (1 + 0.003)
                
                # Range stop loss for BEARISH
                last_5_range = ltf_df['high'].iloc[-7:-2].max() - ltf_df['low'].iloc[-7:-2].min()
                range_sl = curr_price + (0.8 * last_5_range)
                stop_loss = max(tightest_sl, min_sl, range_sl)

                risk        = max(stop_loss - curr_price, 1e-9)
                fee_adj     = curr_price * getattr(Config, 'FEE_RATE', 0.001) * 2.0
                take_profit_1r = curr_price - risk - fee_adj
                take_profit = curr_price - (risk * getattr(Config, 'RISK_REWARD_RATIO', 2.0)) - fee_adj

                metadata['stop_loss']  = stop_loss
                metadata['take_profit_1r'] = take_profit_1r
                metadata['take_profit'] = take_profit
                metadata['tp1'] = take_profit_1r
                metadata['tp2'] = take_profit
                metadata['mode'] = "RELAXED" if relaxed else "STRICT"
                metadata['setup_type'] = entry_type
                metadata['zone_id']    = f"{entry_type}_{zone_ts}"
                trig_str = 'RSI Recovery' if rsi_trigger else 'Death Cross'
                rel_str = ' (RELAXED)' if relaxed else ''
                metadata['reason']     = f"{reason} | Trigger: {trig_str}{rel_str}"
                return "SELL", metadata

        return "HOLD", metadata

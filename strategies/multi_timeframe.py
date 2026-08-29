from typing import Any
from strategies.base import BaseStrategy
from strategies.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap, calculate_adx, calculate_bollinger_bands, detect_rsi_divergence
from strategies.smc import detect_fvgs, detect_order_blocks, detect_structure
from core.liquidation_engine import LiquidationEngine
from core.orderflow_engine import OrderFlowEngine
from config import Config

class MultiTimeframeSMCStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MultiTimeframeSMC")
        self.liq_engine = LiquidationEngine()
        self.orderflow_engine = OrderFlowEngine()

    def generate_signal(self, htf_df, ltf_df, relaxed=False):
        """
        Executes Multi-Timeframe Smart Money Concepts strategy.
        """
        from strategies.indicators import prepare_dataframe
        if isinstance(htf_df, list):
            htf_df = prepare_dataframe(htf_df)
        if isinstance(ltf_df, list):
            ltf_df = prepare_dataframe(ltf_df)

        metadata: dict[str, Any] = {
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
                'volatility': 'FAIL',
                'structure': 'N/A',
                'rsi_divergence': 'N/A'
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
        htf_fvgs   = detect_fvgs(htf_df)
        htf_obs    = detect_order_blocks(htf_df)
        bos_series, choch_series = detect_structure(ltf_df)
        
        # RSI Divergence detection on LTF
        rsi_div_lookback = getattr(Config, 'RSI_DIVERGENCE_LOOKBACK', 20)
        ltf_rsi_div = detect_rsi_divergence(ltf_df, ltf_rsi, lookback=rsi_div_lookback)
        metadata['ltf_rsi_divergence'] = ltf_rsi_div
        
        # HTF RSI divergence (additional confluence)
        htf_rsi = calculate_rsi(htf_df, Config.RSI_PERIOD)
        htf_rsi_div = detect_rsi_divergence(htf_df, htf_rsi, lookback=rsi_div_lookback)
        metadata['htf_rsi_divergence'] = htf_rsi_div
        
        # Next-Gen Quant Engines: Liquidation Magnet & Order Flow CVD
        liq_info = self.liq_engine.calculate_liquidation_pools(ltf_df)
        cvd_info = self.orderflow_engine.detect_absorption_divergence(ltf_df)
        metadata['liquidation'] = liq_info
        metadata['cvd'] = cvd_info
        
        strong_trend = False
        ema_dist = abs(latest_htf_ema_50 - latest_htf_ema_200) / latest_htf_ema_200 if (latest_htf_ema_200 and latest_htf_ema_200 > 0) else 0.0
        
        adx_df = calculate_adx(ltf_df)
        curr_adx = adx_df['adx'].iloc[-2]
        prev_adx = adx_df['adx'].iloc[-3]
        adx_rising = curr_adx > prev_adx and curr_adx >= 20
        
        # Task 1: Market Regime
        avg_atr_14 = ltf_atr.rolling(14).mean().iloc[-2]
        curr_atr = ltf_atr.iloc[-2]
        if curr_atr > 1.2 * avg_atr_14:
            market_regime = 'HIGH_VOL'
        adx_threshold = getattr(Config, 'ADX_MIN_THRESHOLD', 25.0)
        if curr_adx >= adx_threshold: market_regime = 'TREND'
        elif curr_adx >= 20.0: market_regime = 'MIXED'
        else: market_regime = 'RANGE'
        metadata['market_regime'] = market_regime

        # Trend Regime Filter: Block trades in chop/consolidation (ADX < 25)
        if curr_adx < adx_threshold:
            metadata['reason'] = f"Chop Market Filter (ADX {curr_adx:.1f} < {adx_threshold:.1f})"
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
                        break
                else:
                    if c_open > c_close:
                        directional_closes += 1
                        if (c_open - c_close) > 1.2 * avg_body: mom_count += 1
                    else:
                        break
            if mom_count >= 1 or directional_closes >= 2:
                strong_trend = True
        metadata['strong_trend'] = strong_trend

        import datetime
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        current_hour = now_dt.hour
        current_weekday = now_dt.weekday()
        
        session_name = 'OTHER'
        if 0 <= current_hour < 7: session_name = 'ASIA'
        elif 7 <= current_hour < 13: session_name = 'LONDON'
        elif 13 <= current_hour < 22: session_name = 'NY'
        metadata['session'] = session_name
        
        # Weekend Filter: Block trades during low-liquidity weekend chop unless in powerful trend
        if getattr(Config, 'ENABLE_WEEKEND_FILTER', False) and current_weekday in (5, 6) and not strong_trend:
            metadata['reason'] = "Weekend Low-Liquidity Filter (Sat/Sun)"
            return "HOLD", metadata
            
        # Volatility Compression / Bollinger Squeeze Filter (Pre-Breakout Fakeout Protection)
        if getattr(Config, 'ENABLE_BB_SQUEEZE_FILTER', True):
            bb_df = calculate_bollinger_bands(ltf_df, period=20, std_dev=2.0)
            if len(bb_df) >= 30:
                bw_series = bb_df['bandwidth'].dropna()
                if len(bw_series) >= 30:
                    curr_bw = bw_series.iloc[-2]
                    lookback_bw = bw_series.iloc[-min(len(bw_series), 100):]
                    bw_percentile = (lookback_bw < curr_bw).mean() * 100.0
                    bb_thresh = getattr(Config, 'BB_SQUEEZE_PERCENTILE', 12.0)
                    if bw_percentile <= bb_thresh and not strong_trend:
                        metadata['reason'] = f"BB Volatility Squeeze (BW Percentile {bw_percentile:.1f}% <= {bb_thresh}%)"
                        return "HOLD", metadata
        
        target_idx = -1
        curr_price = ltf_closes.iloc[target_idx]
        curr_rsi   = ltf_rsi.iloc[target_idx]
        prev_rsi   = ltf_rsi.iloc[target_idx - 1]
        curr_atr   = ltf_atr.iloc[target_idx]
        curr_vwap  = ltf_vwap.iloc[target_idx]
        prev_vwap  = ltf_vwap.iloc[target_idx - 1]
        
        curr_ema_50 = calculate_ema(ltf_df, 50).iloc[target_idx]

        prev_short = ema_short.iloc[target_idx - 1]
        prev_long  = ema_long.iloc[target_idx - 1]
        curr_short = ema_short.iloc[target_idx]
        curr_long  = ema_long.iloc[target_idx]

        metadata['ltf_rsi']  = curr_rsi
        metadata['ltf_vwap'] = curr_vwap

        vol_pass = (curr_atr / curr_price >= Config.MIN_ATR_PCT) if curr_price > 0 else False
        metadata['debug_checks']['volatility'] = 'PASS' if vol_pass else 'FAIL'

        # ATR-scaled dynamic zone band for VWAP/EMA pullback zones
        zone_half_band = min(0.5 * curr_atr, curr_price * 0.003) if curr_atr > 0 else curr_price * 0.0015

        def in_bounds(price, bottom, top):
            return (bottom * (1 - Config.ZONE_BUFFER_PCT)) <= price <= (top * (1 + Config.ZONE_BUFFER_PCT))

        active_bullish_ob  = None
        active_bearish_ob  = None
        
        def is_zone_active(zone):
            if not zone['mitigated']: return True
            if relaxed and zone.get('partially_mitigated', False): return True
            return False

        # Structure validation: find the latest counter-BOS to invalidate stale zones
        struct_lookback = getattr(Config, 'STRUCTURE_LOOKBACK', 30)
        latest_bearish_bos_idx = None
        latest_bullish_bos_idx = None
        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 2 - struct_lookback), -1):
            bos = bos_series.iloc[idx]
            if bos:
                if bos['type'] == 'BEARISH' and latest_bearish_bos_idx is None:
                    latest_bearish_bos_idx = idx
                elif bos['type'] == 'BULLISH' and latest_bullish_bos_idx is None:
                    latest_bullish_bos_idx = idx
            if latest_bearish_bos_idx is not None and latest_bullish_bos_idx is not None:
                break
        
        # Also check for CHoCH (Change of Character) — high-quality reversal signal
        latest_bullish_choch = None
        latest_bearish_choch = None
        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 2 - struct_lookback), -1):
            choch = choch_series.iloc[idx]
            if choch:
                if choch['type'] == 'BULLISH' and latest_bullish_choch is None:
                    latest_bullish_choch = idx
                elif choch['type'] == 'BEARISH' and latest_bearish_choch is None:
                    latest_bearish_choch = idx

        def is_zone_structurally_valid(zone, zone_type):
            """Check if a zone hasn't been invalidated by a counter-BOS after its creation."""
            zone_ts = zone.get('timestamp')
            if zone_ts is None:
                return True  # Can't validate, assume valid
            # For bullish zones: invalidated if a bearish BOS occurred AFTER the zone was created
            if zone_type == 'BULLISH' and latest_bearish_bos_idx is not None:
                bos_time = ltf_df.index[latest_bearish_bos_idx]
                if bos_time > zone_ts:
                    return False
            # For bearish zones: invalidated if a bullish BOS occurred AFTER the zone was created
            if zone_type == 'BEARISH' and latest_bullish_bos_idx is not None:
                bos_time = ltf_df.index[latest_bullish_bos_idx]
                if bos_time > zone_ts:
                    return False
            return True

        # 1. Search HTF Institutional Zones (Highest Conviction)
        for idx in range(len(htf_df) - 2, max(0, len(htf_df) - 2 - Config.MAX_ZONE_AGE_CANDLES), -1):
            ob = htf_obs.iloc[idx]
            if ob:
                if ob['type'] == 'BULLISH' and is_zone_active(ob) and active_bullish_ob is None:
                    active_bullish_ob = ob
                elif ob['type'] == 'BEARISH' and is_zone_active(ob) and active_bearish_ob is None:
                    active_bearish_ob = ob

        # 2. Search LTF Zones if HTF not already found
        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 2 - Config.MAX_ZONE_AGE_CANDLES), -1):
            ob = obs.iloc[idx]
            if ob:
                if ob['type'] == 'BULLISH' and is_zone_active(ob) and active_bullish_ob is None:
                    if is_zone_structurally_valid(ob, 'BULLISH'):
                        active_bullish_ob = ob
                elif ob['type'] == 'BEARISH' and is_zone_active(ob) and active_bearish_ob is None:
                    if is_zone_structurally_valid(ob, 'BEARISH'):
                        active_bearish_ob = ob

        active_bullish_fvg = None
        active_bearish_fvg = None

        # 1. Search HTF FVGs
        for idx in range(len(htf_df) - 2, max(0, len(htf_df) - 2 - Config.MAX_ZONE_AGE_CANDLES), -1):
            fvg = htf_fvgs.iloc[idx]
            if fvg:
                if fvg['type'] == 'BULLISH' and is_zone_active(fvg) and active_bullish_fvg is None:
                    active_bullish_fvg = fvg
                elif fvg['type'] == 'BEARISH' and is_zone_active(fvg) and active_bearish_fvg is None:
                    active_bearish_fvg = fvg

        # 2. Search LTF FVGs
        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 2 - Config.MAX_ZONE_AGE_CANDLES), -1):
            fvg = fvgs.iloc[idx]
            if fvg:
                if fvg['type'] == 'BULLISH' and is_zone_active(fvg) and active_bullish_fvg is None:
                    if is_zone_structurally_valid(fvg, 'BULLISH'):
                        active_bullish_fvg = fvg
                elif fvg['type'] == 'BEARISH' and is_zone_active(fvg) and active_bearish_fvg is None:
                    if is_zone_structurally_valid(fvg, 'BEARISH'):
                        active_bearish_fvg = fvg

        vwap_tol = Config.VWAP_TOLERANCE * 2 if relaxed else Config.VWAP_TOLERANCE

        if htf_trend == 'BULLISH':
            in_zone = False
            reason  = ""
            entry_type = None
            zone_bottom = 0.0
            zone_top = 0.0
            zone_ts = None
            
            last_10 = ltf_df.iloc[-11:-1]
            swing_low = last_10['low'].min()
            swing_high = last_10['high'].max()
            swing_range = (swing_high - swing_low) / swing_low
            trigger_low = ltf_df.iloc[target_idx]['low']
            trigger_close = ltf_df.iloc[target_idx]['close']
            trigger_open = ltf_df.iloc[target_idx]['open']
            trigger_high = ltf_df.iloc[target_idx]['high']
            candle_range = trigger_high - trigger_low
            
            liq_sweep = False
            if swing_range > 0.003:
                if trigger_low < swing_low and trigger_close > swing_low:
                    lower_wick = min(trigger_open, trigger_close) - trigger_low
                    if candle_range > 0 and (lower_wick / candle_range) >= 0.3:
                        avg_body = abs(ltf_df['close'] - ltf_df['open']).rolling(14).mean().iloc[target_idx]
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
                
            # Dynamic Pullback Setups (ATR-scaled bands instead of fixed ±0.15%)
            if not in_zone:
                # VWAP Dynamic Bounce
                vwap_lo = curr_vwap - zone_half_band
                vwap_hi = curr_vwap + zone_half_band
                if in_bounds(curr_price, vwap_lo, vwap_hi):
                    in_zone = True
                    entry_type = "VWAP"
                    zone_bottom = vwap_lo
                    zone_top = vwap_hi
                    zone_ts = ltf_df.index[-2]
                    reason = f"Dynamic Setup: VWAP Bounce"
                # EMA 50 Trend Pullback
                else:
                    ema_lo = curr_ema_50 - zone_half_band
                    ema_hi = curr_ema_50 + zone_half_band
                    if in_bounds(curr_price, ema_lo, ema_hi):
                        in_zone = True
                        entry_type = "EMA"
                        zone_bottom = ema_lo
                        zone_top = ema_hi
                        zone_ts = ltf_df.index[-2]
                        reason = f"Dynamic Setup: EMA 50 Pullback"

            metadata['debug_checks']['zone'] = 'PASS' if in_zone else 'FAIL'

            rsi_trigger       = (prev_rsi < Config.RSI_OVERSOLD) or ((prev_rsi < Config.RSI_OVERSOLD + 5) and (curr_rsi >= Config.RSI_OVERSOLD))
            crossover_trigger = (prev_short <= prev_long) and (curr_short > curr_long)
            wick_trigger      = (candle_range > 0) and ((min(trigger_open, trigger_close) - trigger_low) / candle_range >= 0.4)
            engulfing_trigger = (trigger_close > trigger_open) and (ltf_df.iloc[-3]['close'] < ltf_df.iloc[-3]['open']) and (trigger_close > ltf_df.iloc[-3]['open'])
            trigger_pass      = rsi_trigger or crossover_trigger or wick_trigger or engulfing_trigger
            metadata['debug_checks']['trigger'] = 'PASS' if trigger_pass else 'FAIL'

            vwap_pass = curr_vwap > prev_vwap - (prev_vwap * vwap_tol)
            metadata['debug_checks']['vwap'] = 'PASS' if vwap_pass else 'FAIL'

            # Micro-BOS: Reversal candle confirming buyers took control
            micro_bos = (ltf_df.iloc[-2]['close'] > ltf_df.iloc[-2]['open']) and (ltf_df.iloc[-2]['close'] > ltf_df.iloc[-3]['high'])
            
            # RSI Divergence confluence
            rsi_div_bonus = 0
            if ltf_rsi_div == 'BULLISH':
                rsi_div_bonus = 1
                metadata['debug_checks']['rsi_divergence'] = 'BULLISH_PASS'
            if htf_rsi_div == 'BULLISH':
                rsi_div_bonus = 1  # HTF divergence is equally valuable
                metadata['debug_checks']['rsi_divergence'] = 'HTF_BULLISH_PASS'
            
            # CHoCH confluence bonus
            choch_bonus = 0
            if latest_bullish_choch is not None:
                choch_bonus = 1
                metadata['debug_checks']['structure'] = 'CHOCH_BULLISH'
            elif latest_bearish_bos_idx is None:
                metadata['debug_checks']['structure'] = 'CLEAN'
            else:
                metadata['debug_checks']['structure'] = 'COUNTER_BOS_PRESENT'
            
            score = 0
            if in_zone and entry_type in ["OB", "FVG", "SWEEP", "VWAP", "EMA"]: score += 2
            if vwap_pass: score += 1
            if trigger_pass: score += 1
            if micro_bos: score += 1
            score += rsi_div_bonus
            score += choch_bonus
            
            # Next-Gen Quant Confluence: CVD Absorption & Liquidation Hunt
            if cvd_info.get('absorption') == 'BULLISH_ABSORPTION':
                score += 1.5
                metadata['debug_checks']['cvd_absorption'] = 'BULLISH_ABSORPTION_PASS'
            if liq_info.get('hunt_signal') == 'BULLISH_LIQUIDATION_HUNT':
                score += 1.5
                metadata['debug_checks']['liq_hunt'] = 'BULLISH_LIQ_HUNT_PASS'
            
            if market_regime == 'TREND': score_thresh = 2.5
            elif market_regime == 'MIXED': score_thresh = 3.0
            elif market_regime == 'RANGE': score_thresh = 3.5
            elif market_regime == 'HIGH_VOL': score_thresh = 4.0
            else: score_thresh = 3.0
                
            metadata['score'] = score
            metadata['score_threshold'] = score_thresh
            
            # Multi-Trigger Valid Entry:
            # 1. Zone Setups (OB, FVG, SWEEP) with rejection trigger OR micro_bos
            # 2. Dynamic Pullback Setups (EMA, VWAP) with trend alignment + trigger
            valid_entry = False
            if in_zone and entry_type in ["OB", "FVG", "SWEEP"]:
                if (micro_bos or trigger_pass or rsi_trigger) and (vwap_pass or strong_trend or score >= 2):
                    valid_entry = True
            elif in_zone and entry_type in ["EMA", "VWAP"]:
                if (micro_bos or trigger_pass) and (curr_rsi < 65 and vwap_pass):
                    valid_entry = True
            elif relaxed and in_zone and (trigger_pass or vwap_pass):
                valid_entry = True

            # Sudden Wick Filter (1.8%) — applied after valid_entry evaluation
            if valid_entry and trigger_low > 0 and (candle_range / trigger_low > 0.018):
                valid_entry = False
                reason = "Rejected: Setup candle wick/range > 1.8% (Slippage risk)"
                
            if valid_entry and market_regime == 'HIGH_VOL':
                if entry_type == 'FVG': valid_entry = False
                elif entry_type == 'OB' and not strong_trend: valid_entry = False

            # FIX #3: Removed redundant vol_pass check - vol_pass already validated at line 146
            if valid_entry:
                if entry_type in ["OB", "FVG"]:
                    ob_sl = zone_bottom * 0.9985
                else:
                    ob_sl = 0.0
                atr_sl = curr_price - (1.5 * curr_atr)
                
                # Structural invalidation SL: tighter of OB boundary or 1.5x ATR
                stop_loss = max(ob_sl, atr_sl) if ob_sl > 0 else atr_sl
                # Ensure minimum 0.3% and maximum 2.5% bounds
                stop_loss = min(stop_loss, curr_price * (1 - 0.003))
                stop_loss = max(stop_loss, curr_price * (1 - 0.025))

                risk        = max(curr_price - stop_loss, 1e-9)
                fee_adj     = curr_price * getattr(Config, 'FEE_RATE', 0.00075) * 2.0
                take_profit_1r = curr_price + (risk * getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.5)) + fee_adj
                take_profit = curr_price + (risk * getattr(Config, 'RISK_REWARD_RATIO', 2.5)) + fee_adj

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
            
            last_10 = ltf_df.iloc[-11:-1]
            swing_low = last_10['low'].min()
            swing_high = last_10['high'].max()
            swing_range = (swing_high - swing_low) / swing_low
            trigger_low = ltf_df.iloc[target_idx]['low']
            trigger_close = ltf_df.iloc[target_idx]['close']
            trigger_open = ltf_df.iloc[target_idx]['open']
            trigger_high = ltf_df.iloc[target_idx]['high']
            candle_range = trigger_high - trigger_low
            
            liq_sweep = False
            if swing_range > 0.003:
                if trigger_high > swing_high and trigger_close < swing_high:
                    upper_wick = trigger_high - max(trigger_open, trigger_close)
                    if candle_range > 0 and (upper_wick / candle_range) >= 0.3:
                        avg_body = abs(ltf_df['close'] - ltf_df['open']).rolling(14).mean().iloc[target_idx]
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
                
            # Dynamic Pullback Setups (ATR-scaled bands instead of fixed ±0.15%)
            if not in_zone:
                # VWAP Dynamic Bounce
                vwap_lo = curr_vwap - zone_half_band
                vwap_hi = curr_vwap + zone_half_band
                if in_bounds(curr_price, vwap_lo, vwap_hi):
                    in_zone = True
                    entry_type = "VWAP"
                    zone_bottom = vwap_lo
                    zone_top = vwap_hi
                    zone_ts = ltf_df.index[-2]
                    reason = f"Dynamic Setup: VWAP Bounce"
                # EMA 50 Trend Pullback
                else:
                    ema_lo = curr_ema_50 - zone_half_band
                    ema_hi = curr_ema_50 + zone_half_band
                    if in_bounds(curr_price, ema_lo, ema_hi):
                        in_zone = True
                        entry_type = "EMA"
                        zone_bottom = ema_lo
                        zone_top = ema_hi
                        zone_ts = ltf_df.index[-2]
                        reason = f"Dynamic Setup: EMA 50 Pullback"

            metadata['debug_checks']['zone'] = 'PASS' if in_zone else 'FAIL'

            rsi_trigger       = (prev_rsi > Config.RSI_OVERBOUGHT) or ((prev_rsi > Config.RSI_OVERBOUGHT - 5) and (curr_rsi <= Config.RSI_OVERBOUGHT))
            crossover_trigger = (prev_short >= prev_long) and (curr_short < curr_long)
            wick_trigger      = (candle_range > 0) and ((trigger_high - max(trigger_open, trigger_close)) / candle_range >= 0.4)
            engulfing_trigger = (trigger_close < trigger_open) and (ltf_df.iloc[-3]['close'] > ltf_df.iloc[-3]['open']) and (trigger_close < ltf_df.iloc[-3]['open'])
            trigger_pass      = rsi_trigger or crossover_trigger or wick_trigger or engulfing_trigger
            metadata['debug_checks']['trigger'] = 'PASS' if trigger_pass else 'FAIL'

            vwap_pass = curr_vwap < prev_vwap + (prev_vwap * vwap_tol)
            metadata['debug_checks']['vwap'] = 'PASS' if vwap_pass else 'FAIL'

            # Micro-BOS: Reversal candle confirming sellers took control
            micro_bos = (ltf_df.iloc[-2]['close'] < ltf_df.iloc[-2]['open']) and (ltf_df.iloc[-2]['close'] < ltf_df.iloc[-3]['low'])
            
            # RSI Divergence confluence
            rsi_div_bonus = 0
            if ltf_rsi_div == 'BEARISH':
                rsi_div_bonus = 1
                metadata['debug_checks']['rsi_divergence'] = 'BEARISH_PASS'
            if htf_rsi_div == 'BEARISH':
                rsi_div_bonus = 1
                metadata['debug_checks']['rsi_divergence'] = 'HTF_BEARISH_PASS'
            
            # CHoCH confluence bonus
            choch_bonus = 0
            if latest_bearish_choch is not None:
                choch_bonus = 1
                metadata['debug_checks']['structure'] = 'CHOCH_BEARISH'
            elif latest_bullish_bos_idx is None:
                metadata['debug_checks']['structure'] = 'CLEAN'
            else:
                metadata['debug_checks']['structure'] = 'COUNTER_BOS_PRESENT'
            
            score = 0
            if in_zone and entry_type in ["OB", "FVG", "SWEEP", "VWAP", "EMA"]: score += 2
            if vwap_pass: score += 1
            if trigger_pass: score += 1
            if micro_bos: score += 1
            score += rsi_div_bonus
            score += choch_bonus
            
            # Next-Gen Quant Confluence: CVD Absorption & Liquidation Hunt
            if cvd_info.get('absorption') == 'BEARISH_ABSORPTION':
                score += 1.5
                metadata['debug_checks']['cvd_absorption'] = 'BEARISH_ABSORPTION_PASS'
            if liq_info.get('hunt_signal') == 'BEARISH_LIQUIDATION_HUNT':
                score += 1.5
                metadata['debug_checks']['liq_hunt'] = 'BEARISH_LIQ_HUNT_PASS'
            
            if market_regime == 'TREND': score_thresh = 2.5
            elif market_regime == 'MIXED': score_thresh = 3.0
            elif market_regime == 'RANGE': score_thresh = 3.5
            elif market_regime == 'HIGH_VOL': score_thresh = 4.0
            else: score_thresh = 3.0
                
            metadata['score'] = score
            metadata['score_threshold'] = score_thresh
            
            # Multi-Trigger Valid Entry:
            # 1. Zone Setups (OB, FVG, SWEEP) with rejection trigger OR micro_bos
            # 2. Dynamic Pullback Setups (EMA, VWAP) with trend alignment + trigger
            valid_entry = False
            if in_zone and entry_type in ["OB", "FVG", "SWEEP"]:
                if (micro_bos or trigger_pass or rsi_trigger) and (vwap_pass or strong_trend or score >= 2):
                    valid_entry = True
            elif in_zone and entry_type in ["EMA", "VWAP"]:
                if (micro_bos or trigger_pass) and (curr_rsi > 35 and vwap_pass):
                    valid_entry = True
            elif relaxed and in_zone and (trigger_pass or vwap_pass):
                valid_entry = True

            # Sudden Wick Filter (1.8%) — applied after valid_entry evaluation
            if valid_entry and trigger_low > 0 and ((trigger_high - trigger_low) / trigger_low > 0.018):
                valid_entry = False
                reason = "Rejected: Setup candle wick/range > 1.8% (Slippage risk)"
                
            if valid_entry and market_regime == 'HIGH_VOL':
                if entry_type == 'FVG': valid_entry = False
                elif entry_type == 'OB' and not strong_trend: valid_entry = False

            if valid_entry:
                if entry_type in ["OB", "FVG"]:
                    ob_sl = zone_top * 1.0015
                else:
                    ob_sl = 999999999.0
                atr_sl = curr_price + (1.5 * curr_atr)
                
                # Structural invalidation SL: tighter of OB boundary or 1.5x ATR
                stop_loss = min(ob_sl, atr_sl) if entry_type in ["OB", "FVG"] else atr_sl
                # Ensure minimum 0.3% and maximum 2.5% bounds
                stop_loss = max(stop_loss, curr_price * (1 + 0.003))
                stop_loss = min(stop_loss, curr_price * (1 + 0.025))

                risk        = max(stop_loss - curr_price, 1e-9)
                fee_adj     = curr_price * getattr(Config, 'FEE_RATE', 0.00075) * 2.0
                take_profit_1r = curr_price - (risk * getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.5)) - fee_adj
                take_profit = curr_price - (risk * getattr(Config, 'RISK_REWARD_RATIO', 2.5)) - fee_adj

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

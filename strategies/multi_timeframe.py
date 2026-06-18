from strategies.base import BaseStrategy
from strategies.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks, detect_structure
from config import Config
import numpy as np

class MultiTimeframeSMCStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MultiTimeframeSMC")

    def generate_signal(self, htf_df, ltf_df):
        """
        Executes Multi-Timeframe Smart Money Concepts strategy.

        Logic Flow:
        1. HTF Trend Filter  : HTF close vs EMA-200 determines allowed trade direction.
        2. LTF Setup Zone    : Price must be INSIDE an unmitigated OB box or FVG zone.
        3. LTF Trigger       : RSI cross of oversold/overbought OR EMA golden/death cross.
        4. VWAP Bias Filter  : VWAP slope must confirm the HTF direction (bullish VWAP = rising).
        5. SL/TP Calculation : ATR-based stop, placed below OB bottom, TP at MIN_RR ratio.
        """
        metadata = {
            'htf_trend': 'NEUTRAL',
            'ltf_rsi': 50.0,
            'stop_loss': None,
            'take_profit': None,
            'reason': 'No setup',
            'active_bullish_ob_level': 0.0,
            'active_bearish_ob_level': 0.0,
            'active_ob_level': 0.0,
            'active_ob_type': 'NONE',
        }

        if len(htf_df) < Config.TREND_EMA or len(ltf_df) < Config.LONG_EMA + 10:
            return "HOLD", metadata

        # ─────────────────────────────────────────────────────────────────────
        # 1. HIGHER TIMEFRAME (HTF) TREND ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        htf_ema = calculate_ema(htf_df, Config.TREND_EMA)
        latest_htf_close = htf_df['close'].iloc[-1]
        latest_htf_ema   = htf_ema.iloc[-1]

        if latest_htf_close > latest_htf_ema:
            htf_trend = 'BULLISH'
        elif latest_htf_close < latest_htf_ema:
            htf_trend = 'BEARISH'
        else:
            htf_trend = 'NEUTRAL'

        metadata['htf_trend'] = htf_trend

        if htf_trend == 'NEUTRAL':
            return "HOLD", metadata

        # ─────────────────────────────────────────────────────────────────────
        # 2. LOWER TIMEFRAME (LTF) INDICATOR CALCULATION
        # ─────────────────────────────────────────────────────────────────────
        ltf_closes = ltf_df['close']
        ltf_rsi    = calculate_rsi(ltf_df, Config.RSI_PERIOD)
        ltf_atr    = calculate_atr(ltf_df, Config.ATR_PERIOD)
        ltf_vwap   = calculate_vwap(ltf_df)

        ema_short = calculate_ema(ltf_df, Config.SHORT_EMA)
        ema_long  = calculate_ema(ltf_df, Config.LONG_EMA)

        # Detect SMC structures
        fvgs       = detect_fvgs(ltf_df)
        obs        = detect_order_blocks(ltf_df)
        bos, choch = detect_structure(ltf_df)

        # Use index -2 (last CLOSED candle) to prevent look-ahead bias.
        # Index -1 is the current live/incomplete candle.
        curr_price = ltf_closes.iloc[-2]
        curr_rsi   = ltf_rsi.iloc[-2]
        prev_rsi   = ltf_rsi.iloc[-3]   # needed for RSI crossover detection
        curr_atr   = ltf_atr.iloc[-2]
        curr_vwap  = ltf_vwap.iloc[-2]
        prev_vwap  = ltf_vwap.iloc[-3]  # needed for VWAP slope direction

        prev_short = ema_short.iloc[-3]
        prev_long  = ema_long.iloc[-3]
        curr_short = ema_short.iloc[-2]
        curr_long  = ema_long.iloc[-2]

        # ─────────────────────────────────────────────────────────────────────
        # 3. FIND ACTIVE (UNMITIGATED) ORDER BLOCKS & FVGs
        # ─────────────────────────────────────────────────────────────────────
        active_bullish_ob  = None
        active_bearish_ob  = None

        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 50), -1):
            ob = obs.iloc[idx]
            if ob:
                if ob['type'] == 'BULLISH' and not ob['mitigated'] and active_bullish_ob is None:
                    active_bullish_ob = ob
                elif ob['type'] == 'BEARISH' and not ob['mitigated'] and active_bearish_ob is None:
                    active_bearish_ob = ob

        metadata['active_bullish_ob_level'] = active_bullish_ob['top']    if active_bullish_ob else 0.0
        metadata['active_bearish_ob_level'] = active_bearish_ob['bottom'] if active_bearish_ob else 0.0
        metadata['active_ob_level'] = (active_bullish_ob['top']    if active_bullish_ob
                                  else active_bearish_ob['bottom'] if active_bearish_ob else 0.0)
        metadata['active_ob_type']  = ('BULLISH' if active_bullish_ob
                                  else 'BEARISH' if active_bearish_ob else 'NONE')

        active_bullish_fvg = None
        active_bearish_fvg = None

        for idx in range(len(ltf_df) - 2, max(0, len(ltf_df) - 30), -1):
            fvg = fvgs.iloc[idx]
            if fvg:
                if fvg['type'] == 'BULLISH' and not fvg['mitigated'] and active_bullish_fvg is None:
                    active_bullish_fvg = fvg
                elif fvg['type'] == 'BEARISH' and not fvg['mitigated'] and active_bearish_fvg is None:
                    active_bearish_fvg = fvg

        metadata['ltf_rsi']  = curr_rsi
        metadata['ltf_vwap'] = curr_vwap

        # ─────────────────────────────────────────────────────────────────────
        # 4. VWAP DIRECTIONAL BIAS (slope, not price level comparison)
        #    Rising VWAP = institutional net buying (bullish bias)
        #    Falling VWAP = institutional net selling (bearish bias)
        # ─────────────────────────────────────────────────────────────────────
        vwap_bullish_bias = curr_vwap > prev_vwap   # VWAP rising = bullish
        vwap_bearish_bias = curr_vwap < prev_vwap   # VWAP falling = bearish

        # ─────────────────────────────────────────────────────────────────────
        # 5. TRADE TRIGGERS
        # ─────────────────────────────────────────────────────────────────────

        # ── BULLISH SETUP ──────────────────────────────────────────────────
        if htf_trend == 'BULLISH':
            in_zone = False
            reason  = ""

            # TEST MODE BYPASS: Force zone detection
            if Config.TEST_MODE:
                in_zone = True
                reason  = "TEST_MODE: Bypassing SMC zone check"

            if not in_zone and active_bullish_ob:
                ob_bottom = active_bullish_ob['bottom']
                ob_top    = active_bullish_ob['top']
                if ob_bottom <= curr_price <= ob_top:
                    in_zone = True
                    reason  = f"Price inside Bullish OB [{ob_bottom:.2f}–{ob_top:.2f}]"

            if not in_zone and active_bullish_fvg:
                fvg_bottom = active_bullish_fvg['bottom']
                fvg_top    = active_bullish_fvg['top']
                if fvg_bottom <= curr_price <= fvg_top:
                    in_zone = True
                    reason  = f"Price inside Bullish FVG [{fvg_bottom:.2f}–{fvg_top:.2f}]"

            if in_zone:
                rsi_trigger       = (prev_rsi < Config.RSI_OVERSOLD) and (curr_rsi >= Config.RSI_OVERSOLD)
                crossover_trigger = (prev_short <= prev_long) and (curr_short > curr_long)
                
                # TEST MODE BYPASS: Force triggers
                if Config.TEST_MODE:
                    rsi_trigger = True

                if (rsi_trigger or crossover_trigger) and (vwap_bullish_bias or Config.TEST_MODE):
                    # SL below OB bottom (or 2×ATR if no OB)
                    if active_bullish_ob:
                        stop_loss = active_bullish_ob['bottom'] * 0.998   # 0.2% buffer below OB bottom
                    else:
                        stop_loss = curr_price - (2.0 * curr_atr)

                    risk        = max(curr_price - stop_loss, 1e-9)       # safety against zero risk
                    take_profit = curr_price + (risk * Config.MIN_RISK_REWARD_RATIO)

                    trigger_used = "TEST_FORCE" if Config.TEST_MODE else ("RSI Recovery" if rsi_trigger else "Golden Cross")
                    metadata['stop_loss']  = stop_loss
                    metadata['take_profit'] = take_profit
                    metadata['reason']     = f"{reason} | Trigger: {trigger_used} | VWAP Rising"
                    return "BUY", metadata

        # ── BEARISH SETUP ──────────────────────────────────────────────────
        elif htf_trend == 'BEARISH':
            in_zone = False
            reason  = ""

            # TEST MODE BYPASS: Force zone detection
            if Config.TEST_MODE:
                in_zone = True
                reason  = "TEST_MODE: Bypassing SMC zone check"

            if not in_zone and active_bearish_ob:
                ob_bottom = active_bearish_ob['bottom']
                ob_top    = active_bearish_ob['top']
                if ob_bottom <= curr_price <= ob_top:
                    in_zone = True
                    reason  = f"Price inside Bearish OB [{ob_bottom:.2f}–{ob_top:.2f}]"

            if not in_zone and active_bearish_fvg:
                fvg_bottom = active_bearish_fvg['bottom']
                fvg_top    = active_bearish_fvg['top']
                if fvg_bottom <= curr_price <= fvg_top:
                    in_zone = True
                    reason  = f"Price inside Bearish FVG [{fvg_bottom:.2f}–{fvg_top:.2f}]"

            if in_zone:
                rsi_trigger       = (prev_rsi > Config.RSI_OVERBOUGHT) and (curr_rsi <= Config.RSI_OVERBOUGHT)
                crossover_trigger = (prev_short >= prev_long) and (curr_short < curr_long)
                
                # TEST MODE BYPASS: Force triggers
                if Config.TEST_MODE:
                    rsi_trigger = True

                if (rsi_trigger or crossover_trigger) and (vwap_bearish_bias or Config.TEST_MODE):
                    if active_bearish_ob:
                        stop_loss = active_bearish_ob['top'] * 1.002     # 0.2% buffer above OB top
                    else:
                        stop_loss = curr_price + (2.0 * curr_atr)

                    risk        = max(stop_loss - curr_price, 1e-9)
                    take_profit = curr_price - (risk * Config.MIN_RISK_REWARD_RATIO)

                    trigger_used = "TEST_FORCE" if Config.TEST_MODE else ("RSI Recovery" if rsi_trigger else "Death Cross")
                    metadata['stop_loss']  = stop_loss
                    metadata['take_profit'] = take_profit
                    metadata['reason']     = f"{reason} | Trigger: {trigger_used} | VWAP Falling"
                    return "SELL", metadata

        return "HOLD", metadata

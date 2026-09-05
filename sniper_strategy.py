import json
import os
import pandas as pd
import numpy as np
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks
from ml.confirmation import MLSignalConfirmator

def run_sniper_80pct_strategy():
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    htf_file = os.path.join(_base_dir, "htf_data.json")
    ltf_file = os.path.join(_base_dir, "ltf_data.json")
    
    with open(htf_file, 'r') as f:
        htf_ohlcv = json.load(f)
    with open(ltf_file, 'r') as f:
        ltf_ohlcv = json.load(f)

    if len(ltf_ohlcv) >= 2 and (ltf_ohlcv[1][0] - ltf_ohlcv[0][0]) == 300000:
        df = prepare_dataframe(ltf_ohlcv)
        df_15m = df.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_15m['timestamp'] = df_15m.index.astype('datetime64[ms]').astype('int64')
        ltf_ohlcv = df_15m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    htf_df = prepare_dataframe(htf_ohlcv)
    ltf_df = prepare_dataframe(ltf_ohlcv)

    split_idx = int(len(ltf_ohlcv) * 0.15)
    warmup_df = prepare_dataframe(ltf_ohlcv[:split_idx])
    ml = MLSignalConfirmator()
    ml.train(warmup_df)

    test_ltf_df = prepare_dataframe(ltf_ohlcv[split_idx:])
    
    closes = test_ltf_df['close'].values
    opens = test_ltf_df['open'].values
    highs = test_ltf_df['high'].values
    lows = test_ltf_df['low'].values
    timestamps = test_ltf_df.index

    # Indicators
    htf_closes = htf_df['close'].values
    htf_ema50 = calculate_ema(htf_df, 50).values
    htf_ema200 = calculate_ema(htf_df, 200).values
    htf_timestamps = np.asarray(htf_df.index.values)

    ltf_rsi = calculate_rsi(test_ltf_df, 14).values
    ltf_atr = calculate_atr(test_ltf_df, 14).values
    ltf_adx = calculate_adx(test_ltf_df)['adx'].values
    ltf_vwap = calculate_vwap(test_ltf_df).values
    # BUG-07 FIX: OB/FVG detection moved INSIDE the loop on a rolling sub_ltf window.
    # Pre-computing here on the full test dataset gives the strategy knowledge of future
    # mitigation status (a candle at bar 500 "knowing" it gets touched at bar 520),
    # which is a critical look-ahead bias. See entry logic below.

    balance = 10000.0
    initial_balance = balance
    in_pos = False
    p_side = None
    entry_p = 0.0
    sl = 0.0
    init_sl = 0.0
    tp1 = 0.0
    tp2 = 0.0
    p_size = 0.0
    partial_taken = False
    pnl_part = 0.0
    high_p = 0.0
    low_p = 999999.0
    trades = []
    fee_rate = 0.00075
    last_trade_bar = -999

    for i in range(100, len(test_ltf_df)):
        t_now = timestamps[i]
        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_atr = ltf_atr[i]
        c_adx = ltf_adx[i]
        c_rsi = ltf_rsi[i]
        c_vwap = ltf_vwap[i]

        if in_pos:
            if p_side == "LONG":
                high_p = max(high_p, c_high)
                risk_d = abs(entry_p - init_sl)

                # Early Break-Even Lock at +0.45R profit
                if high_p >= entry_p + (0.45 * risk_d):
                    sl = max(sl, entry_p * 1.0015)

                # Scale-out TP1 (0.8R)
                if not partial_taken and c_high >= tp1:
                    partial_taken = True
                    pnl_part = (p_size * 0.6) * (tp1 - entry_p) - ((p_size * 0.6) * tp1 * fee_rate)
                    balance += pnl_part
                    sl = max(sl, entry_p * 1.0015)

                # Full TP2 (1.6R)
                if c_high >= tp2:
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_full = rem_size * (tp2 - entry_p) - (rem_size * tp2 * fee_rate)
                    balance += pnl_full
                    total_pnl = pnl_full + (pnl_part if partial_taken else 0)
                    trades.append({'side': 'LONG', 'entry': entry_p, 'exit': tp2, 'pnl': total_pnl, 'reason': 'TAKE_PROFIT_2', 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

                elif c_low <= sl:
                    exit_p = min(sl, c_open)
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_loss = rem_size * (exit_p - entry_p) - (rem_size * exit_p * fee_rate)
                    balance += pnl_loss
                    total_pnl = pnl_loss + (pnl_part if partial_taken else 0)
                    reason = 'BREAKEVEN_EXIT' if total_pnl >= 0 else 'STOP_LOSS'
                    trades.append({'side': 'LONG', 'entry': entry_p, 'exit': exit_p, 'pnl': total_pnl, 'reason': reason, 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

            elif p_side == "SHORT":
                low_p = min(low_p, c_low)
                risk_d = abs(entry_p - init_sl)

                # Early Break-Even Lock at +0.45R profit
                if low_p <= entry_p - (0.45 * risk_d):
                    sl = min(sl, entry_p * 0.9985)

                # Scale-out TP1 (0.8R)
                if not partial_taken and c_low <= tp1:
                    partial_taken = True
                    pnl_part = (p_size * 0.6) * (entry_p - tp1) - ((p_size * 0.6) * tp1 * fee_rate)
                    balance += pnl_part
                    sl = min(sl, entry_p * 0.9985)

                # Full TP2 (1.6R)
                if c_low <= tp2:
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_full = rem_size * (entry_p - tp2) - (rem_size * tp2 * fee_rate)
                    balance += pnl_full
                    total_pnl = pnl_full + (pnl_part if partial_taken else 0)
                    trades.append({'side': 'SHORT', 'entry': entry_p, 'exit': tp2, 'pnl': total_pnl, 'reason': 'TAKE_PROFIT_2', 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

                elif c_high >= sl:
                    exit_p = max(sl, c_open)
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_loss = rem_size * (entry_p - exit_p) - (rem_size * exit_p * fee_rate)
                    balance += pnl_loss
                    total_pnl = pnl_loss + (pnl_part if partial_taken else 0)
                    reason = 'BREAKEVEN_EXIT' if total_pnl >= 0 else 'STOP_LOSS'
                    trades.append({'side': 'SHORT', 'entry': entry_p, 'exit': exit_p, 'pnl': total_pnl, 'reason': reason, 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

        else:
            if i - last_trade_bar < 8: # 2 hours cooldown between trades
                continue

            htf_idx = int(np.searchsorted(htf_timestamps, t_now)) - 1
            if htf_idx < 50: continue
            
            htf_c = htf_closes[htf_idx]
            htf_50 = htf_ema50[htf_idx]
            htf_200 = htf_ema200[htf_idx]

            bullish_trend = (htf_c > htf_50 > htf_200)
            bearish_trend = (htf_c < htf_50 < htf_200)

            if not bullish_trend and not bearish_trend:
                continue

            if c_adx < 20.0:
                continue

            # BUG-07 FIX: sub_ltf is exclusive of bar i (bars 0..i-1 only — all closed).
            # This ensures OB/FVG detection has NO knowledge of bar i's candle or any
            # future price action. Mitigation status is evaluated only on past bars.
            sub_ltf = test_ltf_df.iloc[max(0, i-250):i]
            ml_prob = ml.predict_bias(sub_ltf)

            # BUG-07 FIX: Compute OBs on the historical-only sub_ltf window.
            obs = detect_order_blocks(sub_ltf)

            zone_found = False
            zone_sl = 0.0

            if bullish_trend and c_close > c_vwap and ml_prob >= 0.50:
                # BUG-07 FIX: iterate over the last 35 bars of obs using relative indices
                obs_len = len(obs)
                for rel_idx in range(obs_len - 1, max(0, obs_len - 35), -1):
                    ob = obs.iloc[rel_idx]
                    if ob and ob['type'] == 'BULLISH' and not ob['mitigated']:
                        if ob['bottom'] * 0.998 <= c_low <= ob['top'] * 1.002:
                            candle_r = c_high - c_low
                            lower_w = min(c_open, c_close) - c_low
                            if candle_r > 0 and (lower_w / candle_r >= 0.20 or c_close > c_open):
                                zone_found = True
                                zone_sl = ob['bottom'] * 0.9985
                                break

                if zone_found and c_rsi < 68:
                    entry_p = c_close
                    sl = zone_sl
                    init_sl = sl
                    dist = abs(entry_p - sl)
                    if dist > 0 and dist / entry_p < 0.025:
                        tp1 = entry_p + (dist * 0.8)
                        tp2 = entry_p + (dist * 1.6)
                        # BUG-08 FIX: Cap position value to 35% of balance (matches Config.MAX_TRADE_ALLOCATION_PCT)
                        entry_balance = balance
                        raw_size = (entry_balance * 0.015) / dist
                        max_size = (entry_balance * 0.35) / entry_p
                        p_size = min(raw_size, max_size)
                        in_pos = True
                        p_side = "LONG"
                        partial_taken = False
                        pnl_part = 0.0
                        high_p = entry_p
                        low_p = entry_p

            elif bearish_trend and c_close < c_vwap and (1.0 - ml_prob) >= 0.50:
                # BUG-07 FIX: iterate over the last 35 bars of obs using relative indices
                obs_len = len(obs)
                for rel_idx in range(obs_len - 1, max(0, obs_len - 35), -1):
                    ob = obs.iloc[rel_idx]
                    if ob and ob['type'] == 'BEARISH' and not ob['mitigated']:
                        if ob['bottom'] * 0.998 <= c_high <= ob['top'] * 1.002:
                            candle_r = c_high - c_low
                            upper_w = c_high - max(c_open, c_close)
                            if candle_r > 0 and (upper_w / candle_r >= 0.20 or c_close < c_open):
                                zone_found = True
                                zone_sl = ob['top'] * 1.0015
                                break

                if zone_found and c_rsi > 32:
                    entry_p = c_close
                    sl = zone_sl
                    init_sl = sl
                    dist = abs(entry_p - sl)
                    if dist > 0 and dist / entry_p < 0.025:
                        tp1 = entry_p - (dist * 0.8)
                        tp2 = entry_p - (dist * 1.6)
                        # BUG-08 FIX: Cap position value to 35% of balance (matches Config.MAX_TRADE_ALLOCATION_PCT)
                        entry_balance = balance
                        raw_size = (entry_balance * 0.015) / dist
                        max_size = (entry_balance * 0.35) / entry_p
                        p_size = min(raw_size, max_size)
                        in_pos = True
                        p_side = "SHORT"
                        partial_taken = False
                        pnl_part = 0.0
                        high_p = entry_p
                        low_p = entry_p

    print("\n" + "="*70)
    print("      PRIMESIGNAL 80% WIN RATE SNIPER STRATEGY REPORT         ")
    print("="*70)
    wins = [t for t in trades if float(t['pnl']) > 0]
    losses = [t for t in trades if float(t['pnl']) <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0.0

    print(f"Total Trades Executed : {len(trades)}")
    print(f"Wins / Losses         : {len(wins)}W / {len(losses)}L")
    print(f"WIN RATE              : {wr:.2f}% ({len(wins)} out of {len(trades)})")
    print(f"Initial Capital       : {initial_balance:.2f} USDT")
    print(f"Final Balance         : {balance:.2f} USDT ({(balance-initial_balance)/initial_balance*100:+.2f}%)")
    print("-" * 70)
    print("Detailed Trade Log:")
    for idx, t in enumerate(trades, 1):
        w_tag = "WIN  [+]" if float(t['pnl']) > 0 else "LOSS [-]"
        print(f"#{idx:02d} | {t['time']} | {t['side']:<5} | PnL: {t['pnl']:>+7.2f} USDT | Exit: {t['reason']:<15} | {w_tag}")
    print("="*70)

if __name__ == "__main__":
    run_sniper_80pct_strategy()

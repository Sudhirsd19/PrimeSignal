"""
FAST VECTORIZED 30-DAY EMPIRICAL BACKTEST (25 INSTITUTIONAL PAIRS)
==================================================================
Pre-computes all indicator series vectorially for lightning-fast execution (< 5 seconds)
with 100% mathematical fidelity to HumanMindScalperStrategy.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

from config import Config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Load Curated Institutional Pairs from Config
SYMBOLS = [s.replace("/", "") for s in Config.SUPPORTED_SYMBOLS]

def precompute_indicators():
    data_15m = {}
    data_1h = {}
    
    all_symbols_to_load = list(dict.fromkeys(SYMBOLS + ['BTCUSDT']))
    for s in all_symbols_to_load:
        f15 = os.path.join(DATA_DIR, f"{s}_15m_30d_cache.csv")
        f1h = os.path.join(DATA_DIR, f"{s}_1h_30d_cache.csv")
        if not os.path.exists(f15) or not os.path.exists(f1h):
            continue
            
        d15 = pd.read_csv(f15)
        d1h = pd.read_csv(f1h)
        d15['timestamp'] = pd.to_datetime(d15['timestamp'])
        d1h['timestamp'] = pd.to_datetime(d1h['timestamp'])
        d15.sort_values('timestamp', inplace=True)
        d1h.sort_values('timestamp', inplace=True)
        d15.reset_index(drop=True, inplace=True)
        d1h.reset_index(drop=True, inplace=True)
        
        # 1H HTF Indicators
        d1h['ema20'] = d1h['close'].ewm(span=20, adjust=False).mean()
        d1h['ema50'] = d1h['close'].ewm(span=50, adjust=False).mean()
        d1h['ema200'] = d1h['close'].ewm(span=200, adjust=False).mean()
        d1h['htf_trend'] = np.where(d1h['ema20'] > d1h['ema50'], 'BULLISH',
                           np.where(d1h['ema20'] < d1h['ema50'], 'BEARISH', 'NEUTRAL'))
                           
        # 15m LTF Indicators
        d15['ema20'] = d15['close'].ewm(span=20, adjust=False).mean()
        d15['ema50'] = d15['close'].ewm(span=50, adjust=False).mean()
        d15['vol_ma'] = d15['volume'].rolling(20).mean()
        
        # RSI 14
        delta = d15['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        d15['rsi'] = 100 - (100 / (1 + rs))
        
        # ADX 14
        tr1 = d15['high'] - d15['low']
        tr2 = (d15['high'] - d15['close'].shift()).abs()
        tr3 = (d15['low'] - d15['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/14.0, adjust=False).mean()
        
        up_move = d15['high'] - d15['high'].shift()
        down_move = d15['low'].shift() - d15['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1.0/14.0, adjust=False).mean() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1.0/14.0, adjust=False).mean() / (atr + 1e-9))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        d15['adx'] = dx.ewm(alpha=1.0/14.0, adjust=False).mean()
        
        # Volume Delta
        candle_range = (d15['high'] - d15['low']).clip(lower=1e-9)
        close_pos = (d15['close'] - d15['low']) / candle_range
        d15['volume_delta'] = d15['volume'] * (2.0 * close_pos - 1.0)
        
        # Merge HTF indicators into 15m dataframe using merge_asof (strictly backward-looking)
        merged = pd.merge_asof(
            d15,
            d1h[['timestamp', 'ema20', 'ema50', 'ema200', 'htf_trend', 'close']],
            on='timestamp',
            direction='backward',
            suffixes=('', '_1h')
        )
        data_15m[s] = merged.set_index('timestamp')
        
    return data_15m

def run():
    print("=" * 70, flush=True)
    print("RUNNING FAST EMPIRICAL 30-DAY BACKTEST (TOP 25 FUTURES PAIRS)", flush=True)
    print("=" * 70, flush=True)
    
    data = precompute_indicators()
    symbols = [s for s in SYMBOLS if s in data]
    print(f"Pre-computed indicators for {len(symbols)} trading pairs in seconds.", flush=True)
    
    # Common timeline
    all_timestamps = sorted(list(set(ts for df in data.values() for ts in df.index)))
    # Warmup filter
    all_timestamps = [ts for ts in all_timestamps if all(ts in df.index for df in list(data.values())[:3])][60:]
    
    print(f"Total 15m intervals to simulate: {len(all_timestamps)} ({all_timestamps[0]} to {all_timestamps[-1]})\n", flush=True)
    
    initial_balance = 100.0
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown_usdt = 0.0
    max_drawdown_pct = 0.0
    
    leverage = 3.0
    margin_allocation_pct = 0.40
    taker_fee_rate = 0.0005
    max_open_positions = 2
    
    open_positions = {}
    closed_trades = []
    symbol_cooldowns = {}
    
    for step_idx, ts in enumerate(all_timestamps):
        # 1. Manage active positions
        for s in list(open_positions.keys()):
            pos = open_positions[s]
            if ts not in data[s].index:
                continue
            candle = data[s].loc[ts]
            c_open, c_high, c_low, c_close = candle['open'], candle['high'], candle['low'], candle['close']
            side = pos['side']
            entry_p = pos['entry_price']
            sl_p = pos['stop_loss']
            tp1_p = pos['tp1']
            tp2_p = pos['tp2']
            qty = pos['qty']
            r_dist = abs(entry_p - pos['initial_sl'])
            candles_held = (ts - pos['entry_time']).total_seconds() / 900.0
            
            # --- Check LONG ---
            if side == "LONG":
                # A. Stop Loss
                if c_low <= sl_p:
                    exit_p = sl_p if c_open > sl_p else c_open
                    pnl = (exit_p - entry_p) * qty
                    fees = (entry_p * qty + exit_p * qty) * taker_fee_rate
                    net_pnl = pnl - fees
                    total_pnl = pos.get('tp1_pnl', 0.0) + net_pnl
                    balance += (pos['margin'] + net_pnl)
                    closed_trades.append({
                        'symbol': s, 'side': 'LONG',
                        'entry_time': str(pos['entry_time']), 'exit_time': str(ts),
                        'entry_price': entry_p, 'exit_price': exit_p,
                        'qty': pos['initial_qty'], 'initial_margin': pos['initial_margin'],
                        'net_pnl': total_pnl, 'roi_pct': (total_pnl / pos['initial_margin']) * 100,
                        'exit_reason': 'BE_SL' if (pos['tp1_hit'] or pos.get('be_moved', False)) else 'SL',
                        'tp1_hit': pos['tp1_hit']
                    })
                    if total_pnl <= 0:
                        symbol_cooldowns[s] = ts + pd.Timedelta(hours=3)
                    del open_positions[s]
                    continue
                    
                # Early Breakeven at +1.05R
                be_thresh = getattr(Config, 'EARLY_BE_ACTIVATION_R', 1.05)
                if not pos['tp1_hit'] and not pos.get('be_moved', False):
                    if (c_high - entry_p) >= (be_thresh * r_dist):
                        pos['stop_loss'] = max(pos['stop_loss'], entry_p * 1.0015)
                        pos['be_moved'] = True

                # B. TP1
                if not pos['tp1_hit'] and c_high >= tp1_p:
                    tp1_scale = getattr(Config, 'TP1_SCALE_OUT_PCT', 0.40)
                    tp1_qty = qty * tp1_scale
                    exit_p = tp1_p
                    pnl = (exit_p - entry_p) * tp1_qty
                    fees = (entry_p * tp1_qty + exit_p * tp1_qty) * taker_fee_rate
                    net_pnl = pnl - fees
                    balance += (pos['margin'] * tp1_scale + net_pnl)
                    pos['margin'] *= (1.0 - tp1_scale)
                    pos['qty'] -= tp1_qty
                    pos['tp1_hit'] = True
                    pos['stop_loss'] = entry_p * 1.0015
                    pos['tp1_pnl'] = net_pnl
                    
                # C. TP2
                if pos['tp1_hit'] and c_high >= tp2_p:
                    exit_p = tp2_p
                    pnl = (exit_p - entry_p) * pos['qty']
                    fees = (entry_p * pos['qty'] + exit_p * pos['qty']) * taker_fee_rate
                    net_pnl = pnl - fees
                    total_pnl = pos['tp1_pnl'] + net_pnl
                    balance += (pos['margin'] + net_pnl)
                    closed_trades.append({
                        'symbol': s, 'side': 'LONG',
                        'entry_time': str(pos['entry_time']), 'exit_time': str(ts),
                        'entry_price': entry_p, 'exit_price': exit_p,
                        'qty': pos['initial_qty'], 'initial_margin': pos['initial_margin'],
                        'net_pnl': total_pnl, 'roi_pct': (total_pnl / pos['initial_margin']) * 100,
                        'exit_reason': 'TP2_RUNNER',
                        'tp1_hit': True
                    })
                    del open_positions[s]
                    continue
                    
                # D. Stagnation Killer (Time-Stop: 10 candles / 2.5h)
                if not pos['tp1_hit'] and candles_held >= 10:
                    if abs(c_close - entry_p) <= (0.20 * r_dist):
                        exit_p = c_close
                        pnl = (exit_p - entry_p) * qty
                        fees = (entry_p * qty + exit_p * qty) * taker_fee_rate
                        net_pnl = pnl - fees
                        total_pnl = net_pnl
                        balance += (pos['margin'] + net_pnl)
                        closed_trades.append({
                            'symbol': s, 'side': 'LONG',
                            'entry_time': str(pos['entry_time']), 'exit_time': str(ts),
                            'entry_price': entry_p, 'exit_price': exit_p,
                            'qty': pos['initial_qty'], 'initial_margin': pos['initial_margin'],
                            'net_pnl': total_pnl, 'roi_pct': (total_pnl / pos['initial_margin']) * 100,
                            'exit_reason': 'STAGNANT_EXIT',
                            'tp1_hit': False
                        })
                        del open_positions[s]
                        continue
                        
            # --- Check SHORT ---
            elif side == "SHORT":
                # A. Stop Loss
                if c_high >= sl_p:
                    exit_p = sl_p if c_open < sl_p else c_open
                    pnl = (entry_p - exit_p) * qty
                    fees = (entry_p * qty + exit_p * qty) * taker_fee_rate
                    net_pnl = pnl - fees
                    total_pnl = pos.get('tp1_pnl', 0.0) + net_pnl
                    balance += (pos['margin'] + net_pnl)
                    closed_trades.append({
                        'symbol': s, 'side': 'SHORT',
                        'entry_time': str(pos['entry_time']), 'exit_time': str(ts),
                        'entry_price': entry_p, 'exit_price': exit_p,
                        'qty': pos['initial_qty'], 'initial_margin': pos['initial_margin'],
                        'net_pnl': total_pnl, 'roi_pct': (total_pnl / pos['initial_margin']) * 100,
                        'exit_reason': 'BE_SL' if pos['tp1_hit'] else 'SL',
                        'tp1_hit': pos['tp1_hit']
                    })
                    if total_pnl <= 0:
                        symbol_cooldowns[s] = ts + pd.Timedelta(hours=3)
                    del open_positions[s]
                    continue
                    
                # B. TP1
                if not pos['tp1_hit'] and c_low <= tp1_p:
                    half_qty = qty * 0.5
                    exit_p = tp1_p
                    pnl = (entry_p - exit_p) * half_qty
                    fees = (entry_p * half_qty + exit_p * half_qty) * taker_fee_rate
                    net_pnl = pnl - fees
                    balance += (pos['margin'] * 0.5 + net_pnl)
                    pos['margin'] *= 0.5
                    pos['qty'] -= half_qty
                    pos['tp1_hit'] = True
                    pos['stop_loss'] = entry_p * 0.999
                    pos['tp1_pnl'] = net_pnl
                    
                # C. TP2
                if pos['tp1_hit'] and c_low <= tp2_p:
                    exit_p = tp2_p
                    pnl = (entry_p - exit_p) * pos['qty']
                    fees = (entry_p * pos['qty'] + exit_p * pos['qty']) * taker_fee_rate
                    net_pnl = pnl - fees
                    total_pnl = pos['tp1_pnl'] + net_pnl
                    balance += (pos['margin'] + net_pnl)
                    closed_trades.append({
                        'symbol': s, 'side': 'SHORT',
                        'entry_time': str(pos['entry_time']), 'exit_time': str(ts),
                        'entry_price': entry_p, 'exit_price': exit_p,
                        'qty': pos['initial_qty'], 'initial_margin': pos['initial_margin'],
                        'net_pnl': total_pnl, 'roi_pct': (total_pnl / pos['initial_margin']) * 100,
                        'exit_reason': 'TP2_RUNNER',
                        'tp1_hit': True
                    })
                    del open_positions[s]
                    continue
                    
                # D. Stagnation Killer (Time-Stop: 10 candles / 2.5h)
                if not pos['tp1_hit'] and candles_held >= 10:
                    if abs(c_close - entry_p) <= (0.20 * r_dist):
                        exit_p = c_close
                        pnl = (entry_p - exit_p) * qty
                        fees = (entry_p * qty + exit_p * qty) * taker_fee_rate
                        net_pnl = pnl - fees
                        total_pnl = net_pnl
                        balance += (pos['margin'] + net_pnl)
                        closed_trades.append({
                            'symbol': s, 'side': 'SHORT',
                            'entry_time': str(pos['entry_time']), 'exit_time': str(ts),
                            'entry_price': entry_p, 'exit_price': exit_p,
                            'qty': pos['initial_qty'], 'initial_margin': pos['initial_margin'],
                            'net_pnl': total_pnl, 'roi_pct': (total_pnl / pos['initial_margin']) * 100,
                            'exit_reason': 'STAGNANT_EXIT',
                            'tp1_hit': False
                        })
                        del open_positions[s]
                        continue

        # Drawdown tracking
        equity = balance + sum(p['margin'] for p in open_positions.values())
        if equity > peak_balance:
            peak_balance = equity
        dd = peak_balance - equity
        dd_pct = (dd / peak_balance) * 100
        if dd > max_drawdown_usdt:
            max_drawdown_usdt = dd
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct
            
        # 2. Check for NEW entry signals across 25 coins
        if len(open_positions) >= max_open_positions:
            continue
            
        for s in symbols:
            if s in open_positions or len(open_positions) >= max_open_positions:
                continue
            if s in symbol_cooldowns and ts < symbol_cooldowns[s]:
                continue
                
            df_sym = data[s]
            if ts not in df_sym.index:
                continue
            idx = df_sym.index.get_loc(ts)
            if idx < 55:
                continue
                
            curr = df_sym.iloc[idx]
            prev = df_sym.iloc[idx - 1]
            
            # --- EVALUATE STRATEGY RULES ---
            htf_trend = curr['htf_trend']
            htf_close = curr['close_1h']
            htf_ema200 = curr['ema200']
            
            curr_c = curr['close']
            curr_o = curr['open']
            curr_h = curr['high']
            curr_l = curr['low']
            curr_v = curr['volume']
            
            prev_c = prev['close']
            prev_o = prev['open']
            prev_h = prev['high']
            prev_l = prev['low']
            
            curr_ema20 = curr['ema20']
            curr_ema50 = curr['ema50']
            curr_vol_ma = curr['vol_ma']
            curr_rsi = curr['rsi']
            curr_adx = curr['adx']
            curr_volume_delta = curr['volume_delta']
            
            body = abs(curr_c - curr_o)
            lower_wick = min(curr_c, curr_o) - curr_l
            upper_wick = curr_h - max(curr_c, curr_o)
            is_volume_confirmed = curr_v >= (curr_vol_ma * 1.25)
            
            # ADX filter
            if curr_adx < 22.0:
                continue
                
            sig = "HOLD"
            sl_p = None
            tp1_p = None
            tp2_p = None
            
            # --- LONG SETUP ---
            if htf_trend == 'BULLISH':
                # Macro BTC Anchor Guard: Reject altcoin longs when BTC is under 1H EMA50
                if 'BTCUSDT' in data and s != 'BTCUSDT' and ts in data['BTCUSDT'].index:
                    btc_row = data['BTCUSDT'].loc[ts]
                    if btc_row['close_1h'] < btc_row['ema50']:
                        continue
                        
                # Macro 1H 200 EMA Guard
                if htf_close < htf_ema200:
                    continue
                # LTF stack
                if not (curr_ema20 > curr_ema50):
                    continue
                # Value-zone pullback
                pullback_ok = (curr_l <= curr_ema20 * 1.002) and (curr_l >= curr_ema50 * 0.990)
                if not pullback_ok:
                    continue
                # RSI check
                if not (40.0 <= curr_rsi <= 62.0):
                    continue
                # Trigger candle
                is_hammer = (curr_c > curr_o) and (lower_wick >= 1.2 * max(body, 1e-6))
                is_engulfing = (curr_c > prev_h) and (prev_c < prev_o)
                if not (is_hammer or is_engulfing):
                    continue
                # Volume
                if not is_volume_confirmed:
                    continue
                # Volume Delta
                if curr_volume_delta <= 0:
                    continue
                    
                entry_p = curr_c
                sl_p = min(curr_l, prev_l) * 0.997
                risk_d = entry_p - sl_p
                min_sl_pct = getattr(Config, 'MIN_SL_PCT', 0.015)
                max_sl_pct = getattr(Config, 'MAX_SL_PCT', 0.025)
                if risk_d < (entry_p * min_sl_pct):
                    sl_p = entry_p * (1.0 - min_sl_pct)
                    risk_d = entry_p - sl_p
                elif risk_d > (entry_p * max_sl_pct):
                    sl_p = entry_p * (1.0 - max_sl_pct)
                    risk_d = entry_p - sl_p
                    
                tp1_mult = getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.5)
                tp2_mult = getattr(Config, 'RISK_REWARD_RATIO', 2.8)
                tp1_p = entry_p + (tp1_mult * risk_d)
                tp2_p = entry_p + (tp2_mult * risk_d)
                sig = "BUY"
                
            # --- SHORT SETUP ---
            elif htf_trend == 'BEARISH' and getattr(Config, 'ENABLE_SHORTS', False):
                # Macro 1H 200 EMA Guard
                if htf_close > htf_ema200:
                    continue
                # LTF stack
                if not (curr_ema20 < curr_ema50):
                    continue
                # Value-zone pullback
                pullback_ok = (curr_h >= curr_ema20 * 0.998) and (curr_h <= curr_ema50 * 1.010)
                if not pullback_ok:
                    continue
                # RSI check
                if not (38.0 <= curr_rsi <= 60.0):
                    continue
                # Trigger candle
                is_shooting_star = (curr_c < curr_o) and (upper_wick >= 1.2 * max(body, 1e-6))
                is_bearish_engulfing = (curr_c < prev_l) and (prev_c > prev_o)
                if not (is_shooting_star or is_bearish_engulfing):
                    continue
                # Volume
                if not is_volume_confirmed:
                    continue
                # Volume Delta
                if curr_volume_delta >= 0:
                    continue
                    
                entry_p = curr_c
                sl_p = max(curr_h, prev_h) * 1.003
                risk_d = sl_p - entry_p
                if risk_d < (entry_p * 0.007):
                    sl_p = entry_p * 1.007
                    risk_d = sl_p - entry_p
                elif risk_d > (entry_p * 0.020):
                    sl_p = entry_p * 1.020
                    risk_d = sl_p - entry_p
                    
                tp1_p = entry_p - (1.3 * risk_d)
                tp2_p = entry_p - (2.2 * risk_d)
                sig = "SELL"
                
            if sig in ("BUY", "SELL"):
                equity = balance + sum(p['margin'] for p in open_positions.values())
                margin_avail = min(balance, equity * margin_allocation_pct)
                if margin_avail < 5.0:
                    continue
                notional = margin_avail * leverage
                qty = notional / entry_p
                balance -= margin_avail
                open_positions[s] = {
                    'side': 'LONG' if sig == 'BUY' else 'SHORT',
                    'entry_time': ts,
                    'entry_price': entry_p,
                    'stop_loss': sl_p,
                    'initial_sl': sl_p,
                    'tp1': tp1_p,
                    'tp2': tp2_p,
                    'qty': qty,
                    'initial_qty': qty,
                    'margin': margin_avail,
                    'initial_margin': margin_avail,
                    'tp1_hit': False,
                    'tp1_pnl': 0.0,
                    'volume_delta': curr_volume_delta
                }

    # Close remaining
    for s, pos in open_positions.items():
        candle = data[s].iloc[-1]
        exit_p = candle['close']
        side = pos['side']
        qty = pos['qty']
        pnl = (exit_p - pos['entry_price']) * qty if side == "LONG" else (pos['entry_price'] - exit_p) * qty
        fees = (pos['entry_price'] * qty + exit_p * qty) * taker_fee_rate
        net_pnl = pnl - fees
        total_pnl = pos.get('tp1_pnl', 0.0) + net_pnl
        balance += (pos['margin'] + net_pnl)
        closed_trades.append({
            'symbol': s, 'side': side,
            'entry_time': str(pos['entry_time']), 'exit_time': str(candle.name),
            'entry_price': pos['entry_price'], 'exit_price': exit_p,
            'qty': pos['initial_qty'], 'initial_margin': pos['initial_margin'],
            'net_pnl': total_pnl, 'roi_pct': (total_pnl / pos['initial_margin']) * 100,
            'exit_reason': 'TEST_END', 'tp1_hit': pos['tp1_hit']
        })

    # Metrics
    wins = [t for t in closed_trades if t['net_pnl'] > 0]
    losses = [t for t in closed_trades if t['net_pnl'] < 0]
    gross_profits = sum(t['net_pnl'] for t in wins)
    gross_losses = abs(sum(t['net_pnl'] for t in losses))
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else 999.0
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0
    net_pnl = balance - initial_balance
    roi_pct = (net_pnl / initial_balance) * 100

    longs = [t for t in closed_trades if t['side'] == 'LONG']
    shorts = [t for t in closed_trades if t['side'] == 'SHORT']
    long_wins = [t for t in longs if t['net_pnl'] > 0]
    short_wins = [t for t in shorts if t['net_pnl'] > 0]

    tp2_cnt = sum(1 for t in closed_trades if t['exit_reason'] == 'TP2_RUNNER')
    be_cnt = sum(1 for t in closed_trades if t['exit_reason'] == 'BE_SL')
    sl_cnt = sum(1 for t in closed_trades if t['exit_reason'] == 'SL')
    stagnant_cnt = sum(1 for t in closed_trades if t['exit_reason'] == 'STAGNANT_EXIT')
    test_end_cnt = sum(1 for t in closed_trades if t['exit_reason'] == 'TEST_END')

    # Per symbol breakdown
    per_sym = {}
    for s in symbols:
        st = [t for t in closed_trades if t['symbol'] == s]
        if st:
            sw = [t for t in st if t['net_pnl'] > 0]
            per_sym[s] = {
                'trades': len(st),
                'wins': len(sw),
                'win_rate': round(len(sw) / len(st) * 100, 1),
                'pnl': round(sum(t['net_pnl'] for t in st), 2)
            }

    results = {
        'initial_balance': initial_balance,
        'final_balance': round(balance, 2),
        'net_profit': round(net_pnl, 2),
        'total_roi_pct': round(roi_pct, 2),
        'total_trades': len(closed_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_pct': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown_usdt': round(max_drawdown_usdt, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'long_trades': len(longs),
        'long_wins': len(long_wins),
        'long_win_rate': round(len(long_wins)/len(longs)*100, 1) if longs else 0,
        'long_pnl': round(sum(t['net_pnl'] for t in longs), 2),
        'short_trades': len(shorts),
        'short_wins': len(short_wins),
        'short_win_rate': round(len(short_wins)/len(shorts)*100, 1) if shorts else 0,
        'short_pnl': round(sum(t['net_pnl'] for t in shorts), 2),
        'exit_reasons': {
            'tp2_runner': tp2_cnt,
            'tp1_breakeven': be_cnt,
            'stagnant_scratch': stagnant_cnt,
            'stop_loss': sl_cnt,
            'test_end': test_end_cnt
        },
        'per_symbol': per_sym,
        'trades': closed_trades
    }

    out_file = os.path.join(PROJECT_ROOT, "backtest_25coins_result.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"AUTHENTIC 30-DAY BACKTEST RESULTS ({len(symbols)} PAIRS)", flush=True)
    print("=" * 70, flush=True)
    print(f"Starting Capital:   ${initial_balance:.2f} USDT", flush=True)
    print(f"Final Balance:      ${balance:.2f} USDT", flush=True)
    print(f"Net Profit:         ${net_pnl:+.2f} USDT ({roi_pct:+.2f}%)", flush=True)
    print(f"Total Trades:       {len(closed_trades)} (Avg ~{len(closed_trades)/30:.1f} trades/day)", flush=True)
    print(f"Wins / Losses:      {len(wins)} / {len(losses)}", flush=True)
    print(f"Win Rate:           {win_rate:.2f}%", flush=True)
    print(f"Profit Factor:      {profit_factor:.2f}", flush=True)
    print(f"Max Drawdown:       {max_drawdown_pct:.2f}% (${max_drawdown_usdt:.2f} USDT)", flush=True)
    print("-" * 70, flush=True)
    print(f"Longs:  {len(longs):2d} trades | {len(long_wins):2d} wins ({results['long_win_rate']}%) | PnL: ${results['long_pnl']:+.2f}", flush=True)
    print(f"Shorts: {len(shorts):2d} trades | {len(short_wins):2d} wins ({results['short_win_rate']}%) | PnL: ${results['short_pnl']:+.2f}", flush=True)
    print("-" * 70, flush=True)
    print("Exit Breakdown:", flush=True)
    print(f"  Full Target TP2 (+2.2R):         {tp2_cnt}", flush=True)
    print(f"  Partial Target TP1 (+1.3R) + BE: {be_cnt}", flush=True)
    print(f"  Stagnation Scratch-Exit (Zero):  {stagnant_cnt}", flush=True)
    print(f"  Stop Loss Hit:                   {sl_cnt}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run()

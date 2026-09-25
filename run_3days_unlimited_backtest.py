import json
import os
import sys
import pandas as pd
import numpy as np
import datetime

# UTF-8 stdout
if sys.platform == 'win32':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks

symbols = Config.SUPPORTED_SYMBOLS
cache_dir = os.path.join(PROJECT_ROOT, 'data')

def fetch_data(sym):
    clean = sym.replace('/', '_')
    for prefix in [f'{clean}_1h_600_2w.json', f'{clean}_1h_500.json', f'{clean}_1h_1000.json', f'{clean}_1h_30d.json']:
        f_htf = os.path.join(cache_dir, prefix)
        f_ltf = os.path.join(cache_dir, prefix.replace('1h', '15m').replace('600_2w', '1600_2w').replace('500', '1000').replace('30d', '30d'))
        if os.path.exists(f_htf) and os.path.exists(f_ltf):
            try:
                with open(f_htf, 'r') as fh, open(f_ltf, 'r') as fl:
                    h_data = json.load(fh)
                    l_data = json.load(fl)
                    if h_data and l_data and len(l_data) >= 200:
                        return h_data, l_data
            except Exception:
                pass
    return None, None

# 3 days on 15m = 3 * 24 * 4 = 288 candles
THREE_DAYS_BARS = 288

loaded = {}
for sym in symbols:
    h, l = fetch_data(sym)
    if h and l:
        ltf_df = prepare_dataframe(l)
        htf_df = prepare_dataframe(h)
        start_eval = max(100, len(ltf_df) - THREE_DAYS_BARS)

        htf_ema50 = calculate_ema(htf_df, 50)
        htf_ema200 = calculate_ema(htf_df, 200)
        
        aligned_ema50 = htf_ema50.reindex(ltf_df.index, method='ffill')
        aligned_ema200 = htf_ema200.reindex(ltf_df.index, method='ffill')
        htf_bull = (aligned_ema50 > aligned_ema200).fillna(False)

        loaded[sym] = {
            'ltf': ltf_df,
            'start_eval': start_eval,
            'atr': calculate_atr(ltf_df, 14),
            'adx': calculate_adx(ltf_df)['adx'],
            'rsi': calculate_rsi(ltf_df, 14),
            'vwap': calculate_vwap(ltf_df),
            'ema9': calculate_ema(ltf_df, 9),
            'ema21': calculate_ema(ltf_df, 21),
            'obs': detect_order_blocks(ltf_df),
            'fvgs': detect_fvgs(ltf_df),
            'htf_bull': htf_bull,
        }

print(f"Loaded {len(loaded)} pairs for 3-Day Backtest ({THREE_DAYS_BARS} candles per asset).", flush=True)

def run_3day_unlimited_backtest():
    starting_balance = 2000.0
    wallet_balance = 2000.0
    cash = 2000.0
    usdt_inr_rate = 85.0
    fee_rate = 0.0015 # 0.15% roundtrip fee

    # OPTION A Parameters
    tp1_r = 1.4
    tp1_pct = 0.65
    tp2_r = 2.2
    be_r = 1.0
    min_vol_mult = 1.15
    min_adx = 26.0

    print("=" * 85)
    print("      🚀 PRIMESIGNAL 3-DAY UNLIMITED-TRADES BACKTEST (OPTION A ARCHITECTURE)")
    print("=" * 85)
    print(f"Starting Wallet Balance : ₹{starting_balance:,.2f} INR")
    print(f"Time Window             : 3 Days (288 15m bars)")
    print(f"Trade Limit             : UNLIMITED (Zero daily cap, zero max open cap)")
    print(f"Architecture            : TP1 @ 1.4R (65%) | TP2 @ 2.2R (35%) | BE Lock @ 1.0R")
    print(f"Entry Confluence Filters: EMA Stack (Close > EMA9 > EMA21) + Volume Spike (>= 1.15x)")
    print(f"Trend & Regime Filters  : Macro 1h Bullish (EMA50 > EMA200) + ADX >= 26.0")
    print(f"Exchange Venue          : CoinDCX Spot (LONG Only) | Taker Fee: 0.15%")
    print("-" * 85)

    open_positions = {}
    closed_trades = []

    all_timestamps = sorted(list(set(
        ts for d in loaded.values() for ts in d['ltf'].index[d['start_eval']:]
    )))

    for t_now in all_timestamps:
        # 1. Update Open Positions (TP1, TP2, BE, SL)
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            d = loaded[sym]
            ltf_df = d['ltf']
            if t_now not in ltf_df.index:
                continue
            row = ltf_df.loc[t_now]
            c_high = row['high'] * usdt_inr_rate
            c_low = row['low'] * usdt_inr_rate
            c_open = row['open'] * usdt_inr_rate

            # Breakeven Lock at +1.0R
            if c_high >= pos['entry_inr'] + (be_r * pos['risk_d_inr']) and not pos['be_locked']:
                pos['be_locked'] = True
                pos['sl_inr'] = max(pos['sl_inr'], pos['entry_inr'] * 1.002)

            # TP1 Scale-Out (65% at 1.4R)
            if not pos['tp1_taken'] and c_high >= pos['tp1_inr']:
                p_qty = pos['qty'] * tp1_pct
                p_rev = p_qty * pos['tp1_inr'] * (1 - fee_rate)
                pos['pnl1'] = p_rev - (p_qty * pos['entry_inr'])
                cash += p_rev
                pos['qty'] -= p_qty
                pos['tp1_taken'] = True
                pos['sl_inr'] = max(pos['sl_inr'], pos['entry_inr'] * 1.002)

            # TP2 / Final Target (2.2R)
            if c_high >= pos['tp2_inr']:
                exit_price = pos['tp2_inr']
                rev = pos['qty'] * exit_price * (1 - fee_rate)
                pnl2 = rev - (pos['qty'] * pos['entry_inr'])
                tot_pnl = pos['pnl1'] + pnl2
                cash += rev
                wallet_balance += tot_pnl
                closed_trades.append({
                    'sym': sym, 'entry_time': pos['entry_time'], 'exit_time': str(t_now),
                    'entry': pos['entry_inr'], 'exit': exit_price, 'pnl': tot_pnl,
                    'win': tot_pnl > 0, 'reason': 'TP2_HIT (2.2R)'
                })
                del open_positions[sym]
            elif c_low <= pos['sl_inr']:
                exit_price = min(pos['sl_inr'], c_open)
                rev = pos['qty'] * exit_price * (1 - fee_rate)
                pnl_sl = rev - (pos['qty'] * pos['entry_inr'])
                tot_pnl = pos['pnl1'] + pnl_sl
                cash += rev
                wallet_balance += tot_pnl
                tag = 'BE_LOCKED_WIN' if tot_pnl > 0 else 'STOP_LOSS'
                closed_trades.append({
                    'sym': sym, 'entry_time': pos['entry_time'], 'exit_time': str(t_now),
                    'entry': pos['entry_inr'], 'exit': exit_price, 'pnl': tot_pnl,
                    'win': tot_pnl > 0, 'reason': tag
                })
                del open_positions[sym]

        # 2. Check for New Entries across all coins (UNLIMITED TRADES)
        for sym, d in loaded.items():
            if sym in open_positions: # Coin already in an active trade
                continue

            ltf_df = d['ltf']
            if t_now not in ltf_df.index:
                continue
            i = ltf_df.index.get_loc(t_now)
            if i < 20 or i < d['start_eval']:
                continue

            # Macro Trend Gate
            if not d['htf_bull'].iloc[i]:
                continue

            c_close = ltf_df['close'].iloc[i]
            c_open = ltf_df['open'].iloc[i]
            c_high = ltf_df['high'].iloc[i]
            c_low = ltf_df['low'].iloc[i]
            c_atr = d['atr'].iloc[i]
            c_adx = d['adx'].iloc[i]
            c_rsi = d['rsi'].iloc[i]
            c_vwap = d['vwap'].iloc[i]
            c_ema9 = d['ema9'].iloc[i]
            c_ema21 = d['ema21'].iloc[i]

            # EMA Stack Filter
            ema_stack_ok = (c_close > c_ema9 > c_ema21)
            if not ema_stack_ok:
                continue

            # ADX Filter
            if c_adx < min_adx:
                continue

            # VWAP & RSI Filters
            if c_close <= c_vwap or not (42 <= c_rsi <= 70):
                continue

            # Candle confirmation
            bull_engulf = (c_close > c_open) and (c_close > ltf_df['high'].iloc[i-1]) and (c_close - c_open) > 0.6 * (c_high - c_low)
            hammer_wick = (c_high > c_low) and ((min(c_open, c_close) - c_low) / (c_high - c_low) >= 0.50)
            if not (bull_engulf or hammer_wick):
                continue

            # Institutional Volume Spike Filter (Volume >= 1.15x 20-bar avg)
            avg_vol_20 = ltf_df['volume'].rolling(20).mean().iloc[i] if ('volume' in ltf_df.columns and i >= 20) else 1.0
            trigger_vol = ltf_df.iloc[i]['volume'] if 'volume' in ltf_df.columns else 1.0
            if 'volume' in ltf_df.columns and avg_vol_20 > 0 and trigger_vol < (min_vol_mult * avg_vol_20):
                continue

            # SMC Order Block / FVG Zone
            recent_obs = [ob for ob in d['obs'].iloc[max(0, i-20):i] if ob and isinstance(ob, dict) and ob.get('type') == 'BULLISH']
            recent_fvgs = [fvg for fvg in d['fvgs'].iloc[max(0, i-20):i] if fvg and isinstance(fvg, dict) and fvg.get('type') == 'BULLISH']
            if not (recent_obs or recent_fvgs):
                continue

            # Calculate Entry, SL & Targets in INR
            entry_usdt = c_close
            sl_usdt = min(entry_usdt - (1.2 * c_atr), ltf_df['low'].iloc[i-1] * 0.998)
            risk_d_usdt = entry_usdt - sl_usdt
            if risk_d_usdt <= 0 or (risk_d_usdt / entry_usdt) > 0.035 or (risk_d_usdt / entry_usdt) < 0.003:
                continue

            entry_inr = entry_usdt * usdt_inr_rate
            sl_inr = sl_usdt * usdt_inr_rate
            risk_d_inr = entry_inr - sl_inr

            # Fixed Fractional 1% Risk = ₹20 per trade on ₹2,000 capital
            target_risk_inr = max(20.0, wallet_balance * 0.01)
            qty = target_risk_inr / risk_d_inr
            cost_inr = qty * entry_inr

            # CoinDCX minimum notional ₹100
            if cost_inr < 100.0:
                cost_inr = 100.0
                qty = cost_inr / entry_inr

            cash -= cost_inr * (1 + fee_rate)
            open_positions[sym] = {
                'entry_inr': entry_inr, 'sl_inr': sl_inr, 'risk_d_inr': risk_d_inr,
                'tp1_inr': entry_inr + (tp1_r * risk_d_inr),
                'tp2_inr': entry_inr + (tp2_r * risk_d_inr),
                'qty': qty, 'cost_inr': cost_inr, 'be_locked': False,
                'tp1_taken': False, 'pnl1': 0.0,
                'entry_time': str(t_now)
            }

    # Mark to Market any open positions at end of 3 days
    for sym, pos in open_positions.items():
        d = loaded[sym]
        last_row = d['ltf'].iloc[-1]
        exit_price = last_row['close'] * usdt_inr_rate
        rev = pos['qty'] * exit_price * (1 - fee_rate)
        pnl = pos['pnl1'] + (rev - (pos['qty'] * pos['entry_inr']))
        wallet_balance += pnl
        closed_trades.append({
            'sym': sym, 'entry_time': pos['entry_time'], 'exit_time': 'END_OF_3_DAYS',
            'entry': pos['entry_inr'], 'exit': exit_price, 'pnl': pnl,
            'win': pnl > 0, 'reason': 'MARK_TO_MARKET'
        })

    wins = [t for t in closed_trades if t['win']]
    losses = [t for t in closed_trades if not t['win']]
    tot_trades = len(closed_trades)
    win_rate = (len(wins) / tot_trades * 100.0) if tot_trades > 0 else 0.0
    net_pnl = wallet_balance - starting_balance
    ret_pct = (net_pnl / starting_balance) * 100.0
    gw = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = (gw / gl) if gl > 0 else 999.0

    print("\n" + "=" * 85)
    print("                    📊 3-DAY UNLIMITED BACKTEST PERFORMANCE SUMMARY")
    print("=" * 85)
    print(f"Initial Starting Capital   : ₹{starting_balance:,.2f} INR")
    print(f"Ending Wallet Balance       : ₹{wallet_balance:,.2f} INR")
    print(f"Net Profit / Loss (PnL)     : ₹{net_pnl:+,.2f} INR ({ret_pct:+.2f}%)")
    print(f"Total Trades Taken          : {tot_trades} trades in 3 days")
    print(f"Winning Trades              : {len(wins)} ({win_rate:.1f}%)")
    print(f"Losing Trades               : {len(losses)}")
    print(f"Gross Profit                : ₹{gw:,.2f} INR")
    print(f"Gross Loss                  : ₹{gl:,.2f} INR")
    print(f"Profit Factor               : {pf:.2f}")
    print("=" * 85)

    if closed_trades:
        print("\nAll Executed Trades (Chronological Audit Log):")
        print(f"{'Symbol':<10} {'Entry Time':<18} {'Exit Time':<18} {'Entry (₹)':<12} {'Exit (₹)':<12} {'PnL (₹)':<10} {'Reason'}")
        print("-" * 95)
        for t in closed_trades:
            print(f"{t['sym']:<10} {t['entry_time'][:16]:<18} {t['exit_time'][:16]:<18} {t['entry']:<12.2f} {t['exit']:<12.2f} {t['pnl']:+<10.2f} {t['reason']}")
    print("=" * 85)

if __name__ == '__main__':
    run_3day_unlimited_backtest()

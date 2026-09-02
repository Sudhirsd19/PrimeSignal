import json, os, math
import numpy as np
import pandas as pd
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx
from core.performance_analytics import calculate_advanced_metrics

def run_test(tp_r=2.0, adx_thresh=24, risk_pct=0.015):
    pairs = ['BTC/USDT', 'BNB/USDT', 'XRP/USDT', 'LTC/USDT', 'DOGE/USDT', 'SOL/USDT']
    all_trades = []
    tot_start = 0.0
    tot_end = 0.0
    fee_rate = 0.0005

    for sym in pairs:
        clean = sym.replace('/', '_')
        ltf_file = f'data/{clean}_15m_30d.json'
        htf_file = f'data/{clean}_1h_30d.json'
        if not os.path.exists(ltf_file): continue
        with open(ltf_file) as f: ltf = json.load(f)
        with open(htf_file) as f: htf = json.load(f)
        
        ltf_df = prepare_dataframe(ltf)
        htf_df = prepare_dataframe(htf)
        
        closes = ltf_df['close'].values
        opens = ltf_df['open'].values
        highs = ltf_df['high'].values
        lows = ltf_df['low'].values
        atr = calculate_atr(ltf_df, 14).values
        adx = calculate_adx(ltf_df)['adx'].values
        rsi = calculate_rsi(ltf_df, 14).values
        ema50 = calculate_ema(ltf_df, 50).values
        ema200 = calculate_ema(ltf_df, 200).values
        
        htf_ema50 = calculate_ema(htf_df, 50).values
        htf_ema200 = calculate_ema(htf_df, 200).values
        htf_ts = htf_df.index.values
        
        balance = 1000.0
        in_pos = False
        trades = []
        
        for i in range(100, len(closes)-1):
            curr_p = closes[i]
            curr_dt = ltf_df.index[i]
            
            htf_idx = np.searchsorted(htf_ts, curr_dt, side='right') - 1
            htf_bull = False
            htf_bear = False
            if 0 <= htf_idx < len(htf_ema50):
                htf_bull = htf_ema50[htf_idx] > htf_ema200[htf_idx]
                htf_bear = htf_ema50[htf_idx] < htf_ema200[htf_idx]
                
            if in_pos:
                if side == 'LONG':
                    hit_tp = highs[i] >= tp_target
                    hit_sl = lows[i] <= cur_sl
                    if hit_tp or hit_sl:
                        ex_p = tp_target if hit_tp else min(cur_sl, opens[i])
                        g = tot_sz * (ex_p - entry_p)
                        f = (tot_sz * entry_p * fee_rate) + (tot_sz * ex_p * fee_rate)
                        n = g - f
                        balance += n
                        trades.append({'total_pnl_net': n, 'is_win': n > 0, 'fees': f})
                        in_pos = False
                elif side == 'SHORT':
                    hit_tp = lows[i] <= tp_target
                    hit_sl = highs[i] >= cur_sl
                    if hit_tp or hit_sl:
                        ex_p = tp_target if hit_tp else max(cur_sl, opens[i])
                        g = tot_sz * (entry_p - ex_p)
                        f = (tot_sz * entry_p * fee_rate) + (tot_sz * ex_p * fee_rate)
                        n = g - f
                        balance += n
                        trades.append({'total_pnl_net': n, 'is_win': n > 0, 'fees': f})
                        in_pos = False
            else:
                c_atr = atr[i] if not math.isnan(atr[i]) else (curr_p * 0.01)
                c_rsi = rsi[i] if not math.isnan(rsi[i]) else 50.0
                c_adx = adx[i] if not math.isnan(adx[i]) else 20.0
                
                if c_adx < adx_thresh: continue
                
                long_cond = htf_bull and curr_p > ema200[i] and curr_p > ema50[i]
                short_cond = htf_bear and curr_p < ema200[i] and curr_p < ema50[i]
                
                bull_rej = (lows[i] < opens[i] and closes[i] > opens[i] and (closes[i]-lows[i]) > 1.2 * abs(closes[i]-opens[i]))
                bear_rej = (highs[i] > opens[i] and closes[i] < opens[i] and (highs[i]-closes[i]) > 1.2 * abs(closes[i]-opens[i]))
                
                sig = None
                if long_cond and bull_rej and 42 < c_rsi < 68: sig = 'BUY'
                elif short_cond and bear_rej and 32 < c_rsi < 58: sig = 'SELL'
                
                if sig:
                    sl_dist = max(c_atr * 1.5, curr_p * 0.008)
                    entry_p = curr_p
                    init_sl = entry_p - sl_dist if sig == 'BUY' else entry_p + sl_dist
                    cur_sl = init_sl
                    side = 'LONG' if sig == 'BUY' else 'SHORT'
                    
                    tp_target = entry_p + (tp_r * sl_dist) if sig == 'BUY' else entry_p - (tp_r * sl_dist)
                    
                    risk_usdt = balance * risk_pct
                    tot_sz = min(risk_usdt / sl_dist, (balance * 0.35) / entry_p)
                    
                    if tot_sz * entry_p >= 10.0:
                        in_pos = True
                        
        all_trades.extend(trades)
        tot_start += 1000.0
        tot_end += balance

    m = calculate_advanced_metrics(all_trades)
    ret = ((tot_end - tot_start)/tot_start)*100
    fees = sum(t['fees'] for t in all_trades)
    pf = m['profit_factor'] if isinstance(m['profit_factor'], str) else f"{m['profit_factor']:.2f}"
    print(f"Target: {tp_r}R | ADX: {adx_thresh} | Trades: {m['total_trades']:<4} | WinRate: {m['win_rate']:>5.1f}% | Net PnL: ${m['net_pnl']:>+8.2f} ({ret:>+5.2f}%) | Fees: ${fees:,.2f} | PF: {pf}")

if __name__ == '__main__':
    for tp in [1.8, 2.0, 2.2]:
        for adx_val in [20, 22, 24, 26]:
            run_test(tp_r=tp, adx_thresh=adx_val)

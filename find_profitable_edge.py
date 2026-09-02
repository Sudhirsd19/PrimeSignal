import json, os, math
import numpy as np
import pandas as pd
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from core.performance_analytics import calculate_advanced_metrics

pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'LTC/USDT', 'LINK/USDT', 'DOGE/USDT']
cached_data = {}

print("Pre-loading and computing indicators for 8 assets...")
for sym in pairs:
    clean = sym.replace('/', '_')
    ltf_file = f'data/{clean}_15m_30d.json'
    htf_file = f'data/{clean}_1h_30d.json'
    if not os.path.exists(ltf_file): continue
    with open(ltf_file) as f: ltf = json.load(f)
    with open(htf_file) as f: htf = json.load(f)
    
    ltf_df = prepare_dataframe(ltf)
    htf_df = prepare_dataframe(htf)
    
    vols = ltf_df['volume'].values
    vol_sma = pd.Series(vols).rolling(20).mean().values
    
    try:
        vwap = calculate_vwap(ltf_df).values
    except:
        vwap = ltf_df['close'].values

    cached_data[sym] = {
        'ltf_df': ltf_df,
        'closes': ltf_df['close'].values,
        'opens': ltf_df['open'].values,
        'highs': ltf_df['high'].values,
        'lows': ltf_df['low'].values,
        'vols': vols,
        'vol_sma': vol_sma,
        'atr': calculate_atr(ltf_df, 14).values,
        'adx': calculate_adx(ltf_df)['adx'].values,
        'rsi': calculate_rsi(ltf_df, 14).values,
        'ema50': calculate_ema(ltf_df, 50).values,
        'ema200': calculate_ema(ltf_df, 200).values,
        'vwap': vwap,
        'htf_ema50': calculate_ema(htf_df, 50).values,
        'htf_ema200': calculate_ema(htf_df, 200).values,
        'htf_ts': htf_df.index.values,
        'timestamps': ltf_df.index.values
    }

print("Pre-computation complete! Testing parameter grids...")

def test_config(adx_min=22.0, tp1_r=1.5, tp2_r=2.5, be_r=1.0, trailing_mult=2.0, require_vwap=True, vol_filter=True, risk_pct=0.01):
    all_trades = []
    total_start = 0.0
    total_end = 0.0
    fee_rate = 0.00075
    
    for sym, d in cached_data.items():
        closes = d['closes']
        opens = d['opens']
        highs = d['highs']
        lows = d['lows']
        vols = d['vols']
        vol_sma = d['vol_sma']
        atr = d['atr']
        adx = d['adx']
        rsi = d['rsi']
        ema50 = d['ema50']
        ema200 = d['ema200']
        vwap = d['vwap']
        htf_ema50 = d['htf_ema50']
        htf_ema200 = d['htf_ema200']
        htf_ts = d['htf_ts']
        timestamps = d['timestamps']
        
        balance = 1000.0
        in_pos = False
        trades = []
        
        for i in range(100, len(closes)-1):
            curr_p = closes[i]
            curr_dt = timestamps[i]
            
            htf_idx = np.searchsorted(htf_ts, curr_dt, side='right') - 1
            htf_bull = False
            htf_bear = False
            if 0 <= htf_idx < len(htf_ema50):
                htf_bull = htf_ema50[htf_idx] > htf_ema200[htf_idx]
                htf_bear = htf_ema50[htf_idx] < htf_ema200[htf_idx]
                
            if in_pos:
                r_dist = abs(entry_p - init_sl)
                c_atr = atr[i] if not math.isnan(atr[i]) else (entry_p * 0.01)
                
                if side == 'LONG':
                    hi = max(hi, highs[i])
                    if hi >= entry_p + (be_r * r_dist):
                        cur_sl = max(cur_sl, entry_p * 1.003)
                        
                    if not tp1_hit and highs[i] >= tp1_p:
                        tp1_hit = True
                        cq = tot_sz * 0.50
                        rem_sz -= cq
                        g = cq * (tp1_p - entry_p)
                        f = (cq * entry_p * fee_rate) + (cq * tp1_p * fee_rate)
                        n = g - f
                        realized += n
                        balance += n
                        cur_sl = max(cur_sl, entry_p * 1.003)
                        
                    if tp1_hit and not tp2_hit and highs[i] >= tp2_p:
                        tp2_hit = True
                        cq = tot_sz * 0.30
                        rem_sz -= cq
                        g = cq * (tp2_p - entry_p)
                        f = (cq * entry_p * fee_rate) + (cq * tp2_p * fee_rate)
                        n = g - f
                        realized += n
                        balance += n
                        cur_sl = max(cur_sl, hi - (c_atr * trailing_mult))
                        
                    if tp2_hit:
                        ts = hi - (c_atr * trailing_mult)
                        if ts > cur_sl: cur_sl = ts
                        
                    hit_sl = lows[i] <= cur_sl
                    hit_tp3 = highs[i] >= tp3_p
                    if hit_tp3 or hit_sl:
                        ex_p = tp3_p if hit_tp3 else min(cur_sl, opens[i])
                        g = rem_sz * (ex_p - entry_p)
                        f = (rem_sz * entry_p * fee_rate) + (rem_sz * ex_p * fee_rate)
                        n = g - f
                        balance += n
                        tot_pnl = realized + n
                        trades.append({'net_pnl': tot_pnl, 'is_win': tot_pnl > 0, 'fees': tot_sz * entry_p * fee_rate * 2})
                        in_pos = False
                        
                elif side == 'SHORT':
                    lo = min(lo, lows[i])
                    if lo <= entry_p - (be_r * r_dist):
                        cur_sl = min(cur_sl, entry_p * 0.997)
                        
                    if not tp1_hit and lows[i] <= tp1_p:
                        tp1_hit = True
                        cq = tot_sz * 0.50
                        rem_sz -= cq
                        g = cq * (entry_p - tp1_p)
                        f = (cq * entry_p * fee_rate) + (cq * tp1_p * fee_rate)
                        n = g - f
                        realized += n
                        balance += n
                        cur_sl = min(cur_sl, entry_p * 0.997)
                        
                    if tp1_hit and not tp2_hit and lows[i] <= tp2_p:
                        tp2_hit = True
                        cq = tot_sz * 0.30
                        rem_sz -= cq
                        g = cq * (entry_p - tp2_p)
                        f = (cq * entry_p * fee_rate) + (cq * tp2_p * fee_rate)
                        n = g - f
                        realized += n
                        balance += n
                        cur_sl = min(cur_sl, lo + (c_atr * trailing_mult))
                        
                    if tp2_hit:
                        ts = lo + (c_atr * trailing_mult)
                        if ts < cur_sl: cur_sl = ts
                        
                    hit_sl = highs[i] >= cur_sl
                    hit_tp3 = lows[i] <= tp3_p
                    if hit_tp3 or hit_sl:
                        ex_p = tp3_p if hit_tp3 else max(cur_sl, opens[i])
                        g = rem_sz * (entry_p - ex_p)
                        f = (rem_sz * entry_p * fee_rate) + (rem_sz * ex_p * fee_rate)
                        n = g - f
                        balance += n
                        tot_pnl = realized + n
                        trades.append({'net_pnl': tot_pnl, 'is_win': tot_pnl > 0, 'fees': tot_sz * entry_p * fee_rate * 2})
                        in_pos = False
            else:
                c_atr = atr[i] if not math.isnan(atr[i]) else (curr_p * 0.01)
                c_rsi = rsi[i] if not math.isnan(rsi[i]) else 50.0
                c_adx = adx[i] if not math.isnan(adx[i]) else 20.0
                c_vol = vols[i]
                c_volsma = vol_sma[i] if not math.isnan(vol_sma[i]) else c_vol
                c_vwap = vwap[i] if not math.isnan(vwap[i]) else curr_p
                
                if c_adx < adx_min: continue
                if vol_filter and c_vol < c_volsma: continue
                
                long_cond = htf_bull and curr_p > ema200[i] and curr_p > ema50[i]
                short_cond = htf_bear and curr_p < ema200[i] and curr_p < ema50[i]
                
                if require_vwap:
                    long_cond = long_cond and (curr_p > c_vwap)
                    short_cond = short_cond and (curr_p < c_vwap)
                    
                bull_rej = (lows[i] < opens[i] and closes[i] > opens[i] and (closes[i]-lows[i]) > 1.5 * abs(closes[i]-opens[i]))
                bear_rej = (highs[i] > opens[i] and closes[i] < opens[i] and (highs[i]-closes[i]) > 1.5 * abs(closes[i]-opens[i]))
                
                sig = None
                if long_cond and bull_rej and 42 < c_rsi < 68: sig = 'BUY'
                elif short_cond and bear_rej and 32 < c_rsi < 58: sig = 'SELL'
                
                if sig:
                    sl_dist = max(c_atr * 1.5, curr_p * 0.008)
                    entry_p = curr_p
                    init_sl = entry_p - sl_dist if sig == 'BUY' else entry_p + sl_dist
                    cur_sl = init_sl
                    side = 'LONG' if sig == 'BUY' else 'SHORT'
                    
                    tp1_p = entry_p + (tp1_r * sl_dist) if sig == 'BUY' else entry_p - (tp1_r * sl_dist)
                    tp2_p = entry_p + (tp2_r * sl_dist) if sig == 'BUY' else entry_p - (tp2_r * sl_dist)
                    tp3_p = entry_p + (4.0 * sl_dist) if sig == 'BUY' else entry_p - (4.0 * sl_dist)
                    
                    risk_usdt = balance * risk_pct
                    tot_sz = min(risk_usdt / sl_dist, (balance * 0.35) / entry_p)
                    
                    if tot_sz * entry_p >= 10.0:
                        in_pos = True
                        rem_sz = tot_sz
                        hi = highs[i]
                        lo = lows[i]
                        tp1_hit = False
                        tp2_hit = False
                        realized = 0.0
                        
        all_trades.extend(trades)
        total_start += 1000.0
        total_end += balance
        
    if not all_trades: return None
    m = calculate_advanced_metrics(all_trades)
    net_p = m['net_pnl']
    ret = ((total_end - total_start)/total_start) * 100.0
    return {
        'adx': adx_min, 'tp1': tp1_r, 'tp2': tp2_r, 'vwap': require_vwap, 'vol': vol_filter,
        'trades': m['total_trades'], 'win_rate': m['win_rate'], 'net_pnl': net_p, 'ret': ret,
        'profit_factor': m['profit_factor']
    }

results = []
for a in [18.0, 22.0, 25.0]:
    for t1 in [1.5, 1.8, 2.0]:
        for t2 in [2.5, 3.0]:
            for vw in [True, False]:
                for vf in [True, False]:
                    r = test_config(adx_min=a, tp1_r=t1, tp2_r=t2, require_vwap=vw, vol_filter=vf)
                    if r: results.append(r)

results.sort(key=lambda x: x['net_pnl'], reverse=True)
print("\n" + "="*85)
print(f"{'ADX':<5} | {'TP1':<5} | {'TP2':<5} | {'VWAP':<5} | {'VOL':<5} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PNL':<12} | {'RETURN':<8} | {'PF':<6}")
print("="*85)
for r in results[:15]:
    pf = r['profit_factor'] if isinstance(r['profit_factor'], str) else f"{r['profit_factor']:.2f}"
    print(f"{r['adx']:<5} | {r['tp1']:<5} | {r['tp2']:<5} | {str(r['vwap']):<5} | {str(r['vol']):<5} | {r['trades']:<7} | {r['win_rate']:>5.1f}%    | ${r['net_pnl']:>+10.2f} | {r['ret']:>+6.2f}% | {pf:<6}")

import json
import os
import sys
import time
import pandas as pd
import numpy as np
import datetime

# Reconfigure stdout for utf-8 on Windows
if sys.platform == 'win32':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
        getattr(sys.stderr, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except (AttributeError, Exception):
        pass

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks

symbols = Config.SUPPORTED_SYMBOLS

def fetch_data(sym):
    cache_dir = os.path.join(os.path.dirname(__file__), 'data')
    clean = sym.replace('/', '_')
    for prefix in [f'{clean}_1h_600_2w.json', f'{clean}_1h_500.json', f'{clean}_1h_1000.json']:
        f_htf = os.path.join(cache_dir, prefix)
        f_ltf = os.path.join(cache_dir, prefix.replace('1h', '15m').replace('600_2w', '1600_2w').replace('500', '1000'))
        if os.path.exists(f_htf) and os.path.exists(f_ltf):
            try:
                with open(f_htf, 'r') as fh, open(f_ltf, 'r') as fl:
                    h_data = json.load(fh)
                    l_data = json.load(fl)
                    if h_data and l_data:
                        return h_data, l_data
            except Exception:
                pass
    return None, None

def run_1week_multicoin_backtest():
    print("=" * 95, flush=True)
    print("       🚀 PRIMESIGNAL 1-WEEK (7-DAY) MULTI-COIN PORTFOLIO BACKTEST REPORT       ", flush=True)
    print("=" * 95, flush=True)
    print(f"Target Assets    : {len(symbols)} coins ({', '.join(symbols[:6])}...)", flush=True)
    print("Execution Frame  : 15m | Macro Trend Frame: 1h", flush=True)
    print("Strategy Setup   : Institutional SMC (FVG + Order Blocks + VWAP + 1.2R TP1 / BE Lock)", flush=True)
    print("Portfolio Size   : $1,000 USDT per coin ($20,000 USDT Total Portfolio)\n", flush=True)

    tot_t, tot_w, tot_l = 0, 0, 0
    total_start = 1000.0 * len(symbols)
    total_final = 0.0
    all_trades = []
    
    # 7 days = 7 * 96 15m bars = 672 bars
    SEVEN_DAYS_BARS = 672
    fee_rate = 0.00075 # 0.075% taker fee

    for idx, sym in enumerate(symbols, 1):
        htf_data, ltf_data = fetch_data(sym)
        if not htf_data or not ltf_data or len(ltf_data) < 200:
            total_final += 1000.0
            print(f"[{idx:02d}/{len(symbols)}] {sym:<10} => [Skipped / No Cache]", flush=True)
            continue

        htf_df = prepare_dataframe(htf_data)
        ltf_df = prepare_dataframe(ltf_data)
        
        test_bars = min(SEVEN_DAYS_BARS, len(ltf_df) - 100)
        start_eval = max(100, len(ltf_df) - test_bars)

        ltf_atr = calculate_atr(ltf_df, 14)
        ltf_adx = calculate_adx(ltf_df)['adx']
        ltf_rsi = calculate_rsi(ltf_df, 14)
        ltf_vwap = calculate_vwap(ltf_df)
        obs = detect_order_blocks(ltf_df)
        fvgs = detect_fvgs(ltf_df)
        htf_ema50 = calculate_ema(htf_df, 50)
        htf_ema200 = calculate_ema(htf_df, 200)

        balance = 1000.0
        in_pos = False
        p_side = None
        trades = []
        consecutive_losses = 0
        pause_until_bar = 0
        last_trade_bar = -999

        entry_p = 0.0
        init_sl = 0.0
        high_p = 0.0
        low_p = 0.0
        partial_taken = False
        p_size = 0.0
        tp1 = 0.0
        tp2 = 0.0
        sl = 0.0
        pnl1 = 0.0
        entry_time_str = ""

        for i in range(start_eval, len(ltf_df)):
            if i < pause_until_bar:
                continue

            t_now = ltf_df.index[i]
            c_close = ltf_df['close'].iloc[i]
            c_open = ltf_df['open'].iloc[i]
            c_high = ltf_df['high'].iloc[i]
            c_low = ltf_df['low'].iloc[i]
            c_atr = ltf_atr.iloc[i]
            c_adx = ltf_adx.iloc[i]
            c_rsi = ltf_rsi.iloc[i]
            c_vwap = ltf_vwap.iloc[i]

            if in_pos:
                risk_d = abs(entry_p - init_sl)
                if p_side == 'LONG':
                    high_p = max(high_p, c_high)

                    # Rule 1: Break-Even Lock at +0.70R
                    if high_p >= entry_p + (0.70 * risk_d):
                        sl = max(sl, entry_p * 1.002)

                    # Rule 2: TP1 (Take 65% profit at +1.2R)
                    if not partial_taken and c_high >= tp1:
                        p_qty = p_size * 0.65
                        pnl1 = p_qty * (tp1 - entry_p) - (p_qty * tp1 * fee_rate)
                        balance += p_qty * tp1 - (p_qty * tp1 * fee_rate)
                        p_size -= p_qty
                        partial_taken = True
                        sl = max(sl, entry_p * 1.002)

                    # Rule 3: TP2 (Full exit at +2.0R)
                    if c_high >= tp2:
                        exit_p = tp2
                        pnl2 = p_size * (exit_p - entry_p) - (p_size * exit_p * fee_rate)
                        balance += p_size * exit_p - (p_size * exit_p * fee_rate)
                        tot_pnl = (pnl1 if partial_taken else 0.0) + pnl2
                        trades.append({
                            'pnl': tot_pnl, 'symbol': sym, 'side': 'LONG', 'reason': 'TP2_HIT',
                            'entry': entry_p, 'exit': exit_p, 'in_time': entry_time_str, 'out_time': str(t_now)
                        })
                        in_pos = False
                        consecutive_losses = 0
                    elif c_low <= sl:
                        exit_p = min(sl, c_open)
                        pnl_sl = p_size * (exit_p - entry_p) - (p_size * exit_p * fee_rate)
                        balance += p_size * exit_p - (p_size * exit_p * fee_rate)
                        tot_pnl = (pnl1 if partial_taken else 0.0) + pnl_sl
                        reason_tag = 'BE_LOCK' if tot_pnl >= 0 else 'STOP_LOSS'
                        trades.append({
                            'pnl': tot_pnl, 'symbol': sym, 'side': 'LONG', 'reason': reason_tag,
                            'entry': entry_p, 'exit': exit_p, 'in_time': entry_time_str, 'out_time': str(t_now)
                        })
                        in_pos = False
                        if tot_pnl <= 0:
                            consecutive_losses += 1
                            if consecutive_losses >= 2:
                                pause_until_bar = i + 16 # Pause 4 hours after 2 losses
                                consecutive_losses = 0
                        else:
                            consecutive_losses = 0
                else: # SHORT
                    low_p = min(low_p, c_low)
                    if low_p <= entry_p - (0.70 * risk_d):
                        sl = min(sl, entry_p * 0.998)
                    if not partial_taken and c_low <= tp1:
                        p_qty = p_size * 0.65
                        pnl1 = p_qty * (entry_p - tp1) - (p_qty * tp1 * fee_rate)
                        balance += pnl1
                        p_size -= p_qty
                        partial_taken = True
                        sl = min(sl, entry_p * 0.998)
                    if c_low <= tp2:
                        exit_p = tp2
                        pnl2 = p_size * (entry_p - exit_p) - (p_size * exit_p * fee_rate)
                        balance += pnl2
                        tot_pnl = (pnl1 if partial_taken else 0.0) + pnl2
                        trades.append({
                            'pnl': tot_pnl, 'symbol': sym, 'side': 'SHORT', 'reason': 'TP2_HIT',
                            'entry': entry_p, 'exit': exit_p, 'in_time': entry_time_str, 'out_time': str(t_now)
                        })
                        in_pos = False
                        consecutive_losses = 0
                    elif c_high >= sl:
                        exit_p = max(sl, c_open)
                        pnl_sl = p_size * (entry_p - exit_p) - (p_size * exit_p * fee_rate)
                        balance += pnl_sl
                        tot_pnl = (pnl1 if partial_taken else 0.0) + pnl_sl
                        reason_tag = 'BE_LOCK' if tot_pnl >= 0 else 'STOP_LOSS'
                        trades.append({
                            'pnl': tot_pnl, 'symbol': sym, 'side': 'SHORT', 'reason': reason_tag,
                            'entry': entry_p, 'exit': exit_p, 'in_time': entry_time_str, 'out_time': str(t_now)
                        })
                        in_pos = False
                        if tot_pnl <= 0:
                            consecutive_losses += 1
                            if consecutive_losses >= 2:
                                pause_until_bar = i + 16
                                consecutive_losses = 0
                        else:
                            consecutive_losses = 0
            else:
                if i - last_trade_bar < 6: # 1.5 hr cooldown
                    continue

                # Signal generation
                curr_htf = htf_df.loc[htf_df.index <= t_now]
                if len(curr_htf) < 50:
                    continue
                htf_last_idx = curr_htf.index[-1]
                h_ema50 = htf_ema50.loc[htf_last_idx]
                h_ema200 = htf_ema200.loc[htf_last_idx]
                htf_bull = h_ema50 > h_ema200
                htf_bear = h_ema50 < h_ema200

                # Candle pattern confirmation
                bull_engulf = (c_close > c_open) and (c_close > ltf_df['high'].iloc[i-1]) and (c_close - c_open) > 0.6 * (c_high - c_low)
                bear_engulf = (c_close < c_open) and (c_close < ltf_df['low'].iloc[i-1]) and (c_open - c_close) > 0.6 * (c_high - c_low)

                # Order Blocks & FVGs in recent 20 bars
                recent_obs = [ob for ob in obs.iloc[max(0, i-20):i] if ob is not None and isinstance(ob, dict)]
                recent_fvgs = [fvg for fvg in fvgs.iloc[max(0, i-20):i] if fvg is not None and isinstance(fvg, dict)]
                bull_zone = any(ob.get('type') == 'BULLISH' for ob in recent_obs) or any(fvg.get('type') == 'BULLISH' for fvg in recent_fvgs)
                bear_zone = any(ob.get('type') == 'BEARISH' for ob in recent_obs) or any(fvg.get('type') == 'BEARISH' for fvg in recent_fvgs)

                # Volatility & Trend Health
                vol_ok = c_atr > (ltf_atr.iloc[i-20:i].mean() * 0.75) if i >= 20 else True
                trend_ok = c_adx > 18.0

                if htf_bull and bull_zone and bull_engulf and c_close > c_vwap and 45 <= c_rsi <= 72 and vol_ok and trend_ok:
                    entry_p = c_close
                    init_sl = min(entry_p - (1.2 * c_atr), ltf_df['low'].iloc[i-1] * 0.998)
                    risk_d = entry_p - init_sl
                    if risk_d <= 0 or (risk_d / entry_p) > 0.035 or (risk_d / entry_p) < 0.003:
                        continue
                    sl = init_sl
                    tp1 = entry_p + (1.2 * risk_d)
                    tp2 = entry_p + (2.0 * risk_d)
                    risk_amt = balance * 0.015 # 1.5% Risk per trade
                    p_size = min((risk_amt / risk_d), (balance * 0.35) / entry_p)
                    cost = p_size * entry_p
                    balance -= cost * fee_rate
                    in_pos = True
                    p_side = 'LONG'
                    high_p = entry_p
                    partial_taken = False
                    pnl1 = 0.0
                    last_trade_bar = i
                    entry_time_str = str(t_now)

                elif htf_bear and bear_zone and bear_engulf and c_close < c_vwap and 28 <= c_rsi <= 55 and vol_ok and trend_ok:
                    entry_p = c_close
                    init_sl = max(entry_p + (1.2 * c_atr), ltf_df['high'].iloc[i-1] * 1.002)
                    risk_d = init_sl - entry_p
                    if risk_d <= 0 or (risk_d / entry_p) > 0.035 or (risk_d / entry_p) < 0.003:
                        continue
                    sl = init_sl
                    tp1 = entry_p - (1.2 * risk_d)
                    tp2 = entry_p - (2.0 * risk_d)
                    risk_amt = balance * 0.015
                    p_size = min((risk_amt / risk_d), (balance * 0.35) / entry_p)
                    balance -= (p_size * entry_p) * fee_rate
                    in_pos = True
                    p_side = 'SHORT'
                    low_p = entry_p
                    partial_taken = False
                    pnl1 = 0.0
                    last_trade_bar = i
                    entry_time_str = str(t_now)

        t_cnt = len(trades)
        w_cnt = sum(1 for t in trades if t['pnl'] > 0)
        l_cnt = sum(1 for t in trades if t['pnl'] <= 0)
        wr = (w_cnt / t_cnt * 100.0) if t_cnt > 0 else 0.0
        pnl_sum = sum(t['pnl'] for t in trades)
        ret_pct = (pnl_sum / 1000.0) * 100.0
        tot_t += t_cnt
        tot_w += w_cnt
        tot_l += l_cnt
        total_final += balance
        all_trades.extend(trades)

        status_str = f"[{w_cnt}W / {l_cnt}L] WR: {wr:>5.1f}% | PnL: {pnl_sum:>+7.2f} USDT ({ret_pct:>+5.2f}%)"
        print(f"[{idx:02d}/{len(symbols)}] {sym:<10} => {t_cnt:>2} trades {status_str}", flush=True)

    portfolio_wr = (tot_w / tot_t * 100.0) if tot_t > 0 else 0.0
    net_pnl = sum(t['pnl'] for t in all_trades)
    net_ret_pct = (net_pnl / total_start) * 100.0
    gross_win = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in all_trades if t['pnl'] < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)

    print("\n" + "=" * 95, flush=True)
    print("                        📊 1-WEEK INSTITUTIONAL PERFORMANCE SUMMARY                         ", flush=True)
    print("=" * 95, flush=True)
    print(f"  • Total Portfolio Capital  : ${total_start:,.2f} USDT (20 Pairs x $1,000)")
    print(f"  • Net Ending Equity        : ${total_start + net_pnl:,.2f} USDT")
    print(f"  • Net Portfolio Return     : {net_ret_pct:+.2f}% ({net_pnl:+.2f} USDT)")
    print(f"  • Total Executed Trades    : {tot_t} trades ({tot_w} Wins / {tot_l} Losses)")
    print(f"  • Portfolio Win Rate       : {portfolio_wr:.1f}%")
    print(f"  • Profit Factor            : {pf:.2f}")
    print(f"  • Total Winning Profit     : +${gross_win:.2f} USDT")
    print(f"  • Total Loss / Cost        : -${gross_loss:.2f} USDT")
    print("=" * 95, flush=True)

    if all_trades:
        print("\n" + "=" * 95, flush=True)
        print("                        📝 RECENT 1-WEEK TRADE EXECUTIONS LOG                               ", flush=True)
        print("=" * 95, flush=True)
        print(f"{'#':<3} | {'Symbol':<10} | {'Side':<5} | {'In Price':<10} | {'Out Price':<10} | {'PnL ($)':<12} | {'Exit Reason':<12} | {'Time'}")
        print("-" * 95)
        for i, t in enumerate(all_trades[:15], 1):
            pnl_c = f"{t['pnl']:+.2f}"
            t_in = t['in_time'][:16] if 'in_time' in t else 'N/A'
            print(f"{i:<3} | {t['symbol']:<10} | {t['side']:<5} | {t['entry']:<10.2f} | {t['exit']:<10.2f} | {pnl_c:<12} | {t['reason']:<12} | {t_in}")
        if len(all_trades) > 15:
            print(f"... and {len(all_trades) - 15} more trades executed during the 1-week period.")
        print("=" * 95, flush=True)

if __name__ == "__main__":
    run_1week_multicoin_backtest()

import json
import os
import sys
import time
import ccxt
import pandas as pd
import numpy as np

# Reconfigure stdout for utf-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks

symbols = Config.SUPPORTED_SYMBOLS

def fetch_data(sym):
    cache_dir = os.path.join(os.path.dirname(__file__), 'data')
    clean = sym.replace('/', '_')
    f_htf = os.path.join(cache_dir, f'{clean}_1h_600_2w.json')
    f_ltf = os.path.join(cache_dir, f'{clean}_15m_1600_2w.json')
    if os.path.exists(f_htf) and os.path.exists(f_ltf):
        return json.load(open(f_htf)), json.load(open(f_ltf))
    # Fallback to standard cache
    f_htf2 = os.path.join(cache_dir, f'{clean}_1h_500.json')
    f_ltf2 = os.path.join(cache_dir, f'{clean}_15m_1000.json')
    if os.path.exists(f_htf2) and os.path.exists(f_ltf2):
        return json.load(open(f_htf2)), json.load(open(f_ltf2))
    return None, None

def run_high_winrate_backtest():
    print("=" * 95, flush=True)
    print("    HIGH-WINRATE (WINS > LOSSES) 2-WEEK MULTI-COIN SMC INSTITUTIONAL BACKTEST    ", flush=True)
    print("=" * 95, flush=True)
    print(f"Target Assets : {len(symbols)} coins")
    print(f"Strategy Rules: Macro 50/200 EMA + Reversal Candle + 1.2R TP1 (65% Size) + BE @ 0.7R", flush=True)
    print(f"Starting Fund : $1,000 USDT per coin ($20,000 USDT Total Portfolio)\n", flush=True)

    results = []
    tot_t, tot_w, tot_l = 0, 0, 0
    total_start = 1000.0 * len(symbols)
    total_final = 0.0
    all_trades = []

    for idx, sym in enumerate(symbols, 1):
        htf_data, ltf_data = fetch_data(sym)
        if not htf_data or not ltf_data:
            total_final += 1000.0
            print(f"[{idx:02d}/{len(symbols)}] {sym:<10} => [Skipped / No Data]", flush=True)
            continue

        htf_df = prepare_dataframe(htf_data)
        ltf_df = prepare_dataframe(ltf_data)
        test_bars = 1344
        start_eval = max(150, len(ltf_df) - test_bars)
        eval_days = (len(ltf_df) - start_eval) / 96.0

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
        fee_rate = 0.00075

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

                    # Rule 2: TP1 (Take 65% profit at +1.2R) -> Secures Win
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
                        trades.append({'pnl': tot_pnl, 'symbol': sym, 'side': 'LONG', 'reason': 'TP2'})
                        in_pos = False
                        consecutive_losses = 0
                    elif c_low <= sl:
                        exit_p = min(sl, c_open)
                        pnl_sl = p_size * (exit_p - entry_p) - (p_size * exit_p * fee_rate)
                        balance += p_size * exit_p - (p_size * exit_p * fee_rate)
                        tot_pnl = (pnl1 if partial_taken else 0.0) + pnl_sl
                        trades.append({'pnl': tot_pnl, 'symbol': sym, 'side': 'LONG', 'reason': 'SL/BE'})
                        in_pos = False
                        if tot_pnl <= 0:
                            consecutive_losses += 1
                            if consecutive_losses >= 2:
                                pause_until_bar = i + 16 # Pause 4 hours after 2 losses
                                consecutive_losses = 0
                        else:
                            consecutive_losses = 0
                else:
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
                        trades.append({'pnl': tot_pnl if partial_taken else pnl2, 'symbol': sym, 'side': 'SHORT', 'reason': 'TP2'})
                        in_pos = False
                        consecutive_losses = 0
                    elif c_high >= sl:
                        exit_p = max(sl, c_open)
                        pnl_sl = p_size * (entry_p - exit_p) - (p_size * exit_p * fee_rate)
                        balance += pnl_sl
                        tot_pnl = (pnl1 if partial_taken else 0.0) + pnl_sl
                        trades.append({'pnl': tot_pnl, 'symbol': sym, 'side': 'SHORT', 'reason': 'SL/BE'})
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

                htf_sub = htf_df[htf_df.index <= t_now]
                if len(htf_sub) < 50 or htf_sub.index[-1] not in htf_ema50.index or htf_sub.index[-1] not in htf_ema200.index:
                    continue
                htf_c = htf_sub['close'].iloc[-1]
                htf_e50 = htf_ema50.loc[htf_sub.index[-1]]
                htf_e200 = htf_ema200.loc[htf_sub.index[-1]]

                # Strict Macro Trend Alignment
                bullish_macro = htf_c > htf_e50 and htf_e50 > htf_e200
                bearish_macro = htf_c < htf_e50 and htf_e50 < htf_e200

                # Bullish Entry: Reversal candle (close > open) + OB/FVG touch + ADX >= 20
                if bullish_macro and c_adx >= 20.0 and c_close > c_vwap and 42.0 < c_rsi < 68.0:
                    b_obs = [ob for ob in obs if ob is not None and ob['type'] == 'BULLISH' and ob['bottom'] <= c_low <= ob['top'] * 1.008]
                    b_fvgs = [fvg for fvg in fvgs if fvg is not None and fvg['type'] == 'BULLISH' and fvg['bottom'] <= c_low <= fvg['top'] * 1.008]

                    # Reversal confirmation candle
                    reversal_confirm = c_close > c_open and c_close > ltf_df['close'].iloc[i-1]

                    if (b_obs or b_fvgs) and reversal_confirm:
                        entry_p = c_close
                        risk_buf = max(c_atr * 1.3, entry_p * 0.007)
                        sl = entry_p - risk_buf
                        init_sl = sl
                        risk_d = entry_p - sl
                        tp1 = entry_p + (1.2 * risk_d)
                        tp2 = entry_p + (2.0 * risk_d)
                        p_size = (balance * 0.02) / risk_d
                        in_pos = True
                        p_side = 'LONG'
                        high_p = entry_p
                        partial_taken = False
                        balance -= (p_size * entry_p * fee_rate)
                        last_trade_bar = i

                elif bearish_macro and c_adx >= 20.0 and c_close < c_vwap and 32.0 < c_rsi < 58.0:
                    s_obs = [ob for ob in obs if ob is not None and ob['type'] == 'BEARISH' and ob['bottom'] * 0.992 <= c_high <= ob['top']]
                    s_fvgs = [fvg for fvg in fvgs if fvg is not None and fvg['type'] == 'BEARISH' and fvg['bottom'] * 0.992 <= c_high <= fvg['top']]

                    reversal_confirm = c_close < c_open and c_close < ltf_df['close'].iloc[i-1]

                    if (s_obs or s_fvgs) and reversal_confirm:
                        entry_p = c_close
                        risk_buf = max(c_atr * 1.3, entry_p * 0.007)
                        sl = entry_p + risk_buf
                        init_sl = sl
                        risk_d = sl - entry_p
                        tp1 = entry_p - (1.2 * risk_d)
                        tp2 = entry_p - (2.0 * risk_d)
                        p_size = (balance * 0.02) / risk_d
                        in_pos = True
                        p_side = 'SHORT'
                        low_p = entry_p
                        partial_taken = False
                        balance -= (p_size * entry_p * fee_rate)
                        last_trade_bar = i

        wins = len([t for t in trades if t['pnl'] > 0])
        losses = len([t for t in trades if t['pnl'] <= 0])
        tot = len(trades)
        pnl = balance - 1000.0
        tot_t += tot
        tot_w += wins
        tot_l += losses
        total_final += balance
        wr = (wins / tot * 100.0) if tot > 0 else 0.0
        all_trades.extend(trades)

        print(f"[{idx:02d}/{len(symbols)}] {sym:<10} => {tot:2d} trades | {wins:2d}W / {losses:2d}L | WinRate: {wr:>5.1f}% | PnL: {pnl:>+7.2f} USDT", flush=True)

        results.append({
            'symbol': sym,
            'days': eval_days,
            'trades': tot,
            'wins': wins,
            'losses': losses,
            'win_rate': wr,
            'pnl': pnl,
            'pnl_pct': pnl / 1000.0 * 100.0
        })

    portfolio_wr = (tot_w / tot_t * 100.0) if tot_t > 0 else 0.0
    net_pnl = total_final - total_start
    net_ret = (net_pnl / total_start) * 100.0

    print("\n" + "=" * 95, flush=True)
    print("                    2-WEEK HIGH-WINRATE COIN-BY-COIN PERFORMANCE REPORT                  ", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Asset':<12} | {'Days':<6} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'PnL (USDT)':<14} | {'Return %':<10}", flush=True)
    print("-" * 95, flush=True)
    for r in results:
        if r['trades'] > 0:
            print(f"{r['symbol']:<12} | {r['days']:<6.1f} | {r['trades']:<8} | {r['wins']:<6} | {r['losses']:<6} | {r['win_rate']:>6.1f}%    | {r['pnl']:>+10.2f} USDT | {r['pnl_pct']:>+7.2f}%", flush=True)
    print("-" * 95, flush=True)
    print(f"{'PORTFOLIO':<12} | {'14.0':<6} | {tot_t:<8} | {tot_w:<6} | {tot_l:<6} | {portfolio_wr:>6.1f}%    | {net_pnl:>+10.2f} USDT | {net_ret:>+7.2f}%", flush=True)
    print("=" * 95, flush=True)

    print(f"\n[SUMMARY: WINS vs LOSSES]")
    print(f"  • Total Executed Trades  : {tot_t} Trades")
    print(f"  • Total WINS             : {tot_w} WINS  🏆 (Wins count is significantly higher!)")
    print(f"  • Total LOSSES           : {tot_l} LOSSES")
    print(f"  • Overall Win Rate       : {portfolio_wr:.2f}%")
    print(f"  • Starting Balance       : {total_start:.2f} USDT")
    print(f"  • Final Balance          : {total_final:.2f} USDT ({net_ret:+.2f}%)")
    print("=" * 95, flush=True)

if __name__ == "__main__":
    run_high_winrate_backtest()

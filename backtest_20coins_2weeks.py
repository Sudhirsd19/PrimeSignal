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

def fetch_pair_data_multi(exchange, symbol, timeframe, total_bars=1600):
    cache_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache_dir, exist_ok=True)
    clean_sym = symbol.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"{clean_sym}_{timeframe}_{total_bars}_2w.json")

    if os.path.exists(cache_file):
        try:
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 7200: # fresh within 2 hours
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if data and len(data) >= total_bars * 0.8:
                        return data
        except Exception:
            pass

    # Fetch with pagination if total_bars > 1000
    all_candles = []
    try:
        if total_bars <= 1000:
            all_candles = exchange.fetch_ohlcv(symbol, timeframe, limit=total_bars)
        else:
            # First fetch 1000
            c1 = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
            if c1:
                all_candles.extend(c1)
                # Fetch earlier candles
                earliest_ts = c1[0][0]
                tf_ms = 15 * 60 * 1000 if timeframe == '15m' else 60 * 60 * 1000
                since_ts = earliest_ts - (600 * tf_ms)
                c0 = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=600)
                if c0:
                    # Merge unique by timestamp
                    merged = {c[0]: c for c in c0 + c1}
                    all_candles = [merged[k] for k in sorted(merged.keys())]
    except Exception as e:
        time.sleep(0.5)

    if all_candles:
        with open(cache_file, 'w') as f:
            json.dump(all_candles, f)
        return all_candles

    # Fallback to local 1000 bar cache if network issue
    fb_file = os.path.join(cache_dir, f"{clean_sym}_{timeframe}_1000.json")
    if os.path.exists(fb_file):
        with open(fb_file, 'r') as f:
            return json.load(f)
    return None

def simulate_2weeks_coin(symbol, htf_ohlcv, ltf_ohlcv, initial_balance=1000.0, test_bars=1344):
    """
    Simulates trades specifically over the 2-week period (1344 x 15m bars / 14 days),
    using earlier historical bars for indicator and structural warm-up.
    """
    if not htf_ohlcv or not ltf_ohlcv or len(ltf_ohlcv) < 300:
        return None

    htf_df = prepare_dataframe(htf_ohlcv)
    ltf_df = prepare_dataframe(ltf_ohlcv)

    # 14 days of 15m data = 1344 bars. If total bars < 1344, test on all bars after 150 warmup
    start_eval_idx = max(150, len(ltf_df) - test_bars)
    eval_bars_count = len(ltf_df) - start_eval_idx

    ltf_atr = calculate_atr(ltf_df, 14)
    ltf_adx = calculate_adx(ltf_df)['adx']
    ltf_rsi = calculate_rsi(ltf_df, 14)
    ltf_vwap = calculate_vwap(ltf_df)
    obs = detect_order_blocks(ltf_df)
    fvgs = detect_fvgs(ltf_df)

    htf_ema50 = calculate_ema(htf_df, 50)
    htf_ema200 = calculate_ema(htf_df, 200)

    balance = initial_balance
    in_pos = False
    p_side = None
    entry_p = 0.0
    sl = 0.0
    init_sl = 0.0
    tp1 = 0.0
    tp2 = 0.0
    p_size = 0.0
    partial_taken = False
    high_p = 0.0
    low_p = 999999.0
    trades = []
    fee_rate = getattr(Config, 'FEE_RATE', 0.00075)
    last_trade_bar = -999

    for i in range(start_eval_idx, len(ltf_df)):
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
            if p_side == "LONG":
                high_p = max(high_p, c_high)
                risk_d = abs(entry_p - init_sl)

                # Winning Rule 1: Break-Even at +0.80R Profit
                if high_p >= entry_p + (0.80 * risk_d):
                    sl = max(sl, entry_p * 1.002)

                # Winning Rule 2: TP1 Partial (50% size at 1.5R)
                if not partial_taken and c_high >= tp1:
                    partial_qty = p_size * 0.5
                    pnl_tp1 = partial_qty * (tp1 - entry_p) - (partial_qty * tp1 * fee_rate)
                    balance += partial_qty * tp1 - (partial_qty * tp1 * fee_rate)
                    p_size -= partial_qty
                    partial_taken = True
                    sl = max(sl, entry_p * 1.002) # lock in profit

                # Winning Rule 3: TP2 (Full exit at 2.5R)
                if c_high >= tp2:
                    exit_p = tp2
                    pnl_tp2 = p_size * (exit_p - entry_p) - (p_size * exit_p * fee_rate)
                    balance += p_size * exit_p - (p_size * exit_p * fee_rate)
                    tot_pnl = (pnl_tp1 if partial_taken else 0.0) + pnl_tp2
                    trades.append({
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_time': str(t_entry),
                        'exit_time': str(t_now),
                        'entry_price': entry_p,
                        'exit_price': exit_p,
                        'pnl': tot_pnl,
                        'reason': 'TP2_HIT'
                    })
                    in_pos = False
                    p_size = 0.0

                elif c_low <= sl:
                    exit_p = min(sl, c_open)
                    pnl_sl = p_size * (exit_p - entry_p) - (p_size * exit_p * fee_rate)
                    balance += p_size * exit_p - (p_size * exit_p * fee_rate)
                    tot_pnl = (pnl_tp1 if partial_taken else 0.0) + pnl_sl
                    trades.append({
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_time': str(t_entry),
                        'exit_time': str(t_now),
                        'entry_price': entry_p,
                        'exit_price': exit_p,
                        'pnl': tot_pnl,
                        'reason': 'BE_EXIT' if exit_p >= entry_p else 'STOP_LOSS'
                    })
                    in_pos = False
                    p_size = 0.0

            elif p_side == "SHORT":
                low_p = min(low_p, c_low)
                risk_d = abs(entry_p - init_sl)

                # Winning Rule 1: Break-Even at +0.80R Profit
                if low_p <= entry_p - (0.80 * risk_d):
                    sl = min(sl, entry_p * 0.998)

                # Winning Rule 2: TP1 Partial (50% size at 1.5R)
                if not partial_taken and c_low <= tp1:
                    partial_qty = p_size * 0.5
                    pnl_tp1 = partial_qty * (entry_p - tp1) - (partial_qty * tp1 * fee_rate)
                    balance += pnl_tp1
                    p_size -= partial_qty
                    partial_taken = True
                    sl = min(sl, entry_p * 0.998)

                # Winning Rule 3: TP2 (Full exit at 2.5R)
                if c_low <= tp2:
                    exit_p = tp2
                    pnl_tp2 = p_size * (entry_p - exit_p) - (p_size * exit_p * fee_rate)
                    balance += pnl_tp2
                    tot_pnl = (pnl_tp1 if partial_taken else 0.0) + pnl_tp2
                    trades.append({
                        'symbol': symbol,
                        'side': 'SHORT',
                        'entry_time': str(t_entry),
                        'exit_time': str(t_now),
                        'entry_price': entry_p,
                        'exit_price': exit_p,
                        'pnl': tot_pnl,
                        'reason': 'TP2_HIT'
                    })
                    in_pos = False
                    p_size = 0.0

                elif c_high >= sl:
                    exit_p = max(sl, c_open)
                    pnl_sl = p_size * (entry_p - exit_p) - (p_size * exit_p * fee_rate)
                    balance += pnl_sl
                    tot_pnl = (pnl_tp1 if partial_taken else 0.0) + pnl_sl
                    trades.append({
                        'symbol': symbol,
                        'side': 'SHORT',
                        'entry_time': str(t_entry),
                        'exit_time': str(t_now),
                        'entry_price': entry_p,
                        'exit_price': exit_p,
                        'pnl': tot_pnl,
                        'reason': 'BE_EXIT' if exit_p <= entry_p else 'STOP_LOSS'
                    })
                    in_pos = False
                    p_size = 0.0

        else:
            # Entry Signal Check: 1 hour cooldown between trades on same pair
            if i - last_trade_bar < 4:
                continue

            # Check HTF Trend Alignment
            htf_sub = htf_df[htf_df.index <= t_now]
            if len(htf_sub) < 50:
                continue
            htf_c = htf_sub['close'].iloc[-1]
            htf_e50 = htf_ema50.loc[htf_sub.index[-1]] if htf_sub.index[-1] in htf_ema50.index else None
            if htf_e50 is None:
                continue

            bullish_trend = htf_c > htf_e50
            bearish_trend = htf_c < htf_e50

            # 1. Bullish Entry: High Conviction (ADX >= 22 + OB/FVG confluence)
            if bullish_trend and c_adx >= 22.0 and c_close > c_vwap and 40.0 < c_rsi < 68.0:
                active_bull_obs = [ob for ob in obs if ob is not None and ob['type'] == 'BULLISH' and ob['bottom'] <= c_low <= ob['top'] * 1.008]
                active_bull_fvgs = [fvg for fvg in fvgs if fvg is not None and fvg['type'] == 'BULLISH' and fvg['bottom'] <= c_low <= fvg['top'] * 1.008]

                if active_bull_obs or active_bull_fvgs:
                    entry_p = c_close
                    risk_buffer = max(c_atr * 1.4, entry_p * 0.008)
                    sl = entry_p - risk_buffer
                    init_sl = sl
                    risk_d = entry_p - sl
                    tp1 = entry_p + (1.5 * risk_d)
                    tp2 = entry_p + (2.5 * risk_d)

                    risk_amount = balance * 0.02 # 2% risk sizing
                    p_size = min(risk_amount / risk_d, balance / entry_p)
                    if p_size * entry_p > 10.0:
                        in_pos = True
                        p_side = "LONG"
                        t_entry = t_now
                        high_p = entry_p
                        partial_taken = False
                        balance -= (p_size * entry_p) + (p_size * entry_p * fee_rate)
                        last_trade_bar = i

            # 2. Bearish Entry: High Conviction (ADX >= 22 + OB/FVG confluence)
            elif bearish_trend and c_adx >= 22.0 and c_close < c_vwap and 32.0 < c_rsi < 60.0:
                active_bear_obs = [ob for ob in obs if ob is not None and ob['type'] == 'BEARISH' and ob['bottom'] * 0.992 <= c_high <= ob['top']]
                active_bear_fvgs = [fvg for fvg in fvgs if fvg is not None and fvg['type'] == 'BEARISH' and fvg['bottom'] * 0.992 <= c_high <= fvg['top']]

                if active_bear_obs or active_bear_fvgs:
                    entry_p = c_close
                    risk_buffer = max(c_atr * 1.4, entry_p * 0.008)
                    sl = entry_p + risk_buffer
                    init_sl = sl
                    risk_d = sl - entry_p
                    tp1 = entry_p - (1.5 * risk_d)
                    tp2 = entry_p - (2.5 * risk_d)

                    risk_amount = balance * 0.02
                    p_size = min(risk_amount / risk_d, balance / entry_p)
                    if p_size * entry_p > 10.0:
                        in_pos = True
                        p_side = "SHORT"
                        t_entry = t_now
                        low_p = entry_p
                        partial_taken = False
                        balance -= (p_size * entry_p * fee_rate)
                        last_trade_bar = i

    # Close open position at end of 2-week period
    if in_pos:
        last_c = ltf_df['close'].iloc[-1]
        if p_side == "LONG":
            pnl_final = p_size * (last_c - entry_p) - (p_size * last_c * fee_rate)
            balance += (p_size * last_c) - (p_size * last_c * fee_rate)
        else:
            pnl_final = p_size * (entry_p - last_c) - (p_size * last_c * fee_rate)
            balance += pnl_final
        trades.append({
            'symbol': symbol,
            'side': p_side,
            'entry_time': str(t_entry),
            'exit_time': str(ltf_df.index[-1]),
            'entry_price': entry_p,
            'exit_price': last_c,
            'pnl': (pnl_tp1 if partial_taken else 0.0) + pnl_final,
            'reason': 'EOD_CLOSE'
        })

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = (len(wins) / len(trades) * 100.0) if trades else 0.0
    pnl = balance - initial_balance

    gross_profit = sum([t['pnl'] for t in wins]) if wins else 0.0
    gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 1e-9
    pf = gross_profit / gross_loss

    return {
        'symbol': symbol,
        'days': eval_bars_count / 96.0,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': wr,
        'final_balance': balance,
        'pnl': pnl,
        'pnl_pct': pnl / initial_balance * 100.0,
        'profit_factor': pf,
        'trade_list': trades
    }

def main():
    symbols = Config.SUPPORTED_SYMBOLS
    print("=" * 95, flush=True)
    print("      PRIMESIGNAL 2-WEEK (14-DAY) INSTITUTIONAL SMC MULTI-COIN PORTFOLIO BACKTEST      ", flush=True)
    print("=" * 95, flush=True)
    print(f"Total Assets  : {len(symbols)} coins")
    print(f"Timeframe     : LTF: 15m | HTF: 1h")
    print(f"Period        : 14 Days (1,344 x 15m Candles per Coin)")
    print(f"Allocation    : $1,000 USDT per coin ($20,000 USDT Total Portfolio Starting Capital)\n", flush=True)

    exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 10000})

    results = []
    total_trades = 0
    total_wins = 0
    total_losses = 0
    initial_total = 1000.0 * len(symbols)
    final_total = 0.0
    all_trade_records = []

    for idx, sym in enumerate(symbols, 1):
        print(f"[{idx:02d}/{len(symbols)}] Fetching 2-week history & simulating {sym:<10}...", end="", flush=True)
        htf_data = fetch_pair_data_multi(exchange, sym, "1h", total_bars=600)
        ltf_data = fetch_pair_data_multi(exchange, sym, "15m", total_bars=1600)
        time.sleep(0.05)

        res = simulate_2weeks_coin(sym, htf_data, ltf_data, initial_balance=1000.0, test_bars=1344)
        if res:
            results.append(res)
            total_trades += res['trades']
            total_wins += res['wins']
            total_losses += res['losses']
            final_total += res['final_balance']
            all_trade_records.extend(res['trade_list'])
            print(f" Done ({res['trades']} trades | WR: {res['win_rate']:.1f}% | PnL: {res['pnl']:+7.2f} USDT | PF: {res['profit_factor']:.2f})", flush=True)
        else:
            final_total += 1000.0
            print(" [No Data / Skipped]", flush=True)

    portfolio_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    net_pnl = final_total - initial_total
    net_return_pct = (net_pnl / initial_total) * 100.0

    all_wins = [t for t in all_trade_records if t['pnl'] > 0]
    all_losses = [t for t in all_trade_records if t['pnl'] <= 0]
    portfolio_gp = sum([t['pnl'] for t in all_wins]) if all_wins else 0.0
    portfolio_gl = abs(sum([t['pnl'] for t in all_losses])) if all_losses else 1e-9
    portfolio_pf = portfolio_gp / portfolio_gl

    print("\n" + "=" * 95, flush=True)
    print("                    2-WEEK (14-DAY) COIN-BY-COIN PERFORMANCE REPORT                      ", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Asset':<12} | {'Days':<6} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'PnL (USDT)':<14} | {'Return %':<10}", flush=True)
    print("-" * 95, flush=True)
    for r in results:
        print(f"{r['symbol']:<12} | {r['days']:<6.1f} | {r['trades']:<8} | {r['wins']:<6} | {r['losses']:<6} | {r['win_rate']:>6.1f}%    | {r['pnl']:>+10.2f} USDT | {r['pnl_pct']:>+7.2f}%", flush=True)
    print("-" * 95, flush=True)
    print(f"{'PORTFOLIO':<12} | {'14.0':<6} | {total_trades:<8} | {total_wins:<6} | {total_losses:<6} | {portfolio_wr:>6.1f}%    | {net_pnl:>+10.2f} USDT | {net_return_pct:>+7.2f}%", flush=True)
    print("=" * 95, flush=True)

    print(f"\n[2-WEEK MULTI-COIN PORTFOLIO SUMMARY]")
    print(f"  • Backtest Duration      : 14 Days (2 Weeks)")
    print(f"  • Total Coins Tested     : {len(symbols)} crypto assets")
    print(f"  • Total Executed Trades  : {total_trades} trades (~{total_trades/14:.1f} trades/day)")
    print(f"  • Total Wins / Losses    : {total_wins}W / {total_losses}L")
    print(f"  • Overall Win Rate       : {portfolio_wr:.2f}%")
    print(f"  • Portfolio Profit Factor: {portfolio_pf:.2f}")
    print(f"  • Starting Balance       : {initial_total:.2f} USDT")
    print(f"  • Final Balance          : {final_total:.2f} USDT ({net_return_pct:+.2f}%)")
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()

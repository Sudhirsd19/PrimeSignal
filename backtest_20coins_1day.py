import json
import os
import sys
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Reconfigure stdout for utf-8 on Windows
if sys.platform == 'win32':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
        getattr(sys.stderr, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except (AttributeError, Exception):
        pass

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks, detect_structure

def fetch_pair_data(exchange, symbol, timeframe, limit=1000):
    cache_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache_dir, exist_ok=True)
    clean_sym = symbol.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"{clean_sym}_{timeframe}_{limit}.json")

    # If cache exists and is fresh (< 2 hours old), use it
    if os.path.exists(cache_file):
        try:
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 7200:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if data and len(data) > 100:
                        return data
        except Exception:
            pass

    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                with open(cache_file, 'w') as f:
                    json.dump(ohlcv, f)
                return ohlcv
        except Exception as e:
            time.sleep(0.5)
    return None

def simulate_1day_coin(symbol, htf_ohlcv, ltf_ohlcv, initial_balance=1000.0, test_bars=96):
    """
    Simulates trades specifically over the last 1 day (96 15m bars / 24 hours),
    using earlier historical bars for indicator and structural warm-up.
    """
    if not htf_ohlcv or not ltf_ohlcv or len(ltf_ohlcv) < test_bars + 100:
        return None

    htf_df = prepare_dataframe(htf_ohlcv)
    ltf_df = prepare_dataframe(ltf_ohlcv)

    # 1 day of 15m data = 96 bars
    start_eval_idx = len(ltf_df) - test_bars
    test_ltf = ltf_df.iloc[start_eval_idx:]

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
    pnl_tp1 = 0.0
    high_p = 0.0
    low_p = 999999.0
    t_entry = None
    trades = []
    fee_rate = 0.00075
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

                # Break-Even at +0.80R Profit (Winning Rule)
                if high_p >= entry_p + (0.80 * risk_d):
                    sl = max(sl, entry_p * 1.002)

                # TP1 Partial (50% size at 1.5R)
                if not partial_taken and c_high >= tp1:
                    partial_qty = p_size * 0.5
                    pnl_tp1 = partial_qty * (tp1 - entry_p) - (partial_qty * tp1 * fee_rate)
                    balance += partial_qty * tp1 - (partial_qty * tp1 * fee_rate)
                    p_size -= partial_qty
                    partial_taken = True
                    sl = max(sl, entry_p * 1.002) # lock in profit

                # TP2 (Full exit at 2.5R)
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

                # Break-Even at +0.80R Profit (Winning Rule)
                if low_p <= entry_p - (0.80 * risk_d):
                    sl = min(sl, entry_p * 0.998)

                # TP1 Partial (50% size at 1.5R)
                if not partial_taken and c_low <= tp1:
                    partial_qty = p_size * 0.5
                    pnl_tp1 = partial_qty * (entry_p - tp1) - (partial_qty * tp1 * fee_rate)
                    balance += pnl_tp1
                    p_size -= partial_qty
                    partial_taken = True
                    sl = min(sl, entry_p * 0.998)

                # TP2 (Full exit at 2.5R)
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
            # Entry Signal Check
            if i - last_trade_bar < 4:  # 1 hour cooldown between trades on same pair
                continue

            # Check HTF Trend Alignment
            htf_sub = htf_df[htf_df.index <= (t_now - pd.Timedelta(hours=1))]
            if len(htf_sub) < 50:
                continue
            htf_c = htf_sub['close'].iloc[-1]
            htf_e50 = htf_ema50.loc[htf_sub.index[-1]] if htf_sub.index[-1] in htf_ema50.index else None
            if htf_e50 is None:
                continue

            bullish_trend = htf_c > htf_e50
            bearish_trend = htf_c < htf_e50

            # 1. Bullish Entry (High Conviction: ADX >= 22 + OB/FVG confluence)
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

                    risk_amount = balance * 0.02 # 2% account risk
                    p_size = min(risk_amount / risk_d, balance / entry_p)
                    if p_size * entry_p > 10.0:
                        in_pos = True
                        p_side = "LONG"
                        t_entry = t_now
                        high_p = entry_p
                        partial_taken = False
                        balance -= (p_size * entry_p) + (p_size * entry_p * fee_rate)
                        last_trade_bar = i

            # 2. Bearish Entry (High Conviction: ADX >= 22 + OB/FVG confluence)
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

    # Close open position at end of 1-day period
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

    wins = [t for t in trades if float(t['pnl']) > 0]
    losses = [t for t in trades if float(t['pnl']) <= 0]
    wr = (len(wins) / len(trades) * 100.0) if trades else 0.0
    pnl = balance - initial_balance

    return {
        'symbol': symbol,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': wr,
        'final_balance': balance,
        'pnl': pnl,
        'pnl_pct': pnl / initial_balance * 100.0,
        'trade_list': trades
    }

def main():
    symbols = Config.SUPPORTED_SYMBOLS
    print("=" * 90, flush=True)
    print("        PRIMESIGNAL 1-DAY (24-HOUR) MULTI-COIN INSTITUTIONAL SMC BACKTEST        ", flush=True)
    print("=" * 90, flush=True)
    print(f"Total Assets  : {len(symbols)} coins")
    print(f"Coins Tested  : {', '.join(symbols)}")
    print(f"Window        : Last 24 Hours (96 x 15m Candles)")
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
        print(f"[{idx:02d}/{len(symbols)}] Fetching & backtesting {sym:<10}...", end="", flush=True)
        htf_data = fetch_pair_data(exchange, sym, "1h", limit=500)
        ltf_data = fetch_pair_data(exchange, sym, "15m", limit=1000)
        time.sleep(0.05)

        res = simulate_1day_coin(sym, htf_data, ltf_data, initial_balance=1000.0, test_bars=96)
        if res:
            results.append(res)
            total_trades += res['trades']
            total_wins += res['wins']
            total_losses += res['losses']
            final_total += res['final_balance']
            all_trade_records.extend(res['trade_list'])
            print(f" Done ({res['trades']} trades | WR: {res['win_rate']:.1f}% | PnL: {res['pnl']:+6.2f} USDT)", flush=True)
        else:
            final_total += 1000.0
            print(" [No Data / Skipped]", flush=True)

    portfolio_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    net_pnl = final_total - initial_total
    net_return_pct = (net_pnl / initial_total) * 100.0

    print("\n" + "=" * 90, flush=True)
    print("                    1-DAY (24-HOUR) COIN-BY-COIN PERFORMANCE REPORT                      ", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Asset':<12} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'PnL (USDT)':<14} | {'Return %':<10}", flush=True)
    print("-" * 90, flush=True)
    for r in results:
        print(f"{r['symbol']:<12} | {r['trades']:<8} | {r['wins']:<6} | {r['losses']:<6} | {r['win_rate']:>6.1f}%    | {r['pnl']:>+10.2f} USDT | {r['pnl_pct']:>+7.2f}%", flush=True)
    print("-" * 90, flush=True)
    print(f"{'PORTFOLIO':<12} | {total_trades:<8} | {total_wins:<6} | {total_losses:<6} | {portfolio_wr:>6.1f}%    | {net_pnl:>+10.2f} USDT | {net_return_pct:>+7.2f}%", flush=True)
    print("=" * 90, flush=True)

    # Detailed trade logs
    if all_trade_records:
        print("\n" + "=" * 90, flush=True)
        print("                           EXECUTED TRADE LOG (LAST 24 HOURS)                             ", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Symbol':<10} | {'Side':<6} | {'Entry Price':<12} | {'Exit Price':<12} | {'PnL (USDT)':<12} | {'Reason':<12}", flush=True)
        print("-" * 90, flush=True)
        for t in all_trade_records:
            print(f"{t['symbol']:<10} | {t['side']:<6} | {t['entry_price']:<12.4f} | {t['exit_price']:<12.4f} | {t['pnl']:>+9.2f} USDT | {t['reason']:<12}", flush=True)
        print("=" * 90, flush=True)

    print(f"\n[1-DAY MULTI-COIN PORTFOLIO SUMMARY]")
    print(f"  • Total Coins Tested     : {len(symbols)} crypto assets")
    print(f"  • Total Executed Trades  : {total_trades} trades")
    print(f"  • Total Wins / Losses    : {total_wins}W / {total_losses}L")
    print(f"  • Overall Win Rate       : {portfolio_wr:.2f}%")
    print(f"  • Starting Balance       : {initial_total:.2f} USDT")
    print(f"  • Final Balance          : {final_total:.2f} USDT ({net_return_pct:+.2f}%)")
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()

import json
import os
import sys
import time
import ccxt
import pandas as pd
import numpy as np
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks

def fetch_pair_data(exchange, symbol, timeframe, limit=1000):
    cache_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache_dir, exist_ok=True)
    clean_sym = symbol.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"{clean_sym}_{timeframe}_{limit}.json")

    if os.path.exists(cache_file):
        try:
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
        except Exception:
            time.sleep(0.5)
    return None

def sniper_structure_simulate(symbol, htf_ohlcv, ltf_ohlcv, initial_balance=1000.0):
    if not htf_ohlcv or not ltf_ohlcv or len(ltf_ohlcv) < 200:
        return None

    htf_df = prepare_dataframe(htf_ohlcv)
    ltf_df = prepare_dataframe(ltf_ohlcv)

    split_idx = int(len(ltf_df) * 0.10)
    test_ltf = ltf_df.iloc[split_idx:]
    
    closes = test_ltf['close'].values
    opens = test_ltf['open'].values
    highs = test_ltf['high'].values
    lows = test_ltf['low'].values
    timestamps = test_ltf.index

    ltf_atr = calculate_atr(test_ltf, 14).values
    ltf_adx = calculate_adx(test_ltf)['adx'].values
    ltf_rsi = calculate_rsi(test_ltf, 14).values
    ltf_vwap = calculate_vwap(test_ltf).values
    obs = detect_order_blocks(test_ltf)
    fvgs = detect_fvgs(test_ltf)

    htf_timestamps = htf_df.index.values
    htf_closes = htf_df['close'].values
    htf_ema50 = calculate_ema(htf_df, 50).values
    htf_ema200 = calculate_ema(htf_df, 200).values

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
    pnl_part = 0.0
    high_p = 0.0
    low_p = 999999.0
    trades = []
    fee_rate = 0.00075
    last_trade_bar = -999

    for i in range(50, len(test_ltf)):
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

                # Rule 1: Break-Even at +0.45R Profit
                if high_p >= entry_p + (0.45 * risk_d):
                    sl = max(sl, entry_p * 1.0015)

                # Rule 2: Scale-Out Partial TP1 at 0.8R (60% profit secured)
                if not partial_taken and c_high >= tp1:
                    partial_taken = True
                    pnl_part = (p_size * 0.6) * (tp1 - entry_p) - ((p_size * 0.6) * tp1 * fee_rate)
                    balance += pnl_part
                    sl = max(sl, entry_p * 1.0015)

                # Rule 3: Full TP2 at 1.5R
                if c_high >= tp2:
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_full = rem_size * (tp2 - entry_p) - (rem_size * tp2 * fee_rate)
                    balance += pnl_full
                    total_pnl = pnl_full + (pnl_part if partial_taken else 0)
                    trades.append({'symbol': symbol, 'side': 'LONG', 'pnl': total_pnl, 'is_win': True, 'reason': 'TAKE_PROFIT', 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

                elif c_low <= sl:
                    exit_p = min(sl, c_open)
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_loss = rem_size * (exit_p - entry_p) - (rem_size * exit_p * fee_rate)
                    balance += pnl_loss
                    total_pnl = pnl_loss + (pnl_part if partial_taken else 0)
                    is_win = total_pnl >= 0
                    trades.append({'symbol': symbol, 'side': 'LONG', 'pnl': total_pnl, 'is_win': is_win, 'reason': 'BREAKEVEN' if is_win else 'STOP_LOSS', 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

            elif p_side == "SHORT":
                low_p = min(low_p, c_low)
                risk_d = abs(entry_p - init_sl)

                # Rule 1: Break-Even at +0.45R Profit
                if low_p <= entry_p - (0.45 * risk_d):
                    sl = min(sl, entry_p * 0.9985)

                # Rule 2: Scale-Out Partial TP1 at 0.8R
                if not partial_taken and c_low <= tp1:
                    partial_taken = True
                    pnl_part = (p_size * 0.6) * (entry_p - tp1) - ((p_size * 0.6) * tp1 * fee_rate)
                    balance += pnl_part
                    sl = min(sl, entry_p * 0.9985)

                # Rule 3: Full TP2 at 1.5R
                if c_low <= tp2:
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_full = rem_size * (entry_p - tp2) - (rem_size * tp2 * fee_rate)
                    balance += pnl_full
                    total_pnl = pnl_full + (pnl_part if partial_taken else 0)
                    trades.append({'symbol': symbol, 'side': 'SHORT', 'pnl': total_pnl, 'is_win': True, 'reason': 'TAKE_PROFIT', 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

                elif c_high >= sl:
                    exit_p = max(sl, c_open)
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_loss = rem_size * (entry_p - exit_p) - (rem_size * exit_p * fee_rate)
                    balance += pnl_loss
                    total_pnl = pnl_loss + (pnl_part if partial_taken else 0)
                    is_win = total_pnl >= 0
                    trades.append({'symbol': symbol, 'side': 'SHORT', 'pnl': total_pnl, 'is_win': is_win, 'reason': 'BREAKEVEN' if is_win else 'STOP_LOSS', 'time': t_now})
                    in_pos = False
                    last_trade_bar = i

        else:
            if i - last_trade_bar < 4:
                continue

            htf_idx = np.searchsorted(htf_timestamps, t_now) - 1
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

            setup_found = False
            zone_sl = 0.0

            # BULLISH (OB + FVG + Rejection Wick)
            if bullish_trend and c_close > c_vwap:
                # OB Entry
                for ob_idx in range(i-1, max(0, i-35), -1):
                    ob = obs.iloc[ob_idx]
                    if ob and ob['type'] == 'BULLISH' and not ob['mitigated']:
                        if ob['bottom'] * 0.998 <= c_low <= ob['top'] * 1.002:
                            candle_r = c_high - c_low
                            lower_w = min(c_open, c_close) - c_low
                            if candle_r > 0 and (lower_w / candle_r >= 0.20 or c_close > c_open):
                                setup_found = True
                                zone_sl = ob['bottom'] * 0.9985
                                break

                # FVG Entry with Micro-BOS
                if not setup_found:
                    for fvg_idx in range(i-1, max(0, i-25), -1):
                        fvg = fvgs.iloc[fvg_idx]
                        if fvg and fvg['type'] == 'BULLISH' and not fvg['mitigated']:
                            if fvg['bottom'] <= c_low <= fvg['top'] * 1.002:
                                if c_close > highs[i-1]: # Micro-BOS breakout
                                    setup_found = True
                                    zone_sl = fvg['bottom'] * 0.9985
                                    break

                if setup_found and c_rsi < 68:
                    entry_p = c_close
                    sl = max(zone_sl, entry_p - (1.5 * c_atr))
                    sl = min(sl, entry_p * 0.997) # min 0.3%
                    init_sl = sl
                    dist = abs(entry_p - sl)
                    if 0 < dist / entry_p < 0.025:
                        tp1 = entry_p + (dist * 0.8)
                        tp2 = entry_p + (dist * 1.5)
                        p_size = (balance * 0.02) / dist
                        in_pos = True
                        p_side = "LONG"
                        partial_taken = False
                        pnl_part = 0.0
                        high_p = entry_p
                        low_p = entry_p

            # BEARISH (OB + FVG + Rejection Wick)
            elif bearish_trend and c_close < c_vwap:
                # OB Entry
                for ob_idx in range(i-1, max(0, i-35), -1):
                    ob = obs.iloc[ob_idx]
                    if ob and ob['type'] == 'BEARISH' and not ob['mitigated']:
                        if ob['bottom'] * 0.998 <= c_high <= ob['top'] * 1.002:
                            candle_r = c_high - c_low
                            upper_w = c_high - max(c_open, c_close)
                            if candle_r > 0 and (upper_w / candle_r >= 0.20 or c_close < c_open):
                                setup_found = True
                                zone_sl = ob['top'] * 1.0015
                                break

                # FVG Entry with Micro-BOS
                if not setup_found:
                    for fvg_idx in range(i-1, max(0, i-25), -1):
                        fvg = fvgs.iloc[fvg_idx]
                        if fvg and fvg['type'] == 'BEARISH' and not fvg['mitigated']:
                            if fvg['bottom'] * 0.998 <= c_high <= fvg['top']:
                                if c_close < lows[i-1]: # Micro-BOS breakdown
                                    setup_found = True
                                    zone_sl = fvg['top'] * 1.0015
                                    break

                if setup_found and c_rsi > 32:
                    entry_p = c_close
                    sl = min(zone_sl, entry_p + (1.5 * c_atr))
                    sl = max(sl, entry_p * 1.003) # min 0.3%
                    init_sl = sl
                    dist = abs(entry_p - sl)
                    if 0 < dist / entry_p < 0.025:
                        tp1 = entry_p - (dist * 0.8)
                        tp2 = entry_p - (dist * 1.5)
                        p_size = (balance * 0.02) / dist
                        in_pos = True
                        p_side = "SHORT"
                        partial_taken = False
                        pnl_part = 0.0
                        high_p = entry_p
                        low_p = entry_p

    wins = [t for t in trades if t['is_win']]
    losses = [t for t in trades if not t['is_win']]
    wr = len(wins) / len(trades) * 100 if trades else 0.0
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
    symbols = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"
    ]

    print("=" * 85, flush=True)
    print("      PRIMESIGNAL HIGH-CONVICTION SMC PORTFOLIO ENGINE (OB + FVG MICRO-BOS)     ", flush=True)
    print("=" * 85, flush=True)
    print(f"Universe   : {', '.join(symbols)}", flush=True)
    print(f"Timeframes : LTF: 15m | HTF: 1h | Allocation: $1,000 per pair ($10,000 Total)\n", flush=True)

    exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 10000})

    _base_dir = os.path.dirname(os.path.abspath(__file__))
    htf_file = os.path.join(_base_dir, "htf_data.json")
    ltf_file = os.path.join(_base_dir, "ltf_data.json")
    btc_htf = json.load(open(htf_file)) if os.path.exists(htf_file) else None
    btc_ltf = json.load(open(ltf_file)) if os.path.exists(ltf_file) else None

    results = []
    total_trades = 0
    total_wins = 0
    total_losses = 0
    initial_total = 1000.0 * len(symbols)
    final_total = 0.0

    for idx, sym in enumerate(symbols, 1):
        print(f"[{idx}/10] Evaluating {sym}...", flush=True)
        if sym == "BTC/USDT" and btc_htf and btc_ltf:
            htf_data = btc_htf
            ltf_data = btc_ltf
        else:
            htf_data = fetch_pair_data(exchange, sym, "1h", limit=500)
            ltf_data = fetch_pair_data(exchange, sym, "15m", limit=1000)
            time.sleep(0.1)

        res = sniper_structure_simulate(sym, htf_data, ltf_data, initial_balance=1000.0)
        if res:
            results.append(res)
            total_trades += res['trades']
            total_wins += res['wins']
            total_losses += res['losses']
            final_total += res['final_balance']
        else:
            final_total += 1000.0

    portfolio_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    net_pnl = final_total - initial_total
    net_return_pct = (net_pnl / initial_total) * 100.0

    print("\n" + "=" * 85, flush=True)
    print("                 HIGH-CONVICTION SMC PORTFOLIO PERFORMANCE                           ", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Asset':<12} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'PnL (USDT)':<12} | {'Return %':<10}", flush=True)
    print("-" * 85, flush=True)
    for r in results:
        print(f"{r['symbol']:<12} | {r['trades']:<8} | {r['wins']:<6} | {r['losses']:<6} | {r['win_rate']:>6.1f}%    | {r['pnl']:>+9.2f} USDT | {r['pnl_pct']:>+7.2f}%", flush=True)
    print("-" * 85, flush=True)
    print(f"{'PORTFOLIO':<12} | {total_trades:<8} | {total_wins:<6} | {total_losses:<6} | {portfolio_wr:>6.1f}%    | {net_pnl:>+9.2f} USDT | {net_return_pct:>+7.2f}%", flush=True)
    print("=" * 85, flush=True)

    trades_per_day = total_trades / 10.4
    print(f"\n[SMC PORTFOLIO SUMMARY]")
    print(f"  • Total Portfolio Trades : {total_trades} trades executed")
    print(f"  • Overall Win Rate       : {portfolio_wr:.2f}% ({total_wins} Wins out of {total_trades} Trades)")
    print(f"  • Starting Capital       : {initial_total:.2f} USDT")
    print(f"  • Final Account Value    : {final_total:.2f} USDT ({net_return_pct:+.2f}%)")
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()

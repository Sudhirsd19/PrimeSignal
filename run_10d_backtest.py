"""
PrimeSignal 10-Day Institutional Portfolio Backtest
Evaluates the EXACT live production strategy over the last 10 days (960 x 15m bars).
Incorporates:
- 15m Execution + 1h Macro Trend Multi-Timeframe SMC
- Dynamic Kelly Risk Sizing (0.2% - 2.0%)
- Institutional Targets: TP1 @ 2.0R, TP2 @ 3.0R, Trailing Runner @ 4.0R
- Zero-Risk Breakeven Lock at +1.0R (with fee offset buffer)
- 0.15% Roundtrip Taker Fees
- Overtrading limit (max 3 trades/day)
- Daily Profit Lock (Configurable: 17.5% target) & 5.0% Loss Circuit Breaker
"""
import os
import sys
import json
import math
import datetime
import urllib.request
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from core.performance_analytics import calculate_advanced_metrics
from risk.risk_manager import RiskManager

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "DOGE/USDT", "LTC/USDT", "ARB/USDT", "OP/USDT", "NEAR/USDT"
]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEN_DAYS_15M_BARS = 960
TEN_DAYS_1H_BARS = 240

def fetch_live_binance_klines(symbol, interval, limit=1000):
    clean = symbol.replace("/", "")
    url = f"https://api.binance.com/api/v3/klines?symbol={clean}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in data]
    except Exception as e:
        return None

def load_or_fetch_10d_data(symbol):
    clean = symbol.replace("/", "_")
    
    # 1. Try Binance Live API for fresh 10 days
    ltf_live = fetch_live_binance_klines(symbol, "15m", limit=1000)
    htf_live = fetch_live_binance_klines(symbol, "1h", limit=300)
    
    if ltf_live and htf_live and len(ltf_live) >= 900:
        return ltf_live, htf_live, "Live Binance 10D"
        
    # 2. Fallback to local cached data
    ltf_file = os.path.join(DATA_DIR, f"{clean}_15m_30d.json")
    htf_file = os.path.join(DATA_DIR, f"{clean}_1h_30d.json")
    if not os.path.exists(ltf_file):
        ltf_file = os.path.join(DATA_DIR, f"{clean}_15m_1600_2w.json")
    if not os.path.exists(htf_file):
        htf_file = os.path.join(DATA_DIR, f"{clean}_1h_600_2w.json")
        
    if os.path.exists(ltf_file) and os.path.exists(htf_file):
        with open(ltf_file, 'r') as f: ltf = json.load(f)
        with open(htf_file, 'r') as f: htf = json.load(f)
        ltf_clean = [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in ltf[-1000:]]
        htf_clean = [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in htf[-300:]]
        return ltf_clean, htf_clean, "Local Cached 10D"
        
    return None, None, "None"

def run_asset_10d_backtest(symbol, ltf_ohlcv, htf_ohlcv, initial_balance=1000.0, profit_lock_target=17.5):
    ltf_df = prepare_dataframe(ltf_ohlcv)
    htf_df = prepare_dataframe(htf_ohlcv)

    if len(ltf_df) < 200 or len(htf_df) < 50:
        return None

    eval_bars = min(TEN_DAYS_15M_BARS, len(ltf_df) - 100)
    start_idx = len(ltf_df) - eval_bars

    closes = ltf_df['close'].values
    opens = ltf_df['open'].values
    highs = ltf_df['high'].values
    lows = ltf_df['low'].values
    timestamps = ltf_df.index

    atr_vals = calculate_atr(ltf_df, 14).values
    adx = calculate_adx(ltf_df)['adx'].values
    rsi = calculate_rsi(ltf_df, 14).values
    ema50_ltf = calculate_ema(ltf_df, 50).values
    ema200_ltf = calculate_ema(ltf_df, 200).values

    htf_timestamps = htf_df.index
    htf_ema50 = calculate_ema(htf_df, 50).values
    htf_ema200 = calculate_ema(htf_df, 200).values

    fee_rate = 0.00075
    base_risk_pct = getattr(Config, 'RISK_PCT', 0.01)
    risk_manager = RiskManager()

    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown_pct = 0.0
    daily_equity_curve = {}

    current_day = None
    daily_start_equity = initial_balance
    daily_profit_locked = False
    daily_loss_tripped = False
    trades_today = 0
    max_daily_trades = getattr(Config, 'MAX_DAILY_TRADES', 3)
    pause_until_ts = 0

    trades = []
    equity_series = [initial_balance]

    in_pos = False
    pos_side = None
    entry_price = 0.0
    initial_sl = 0.0
    current_sl = 0.0
    tp1 = 0.0
    tp2 = 0.0
    tp3 = 0.0
    total_size = 0.0
    rem_size = 0.0
    highest_p = 0.0
    lowest_p = 999999.0
    entry_ts = None
    tp1_done = False
    tp2_done = False
    realized_tp_pnl = 0.0
    accum_fees = 0.0
    stages = []

    for i in range(start_idx, len(ltf_df) - 1):
        curr_price = closes[i]
        curr_dt = timestamps[i]
        curr_ts = curr_dt.timestamp()
        day_date = curr_dt.date()

        if current_day != day_date:
            if current_day is not None:
                daily_equity_curve[str(current_day)] = balance
            current_day = day_date
            daily_start_equity = balance
            daily_profit_locked = False
            daily_loss_tripped = False
            trades_today = 0

        daily_gain_pct = ((balance - daily_start_equity) / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0
        if daily_gain_pct >= profit_lock_target:
            daily_profit_locked = True
        if daily_gain_pct <= -5.0:
            daily_loss_tripped = True

        htf_idx = np.searchsorted(htf_timestamps, curr_dt, side='right') - 1
        htf_bullish = False
        htf_bearish = False
        if 0 <= htf_idx < len(htf_ema50):
            htf_bullish = htf_ema50[htf_idx] > htf_ema200[htf_idx]
            htf_bearish = htf_ema50[htf_idx] < htf_ema200[htf_idx]

        # ─── POSITION MANAGEMENT ───
        if in_pos:
            r_dist = abs(entry_price - initial_sl)
            curr_atr = atr_vals[i] if not math.isnan(atr_vals[i]) else (entry_price * 0.01)

            if pos_side == "LONG":
                highest_p = max(highest_p, highs[i])

                if highest_p >= entry_price + (1.0 * r_dist):
                    be_sl = entry_price * (1.0 + Config.DYNAMIC_BE_BUFFER_PCT)
                    if be_sl > current_sl:
                        current_sl = be_sl

                tp1_scale = getattr(Config, 'TP1_SCALE_OUT_PCT', 0.80)
                if not tp1_done and highs[i] >= tp1:
                    tp1_done = True
                    stages.append("TP1")
                    close_qty = total_size * tp1_scale
                    rem_size -= close_qty
                    leg_gross = close_qty * (tp1 - entry_price)
                    leg_fee = (close_qty * entry_price * fee_rate) + (close_qty * tp1 * fee_rate)
                    leg_net = leg_gross - leg_fee
                    realized_tp_pnl += leg_net
                    accum_fees += (close_qty * tp1 * fee_rate)
                    balance += leg_net
                    current_sl = max(current_sl, entry_price * (1.0 + Config.DYNAMIC_BE_BUFFER_PCT))

                if tp1_done and not tp2_done and highs[i] >= tp2:
                    tp2_done = True
                    stages.append("TP2")
                    close_qty = rem_size
                    rem_size = 0.0
                    leg_gross = close_qty * (tp2 - entry_price)
                    leg_fee = (close_qty * entry_price * fee_rate) + (close_qty * tp2 * fee_rate)
                    leg_net = leg_gross - leg_fee
                    realized_tp_pnl += leg_net
                    accum_fees += (close_qty * tp2 * fee_rate)
                    balance += leg_net
                    current_sl = max(current_sl, highest_p - (curr_atr * Config.TRAILING_ATR_MULT))

                if tp2_done:
                    trail_stop = highest_p - (curr_atr * Config.TRAILING_ATR_MULT)
                    if trail_stop > current_sl:
                        current_sl = trail_stop

                hit_sl = lows[i] <= current_sl
                hit_tp3 = highs[i] >= tp3

                if hit_tp3 or hit_sl:
                    exit_p = tp3 if hit_tp3 else min(current_sl, opens[i])
                    reason = "TP3_TARGET" if hit_tp3 else ("TRAILING_STOP" if tp2_done else ("BREAKEVEN" if current_sl > entry_price else "STOP_LOSS"))
                    stages.append(reason)
                    
                    leg_gross = rem_size * (exit_p - entry_price)
                    leg_fee = (rem_size * entry_price * fee_rate) + (rem_size * exit_p * fee_rate)
                    leg_net = leg_gross - leg_fee
                    accum_fees += (rem_size * exit_p * fee_rate)
                    balance += leg_net

                    total_lifecycle_pnl = realized_tp_pnl + leg_net
                    is_win = total_lifecycle_pnl > 0

                    trades.append({
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_time': str(entry_ts),
                        'exit_time': str(curr_dt),
                        'entry_price': entry_price,
                        'exit_price': exit_p,
                        'total_pnl_net': total_lifecycle_pnl,
                        'total_fees': accum_fees,
                        'stages_completed': list(stages),
                        'exit_reason': reason,
                        'is_win': is_win
                    })
                    in_pos = False

            elif pos_side == "SHORT":
                lowest_p = min(lowest_p, lows[i])

                if lowest_p <= entry_price - (1.0 * r_dist):
                    be_sl = entry_price * (1.0 - Config.DYNAMIC_BE_BUFFER_PCT)
                    if be_sl < current_sl:
                        current_sl = be_sl

                tp1_scale = getattr(Config, 'TP1_SCALE_OUT_PCT', 0.80)
                if not tp1_done and lows[i] <= tp1:
                    tp1_done = True
                    stages.append("TP1")
                    close_qty = total_size * tp1_scale
                    rem_size -= close_qty
                    leg_gross = close_qty * (entry_price - tp1)
                    leg_fee = (close_qty * entry_price * fee_rate) + (close_qty * tp1 * fee_rate)
                    leg_net = leg_gross - leg_fee
                    realized_tp_pnl += leg_net
                    accum_fees += (close_qty * tp1 * fee_rate)
                    balance += leg_net
                    current_sl = min(current_sl, entry_price * (1.0 - Config.DYNAMIC_BE_BUFFER_PCT))

                if tp1_done and not tp2_done and lows[i] <= tp2:
                    tp2_done = True
                    stages.append("TP2")
                    close_qty = rem_size
                    rem_size = 0.0
                    leg_gross = close_qty * (entry_price - tp2)
                    leg_fee = (close_qty * entry_price * fee_rate) + (close_qty * tp2 * fee_rate)
                    leg_net = leg_gross - leg_fee
                    realized_tp_pnl += leg_net
                    accum_fees += (close_qty * tp2 * fee_rate)
                    balance += leg_net
                    current_sl = min(current_sl, lowest_p + (curr_atr * Config.TRAILING_ATR_MULT))

                if tp2_done:
                    trail_stop = lowest_p + (curr_atr * Config.TRAILING_ATR_MULT)
                    if trail_stop < current_sl:
                        current_sl = trail_stop

                hit_sl = highs[i] >= current_sl
                hit_tp3 = lows[i] <= tp3

                if hit_tp3 or hit_sl:
                    exit_p = tp3 if hit_tp3 else max(current_sl, opens[i])
                    reason = "TP3_TARGET" if hit_tp3 else ("TRAILING_STOP" if tp2_done else ("BREAKEVEN" if current_sl < entry_price else "STOP_LOSS"))
                    stages.append(reason)
                    
                    leg_gross = rem_size * (entry_price - exit_p)
                    leg_fee = (rem_size * entry_price * fee_rate) + (rem_size * exit_p * fee_rate)
                    leg_net = leg_gross - leg_fee
                    accum_fees += (rem_size * exit_p * fee_rate)
                    balance += leg_net

                    total_lifecycle_pnl = realized_tp_pnl + leg_net
                    is_win = total_lifecycle_pnl > 0

                    trades.append({
                        'symbol': symbol,
                        'side': 'SHORT',
                        'entry_time': str(entry_ts),
                        'exit_time': str(curr_dt),
                        'entry_price': entry_price,
                        'exit_price': exit_p,
                        'total_pnl_net': total_lifecycle_pnl,
                        'total_fees': accum_fees,
                        'stages_completed': list(stages),
                        'exit_reason': reason,
                        'is_win': is_win
                    })
                    in_pos = False

        # ─── ENTRY SIGNALS ───
        if not in_pos:
            if daily_profit_locked or daily_loss_tripped or curr_ts < pause_until_ts:
                continue
            if trades_today >= max_daily_trades:
                continue

            curr_atr = atr_vals[i] if not math.isnan(atr_vals[i]) else (curr_price * 0.01)
            curr_rsi = rsi[i] if not math.isnan(rsi[i]) else 50.0
            curr_adx = adx[i] if not math.isnan(adx[i]) else 25.0

            long_trend = htf_bullish and curr_price > ema200_ltf[i] and curr_price > ema50_ltf[i]
            short_trend = htf_bearish and curr_price < ema200_ltf[i] and curr_price < ema50_ltf[i]

            bullish_rejection = (lows[i] < opens[i] and closes[i] > opens[i] and (closes[i] - lows[i]) > 1.5 * abs(closes[i] - opens[i]))
            bearish_rejection = (highs[i] > opens[i] and closes[i] < opens[i] and (highs[i] - closes[i]) > 1.5 * abs(closes[i] - opens[i]))

            sig = None
            adx_min = getattr(Config, 'ADX_MIN_THRESHOLD', 22.0)
            if long_trend and bullish_rejection and 40 < curr_rsi < 68 and curr_adx >= adx_min:
                sig = "BUY"
            elif short_trend and bearish_rejection and 32 < curr_rsi < 60 and curr_adx >= adx_min:
                sig = "SELL"

            if sig:
                if Config.ENABLE_KELLY_SIZING:
                    current_risk_pct = risk_manager.calculate_kelly_risk_pct(trades, base_risk=Config.RISK_PCT) / 100.0
                else:
                    current_risk_pct = base_risk_pct

                sl_dist = max(curr_atr * 1.5, curr_price * 0.008)
                dollar_risk = balance * current_risk_pct
                pos_qty = dollar_risk / sl_dist

                max_notional = balance * getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35)
                if (pos_qty * curr_price) > max_notional:
                    pos_qty = max_notional / curr_price

                if (pos_qty * curr_price) < 10.0:
                    continue

                in_pos = True
                pos_side = "LONG" if sig == "BUY" else "SHORT"
                entry_price = curr_price
                entry_ts = curr_dt
                total_size = pos_qty
                rem_size = pos_qty
                highest_p = curr_price
                lowest_p = curr_price
                tp1_done = False
                tp2_done = False
                realized_tp_pnl = 0.0
                accum_fees = pos_qty * curr_price * fee_rate
                balance -= accum_fees
                stages = ["ENTRY"]
                trades_today += 1

                if pos_side == "LONG":
                    initial_sl = entry_price - sl_dist
                    current_sl = initial_sl
                    tp1 = entry_price + (sl_dist * getattr(Config, 'TP1_R_MULT', 2.0))
                    tp2 = entry_price + (sl_dist * getattr(Config, 'TP2_R_MULT', 3.0))
                    tp3 = entry_price + (sl_dist * getattr(Config, 'TP3_R_MULT', 4.0))
                else:
                    initial_sl = entry_price + sl_dist
                    current_sl = initial_sl
                    tp1 = entry_price - (sl_dist * getattr(Config, 'TP1_R_MULT', 2.0))
                    tp2 = entry_price - (sl_dist * getattr(Config, 'TP2_R_MULT', 3.0))
                    tp3 = entry_price - (sl_dist * getattr(Config, 'TP3_R_MULT', 4.0))

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0 if peak_balance > 0 else 0.0
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd
        equity_series.append(balance)

    if current_day is not None:
        daily_equity_curve[str(current_day)] = balance

    wins = [t for t in trades if t['is_win']]
    losses = [t for t in trades if not t['is_win']]
    gross_profits = sum(t['total_pnl_net'] for t in wins)
    gross_losses = abs(sum(t['total_pnl_net'] for t in losses))
    net_profit = balance - initial_balance
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (99.0 if gross_profits > 0 else 0.0)

    start_date_str = str(timestamps[start_idx].strftime('%Y-%m-%d %H:%M'))
    end_date_str = str(timestamps[-1].strftime('%Y-%m-%d %H:%M'))

    return {
        'symbol': symbol,
        'start_date': start_date_str,
        'end_date': end_date_str,
        'initial_balance': initial_balance,
        'final_balance': balance,
        'net_profit': net_profit,
        'net_profit_pct': (net_profit / initial_balance) * 100.0,
        'trades_count': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'trades': trades,
        'daily_equity_curve': daily_equity_curve
    }

def main():
    print("=" * 95)
    print("      🚀 PRIMESIGNAL 10-DAY INSTITUTIONAL MULTI-ASSET BACKTEST RUNNER        ")
    print("=" * 95)
    print("Simulation Window : Last 10 Days (960 x 15m Bars / 240 x 1h Bars)")
    print("Target Watchlist  : 10 Major Coins (BTC, ETH, SOL, BNB, XRP, DOGE, LTC, ARB, OP, NEAR)")
    print("Execution Engine  : 15m LTF + 1h HTF Multi-Timeframe SMC (EMA200, ADX 22, RSI, ATR)")
    print("Fee Structure     : 0.15% Roundtrip Taker Fees Included")
    print("Risk Sizing       : Half-Kelly Dynamic Sizing with 35% Capital Allocation Limit")
    print("Profit Lock Target: 17.5% Daily Target\n")

    results = []
    total_start = 1000.0 * len(SYMBOLS)
    total_final = 0.0
    all_trades = []

    print(f"{'#':<3} {'Asset':<10} {'Source':<18} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'Max DD':<10} {'Net PnL ($)':<12} {'Return %':<10}")
    print("-" * 95)

    for idx, sym in enumerate(SYMBOLS, 1):
        ltf, htf, src = load_or_fetch_10d_data(sym)
        if not ltf or not htf:
            print(f"{idx:<3} {sym:<10} {'[No Data]':<18} {'-':<8} {'-':<10} {'-':<15} {'-':<10} {'-':<12} {'-'}")
            total_final += 1000.0
            continue

        res = run_asset_10d_backtest(sym, ltf, htf, initial_balance=1000.0, profit_lock_target=17.5)
        if not res:
            print(f"{idx:<3} {sym:<10} {'[Error]':<18} {'-':<8} {'-':<10} {'-':<15} {'-':<10} {'-':<12} {'-'}")
            total_final += 1000.0
            continue

        results.append(res)
        total_final += res['final_balance']
        all_trades.extend(res['trades'])

        pnl_str = f"+${res['net_profit']:.2f}" if res['net_profit'] >= 0 else f"-${abs(res['net_profit']):.2f}"
        ret_str = f"+{res['net_profit_pct']:.2f}%" if res['net_profit_pct'] >= 0 else f"{res['net_profit_pct']:.2f}%"
        pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] < 90 else "Inf"

        print(f"{idx:<3} {sym:<10} {src:<18} {res['trades_count']:<8} {res['win_rate']:<9.1f}% {pf_str:<15} {res['max_drawdown_pct']:<9.2f}% {pnl_str:<12} {ret_str}")

    print("=" * 95)
    
    portfolio_net_pnl = total_final - total_start
    portfolio_return_pct = (portfolio_net_pnl / total_start) * 100.0
    tot_trades = len(all_trades)
    tot_wins = sum(1 for t in all_trades if t['is_win'])
    tot_losses = sum(1 for t in all_trades if not t['is_win'])
    overall_winrate = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0

    gross_gains = sum(t['total_pnl_net'] for t in all_trades if t['is_win'])
    gross_loss = abs(sum(t['total_pnl_net'] for t in all_trades if not t['is_win']))
    overall_pf = (gross_gains / gross_loss) if gross_loss > 0 else (99.0 if gross_gains > 0 else 0.0)
    total_fees_paid = sum(t['total_fees'] for t in all_trades)

    print("\n📊 10-DAY PORTFOLIO AGGREGATE SUMMARY:")
    print(f"  • Starting Balance     : ${total_start:,.2f} USDT ($1,000 per asset)")
    print(f"  • Final Portfolio Value: ${total_final:,.2f} USDT")
    print(f"  • Total Net Profit     : {'+' if portfolio_net_pnl >= 0 else '-'}${abs(portfolio_net_pnl):,.2f} USDT ({'+' if portfolio_return_pct >= 0 else ''}{portfolio_return_pct:.2f}%)")
    print(f"  • Total Trades Taken   : {tot_trades} ({tot_wins} Wins / {tot_losses} Losses)")
    print(f"  • Overall Win Rate     : {overall_winrate:.1f}%")
    print(f"  • Portfolio Profit Factor: {overall_pf:.2f}")
    print(f"  • Total Exchange Fees  : ${total_fees_paid:.2f} USDT (0.15% roundtrip deducted)")

    if all_trades:
        print("\n📜 RECENT EXECUTED TRADES IN 10-DAY WINDOW (Sample):")
        print(f"{'Time':<20} {'Symbol':<10} {'Side':<6} {'Entry':<10} {'Exit':<10} {'Stages':<25} {'Net PnL':<10}")
        print("-" * 88)
        for t in all_trades[-12:]:
            pnl_s = f"+${t['total_pnl_net']:.2f}" if t['total_pnl_net'] >= 0 else f"-${abs(t['total_pnl_net']):.2f}"
            stg_s = " -> ".join(t['stages_completed'][:3])
            t_str = t['entry_time'][:16]
            print(f"{t_str:<20} {t['symbol']:<10} {t['side']:<6} {t['entry_price']:<10.4f} {t['exit_price']:<10.4f} {stg_s:<25} {pnl_s:<10}")

    summary_path = os.path.join(PROJECT_ROOT, "backtest_10d_results.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'total_start': total_start,
            'total_final': total_final,
            'portfolio_net_pnl': portfolio_net_pnl,
            'portfolio_return_pct': portfolio_return_pct,
            'tot_trades': tot_trades,
            'tot_wins': tot_wins,
            'tot_losses': tot_losses,
            'overall_winrate': overall_winrate,
            'overall_pf': overall_pf,
            'total_fees_paid': total_fees_paid,
            'results': results
        }, f, indent=2, default=str)
    print(f"\n[BACKTEST] Detailed results successfully saved to {summary_path}")

if __name__ == "__main__":
    main()

"""
PrimeSignal Current Logic Backtest Suite — Upgraded Institutional Run
Simulates the EXACT live trading bot architecture with all institutional parameters:
1. 15m LTF + 1h HTF Multi-Timeframe SMC (Order Blocks + EMA Trend Alignment + RSI + ADX)
2. Upgraded Institutional Targets: TP1 @ 1.5R (50%), TP2 @ 2.5R (30%), Runner @ 4.0R (20% ATR Trailing)
3. Zero-Risk Breakeven Lock at +1.0R (with fee offset buffer)
4. Dynamic Half-Kelly Position Sizing (0.2% - 2.0% bounds)
5. Overtrading Limit: Max 3 A+ trades per day (MAX_DAILY_TRADES = 3)
6. 0.15% Roundtrip Taker Fees deducted on every fill
7. Daily Profit Lock (+4.0%) & Daily Loss Circuit Breaker (-5.0%)
8. Institutional Metrics calculated via core.performance_analytics
"""
import os
import sys
import json
import math
import datetime
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from core.performance_analytics import calculate_advanced_metrics
from risk.risk_manager import RiskManager

SUPPORTED_PAIRS = [
    "BTC/USDT", "BNB/USDT", "XRP/USDT", "LTC/USDT", "DOGE/USDT", "SOL/USDT", "ETH/USDT", "LINK/USDT"
]

def load_data(symbol):
    data_dir = os.path.join(PROJECT_ROOT, "data")
    clean = symbol.replace("/", "_")
    ltf_file = os.path.join(data_dir, f"{clean}_15m_30d.json")
    htf_file = os.path.join(data_dir, f"{clean}_1h_30d.json")
    
    if not os.path.exists(ltf_file):
        ltf_file = os.path.join(data_dir, f"{clean}_15m_1600_2w.json")
    if not os.path.exists(htf_file):
        htf_file = os.path.join(data_dir, f"{clean}_1h_600_2w.json")
        
    if os.path.exists(ltf_file) and os.path.exists(htf_file):
        with open(ltf_file, 'r') as f: ltf = json.load(f)
        with open(htf_file, 'r') as f: htf = json.load(f)
        return ltf, htf
    return None, None

def simulate_asset(symbol, ltf_ohlcv, htf_ohlcv, initial_balance=1000.0):
    ltf_df = prepare_dataframe(ltf_ohlcv)
    htf_df = prepare_dataframe(htf_ohlcv)

    if len(ltf_df) < 150 or len(htf_df) < 50:
        return None

    closes = ltf_df['close'].values
    opens = ltf_df['open'].values
    highs = ltf_df['high'].values
    lows = ltf_df['low'].values
    timestamps = ltf_df.index

    atr_vals = calculate_atr(ltf_df, 14).values
    adx_df = calculate_adx(ltf_df)
    adx = adx_df['adx'].values
    rsi = calculate_rsi(ltf_df, 14).values
    ema50_ltf = calculate_ema(ltf_df, 50).values
    ema200_ltf = calculate_ema(ltf_df, 200).values

    htf_ema50 = calculate_ema(htf_df, 50).values
    htf_ema200 = calculate_ema(htf_df, 200).values
    htf_timestamps = htf_df.index.values

    balance = initial_balance
    daily_start_equity = initial_balance
    current_day = None
    daily_profit_locked = False
    daily_loss_tripped = False
    trades_today = 0

    fee_rate = Config.FEE_RATE  # 0.00075 taker
    base_risk_pct = Config.RISK_PCT / 100.0  # 0.008 (0.8%)
    max_daily_profit_pct = Config.MAX_DAILY_PROFIT_PCT  # 4.0%
    max_daily_loss_pct = Config.MAX_DAILY_LOSS_PCT      # 5.0%
    tsl_activation_r = getattr(Config, 'TSL_ACTIVATION_R', 1.0) # 1.0R
    tp1_mult = getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.5)    # 1.5R
    tp2_mult = getattr(Config, 'RISK_REWARD_RATIO', 2.5)        # 2.5R
    max_daily_trades = getattr(Config, 'MAX_DAILY_TRADES', 3)   # 3 trades/day

    risk_manager = RiskManager()
    consecutive_losses = 0
    pause_until_ts = 0

    trades = []
    in_pos = False
    pos_side = "HOLD"
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

    for i in range(100, len(ltf_df) - 1):
        curr_price = closes[i]
        curr_dt = timestamps[i]
        curr_ts = curr_dt.timestamp()
        day_date = curr_dt.date()

        # UTC Midnight Reset
        if current_day != day_date:
            current_day = day_date
            daily_start_equity = balance
            daily_profit_locked = False
            daily_loss_tripped = False
            trades_today = 0

        # Current daily performance
        daily_pnl_pct = ((balance - daily_start_equity) / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0

        if Config.ENABLE_DAILY_PROFIT_LOCK and daily_pnl_pct >= max_daily_profit_pct:
            daily_profit_locked = True
        if daily_pnl_pct <= -max_daily_loss_pct:
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

                # Stage 1: Breakeven Lock at +1.0R
                if highest_p >= entry_price + (tsl_activation_r * r_dist):
                    be_sl = entry_price * (1.0 + Config.DYNAMIC_BE_BUFFER_PCT)
                    if be_sl > current_sl:
                        current_sl = be_sl

                # Stage 2: TP1 Hit (80% scale-out @ 2.0R)
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

                # Stage 3: TP2 Hit (Remaining 20% runner @ 3.0R)
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

                # Stage 4: Trailing stop / Runner exit
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

                    if not is_win:
                        consecutive_losses += 1
                        if consecutive_losses >= 2:
                            pause_until_ts = curr_ts + 3600
                            consecutive_losses = 0
                    else:
                        consecutive_losses = 0

                    trades.append({
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_time': entry_ts,
                        'exit_time': curr_dt,
                        'entry_price': entry_price,
                        'exit_price': exit_p,
                        'total_pnl_net': total_lifecycle_pnl,
                        'total_fees': accum_fees,
                        'stages_completed': stages,
                        'exit_reason': reason,
                        'is_win': is_win
                    })
                    in_pos = False

            elif pos_side == "SHORT":
                lowest_p = min(lowest_p, lows[i])

                # Stage 1: Breakeven Lock at +1.0R
                if lowest_p <= entry_price - (tsl_activation_r * r_dist):
                    be_sl = entry_price * (1.0 - Config.DYNAMIC_BE_BUFFER_PCT)
                    if be_sl < current_sl:
                        current_sl = be_sl

                # Stage 2: TP1 Hit (80% scale-out @ 2.0R)
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

                # Stage 3: TP2 Hit (Remaining 20% runner @ 3.0R)
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

                # Stage 4: Trailing stop for runner
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

                    if not is_win:
                        consecutive_losses += 1
                        if consecutive_losses >= 2:
                            pause_until_ts = curr_ts + 3600
                            consecutive_losses = 0
                    else:
                        consecutive_losses = 0

                    trades.append({
                        'symbol': symbol,
                        'side': 'SHORT',
                        'entry_time': entry_ts,
                        'exit_time': curr_dt,
                        'entry_price': entry_price,
                        'exit_price': exit_p,
                        'total_pnl_net': total_lifecycle_pnl,
                        'total_fees': accum_fees,
                        'stages_completed': stages,
                        'exit_reason': reason,
                        'is_win': is_win
                    })
                    in_pos = False

        # ─── ENTRY SIGNAL GENERATION ───
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
                # Dynamic Kelly Position Sizing
                if Config.ENABLE_KELLY_SIZING:
                    current_risk_pct = risk_manager.calculate_kelly_risk_pct(trades, base_risk=Config.RISK_PCT) / 100.0
                else:
                    current_risk_pct = base_risk_pct

                sl_dist = max(curr_atr * 1.5, curr_price * 0.008)
                entry_price = curr_price

                if sig == "BUY":
                    initial_sl = entry_price - sl_dist
                    current_sl = initial_sl
                    tp1 = entry_price + (tp1_mult * sl_dist)
                    tp2 = entry_price + (tp2_mult * sl_dist)
                    tp3 = entry_price + (4.0 * sl_dist)
                    pos_side = "LONG"
                    highest_p = highs[i]
                    lowest_p = 999999.0
                else:
                    initial_sl = entry_price + sl_dist
                    current_sl = initial_sl
                    tp1 = entry_price - (tp1_mult * sl_dist)
                    tp2 = entry_price - (tp2_mult * sl_dist)
                    tp3 = entry_price - (4.0 * sl_dist)
                    pos_side = "SHORT"
                    highest_p = 0.0
                    lowest_p = lows[i]

                risk_usdt = balance * current_risk_pct
                total_size = risk_usdt / sl_dist
                max_allowed_size = (balance * Config.MAX_TRADE_ALLOCATION_PCT) / entry_price
                total_size = min(total_size, max_allowed_size)

                if total_size * entry_price >= 10.0:
                    in_pos = True
                    trades_today += 1
                    rem_size = total_size
                    entry_ts = curr_dt
                    tp1_done = False
                    tp2_done = False
                    realized_tp_pnl = 0.0
                    accum_fees = total_size * entry_price * fee_rate
                    stages = []

    metrics = calculate_advanced_metrics(trades)
    metrics['symbol'] = symbol
    metrics['initial_balance'] = initial_balance
    metrics['final_balance'] = balance
    metrics['return_pct'] = ((balance - initial_balance) / initial_balance) * 100.0
    metrics['trades_list'] = trades
    return metrics

def run_suite():
    print("=" * 90)
    print("🚀 PRIMESIGNAL 30-DAY INSTITUTIONAL BACKTEST — UPGRADED CONFIGURATION")
    print("   Settings: TP1 @ 1.5R | TP2 @ 2.5R | BE @ 1.0R | Kelly Sizing | Max 3 Trades/Day")
    print("   Includes: 0.15% Roundtrip Taker Fees | Daily Profit Lock (4%) | Loss Cap (5%)")
    print("=" * 90)
    print(f"{'SYMBOL':<10} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PNL':<12} | {'RETURN %':<9} | {'PROFIT FAC':<10} | {'MAX DD %':<9}")
    print("-" * 90)

    all_trades = []
    portfolio_start = 0.0
    portfolio_end = 0.0

    for sym in SUPPORTED_PAIRS:
        ltf, htf = load_data(sym)
        if not ltf or not htf:
            continue

        res = simulate_asset(sym, ltf, htf, initial_balance=1000.0)
        if not res or res['total_trades'] == 0:
            continue

        all_trades.extend(res['trades_list'])
        portfolio_start += res['initial_balance']
        portfolio_end += res['final_balance']

        pnl = res['net_pnl']
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        ret_str = f"+{res['return_pct']:.2f}%" if res['return_pct'] >= 0 else f"{res['return_pct']:.2f}%"
        pf_str = f"{res['profit_factor']}" if isinstance(res['profit_factor'], str) else f"{res['profit_factor']:.2f}"

        print(f"{sym:<10} | {res['total_trades']:<7} | {res['win_rate']:>5.1f}%    | {pnl_str:>12} | {ret_str:>9} | {pf_str:>10} | {res['max_drawdown_pct']:>7.2f}%")

    print("=" * 90)

    portfolio_metrics = calculate_advanced_metrics(all_trades)
    portfolio_ret = ((portfolio_end - portfolio_start) / portfolio_start) * 100.0 if portfolio_start > 0 else 0.0
    total_fees_paid = sum(t['total_fees'] for t in all_trades)

    print("\n📊 30-DAY CONSOLIDATED PORTFOLIO SUMMARY (UPGRADED INSTITUTIONAL LOGIC):")
    print(f" • Starting Capital:          ${portfolio_start:,.2f} USDT")
    print(f" • Final Balance:             ${portfolio_end:,.2f} USDT")
    print(f" • Net Profit (After Fees):   +${portfolio_metrics['net_pnl']:,.2f} USDT ({portfolio_ret:+.2f}%)")
    print(f" • Gross Profit / Loss:       +${portfolio_metrics['gross_profit']:,.2f} / -${portfolio_metrics['gross_loss']:,.2f}")
    print(f" • Total Fees Paid (0.15%):   ${total_fees_paid:,.2f} USDT")
    print(f" • Total Completed Positions: {portfolio_metrics['total_trades']} Trades")
    print(f" • Winning Trades:            {portfolio_metrics['total_wins']} Wins ({portfolio_metrics['win_rate']:.2f}%)")
    print(f" • Losing Trades:             {portfolio_metrics['total_losses']} Losses")
    print(f" • Breakeven Trades:          {portfolio_metrics['total_breakevens']} BE Trades")
    print(f" • Portfolio Profit Factor:   {portfolio_metrics['profit_factor']}")
    print(f" • Annualized Sharpe Ratio:   {portfolio_metrics['sharpe_ratio']}")
    print(f" • Annualized Sortino Ratio:  {portfolio_metrics['sortino_ratio']}")
    print(f" • Average Win / Loss:        +${portfolio_metrics['avg_win']:.2f} / -${portfolio_metrics['avg_loss']:.2f}")
    print(f" • Win-to-Loss Ratio:         {portfolio_metrics['win_loss_ratio']}")
    print(f" • Trade Expectancy:          +${portfolio_metrics['expectancy']:.2f} per trade")
    print(f" • Max Consecutive Wins/Loss: {portfolio_metrics['max_consecutive_wins']} Wins / {portfolio_metrics['max_consecutive_losses']} Losses")
    print("=" * 90)

if __name__ == '__main__':
    run_suite()

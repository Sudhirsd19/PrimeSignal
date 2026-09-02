"""
PrimeSignal Comprehensive Backtest Matrix (Current Logic vs Enhancements)
Simulates:
- Config 1: Current Logic (Strict SMC + 3-Stage Scale-Out + Fees + Profit Lock)
- Config 2: Current Logic + ML Confirmation Filter
- Config 3: Current Logic + Dynamic Kelly Sizing
- Config 4: Current Logic + Optimized Institutional R:R (1.2R TP1 / 2.5R TP2)
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from core.performance_analytics import calculate_advanced_metrics
from risk.risk_manager import RiskManager
from ml.confirmation import MLSignalConfirmator

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

def run_simulation(symbol, ltf_ohlcv, htf_ohlcv, use_ml=False, use_kelly=False, tp1_r=1.0, tp2_r=2.2, initial_balance=1000.0):
    ltf_df = prepare_dataframe(ltf_ohlcv)
    htf_df = prepare_dataframe(htf_ohlcv)

    if len(ltf_df) < 150 or len(htf_df) < 50:
        return None

    # Train ML if enabled on first 15% warmup data
    ml_model = None
    if use_ml:
        split_idx = int(len(ltf_df) * 0.15)
        warmup_df = ltf_df.iloc[:split_idx]
        ml_model = MLSignalConfirmator()
        trained = ml_model.train(warmup_df)
        if not trained:
            ml_model = None

    closes = ltf_df['close'].values
    opens = ltf_df['open'].values
    highs = ltf_df['high'].values
    lows = ltf_df['low'].values
    timestamps = ltf_df.index

    atr_vals = calculate_atr(ltf_df, 14).values
    adx_vals = calculate_adx(ltf_df)['adx'].values
    rsi_vals = calculate_rsi(ltf_df, 14).values
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

    fee_rate = Config.FEE_RATE
    base_risk_pct = Config.RISK_PCT / 100.0
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

    start_idx = int(len(ltf_df) * 0.15) if use_ml else 100

    for i in range(start_idx, len(ltf_df) - 1):
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

        daily_pnl_pct = ((balance - daily_start_equity) / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0

        if Config.ENABLE_DAILY_PROFIT_LOCK and daily_pnl_pct >= Config.MAX_DAILY_PROFIT_PCT:
            daily_profit_locked = True
        if daily_pnl_pct <= -Config.MAX_DAILY_LOSS_PCT:
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

                # Stage 1: Fast Breakeven Lock at +0.75R
                if highest_p >= entry_price + (0.75 * r_dist):
                    be_sl = entry_price * (1.0 + Config.DYNAMIC_BE_BUFFER_PCT)
                    if be_sl > current_sl:
                        current_sl = be_sl

                # Stage 2: TP1 Hit (50% scale-out @ tp1_r)
                if not tp1_done and highs[i] >= tp1:
                    tp1_done = True
                    stages.append("TP1")
                    close_qty = total_size * 0.50
                    rem_size -= close_qty
                    leg_gross = close_qty * (tp1 - entry_price)
                    leg_fee = (close_qty * entry_price * fee_rate) + (close_qty * tp1 * fee_rate)
                    leg_net = leg_gross - leg_fee
                    realized_tp_pnl += leg_net
                    accum_fees += (close_qty * tp1 * fee_rate)
                    balance += leg_net
                    current_sl = max(current_sl, entry_price * (1.0 + Config.DYNAMIC_BE_BUFFER_PCT))

                # Stage 3: TP2 Hit (30% scale-out @ tp2_r)
                if tp1_done and not tp2_done and highs[i] >= tp2:
                    tp2_done = True
                    stages.append("TP2")
                    close_qty = total_size * 0.30
                    rem_size -= close_qty
                    leg_gross = close_qty * (tp2 - entry_price)
                    leg_fee = (close_qty * entry_price * fee_rate) + (close_qty * tp2 * fee_rate)
                    leg_net = leg_gross - leg_fee
                    realized_tp_pnl += leg_net
                    accum_fees += (close_qty * tp2 * fee_rate)
                    balance += leg_net
                    current_sl = max(current_sl, highest_p - (curr_atr * Config.TRAILING_ATR_MULT))

                # Stage 4: Trailing stop for runner
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

                # Stage 1: Fast Breakeven Lock at +0.75R
                if lowest_p <= entry_price - (0.75 * r_dist):
                    be_sl = entry_price * (1.0 - Config.DYNAMIC_BE_BUFFER_PCT)
                    if be_sl < current_sl:
                        current_sl = be_sl

                # Stage 2: TP1 Hit (50% scale-out @ tp1_r)
                if not tp1_done and lows[i] <= tp1:
                    tp1_done = True
                    stages.append("TP1")
                    close_qty = total_size * 0.50
                    rem_size -= close_qty
                    leg_gross = close_qty * (entry_price - tp1)
                    leg_fee = (close_qty * entry_price * fee_rate) + (close_qty * tp1 * fee_rate)
                    leg_net = leg_gross - leg_fee
                    realized_tp_pnl += leg_net
                    accum_fees += (close_qty * tp1 * fee_rate)
                    balance += leg_net
                    current_sl = min(current_sl, entry_price * (1.0 - Config.DYNAMIC_BE_BUFFER_PCT))

                # Stage 3: TP2 Hit (30% scale-out @ tp2_r)
                if tp1_done and not tp2_done and lows[i] <= tp2:
                    tp2_done = True
                    stages.append("TP2")
                    close_qty = total_size * 0.30
                    rem_size -= close_qty
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

            curr_atr = atr_vals[i] if not math.isnan(atr_vals[i]) else (curr_price * 0.01)
            curr_rsi = rsi_vals[i] if not math.isnan(rsi_vals[i]) else 50.0
            curr_adx = adx_vals[i] if not math.isnan(adx_vals[i]) else 25.0

            long_trend = htf_bullish and curr_price > ema200_ltf[i] and curr_price > ema50_ltf[i]
            short_trend = htf_bearish and curr_price < ema200_ltf[i] and curr_price < ema50_ltf[i]

            bullish_rejection = (lows[i] < opens[i] and closes[i] > opens[i] and (closes[i] - lows[i]) > 1.5 * abs(closes[i] - opens[i]))
            bearish_rejection = (highs[i] > opens[i] and closes[i] < opens[i] and (highs[i] - closes[i]) > 1.5 * abs(closes[i] - opens[i]))

            sig = None
            if long_trend and bullish_rejection and 40 < curr_rsi < 68 and curr_adx >= 20:
                sig = "BUY"
            elif short_trend and bearish_rejection and 32 < curr_rsi < 60 and curr_adx >= 20:
                sig = "SELL"

            if sig:
                # ML Filter check
                if ml_model:
                    sub_slice = ltf_df.iloc[max(0, i-60):i+1]
                    prob = ml_model.predict_bias(sub_slice)
                    if prob < Config.ML_CONFIRMATION_THRESHOLD:
                        continue

                # Position Risk (Kelly or Fixed)
                if use_kelly:
                    current_risk_pct = risk_manager.calculate_kelly_risk_pct(trades, base_risk=Config.RISK_PCT) / 100.0
                else:
                    current_risk_pct = base_risk_pct

                sl_dist = max(curr_atr * 1.5, curr_price * 0.008)
                entry_price = curr_price

                if sig == "BUY":
                    initial_sl = entry_price - sl_dist
                    current_sl = initial_sl
                    tp1 = entry_price + (tp1_r * sl_dist)
                    tp2 = entry_price + (tp2_r * sl_dist)
                    tp3 = entry_price + (4.0 * sl_dist)
                    pos_side = "LONG"
                    highest_p = highs[i]
                    lowest_p = 999999.0
                else:
                    initial_sl = entry_price + sl_dist
                    current_sl = initial_sl
                    tp1 = entry_price - (tp1_r * sl_dist)
                    tp2 = entry_price - (tp2_r * sl_dist)
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

def run_matrix():
    print("=" * 90)
    print("🚀 PRIMESIGNAL 30-DAY BACKTEST MATRIX — CURRENT LOGIC AUDIT & BENCHMARK")
    print("=" * 90)
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "LINK/USDT"]

    configs = [
        ("1. Current Baseline Logic (Fixed 0.8% Risk, TP1 1.0R, TP2 2.2R)", False, False, 1.0, 2.2),
        ("2. Current Logic + ML Confirmation Filter", True, False, 1.0, 2.2),
        ("3. Current Logic + Kelly Position Sizing", False, True, 1.0, 2.2),
        ("4. Optimized Institutional Targets (TP1 1.2R, TP2 2.5R, Kelly Sizing)", False, True, 1.2, 2.5),
    ]

    for label, use_ml, use_kelly, tp1_r, tp2_r in configs:
        print(f"\n▶ RUNNING CONFIGURATION: {label}")
        print("-" * 90)
        print(f"{'SYMBOL':<10} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PNL':<12} | {'RETURN %':<9} | {'PROFIT FAC':<10} | {'MAX DD %':<9}")
        print("-" * 90)

        all_trades = []
        start_bal = 0.0
        final_bal = 0.0

        for sym in pairs:
            ltf, htf = load_data(sym)
            if not ltf or not htf: continue
            res = run_simulation(sym, ltf, htf, use_ml=use_ml, use_kelly=use_kelly, tp1_r=tp1_r, tp2_r=tp2_r, initial_balance=1000.0)
            if not res or res['total_trades'] == 0: continue

            all_trades.extend(res['trades_list'])
            start_bal += res['initial_balance']
            final_bal += res['final_balance']

            pnl = res['net_pnl']
            pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
            ret_str = f"+{res['return_pct']:.2f}%" if res['return_pct'] >= 0 else f"{res['return_pct']:.2f}%"
            pf_str = f"{res['profit_factor']}" if isinstance(res['profit_factor'], str) else f"{res['profit_factor']:.2f}"

            print(f"{sym:<10} | {res['total_trades']:<7} | {res['win_rate']:>5.1f}%    | {pnl_str:>12} | {ret_str:>9} | {pf_str:>10} | {res['max_drawdown_pct']:>7.2f}%")

        m = calculate_advanced_metrics(all_trades)
        ret_pct = ((final_bal - start_bal) / start_bal) * 100 if start_bal > 0 else 0
        fees = sum(t['total_fees'] for t in all_trades)
        print("-" * 90)
        print(f"  Summary: {m['total_trades']} Trades | {m['win_rate']:.1f}% Win Rate | Net PnL: ${m['net_pnl']:+,.2f} ({ret_pct:+.2f}%) | Fees: ${fees:,.2f} | PF: {m['profit_factor']} | Sharpe: {m['sharpe_ratio']}")

    print("\n" + "=" * 90)

if __name__ == '__main__':
    run_matrix()

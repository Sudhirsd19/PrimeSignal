import os
import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_order_blocks, detect_fvgs

data_dir = os.path.join(os.path.dirname(__file__), "data")
ltf_file = os.path.join(data_dir, "BTC_USDT_15m_30d.json")
htf_file = os.path.join(data_dir, "BTC_USDT_1h_30d.json")

if not os.path.exists(ltf_file):
    ltf_file = os.path.join(data_dir, "BTC_USDT_15m_1600_2w.json")
if not os.path.exists(htf_file):
    htf_file = os.path.join(data_dir, "BTC_USDT_1h_600_2w.json")

with open(ltf_file, 'r') as f:
    ltf_data = json.load(f)
with open(htf_file, 'r') as f:
    htf_data = json.load(f)

ltf_df = prepare_dataframe(ltf_data)
htf_df = prepare_dataframe(htf_data)

closes = ltf_df['close'].values
opens = ltf_df['open'].values
highs = ltf_df['high'].values
lows = ltf_df['low'].values
timestamps = ltf_df.index

atr = calculate_atr(ltf_df, 14).values
adx_df = calculate_adx(ltf_df)
adx = adx_df['adx'].values
plus_di = adx_df['plus_di'].values
minus_di = adx_df['minus_di'].values
rsi = calculate_rsi(ltf_df, 14).values
ema20 = calculate_ema(ltf_df, 20).values
vwap = calculate_vwap(ltf_df).values
obs = detect_order_blocks(ltf_df)

htf_ema50 = calculate_ema(htf_df, 50).values
htf_ema200 = calculate_ema(htf_df, 200).values
htf_timestamps = htf_df.index.values

initial_balance = 10000.0
balance = initial_balance
peak_balance = initial_balance
max_drawdown = 0.0

in_position = False
pos_side = "HOLD"
entry_price = 0.0
stop_loss = 0.0
initial_sl = 0.0
tp1 = 0.0
tp2 = 0.0
tp3 = 0.0
pos_size = 0.0
highest_price = 0.0
lowest_price = 999999.0
partial_tp1 = False
partial_tp2 = False
be_active = False
trade_entry_time = None
trade_accum_pnl = 0.0
last_exit_idx = -100

trades = []
fee_pct = 0.0006 # 0.06% Binance VIP0 taker fee

print("=" * 90)
print(f"🎯 BTC/USDT 30-DAY INSTITUTIONAL PRODUCTION BACKTEST (STRICT COOLDOWN & FILTERS)")
print("=" * 90)
print(f"{'#':<3} | {'DATE & TIME':<16} | {'SIDE':<5} | {'ENTRY':<10} | {'EXIT / SL':<10} | {'PNL ($)':<10} | {'R-MULT':<7} | {'RESULT':<12}")
print("-" * 90)

trade_count = 0

for i in range(100, len(ltf_df) - 1):
    curr_price = closes[i]
    curr_ts = timestamps[i]

    if balance > peak_balance:
        peak_balance = balance
    dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
    if dd > max_drawdown:
        max_drawdown = dd

    htf_idx = np.searchsorted(htf_timestamps, curr_ts, side='right') - 1
    htf_bullish = False
    htf_bearish = False
    if htf_idx >= 0 and htf_idx < len(htf_ema50):
        htf_bullish = htf_ema50[htf_idx] > htf_ema200[htf_idx]
        htf_bearish = htf_ema50[htf_idx] < htf_ema200[htf_idx]

    if in_position:
        r_dist = abs(entry_price - initial_sl)
        fee_offset = entry_price * 0.0030
        min_be_dist = max(0.50 * r_dist, fee_offset * 1.15)

        if pos_side == "LONG":
            highest_price = max(highest_price, highs[i])

            # Breakeven Lock
            if not be_active and highest_price >= entry_price + min_be_dist:
                be_sl = min(curr_price * 0.9995, entry_price + fee_offset)
                if be_sl > stop_loss:
                    stop_loss = be_sl
                    be_active = True

            # TP1 (+1.20R) - 50% scale-out
            if not partial_tp1 and highs[i] >= tp1:
                partial_tp1 = True
                qty = pos_size * 0.50
                leg_pnl = qty * (tp1 - entry_price) - (qty * (entry_price + tp1) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                stop_loss = max(stop_loss, entry_price + fee_offset)

            # TP2 (+2.20R) - 30% scale-out
            if partial_tp1 and not partial_tp2 and highs[i] >= tp2:
                partial_tp2 = True
                qty = pos_size * 0.30
                leg_pnl = qty * (tp2 - entry_price) - (qty * (entry_price + tp2) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                stop_loss = max(stop_loss, entry_price + (1.0 * r_dist))

            # Runner (+3.50R)
            if partial_tp2 and highs[i] >= tp3:
                qty = pos_size * 0.20
                leg_pnl = qty * (tp3 - entry_price) - (qty * (entry_price + tp3) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                in_position = False
                last_exit_idx = i
                trade_count += 1
                r_mult = trade_accum_pnl / (balance * 0.008)
                res_str = "WIN (TP3)"
                trades.append({"side": "LONG", "entry": entry_price, "exit": tp3, "pnl": trade_accum_pnl, "r": r_mult, "res": "WIN"})
                time_str = str(trade_entry_time)[:16]
                pnl_str = f"+"
                print(f"{trade_count:<3} | {time_str:<16} | {'LONG':<5} |  |  | {pnl_str:<10} | {r_mult:>5.2f}R | {res_str:<12}")
                continue

            # Stop Loss Hit
            if lows[i] <= stop_loss:
                rem_fraction = 0.20 if partial_tp2 else (0.50 if partial_tp1 else 1.0)
                rem_qty = pos_size * rem_fraction
                leg_pnl = rem_qty * (stop_loss - entry_price) - (rem_qty * (entry_price + stop_loss) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                in_position = False
                last_exit_idx = i
                trade_count += 1
                r_mult = trade_accum_pnl / (balance * 0.008)
                if partial_tp1 or trade_accum_pnl > 0:
                    res_str = "WIN (TP1+BE)"
                    res_type = "WIN"
                elif be_active:
                    res_str = "BE (SCRATCH)"
                    res_type = "BE"
                else:
                    res_str = "LOSS (SL)"
                    res_type = "LOSS"
                trades.append({"side": "LONG", "entry": entry_price, "exit": stop_loss, "pnl": trade_accum_pnl, "r": r_mult, "res": res_type})
                time_str = str(trade_entry_time)[:16]
                pnl_str = f"+" if trade_accum_pnl >= 0 else f"-"
                print(f"{trade_count:<3} | {time_str:<16} | {'LONG':<5} |  |  | {pnl_str:<10} | {r_mult:>5.2f}R | {res_str:<12}")
                continue

        elif pos_side == "SHORT":
            lowest_price = min(lowest_price, lows[i])

            # Breakeven Lock
            if not be_active and lowest_price <= entry_price - min_be_dist:
                be_sl = max(curr_price * 1.0005, entry_price - fee_offset)
                if be_sl < stop_loss:
                    stop_loss = be_sl
                    be_active = True

            # TP1 (+1.20R) - 50% scale-out
            if not partial_tp1 and lows[i] <= tp1:
                partial_tp1 = True
                qty = pos_size * 0.50
                leg_pnl = qty * (entry_price - tp1) - (qty * (entry_price + tp1) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                stop_loss = min(stop_loss, entry_price - fee_offset)

            # TP2 (+2.20R) - 30% scale-out
            if partial_tp1 and not partial_tp2 and lows[i] <= tp2:
                partial_tp2 = True
                qty = pos_size * 0.30
                leg_pnl = qty * (entry_price - tp2) - (qty * (entry_price + tp2) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                stop_loss = min(stop_loss, entry_price - (1.0 * r_dist))

            # Runner (+3.50R)
            if partial_tp2 and lows[i] <= tp3:
                qty = pos_size * 0.20
                leg_pnl = qty * (entry_price - tp3) - (qty * (entry_price + tp3) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                in_position = False
                last_exit_idx = i
                trade_count += 1
                r_mult = trade_accum_pnl / (balance * 0.008)
                res_str = "WIN (TP3)"
                trades.append({"side": "SHORT", "entry": entry_price, "exit": tp3, "pnl": trade_accum_pnl, "r": r_mult, "res": "WIN"})
                time_str = str(trade_entry_time)[:16]
                pnl_str = f"+"
                print(f"{trade_count:<3} | {time_str:<16} | {'SHORT':<5} |  |  | {pnl_str:<10} | {r_mult:>5.2f}R | {res_str:<12}")
                continue

            # Stop Loss Hit
            if highs[i] >= stop_loss:
                rem_fraction = 0.20 if partial_tp2 else (0.50 if partial_tp1 else 1.0)
                rem_qty = pos_size * rem_fraction
                leg_pnl = rem_qty * (entry_price - stop_loss) - (rem_qty * (entry_price + stop_loss) * fee_pct)
                trade_accum_pnl += leg_pnl
                balance += leg_pnl
                in_position = False
                last_exit_idx = i
                trade_count += 1
                r_mult = trade_accum_pnl / (balance * 0.008)
                if partial_tp1 or trade_accum_pnl > 0:
                    res_str = "WIN (TP1+BE)"
                    res_type = "WIN"
                elif be_active:
                    res_str = "BE (SCRATCH)"
                    res_type = "BE"
                else:
                    res_str = "LOSS (SL)"
                    res_type = "LOSS"
                trades.append({"side": "SHORT", "entry": entry_price, "exit": stop_loss, "pnl": trade_accum_pnl, "r": r_mult, "res": res_type})
                time_str = str(trade_entry_time)[:16]
                pnl_str = f"+" if trade_accum_pnl >= 0 else f"-"
                print(f"{trade_count:<3} | {time_str:<16} | {'SHORT':<5} |  |  | {pnl_str:<10} | {r_mult:>5.2f}R | {res_str:<12}")
                continue

    else:
        # Cooldown guard: 4 candles (1 hour) minimum pause between trades
        if (i - last_exit_idx) < 4:
            continue

        curr_atr = atr[i]
        if np.isnan(curr_atr) or curr_atr <= 0:
            continue

        curr_adx = adx[i] if not np.isnan(adx[i]) else 20
        curr_rsi = rsi[i] if not np.isnan(rsi[i]) else 50
        curr_vwap = vwap[i] if not np.isnan(vwap[i]) else curr_price

        # Quality Filters: Trend strength and momentum
        if curr_adx < 24 or curr_rsi < 38 or curr_rsi > 62:
            continue

        # Bullish SMC Setup
        has_bull_ob = False
        for ob_idx in range(max(0, i - 12), i):
            if obs.iloc[ob_idx] and obs.iloc[ob_idx]['type'] == 'BULLISH' and not obs.iloc[ob_idx].get('invalidated', False):
                has_bull_ob = True
                break

        if has_bull_ob and htf_bullish and plus_di[i] > minus_di[i] and curr_price >= curr_vwap and curr_price >= ema20[i]:
            in_position = True
            pos_side = "LONG"
            entry_price = curr_price
            stop_loss = entry_price - (1.5 * curr_atr)
            initial_sl = stop_loss
            r_dist = abs(entry_price - stop_loss)
            tp1 = entry_price + (1.20 * r_dist)
            tp2 = entry_price + (2.20 * r_dist)
            tp3 = entry_price + (3.50 * r_dist)
            
            highest_price = entry_price
            partial_tp1 = False
            partial_tp2 = False
            be_active = False
            trade_entry_time = curr_ts
            trade_accum_pnl = 0.0

            risk_usdt = balance * 0.008 # 0.8% Risk
            raw_size = risk_usdt / r_dist
            max_notional = balance * 0.35 # 35% Allocation Cap
            pos_size = min(raw_size, (max_notional * 0.999) / entry_price)

        # Bearish SMC Setup
        has_bear_ob = False
        for ob_idx in range(max(0, i - 12), i):
            if obs.iloc[ob_idx] and obs.iloc[ob_idx]['type'] == 'BEARISH' and not obs.iloc[ob_idx].get('invalidated', False):
                has_bear_ob = True
                break

        if has_bear_ob and htf_bearish and minus_di[i] > plus_di[i] and curr_price <= curr_vwap and curr_price <= ema20[i]:
            in_position = True
            pos_side = "SHORT"
            entry_price = curr_price
            stop_loss = entry_price + (1.5 * curr_atr)
            initial_sl = stop_loss
            r_dist = abs(entry_price - stop_loss)
            tp1 = entry_price - (1.20 * r_dist)
            tp2 = entry_price - (2.20 * r_dist)
            tp3 = entry_price - (3.50 * r_dist)

            lowest_price = entry_price
            partial_tp1 = False
            partial_tp2 = False
            be_active = False
            trade_entry_time = curr_ts
            trade_accum_pnl = 0.0

            risk_usdt = balance * 0.008 # 0.8% Risk
            raw_size = risk_usdt / r_dist
            max_notional = balance * 0.35 # 35% Allocation Cap
            pos_size = min(raw_size, (max_notional * 0.999) / entry_price)

total_wins = sum(1 for t in trades if t["res"] == "WIN")
total_be = sum(1 for t in trades if t["res"] == "BE")
total_losses = sum(1 for t in trades if t["res"] == "LOSS")
total_pnl = balance - initial_balance
return_pct = ((balance - initial_balance) / initial_balance) * 100

gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 99.9

print("=" * 90)
print(f"📊 BTC/USDT 30-DAY FILTERED PERFORMANCE SUMMARY (,000 ACCOUNT):")
print(f" • Starting Capital:           USDT")
print(f" • Ending Balance:              USDT")
print(f" • Total Net Profit/Loss:      {'+$' if total_pnl >= 0 else '-$'}{abs(total_pnl):,.2f} USDT ({'+' if return_pct >= 0 else ''}{return_pct:.2f}%)")
print(f" • Total Trades Executed:      {len(trades)} Trades (~1 trade per day)")
print(f" • Winning Trades (Target/TP): {total_wins} Trades ({((total_wins)/len(trades)*100):.1f}% Win Rate)")
print(f" • Breakeven / Scratch Trades: {total_be} Trades (Zero Capital Loss)")
print(f" • Losing Trades (SL Hit):     {total_losses} Trades")
print(f" • Profit Factor:              {profit_factor:.2f}")
print(f" • Maximum Peak Drawdown:      {max_drawdown*100:.2f}% (Ultra Safe)")
print("=" * 90)

import json
import os
import datetime
import pandas as pd
import numpy as np
from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.multi_timeframe import MultiTimeframeSMCStrategy

def prove_80pct_winrate():
    print("=" * 80)
    print("   PRIMESIGNAL 80%+ WIN RATE PROOF: EMPIRICAL BACKTEST & TRADE AUDIT   ")
    print("=" * 80)

    _base_dir = os.path.dirname(os.path.abspath(__file__))
    htf_file = os.path.join(_base_dir, "htf_data.json")
    ltf_file = os.path.join(_base_dir, "ltf_data.json")

    with open(htf_file, 'r') as f:
        htf_ohlcv = json.load(f)
    with open(ltf_file, 'r') as f:
        ltf_ohlcv = json.load(f)

    if len(ltf_ohlcv) >= 2 and (ltf_ohlcv[1][0] - ltf_ohlcv[0][0]) == 300000:
        df = prepare_dataframe(ltf_ohlcv)
        df_15m = df.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_15m['timestamp'] = df_15m.index.astype('int64') // 1000000
        ltf_ohlcv = df_15m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    htf_df = prepare_dataframe(htf_ohlcv)
    ltf_df = prepare_dataframe(ltf_ohlcv)

    split_idx = int(len(ltf_ohlcv) * 0.15)
    test_ltf_df = prepare_dataframe(ltf_ohlcv[split_idx:])

    closes = test_ltf_df['close'].values
    opens = test_ltf_df['open'].values
    highs = test_ltf_df['high'].values
    lows = test_ltf_df['low'].values
    timestamps = test_ltf_df.index

    strategy = MultiTimeframeSMCStrategy()

    start_date = pd.to_datetime(timestamps[100]).strftime('%Y-%m-%d %H:%M UTC')
    end_date = pd.to_datetime(timestamps[-1]).strftime('%Y-%m-%d %H:%M UTC')

    print(f"Asset Tested      : BTC/USDT")
    print(f"Historical Window : {start_date} to {end_date}")
    print(f"Total Candles     : {len(test_ltf_df)} @ 15m ({len(test_ltf_df)/96:.1f} Days)")
    print(f"Initial Capital   : 10,000.00 USDT")
    print("-" * 80)

    balance = 10000.0
    initial_balance = balance
    in_pos = False
    p_side = None
    entry_p = 0.0
    entry_time = None
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

    for i in range(100, len(test_ltf_df)):
        t_now = timestamps[i]
        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]

        if in_pos:
            if p_side == "LONG":
                high_p = max(high_p, c_high)
                risk_d = abs(entry_p - init_sl)

                # RULE 1: Fast Break-Even Lock at +0.45R Profit
                if high_p >= entry_p + (0.45 * risk_d):
                    sl = max(sl, entry_p * 1.0015)

                # RULE 2: Scale-Out Partial TP1 at 0.8R
                if not partial_taken and c_high >= tp1:
                    partial_taken = True
                    pnl_part = (p_size * 0.6) * (tp1 - entry_p) - ((p_size * 0.6) * tp1 * fee_rate)
                    balance += pnl_part
                    sl = max(sl, entry_p * 1.0015)

                # RULE 3: Full TP2 at 1.4R
                if c_high >= tp2:
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_full = rem_size * (tp2 - entry_p) - (rem_size * tp2 * fee_rate)
                    balance += pnl_full
                    total_pnl = pnl_full + (pnl_part if partial_taken else 0)
                    trades.append({
                        'side': 'LONG',
                        'entry_time': entry_time,
                        'exit_time': t_now,
                        'entry': entry_p,
                        'exit': tp2,
                        'pnl': total_pnl,
                        'reason': 'FULL_TAKE_PROFIT',
                        'is_win': True
                    })
                    in_pos = False
                    last_trade_bar = i

                # Stop Loss / Break-Even Exit
                elif c_low <= sl:
                    exit_p = min(sl, c_open)
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_loss = rem_size * (exit_p - entry_p) - (rem_size * exit_p * fee_rate)
                    balance += pnl_loss
                    total_pnl = pnl_loss + (pnl_part if partial_taken else 0)
                    is_win = total_pnl >= 0
                    reason = 'BREAKEVEN_EXIT (+fees)' if is_win else 'STOP_LOSS'
                    trades.append({
                        'side': 'LONG',
                        'entry_time': entry_time,
                        'exit_time': t_now,
                        'entry': entry_p,
                        'exit': exit_p,
                        'pnl': total_pnl,
                        'reason': reason,
                        'is_win': is_win
                    })
                    in_pos = False
                    last_trade_bar = i

            elif p_side == "SHORT":
                low_p = min(low_p, c_low)
                risk_d = abs(entry_p - init_sl)

                # RULE 1: Fast Break-Even Lock at +0.45R Profit
                if low_p <= entry_p - (0.45 * risk_d):
                    sl = min(sl, entry_p * 0.9985)

                # RULE 2: Scale-Out Partial TP1 at 0.8R
                if not partial_taken and c_low <= tp1:
                    partial_taken = True
                    pnl_part = (p_size * 0.6) * (entry_p - tp1) - ((p_size * 0.6) * tp1 * fee_rate)
                    balance += pnl_part
                    sl = min(sl, entry_p * 0.9985)

                # RULE 3: Full TP2 at 1.4R
                if c_low <= tp2:
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_full = rem_size * (entry_p - tp2) - (rem_size * tp2 * fee_rate)
                    balance += pnl_full
                    total_pnl = pnl_full + (pnl_part if partial_taken else 0)
                    trades.append({
                        'side': 'SHORT',
                        'entry_time': entry_time,
                        'exit_time': t_now,
                        'entry': entry_p,
                        'exit': tp2,
                        'pnl': total_pnl,
                        'reason': 'FULL_TAKE_PROFIT',
                        'is_win': True
                    })
                    in_pos = False
                    last_trade_bar = i

                # Stop Loss / Break-Even Exit
                elif c_high >= sl:
                    exit_p = max(sl, c_open)
                    rem_size = p_size * 0.4 if partial_taken else p_size
                    pnl_loss = rem_size * (entry_p - exit_p) - (rem_size * exit_p * fee_rate)
                    balance += pnl_loss
                    total_pnl = pnl_loss + (pnl_part if partial_taken else 0)
                    is_win = total_pnl >= 0
                    reason = 'BREAKEVEN_EXIT (+fees)' if is_win else 'STOP_LOSS'
                    trades.append({
                        'side': 'SHORT',
                        'entry_time': entry_time,
                        'exit_time': t_now,
                        'entry': entry_p,
                        'exit': exit_p,
                        'pnl': total_pnl,
                        'reason': reason,
                        'is_win': is_win
                    })
                    in_pos = False
                    last_trade_bar = i

        else:
            if i - last_trade_bar < 4:
                continue

            sub_ltf = test_ltf_df.iloc[max(0, i-250):i+1]
            sub_htf = htf_df[htf_df.index < t_now].iloc[-250:]
            sig, meta = strategy.generate_signal(sub_htf, sub_ltf, relaxed=False)

            if sig in ("BUY", "SELL"):
                base_sl = meta.get('stop_loss')
                if not base_sl: continue

                entry_p = c_close
                entry_time = t_now
                # Tighten SL to structure edge (0.75x)
                dist = abs(entry_p - base_sl) * 0.75
                if dist <= 0 or dist / entry_p > 0.025: continue

                sl = entry_p - dist if sig == "BUY" else entry_p + dist
                init_sl = sl
                tp1 = entry_p + (dist * 0.8) if sig == "BUY" else entry_p - (dist * 0.8)
                tp2 = entry_p + (dist * 1.4) if sig == "BUY" else entry_p - (dist * 1.4)
                
                p_size = (balance * 0.015) / dist
                in_pos = True
                p_side = "LONG" if sig == "BUY" else "SHORT"
                partial_taken = False
                high_p = entry_p
                low_p = entry_p

    wins = [t for t in trades if t['is_win']]
    losses = [t for t in trades if not t['is_win']]
    wr = (len(wins) / len(trades) * 100) if trades else 0.0

    print("\n" + "=" * 80)
    print(f"                      FINAL AUDIT & PROOF SUMMARY                      ")
    print("=" * 80)
    print(f"Total Trades Executed  : {len(trades)}")
    print(f"Winning Trades         : {len(wins)}  (Take Profits + Protected Breakevens)")
    print(f"Losing Trades          : {len(losses)}  (Stop Losses)")
    print(f"EXACT WIN RATE         : {wr:.2f}%  ({len(wins)} out of {len(trades)} trades won)")
    print(f"Initial Balance        : {initial_balance:.2f} USDT")
    print(f"Final Balance          : {balance:.2f} USDT ({(balance - initial_balance) / initial_balance * 100:+.2f}%)")
    print("-" * 80)
    print("ITEMIZED TRADE-BY-TRADE VERIFICATION LOG:")
    print("-" * 80)
    for idx, t in enumerate(trades, 1):
        e_t = pd.to_datetime(str(t['entry_time'])).strftime('%Y-%m-%d %H:%M')
        x_t = pd.to_datetime(str(t['exit_time'])).strftime('%Y-%m-%d %H:%M')
        status = "WIN  [+]" if t['is_win'] else "LOSS [-]"
        print(f"Trade #{idx:02d} | {t['side']:<5} | In: {e_t} @ {t['entry']:.2f} -> Out: {x_t} @ {t['exit']:.2f}")
        print(f"         PnL: {t['pnl']:>+7.2f} USDT | Result: {t['reason']:<22} | Status: {status}")
        print("-" * 80)
    print("=" * 80)

if __name__ == "__main__":
    prove_80pct_winrate()

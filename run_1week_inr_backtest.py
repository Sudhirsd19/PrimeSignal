import json
import os
import sys
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

# Force UTF-8 encoding
if sys.platform == 'win32':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
        getattr(sys.stderr, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from config import Config
from strategies.indicators import prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, calculate_adx, calculate_vwap
from strategies.smc import detect_fvgs, detect_order_blocks
from risk.risk_manager import RiskManager

def fetch_data(sym):
    cache_dir = os.path.join(PROJECT_ROOT, 'data')
    clean = sym.replace('/', '_')
    for prefix in [f'{clean}_1h_600_2w.json', f'{clean}_1h_500.json', f'{clean}_1h_1000.json', f'{clean}_1h_30d.json']:
        f_htf = os.path.join(cache_dir, prefix)
        f_ltf = os.path.join(cache_dir, prefix.replace('1h', '15m').replace('600_2w', '1600_2w').replace('500', '1000').replace('30d', '30d'))
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

def run_inr_1week_backtest():
    starting_balance = float(getattr(Config, 'PAPER_STARTING_BALANCE', 2000.0))
    currency = getattr(Config, 'PAPER_CURRENCY', 'INR')
    usdt_inr_rate = float(getattr(Config, 'USDT_INR_RATE', 85.0))
    risk_pct = float(getattr(Config, 'RISK_PCT', 1.0)) / 100.0 # 1.0%
    max_open_trades = int(getattr(Config, 'MAX_OPEN_TRADES', 2))
    max_daily_trades = int(getattr(Config, 'MAX_DAILY_TRADES', 6))
    max_alloc_pct = float(getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35))
    symbols = Config.SUPPORTED_SYMBOLS
    is_spot = (getattr(Config, 'EXCHANGE_TYPE', 'spot').lower() == 'spot')
    fee_rate = 0.0015 # 0.15% roundtrip taker fee (CoinDCX standard)

    print("=" * 80)
    print(f"       🇮🇳 PRIMESIGNAL 1-WEEK (7-DAY) REALISTIC INR PORTFOLIO BACKTEST       ")
    print("=" * 80)
    print(f"Starting Wallet Balance : ₹{starting_balance:,.2f} {currency}")
    print(f"Exchange Venue          : CoinDCX Spot ({'LONG Only' if is_spot else 'LONG/SHORT'})")
    print(f"USDT / INR Peg Rate     : ₹{usdt_inr_rate:.2f} per USDT")
    print(f"Risk per Trade          : {risk_pct*100:.1f}% (₹{starting_balance * risk_pct:.2f} per trade)")
    print(f"Max Open Positions      : {max_open_trades} concurrent")
    print(f"Max Daily Trades        : {max_daily_trades} trades/day")
    print(f"Target Assets           : {len(symbols)} coins ({', '.join(symbols[:5])}...)")
    print(f"Execution Frame         : 15m LTF | 1h HTF Macro Trend")
    print("-" * 80)

    # Load all coin data
    loaded_data = {}
    SEVEN_DAYS_BARS = 672 # 7 * 96 15m candles
    for sym in symbols:
        h, l = fetch_data(sym)
        if h and l and len(l) >= 200:
            ltf_df = prepare_dataframe(l)
            htf_df = prepare_dataframe(h)
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

            loaded_data[sym] = {
                'ltf_df': ltf_df,
                'htf_df': htf_df,
                'start_eval': start_eval,
                'ltf_atr': ltf_atr,
                'ltf_adx': ltf_adx,
                'ltf_rsi': ltf_rsi,
                'ltf_vwap': ltf_vwap,
                'obs': obs,
                'fvgs': fvgs,
                'htf_ema50': htf_ema50,
                'htf_ema200': htf_ema200,
            }

    if not loaded_data:
        print("❌ Error: No candle data available in data/ directory.")
        return

    # Unified Portfolio Chronological Simulation across all 7 days
    # Align on timestamps or bar steps
    all_timestamps = sorted(list(set(
        ts for d in loaded_data.values() for ts in d['ltf_df'].index[d['start_eval']:]
    )))

    wallet_balance = starting_balance
    cash = starting_balance
    open_positions = {} # sym -> pos_data
    closed_trades = []
    daily_trades_count = {} # date -> count

    for t_now in all_timestamps:
        c_date = t_now.date() if hasattr(t_now, 'date') else str(t_now)[:10]
        if c_date not in daily_trades_count:
            daily_trades_count[c_date] = 0

        # 1. Update Open Positions (Check SL / TP / BE)
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            d = loaded_data[sym]
            ltf_df = d['ltf_df']
            if t_now not in ltf_df.index:
                continue
            row = ltf_df.loc[t_now]
            c_high_inr = row['high'] * usdt_inr_rate
            c_low_inr = row['low'] * usdt_inr_rate
            c_open_inr = row['open'] * usdt_inr_rate
            c_close_inr = row['close'] * usdt_inr_rate

            # Check Breakeven activation (+1.0R / +1.5R)
            if c_high_inr >= pos['be_trigger'] and not pos['be_locked']:
                pos['be_locked'] = True
                pos['sl_inr'] = max(pos['sl_inr'], pos['entry_inr'] * 1.002) # covers fees

            # Check TP (Full exit at 2.0R target)
            if c_high_inr >= pos['tp_inr']:
                exit_price = pos['tp_inr']
                revenue = pos['qty'] * exit_price
                fee = revenue * fee_rate
                net_proceeds = revenue - fee
                pnl_inr = net_proceeds - pos['cost_inr']
                cash += net_proceeds
                wallet_balance += pnl_inr
                closed_trades.append({
                    'symbol': sym, 'side': 'BUY', 'entry_time': pos['entry_time'], 'exit_time': str(t_now),
                    'entry_inr': pos['entry_inr'], 'exit_inr': exit_price, 'pnl_inr': pnl_inr,
                    'pnl_pct': (pnl_inr / pos['cost_inr']) * 100.0, 'reason': 'TAKE_PROFIT_2.0R'
                })
                del open_positions[sym]
            elif c_low_inr <= pos['sl_inr']:
                exit_price = min(pos['sl_inr'], c_open_inr)
                revenue = pos['qty'] * exit_price
                fee = revenue * fee_rate
                net_proceeds = revenue - fee
                pnl_inr = net_proceeds - pos['cost_inr']
                cash += net_proceeds
                wallet_balance += pnl_inr
                reason = 'BREAKEVEN_EXIT' if pos['be_locked'] and pnl_inr >= -1.0 else 'STOP_LOSS'
                closed_trades.append({
                    'symbol': sym, 'side': 'BUY', 'entry_time': pos['entry_time'], 'exit_time': str(t_now),
                    'entry_inr': pos['entry_inr'], 'exit_inr': exit_price, 'pnl_inr': pnl_inr,
                    'pnl_pct': (pnl_inr / pos['cost_inr']) * 100.0, 'reason': reason
                })
                del open_positions[sym]

        # 2. Check for New Entries across symbols
        if len(open_positions) >= max_open_trades:
            continue
        if daily_trades_count[c_date] >= max_daily_trades:
            continue

        for sym, d in loaded_data.items():
            if sym in open_positions:
                continue
            if len(open_positions) >= max_open_trades:
                break
            if daily_trades_count[c_date] >= max_daily_trades:
                break

            ltf_df = d['ltf_df']
            if t_now not in ltf_df.index:
                continue
            i = ltf_df.index.get_loc(t_now)
            if i < 20 or i < d['start_eval']:
                continue

            c_close = ltf_df['close'].iloc[i]
            c_open = ltf_df['open'].iloc[i]
            c_high = ltf_df['high'].iloc[i]
            c_low = ltf_df['low'].iloc[i]
            c_atr = d['ltf_atr'].iloc[i]
            c_adx = d['ltf_adx'].iloc[i]
            c_rsi = d['ltf_rsi'].iloc[i]
            c_vwap = d['ltf_vwap'].iloc[i]

            # Signal conditions
            htf_df = d['htf_df']
            curr_htf = htf_df.loc[htf_df.index <= t_now]
            if len(curr_htf) < 50:
                continue
            htf_last_idx = curr_htf.index[-1]
            h_ema50 = d['htf_ema50'].loc[htf_last_idx]
            h_ema200 = d['htf_ema200'].loc[htf_last_idx]
            htf_bull = h_ema50 > h_ema200

            bull_engulf = (c_close > c_open) and (c_close > ltf_df['high'].iloc[i-1]) and (c_close - c_open) > 0.6 * (c_high - c_low)
            recent_obs = [ob for ob in d['obs'].iloc[max(0, i-20):i] if ob is not None and isinstance(ob, dict)]
            recent_fvgs = [fvg for fvg in d['fvgs'].iloc[max(0, i-20):i] if fvg is not None and isinstance(fvg, dict)]
            bull_zone = any(ob.get('type') == 'BULLISH' for ob in recent_obs) or any(fvg.get('type') == 'BULLISH' for fvg in recent_fvgs)
            vol_ok = c_atr > (d['ltf_atr'].iloc[i-20:i].mean() * 0.75) if i >= 20 else True
            trend_ok = c_adx > 25.0

            if htf_bull and bull_zone and bull_engulf and c_close > c_vwap and 45 <= c_rsi <= 72 and vol_ok and trend_ok:
                # LONG setup confirmed
                entry_usdt = c_close
                sl_usdt = min(entry_usdt - (1.2 * c_atr), ltf_df['low'].iloc[i-1] * 0.998)
                risk_d_usdt = entry_usdt - sl_usdt
                if risk_d_usdt <= 0 or (risk_d_usdt / entry_usdt) > 0.035 or (risk_d_usdt / entry_usdt) < 0.003:
                    continue

                # INR Conversion
                entry_inr = entry_usdt * usdt_inr_rate
                sl_inr = sl_usdt * usdt_inr_rate
                risk_d_inr = entry_inr - sl_inr

                # Sizing in INR
                target_risk_inr = wallet_balance * risk_pct # 1% = ₹20 on ₹2000
                qty = target_risk_inr / risk_d_inr
                cost_inr = qty * entry_inr

                # Apply max trade allocation (35% of wallet = ₹700 on ₹2000)
                max_cost_inr = min(cash, wallet_balance * max_alloc_pct)
                if cost_inr > max_cost_inr:
                    cost_inr = max_cost_inr
                    qty = cost_inr / entry_inr

                # CoinDCX Spot min notional is ₹100
                if cost_inr < 100.0 or cash < cost_inr:
                    continue

                entry_fee = cost_inr * fee_rate
                cash -= (cost_inr + entry_fee)

                # Targets
                tp_inr = entry_inr + (2.0 * risk_d_inr) # 2.0R target
                be_trigger = entry_inr + (1.2 * risk_d_inr)

                open_positions[sym] = {
                    'entry_inr': entry_inr,
                    'sl_inr': sl_inr,
                    'tp_inr': tp_inr,
                    'be_trigger': be_trigger,
                    'be_locked': False,
                    'qty': qty,
                    'cost_inr': cost_inr,
                    'entry_time': str(t_now),
                }
                daily_trades_count[c_date] += 1

    # End of 1 week: Mark-to-Market any remaining open positions
    for sym, pos in open_positions.items():
        d = loaded_data[sym]
        last_row = d['ltf_df'].iloc[-1]
        exit_price = last_row['close'] * usdt_inr_rate
        revenue = pos['qty'] * exit_price
        fee = revenue * fee_rate
        net_proceeds = revenue - fee
        pnl_inr = net_proceeds - pos['cost_inr']
        wallet_balance += pnl_inr
        closed_trades.append({
            'symbol': sym, 'side': 'BUY', 'entry_time': pos['entry_time'], 'exit_time': 'MARK_TO_MARKET',
            'entry_inr': pos['entry_inr'], 'exit_inr': exit_price, 'pnl_inr': pnl_inr,
            'pnl_pct': (pnl_inr / pos['cost_inr']) * 100.0, 'reason': 'MARK_TO_MARKET'
        })

    # Summary Statistics
    wins = [t for t in closed_trades if t['pnl_inr'] > 0]
    losses = [t for t in closed_trades if t['pnl_inr'] < 0]
    bes = [t for t in closed_trades if abs(t['pnl_inr']) <= 1.0]

    tot_trades = len(closed_trades)
    win_rate = (len(wins) / tot_trades * 100.0) if tot_trades > 0 else 0.0
    total_net_pnl_inr = wallet_balance - starting_balance
    total_return_pct = (total_net_pnl_inr / starting_balance) * 100.0
    tot_win_val = sum(t['pnl_inr'] for t in wins)
    tot_loss_val = abs(sum(t['pnl_inr'] for t in losses))
    profit_factor = (tot_win_val / tot_loss_val) if tot_loss_val > 0 else float('inf')

    print("\n" + "=" * 80)
    print("                      📊 1-WEEK INR BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Initial Starting Capital   : ₹{starting_balance:,.2f} INR")
    print(f"Ending Wallet Balance       : ₹{wallet_balance:,.2f} INR")
    print(f"Net Profit / Loss (PnL)     : ₹{total_net_pnl_inr:+,.2f} INR ({total_return_pct:+.2f}%)")
    print(f"Total Trades Executed       : {tot_trades}")
    print(f"Winning Trades              : {len(wins)} ({win_rate:.1f}%)")
    print(f"Losing Trades               : {len(losses)}")
    print(f"Breakeven / Neutral Trades  : {len(bes)}")
    print(f"Gross Profit                : ₹{tot_win_val:,.2f} INR")
    print(f"Gross Loss                  : ₹{tot_loss_val:,.2f} INR")
    print(f"Profit Factor               : {profit_factor:.2f}")
    print("=" * 80)

    if closed_trades:
        print("\nRecent Executed Trades Log (Sample):")
        print(f"{'Symbol':<10} {'Entry Time':<20} {'Entry (₹)':<12} {'Exit (₹)':<12} {'PnL (₹)':<10} {'Reason'}")
        print("-" * 75)
        for t in closed_trades[-8:]:
            print(f"{t['symbol']:<10} {t['entry_time'][:16]:<20} {t['entry_inr']:<12.2f} {t['exit_inr']:<12.2f} {t['pnl_inr']:+<10.2f} {t['reason']}")
    print("=" * 80)

if __name__ == '__main__':
    run_inr_1week_backtest()

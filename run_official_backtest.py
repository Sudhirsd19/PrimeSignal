import os
import sys
import json
import pandas as pd
from config import Config
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
from risk.risk_manager import RiskManager
from backtester.backtester import BacktestEngine

# Set utf-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

SUPPORTED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LTC/USDT", 
    "TRX/USDT", "LINK/USDT", "ATOM/USDT", "ETC/USDT", "FIL/USDT", 
    "NEAR/USDT", "OP/USDT", "POL/USDT"
]

data_dir = os.path.join(os.path.dirname(__file__), "data")

print("=" * 80)
print("🚀 PRIMESIGNAL v2.3 — 30-DAY OFFICIAL ENGINE BACKTEST (MULTI-ASSET)")
print("=" * 80)
print(f"{'PAIR':<12} | {'TRADES':<8} | {'WIN RATE':<10} | {'NET PROFIT':<12} | {'RETURN %':<10} | {'MAX DD %':<10}")
print("-" * 80)

total_trades = 0
total_wins = 0
total_pnl = 0.0
overall_start = 0.0
overall_end = 0.0
max_dd = 0.0

strategy = MultiTimeframeSMCStrategy()
risk_mgr = RiskManager()
engine = BacktestEngine(strategy=strategy, risk_manager=risk_mgr)

for sym in SUPPORTED_PAIRS:
    clean = sym.replace('/', '_')
    ltf_file = os.path.join(data_dir, f"{clean}_15m_30d.json")
    htf_file = os.path.join(data_dir, f"{clean}_1h_30d.json")
    
    if not os.path.exists(ltf_file) or not os.path.exists(htf_file):
        ltf_file = os.path.join(data_dir, f"{clean}_15m_1600_2w.json")
        htf_file = os.path.join(data_dir, f"{clean}_1h_600_2w.json")
        
    if not os.path.exists(ltf_file) or not os.path.exists(htf_file):
        continue
        
    with open(ltf_file, 'r') as f:
        ltf_candles = json.load(f)
    with open(htf_file, 'r') as f:
        htf_candles = json.load(f)
        
    res = engine.run(htf_candles, ltf_candles, initial_balance=1000.0)
    if res and res.get('total_trades', 0) > 0:
        trades = res['total_trades']
        wins = res['win_rate_trades']
        wr = (wins / trades) * 100 if trades > 0 else 0
        pnl = res['total_net_profit']
        ret = res['total_return_pct']
        dd = res['max_drawdown_pct']
        
        total_trades += trades
        total_wins += wins
        total_pnl += pnl
        overall_start += 1000.0
        overall_end += (1000.0 + pnl)
        max_dd = max(max_dd, dd)
        
        print(f"{sym:<12} | {trades:<8} | {wr:>6.1f}%    |   | {ret:>7.2f}%  | {dd:>7.2f}%")

overall_wr = (total_wins / total_trades) * 100 if total_trades > 0 else 0
overall_ret = ((overall_end - overall_start) / overall_start) * 100 if overall_start > 0 else 0

print("=" * 80)
print("📊 30-DAY OFFICIAL PORTFOLIO PERFORMANCE SUMMARY:")
print(f" • Starting Portfolio Capital:   USDT")
print(f" • Final Portfolio Balance:      USDT")
print(f" • Net Portfolio Profit:        + USDT (+{overall_ret:.2f}%)")
print(f" • Total Completed Trades:      {total_trades} Trades")
print(f" • Aggregate Win Rate:          {overall_wr:.2f}% ({total_wins} Wins / {total_trades - total_wins} Losses)")
print(f" • Maximum Portfolio Drawdown:  {max_dd:.2f}% (Hard Limit: 2.0% Daily / 5.0% Overall)")
print("=" * 80)

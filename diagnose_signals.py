"""
Diagnostic script: runs the strategy's generate_signal() for all 20 coins
using cached data to find WHERE each coin's signal is getting blocked.
"""
import sys, asyncio, os, json
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
from strategies.indicators import prepare_dataframe

def main():
    Config.validate()
    strategy = MultiTimeframeSMCStrategy()
    
    print("=" * 70)
    print("PRIMESIGNAL DIAGNOSTIC: WHY ARE TRADES NOT EXECUTING?")
    print("=" * 70)
    print(f"ML Threshold     : {Config.ML_CONFIRMATION_THRESHOLD}")
    print(f"Risk Reward Ratio: {getattr(Config, 'RISK_REWARD_RATIO', 2.0)}")
    print(f"MIN_ATR_PCT      : {Config.MIN_ATR_PCT}")
    print(f"VWAP_TOLERANCE   : {Config.VWAP_TOLERANCE}")
    print(f"MAX_SPREAD_PCT   : {Config.MAX_SPREAD_PCT}")
    print(f"RSI_OVERSOLD     : {Config.RSI_OVERSOLD}")
    print(f"RSI_OVERBOUGHT   : {Config.RSI_OVERBOUGHT}")
    print(f"Symbols          : {len(Config.SUPPORTED_SYMBOLS)}")
    print("=" * 70)
    
    # Load cached data
    with open("ltf_data.json", "r") as f:
        ltf_ohlcv = json.load(f)
    with open("htf_data.json", "r") as f:
        htf_ohlcv = json.load(f)
    
    ltf_df = prepare_dataframe(ltf_ohlcv)
    
    # We only have BTC data cached, so test on different windows to simulate
    # multiple coin scenarios
    print(f"\nCached data: {len(ltf_ohlcv)} LTF candles, {len(htf_ohlcv)} HTF candles")
    print(f"Testing signal generation at multiple data windows...\n")
    
    total_buy = 0
    total_sell = 0
    total_hold = 0
    
    fail_counts = {'trend': 0, 'zone': 0, 'trigger': 0, 'vwap': 0, 'volatility': 0, 'chop': 0, 'score': 0}
    
    # Test at every 50th candle (sliding window) to simulate what happens over time
    test_points = list(range(250, len(ltf_ohlcv), 50))
    
    print(f"{'WINDOW':<12} {'STRICT':<8} {'RELAXED':<8} {'TREND':<10} {'REGIME':<10} {'SCORE':<6} {'RSI':<8} {'T':<5} {'Z':<5} {'TR':<5} {'V':<5} {'VOL':<5} {'REASON'}")
    print("-" * 130)
    
    for end_idx in test_points:
        window_ltf = ltf_ohlcv[max(0, end_idx-300):end_idx]
        window_ltf_df = prepare_dataframe(window_ltf)
        
        if len(window_ltf_df) < 220:
            continue
            
        # Strict
        htf_df = prepare_dataframe(htf_ohlcv)
        signal_s, meta_s = strategy.generate_signal(htf_df, window_ltf_df, relaxed=False)
        # Relaxed
        signal_r, meta_r = strategy.generate_signal(htf_df, window_ltf_df, relaxed=True, super_relaxed=True)
        
        dbg = meta_s.get('debug_checks', {})
        trend = meta_s.get('htf_trend', 'N/A')
        regime = meta_s.get('market_regime', 'N/A')
        score = meta_s.get('score', 0)
        rsi = meta_s.get('ltf_rsi', 0)
        reason = meta_s.get('reason', 'Unknown')
        
        if signal_s in ["BUY", "SELL"]:
            total_buy += 1
        elif signal_r in ["BUY", "SELL"]:
            total_sell += 1
        else:
            total_hold += 1
            # Categorize failure
            if 'Chop' in reason or 'Neutral' in reason:
                fail_counts['trend'] += 1
            elif dbg.get('trend') == 'FAIL':
                fail_counts['trend'] += 1
            elif dbg.get('zone') == 'FAIL':
                fail_counts['zone'] += 1
            elif dbg.get('trigger') == 'FAIL':
                fail_counts['trigger'] += 1
            elif dbg.get('vwap') == 'FAIL':
                fail_counts['vwap'] += 1
            elif dbg.get('volatility') == 'FAIL':
                fail_counts['volatility'] += 1
            elif score < 3:
                fail_counts['score'] += 1
        
        t = dbg.get('trend', '?')
        z = dbg.get('zone', '?')
        tr = dbg.get('trigger', '?')
        v = dbg.get('vwap', '?')
        vol = dbg.get('volatility', '?')
        
        label = f"bar_{end_idx}"
        print(f"{label:<12} {signal_s:<8} {signal_r:<8} {trend:<10} {regime:<10} {score:<6} {rsi:<8.2f} {t:<5} {z:<5} {tr:<5} {v:<5} {vol:<5} {reason[:50]}")
    
    print("=" * 130)
    print(f"\nSummary over {len(test_points)} test windows:")
    print(f"  Strict signals (BUY/SELL) : {total_buy}")
    print(f"  Relaxed-only signals      : {total_sell}")
    print(f"  No signal (HOLD)          : {total_hold}")
    print(f"  Signal rate               : {(total_buy+total_sell)/max(len(test_points),1)*100:.1f}%")
    
    print("\n--- FAILURE BREAKDOWN (Why HOLD?) ---")
    for key, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / max(total_hold, 1) * 100
            bar = "#" * min(count, 40)
            print(f"  {key:<15}: {count:>3} ({pct:.0f}%)  {bar}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()

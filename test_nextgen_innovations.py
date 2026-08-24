import asyncio
import sys
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from config import Config
from core.liquidation_engine import LiquidationEngine
from core.orderflow_engine import OrderFlowEngine
from core.lead_lag_arbitrage import LeadLagArbitrageEngine
from ml.adversarial_debate import AdversarialDebateCourtroom

def test_liquidation_engine():
    print("[TEST 1] Testing Liquidation Magnetic Heatmap Engine...")
    engine = LiquidationEngine()
    dates = pd.date_range('2026-01-01', periods=50, freq='15min')
    prices = [100.0 + np.sin(i / 3.0) * 5.0 for i in range(50)]
    df = pd.DataFrame({'open': prices, 'high': [p + 1.0 for p in prices], 'low': [p - 1.0 for p in prices], 'close': prices, 'volume': [1000]*50}, index=dates)
    
    liq_res = engine.calculate_liquidation_pools(df)
    assert 'nearest_short_liq' in liq_res, "Missing nearest_short_liq"
    assert 'nearest_long_liq' in liq_res, "Missing nearest_long_liq"
    assert liq_res['nearest_short_liq'] > df['close'].iloc[-1], "Short liq should be above close"
    assert liq_res['nearest_long_liq'] < df['close'].iloc[-1], "Long liq should be below close"
    print(f"  --> Liquidation Pools: Short Liq: {liq_res['nearest_short_liq']}, Long Liq: {liq_res['nearest_long_liq']}: ALL PASS")

def test_orderflow_engine():
    print("[TEST 2] Testing Order Flow Footprint & CVD Absorption Engine...")
    engine = OrderFlowEngine()
    dates = pd.date_range('2026-01-01', periods=30, freq='15min')
    # Price dropping, but close near top of bar (passive buyer absorption)
    closes = [100.0 - (i * 0.2) for i in range(30)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.1 for c in closes]
    opens = [c - 0.05 for c in closes]
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': [1500]*30}, index=dates)
    
    cvd_res = engine.detect_absorption_divergence(df)
    assert 'absorption' in cvd_res, "Missing absorption"
    assert 'cvd_trend' in cvd_res, "Missing cvd_trend"
    print(f"  --> CVD Absorption detected: {cvd_res['absorption']}, Trend: {cvd_res['cvd_trend']}: ALL PASS")

def test_lead_lag_arbitrage():
    print("[TEST 3] Testing Cross-Asset Lead-Lag Latency Arbitrage Engine...")
    engine = LeadLagArbitrageEngine()
    
    # Simulate BTC price history with sudden +0.50% impulse
    for i in range(20):
        engine.record_tick("BTC/USDT", 80000.0)
    for i in range(10):
        engine.record_tick("BTC/USDT", 80450.0) # +0.56% surge

    # Simulate SOL lagging (0% move)
    for i in range(30):
        engine.record_tick("SOL/USDT", 180.0)

    opp = engine.evaluate_lead_lag("SOL/USDT")
    assert opp['signal'] == 'BULLISH_LEAD_LAG_ARBITRAGE', f"Expected BULLISH_LEAD_LAG_ARBITRAGE, got {opp['signal']}"
    assert opp['edge_pct'] > 0, "Edge should be positive"
    print(f"  --> Lead-Lag Signal: {opp['signal']} (Edge: +{opp['edge_pct']}%): ALL PASS")

def test_adversarial_courtroom():
    print("[TEST 4] Testing Dual-Brain Adversarial AI Debate Courtroom...")
    courtroom = AdversarialDebateCourtroom()
    
    # Case 1: High conviction A+ setup
    metadata_pass = {
        'htf_trend': 'BULLISH',
        'setup_type': 'OB',
        'ltf_rsi': 48.0
    }
    context_pass = {
        'cvd': {'absorption': 'BULLISH_ABSORPTION'},
        'liquidation': {'hunt_signal': 'BULLISH_LIQUIDATION_HUNT'},
        'ml_confidence': 0.72,
        'funding_rate': 0.0001,
        'bb_squeeze': False,
        'spread_pct': 0.0004
    }
    res_pass = courtroom.conduct_debate("BUY", metadata_pass, context_pass)
    assert res_pass['verdict'] == 'APPROVED', f"Expected APPROVED, got {res_pass['verdict']}"
    print(f"  --> High Conviction Setup Verdict: {res_pass['verdict']} ({res_pass['conviction_pct']}% conviction): ALL PASS")

    # Case 2: Trap trade with severe objections (overbought RSI, extreme funding rate, squeeze trap)
    metadata_trap = {
        'htf_trend': 'BULLISH',
        'setup_type': 'OB',
        'ltf_rsi': 74.0 # Overbought
    }
    context_trap = {
        'cvd': {'absorption': 'NEUTRAL'},
        'liquidation': {'hunt_signal': 'NONE'},
        'ml_confidence': 0.52,
        'funding_rate': 0.00045, # Extreme positive funding rate
        'bb_squeeze': True, # Squeeze trap
        'spread_pct': 0.0015
    }
    res_trap = courtroom.conduct_debate("BUY", metadata_trap, context_trap)
    assert res_trap['verdict'] == 'REJECTED_BY_AI_COURTROOM', f"Expected REJECTED, got {res_trap['verdict']}"
    print(f"  --> Trap Setup Verdict: {res_trap['verdict']} (Prosecutor blocked trade): ALL PASS")

def run_all():
    print("=" * 65)
    print("  RUNNING NEXT-GEN PROPRIETARY QUANT INNOVATIONS TEST SUITE")
    print("=" * 65)
    test_liquidation_engine()
    test_orderflow_engine()
    test_lead_lag_arbitrage()
    test_adversarial_courtroom()
    print("=" * 65)
    print("  🎉 ALL 4 NEXT-GEN PROPRIETARY ENGINES PASSED 100%!")
    print("=" * 65)

if __name__ == "__main__":
    run_all()

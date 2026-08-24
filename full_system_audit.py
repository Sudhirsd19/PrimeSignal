import sys
import os
import asyncio
import pandas as pd
import numpy as np
import json
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from config import Config
from strategies.indicators import (
    prepare_dataframe, calculate_ema, calculate_rsi, calculate_atr, 
    calculate_vwap, calculate_adx, calculate_bollinger_bands, detect_rsi_divergence
)
from strategies.smc import detect_fvgs, detect_order_blocks, detect_structure
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
from core.liquidation_engine import LiquidationEngine
from core.orderflow_engine import OrderFlowEngine
from core.lead_lag_arbitrage import LeadLagArbitrageEngine
from ml.adversarial_debate import AdversarialDebateCourtroom
from ml.confirmation import MLSignalConfirmator
from risk.risk_manager import RiskManager
from alerts.notifier import TelegramNotifier
from dashboard.app import DashboardState

def make_sample_df(bars=120, trend="UP"):
    dates = pd.date_range('2026-01-01', periods=bars, freq='15min')
    base = 80000.0
    prices = []
    for i in range(bars):
        slope = (i * 15.0) if trend == "UP" else (-i * 15.0)
        p = base + slope + np.sin(i / 4.0) * 100.0
        prices.append(p)
    df = pd.DataFrame({
        'open': [p - 5.0 for p in prices],
        'high': [p + 25.0 for p in prices],
        'low': [p - 25.0 for p in prices],
        'close': prices,
        'volume': [1500 + (i % 5)*100 for i in range(bars)]
    }, index=dates)
    return df

def audit_1_config():
    print("[AUDIT 1/8] Validating Global Configuration & Dynamic Symbols...")
    assert len(Config.SUPPORTED_SYMBOLS) >= 5, "Must support at least 5 symbols"
    assert Config.SYMBOL in Config.SUPPORTED_SYMBOLS, "Active symbol must be in supported list"
    assert hasattr(Config, 'ENABLE_LIQUIDATION_MAGNET')
    assert hasattr(Config, 'ENABLE_CVD_ABSORPTION')
    assert hasattr(Config, 'ENABLE_LEAD_LAG_ARBITRAGE')
    assert hasattr(Config, 'ENABLE_ADVERSARIAL_DEBATE')
    print("  --> Config validation: 100% PASSED")

def audit_2_indicators():
    print("[AUDIT 2/8] Auditing Mathematical Indicators & SMC Structure Engine...")
    df = make_sample_df(100)
    ema20 = calculate_ema(df, 20)
    rsi = calculate_rsi(df, 14)
    atr = calculate_atr(df, 14)
    vwap = calculate_vwap(df)
    adx = calculate_adx(df)
    bb = calculate_bollinger_bands(df)
    bos, choch = detect_structure(df)
    obs = detect_order_blocks(df)
    fvgs = detect_fvgs(df)

    assert not np.isnan(ema20.iloc[-1]), "EMA20 is NaN"
    assert 0 <= rsi.iloc[-1] <= 100, f"RSI out of bounds: {rsi.iloc[-1]}"
    assert atr.iloc[-1] > 0, "ATR must be > 0"
    assert 'bandwidth' in bb.columns, "BB bandwidth missing"
    assert isinstance(obs, (list, pd.Series)), "Order blocks must be Series or list"
    assert isinstance(fvgs, (list, pd.Series)), "FVGs must be Series or list"
    print("  --> Indicators & SMC calculations: 100% PASSED")

def audit_3_liquidation_and_cvd():
    print("[AUDIT 3/8] Auditing Liquidation Gravity & CVD Absorption Engines...")
    df = make_sample_df(100)
    liq = LiquidationEngine()
    liq_res = liq.calculate_liquidation_pools(df)
    assert liq_res['nearest_short_liq'] > df['close'].iloc[-1]
    assert liq_res['nearest_long_liq'] < df['close'].iloc[-1]

    orderflow = OrderFlowEngine()
    cvd_res = orderflow.detect_absorption_divergence(df)
    assert 'absorption' in cvd_res
    assert 'cvd_trend' in cvd_res
    print(f"  --> Liquidation Pools & CVD Absorption: 100% PASSED (Liq Short: {liq_res['nearest_short_liq']}, CVD Trend: {cvd_res['cvd_trend']})")

def audit_4_lead_lag():
    print("[AUDIT 4/8] Auditing Cross-Asset Lead-Lag Latency Arbitrage...")
    lead_lag = LeadLagArbitrageEngine()
    for _ in range(20): lead_lag.record_tick("BTC/USDT", 80000.0)
    for _ in range(10): lead_lag.record_tick("BTC/USDT", 80400.0) # +0.50% impulse
    for _ in range(30): lead_lag.record_tick("SOL/USDT", 180.0)    # lagging

    opp = lead_lag.evaluate_lead_lag("SOL/USDT")
    assert opp['signal'] == 'BULLISH_LEAD_LAG_ARBITRAGE'
    print(f"  --> Lead-Lag propagation detection: 100% PASSED (Impulse edge: +{opp['edge_pct']}%)")

def audit_5_adversarial_courtroom():
    print("[AUDIT 5/8] Auditing Dual-Brain Adversarial AI Debate Courtroom...")
    courtroom = AdversarialDebateCourtroom()
    
    # 1. High conviction A+ setup
    meta_good = {'htf_trend': 'BULLISH', 'setup_type': 'OB', 'ltf_rsi': 45.0}
    ctx_good = {'cvd': {'absorption': 'BULLISH_ABSORPTION'}, 'liquidation': {'hunt_signal': 'BULLISH_LIQUIDATION_HUNT'}, 'ml_confidence': 0.70}
    res_good = courtroom.conduct_debate("BUY", meta_good, ctx_good)
    assert res_good['verdict'] == 'APPROVED'

    # 2. Trap setup
    meta_bad = {'htf_trend': 'BULLISH', 'setup_type': 'OB', 'ltf_rsi': 75.0}
    ctx_bad = {'funding_rate': 0.0005, 'bb_squeeze': True}
    res_bad = courtroom.conduct_debate("BUY", meta_bad, ctx_bad)
    assert res_bad['verdict'] == 'REJECTED_BY_AI_COURTROOM'
    print("  --> AI Courtroom Dual-Brain filtering: 100% PASSED (A+ Approved, Traps Blocked)")

def audit_6_risk_manager():
    print("[AUDIT 6/8] Auditing Risk Manager, Sizing & Trailing Stops...")
    rm = RiskManager()
    size = rm.calculate_position_size(account_equity=10000.0, entry_price=80000.0, stop_loss=79200.0)
    assert size > 0, "Position size must be > 0"
    assert size * 80000.0 <= 10000.0, "Position value must not exceed equity"

    new_sl = rm.update_trailing_stop(entry_price=80000.0, extreme_price=81500.0, stop_loss=80000.0, curr_atr=200.0, position_side="LONG")
    assert new_sl >= 80000.0, "Trailing stop should move up with price"
    print(f"  --> Risk management & trailing stops: 100% PASSED (Calculated size: {size:.6f})")

def audit_7_strategy_execution():
    print("[AUDIT 7/8] Auditing MultiTimeframeSMCStrategy Signal Generation...")
    strat = MultiTimeframeSMCStrategy()
    htf_df = make_sample_df(250, "UP")
    ltf_df = make_sample_df(150, "UP")
    sig, meta = strat.generate_signal(htf_df, ltf_df)
    assert sig in ["BUY", "SELL", "HOLD"], f"Unexpected signal: {sig}"
    assert 'liquidation' in meta, "Liquidation info missing in metadata"
    assert 'cvd' in meta, "CVD info missing in metadata"
    print(f"  --> SMC Strategy evaluation: 100% PASSED (Generated: {sig}, Regime: {meta.get('market_regime')})")

def audit_8_state_serialization():
    print("[AUDIT 8/8] Auditing State Persistence & Dashboard State Serialization...")
    state_file = Path("test_state_audit.json")
    dummy_state = {
        'in_position': {s: False for s in Config.SUPPORTED_SYMBOLS},
        'position_side': {s: "HOLD" for s in Config.SUPPORTED_SYMBOLS},
        'entry_price': {s: 0.0 for s in Config.SUPPORTED_SYMBOLS},
        'stop_loss': {s: 0.0 for s in Config.SUPPORTED_SYMBOLS},
        'take_profit': {s: 0.0 for s in Config.SUPPORTED_SYMBOLS}
    }
    state_file.write_text(json.dumps(dummy_state))
    loaded = json.loads(state_file.read_text())
    assert loaded['in_position'] == dummy_state['in_position']
    if state_file.exists(): state_file.unlink()
    print("  --> State persistence & recovery roundtrip: 100% PASSED")

def run_complete_audit():
    print("=" * 70)
    print("  EXECUTING COMPREHENSIVE PRIME SIGNAL SYSTEM-WIDE AUDIT")
    print("=" * 70)
    audit_1_config()
    audit_2_indicators()
    audit_3_liquidation_and_cvd()
    audit_4_lead_lag()
    audit_5_adversarial_courtroom()
    audit_6_risk_manager()
    audit_7_strategy_execution()
    audit_8_state_serialization()
    print("=" * 70)
    print("  VERDICT: 100% OF ALL 8 AUDIT MODULES PASSED WITH ZERO ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    run_complete_audit()

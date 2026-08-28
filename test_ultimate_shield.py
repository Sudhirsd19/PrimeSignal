import asyncio
import pandas as pd
import numpy as np
from config import Config
from strategies.indicators import calculate_bollinger_bands
from execution.execution_engine import ExecutionEngine
from main import PrimeSignalBot

def test_config():
    print("[TEST 1] Testing Ultimate Shield Configurations...")
    assert hasattr(Config, 'ENABLE_TIME_STOP'), "Missing ENABLE_TIME_STOP"
    assert hasattr(Config, 'MAX_STAGNANT_CANDLES'), "Missing MAX_STAGNANT_CANDLES"
    assert hasattr(Config, 'ENABLE_STRUCTURAL_EXIT'), "Missing ENABLE_STRUCTURAL_EXIT"
    assert hasattr(Config, 'ENABLE_FUNDING_RATE_FILTER'), "Missing ENABLE_FUNDING_RATE_FILTER"
    assert hasattr(Config, 'ENABLE_BB_SQUEEZE_FILTER'), "Missing ENABLE_BB_SQUEEZE_FILTER"
    assert hasattr(Config, 'ENABLE_MACRO_NEWS_FILTER'), "Missing ENABLE_MACRO_NEWS_FILTER"
    print("  --> Config attributes verified: ALL PASS")

def test_bollinger_bands():
    print("[TEST 2] Testing Bollinger Bands & Squeeze calculation...")
    dates = pd.date_range('2026-01-01', periods=50, freq='15min')
    prices = [100.0 + np.sin(i / 5.0) * 0.5 for i in range(50)]
    df = pd.DataFrame({'close': prices, 'open': prices, 'high': prices, 'low': prices, 'volume': [100]*50}, index=dates)
    
    bb = calculate_bollinger_bands(df, period=20, std_dev=2.0)
    assert 'bandwidth' in bb.columns, "Missing bandwidth column"
    assert 'upper' in bb.columns, "Missing upper column"
    assert 'lower' in bb.columns, "Missing lower column"
    assert not np.isnan(bb['bandwidth'].iloc[-1]), "Bandwidth is NaN"
    print(f"  --> BB Bandwidth calculated: {bb['bandwidth'].iloc[-1]:.6f}: ALL PASS")

async def test_funding_rate():
    print("[TEST 3] Testing Funding Rate Fetcher...")
    engine = ExecutionEngine()
    fr = await engine.fetch_funding_rate("BTC/USDT")
    assert isinstance(fr, float), f"Expected float, got {type(fr)}"
    print(f"  --> Real-time Funding Rate for BTC/USDT: {fr*100:+.4f}%: ALL PASS")
    await engine.close()

def test_macro_news_blackout():
    print("[TEST 4] Testing Macro News Blackout checker...")
    bot = PrimeSignalBot()
    is_news, reason = bot.is_macro_news_blackout()
    assert isinstance(is_news, bool)
    assert isinstance(reason, str)
    print(f"  --> Macro news status: active={is_news}, reason='{reason}': ALL PASS")

async def run_all():
    print("=" * 60)
    print("  RUNNING ALL-IN-ONE ULTIMATE SHIELD VERIFICATION SUITE")
    print("=" * 60)
    test_config()
    test_bollinger_bands()
    await test_funding_rate()
    test_macro_news_blackout()
    print("=" * 60)
    print("  ALL ULTIMATE SHIELD CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_all())

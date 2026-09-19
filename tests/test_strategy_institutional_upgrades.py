"""
Institutional Strategy Upgrade Verification Suite
=================================================
Tests:
1. BTC Master Correlation Anchor Engine (flash drops, HTF alignment, score boost).
2. Dynamic Market Regime Classifier (Trend, Chop Compression, Volatility Shock).
3. Institutional Liquidity Sweep Engine (PDH/PDL sweeps, rejection wicks).
"""

import math
import unittest
import pandas as pd
import numpy as np

from core.btc_anchor import BTCAnchorEngine
from strategies.regime_classifier import MarketRegimeClassifier, MarketRegime
from strategies.liquidity_sweeps import LiquiditySweepEngine


class TestStrategyInstitutionalUpgrades(unittest.TestCase):

    def setUp(self):
        # Generate synthetic 15m OHLCV data with strong upward trend
        n = 100
        dates = pd.date_range("2026-09-01", periods=n, freq="15min", tz="UTC")
        prices = np.linspace(50000, 58000, n)
        
        self.df_trend = pd.DataFrame({
            'open': prices - 10,
            'high': prices + 25,
            'low': prices - 15,
            'close': prices + 15,
            'volume': np.full(n, 200.0)
        }, index=dates)

    def test_btc_anchor_btc_self_reference(self):
        """BTC/USDT must never filter itself."""
        engine = BTCAnchorEngine()
        allowed, reason, boost = engine.evaluate_btc_confluence("BTC/USDT", "BUY", self.df_trend)
        self.assertTrue(allowed)
        self.assertIn("Primary Asset", reason)
        self.assertEqual(boost, 0.0)

    def test_btc_anchor_flash_drop_blocks_altcoin_longs(self):
        """Flash drop in BTC (>0.6% in 15m) must immediately block altcoin LONGs."""
        engine = BTCAnchorEngine(flash_drop_threshold=0.006)
        
        # Create a flash drop candle in BTC
        btc_flash = self.df_trend.copy()
        btc_flash.iloc[-1, btc_flash.columns.get_loc('open')] = 50000.0
        btc_flash.iloc[-1, btc_flash.columns.get_loc('close')] = 49500.0  # -1.0% drop!

        allowed, reason, boost = engine.evaluate_btc_confluence("ETH/USDT", "BUY", btc_flash)
        self.assertFalse(allowed)
        self.assertIn("Flash Flush", reason)
        self.assertEqual(boost, 0.0)

    def test_btc_anchor_flash_pump_blocks_altcoin_shorts(self):
        """Flash pump in BTC (>0.6% in 15m) must block altcoin SHORTs."""
        engine = BTCAnchorEngine(flash_pump_threshold=0.006)
        
        btc_pump = self.df_trend.copy()
        btc_pump.iloc[-1, btc_pump.columns.get_loc('open')] = 50000.0
        btc_pump.iloc[-1, btc_pump.columns.get_loc('close')] = 50500.0  # +1.0% surge!

        allowed, reason, boost = engine.evaluate_btc_confluence("SOL/USDT", "SELL", btc_pump)
        self.assertFalse(allowed)
        self.assertIn("Flash Surge", reason)

    def test_btc_anchor_bullish_confluence_boosts_score(self):
        """When BTC is in strong 1h uptrend and RSI >= 50, altcoin longs receive +0.5 boost."""
        engine = BTCAnchorEngine()
        
        # Strong uptrend HTF
        dates_htf = pd.date_range("2026-08-01", periods=120, freq="1h", tz="UTC")
        prices_htf = np.linspace(40000, 60000, 120)
        df_htf = pd.DataFrame({
            'open': prices_htf - 20,
            'high': prices_htf + 50,
            'low': prices_htf - 50,
            'close': prices_htf + 20,
            'volume': np.random.uniform(100, 500, 120)
        }, index=dates_htf)

        allowed, reason, boost = engine.evaluate_btc_confluence("AVAX/USDT", "BUY", self.df_trend, df_htf)
        self.assertTrue(allowed)
        self.assertIn("Bullish Confluence", reason)
        self.assertAlmostEqual(boost, 0.5)

    def test_regime_classifier_trending_vs_chop(self):
        """Regime classifier correctly identifies trend vs chop and adjusts parameters."""
        classifier = MarketRegimeClassifier()

        # Trend data
        res_trend = classifier.classify_regime(self.df_trend)
        self.assertIn(res_trend['regime'], (MarketRegime.TRENDING_BULL.value, MarketRegime.NEUTRAL_RANGE.value))
        self.assertGreaterEqual(res_trend['tp2_mult'], 2.0)

        # Chop data (flat price oscillation)
        n = 100
        dates = pd.date_range("2026-09-01", periods=n, freq="15min", tz="UTC")
        chop_prices = 50000 + 10 * np.sin(np.linspace(0, 20 * np.pi, n))
        df_chop = pd.DataFrame({
            'open': chop_prices - 2,
            'high': chop_prices + 5,
            'low': chop_prices - 5,
            'close': chop_prices + 1,
            'volume': np.random.uniform(50, 100, n)
        }, index=dates)

        res_chop = classifier.classify_regime(df_chop)
        self.assertEqual(res_chop['regime'], MarketRegime.CHOP_COMPRESSION.value)
        self.assertFalse(res_chop['allow_breakout'])
        self.assertTrue(res_chop['allow_mean_reversion'])
        self.assertLessEqual(res_chop['tp1_mult'], 1.3)
        self.assertLessEqual(res_chop['risk_mult'], 0.8)

    def test_liquidity_sweep_bullish_sweep(self):
        """Sweep below PDL with large rejection wick generates a high-probability BUY setup."""
        sweep_engine = LiquiditySweepEngine(min_wick_ratio=0.30)
        
        # 3 days of 15m data (288 bars)
        n = 288
        dates = pd.date_range("2026-09-01", periods=n, freq="15min", tz="UTC")
        prices = np.full(n, 50000.0)
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 100,
            'low': prices - 100,
            'close': prices,
            'volume': 100
        }, index=dates)

        # Force yesterday's (Day 2) low to be 49800 (index 150 is Day 2)
        pdl_target = 49800.0
        df.iloc[150, df.columns.get_loc('low')] = pdl_target

        # Make last closed candle (index -2) pierce below 49800 to 49600, but close at 49850 (huge lower wick!)
        idx_eval = -2
        df.iloc[idx_eval, df.columns.get_loc('open')] = 49820.0
        df.iloc[idx_eval, df.columns.get_loc('high')] = 49860.0
        df.iloc[idx_eval, df.columns.get_loc('low')] = 49600.0  # Swept 49800 by $200!
        df.iloc[idx_eval, df.columns.get_loc('close')] = 49850.0 # Closed back above 49800

        setup = sweep_engine.detect_sweep_setup(df)
        self.assertTrue(setup['is_setup'])
        self.assertEqual(setup['signal'], 'BUY')
        self.assertEqual(setup['sweep_level_type'], 'PDL')
        self.assertLess(setup['stop_loss'], 49600.0)
        self.assertGreater(setup['suggested_tp'], 49850.0)


if __name__ == "__main__":
    unittest.main()

import unittest
import asyncio
from unittest.mock import AsyncMock
from pathlib import Path
import numpy as np
import pandas as pd

from config import Config
from core.data_pipeline import RealTimeDataPipeline
from ml.confirmation import MLSignalConfirmator

class TestMLBalancedAndDepth(unittest.IsolatedAsyncioTestCase):
    async def test_candle_caching_and_delta_sync(self):
        """Candle pipeline persists bars to disk and syncs only missing delta on subsequent calls."""
        cache_path = Path("data") / "candles_cache_btc_usdt_15m.json"
        if cache_path.exists():
            cache_path.unlink()

        mock_execution = AsyncMock()
        # Initial 50 bars
        t0 = 1700000000000
        initial_bars = [[t0 + (i * 900000), 80000.0, 80100.0, 79900.0, 80050.0, 10.0] for i in range(50)]
        mock_execution.fetch_ohlcv.return_value = initial_bars

        pipeline = RealTimeDataPipeline(mock_execution)
        # First fetch: empty cache
        result1 = await pipeline._fetch_ohlcv_paged("BTC/USDT", "15m", 50)
        self.assertEqual(len(result1), 50)
        self.assertTrue(cache_path.exists())

        # Delta fetch: return 10 newer bars
        new_bars = [[t0 + ((50 + i) * 900000), 80100.0, 80200.0, 80050.0, 80150.0, 12.0] for i in range(10)]
        mock_execution.fetch_ohlcv.return_value = new_bars

        result2 = await pipeline._fetch_ohlcv_paged("BTC/USDT", "15m", 60)
        self.assertEqual(len(result2), 60)
        self.assertEqual(result2[0][0], initial_bars[0][0])
        self.assertEqual(result2[-1][0], new_bars[-1][0])

        if cache_path.exists():
            cache_path.unlink()

    def test_ml_balanced_sample_weight_on_skewed_data(self):
        """Model trains cleanly with balanced weights and non-crashing TimeSeriesSplit on skewed data."""
        confirmator = MLSignalConfirmator()
        
        # Generate 600 synthetic candles with strong upward drift (simulating extreme bull market)
        np.random.seed(42)
        n = 600
        dates = pd.date_range("2026-01-01", periods=n, freq="15min")
        returns = np.random.normal(0.002, 0.005, n) # Positive drift
        price = 80000.0 * np.cumprod(1 + returns)
        high = price * (1 + np.abs(np.random.normal(0, 0.002, n)))
        low = price * (1 - np.abs(np.random.normal(0, 0.002, n)))
        volume = np.random.uniform(10, 100, n)

        df = pd.DataFrame({
            "open": price,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume
        }, index=dates)

        success = confirmator.train(df)
        self.assertTrue(success)
        self.assertTrue(confirmator.is_trained)
        self.assertIsNotNone(confirmator.cv_score)
        self.assertGreaterEqual(confirmator.cv_score, 0.0)

if __name__ == "__main__":
    unittest.main()

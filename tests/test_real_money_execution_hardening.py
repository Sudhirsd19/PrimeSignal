import unittest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
import ccxt.async_support as ccxt

from config import Config
from execution.execution_engine import ExecutionEngine
from execution.execution_result import ExecutionState

class TestRealMoneyHardening(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = ExecutionEngine(intent_journal_path="__test_journal_hardening__.jsonl")
        self.engine._futures_initialized_symbols.add("BTC/USDT")
        self.engine._futures_initialized_symbols.add(Config.SYMBOL)
        self.engine.coindcx_client = None

    async def tearDown(self):
        await self.engine.close()
        import os
        if os.path.exists("__test_journal_hardening__.jsonl"):
            os.remove("__test_journal_hardening__.jsonl")

    async def test_rate_limit_429_triggers_circuit_breaker(self):
        """Exchange 429/418 error sets rate_limited_until and halts subsequent calls."""
        call_count = 0

        async def mock_failing_api():
            nonlocal call_count
            call_count += 1
            raise ccxt.RateLimitExceeded("binance -1003 WAY_TOO_MANY_REQUESTS 429")

        # Initial call hits rate limit and sets cooldown
        with self.assertRaises(ccxt.RateLimitExceeded):
            await self.engine.execute_with_retry(mock_failing_api, retries=3)

        self.assertEqual(call_count, 1) # Must NOT retry into a 429!
        self.assertGreater(self.engine.rate_limited_until, time.time() + 50.0)

        # Subsequent call before cooldown expires fails fast
        with self.assertRaises(ccxt.RateLimitExceeded):
            await self.engine.execute_with_retry(mock_failing_api, retries=3)
        self.assertEqual(call_count, 1) # Unchanged: blocked by circuit breaker

        # place_order also rejects without touching exchange
        res = await self.engine.place_order(side='buy', order_type='market', amount=0.01)
        self.assertEqual(res.state, ExecutionState.NOT_SUBMITTED)
        self.assertIn("rate limit cooldown active", res.error.lower())

    async def test_order_book_spread_guard_blocks_entry(self):
        """Market orders are rejected when order book bid-ask spread exceeds Config.MAX_BID_ASK_SPREAD_PCT."""
        # Spread: (80500 - 80000)/80000 = 0.625% > 0.35% limit
        mock_wide_ticker = {
            'bid': 80000.0,
            'ask': 80500.0,
            'last': 80250.0
        }

        with patch.object(self.engine, "execute_with_retry", AsyncMock(return_value=mock_wide_ticker)):
            # 1. Entry order must be aborted
            res = await self.engine.place_order(
                side='buy', order_type='market', amount=0.01,
                price=80250.0, is_exit_order=False, order_role="ENTRY"
            )
            self.assertEqual(res.state, ExecutionState.NOT_SUBMITTED)
            self.assertIn("spread too wide", res.error.lower())

    async def test_exit_order_bypasses_spread_guard(self):
        """Emergency/SL exit orders bypass spread guard to ensure protective risk reduction."""
        mock_wide_ticker = {
            'bid': 80000.0,
            'ask': 80500.0,
            'last': 80250.0
        }

        # Mock mutation so exit order proceeds
        fake_order = {'id': 'EX_EXIT_1', 'status': 'closed', 'filled': 0.01, 'price': 80000.0}
        with patch.object(self.engine, "execute_with_retry", AsyncMock(return_value=mock_wide_ticker)), \
             patch.object(self.engine, "_execute_mutation_once", AsyncMock(return_value=fake_order)):
            res = await self.engine.place_order(
                side='sell', order_type='market', amount=0.01,
                price=80000.0, is_exit_order=True, order_role="SL"
            )
            self.assertNotEqual(res.state, ExecutionState.NOT_SUBMITTED)

    def test_dynamic_sl_reanchoring_preserves_risk_reward(self):
        """When fill price slips, SL is shifted by the exact planned distance so R:R is preserved."""
        planned_entry = 80000.0
        planned_sl = 79200.0
        planned_dist = abs(planned_entry - planned_sl)  # 800.0 (1.0%)

        # Case 1: Adverse Long slippage (bought higher at 80100)
        fill_price = 80100.0
        slippage_pct = (fill_price - planned_entry) / planned_entry
        self.assertAlmostEqual(slippage_pct, 0.00125) # +0.125% slippage

        # Re-anchor
        min_sl_dist = fill_price * 0.005
        effective_dist = max(planned_dist, min_sl_dist)
        reanchored_sl = round(fill_price - effective_dist, 4)

        self.assertEqual(reanchored_sl, 79300.0)
        # Verify dollar risk is unchanged
        self.assertEqual(fill_price - reanchored_sl, 800.0)

        # Take Profit 1R & 2R targets
        tp1 = fill_price + (1.0 * (fill_price - reanchored_sl))
        tp2 = fill_price + (2.0 * (fill_price - reanchored_sl))
        self.assertEqual(tp1, 80900.0)
        self.assertEqual(tp2, 81700.0)

if __name__ == "__main__":
    unittest.main()

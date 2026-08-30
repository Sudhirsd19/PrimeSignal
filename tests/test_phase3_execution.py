import unittest
import asyncio
import time
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

from execution.execution_result import ExecutionResult, ExecutionState
from core.order_state_machine import OrderState
from main import PrimeSignalBot
from config import Config

class TestPhase3ExecutionQuantity(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        Config.PAPER_TRADING = False  # Ensure live mode paths are tested
        self.bot = PrimeSignalBot()
        self.bot.has_keys = True
        self.bot.reconciliation.initial_reconciliation_done = True
        self.bot.global_pause_until = 0
        
        # Mocks
        self.bot.execution = AsyncMock()
        self.bot.execution.fetch_balance = AsyncMock(return_value={'total': {'USDT': 10000.0}})
        self.bot.execution.fetch_funding_rate = AsyncMock(return_value=0.0001)
        self.bot.execution.cancel_order_safe = AsyncMock(return_value=True)
        self.bot.execution.emergency_flatten_position = AsyncMock()
        self.bot.execution.verify_order_active = AsyncMock(return_value="ACTIVE")
        self.bot.execution.fetch_ticker_data = AsyncMock(return_value={'bid': 100.0, 'ask': 100.01, 'quoteVolume': 100000000})
        self.bot.execution.fetch_all_tickers = AsyncMock(return_value={})
        
        # Strategy & ML mocks
        self.bot.strategy = MagicMock()
        self.bot.strategy.generate_signal.return_value = (
            "BUY",
            {
                'mode': 'STRICT',
                'setup_type': 'SMC',
                'entry_price': 100.0,
                'stop_loss': 90.0,
                'take_profit': 150.0,
                'tp1': 110.0,
                'tp2': 125.0,
                'score': 4.0,
                'reason': 'SMC Test Signal',
                'zone_id': 'ZONE_1',
                'prob': 0.85
            }
        )
        self.bot.courtroom = MagicMock()
        self.bot.courtroom.conduct_debate = MagicMock(return_value={'verdict': 'APPROVED', 'conviction_pct': 90.0, 'prosecutor_objections': []})

        # Pipeline candles mock
        dummy_candles = [[time.time()*1000 - (i*900000), 100.0, 105.0, 95.0, 100.0, 1000] for i in range(100, 0, -1)]
        self.bot.pipeline.ltf_candles['BTC/USDT'] = dummy_candles
        self.bot.pipeline.htf_candles['BTC/USDT'] = dummy_candles
        self.bot.pipeline.latest_prices['BTC/USDT'] = 100.0

    def create_mock_fill(self, state, req, filled, price=100.0, avg=100.0):
        res = ExecutionResult(
            state=state,
            requested_qty=req,
            client_order_id="MOCK_ID",
            intent_id="MOCK_INTENT",
            venue="BINANCE"
        )
        res.filled_qty = filled
        res.average_fill_price = avg
        return res

    async def test_entry_partial_fill_authoritative_quantity(self):
        # Request calculated size, fill exactly 0.5
        mock_result = self.create_mock_fill(ExecutionState.PARTIALLY_FILLED, 1.0, 0.5)
        self.bot.execution.place_order = AsyncMock(return_value=mock_result)
        self.bot.execution.place_native_stop_loss = AsyncMock(return_value={'id': 'MOCK_SL', 'status': 'open'})
        
        await self.bot._on_candle_close_impl('BTC/USDT')
        
        self.assertTrue(self.bot.in_position['BTC/USDT'])
        # MUST EQUAL 0.5 (authoritative fill), NOT original requested size
        self.assertEqual(self.bot.position_size['BTC/USDT'], 0.5)

    async def test_entry_rejected_no_quantity(self):
        # Request rejected
        mock_result = self.create_mock_fill(ExecutionState.REJECTED, 1.0, 0.0)
        self.bot.execution.place_order = AsyncMock(return_value=mock_result)

        await self.bot._on_candle_close_impl('BTC/USDT')
        
        self.assertFalse(self.bot.in_position.get('BTC/USDT', False))
        self.assertEqual(self.bot.position_size.get('BTC/USDT', 0.0), 0.0)

    async def test_exit_partial_fill_retains_remainder(self):
        self.bot.in_position['BTC/USDT'] = True
        self.bot.position_size['BTC/USDT'] = 1.0
        self.bot.position_side['BTC/USDT'] = 'LONG'
        self.bot.entry_price['BTC/USDT'] = 100.0
        ctx = self.bot.order_state_machine.get_context('BTC/USDT')
        ctx.transition_to(OrderState.PROTECTED)
        ctx.filled_qty = 1.0
        self.bot.pipeline.latest_prices['BTC/USDT'] = 150.0
        
        # Exit partial fill of 0.6 out of 1.0
        mock_result = self.create_mock_fill(ExecutionState.PARTIALLY_FILLED, 1.0, 0.6)
        self.bot.execution.place_order = AsyncMock(return_value=mock_result)
        
        await self.bot.exit_position('BTC/USDT', "TEST_PARTIAL_EXIT")
        
        # Should still be in position because 0.4 is left
        self.assertTrue(self.bot.in_position['BTC/USDT'])
        self.assertAlmostEqual(self.bot.position_size['BTC/USDT'], 0.4)

    async def test_emergency_flatten_timeout_preserves_state(self):
        # Entry order succeeds, but SL fails -> triggers emergency flatten which times out (EXECUTION_UNKNOWN)
        entry_result = self.create_mock_fill(ExecutionState.FILLED, 1.0, 0.5)
        self.bot.execution.place_order = AsyncMock(return_value=entry_result)
        self.bot.execution.place_native_stop_loss = AsyncMock(return_value=None)  # force failure
        flatten_unknown = self.create_mock_fill(ExecutionState.EXECUTION_UNKNOWN, 0.5, 0.0)
        self.bot.execution.emergency_flatten_position = AsyncMock(return_value=flatten_unknown)
        
        ctx = self.bot.order_state_machine.get_context('BTC/USDT')
        
        await self.bot._on_candle_close_impl('BTC/USDT')
        
        # The emergency flatten failed (UNKNOWN). Local state must remain exposed / quarantined in EXIT_UNKNOWN.
        self.assertTrue(self.bot.in_position['BTC/USDT'])
        self.assertEqual(self.bot.position_size['BTC/USDT'], 0.5)
        self.assertEqual(ctx.state, OrderState.EXIT_UNKNOWN)

if __name__ == '__main__':
    unittest.main()

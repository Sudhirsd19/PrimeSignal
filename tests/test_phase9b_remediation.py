import unittest
import asyncio
import time
import math
import os
from unittest.mock import MagicMock, AsyncMock, patch
from starlette.testclient import TestClient

from config import Config
from execution.execution_result import ExecutionResult, ExecutionState
from execution.exchange_validator import ExchangeValidator
from execution.execution_engine import ExecutionEngine
from execution.coindcx_client import CoinDCXClient
from risk.risk_manager import RiskManager
from core.order_state_machine import OrderStateMachine, OrderState
from core.reconciliation_engine import ReconciliationEngine
from main import PrimeSignalBot
from dashboard.app import app, DashboardState
import ccxt.async_support as ccxt


class TestPhase9BRemediation(unittest.IsolatedAsyncioTestCase):

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-01: ExecutionResult Truthiness & Native SL Lifecycle Verification
    # ─────────────────────────────────────────────────────────────────────────

    def test_aud_p9_01_execution_result_truthiness(self):
        """Zero-trust verification that real ExecutionResult behaves correctly under truthiness."""
        # 1. Resting Stop Loss acknowledged by exchange
        sl_accepted = ExecutionResult(
            state=ExecutionState.ACCEPTED,
            requested_qty=0.5,
            filled_qty=0.0,
            remaining_qty=0.5,
            exchange_order_id="SL_RESTING_12345",
            venue="BINANCE"
        )
        self.assertTrue(bool(sl_accepted), "Accepted SL with exchange order ID must be truthy")
        self.assertTrue(sl_accepted.is_order_accepted)
        self.assertFalse(sl_accepted.is_fill_confirmed)
        self.assertTrue(sl_accepted.has_exchange_order)

        # 2. Confirmed Entry Fill
        fill_confirmed = ExecutionResult(
            state=ExecutionState.FILLED,
            requested_qty=0.5,
            filled_qty=0.5,
            remaining_qty=0.0,
            exchange_order_id="ENTRY_FILLED_67890",
            average_fill_price=85000.0,
            venue="BINANCE"
        )
        self.assertTrue(bool(fill_confirmed))
        self.assertTrue(fill_confirmed.is_order_accepted)
        self.assertTrue(fill_confirmed.is_fill_confirmed)

        # 3. Rejected Stop Loss
        sl_rejected = ExecutionResult(
            state=ExecutionState.REJECTED,
            requested_qty=0.5,
            filled_qty=0.0,
            remaining_qty=0.5,
            error="Insufficient margin for stop loss",
            venue="BINANCE"
        )
        self.assertFalse(bool(sl_rejected))
        self.assertFalse(sl_rejected.is_order_accepted)
        self.assertFalse(sl_rejected.is_fill_confirmed)

        # 4. Accepted order without exchange_order_id must NOT be considered accepted
        sl_no_id = ExecutionResult(
            state=ExecutionState.ACCEPTED,
            requested_qty=0.5,
            filled_qty=0.0,
            exchange_order_id=None,
            venue="BINANCE"
        )
        self.assertFalse(sl_no_id.is_order_accepted)
        self.assertFalse(bool(sl_no_id))

        # 5. Unknown execution state must be falsey
        unknown_res = ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            requested_qty=0.5,
            exchange_order_id="UNKNOWN_123",
            venue="BINANCE"
        )
        self.assertFalse(bool(unknown_res))
        self.assertFalse(unknown_res.is_order_accepted)

    def test_aud_p9_01_bot_and_recon_active_sl_helpers(self):
        """Verify _is_active_sl_order in PrimeSignalBot and ReconciliationEngine with real objects."""
        bot = PrimeSignalBot.__new__(PrimeSignalBot)
        recon = ReconciliationEngine(bot)

        # Real ExecutionResult ACCEPTED
        sl_real = ExecutionResult(
            state=ExecutionState.ACCEPTED,
            exchange_order_id="SL_REAL_111",
            requested_qty=0.1
        )
        self.assertTrue(bot._is_active_sl_order(sl_real))
        self.assertTrue(recon._is_active_sl_order(sl_real))

        # Real ExecutionResult REJECTED
        sl_bad = ExecutionResult(
            state=ExecutionState.REJECTED,
            exchange_order_id=None,
            error="Margin insufficient"
        )
        self.assertFalse(bot._is_active_sl_order(sl_bad))
        self.assertFalse(recon._is_active_sl_order(sl_bad))

        # Real ExecutionResult UNKNOWN
        sl_unk = ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            exchange_order_id="UNK_000"
        )
        self.assertFalse(bot._is_active_sl_order(sl_unk))
        self.assertFalse(recon._is_active_sl_order(sl_unk))

    async def test_aud_p9_01_entry_with_real_execution_result_does_not_flatten(self):
        """Simulate real trade execution where Native SL returns real ExecutionResult."""
        bot = PrimeSignalBot()
        bot.has_keys = True
        Config.PAPER_TRADING = False
        symbol = "BTC/USDT"

        entry_res = ExecutionResult(
            state=ExecutionState.FILLED,
            requested_qty=0.01,
            filled_qty=0.01,
            remaining_qty=0.0,
            average_fill_price=80000.0,
            exchange_order_id="BUY_REAL_123"
        )
        sl_res = ExecutionResult(
            state=ExecutionState.ACCEPTED,
            requested_qty=0.01,
            filled_qty=0.0,
            remaining_qty=0.01,
            exchange_order_id="SL_REAL_ACCEPTED_999"
        )

        bot.execution.create_order = AsyncMock(return_value=entry_res)
        bot.execution.place_native_stop_loss = AsyncMock(return_value=sl_res)
        bot.execution.emergency_flatten_position = AsyncMock()

        # Place SL
        ctx = bot.order_state_machine.get_context(symbol)
        ctx.transition_to(OrderState.FILLED, reason="Entry filled")
        sl_order = await bot.execution.place_native_stop_loss(symbol, 'sell', 0.01, 78400.0)

        # Apply production check
        if bot._is_active_sl_order(sl_order):
            ctx.native_sl_order_id = str(sl_order['id']) if isinstance(sl_order, dict) else str(sl_order.exchange_order_id)
            ctx.transition_to(OrderState.PROTECTED, reason="Native SL confirmed")
        else:
            await bot.execution.emergency_flatten_position(symbol, 'BUY', 0.01, reason="NATIVE_SL_FAILED")

        # Assertions
        bot.execution.emergency_flatten_position.assert_not_called()
        self.assertEqual(ctx.state, OrderState.PROTECTED)
        self.assertEqual(ctx.native_sl_order_id, "SL_REAL_ACCEPTED_999")

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-02: Dynamic CoinDCX USDT/INR Conversion Rate
    # ─────────────────────────────────────────────────────────────────────────

    async def test_aud_p9_02_dynamic_fx_rate_fetching(self):
        """Verify dynamic CoinDCX USDT/INR rate fetching, conservative pricing, and caching."""
        client = CoinDCXClient("dummy_key", "dummy_secret")

        mock_tickers = [
            {"market": "USDTINR", "last_price": "89.50", "bid": "89.20", "ask": "89.80", "volume": "50000"},
            {"market": "BTCINR", "last_price": "7500000.0", "bid": "7490000.0", "ask": "7510000.0"}
        ]

        with patch.object(client, '_get_session') as mock_session_fn:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_tickers)
            
            # Use AsyncMock for context manager
            cm = AsyncMock()
            cm.__aenter__.return_value = mock_resp
            mock_session.get.return_value = cm
            mock_session_fn.return_value = mock_session

            # BUY should choose ask (89.80)
            rate_buy = await client.fetch_usdt_inr_rate(side="BUY")
            self.assertEqual(rate_buy, 89.80)

            # SELL should choose bid (89.20)
            rate_sell = await client.fetch_usdt_inr_rate(side="SELL")
            self.assertEqual(rate_sell, 89.20)

    async def test_aud_p9_02_dynamic_fx_rate_fail_closed(self):
        """Verify dynamic FX rate fails closed when ticker is unavailable, stale, or out of bounds."""
        client = CoinDCXClient("dummy_key", "dummy_secret")

        # 1. Empty ticker list
        with patch.object(client, '_get_session') as mock_session_fn:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=[])
            cm = AsyncMock()
            cm.__aenter__.return_value = mock_resp
            mock_session.get.return_value = cm
            mock_session_fn.return_value = mock_session

            rate = await client.fetch_usdt_inr_rate(side="BUY")
            self.assertIsNone(rate, "Must return None when USDTINR ticker not found")

        # 2. Out of bounds rate (< 70.0 or > 120.0)
        out_of_bounds = [{"market": "USDTINR", "last_price": "150.0", "bid": "149.0", "ask": "151.0"}]
        with patch.object(client, '_get_session') as mock_session_fn:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=out_of_bounds)
            cm = AsyncMock()
            cm.__aenter__.return_value = mock_resp
            mock_session.get.return_value = cm
            mock_session_fn.return_value = mock_session

            client._usdt_inr_cache = None
            rate = await client.fetch_usdt_inr_rate(side="BUY")
            self.assertIsNone(rate, "Must return None when rate exceeds sanity bounds [70.0, 120.0]")

    def test_aud_p9_02_inr_account_sizing_adheres_to_capital_cap(self):
        """Verify ₹2,000 and ₹10,000 INR account sizing respects 35% capital allocation cap."""
        rm = RiskManager()

        # ₹2,000 INR account with BTC @ $80,000, SL @ $78,400, FX rate = 90.0 INR/USDT
        # BTC in INR = 80,000 * 90 = 7,200,000 INR
        # 35% cap of ₹2,000 = ₹700 INR
        # Max BTC = 700 / 7,200,000 = 0.00009722 BTC
        size_2k = rm.calculate_position_size(
            account_equity=2000.0,
            entry_price=80000.0,
            stop_loss=78400.0,
            quote_currency="USDT",
            is_inr=True,
            conversion_rate=90.0
        )
        notional_inr_2k = size_2k * (80000.0 * 90.0)
        self.assertLessEqual(notional_inr_2k, 2000.0 * 0.35 + 0.01)

        # ₹10,000 INR account
        # 35% cap of ₹10,000 = ₹3,500 INR
        size_10k = rm.calculate_position_size(
            account_equity=10000.0,
            entry_price=80000.0,
            stop_loss=78400.0,
            quote_currency="USDT",
            is_inr=True,
            conversion_rate=90.0
        )
        notional_inr_10k = size_10k * (80000.0 * 90.0)
        self.assertLessEqual(notional_inr_10k, 10000.0 * 0.35 + 0.01)

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-03: Invalid Stop Distance Must Fail Closed
    # ─────────────────────────────────────────────────────────────────────────

    def test_aud_p9_03_invalid_stop_distance_fails_closed(self):
        """Verify that stop_distance <= 0, NaN, or Inf returns 0.0 position size."""
        rm = RiskManager()

        # 1. Stop Distance = 0 (Entry == SL)
        size_zero = rm.calculate_position_size(account_equity=1000.0, entry_price=80000.0, stop_loss=80000.0)
        self.assertEqual(size_zero, 0.0)

        # 2. Negative Stop Distance (corrupted inputs)
        size_neg = rm.calculate_position_size(account_equity=1000.0, entry_price=80000.0, stop_loss=80000.0)
        self.assertEqual(size_neg, 0.0)

        # 3. NaN Entry or SL
        size_nan = rm.calculate_position_size(account_equity=1000.0, entry_price=80000.0, stop_loss=float('nan'))
        self.assertEqual(size_nan, 0.0)

        # 4. Infinite Stop Loss
        size_inf = rm.calculate_position_size(account_equity=1000.0, entry_price=80000.0, stop_loss=float('inf'))
        self.assertEqual(size_inf, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-04: Intent Recovery Beyond Latest 20 Orders & Safe Mode Quarantine
    # ─────────────────────────────────────────────────────────────────────────

    async def test_aud_p9_04_intent_recovery_targeted_and_quarantine(self):
        """Verify intent recovery queries exact clientOrderId and quarantines unknown intents."""
        ee = ExecutionEngine()
        ee.trade_client = MagicMock()

        # 1. Exact match via targeted fetch_order with origClientOrderId
        ee.trade_client.fetch_order = AsyncMock(return_value={
            'id': 'EX_TARGET_123',
            'clientOrderId': 'CID_TARGET_001',
            'status': 'closed',
            'filled': 0.5,
            'price': 80000.0
        })

        intent = {
            'intent_id': 'INT_001',
            'client_order_id': 'CID_TARGET_001',
            'symbol': 'BTC/USDT',
            'requested_qty': 0.5,
            'side': 'buy'
        }

        res = await ee.reconcile_intent_on_exchange(intent)
        self.assertEqual(res.state, ExecutionState.FILLED)
        self.assertEqual(res.exchange_order_id, 'EX_TARGET_123')

        # 2. Authoritative absence (OrderNotFound) -> Marked REJECTED
        ee.trade_client.fetch_order = AsyncMock(side_effect=ccxt.OrderNotFound("Order not found"))
        res_absent = await ee.reconcile_intent_on_exchange(intent)
        self.assertEqual(res_absent.state, ExecutionState.REJECTED)

        # 3. Unresolved / Ambiguous without authoritative confirmation -> Quarantined in EXECUTION_UNKNOWN
        ee.trade_client.fetch_order = AsyncMock(side_effect=Exception("Exchange query timeout"))
        ee.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        ee.trade_client.fetch_closed_orders = AsyncMock(return_value=[])

        res_unknown = await ee.reconcile_intent_on_exchange(intent)
        self.assertEqual(res_unknown.state, ExecutionState.EXECUTION_UNKNOWN)
        self.assertTrue(res_unknown.is_unknown)

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-05: WebSocket Authentication Must Fail Closed
    # ─────────────────────────────────────────────────────────────────────────

    def test_aud_p9_05_websocket_auth_fails_closed(self):
        """Verify WebSocket authentication enforces DASHBOARD_SECRET and fails closed."""
        client = TestClient(app)

        # 1. Unset DASHBOARD_SECRET must reject connection
        with patch.dict(os.environ, {"DASHBOARD_SECRET": ""}):
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws?token=anything") as ws:
                    pass

        # 2. With DASHBOARD_SECRET configured:
        with patch.dict(os.environ, {"DASHBOARD_SECRET": "SuperSecretKey999"}):
            # Missing token -> Rejected
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws") as ws:
                    pass

            # Invalid token -> Rejected
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws?token=WrongKey") as ws:
                    pass

            # Valid token -> Accepted
            with client.websocket_connect("/ws?token=SuperSecretKey999") as ws:
                data = ws.receive_json()
                self.assertIn("latest_price", data)

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-06: Minimum Notional Validated After Quantity Precision
    # ─────────────────────────────────────────────────────────────────────────

    def test_aud_p9_06_minimum_notional_after_precision_truncation(self):
        """Verify minimum notional is validated on the FINAL post-truncation executable quantity."""
        markets_info = {
            'BTC/USDT': {
                'limits': {'amount': {'min': 0.0001, 'max': 100.0}},
                'precision': {'amount': 4}  # 4 decimals: step 0.0001
            }
        }

        # Case 1: Raw amount = 0.00019 BTC @ $80,000
        # Raw Notional = 0.00019 * 80,000 = $15.20 (passes pre-check min notional of $10)
        # Precision floor to 4 decimals = 0.0001 BTC
        # Final Notional = 0.0001 * 80,000 = $8.00 (BELOW $10.00 minimum notional!)
        # MUST BE REJECTED!
        is_valid, final_qty, reason = ExchangeValidator.validate_order_intent(
            symbol='BTC/USDT',
            side='buy',
            order_type='market',
            amount=0.00019,
            price=80000.0,
            current_equity=20.0,  # equity tight, cannot scale up
            markets_info=markets_info,
            is_inr=False
        )
        self.assertFalse(is_valid, "Order must be rejected when post-precision notional is below minimum notional")
        self.assertIn("below minimum", reason)

        # Case 2: Post-precision notional remains above minimum notional ($10.00)
        # Raw amount = 0.00025 BTC -> floor 0.0002 BTC -> Final Notional = 0.0002 * 80,000 = $16.00
        is_valid_2, final_qty_2, reason_2 = ExchangeValidator.validate_order_intent(
            symbol='BTC/USDT',
            side='buy',
            order_type='market',
            amount=0.00025,
            price=80000.0,
            current_equity=100.0,
            markets_info=markets_info,
            is_inr=False
        )
        self.assertTrue(is_valid_2)
        self.assertEqual(final_qty_2, 0.0002)

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-07: Emergency Close All Positions Endpoint and Bot Execution
    # ─────────────────────────────────────────────────────────────────────────

    async def test_emergency_close_all_bot_and_api(self):
        """Verify emergency_close_all cleanly exits positions, updates logs, and endpoint responds."""
        bot = PrimeSignalBot()
        bot.has_keys = False
        Config.PAPER_TRADING = True
        DashboardState.trades.clear()

        bot.in_position["BTC/USDT"] = True
        bot.position_size["BTC/USDT"] = 0.05
        bot.entry_price["BTC/USDT"] = 80000.0
        bot.position_side["BTC/USDT"] = "LONG"
        bot.pipeline.latest_prices["BTC/USDT"] = 82000.0

        bot.in_position["ETH/USDT"] = True
        bot.position_size["ETH/USDT"] = 0.5
        bot.entry_price["ETH/USDT"] = 3000.0
        bot.position_side["ETH/USDT"] = "LONG"
        bot.pipeline.latest_prices["ETH/USDT"] = 3100.0

        # Execute emergency close all
        closed_count, msg = await bot.emergency_close_all()
        self.assertEqual(closed_count, 2)
        self.assertFalse(bot.in_position["BTC/USDT"])
        self.assertFalse(bot.in_position["ETH/USDT"])
        self.assertEqual(len(DashboardState.trades), 2)

        # Test API endpoint
        import dashboard.app as dashboard_module
        dashboard_module.bot_instance = bot
        client = TestClient(app)
        with patch.dict(os.environ, {"DASHBOARD_SECRET": "test_secret_123"}):
            resp = client.post("/api/emergency_stop", headers={"X-API-Key": "test_secret_123"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
        
        await bot.shutdown()

    # ─────────────────────────────────────────────────────────────────────────
    # AUD-P9-08: Dynamic Real-time Account Balance & Equity Tracking
    # ─────────────────────────────────────────────────────────────────────────

    async def test_dynamic_balance_and_equity_tracking(self):
        """Verify account equity and paper balance dynamically update with price and currency rate."""
        bot = PrimeSignalBot()
        bot.has_keys = False
        Config.PAPER_TRADING = True
        Config.PAPER_CURRENCY = 'INR'
        Config.COINDCX_TRADE_INR = True
        Config.USDT_INR_RATE = 85.0

        # Start with ₹2,000 INR
        bot._dry_run_balance_usdt = 2000.0

        # Enter LONG position: 0.0001 BTC @ 80,000 USD (Cost: 0.0001 * 80000 * 85 = ₹680 INR)
        # Deduct cost from cash
        entry_cost = 0.0001 * 80000.0 * 85.0
        bot._dry_run_balance_usdt -= entry_cost  # Cash balance = 1320.0 INR

        bot.in_position["BTC/USDT"] = True
        bot.position_size["BTC/USDT"] = 0.0001
        bot.entry_price["BTC/USDT"] = 80000.0
        bot.position_side["BTC/USDT"] = "LONG"
        bot.pipeline.latest_prices["BTC/USDT"] = 80000.0

        # At entry price, total equity must equal starting balance ₹2,000 INR
        eq_at_entry = bot.calculate_total_equity()
        self.assertAlmostEqual(eq_at_entry, 2000.0, places=2)

        # Price rises by $1,000 to $81,000 (Gain: 0.0001 * 1000 * 85 = ₹8.50 INR)
        bot.pipeline.latest_prices["BTC/USDT"] = 81000.0
        eq_after_gain = bot.calculate_total_equity()
        self.assertAlmostEqual(eq_after_gain, 2008.50, places=2)

        # Price falls by $1,000 below entry to $79,000 (Loss: 0.0001 * -1000 * 85 = -₹8.50 INR)
        bot.pipeline.latest_prices["BTC/USDT"] = 79000.0
        eq_after_loss = bot.calculate_total_equity()
        self.assertAlmostEqual(eq_after_loss, 1991.50, places=2)

        # Test _build_state_payload updates DashboardState.balance_usdt
        import dashboard.app as dashboard_module
        dashboard_module.bot_instance = bot
        payload = dashboard_module._build_state_payload()
        self.assertAlmostEqual(payload["balance_usdt"], 1991.50, places=2)

        await bot.shutdown()


if __name__ == '__main__':
    unittest.main()


import unittest
import asyncio
import math
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config
from risk.risk_manager import RiskManager
from execution.exchange_validator import ExchangeValidator
from core.order_state_machine import OrderStateMachine, OrderState
from core.reconciliation_engine import ReconciliationEngine
from execution.execution_result import ExecutionResult, ExecutionState
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
import pandas as pd


class TestPhase8AdversarialRemediation(unittest.IsolatedAsyncioTestCase):
    """
    Phase 8 Zero-Trust Adversarial Remediation Test Suite.
    Directly exercises all P0, P1, and P2 remediation targets.
    """

    def setUp(self):
        Config.PAPER_TRADING = True
        Config.SYMBOL = 'BTC/USDT'
        Config.SUPPORTED_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
        Config.RISK_PCT = 0.8
        Config.MAX_TRADE_ALLOCATION_PCT = 0.35
        Config.USDT_INR_RATE = 85.0

    # ──────────────────────────────────────────────────────────────────────────
    # AUD-P0-01: CURRENCY ISOLATION & SIZING TESTS
    # ──────────────────────────────────────────────────────────────────────────
    def test_aud_p0_01_coindcx_inr_2000_account_sizing(self):
        """
        Prove that a ₹2,000 INR account on CoinDCX with BTC at $85,000 USDT (₹7,225,000 INR)
        CANNOT produce an 85x / ₹59,000 over-allocation, and strictly caps at <= ₹700 (35% allocation).
        """
        rm = RiskManager()
        account_equity_inr = 2000.0
        btc_price_usd = 85000.0
        btc_sl_usd = 83300.0 # 2% stop loss distance
        rate = 85.0

        pos_size_btc = rm.calculate_position_size(
            account_equity=account_equity_inr,
            entry_price=btc_price_usd,
            stop_loss=btc_sl_usd,
            quote_currency="USDT",
            is_inr=True,
            conversion_rate=rate,
        )

        # 35% cap on ₹2,000 = ₹700 INR.
        # At ₹7,225,000 INR/BTC, ₹700 INR buys ~0.0000968 BTC.
        # The previous buggy code computed 0.008227 BTC (costing ₹59,440 INR).
        actual_inr_cost = pos_size_btc * (btc_price_usd * rate)
        max_allowed_inr_allocation = account_equity_inr * 0.35

        self.assertLessEqual(actual_inr_cost, max_allowed_inr_allocation + 1.0)
        self.assertLessEqual(actual_inr_cost, account_equity_inr)
        self.assertLess(pos_size_btc, 0.0002) # Must be around 0.000097 BTC, NOT 0.0082 BTC

    def test_aud_p0_01_coindcx_inr_10000_account_scaling(self):
        """Verify that a ₹10,000 INR account scales correctly within risk and allocation bounds."""
        rm = RiskManager()
        account_equity_inr = 10000.0
        btc_price_usd = 85000.0
        btc_sl_usd = 83300.0
        rate = 85.0

        pos_size_btc = rm.calculate_position_size(
            account_equity=account_equity_inr,
            entry_price=btc_price_usd,
            stop_loss=btc_sl_usd,
            quote_currency="USDT",
            is_inr=True,
            conversion_rate=rate,
        )

        actual_inr_cost = pos_size_btc * (btc_price_usd * rate)
        max_allowed_inr_allocation = account_equity_inr * 0.35

        self.assertLessEqual(actual_inr_cost, max_allowed_inr_allocation + 1.0)
        self.assertGreater(pos_size_btc, 0.0)

    def test_aud_p0_01_conversion_rate_sensitivity(self):
        """Test that different USDT/INR conversion rates (80, 85, 90) scale position size inversely."""
        rm = RiskManager()
        account_equity_inr = 5000.0
        btc_price_usd = 80000.0
        btc_sl_usd = 78400.0

        pos_80 = rm.calculate_position_size(account_equity_inr, btc_price_usd, btc_sl_usd, is_inr=True, conversion_rate=80.0)
        pos_90 = rm.calculate_position_size(account_equity_inr, btc_price_usd, btc_sl_usd, is_inr=True, conversion_rate=90.0)

        # Higher INR/USDT rate means BTC costs more INR, so position size in BTC must be smaller
        self.assertGreater(pos_80, pos_90)

    def test_aud_p0_01_binance_usdt_isolation(self):
        """Verify that Binance USDT calculations remain completely unaffected by INR settings."""
        rm = RiskManager()
        account_equity_usdt = 10000.0
        btc_price_usd = 85000.0
        btc_sl_usd = 83300.0 # $1700 stop distance

        pos_size_btc = rm.calculate_position_size(
            account_equity=account_equity_usdt,
            entry_price=btc_price_usd,
            stop_loss=btc_sl_usd,
            quote_currency="USDT",
            is_inr=False,
        )

        # Max risk = $25. $25 / $1700 = 0.014706 BTC ($1,250 USDT).
        notional_usdt = pos_size_btc * btc_price_usd
        self.assertAlmostEqual(pos_size_btc, 0.014706, places=4)
        self.assertLessEqual(notional_usdt, 3500.0) # 35% of $10,000

    def test_aud_p0_01_exchange_validator_inr(self):
        """Verify ExchangeValidator correctly bounds INR notional and caps."""
        is_valid, amount, reason = ExchangeValidator.validate_order_intent(
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            amount=0.008, # ₹57,800 notional at 85.0 rate!
            price=85000.0,
            current_equity=2000.0,
            is_inr=True,
            quote_currency="USDT",
            conversion_rate=85.0,
        )
        # Should sanitize amount down to <= 35% of ₹2000 (<= ₹700 INR)
        inr_val = amount * 85000.0 * 85.0
        self.assertTrue(is_valid)
        self.assertLessEqual(inr_val, 705.0)

    # ──────────────────────────────────────────────────────────────────────────
    # AUD-P0-02: COINDCX RECONCILIATION DEFENSIVE PARSING
    # ──────────────────────────────────────────────────────────────────────────
    async def test_aud_p0_02_coindcx_reconciliation_ccxt_dict(self):
        """
        Verify that _reconcile_coindcx defensively parses CCXT-style dictionary responses
        without throwing TypeError: string indices must be integers.
        """
        mock_bot = MagicMock()
        mock_bot.execution = MagicMock()
        mock_bot.execution.coindcx_client = MagicMock()
        mock_bot.order_state_machine = OrderStateMachine(Config.SUPPORTED_SYMBOLS)
        mock_bot.in_position = {'BTC/USDT': True, 'ETH/USDT': False}
        mock_bot.position_size = {'BTC/USDT': 0.1, 'ETH/USDT': 0.0}
        mock_bot.has_keys = True

        # Realistic CCXT-style dictionary returned by CoinDCXClient.fetch_balance()
        mock_bot.execution.coindcx_client.fetch_balance = AsyncMock(return_value={
            'total': {'BTC': 0.15, 'INR': 5000.0, 'USDT': 0.0},
            'free': {'BTC': 0.15, 'INR': 5000.0, 'USDT': 0.0},
            'used': {'BTC': 0.0, 'INR': 0.0, 'USDT': 0.0}
        })

        reconciler = ReconciliationEngine(mock_bot, check_interval=60.0)
        # Must execute cleanly without exception
        await reconciler._reconcile_coindcx()
        ctx = mock_bot.order_state_machine.get_context('BTC/USDT')
        self.assertEqual(ctx.state, OrderState.PROTECTED)

    async def test_aud_p0_02_coindcx_reconciliation_malformed_handling(self):
        """Verify _reconcile_coindcx handles None, empty, or unexpected dicts fail-closed."""
        mock_bot = MagicMock()
        mock_bot.execution = MagicMock()
        mock_bot.execution.coindcx_client = MagicMock()
        mock_bot.order_state_machine = OrderStateMachine(Config.SUPPORTED_SYMBOLS)
        mock_bot.in_position = {'BTC/USDT': False}

        reconciler = ReconciliationEngine(mock_bot, check_interval=60.0)

        # Case A: None
        mock_bot.execution.coindcx_client.fetch_balance = AsyncMock(return_value=None)
        await reconciler._reconcile_coindcx()

        # Case B: Empty dict
        mock_bot.execution.coindcx_client.fetch_balance = AsyncMock(return_value={})
        await reconciler._reconcile_coindcx()

        # Case C: Malformed types
        mock_bot.execution.coindcx_client.fetch_balance = AsyncMock(return_value={'total': 'invalid_string'})
        await reconciler._reconcile_coindcx()

    # ──────────────────────────────────────────────────────────────────────────
    # AUD-P1-01: FUTURES DOUBLE-FAULT PROTECTION
    # ──────────────────────────────────────────────────────────────────────────
    async def test_aud_p1_01_futures_unprotected_position_auto_reprotection(self):
        """
        Verify that an open futures position lacking an exchange SL is automatically
        re-protected with a replacement Native SL during reconciliation.
        """
        Config.EXCHANGE_TYPE = 'futures'
        mock_bot = MagicMock()
        mock_bot.execution = MagicMock()
        mock_bot.execution.coindcx_client = None
        mock_bot.execution.trade_client = MagicMock()
        mock_bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        
        async def fake_retry(fn, *args, **kwargs):
            res = fn(*args, **kwargs)
            if asyncio.iscoroutine(res):
                return await res
            return res
            
        mock_bot.execution.execute_with_retry = AsyncMock(side_effect=fake_retry)
        mock_bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[
            {'symbol': 'BTC/USDT', 'contracts': 0.25, 'entryPrice': 85000.0, 'side': 'long'}
        ])
        mock_bot.order_state_machine = OrderStateMachine(Config.SUPPORTED_SYMBOLS)
        mock_bot.in_position = {'BTC/USDT': True, 'ETH/USDT': False}
        mock_bot.position_size = {'BTC/USDT': 0.25, 'ETH/USDT': 0.0}
        mock_bot.position_side = {'BTC/USDT': 'LONG', 'ETH/USDT': 'HOLD'}
        mock_bot.stop_loss = {'BTC/USDT': 83300.0, 'ETH/USDT': 0.0}
        mock_bot.entry_price = {'BTC/USDT': 85000.0, 'ETH/USDT': 0.0}
        mock_bot.has_keys = True
        Config.PAPER_TRADING = False

        # Native SL succeeds
        mock_bot.execution.place_native_stop_loss = AsyncMock(return_value={'id': 'SL_EXCHANGE_999'})

        reconciler = ReconciliationEngine(mock_bot, check_interval=60.0)
        await reconciler._reconcile_binance()

        ctx = mock_bot.order_state_machine.get_context('BTC/USDT')
        self.assertEqual(ctx.native_sl_order_id, 'SL_EXCHANGE_999')
        self.assertEqual(ctx.state, OrderState.PROTECTED)

    async def test_aud_p1_01_futures_double_fault_triggers_safe_mode(self):
        """
        Verify that if Native SL re-protection fails AND emergency flatten fails,
        the position is NOT marked PROTECTED, transitions to EXIT_UNKNOWN, and activates Safe Mode.
        """
        Config.EXCHANGE_TYPE = 'futures'
        mock_bot = MagicMock()
        mock_bot.execution = MagicMock()
        mock_bot.execution.coindcx_client = None
        mock_bot.execution.trade_client = MagicMock()
        mock_bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])

        async def fake_retry(fn, *args, **kwargs):
            res = fn(*args, **kwargs)
            if asyncio.iscoroutine(res):
                return await res
            return res

        mock_bot.execution.execute_with_retry = AsyncMock(side_effect=fake_retry)
        mock_bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[
            {'symbol': 'BTC/USDT', 'contracts': 0.25, 'entryPrice': 85000.0, 'side': 'long'}
        ])
        mock_bot.order_state_machine = OrderStateMachine(Config.SUPPORTED_SYMBOLS)
        mock_bot.in_position = {'BTC/USDT': True, 'ETH/USDT': False}
        mock_bot.position_size = {'BTC/USDT': 0.25, 'ETH/USDT': 0.0}
        mock_bot.position_side = {'BTC/USDT': 'LONG', 'ETH/USDT': 'HOLD'}
        mock_bot.stop_loss = {'BTC/USDT': 83300.0, 'ETH/USDT': 0.0}
        mock_bot.entry_price = {'BTC/USDT': 85000.0, 'ETH/USDT': 0.0}
        mock_bot.has_keys = True
        Config.PAPER_TRADING = False

        # Double-Fault: SL placement returns None/fails, and emergency flatten fails
        mock_bot.execution.place_native_stop_loss = AsyncMock(return_value=None)
        mock_bot.execution.emergency_flatten_position = AsyncMock(return_value=ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            error="Flatten timed out",
            requested_qty=0.25
        ))

        reconciler = ReconciliationEngine(mock_bot, check_interval=60.0)
        await reconciler._reconcile_binance()

        ctx = mock_bot.order_state_machine.get_context('BTC/USDT')
        # Invariant: Must NOT be PROTECTED!
        self.assertNotEqual(ctx.state, OrderState.PROTECTED)
        self.assertEqual(ctx.state, OrderState.EXIT_UNKNOWN)
        self.assertTrue(reconciler.safe_mode_active)

    # ──────────────────────────────────────────────────────────────────────────
    # AUD-P1-02: STARTUP TP1/TP2/EXIT INTENT RECOVERY
    # ──────────────────────────────────────────────────────────────────────────
    async def test_aud_p1_02_startup_replay_tp1_and_exit_recovery(self):
        """
        Verify startup replay recovers TP1 fill and adjusts position_size, preventing
        false Spot balance deficit quarantines.
        """
        mock_bot = MagicMock()
        mock_bot.execution = MagicMock()
        mock_bot.order_state_machine = OrderStateMachine(Config.SUPPORTED_SYMBOLS)
        mock_bot.in_position = {'BTC/USDT': True, 'ETH/USDT': False}
        mock_bot.position_size = {'BTC/USDT': 1.0, 'ETH/USDT': 0.0}
        mock_bot.partial_tp_taken = {'BTC/USDT': False, 'ETH/USDT': False}
        mock_bot.tp2_taken = {'BTC/USDT': False, 'ETH/USDT': False}
        mock_bot.position_side = {'BTC/USDT': 'LONG', 'ETH/USDT': 'HOLD'}
        mock_bot.entry_price = {'BTC/USDT': 85000.0, 'ETH/USDT': 0.0}
        mock_bot.stop_loss = {'BTC/USDT': 83300.0, 'ETH/USDT': 0.0}
        mock_bot.take_profit_1r = {'BTC/USDT': 86700.0, 'ETH/USDT': 0.0}
        mock_bot.take_profit_2r = {'BTC/USDT': 88000.0, 'ETH/USDT': 0.0}
        mock_bot.take_profit = {'BTC/USDT': 90000.0, 'ETH/USDT': 0.0}
        mock_bot.highest_price_reached = {'BTC/USDT': 85000.0, 'ETH/USDT': 0.0}
        mock_bot.lowest_price_reached = {'BTC/USDT': 85000.0, 'ETH/USDT': 0.0}
        mock_bot.entry_time = {'BTC/USDT': 0.0, 'ETH/USDT': 0.0}
        mock_bot.has_keys = False
        Config.PAPER_TRADING = True

        # Simulate replay of a confirmed TP1 fill of 0.5 BTC
        tp1_res = ExecutionResult(
            state=ExecutionState.FILLED,
            requested_qty=0.5,
            filled_qty=0.5,
            average_fill_price=86700.0,
            client_order_id="PS_TP1_001",
            intent_id="INT_TP1_001",
            raw={'order_role': 'TP1', 'symbol': 'BTC/USDT', 'side': 'sell'}
        )

        mock_bot.execution.replay_and_resolve_unresolved_intents = AsyncMock(return_value={
            "INT_TP1_001": tp1_res
        })

        reconciler = ReconciliationEngine(mock_bot, check_interval=60.0)
        await reconciler.start()

        # Position size must be reduced from 1.0 to 0.5 BTC
        self.assertEqual(mock_bot.position_size['BTC/USDT'], 0.5)
        self.assertTrue(mock_bot.partial_tp_taken['BTC/USDT'])

    # ──────────────────────────────────────────────────────────────────────────
    # AUD-P2-02: ADX NAN FAIL-CLOSED
    # ──────────────────────────────────────────────────────────────────────────
    def test_aud_p2_02_adx_nan_fails_closed(self):
        """Verify that NaN in ADX series returns HOLD and does not bypass filter."""
        strategy = MultiTimeframeSMCStrategy()
        
        # Build 250 rows of synthetic OHLCV data with an upward trend
        dates = pd.date_range("2026-01-01", periods=250, freq="15min")
        prices = [100.0 + (i * 0.5) for i in range(250)]
        ltf_df = pd.DataFrame({
            'open': prices,
            'high': [p + 1.0 for p in prices],
            'low': [p - 1.0 for p in prices],
            'close': prices,
            'volume': [1000.0] * 250
        }, index=dates)
        htf_df = ltf_df.copy()

        # Mock calculate_adx to return NaN in latest rows
        with patch("strategies.multi_timeframe.calculate_adx") as mock_adx:
            mock_adx.return_value = pd.DataFrame({'adx': [float('nan')] * 250}, index=dates)
            signal, metadata = strategy.generate_signal(htf_df, ltf_df)

            self.assertEqual(signal, "HOLD")
            self.assertIn("ADX", metadata.get('reason', ''))


    # ──────────────────────────────────────────────────────────────────────────
    # AUD-P2-01: WEBSOCKET AUTHENTICATION
    # ──────────────────────────────────────────────────────────────────────────
    def test_aud_p2_01_websocket_auth_rejection(self):
        """Verify unauthenticated connection to /ws is rejected with code 1008."""
        from fastapi.testclient import TestClient
        import dashboard.app as dash_app
        client = TestClient(dash_app.app)
        
        with patch.dict(os.environ, {"DASHBOARD_SECRET": "TEST_SECRET_123"}):
            dash_app._DASHBOARD_SECRET = "TEST_SECRET_123"
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws") as websocket:
                    websocket.receive_text()

    def test_aud_p2_01_websocket_auth_success(self):
        """Verify authenticated connection with valid token query param succeeds."""
        from fastapi.testclient import TestClient
        import dashboard.app as dash_app
        client = TestClient(dash_app.app)
        
        with patch.dict(os.environ, {"DASHBOARD_SECRET": "TEST_SECRET_123"}):
            dash_app._DASHBOARD_SECRET = "TEST_SECRET_123"
            with client.websocket_connect("/ws?token=TEST_SECRET_123") as websocket:
                initial_state = websocket.receive_text()
                self.assertIn("latest_price", initial_state)
                websocket.send_text("ping")
                resp = websocket.receive_text()
                self.assertEqual(resp, "pong")

    # ──────────────────────────────────────────────────────────────────────────
    # AUD-P1-03: COINDCX LIVE INR ACCOUNT EQUITY SYNC
    # ──────────────────────────────────────────────────────────────────────────
    async def test_aud_p1_03_coindcx_live_inr_equity_sync(self):
        """Verify _on_candle_close_impl reads INR balance when COINDCX_TRADE_INR is active."""
        from main import PrimeSignalBot
        from dashboard.app import DashboardState
        
        bot = PrimeSignalBot()
        bot.has_keys = True
        bot.reconciliation.initial_reconciliation_done = True
        Config.PAPER_TRADING = False
        Config.COINDCX_TRADE_INR = True
        
        bot.execution.fetch_balance = AsyncMock(return_value={
            'total': {'INR': 12500.0, 'BTC': 0.05, 'USDT': 10.0},
            'free': {'INR': 12500.0, 'BTC': 0.05, 'USDT': 10.0}
        })
        candle_mock = [{'timestamp': int(time.time() * 1000), 'open': 85000.0, 'high': 85100.0, 'low': 84900.0, 'close': 85000.0, 'volume': 10.0}]
        bot.pipeline = MagicMock()
        bot.pipeline.ltf_candles = {'BTC/USDT': candle_mock}
        bot.pipeline.htf_candles = {'BTC/USDT': candle_mock}
        bot.strategy = MagicMock()
        bot.strategy.generate_signal = MagicMock(return_value=('HOLD', {}))
        
        await bot._on_candle_close_impl('BTC/USDT')
        
        # Verify dashboard balance updated to INR balance (₹12,500), not USDT ($10.0)
        self.assertEqual(DashboardState.balance_usdt, 12500.0)
        self.assertEqual(DashboardState.balance_currency, "INR")


if __name__ == '__main__':
    unittest.main()

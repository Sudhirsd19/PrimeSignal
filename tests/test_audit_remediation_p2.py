import unittest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from config import Config
from execution.execution_result import ExecutionState


class TestSecondAuditRemediations(unittest.TestCase):
    def test_p0_1_futures_leverage_fail_closed(self):
        """P0-1: Test that futures leverage/margin configuration failure rejects orders fail-closed."""
        from execution.execution_engine import ExecutionEngine
        
        engine = ExecutionEngine()
        engine.trade_client = MagicMock()
        engine.trade_client.load_markets = AsyncMock(return_value={})
        # Simulate fatal failure in set_leverage
        engine.trade_client.set_margin_mode = AsyncMock(return_value={})
        engine.trade_client.set_leverage = AsyncMock(side_effect=Exception("API-key lacks futures trading permission (-2015)"))

        with patch.object(Config, 'EXCHANGE_TYPE', 'futures'):
            # 1. _init_futures must raise and reset _futures_initialized
            with self.assertRaises(RuntimeError) as cm:
                asyncio.run(engine._init_futures("BTC/USDT"))
            self.assertIn("Failed to set futures leverage", str(cm.exception))
            self.assertFalse(engine._futures_initialized)

            # 2. place_order must reject order fail-closed
            res = asyncio.run(engine.place_order("BUY", "market", 0.01, symbol="BTC/USDT"))
            self.assertEqual(res.state, ExecutionState.REJECTED)
            self.assertIn("Futures leverage/margin configuration failed", res.error)

    def test_p0_2_fetch_positions_safe_mode(self):
        """P0-2: Test that futures fetch_positions failure triggers SAFE MODE in reconciliation."""
        from core.reconciliation_engine import ReconciliationEngine
        
        mock_bot = MagicMock()
        mock_bot.has_keys = True
        mock_bot.execution = MagicMock()
        mock_bot.execution.trade_client = MagicMock()
        mock_bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        # Return None or throw exception on fetch_positions
        mock_bot.execution.execute_with_retry = AsyncMock(return_value=None)
        
        rec_engine = ReconciliationEngine(mock_bot, check_interval=15.0)
        with patch.object(Config, 'EXCHANGE_TYPE', 'futures'):
            asyncio.run(rec_engine._reconcile_binance())
            self.assertTrue(rec_engine.safe_mode_active)

    def test_p1_1_funding_rate_none_on_error(self):
        """P1-1: Test that fetch_funding_rate returns None on HTTP failure instead of 0.0."""
        from execution.execution_engine import ExecutionEngine
        
        engine = ExecutionEngine()
        # Ensure clean cache
        if hasattr(engine, '_funding_rates_cache'):
            engine._funding_rates_cache.clear()
            engine._funding_rates_cache_time.clear()

        # Mock aiohttp to simulate HTTP 500 error
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 500
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            rate = asyncio.run(engine.fetch_funding_rate("BTC/USDT"))
            self.assertIsNone(rate, "fetch_funding_rate must return None on error to enable fail-closed caller behavior")

    def test_p1_2_explicit_venue_selection(self):
        """P1-2: Test that venue selection is strictly controlled by Config.TRADING_VENUE."""
        from execution.execution_engine import ExecutionEngine

        # Even with CoinDCX keys populated, TRADING_VENUE="BINANCE" must not route to CoinDCX
        with patch.object(Config, 'TRADING_VENUE', 'BINANCE'), \
             patch.object(Config, 'COINDCX_API_KEY', 'valid_mock_key'), \
             patch.object(Config, 'COINDCX_SECRET_KEY', 'valid_mock_secret'), \
             patch.object(Config, 'EXCHANGE_TYPE', 'spot'):
            engine = ExecutionEngine()
            self.assertIsNone(engine.coindcx_client)

        # TRADING_VENUE="COINDCX" in futures mode must fail closed
        with patch.object(Config, 'TRADING_VENUE', 'COINDCX'), \
             patch.object(Config, 'EXCHANGE_TYPE', 'futures'):
            with self.assertRaises(ValueError) as cm:
                ExecutionEngine()
            self.assertIn("CoinDCX venue selected but EXCHANGE_TYPE='futures'", str(cm.exception))

    def test_p1_3_mode_switch_broker_handshake(self):
        """P1-3: Test that /api/set_mode performs authoritative broker check and blocks switch if orders exist."""
        from dashboard.app import set_mode, ModeRequest
        import dashboard.app as app_mod

        mock_bot = MagicMock()
        mock_bot.has_keys = True
        mock_bot.in_position = {s: False for s in Config.SUPPORTED_SYMBOLS}
        mock_bot.position_size = {s: 0.0 for s in Config.SUPPORTED_SYMBOLS}
        mock_bot.reconciliation = MagicMock(safe_mode_active=False)
        mock_bot.order_state_machine = MagicMock()
        mock_bot.order_state_machine.get_context.return_value = MagicMock(state="IDLE")
        mock_bot.execution = MagicMock()
        mock_bot.execution.coindcx_client = None
        mock_bot.execution.trade_client = MagicMock()
        # Authoritative broker check returns an active order on exchange
        mock_bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[{"id": "ORD_123", "symbol": "BTC/USDT"}])

        app_mod.bot_instance = mock_bot
        app_mod._DASHBOARD_SECRET = "test_secret"

        req = ModeRequest(paper_trading=False)
        res = asyncio.run(set_mode(req))
        self.assertEqual(res["status"], "error")
        self.assertIn("Authoritative broker check found 1 active order(s)", res["message"])

    def test_p1_4_config_journal_and_ledger_binding(self):
        """P1-4: Test that risk settings updates are journaled and config_hash is computed deterministically."""
        from core.config_journal import ConfigAuditJournal
        
        temp_dir = tempfile.mkdtemp()
        try:
            journal_path = Path(temp_dir) / "config_journal.jsonl"
            journal = ConfigAuditJournal(str(journal_path))

            snap1 = Config.get_risk_config_snapshot()
            hash1 = Config.get_risk_config_hash()
            self.assertTrue(len(hash1) == 64)

            # Record change
            journal.record_change(event="TEST_UPDATE", source="TEST", old_snapshot=snap1, new_snapshot=snap1)
            latest = journal.get_latest_record()
            self.assertIsNotNone(latest)
            self.assertEqual(latest["config_hash"], hash1)
            self.assertEqual(latest["event"], "TEST_UPDATE")

            # Verify that changing a risk parameter changes the hash
            with patch.object(Config, 'MAX_DAILY_TRADES', 99):
                hash2 = Config.get_risk_config_hash()
                self.assertNotEqual(hash1, hash2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_p2_1_max_drawdown_initial_loss(self):
        """P2-1: Test that early/initial losses are accurately reflected in max drawdown rather than zeroed out."""
        from core.performance_analytics import calculate_advanced_metrics
        
        # Scenario: First trade is a -$500 loss on a $10,000 starting account
        trades = [{"pnl_usdt": -500.0}]
        metrics = calculate_advanced_metrics(trades, initial_capital=10000.0)
        
        # Max drawdown must be $500 (5.0%), NOT 0.0%
        self.assertEqual(metrics["max_drawdown"], 500.0)
        self.assertEqual(metrics["max_drawdown_pct"], 5.0)


if __name__ == "__main__":
    unittest.main()

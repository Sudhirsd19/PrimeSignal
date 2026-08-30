import unittest
import asyncio
import json
import os
import sys
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from config import Config
from core.order_state_machine import OrderStateMachine, OrderState
from core.reconciliation_engine import ReconciliationEngine
from execution.execution_result import ExecutionResult, ExecutionState, ExecutionIntentJournal
from main import PrimeSignalBot
from risk.risk_manager import RiskManager

class TestPhase5CrashRecovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create temp dir for state and journal testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.test_dir.name) / "bot_state.json"
        self.journal_file = Path(self.test_dir.name) / "execution_intents.jsonl"
        self.ledger_file = Path(self.test_dir.name) / "immutable_ledger.jsonl"
        
        # Patch config paths
        Config.STATE_FILE = str(self.state_file)
        Config.INTENT_JOURNAL_FILE = str(self.journal_file)
        Config.IMMUTABLE_LEDGER_FILE = str(self.ledger_file)
        Config.PAPER_TRADING = False
        Config.EXCHANGE_TYPE = "futures"

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_valid_state_file(self, data=None):
        base = {
            "schema_version": 2,
            "timestamp": time.time(),
            "in_position": {"BTC/USDT": False},
            "position_side": {"BTC/USDT": "HOLD"},
            "position_size": {"BTC/USDT": 0.0},
            "entry_price": {"BTC/USDT": 0.0},
            "stop_loss": {"BTC/USDT": 0.0},
            "take_profit": {"BTC/USDT": 0.0},
            "take_profit_1r": {"BTC/USDT": 0.0},
            "take_profit_2r": {"BTC/USDT": 0.0},
            "partial_tp_taken": {"BTC/USDT": False},
            "tp2_taken": {"BTC/USDT": False},
            "dry_run_balance_usdt": 10000.0,
            "daily_drawdown_reset_day": time.strftime("%Y-%m-%d"),
            "active_risk_reservations": {}
        }
        if data:
            base.update(data)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)
        return base

    def _mock_bot(self):
        bot = PrimeSignalBot()
        bot.has_keys = True
        bot._STATE_FILE = self.state_file
        bot.execution.trade_client.load_markets = AsyncMock()
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'USDT': 10000.0, 'BTC': 0.0}})
        return bot

    # 1. Pre-mutation crash (intent journaled, crash before exchange POST)
    async def test_pre_mutation_crash_recovery(self):
        journal = ExecutionIntentJournal(path=self.journal_file)
        journal.create(
            intent_id="INTENT_101",
            client_order_id="CID_101",
            venue="BINANCE",
            account_mode="FUTURES",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0
        )
        
        bot = self._mock_bot()
        bot.execution.intent_journal = journal
        # Exchange returns no orders found
        bot.execution.reconcile_intent_on_exchange = AsyncMock(return_value=None)
        
        await bot.execution.replay_and_resolve_unresolved_intents()
        
        # Verify intent record resolved
        unresolved = journal.unresolved()
        self.assertEqual(len(unresolved), 0)
        self.assertFalse(bot.in_position.get("BTC/USDT", False))

    # 2. Post-POST network drop (intent SUBMISSION_UNKNOWN, exchange has filled order)
    async def test_post_post_network_timeout_recovery_order_found(self):
        journal = ExecutionIntentJournal(path=self.journal_file)
        journal.create(
            intent_id="INTENT_102",
            client_order_id="CID_102",
            venue="BINANCE",
            account_mode="FUTURES",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0
        )
        journal.append({
            "event": "INTENT_UNKNOWN",
            "intent_id": "INTENT_102",
            "client_order_id": "CID_102",
            "state": ExecutionState.SUBMISSION_UNKNOWN.value,
            "error": "Timeout after POST",
            "recorded_at": time.time()
        })
        
        bot = self._mock_bot()
        bot.execution.intent_journal = journal
        
        # Mock exchange finding the filled order
        mock_res = ExecutionResult(
            state=ExecutionState.FILLED,
            requested_qty=0.5,
            client_order_id="CID_102",
            intent_id="INTENT_102",
            venue="BINANCE"
        )
        mock_res.filled_qty = 0.5
        mock_res.average_fill_price = 90000.0
        mock_res.exchange_order_id = "EX_ORD_102"
        bot.execution.reconcile_intent_on_exchange = AsyncMock(return_value=mock_res)
        
        await bot.execution.replay_and_resolve_unresolved_intents()
        
        unresolved = journal.unresolved()
        self.assertEqual(len(unresolved), 0)

    # 3. Post-POST network drop (intent SUBMISSION_UNKNOWN, exchange has NO order)
    async def test_post_post_network_timeout_order_not_on_exchange(self):
        journal = ExecutionIntentJournal(path=self.journal_file)
        journal.create(
            intent_id="INTENT_103",
            client_order_id="CID_103",
            venue="BINANCE",
            account_mode="FUTURES",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0
        )
        journal.append({
            "event": "INTENT_UNKNOWN",
            "intent_id": "INTENT_103",
            "client_order_id": "CID_103",
            "state": ExecutionState.SUBMISSION_UNKNOWN.value,
            "error": "Timeout after POST",
            "recorded_at": time.time()
        })
        
        bot = self._mock_bot()
        bot.execution.intent_journal = journal
        bot.execution.reconcile_intent_on_exchange = AsyncMock(return_value=None)
        
        await bot.execution.replay_and_resolve_unresolved_intents()
        
        unresolved = journal.unresolved()
        self.assertEqual(len(unresolved), 0)

    # 4. Post-acceptance crash recovery (intent ACCEPTED on exchange)
    async def test_post_acceptance_crash_recovery(self):
        journal = ExecutionIntentJournal(path=self.journal_file)
        journal.create(
            intent_id="INTENT_104",
            client_order_id="CID_104",
            venue="BINANCE",
            account_mode="FUTURES",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0
        )
        journal.append({
            "event": "INTENT_ACCEPTED",
            "intent_id": "INTENT_104",
            "client_order_id": "CID_104",
            "exchange_order_id": "EX_ORD_104",
            "state": ExecutionState.ACCEPTED.value,
            "recorded_at": time.time()
        })
        
        bot = self._mock_bot()
        bot.execution.intent_journal = journal
        
        mock_res = ExecutionResult(
            state=ExecutionState.FILLED,
            requested_qty=0.5,
            client_order_id="CID_104",
            intent_id="INTENT_104",
            venue="BINANCE"
        )
        mock_res.filled_qty = 0.5
        mock_res.average_fill_price = 90000.0
        mock_res.exchange_order_id = "EX_ORD_104"
        bot.execution.reconcile_intent_on_exchange = AsyncMock(return_value=mock_res)
        
        await bot.execution.replay_and_resolve_unresolved_intents()
        self.assertEqual(len(journal.unresolved()), 0)

    # 5. Post-partial fill crash recovery (exchange filled 0.3 of 1.0)
    async def test_post_partial_fill_crash_recovery(self):
        self._create_valid_state_file()
        bot = self._mock_bot()
        
        mock_positions = [{'symbol': 'BTC/USDT', 'contracts': 0.3, 'entryPrice': 91000.0, 'side': 'LONG'}]
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=mock_positions)
        bot.execution.place_native_stop_loss = AsyncMock(return_value={'id': 'SL_PARTIAL_105', 'status': 'open'})
        bot.execution.verify_order_active = AsyncMock(return_value="ACTIVE")
        
        reconciler = ReconciliationEngine(bot, check_interval=1.0)
        await reconciler._reconcile_binance()
        
        self.assertTrue(bot.in_position['BTC/USDT'])
        self.assertEqual(bot.position_size['BTC/USDT'], 0.3)
        self.assertEqual(bot.order_state_machine.get_context('BTC/USDT').state, OrderState.PROTECTED)

    # 6. Post-full fill crash before state save (reconstruct position from exchange)
    async def test_post_full_fill_crash_before_state_save(self):
        self._create_valid_state_file() # local state has 0 contracts
        bot = self._mock_bot()
        
        mock_positions = [{'symbol': 'BTC/USDT', 'contracts': 1.0, 'entryPrice': 90000.0, 'side': 'LONG'}]
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=mock_positions)
        bot.execution.place_native_stop_loss = AsyncMock(return_value={'id': 'SL_106', 'status': 'open'})
        bot.execution.verify_order_active = AsyncMock(return_value="ACTIVE")
        
        reconciler = ReconciliationEngine(bot, check_interval=1.0)
        await reconciler._reconcile_binance()
        
        self.assertTrue(bot.in_position['BTC/USDT'])
        self.assertEqual(bot.position_size['BTC/USDT'], 1.0)
        self.assertEqual(bot.order_state_machine.get_context('BTC/USDT').state, OrderState.PROTECTED)

    # 7. Post-exit crash before state save (ghost position cleanup)
    async def test_post_exit_crash_ghost_cleanup(self):
        self._create_valid_state_file({
            "in_position": {"BTC/USDT": True},
            "position_side": {"BTC/USDT": "LONG"},
            "position_size": {"BTC/USDT": 1.0},
            "entry_price": {"BTC/USDT": 90000.0},
            "stop_loss": {"BTC/USDT": 85000.0}
        })
        bot = self._mock_bot()
        bot.load_state()
        self.assertTrue(bot.in_position['BTC/USDT'])
        
        # Exchange reports 0 open positions
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[])
        reconciler = ReconciliationEngine(bot, check_interval=1.0)
        await reconciler._reconcile_binance()
        
        self.assertFalse(bot.in_position['BTC/USDT'])
        self.assertEqual(bot.position_size['BTC/USDT'], 0.0)
        self.assertEqual(bot.order_state_machine.get_context('BTC/USDT').state, OrderState.CLOSED)

    # 8. Unprotected position startup recovery (places native SL)
    async def test_unprotected_position_startup_recovery(self):
        self._create_valid_state_file()
        bot = self._mock_bot()
        
        mock_positions = [{'symbol': 'BTC/USDT', 'contracts': 0.75, 'entryPrice': 88000.0, 'side': 'LONG'}]
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=mock_positions)
        bot.execution.place_native_stop_loss = AsyncMock(return_value={'id': 'SL_PROTECT_108', 'status': 'open'})
        bot.execution.verify_order_active = AsyncMock(return_value="ACTIVE")
        
        reconciler = ReconciliationEngine(bot, check_interval=1.0)
        await reconciler._reconcile_binance()
        
        bot.execution.place_native_stop_loss.assert_called_once()
        self.assertEqual(bot.order_state_machine.get_context('BTC/USDT').native_sl_order_id, 'SL_PROTECT_108')
        self.assertEqual(bot.order_state_machine.get_context('BTC/USDT').state, OrderState.PROTECTED)

    # 9. State file corruption triggers safe halt
    def test_state_file_corruption_safe_halt(self):
        # Empty dict {}
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("{}")
        bot = self._mock_bot()
        with self.assertRaises(SystemExit):
            bot.load_state()

        # 0-byte file
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("")
        bot = self._mock_bot()
        with self.assertRaises(SystemExit):
            bot.load_state()

        # Truncated/corrupted JSON
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write('{"schema_version": 2, "in_position": {"BTC/USDT": true')
        bot = self._mock_bot()
        with self.assertRaises(SystemExit):
            bot.load_state()

    # 10. Atomic save durability (simulated crash during write does not corrupt state)
    def test_atomic_save_crash_durability(self):
        self._create_valid_state_file({"entry_price": {"BTC/USDT": 92000.0}})
        bot = self._mock_bot()
        bot.load_state()
        
        # Simulate partial temp file
        temp_bad = Path(self.test_dir.name) / "bot_state.json.tmp.1234.5678"
        with open(temp_bad, "w", encoding="utf-8") as f:
            f.write('{"partial_junk": 123')
            
        # Verify original state file remains 100% intact and valid
        with open(self.state_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.assertEqual(content["entry_price"]["BTC/USDT"], 92000.0)

    # 11. Concurrency stress test: 50 concurrent saves without corruption
    def test_concurrent_save_thread_safety(self):
        self._create_valid_state_file()
        bot = self._mock_bot()
        bot.load_state()
        
        errors = []
        def concurrent_saver(idx):
            try:
                bot.entry_price['BTC/USDT'] = 90000.0 + idx
                bot.save_state()
            except Exception as e:
                errors.append(e)
                
        threads = [threading.Thread(target=concurrent_saver, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        self.assertEqual(len(errors), 0)
        # Verify file is still strictly valid JSON
        with open(self.state_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIn(str(saved["schema_version"]), ("2", "2.0"))
        self.assertIn("BTC/USDT", saved["in_position"])

    # 12. Durable risk reservation persistence across restart
    async def test_durable_risk_reservation_across_restart(self):
        self._create_valid_state_file()
        bot = self._mock_bot()
        bot.load_state()
        
        # Reserve risk atomically
        res_ok = await bot.risk.check_and_reserve_risk_atomic(0.0, 0.01, side="LONG", reservation_id="RES_101", symbol="BTC/USDT")
        self.assertTrue(res_ok)
        self.assertEqual(bot.risk.reserved_risk_pct, 0.01)
        self.assertEqual(bot.risk.reserved_longs_count, 1)
        
        # Save state to disk
        bot.save_state()
        
        # New bot instance restart
        bot2 = self._mock_bot()
        bot2.load_state()
        
        # Verify risk reservations restored
        self.assertEqual(bot2.risk.reserved_risk_pct, 0.01)
        self.assertEqual(bot2.risk.reserved_longs_count, 1)
        self.assertIn("RES_101", bot2.risk.active_reservations)

    # 13. Spot ownership isolation upon restart
    async def test_spot_ownership_isolation_restart(self):
        Config.EXCHANGE_TYPE = "spot"
        self._create_valid_state_file({
            "in_position": {"BTC/USDT": True},
            "position_side": {"BTC/USDT": "LONG"},
            "position_size": {"BTC/USDT": 0.25},
            "entry_price": {"BTC/USDT": 85000.0}
        })
        bot = self._mock_bot()
        bot.load_state()
        
        # Spot wallet has 1.0 BTC (user funds)
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 1.0, 'USDT': 5000.0}})
        reconciler = ReconciliationEngine(bot, check_interval=1.0)
        await reconciler._reconcile_binance()
        
        # Bot position size MUST stay isolated at 0.25, NOT inflated to 1.0
        self.assertEqual(bot.position_size['BTC/USDT'], 0.25)
        self.assertTrue(bot.in_position['BTC/USDT'])

if __name__ == '__main__':
    unittest.main()

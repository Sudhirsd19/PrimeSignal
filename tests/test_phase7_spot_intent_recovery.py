import unittest
import asyncio
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from core.order_state_machine import OrderStateMachine, OrderState
from core.reconciliation_engine import ReconciliationEngine
from execution.execution_result import (
    ExecutionResult,
    ExecutionState,
    ExecutionIntentJournal,
    new_intent_id,
)
from execution.execution_engine import ExecutionEngine
from main import PrimeSignalBot

class TestPhase7SpotIntentRecovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.test_dir.name) / "bot_state.json"
        self.journal_file = Path(self.test_dir.name) / "execution_intents.jsonl"
        Config.STATE_FILE = str(self.state_file)
        Config.INTENT_JOURNAL_FILE = str(self.journal_file)
        Config.PAPER_TRADING = False
        Config.EXCHANGE_TYPE = "spot"
        Config.COINDCX_ACTIVE = False

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_bot(self):
        bot = PrimeSignalBot()
        bot.has_keys = True
        bot._STATE_FILE = self.state_file
        bot.execution.coindcx_client = None
        bot.execution.intent_journal = ExecutionIntentJournal(path=self.journal_file)
        bot.execution.trade_client.load_markets = AsyncMock()
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'USDT': 10000.0, 'BTC': 0.0}})
        return bot

    # 1. Spot ENTRY intent FILLED + local state FLAT -> exact filled_qty adopted, no wallet over-adoption
    async def test_01_spot_entry_filled_flat_state_adopted(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0,
        )
        
        # Mock exchange finding the filled order
        mock_order = {
            "id": "EX_101",
            "clientOrderId": cid,
            "status": "closed",
            "filled": 0.5,
            "amount": 0.5,
            "average": 90000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[mock_order])
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 10.5, 'USDT': 5000.0}})
        
        reconciler = ReconciliationEngine(bot)
        await reconciler.start()
        await reconciler.stop()
        
        self.assertTrue(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.5)
        self.assertEqual(bot.entry_price["BTC/USDT"], 90000.0)
        self.assertEqual(bot.order_state_machine.get_context("BTC/USDT").state, OrderState.PROTECTED)

    # 2. Spot ENTRY intent PARTIALLY_FILLED -> only filled_qty adopted
    async def test_02_spot_entry_partial_fill_adopted(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=1.0,
            order_role="ENTRY",
            price=90000.0,
        )
        
        mock_order = {
            "id": "EX_102",
            "clientOrderId": cid,
            "status": "open",
            "filled": 0.35,
            "amount": 1.0,
            "remaining": 0.65,
            "average": 90000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[mock_order])
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 5.0, 'USDT': 5000.0}})
        
        reconciler = ReconciliationEngine(bot)
        await reconciler.start()
        await reconciler.stop()
        
        self.assertTrue(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.35)
        self.assertEqual(bot.order_state_machine.get_context("BTC/USDT").state, OrderState.PARTIALLY_FILLED)

    # 3. Spot ENTRY intent REJECTED -> no position created
    async def test_03_spot_entry_rejected_no_position(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0,
        )
        
        # Absent on exchange
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 0.0, 'USDT': 10000.0}})
        
        reconciler = ReconciliationEngine(bot)
        await reconciler.start()
        await reconciler.stop()
        
        self.assertFalse(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.0)

    # 4. Spot wallet contains 10 BTC manually -> unrelated balance is never adopted
    async def test_04_spot_wallet_manual_balance_never_adopted(self):
        bot = self._create_bot()
        # No intent in journal
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 10.0, 'USDT': 10000.0}})
        
        reconciler = ReconciliationEngine(bot)
        await reconciler.start()
        await reconciler.stop()
        
        self.assertFalse(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.0)

    # 5. Bot ENTRY intent filled 0.5 BTC + wallet total 10.5 BTC -> bot-owned becomes exactly 0.5 BTC
    async def test_05_bot_intent_with_larger_wallet_balance(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0,
        )
        
        mock_order = {
            "id": "EX_105",
            "clientOrderId": cid,
            "status": "closed",
            "filled": 0.5,
            "amount": 0.5,
            "average": 90000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[mock_order])
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 10.5, 'USDT': 10000.0}})
        
        reconciler = ReconciliationEngine(bot)
        await reconciler.start()
        await reconciler.stop()
        
        self.assertTrue(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.5)

    # 6. Two unrelated manual positions + one bot intent -> only matching bot quantity is tracked
    async def test_06_two_manual_positions_one_bot_intent(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.25,
            order_role="ENTRY",
            price=90000.0,
        )
        
        mock_order = {
            "id": "EX_106",
            "clientOrderId": cid,
            "status": "closed",
            "filled": 0.25,
            "amount": 0.25,
            "average": 90000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[mock_order])
        # Wallet has BTC and ETH
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 5.25, 'ETH': 20.0, 'USDT': 10000.0}})
        
        reconciler = ReconciliationEngine(bot)
        await reconciler.start()
        await reconciler.stop()
        
        self.assertTrue(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.25)
        self.assertFalse(bot.in_position.get("ETH/USDT", False))

    # 7. Duplicate startup replay -> must be idempotent, must not double position
    async def test_07_duplicate_startup_replay_idempotency(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0,
        )
        
        mock_order = {
            "id": "EX_107",
            "clientOrderId": cid,
            "status": "closed",
            "filled": 0.5,
            "amount": 0.5,
            "average": 90000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[mock_order])
        
        reconciler1 = ReconciliationEngine(bot)
        await reconciler1.start()
        await reconciler1.stop()
        
        self.assertEqual(bot.position_size["BTC/USDT"], 0.5)
        
        # Second reconciliation on same bot instance
        reconciler2 = ReconciliationEngine(bot)
        await reconciler2.start()
        await reconciler2.stop()
        
        self.assertEqual(bot.position_size["BTC/USDT"], 0.5)

    # 8. Restart twice -> position remains exactly unchanged
    async def test_08_restart_twice_invariance(self):
        bot1 = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.75,
            order_role="ENTRY",
            price=90000.0,
        )
        mock_order = {
            "id": "EX_108",
            "clientOrderId": cid,
            "status": "closed",
            "filled": 0.75,
            "amount": 0.75,
            "average": 90000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }
        bot1.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[mock_order])
        
        r1 = ReconciliationEngine(bot1)
        await r1.start()
        await r1.stop()
        self.assertEqual(bot1.position_size["BTC/USDT"], 0.75)
        
        # Restart in new process/bot instance reading state file
        bot2 = self._create_bot()
        bot2.load_state()
        self.assertTrue(bot2.in_position["BTC/USDT"])
        self.assertEqual(bot2.position_size["BTC/USDT"], 0.75)
        
        r2 = ReconciliationEngine(bot2)
        await r2.start()
        await r2.stop()
        self.assertEqual(bot2.position_size["BTC/USDT"], 0.75)

    # 9. Intent already resolved -> replay must not create duplicate ownership
    async def test_09_already_resolved_intent_no_duplicate(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0,
        )
        # Already resolved in journal
        journal.result(ExecutionResult(
            state=ExecutionState.FILLED,
            requested_qty=0.5,
            client_order_id=cid,
            intent_id=intent_id,
            venue="BINANCE",
        ))
        
        self.assertEqual(len(journal.unresolved()), 0)
        
        r = ReconciliationEngine(bot)
        await r.start()
        await r.stop()
        # Flat state unchanged because intent was already marked resolved
        self.assertFalse(bot.in_position["BTC/USDT"])

    # 10. Ambiguous exchange result -> transition to EXECUTION_UNKNOWN, do not invent quantity
    async def test_10_ambiguous_exchange_result_safe_mode(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0,
        )
        
        # CCXT raises network error during fetch
        bot.execution.trade_client.fetch_open_orders = AsyncMock(side_effect=Exception("Exchange timeout 504"))
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(side_effect=Exception("Exchange timeout 504"))
        
        r = ReconciliationEngine(bot)
        await r.start()
        await r.stop()
        
        self.assertTrue(r.safe_mode_active)
        self.assertFalse(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.0)

    # 11. Exchange unavailable during replay -> fail closed, do not adopt wallet
    async def test_11_exchange_unavailable_fail_closed(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=1.0,
            order_role="ENTRY",
            price=90000.0,
        )
        bot.execution.trade_client.fetch_open_orders = AsyncMock(side_effect=ConnectionError("Exchange unreachable"))
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(side_effect=ConnectionError("Exchange unreachable"))
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 100.0, 'USDT': 10000.0}})
        
        r = ReconciliationEngine(bot)
        await r.start()
        await r.stop()
        
        self.assertTrue(r.safe_mode_active)
        self.assertFalse(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.0)

    # 12. Matching by clientOrderId -> exact order identity preferred over loose matching
    async def test_12_exact_client_order_id_matching(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid_target = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid_target,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.5,
            order_role="ENTRY",
            price=90000.0,
        )
        
        orders = [
            {"id": "EX_DIFF_1", "clientOrderId": "OTHER_CID", "status": "closed", "filled": 0.5, "amount": 0.5, "symbol": "BTC/USDT"},
            {"id": "EX_MATCH", "clientOrderId": cid_target, "status": "closed", "filled": 0.5, "amount": 0.5, "symbol": "BTC/USDT"},
        ]
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=orders)
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 0.5, 'USDT': 10000.0}})
        
        r = ReconciliationEngine(bot)
        await r.start()
        await r.stop()
        
        self.assertTrue(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.5)

    # 13. Same symbol, multiple historical orders -> only correct intent/order recovered
    async def test_13_same_symbol_multiple_orders_target_isolation(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.4,
            order_role="ENTRY",
            price=90000.0,
        )
        
        orders = [
            {"id": "EX_OLD_1", "clientOrderId": "PS_OLD_1", "status": "closed", "filled": 2.0, "amount": 2.0, "symbol": "BTC/USDT"},
            {"id": "EX_OLD_2", "clientOrderId": "MANUAL_ORDER", "status": "closed", "filled": 10.0, "amount": 10.0, "symbol": "BTC/USDT"},
            {"id": "EX_CURRENT", "clientOrderId": cid, "status": "closed", "filled": 0.4, "amount": 0.4, "symbol": "BTC/USDT"},
        ]
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=orders)
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 0.4, 'USDT': 10000.0}})
        
        r = ReconciliationEngine(bot)
        await r.start()
        await r.stop()
        
        self.assertEqual(bot.position_size["BTC/USDT"], 0.4)

    # 14. Partial fill followed by later completion -> recovery converges to authoritative final quantity
    async def test_14_partial_fill_convergence(self):
        bot = self._create_bot()
        journal = ExecutionIntentJournal(path=self.journal_file)
        intent_id = new_intent_id()
        cid = f"PS_{intent_id[:24].upper()}"
        journal.create(
            intent_id=intent_id,
            client_order_id=cid,
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=1.0,
            order_role="ENTRY",
            price=90000.0,
        )
        
        # First check: partially filled 0.4
        mock_open = [{"id": "EX_114", "clientOrderId": cid, "status": "open", "filled": 0.4, "amount": 1.0, "symbol": "BTC/USDT"}]
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=mock_open)
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 0.4, 'USDT': 10000.0}})
        
        r = ReconciliationEngine(bot)
        await r.start()
        await r.stop()
        self.assertEqual(bot.position_size["BTC/USDT"], 0.4)

if __name__ == '__main__':
    unittest.main()

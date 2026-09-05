import unittest
import asyncio
import json
import os
import sys
import time
import tempfile
import threading
import subprocess
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
from risk.risk_manager import RiskManager
import dashboard.app as dash_app

class TestPhase7FinalGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.test_dir.name) / "bot_state.json"
        self.journal_file = Path(self.test_dir.name) / "execution_intents.jsonl"
        Config.STATE_FILE = str(self.state_file)
        Config.INTENT_JOURNAL_FILE = str(self.journal_file)
        Config.PAPER_TRADING = False
        Config.EXCHANGE_TYPE = "futures"
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

    # ---------------------------------------------------------
    # 1. REAL SUBPROCESS SIGKILL & ATOMIC PERSISTENCE TESTS
    # ---------------------------------------------------------
    def test_01_subprocess_sigkill_mid_save(self):
        """Spawns child process writing state in a tight loop and abruptly kills it with SIGKILL / proc.kill()."""
        worker_code = f"""
import os, sys, time, json
sys.path.insert(0, r"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
from config import Config
from main import PrimeSignalBot
Config.STATE_FILE = r"{self.state_file}"
Config.PAPER_TRADING = False
bot = PrimeSignalBot()
bot._STATE_FILE = Config.STATE_FILE
for i in range(10000):
    bot.in_position['BTC/USDT'] = (i % 2 == 0)
    bot.position_size['BTC/USDT'] = float(i)
    bot.save_state()
"""
        proc = subprocess.Popen([sys.executable, "-c", worker_code])
        time.sleep(0.15)
        proc.kill()
        proc.wait()

        # State file must remain intact and valid JSON
        if self.state_file.exists():
            content = self.state_file.read_text(encoding="utf-8")
            data = json.loads(content)
            self.assertIn("in_position", data)
            self.assertIn("position_size", data)
            self.assertIn("BTC/USDT", data["in_position"])

    def test_02_50_concurrent_thread_saves(self):
        """Stress tests 50 simultaneous worker threads executing save_state on 20 symbols."""
        bot = self._create_bot()
        symbols = [f"SYM_{i}/USDT" for i in range(20)]
        for s in symbols:
            bot.in_position[s] = True
            bot.position_size[s] = 1.0
            bot.entry_price[s] = 100.0
            bot.stop_loss[s] = 98.0

        errors = []
        def worker(thread_id):
            try:
                for cycle in range(20):
                    sym = symbols[cycle % len(symbols)]
                    bot.position_size[sym] = float(thread_id * 100 + cycle)
                    bot.save_state()
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertTrue(self.state_file.exists())
        data = json.loads(self.state_file.read_text(encoding="utf-8"))
        for s in symbols:
            self.assertIn(s, data["position_size"])

    # ---------------------------------------------------------
    # 2. DOUBLE-RESTART & CONVERGENCE TESTS
    # ---------------------------------------------------------
    async def test_03_double_restart_and_reconcile_convergence(self):
        """Tests: start -> mutate -> kill -> restart -> reconcile -> kill -> restart -> reconcile."""
        Config.EXCHANGE_TYPE = "spot"
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
            requested_qty=0.6,
            order_role="ENTRY",
            price=92000.0,
        )

        mock_order = {
            "id": "EX_RESTART_1",
            "clientOrderId": cid,
            "status": "closed",
            "filled": 0.6,
            "amount": 0.6,
            "average": 92000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }
        bot1.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[mock_order])
        bot1.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 0.6, 'USDT': 5000.0}})

        # Cycle 1: First restart & reconcile
        r1 = ReconciliationEngine(bot1)
        await r1.start()
        await r1.stop()
        self.assertEqual(bot1.position_size["BTC/USDT"], 0.6)

        # "Crash" -> Instantiate fresh bot2 from disk
        bot2 = self._create_bot()
        bot2.load_state()
        self.assertTrue(bot2.in_position["BTC/USDT"])
        self.assertEqual(bot2.position_size["BTC/USDT"], 0.6)
        bot2.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[mock_order])
        bot2.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 0.6, 'USDT': 5000.0}})

        # Cycle 2: Second restart & reconcile
        r2 = ReconciliationEngine(bot2)
        await r2.start()
        await r2.stop()
        self.assertEqual(bot2.position_size["BTC/USDT"], 0.6)

    # ---------------------------------------------------------
    # 3. PARTIAL FILL & ACCOUNTING INVARIANTS
    # ---------------------------------------------------------
    async def test_04_partial_fill_authoritative_accounting(self):
        """Verifies requested_qty != filled_qty uses ONLY authoritative filled_qty."""
        Config.EXCHANGE_TYPE = "spot"
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
            requested_qty=2.0, # Requested 2.0 BTC
            order_role="ENTRY",
            price=90000.0,
        )

        mock_open = [{
            "id": "EX_PARTIAL_1",
            "clientOrderId": cid,
            "status": "open",
            "filled": 0.45, # Only 0.45 filled
            "amount": 2.0,
            "average": 90000.0,
            "symbol": "BTC/USDT",
            "side": "buy",
        }]
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=mock_open)
        bot.execution.trade_client.fetch_balance = AsyncMock(return_value={'total': {'BTC': 0.45, 'USDT': 5000.0}})

        r = ReconciliationEngine(bot)
        await r.start()
        await r.stop()

        self.assertTrue(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_size["BTC/USDT"], 0.45) # MUST NOT be 2.0!
        self.assertEqual(bot.order_state_machine.get_context("BTC/USDT").state, OrderState.PARTIALLY_FILLED)

    # ---------------------------------------------------------
    # 4. EMERGENCY FLATTEN & TIMEOUT QUARANTINE
    # ---------------------------------------------------------
    async def test_05_emergency_flatten_failure_retains_quarantine(self):
        """Tests: SL failure -> emergency flatten -> timeout -> state remains retained/quarantined."""
        bot = self._create_bot()
        bot.in_position['BTC/USDT'] = True
        bot.position_side['BTC/USDT'] = 'LONG'
        bot.position_size['BTC/USDT'] = 1.0
        bot.entry_price['BTC/USDT'] = 90000.0
        bot.stop_loss['BTC/USDT'] = 88200.0

        ctx = bot.order_state_machine.get_context("BTC/USDT")
        ctx.transition_to(OrderState.PROTECTED, reason="Active position")

        # Mock emergency flatten network failure
        bot.execution.place_order = AsyncMock(side_effect=Exception("Exchange Network Dropped 504"))

        await bot.exit_position('BTC/USDT', reason="TEST_FLATTEN")

        # Must NOT silently reset to flat; must retain position context
        self.assertIn(ctx.state, (OrderState.EXIT_UNKNOWN, OrderState.EXECUTION_UNKNOWN, OrderState.PROTECTED, OrderState.CLOSING))

    # ---------------------------------------------------------
    # 5. RISK RESERVATION DURABILITY & RESTORATION
    # ---------------------------------------------------------
    async def test_06_risk_reservation_serialization_and_durability(self):
        """Verifies risk reservations survive restart and double release."""
        bot = self._create_bot()
        risk = bot.risk

        # 1. Reserve risk within cap
        res_id = "RES_GATE_1"
        reserved = await risk.check_and_reserve_risk_atomic(0.0, 0.008, side="LONG", reservation_id=res_id, symbol="BTC/USDT")
        self.assertTrue(reserved)
        self.assertEqual(risk.reserved_risk_pct, 0.008)

        # 2. Serialize to state and save
        bot.save_state()

        # 3. Fresh bot load
        bot2 = self._create_bot()
        bot2.load_state()
        self.assertEqual(bot2.risk.reserved_risk_pct, 0.008)
        self.assertIn(res_id, bot2.risk.active_reservations)

        # 4. Release risk on fresh bot
        await bot2.risk.release_risk(0.008, side="LONG", reservation_id=res_id)
        self.assertEqual(bot2.risk.reserved_risk_pct, 0.0)

        # 5. Duplicate release (must remain 0.0)
        await bot2.risk.release_risk(0.008, side="LONG", reservation_id=res_id)
        self.assertEqual(bot2.risk.reserved_risk_pct, 0.0)

    # ---------------------------------------------------------
    # 6. DASHBOARD API SECURITY & FAIL-CLOSED GUARDS
    # ---------------------------------------------------------
    async def test_07_dashboard_security_and_live_test_trade_guard(self):
        """Verifies dashboard API key security and hard-blocking of test trade in LIVE mode."""
        # 1. Missing DASHBOARD_SECRET -> 500
        dash_app._DASHBOARD_SECRET = ""
        with patch.dict(os.environ, {"DASHBOARD_SECRET": ""}):
            with self.assertRaises(dash_app.HTTPException) as cm:
                await dash_app.verify_dashboard_key("any_key")
            self.assertIn(cm.exception.status_code, (500, 503))

        # 2. Valid secret set
        dash_app._DASHBOARD_SECRET = "INSTITUTIONAL_SECRET_777"
        with patch.dict(os.environ, {"DASHBOARD_SECRET": "INSTITUTIONAL_SECRET_777"}):
            with self.assertRaises(dash_app.HTTPException) as cm:
                await dash_app.verify_dashboard_key("WRONG_KEY")
            self.assertEqual(cm.exception.status_code, 403)

            # Valid key succeeds without exception
            await dash_app.verify_dashboard_key("INSTITUTIONAL_SECRET_777")

        # 3. Test trade in LIVE mode (Config.PAPER_TRADING = False) MUST be rejected
        Config.PAPER_TRADING = False
        res = await dash_app.trigger_test_trade(dash_app.TestTradeRequest(symbol="BTC/USDT", side="BUY"))
        self.assertEqual(res.get("status"), "error")
        self.assertIn("strictly disabled in LIVE mode", res.get("message", ""))

if __name__ == '__main__':
    unittest.main()

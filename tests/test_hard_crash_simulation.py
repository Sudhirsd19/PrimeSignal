import unittest
import asyncio
import json
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
import ccxt

from config import Config

class TestHardCrashProcessSimulation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.test_dir.name) / "bot_state.json"
        self.journal_file = Path(self.test_dir.name) / "execution_intents.jsonl"
        
        # Initial clean state
        base_state = {
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
            "daily_drawdown_reset_day": str(time.strftime("%Y-%m-%d")),
            "active_risk_reservations": {}
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(base_state, f, indent=2)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_sigkill_during_rapid_state_persistence(self):
        """
        Spawns a child process continuously updating state and journals,
        kills it abruptly with terminate/kill (SIGKILL equivalent),
        and verifies the state file remains 100% parseable and uncorrupted.
        """
        escaped_state_file = str(self.state_file).replace('\\', '\\\\')
        escaped_journal_file = str(self.journal_file).replace('\\', '\\\\')
        
        child_code = f"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, os.getcwd())
from config import Config
Config.STATE_FILE = "{escaped_state_file}"
Config.INTENT_JOURNAL_FILE = "{escaped_journal_file}"
Config.PAPER_TRADING = True

from main import PrimeSignalBot
bot = PrimeSignalBot()
bot._STATE_FILE = Path(Config.STATE_FILE)
bot.load_state()

for i in range(10000):
    bot.entry_price['BTC/USDT'] = 90000.0 + i
    bot.save_state()
    if i == 10:
        print("STARTED_WRITING", flush=True)
    time.sleep(0.001)
"""
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Read until child indicates it has written state
        started = False
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if "STARTED_WRITING" in line:
                started = True
                break

        self.assertTrue(started, "Child process failed to start writing")
        
        # Forcefully kill it mid-write
        time.sleep(0.05)
        proc.kill()
        proc.wait()
        if proc.stdout: proc.stdout.close()
        if proc.stderr: proc.stderr.close()

        # On Windows, brief retry loop for file handle release after process kill
        data = None
        for _ in range(30):
            try:
                if self.state_file.exists():
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    break
            except (PermissionError, json.JSONDecodeError):
                time.sleep(0.05)

        self.assertIsNotNone(data, "Could not load state file after process kill")
        self.assertIn("in_position", data)
        self.assertIn("BTC/USDT", data["in_position"])
        self.assertGreaterEqual(data["entry_price"]["BTC/USDT"], 90000.0)

    def test_sigkill_restart_reconciliation_cycle(self):
        """
        Simulates crash immediately after intent creation in child process,
        kills child, then runs restart process and verifies clean resolution.
        """
        escaped_state_file = str(self.state_file).replace('\\', '\\\\')
        escaped_journal_file = str(self.journal_file).replace('\\', '\\\\')
        
        intent_child_code = f"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, os.getcwd())
from config import Config
from execution.execution_result import ExecutionIntentJournal

Config.STATE_FILE = "{escaped_state_file}"
Config.INTENT_JOURNAL_FILE = "{escaped_journal_file}"

journal = ExecutionIntentJournal(path=Config.INTENT_JOURNAL_FILE)
journal.create(
    intent_id='HARD_CRASH_INTENT_1',
    client_order_id='HARD_CRASH_CID_1',
    venue='BINANCE',
    account_mode='FUTURES',
    symbol='BTC/USDT',
    side='buy',
    requested_qty=0.5,
    order_role='ENTRY',
    price=95000.0
)
print('INTENT_WRITTEN', flush=True)
time.sleep(10) # Wait to be killed
"""
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", intent_child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait until child has written intent to journal
        started = False
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if 'INTENT_WRITTEN' in line:
                started = True
                break

        self.assertTrue(started, "Child failed to write intent record")
        
        # Hard kill the child
        proc.kill()
        proc.wait()
        if proc.stdout: proc.stdout.close()
        if proc.stderr: proc.stderr.close()

        # Startup recovery in parent process
        from execution.execution_result import ExecutionIntentJournal
        from main import PrimeSignalBot
        from unittest.mock import AsyncMock
        
        Config.STATE_FILE = str(self.state_file)
        Config.INTENT_JOURNAL_FILE = str(self.journal_file)
        
        bot = PrimeSignalBot()
        bot.has_keys = True
        bot.execution.coindcx_client = None
        bot.execution.intent_journal = ExecutionIntentJournal(path=self.journal_file)
        bot.execution.trade_client.load_markets = AsyncMock()
        bot.execution.trade_client.fetch_order = AsyncMock(side_effect=ccxt.OrderNotFound("Order not found"))
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[])
        
        # Replay and resolve
        loop = asyncio.new_event_loop()
        resolutions = loop.run_until_complete(bot.execution.replay_and_resolve_unresolved_intents())
        loop.close()
        
        self.assertIn('HARD_CRASH_INTENT_1', resolutions)
        self.assertEqual(resolutions['HARD_CRASH_INTENT_1'].state.value, 'REJECTED')
        
        # Journal must now have 0 unresolved intents
        journal = ExecutionIntentJournal(path=self.journal_file)
        self.assertEqual(len(journal.unresolved()), 0)

if __name__ == '__main__':
    unittest.main()

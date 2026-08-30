import unittest
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from main import PrimeSignalBot

class TestPhase7StateSchema(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.test_dir.name) / "bot_state.json"
        Config.STATE_FILE = str(self.state_file)
        Config.PAPER_TRADING = False
        Config.EXCHANGE_TYPE = "futures"

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_bot(self):
        bot = PrimeSignalBot()
        bot.has_keys = True
        bot._STATE_FILE = self.state_file
        bot.execution.trade_client.load_markets = AsyncMock()
        bot.execution.trade_client.fetch_open_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_closed_orders = AsyncMock(return_value=[])
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[])
        return bot

    def _assert_safe_halt(self, file_content: str, description: str):
        self.state_file.write_text(file_content, encoding="utf-8")
        bot = self._create_bot()
        with self.assertRaises(SystemExit, msg=f"Safe halt (SystemExit) not triggered for: {description}"):
            bot.load_state()

    # 1. Empty file (0 bytes) -> Safe Halt
    def test_01_empty_file_safe_halt(self):
        self._assert_safe_halt("", "empty file (0 bytes)")

    # 2. Whitespace-only file -> Safe Halt
    def test_02_whitespace_only_safe_halt(self):
        self._assert_safe_halt("   \n\t  \r\n  ", "whitespace-only file")

    # 3. Invalid JSON syntax -> Safe Halt
    def test_03_invalid_json_safe_halt(self):
        self._assert_safe_halt('{"in_position": {"BTC/USDT": true}, broken...', "invalid JSON syntax")

    # 4. JSON null literal -> Safe Halt
    def test_04_null_literal_safe_halt(self):
        self._assert_safe_halt("null", "JSON 'null' literal")

    # 5. JSON empty array [] -> Safe Halt
    def test_05_array_safe_halt(self):
        self._assert_safe_halt("[]", "JSON array '[]'")

    # 6. JSON string "" -> Safe Halt
    def test_06_string_safe_halt(self):
        self._assert_safe_halt('""', 'JSON empty string')

    # 7. JSON number 0 -> Safe Halt
    def test_07_zero_safe_halt(self):
        self._assert_safe_halt("0", "JSON number '0'")

    # 8. JSON boolean false -> Safe Halt
    def test_08_false_safe_halt(self):
        self._assert_safe_halt("false", "JSON boolean 'false'")

    # 9. JSON empty dictionary {} -> Safe Halt
    def test_09_empty_dict_safe_halt(self):
        self._assert_safe_halt("{}", "JSON empty dict '{}'")

    # 10. Missing mandatory keys -> Safe Halt
    def test_10_missing_mandatory_keys_safe_halt(self):
        incomplete = {
            "schema_version": "2.0",
            "in_position": {"BTC/USDT": True},
            "position_side": {"BTC/USDT": "LONG"},
            # missing position_size, entry_price, stop_loss
        }
        self._assert_safe_halt(json.dumps(incomplete), "missing mandatory keys")

    # 11. Wrong data types for mandatory fields -> Safe Halt
    def test_11_wrong_data_types_safe_halt(self):
        wrong_types = {
            "in_position": "NOT_A_DICT",
            "position_side": {"BTC/USDT": "LONG"},
            "position_size": {"BTC/USDT": 1.0},
            "entry_price": {"BTC/USDT": 90000.0},
            "stop_loss": {"BTC/USDT": 88000.0},
        }
        self._assert_safe_halt(json.dumps(wrong_types), "wrong data types (non-dict in_position)")

    # 12. Genuinely non-existent file -> Valid clean startup (no SystemExit)
    def test_12_non_existent_file_clean_startup(self):
        if self.state_file.exists():
            self.state_file.unlink()
        bot = self._create_bot()
        # Must not raise SystemExit
        bot.load_state()
        self.assertFalse(bot.in_position.get("BTC/USDT", False))
        self.assertEqual(bot.position_size.get("BTC/USDT", 0.0), 0.0)

    # 13. Valid complete state -> Successfully loaded
    def test_13_valid_state_successfully_loaded(self):
        valid_state = {
            "schema_version": "2.0",
            "in_position": {"BTC/USDT": True},
            "position_side": {"BTC/USDT": "LONG"},
            "position_size": {"BTC/USDT": 0.5},
            "entry_price": {"BTC/USDT": 90000.0},
            "stop_loss": {"BTC/USDT": 88200.0},
            "take_profit_1r": {"BTC/USDT": 91800.0},
            "take_profit_2r": {"BTC/USDT": 93960.0},
            "take_profit": {"BTC/USDT": 97200.0},
            "highest_price_reached": {"BTC/USDT": 90500.0},
            "lowest_price_reached": {"BTC/USDT": 89800.0},
            "partial_tp_taken": {"BTC/USDT": False},
            "tp2_taken": {"BTC/USDT": False},
            "trailing_active": {"BTC/USDT": False},
            "entry_time": {"BTC/USDT": 1700000000000.0},
            "active_risk_reservations": {},
        }
        self.state_file.write_text(json.dumps(valid_state), encoding="utf-8")
        bot = self._create_bot()
        bot.load_state()
        self.assertTrue(bot.in_position["BTC/USDT"])
        self.assertEqual(bot.position_side["BTC/USDT"], "LONG")
        self.assertEqual(bot.position_size["BTC/USDT"], 0.5)
        self.assertEqual(bot.entry_price["BTC/USDT"], 90000.0)

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config
from core.data_pipeline import parse_timeframe_to_minutes
import dashboard.app as dash_app
from fastapi.testclient import TestClient


class TestTimeframeSwitch(unittest.TestCase):
    def setUp(self):
        self.orig_ltf = Config.LTF_TIMEFRAME
        self.orig_secret = dash_app._DASHBOARD_SECRET
        dash_app._DASHBOARD_SECRET = "test_key_tf"
        import os
        self.env_patcher = patch.dict(os.environ, {"DASHBOARD_SECRET": "test_key_tf"})
        self.env_patcher.start()
        self.client = TestClient(dash_app.app)
        self.headers = {"X-API-Key": "test_key_tf"}

    def tearDown(self):
        Config.LTF_TIMEFRAME = self.orig_ltf
        dash_app._DASHBOARD_SECRET = self.orig_secret
        self.env_patcher.stop()
        dash_app.bot_instance = None

    def test_parse_timeframe_to_minutes(self):
        self.assertEqual(parse_timeframe_to_minutes("1m"), 1)
        self.assertEqual(parse_timeframe_to_minutes("3m"), 3)
        self.assertEqual(parse_timeframe_to_minutes("5m"), 5)
        self.assertEqual(parse_timeframe_to_minutes("15m"), 15)
        self.assertEqual(parse_timeframe_to_minutes("30m"), 30)
        self.assertEqual(parse_timeframe_to_minutes("1h"), 60)
        self.assertEqual(parse_timeframe_to_minutes("4h"), 240)
        self.assertEqual(parse_timeframe_to_minutes("1d"), 1440)
        self.assertEqual(parse_timeframe_to_minutes(""), 15)

    def test_api_invalid_timeframe_rejected_without_mutating_config(self):
        Config.LTF_TIMEFRAME = "15m"
        res = self.client.post(
            "/api/set_timeframe",
            json={"timeframe": "2m"},
            headers=self.headers
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Invalid timeframe", data["message"])
        self.assertEqual(Config.LTF_TIMEFRAME, "15m")

    def test_api_standalone_switch(self):
        dash_app.bot_instance = None
        Config.LTF_TIMEFRAME = "15m"
        res = self.client.post(
            "/api/set_timeframe",
            json={"timeframe": "1h"},
            headers=self.headers
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(Config.LTF_TIMEFRAME, "1h")

    def test_api_bot_instance_successful_switch(self):
        mock_bot = MagicMock()
        mock_bot.change_execution_timeframe = AsyncMock(return_value=(True, "Switched successfully"))
        dash_app.bot_instance = mock_bot
        Config.LTF_TIMEFRAME = "15m"

        res = self.client.post(
            "/api/set_timeframe",
            json={"timeframe": "4h"},
            headers=self.headers
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        mock_bot.change_execution_timeframe.assert_awaited_once_with("4h")

    def test_api_bot_instance_rejected_switch_preserves_config(self):
        mock_bot = MagicMock()
        mock_bot.change_execution_timeframe = AsyncMock(return_value=(False, "Order execution in progress"))
        dash_app.bot_instance = mock_bot
        Config.LTF_TIMEFRAME = "15m"

        res = self.client.post(
            "/api/set_timeframe",
            json={"timeframe": "5m"},
            headers=self.headers
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Order execution in progress", data["message"])
        self.assertEqual(Config.LTF_TIMEFRAME, "15m")

    def test_bot_change_execution_timeframe_all_timeframes_supported(self):
        import main
        bot = main.PrimeSignalBot.__new__(main.PrimeSignalBot)
        bot.SUPPORTED_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
        bot.pipeline = MagicMock()
        bot.pipeline.restart_streams = AsyncMock()
        bot.pipeline.ltf_candles = {s: [[1000, 100, 105, 95, 102, 10]] for s in Config.SUPPORTED_SYMBOLS}
        bot.ml_models = {s: MagicMock() for s in Config.SUPPORTED_SYMBOLS}
        bot.order_state_machine = MagicMock()
        bot.order_state_machine.get_context.return_value = MagicMock(state="IDLE")

        for tf in bot.SUPPORTED_TIMEFRAMES:
            Config.LTF_TIMEFRAME = "15m" if tf != "15m" else "1h"
            import asyncio
            success, msg = asyncio.run(bot.change_execution_timeframe(tf))
            self.assertTrue(success, f"Failed for valid timeframe {tf}: {msg}")
            self.assertEqual(Config.LTF_TIMEFRAME, tf)

    def test_bot_change_execution_timeframe_rejects_unsupported(self):
        import main
        bot = main.PrimeSignalBot.__new__(main.PrimeSignalBot)
        bot.SUPPORTED_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
        Config.LTF_TIMEFRAME = "15m"

        import asyncio
        success, msg = asyncio.run(bot.change_execution_timeframe("2m"))
        self.assertFalse(success)
        self.assertIn("Rejected timeframe", msg)
        self.assertEqual(Config.LTF_TIMEFRAME, "15m")

    def test_bot_change_execution_timeframe_rolls_back_on_failure(self):
        import main
        bot = main.PrimeSignalBot.__new__(main.PrimeSignalBot)
        bot.SUPPORTED_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
        bot.pipeline = MagicMock()
        bot.pipeline.restart_streams = AsyncMock(side_effect=RuntimeError("Exchange connection dropped"))
        bot.pipeline.ltf_candles = {s: [] for s in Config.SUPPORTED_SYMBOLS}
        bot.ml_models = {}
        bot.order_state_machine = MagicMock()
        bot.order_state_machine.get_context.return_value = MagicMock(state="IDLE")
        Config.LTF_TIMEFRAME = "15m"

        import asyncio
        success, msg = asyncio.run(bot.change_execution_timeframe("1h"))
        self.assertFalse(success)
        self.assertIn("Exchange connection dropped", msg)
        self.assertEqual(Config.LTF_TIMEFRAME, "15m")


if __name__ == "__main__":
    unittest.main()

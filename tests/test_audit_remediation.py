import unittest
import asyncio
import glob
from pathlib import Path
from unittest.mock import AsyncMock

class TestForensicAuditRemediations(unittest.TestCase):
    def test_f12_no_utf8_bom(self):
        """F-12: Ensure no project python files contain UTF-8 BOM byte sequence."""
        bom = b'\xef\xbb\xbf'
        bom_files = []
        for py_path in glob.glob("**/*.py", recursive=True):
            if ".venv" in py_path:
                continue
            with open(py_path, 'rb') as f:
                content = f.read(3)
                if content == bom:
                    bom_files.append(py_path)
        self.assertEqual(bom_files, [], f"Files with UTF-8 BOM detected: {bom_files}")

    def test_f01_dashboard_secret_no_hardcoded_default(self):
        """F-01: Ensure no hardcoded fallback dashboard secret exists."""
        with open("dashboard/app.py", "r", encoding="utf-8") as f:
            app_code = f.read()
        self.assertNotIn('"primesignal_secret_key"', app_code)
        self.assertNotIn("'primesignal_secret_key'", app_code)
        self.assertNotIn("Devsd@19", app_code)
        self.assertNotIn("7072696d657369676e616c5f7365637265745f6b6579", app_code)
        
        with open("dashboard/templates/index.html", "r", encoding="utf-8") as f:
            html_code = f.read()
        self.assertNotIn('"primesignal_secret_key"', html_code)
        self.assertNotIn("'primesignal_secret_key'", html_code)
        self.assertNotIn("Devsd@19", html_code)

    def test_f08_coindcx_strict_order_identity(self):
        """F-08: Verify CoinDCX reconciliation refuses heuristic matching on ambiguous orders."""
        from execution.coindcx_client import CoinDCXClient
        client = CoinDCXClient(api_key="mock", secret_key="mock")
        
        # Mock active orders with matching side/qty but mismatching client_order_id
        client.fetch_active_orders = AsyncMock(return_value=[
            {
                "client_order_id": "DIFFERENT_CID",
                "side": "buy",
                "total_quantity": 0.5,
                "created_at": 1000000
            }
        ])
        client.fetch_recent_trades = AsyncMock(return_value=[
            {
                "client_order_id": "ANOTHER_DIFFERENT_CID",
                "side": "buy",
                "quantity": 0.5,
                "created_at": 1000000
            }
        ])
        
        result = asyncio.run(client._reconcile_ambiguous_order(
            market="BTCUSDT",
            side="buy",
            amount=0.5,
            created_after_ts=999000,
            client_order_id="TARGET_CID"
        ))
        # Must NOT heuristically adopt different client ID candidate
        self.assertIsNone(result)

    def test_f11_mtf_strategy_closed_candle(self):
        """F-11: Verify multi_timeframe uses last closed candle (iloc[-2]) for HTF trend."""
        with open("strategies/multi_timeframe.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("htf_eval_idx = -2", code)
        self.assertIn("htf_df['close'].iloc[htf_eval_idx]", code)

    def test_f06_funding_rate_fail_closed(self):
        """F-06: Verify main fails closed when funding rate is unavailable."""
        target_file = "main_legacy.py" if Path("main_legacy.py").exists() else "main.py"
        with open(target_file, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("Funding rate data unavailable from exchange (fail-closed protection)", code)

    def test_f07_reconciliation_fail_closed(self):
        """F-07: Verify reconciliation engine activates safe mode if open orders fetch fails."""
        with open("core/reconciliation_engine.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("Binance fetch_open_orders failed", code)
        self.assertIn("self.safe_mode_active = True", code)

    def test_websocket_rejects_unauthorized_and_default_tokens(self):
        """Verify WebSocket closes with 1008 for unauthorized or legacy keys."""
        from fastapi.testclient import TestClient
        import dashboard.app as dash_app
        import os
        from unittest.mock import patch
        from starlette.websockets import WebSocketDisconnect

        orig_secret = dash_app._DASHBOARD_SECRET
        try:
            dash_app._DASHBOARD_SECRET = "secure_secret_123"
            with patch.dict(os.environ, {"DASHBOARD_SECRET": "secure_secret_123"}):
                client = TestClient(dash_app.app)
                # Default or legacy keys must be rejected with 1008
                with self.assertRaises(WebSocketDisconnect) as cm1:
                    with client.websocket_connect("/ws?token=Devsd@19"):
                        pass
                self.assertEqual(cm1.exception.code, 1008)

                with self.assertRaises(WebSocketDisconnect) as cm2:
                    with client.websocket_connect("/ws?token=primesignal_secret_key"):
                        pass
                self.assertEqual(cm2.exception.code, 1008)

                # Valid token must succeed
                with client.websocket_connect("/ws?token=secure_secret_123") as ws:
                    data = ws.receive_json()
                    self.assertIn("symbol", data)
        finally:
            dash_app._DASHBOARD_SECRET = orig_secret

    def test_ml_gate_decision_untrained_hard_gate(self):
        """Verify ML gate_decision blocks when untrained and ML_GATE_MODE='gate'."""
        from ml.confirmation import MLSignalConfirmator
        from config import Config
        ml = MLSignalConfirmator()
        ml.is_trained = False
        with unittest.mock.patch.object(Config, 'ML_GATE_MODE', 'gate'):
            allowed, prob, reason = ml.gate_decision(None, "BUY")
            self.assertFalse(allowed)
            self.assertIn("hard gate BLOCKED", reason)

        with unittest.mock.patch.object(Config, 'ML_GATE_MODE', 'auto'):
            allowed, prob, reason = ml.gate_decision(None, "BUY")
            self.assertTrue(allowed)
            self.assertIn("not gating", reason)

    def test_btc_anchor_default_closed_only(self):
        """Verify evaluate_btc_confluence defaults to eval_closed_only=True."""
        import inspect
        from core.btc_anchor import BTCAnchorEngine
        sig = inspect.signature(BTCAnchorEngine.evaluate_btc_confluence)
        self.assertIn('eval_closed_only', sig.parameters)
        self.assertIs(sig.parameters['eval_closed_only'].default, True)

    def test_exit_position_currency_isolation(self):
        """Verify exit_position uses self._is_inr_account() rather than COINDCX_TRADE_INR directly."""
        with open("main_legacy.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("is_inr = self._is_inr_account()", code)
        self.assertNotIn("is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)", code)

    def test_data_pipeline_refresh_ltf_history_atomic(self):
        """Verify refresh_ltf_history raises RuntimeError and does not mutate candles on failure."""
        from core.data_pipeline import RealTimeDataPipeline
        from config import Config
        pipeline = RealTimeDataPipeline(None)
        orig_candles = {s: [[100, 1, 2, 0.5, 1.5, 10]] for s in Config.SUPPORTED_SYMBOLS}
        pipeline.ltf_candles = dict(orig_candles)

        # Mock one symbol failing
        async def mock_paged(symbol, timeframe, total_bars):
            if symbol == Config.SUPPORTED_SYMBOLS[0]:
                raise RuntimeError("Network timeout")
            return [[200, 2, 3, 1, 2.5, 20]] * 15

        pipeline._fetch_ohlcv_paged = AsyncMock(side_effect=mock_paged)
        with self.assertRaises(RuntimeError):
            asyncio.run(pipeline.refresh_ltf_history())

        # Original candles must remain untouched
        self.assertEqual(pipeline.ltf_candles, orig_candles)


if __name__ == "__main__":
    unittest.main()

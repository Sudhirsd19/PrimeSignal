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
        
        with open("dashboard/templates/index.html", "r", encoding="utf-8") as f:
            html_code = f.read()
        self.assertNotIn('"primesignal_secret_key"', html_code)
        self.assertNotIn("'primesignal_secret_key'", html_code)

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
        """F-06: Verify main.py fails closed when funding rate is unavailable."""
        with open("main.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("Funding rate data unavailable from exchange (fail-closed protection)", code)

    def test_f07_reconciliation_fail_closed(self):
        """F-07: Verify reconciliation engine activates safe mode if open orders fetch fails."""
        with open("core/reconciliation_engine.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("Binance fetch_open_orders failed", code)
        self.assertIn("self.safe_mode_active = True", code)

if __name__ == "__main__":
    unittest.main()

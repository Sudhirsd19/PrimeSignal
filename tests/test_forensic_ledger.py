import unittest
import json
import hashlib
from pathlib import Path
import tempfile
import shutil
from core.immutable_ledger import ImmutableLedger, STRATEGY_VERSION

class TestForensicLedger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ledger_genesis_validation(self):
        # Test that valid hash chain initializes smoothly
        ledger = ImmutableLedger(str(self.ledger_path))
        self.assertEqual(ledger.last_record_hash, "GENESIS_HASH_PRIMESIGNAL_V250")
        
        # Add entry
        r1 = ledger.record_entry("BTC/USDT", "BUY", 1.0, 1.0, 50000.0, 49000.0, 52000.0, 55000.0, 60000.0, "CID_1")
        hash1 = ledger.last_record_hash
        self.assertNotEqual(hash1, "GENESIS_HASH_PRIMESIGNAL_V250")
        
        # Add exit
        r2 = ledger.record_exit("BTC/USDT", "BUY", 1.0, 52000.0, 50000.0, 2000.0, 4.0, "TP1", "CID_1")
        hash2 = ledger.last_record_hash
        self.assertNotEqual(hash2, hash1)
        
        # Re-initialize ledger from file - should succeed and restore hash2
        ledger_reloaded = ImmutableLedger(str(self.ledger_path))
        self.assertEqual(ledger_reloaded.last_record_hash, hash2)

    def test_ledger_tampering_detected(self):
        ledger = ImmutableLedger(str(self.ledger_path))
        ledger.record_entry("BTC/USDT", "BUY", 1.0, 1.0, 50000.0, 49000.0, 52000.0, 55000.0, 60000.0, "CID_1")
        ledger.record_exit("BTC/USDT", "BUY", 1.0, 52000.0, 50000.0, 2000.0, 4.0, "TP1", "CID_1")
        
        # Tamper with the file: change realized_pnl in the second record
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        rec = json.loads(lines[1])
        rec["realized_pnl"] = 999999.0  # Tampered!
        lines[1] = json.dumps(rec) + "\n"
        
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        # Re-initialization must raise ValueError due to tamper detection
        with self.assertRaises(ValueError) as cm:
            ImmutableLedger(str(self.ledger_path))
        self.assertIn("CRITICAL: Immutable ledger integrity check failed", str(cm.exception))

    def test_production_ledger_integrity(self):
        # Also test the actual production ledger file if present
        prod_ledger = Path("data/immutable_trade_ledger.jsonl")
        if prod_ledger.exists():
            ledger = ImmutableLedger(str(prod_ledger))
            self.assertTrue(ledger.last_record_hash != "")

if __name__ == "__main__":
    unittest.main()

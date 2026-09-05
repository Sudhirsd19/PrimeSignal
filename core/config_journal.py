import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from config import Config

class ConfigAuditJournal:
    """
    Append-only forensic journal tracking all governing risk configurations and runtime changes.
    Guarantees every trade can be audited back to the exact risk parameters governing execution.
    """
    def __init__(self, journal_path: str = "data/config_audit_journal.jsonl"):
        self.journal_file = Path(journal_path)
        self.journal_file.parent.mkdir(parents=True, exist_ok=True)

    def record_change(
        self,
        event: str,
        source: str,
        old_snapshot: Optional[Dict[str, Any]] = None,
        new_snapshot: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Records a risk configuration modification or startup snapshot."""
        now = time.time()
        new_snap = new_snapshot or Config.get_risk_config_snapshot()
        config_hash = Config.get_risk_config_hash()

        record = {
            "timestamp": now,
            "iso_time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now)),
            "event": event,
            "source": source,
            "old_snapshot": old_snapshot or {},
            "new_snapshot": new_snap,
            "config_hash": config_hash,
            "details": details or {}
        }

        try:
            with open(self.journal_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                import os
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[CONFIG JOURNAL ERROR] Failed to write config audit journal: {e}")

        return record

    def get_latest_record(self) -> Optional[Dict[str, Any]]:
        """Reads the latest recorded config journal entry."""
        if not self.journal_file.exists():
            return None
        last_line = None
        try:
            with open(self.journal_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        last_line = line
            if last_line:
                return json.loads(last_line)
        except Exception as e:
            print(f"[CONFIG JOURNAL ERROR] Error reading latest config record: {e}")
        return None

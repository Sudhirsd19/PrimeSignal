import unittest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path
from core.macro_calendar import MacroNewsCalendar
from config import Config

class TestMacroCalendarFailover(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache_file = Path("__test_cache_failover__.json")
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.cal = MacroNewsCalendar(overlay_path="__nonexistent__.json", cache_path=str(self.cache_file))
        self.cal.events = []
        self.cal.source = "NONE"

    def tearDown(self):
        if self.cache_file.exists():
            self.cache_file.unlink()

    async def test_failover_to_backup_when_primary_fails(self):
        """When primary calendar URL returns 500, backup URL is queried."""
        primary_called = False
        backup_called = False

        async def mock_fetch(session, url):
            nonlocal primary_called, backup_called
            if "faireconomy" in url:
                primary_called = True
                return None
            if "datasets" in url or "backup" in url:
                backup_called = True
                return [{
                    "title": "US CPI Test",
                    "time": datetime(2026, 3, 11, 12, 30, tzinfo=timezone.utc),
                    "impact": "high",
                    "impact_rank": 3,
                    "currency": "USD"
                }]
            return None

        with patch.object(self.cal, "_fetch_from_endpoint", side_effect=mock_fetch):
            success = await self.cal.refresh(force=True)
            self.assertTrue(success)
            self.assertTrue(primary_called)
            self.assertTrue(backup_called)
            self.assertEqual(self.cal.source, "BACKUP_FEED")
            self.assertEqual(len(self.cal.events), 1)
            self.assertEqual(self.cal.events[0]["title"], "US CPI Test")

    async def test_fallback_to_builtin_schedule_when_all_network_fails(self):
        """When all network calls fail and cache is missing, builtin schedule is activated."""
        with patch.object(self.cal, "_fetch_from_endpoint", return_value=None):
            success = await self.cal.refresh(force=True)
            self.assertTrue(success)
            self.assertEqual(self.cal.source, "BUILTIN_SCHEDULE")
            self.assertGreater(len(self.cal.events), 10)
            # Verify FOMC/CPI events exist in builtin schedule
            titles = [e["title"] for e in self.cal.events]
            self.assertTrue(any("Interest Rate" in t or "CPI" in t for t in titles))

    async def test_blackout_in_builtin_schedule(self):
        """Verifies blackout window calculation works with builtin schedule."""
        with patch.object(self.cal, "_fetch_from_endpoint", return_value=None):
            await self.cal.refresh(force=True)
            # 2026-03-18 18:00 UTC is FOMC meeting
            fomc_time = datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc)
            in_window, reason = self.cal.is_blackout(fomc_time)
            self.assertTrue(in_window)
            self.assertIn("Interest Rate", reason)

if __name__ == "__main__":
    unittest.main()

"""PrimeSignal economic-calendar safety filter.

When the macro filter is enabled, absence or failure of calendar data is a
hard safety condition. The bot must not silently trade while its news blackout
source is unavailable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from config import Config

_IMPACT_RANK = {"low": 1, "medium": 2, "med": 2, "moderate": 2, "high": 3, "holiday": 0, "bank holiday": 0}
_MIN_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
_RELEVANT_CURRENCIES = ("USD", "US")

class MacroNewsCalendar:
    LOCAL_OVERLAY_PATH = Path("data/macro_calendar.json")
    CACHE_PATH = Path("data/macro_calendar_cache.json")

    def __init__(self, overlay_path: Optional[str] = None, cache_path: Optional[str] = None):
        self.overlay_path = Path(overlay_path) if overlay_path else self.LOCAL_OVERLAY_PATH
        self.cache_path = Path(cache_path) if cache_path else self.CACHE_PATH
        self.events: list[dict[str, Any]] = []
        self.source = "NONE"
        self.last_refresh = 0.0
        self.last_error: Optional[str] = None
        self._load_overlay()
        self._load_cache()

    @staticmethod
    def _coerce_utc(stamp: Any) -> Optional[datetime]:
        if stamp is None: return None
        try:
            if isinstance(stamp, (int, float)) or (isinstance(stamp, str) and stamp.isdigit()):
                ts = float(stamp)
                if ts > 1e11: ts /= 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            text = str(stamp).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError, OSError):
            return None

    @classmethod
    def _parse_feed(cls, raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, dict): raw = raw.get("events") or raw.get("data") or []
        if not isinstance(raw, list): return []
        parsed = []
        for item in raw:
            if not isinstance(item, dict): continue
            title = str(item.get("title") or item.get("event") or item.get("name") or "").strip()
            when = cls._coerce_utc(item.get("date") or item.get("time") or item.get("datetime") or item.get("timestamp"))
            if not title or when is None: continue
            impact = str(item.get("impact") or item.get("importance") or "high").strip().lower()
            currency = str(item.get("country") or item.get("currency") or "USD").strip().upper()
            parsed.append({"title": title, "time": when, "impact": impact, "impact_rank": _IMPACT_RANK.get(impact, 3), "currency": currency})
        parsed.sort(key=lambda e: e["time"])
        return parsed

    def _load_overlay(self):
        try:
            if self.overlay_path.exists():
                parsed = self._parse_feed(json.loads(self.overlay_path.read_text(encoding="utf-8")))
                if parsed:
                    self.events, self.source = parsed, "LOCAL_OVERLAY"
        except (OSError, json.JSONDecodeError) as exc:
            self.last_error = f"overlay load failed: {exc}"

    def _load_cache(self):
        if self.source == "LOCAL_OVERLAY": return
        try:
            if self.cache_path.exists():
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                parsed = self._parse_feed(payload.get("events") if isinstance(payload, dict) else payload)
                if parsed:
                    self.events = parsed
                    self.source = "DISK_CACHE"
                    self.last_refresh = float(payload.get("fetched_at", 0.0)) if isinstance(payload, dict) else 0.0
        except (OSError, json.JSONDecodeError) as exc:
            self.last_error = f"cache load failed: {exc}"

    async def refresh(self, force: bool = False) -> bool:
        url = str(getattr(Config, "ECONOMIC_CALENDAR_URL", "") or "").strip()
        if not url:
            self.last_error = "economic calendar URL is not configured"
            return False
        if self.source == "LOCAL_OVERLAY" and not force: return True
        ttl = max(60, int(getattr(Config, "ECONOMIC_CALENDAR_CACHE_MINS", 30)) * 60)
        if not force and self.events and (time.time() - self.last_refresh) < ttl: return True
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=8.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "PrimeSignal/2.7"}) as resp:
                    if resp.status != 200:
                        self.last_error = f"calendar feed HTTP {resp.status}"
                        return False
                    raw = await resp.json(content_type=None)
            parsed = self._parse_feed(raw)
            if not parsed:
                self.last_error = "calendar feed returned no parsable events"
                return False
            self.events, self.source, self.last_refresh, self.last_error = parsed, "LIVE_FEED", time.time(), None
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps({"fetched_at": self.last_refresh, "source_url": url, "events": [{**e, "time": e["time"].isoformat()} for e in self.events]}, indent=2), encoding="utf-8")
            except OSError as exc:
                self.last_error = f"cache write failed: {exc}"
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def _min_impact_rank(self):
        return _MIN_IMPACT_RANK.get(str(getattr(Config, "NEWS_MIN_IMPACT", "high")).strip().lower(), 3)

    def relevant_events(self):
        floor = self._min_impact_rank()
        return [e for e in self.events if e.get("impact_rank", 3) >= floor and str(e.get("currency", "USD")).upper() in _RELEVANT_CURRENCIES]

    def is_blackout(self, now_utc: Optional[datetime] = None) -> tuple[bool, str]:
        if not getattr(Config, "ENABLE_MACRO_NEWS_FILTER", True): return False, ""
        now = now_utc or datetime.now(timezone.utc)
        events = self.relevant_events()
        # P0 safety invariant: enabled + unavailable calendar = halt new entries.
        if not events:
            reason = f"Macro calendar unavailable — trading halted ({self.last_error or 'no relevant calendar events loaded'})"
            return True, reason
        before = timedelta(minutes=int(getattr(Config, "NEWS_BLACKOUT_BEFORE_MIN", 15)))
        after = timedelta(minutes=int(getattr(Config, "NEWS_BLACKOUT_AFTER_MIN", 20)))
        for event in events:
            start, end = event["time"] - before, event["time"] + after
            if start <= now <= end:
                return True, f"{event['currency']} {event['title']} ({event['impact']} impact) at {event['time'].strftime('%H:%M')} UTC [window {start.strftime('%H:%M')}-{end.strftime('%H:%M')} UTC]"
        if getattr(Config, "ENABLE_RECURRING_NEWS_WINDOWS", False): return self._recurring_fallback(now)
        return False, ""

    def _recurring_fallback(self, now):
        if now.weekday() >= 5: return False, ""
        if now.hour == 12 and 20 <= now.minute <= 45: return True, "Recurring window: US morning macro data"
        if now.hour == 13 and 20 <= now.minute <= 45: return True, "Recurring window: US main macro data"
        if now.weekday() == 2 and (now.hour == 18 or (now.hour == 19 and now.minute <= 30)): return True, "Recurring window: Fed FOMC legacy window"
        return False, ""

    def next_event(self, now_utc: Optional[datetime] = None):
        now = now_utc or datetime.now(timezone.utc)
        return next((e for e in self.relevant_events() if e["time"] >= now), None)

    def status(self):
        upcoming = self.next_event()
        return {"enabled": bool(getattr(Config, "ENABLE_MACRO_NEWS_FILTER", True)), "source": self.source, "events_loaded": len(self.events), "relevant_events": len(self.relevant_events()), "last_refresh": self.last_refresh, "last_error": self.last_error, "min_impact": str(getattr(Config, "NEWS_MIN_IMPACT", "high")), "recurring_fallback": bool(getattr(Config, "ENABLE_RECURRING_NEWS_WINDOWS", False)), "next_event": None if upcoming is None else {"title": upcoming["title"], "currency": upcoming["currency"], "impact": upcoming["impact"], "time_utc": upcoming["time"].isoformat()}}

    def describe_mode(self):
        if not getattr(Config, "ENABLE_MACRO_NEWS_FILTER", True): return "DISABLED"
        if self.events: return f"CALENDAR ({self.source}, {len(self.relevant_events())} relevant events)"
        return f"CALENDAR UNAVAILABLE — trading halted ({self.last_error or 'not yet fetched'})"

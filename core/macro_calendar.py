"""
PrimeSignal Macro Economic Calendar
===================================

H-06 FIX — the previous "macro news blackout" was not news-aware at all. It
blocked three fixed clock windows on EVERY weekday (12:20-12:45, 13:20-13:45,
and Wednesday 18:00-19:30 UTC) whether or not any data was actually scheduled,
and it could never see a NFP/PPI/PCE release outside those slots. It was
nonetheless surfaced to the user as an "Economic News Calendar Filter".

This module implements the real thing:

  * A live economic calendar feed (default: the free ForexFactory weekly JSON,
    no API key required) fetched with aiohttp and cached to disk.
  * An optional local overlay at ``data/macro_calendar.json`` for manual or
    air-gapped overrides — entries there always win.
  * Per-event impact filtering (``NEWS_MIN_IMPACT``) and a configurable
    ±window around each release (``NEWS_BLACKOUT_BEFORE_MIN`` /
    ``NEWS_BLACKOUT_AFTER_MIN``).
  * The legacy recurring-clock heuristic is preserved but OPT-IN
    (``ENABLE_RECURRING_NEWS_WINDOWS``), so default behaviour no longer burns
    ~90 minutes of every weekday.

Design note: this filter FAILS OPEN. If the calendar cannot be fetched we do
not halt trading — a missing feed must not become an unannounced trading
outage. The dashboard/status surface reports the degraded mode instead.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from config import Config

# Impact strings seen across calendar feeds, normalised to a rank.
_IMPACT_RANK = {
    "low": 1,
    "medium": 2,
    "med": 2,
    "moderate": 2,
    "high": 3,
    "holiday": 0,
    "bank holiday": 0,
}

_MIN_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}

# Currencies that move crypto materially. Non-USD events (e.g. JPY BoJ) rarely
# cause the liquidation cascades this filter exists to avoid.
_RELEVANT_CURRENCIES = ("USD", "US")


class MacroNewsCalendar:
    """Live economic calendar with disk cache and a manual local overlay."""

    LOCAL_OVERLAY_PATH = Path("data/macro_calendar.json")
    CACHE_PATH = Path("data/macro_calendar_cache.json")

    def __init__(self, overlay_path: Optional[str] = None, cache_path: Optional[str] = None):
        self.overlay_path = Path(overlay_path) if overlay_path else self.LOCAL_OVERLAY_PATH
        self.cache_path = Path(cache_path) if cache_path else self.CACHE_PATH
        self.events: list[dict[str, Any]] = []
        self.source: str = "NONE"
        self.last_refresh: float = 0.0
        self.last_error: Optional[str] = None
        self._load_overlay()
        self._load_cache()

    # ── Ingestion ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_feed(raw: Any) -> list[dict[str, Any]]:
        """Normalises a calendar feed payload into UTC-stamped event dicts."""
        if isinstance(raw, dict):
            raw = raw.get("events") or raw.get("data") or []
        if not isinstance(raw, list):
            return []

        parsed: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("event") or item.get("name") or "").strip()
            if not title:
                continue
            stamp = item.get("date") or item.get("time") or item.get("datetime") or item.get("timestamp")
            when = MacroNewsCalendar._coerce_utc(stamp)
            if when is None:
                continue
            impact = str(item.get("impact") or item.get("importance") or "high").strip().lower()
            currency = str(item.get("country") or item.get("currency") or "USD").strip().upper()
            parsed.append({
                "title": title,
                "time": when,
                "impact": impact,
                "impact_rank": _IMPACT_RANK.get(impact, 3),
                "currency": currency,
            })
        parsed.sort(key=lambda e: e["time"])
        return parsed

    @staticmethod
    def _coerce_utc(stamp: Any) -> Optional[datetime]:
        if stamp is None:
            return None
        try:
            if isinstance(stamp, (int, float)) or (isinstance(stamp, str) and stamp.isdigit()):
                ts = float(stamp)
                if ts > 1e11:  # milliseconds
                    ts /= 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            text = str(stamp).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError, OSError):
            return None

    def _load_overlay(self) -> None:
        """Manual overlay always wins — useful for air-gapped or pinned runs."""
        try:
            if self.overlay_path.exists():
                raw = json.loads(self.overlay_path.read_text(encoding="utf-8"))
                parsed = self._parse_feed(raw)
                if parsed:
                    self.events = parsed
                    self.source = "LOCAL_OVERLAY"
        except (OSError, json.JSONDecodeError) as exc:
            self.last_error = f"overlay load failed: {exc}"

    def _load_cache(self) -> None:
        if self.source in ("LOCAL_OVERLAY",):
            return
        try:
            if self.cache_path.exists():
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                raw_events = payload.get("events") if isinstance(payload, dict) else payload
                parsed = self._parse_feed(raw_events)
                if parsed:
                    self.events = parsed
                    self.source = "DISK_CACHE"
                    self.last_refresh = float(payload.get("fetched_at", 0.0)) if isinstance(payload, dict) else 0.0
        except (OSError, json.JSONDecodeError) as exc:
            self.last_error = f"cache load failed: {exc}"

    async def refresh(self, force: bool = False) -> bool:
        """Fetches the calendar feed, honouring the configured cache TTL."""
        url = str(getattr(Config, "ECONOMIC_CALENDAR_URL", "") or "").strip()
        if not url:
            return False
        if self.source == "LOCAL_OVERLAY" and not force:
            return True

        ttl_secs = max(60, int(getattr(Config, "ECONOMIC_CALENDAR_CACHE_MINS", 30)) * 60)
        if not force and (time.time() - self.last_refresh) < ttl_secs and self.events:
            return True

        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=8.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "PrimeSignal/2.6"}) as resp:
                    if resp.status != 200:
                        self.last_error = f"calendar feed HTTP {resp.status}"
                        return False
                    raw = await resp.json(content_type=None)

            parsed = self._parse_feed(raw)
            if not parsed:
                self.last_error = "calendar feed returned no parsable events"
                return False

            self.events = parsed
            self.source = "LIVE_FEED"
            self.last_refresh = time.time()
            self.last_error = None
            self._persist_cache()
            return True
        except Exception as exc:  # network, JSON, or aiohttp missing
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def _persist_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps({
                    "fetched_at": self.last_refresh,
                    "source_url": str(getattr(Config, "ECONOMIC_CALENDAR_URL", "")),
                    "events": [
                        {**e, "time": e["time"].isoformat()} for e in self.events
                    ],
                }, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self.last_error = f"cache write failed: {exc}"

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _min_impact_rank(self) -> int:
        return _MIN_IMPACT_RANK.get(
            str(getattr(Config, "NEWS_MIN_IMPACT", "high")).strip().lower(), 3
        )

    def relevant_events(self) -> list[dict[str, Any]]:
        floor = self._min_impact_rank()
        return [
            e for e in self.events
            if e.get("impact_rank", 3) >= floor
            and str(e.get("currency", "USD")).upper() in _RELEVANT_CURRENCIES
        ]

    def is_blackout(self, now_utc: Optional[datetime] = None) -> tuple[bool, str]:
        """Returns (in_blackout, reason)."""
        if not getattr(Config, "ENABLE_MACRO_NEWS_FILTER", True):
            return False, ""

        now = now_utc or datetime.now(timezone.utc)
        before = timedelta(minutes=int(getattr(Config, "NEWS_BLACKOUT_BEFORE_MIN", 15)))
        after = timedelta(minutes=int(getattr(Config, "NEWS_BLACKOUT_AFTER_MIN", 20)))

        for event in self.relevant_events():
            start = event["time"] - before
            end = event["time"] + after
            if start <= now <= end:
                return True, (
                    f"{event['currency']} {event['title']} ({event['impact']} impact) "
                    f"at {event['time'].strftime('%H:%M')} UTC "
                    f"[window {start.strftime('%H:%M')}-{end.strftime('%H:%M')} UTC]"
                )

        if getattr(Config, "ENABLE_RECURRING_NEWS_WINDOWS", False):
            return self._recurring_fallback(now)

        return False, ""

    def _recurring_fallback(self, now: datetime) -> tuple[bool, str]:
        """Legacy fixed-clock heuristic. Opt-in only (see module docstring)."""
        hour, minute, weekday = now.hour, now.minute, now.weekday()
        if weekday >= 5:
            return False, ""
        if hour == 12 and 20 <= minute <= 45:
            return True, "Recurring window: US morning macro data (12:20-12:45 UTC, legacy heuristic)"
        if hour == 13 and 20 <= minute <= 45:
            return True, "Recurring window: US main macro data (13:20-13:45 UTC, legacy heuristic)"
        if weekday == 2 and (hour == 18 or (hour == 19 and minute <= 30)):
            return True, "Recurring window: Fed FOMC (Wed 18:00-19:30 UTC, legacy heuristic)"
        return False, ""

    def next_event(self, now_utc: Optional[datetime] = None) -> Optional[dict[str, Any]]:
        now = now_utc or datetime.now(timezone.utc)
        for event in self.relevant_events():
            if event["time"] >= now:
                return event
        return None

    def status(self) -> dict[str, Any]:
        """Structured status for the dashboard / logs."""
        now = datetime.now(timezone.utc)
        upcoming = self.next_event(now)
        return {
            "enabled": bool(getattr(Config, "ENABLE_MACRO_NEWS_FILTER", True)),
            "source": self.source,
            "events_loaded": len(self.events),
            "relevant_events": len(self.relevant_events()),
            "last_refresh": self.last_refresh,
            "last_error": self.last_error,
            "min_impact": str(getattr(Config, "NEWS_MIN_IMPACT", "high")),
            "recurring_fallback": bool(getattr(Config, "ENABLE_RECURRING_NEWS_WINDOWS", False)),
            "next_event": None if upcoming is None else {
                "title": upcoming["title"],
                "currency": upcoming["currency"],
                "impact": upcoming["impact"],
                "time_utc": upcoming["time"].isoformat(),
            },
        }

    def describe_mode(self) -> str:
        """One-line, human-honest description of what is actually active."""
        if not getattr(Config, "ENABLE_MACRO_NEWS_FILTER", True):
            return "DISABLED"
        if self.events:
            return f"CALENDAR ({self.source}, {len(self.relevant_events())} relevant events)"
        if getattr(Config, "ENABLE_RECURRING_NEWS_WINDOWS", False):
            return "RECURRING-WINDOW FALLBACK (no calendar data)"
        return f"CALENDAR UNAVAILABLE — filter failing open ({self.last_error or 'not yet fetched'})"

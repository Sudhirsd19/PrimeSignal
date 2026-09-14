from datetime import datetime, timezone

from core import macro_calendar


class _PaperConfig:
    ENABLE_MACRO_NEWS_FILTER = True
    PAPER_TRADING = True
    NEWS_MIN_IMPACT = "high"
    ENABLE_RECURRING_NEWS_WINDOWS = False
    NEWS_BLACKOUT_BEFORE_MIN = 15
    NEWS_BLACKOUT_AFTER_MIN = 20


class _LiveConfig(_PaperConfig):
    PAPER_TRADING = False


def test_empty_calendar_does_not_halt_paper_mode(monkeypatch):
    monkeypatch.setattr(macro_calendar, "Config", _PaperConfig)
    cal = macro_calendar.MacroNewsCalendar()
    result = cal.is_blackout(datetime(2026, 9, 14, tzinfo=timezone.utc))
    assert result[0] is False
    assert "Paper mode" in result[1]


def test_empty_calendar_fails_closed_for_live_mode(monkeypatch):
    monkeypatch.setattr(macro_calendar, "Config", _LiveConfig)
    cal = macro_calendar.MacroNewsCalendar()
    result = cal.is_blackout(datetime(2026, 9, 14, tzinfo=timezone.utc))
    assert result[0] is True
    assert "trading halted" in result[1]


def test_loaded_event_still_blocks_inside_blackout_window(monkeypatch):
    monkeypatch.setattr(macro_calendar, "Config", _PaperConfig)
    cal = macro_calendar.MacroNewsCalendar()
    cal.events = [{
        "title": "CPI",
        "time": datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc),
        "impact": "high",
        "impact_rank": 3,
        "currency": "USD",
    }]
    result = cal.is_blackout(datetime(2026, 9, 14, 12, 10, tzinfo=timezone.utc))
    assert result[0] is True
    assert "CPI" in result[1]

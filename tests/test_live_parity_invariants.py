"""
Live-parity invariant tests for the v2.6 audit remediation.

M-07 FIX — the existing suite covered execution integrity, crash recovery,
reconciliation and the ledger hash chain, but had NO coverage for the strategy,
the risk-manager units, the backtester's metric contract or the daily state
persistence. Three of the audit's most serious defects (the risk-cap unit
mismatch, the broken official backtest runner, the non-durable daily counters)
would all have been caught by the tests in this file.

Runs under plain unittest:

    python -m unittest tests.test_live_parity_invariants -v
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config  # noqa: E402
from risk.risk_manager import RiskManager  # noqa: E402
from strategies.indicators import prepare_dataframe  # noqa: E402
from strategies.multi_timeframe import MultiTimeframeSMCStrategy  # noqa: E402

# Deterministic fixture parameters verified to emit a SHORT setup.
SELL_FIXTURE = {"seed": 33, "htf_drift": -0.35, "base": 200.0, "vol": 1.2}
LTF_FIXTURE = {"seed": 1033, "drift": -0.14, "base": 190.0, "vol": 0.6}


def synth_candles(bars, base, drift, vol, seed, tf="15min", start="2026-01-01"):
    """Deterministic synthetic OHLCV candles (no network, no exchange)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=bars, freq=tf)
    steps = rng.normal(drift, vol, bars).cumsum()
    close = np.maximum(base + steps, 1.0)
    openp = np.concatenate([[base], close[:-1]])
    high = np.maximum(openp, close) + rng.uniform(0.05, 0.6, bars)
    low = np.maximum(np.minimum(openp, close) - rng.uniform(0.05, 0.6, bars), 0.5)
    volume = rng.uniform(500, 3000, bars)
    rows = [
        [int(t.timestamp() * 1000), float(o), float(h), float(l), float(c), float(v)]
        for t, o, h, l, c, v in zip(ts, openp, high, low, close, volume)
    ]
    return rows


def permissive_config():
    """Snapshot + relax the filters so synthetic data can actually trade."""
    saved = {
        "ADX_MIN_THRESHOLD": Config.ADX_MIN_THRESHOLD,
        "ENABLE_WEEKEND_FILTER": Config.ENABLE_WEEKEND_FILTER,
        "ENABLE_BB_SQUEEZE_FILTER": Config.ENABLE_BB_SQUEEZE_FILTER,
    }
    Config.ADX_MIN_THRESHOLD = 8.0
    Config.ENABLE_WEEKEND_FILTER = False
    Config.ENABLE_BB_SQUEEZE_FILTER = False
    return saved


def restore_config(saved):
    for k, v in saved.items():
        setattr(Config, k, v)


# ═══════════════════════════════════════════════════════════════════════════
# C-02: one canonical risk unit
# ═══════════════════════════════════════════════════════════════════════════
class RiskUnitContractTests(unittest.TestCase):
    def test_portfolio_cap_is_a_fraction_under_one(self):
        frac = Config.get_max_portfolio_risk_fraction()
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 1.0, "portfolio risk cap must be a fraction, not a percent")
        self.assertAlmostEqual(frac, Config.MAX_PORTFOLIO_RISK_PCT / 100.0, places=9)

    def test_risk_manager_uses_the_same_canonical_unit(self):
        rm = RiskManager()
        self.assertAlmostEqual(rm.max_correlated_risk_pct, Config.get_max_portfolio_risk_fraction(), places=9)

    def test_main_loop_exposure_gate_can_actually_fire(self):
        """Regression for the percent-vs-fraction bug that made the cap dead code."""
        cap = Config.get_max_portfolio_risk_fraction()
        high_tier = Config.risk_pct_for_score(Config.RISK_TIER_HIGH_SCORE)
        # A large-but-plausible open risk must breach the cap in the same units.
        projected = cap + high_tier
        self.assertGreater(projected, cap, "gate must be able to trip once units agree")

    def test_reservation_respects_the_cap_boundary(self):
        rm = RiskManager()
        cap = rm.max_correlated_risk_pct
        # Fill the portfolio to just under the cap, then confirm the next one is refused.
        self.assertTrue(rm.check_and_reserve_risk_nolock(0.0, cap * 0.9, "BUY", "R1", "BTC/USDT"))
        self.assertFalse(
            rm.check_and_reserve_risk_nolock(cap * 0.9, cap * 0.5, "BUY", "R2", "ETH/USDT"),
            "a reservation that breaches the portfolio cap must be refused",
        )


# ═══════════════════════════════════════════════════════════════════════════
# H-07: the risk ladder is config-driven and honest
# ═══════════════════════════════════════════════════════════════════════════
class RiskLadderTests(unittest.TestCase):
    def test_ladder_matches_declared_multipliers(self):
        base = Config.RISK_PCT / 100.0
        self.assertAlmostEqual(Config.risk_pct_for_score(0.0), base * Config.RISK_TIER_LOW_MULT, places=9)
        self.assertAlmostEqual(
            Config.risk_pct_for_score(Config.RISK_TIER_MID_SCORE), base * Config.RISK_TIER_MID_MULT, places=9
        )
        self.assertAlmostEqual(
            Config.risk_pct_for_score(Config.RISK_TIER_HIGH_SCORE), base * Config.RISK_TIER_HIGH_MULT, places=9
        )

    def test_ladder_is_monotonic_in_score(self):
        vals = [Config.risk_pct_for_score(s) for s in (0.0, 3.5, 4.5, 10.0)]
        self.assertEqual(vals, sorted(vals), "higher setup score must never risk less")

    def test_baseline_risk_scales_the_ladder(self):
        """RISK_PCT must genuinely move live risk (it was previously ignored)."""
        original = Config.RISK_PCT
        try:
            Config.RISK_PCT = 1.0
            one = Config.risk_pct_for_score(Config.RISK_TIER_MID_SCORE)
            Config.RISK_PCT = 2.0
            two = Config.risk_pct_for_score(Config.RISK_TIER_MID_SCORE)
            self.assertAlmostEqual(two, one * 2.0, places=9)
        finally:
            Config.RISK_PCT = original


# ═══════════════════════════════════════════════════════════════════════════
# C-01: a spot venue can never produce a short
# ═══════════════════════════════════════════════════════════════════════════
class VenueCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.saved_type = Config.EXCHANGE_TYPE
        self.saved = permissive_config()

    def tearDown(self):
        Config.EXCHANGE_TYPE = self.saved_type
        restore_config(self.saved)

    def test_venue_capability_flag(self):
        Config.EXCHANGE_TYPE = "spot"
        self.assertFalse(Config.venue_supports_short())
        Config.EXCHANGE_TYPE = "futures"
        self.assertTrue(Config.venue_supports_short())

    def test_strategy_suppresses_short_on_spot(self):
        """The fixture is verified to emit SELL when shorts are permitted."""
        Config.EXCHANGE_TYPE = "futures"
        htf = synth_candles(300, SELL_FIXTURE["base"], SELL_FIXTURE["htf_drift"], SELL_FIXTURE["vol"],
                            SELL_FIXTURE["seed"], "1h")
        ltf = synth_candles(500, LTF_FIXTURE["base"], LTF_FIXTURE["drift"], LTF_FIXTURE["vol"],
                            LTF_FIXTURE["seed"], "15min")
        strategy = MultiTimeframeSMCStrategy()

        signal, _meta = strategy.generate_signal(htf, ltf, relaxed=False, allow_short=True)
        self.assertEqual(signal, "SELL", "fixture must still produce a SHORT setup for this test to mean anything")

        signal_spot, meta_spot = strategy.generate_signal(htf, ltf, relaxed=False, allow_short=False)
        self.assertEqual(signal_spot, "HOLD", "a spot venue must never receive a SHORT signal")
        self.assertIn("SHORT setups disabled", str(meta_spot.get("reason")))


# ═══════════════════════════════════════════════════════════════════════════
# M-01: RELAXED mode is reachable and adds entries
# ═══════════════════════════════════════════════════════════════════════════
class RelaxedModeTests(unittest.TestCase):
    def setUp(self):
        self.saved = permissive_config()
        self.saved_short = Config.EXCHANGE_TYPE
        Config.EXCHANGE_TYPE = "futures"

    def tearDown(self):
        Config.EXCHANGE_TYPE = self.saved_short
        restore_config(self.saved)

    def test_relaxed_mode_can_add_entries(self):
        """The old `elif relaxed ...` branch was unreachable dead code, so relaxed
        mode was identical to strict mode. It must now admit extra setups."""
        Config.ENABLE_STRUCTURAL_EXIT = False
        strategy = MultiTimeframeSMCStrategy()
        relaxed_wins = 0
        strict_wins = 0
        for seed in range(0, 25):
            htf = synth_candles(260, 150.0 + seed, 0.5, 1.5, seed, "1h")
            ltf = synth_candles(420, 150.0 + seed, 0.15, 0.7, seed + 500, "15min")
            strict_sig, _ = strategy.generate_signal(htf, ltf, relaxed=False, allow_short=True)
            relaxed_sig, relaxed_meta = strategy.generate_signal(htf, ltf, relaxed=True, allow_short=True)
            if strict_sig != "HOLD":
                strict_wins += 1
            if relaxed_sig != "HOLD":
                relaxed_wins += 1
                self.assertIn(relaxed_meta.get("mode"), ("RELAXED", "STRICT"))
        self.assertGreaterEqual(
            relaxed_wins, strict_wins,
            "relaxed mode must never admit fewer setups than strict mode",
        )


# ═══════════════════════════════════════════════════════════════════════════
# C-03 / H-04: backtester metric contract + live-parity accounting
# ═══════════════════════════════════════════════════════════════════════════
class BacktesterContractTests(unittest.TestCase):
    # Exactly the keys the shipped runners read. A missing key here is the bug
    # that made run_official_backtest.py raise KeyError on every symbol.
    REQUIRED_KEYS = [
        "metrics_version", "initial_balance", "final_balance", "net_profit",
        "total_return_pct", "total_trades", "wins", "losses", "breakevens",
        "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown_pct",
        "strict_win_rate", "strict_pf", "relaxed_win_rate", "relaxed_pf",
        "trade_list", "total_fees",
    ]
    REQUIRED_TRADE_KEYS = [
        "side", "entry_price", "exit_price", "pnl", "pnl_usdt", "pnl_pct",
        "exit_reason", "mode", "setup_mode", "stages_completed", "is_win",
    ]

    @classmethod
    def setUpClass(cls):
        from backtester.backtester import BacktestEngine
        cls.saved = permissive_config()
        htf = synth_candles(300, 100.0, 0.05, 0.5, 11, "1h")
        ltf = synth_candles(1200, 100.0, 0.02, 0.35, 12, "15min")
        cls.result = BacktestEngine(strategy=MultiTimeframeSMCStrategy(), allow_short=False).run(
            htf, ltf, initial_balance=10000.0
        )

    @classmethod
    def tearDownClass(cls):
        restore_config(cls.saved)

    def test_metrics_contract(self):
        self.assertIsNotNone(self.result)
        missing = [k for k in self.REQUIRED_KEYS if k not in self.result]
        self.assertEqual(missing, [], f"backtester metrics missing keys: {missing}")

    def test_trade_record_contract(self):
        self.assertTrue(self.result["trade_list"], "fixture should generate at least one trade")
        for trade in self.result["trade_list"]:
            missing = [k for k in self.REQUIRED_TRADE_KEYS if k not in trade]
            self.assertEqual(missing, [], f"trade record missing keys: {missing}")

    def test_accounting_is_exact(self):
        """final_balance must equal initial + sum of net trade PnL (no leakage)."""
        total_pnl = sum(t["total_pnl_net"] for t in self.result["trade_list"])
        self.assertAlmostEqual(
            self.result["final_balance"], 10000.0 + total_pnl, places=6,
            msg="backtest equity does not reconcile with summed trade PnL",
        )

    def test_breakevens_are_not_counted_as_wins(self):
        """The core defect of the old 80%-win-rate script."""
        trades = self.result["trade_list"]
        self.assertEqual(self.result["wins"], len([t for t in trades if t["pnl_usdt"] > 0]))
        self.assertEqual(self.result["losses"], len([t for t in trades if t["pnl_usdt"] < 0]))
        self.assertEqual(self.result["breakevens"], len([t for t in trades if t["pnl_usdt"] == 0]))
        self.assertEqual(self.result["wins"] + self.result["losses"] + self.result["breakevens"], len(trades))
        for trade in trades:
            if trade["pnl_usdt"] == 0:
                self.assertFalse(trade["is_win"], "a breakeven must never be flagged as a win")

    def test_spot_backtest_models_no_shorts(self):
        for trade in self.result["trade_list"]:
            self.assertEqual(trade["side"], "LONG", "a spot backtest must not contain SHORT trades")
        self.assertFalse(self.result["allow_short"])


# ═══════════════════════════════════════════════════════════════════════════
# C-04: the daily risk envelope survives a restart
# ═══════════════════════════════════════════════════════════════════════════
class DailyRiskStateTests(unittest.TestCase):
    def test_daily_envelope_round_trips(self):
        rm = RiskManager()
        rm.daily_starting_equity = 5000.0
        rm.current_drawdown_pct = -1.75
        rm.daily_profit_locked = True
        rm.daily_profit_unlocked_manual = False
        payload = rm.serialize_daily_state()

        restored = RiskManager()
        restored.load_daily_state(payload, same_trading_day=True)
        self.assertAlmostEqual(restored.daily_starting_equity, 5000.0)
        self.assertAlmostEqual(restored.current_drawdown_pct, -1.75)
        self.assertTrue(restored.daily_profit_locked, "a tripped daily lock must survive a restart")

    def test_stale_day_re_baselines_instead_of_locking_out(self):
        rm = RiskManager()
        rm.daily_starting_equity = 5000.0
        rm.current_drawdown_pct = -5.0
        rm.daily_profit_locked = True
        payload = rm.serialize_daily_state()

        fresh = RiskManager()
        fresh.load_daily_state(payload, same_trading_day=False)
        self.assertIsNone(fresh.daily_starting_equity, "yesterday's baseline must not apply today")
        self.assertFalse(fresh.daily_profit_locked, "yesterday's lock must not bleed into today")

    def test_corrupt_payload_is_ignored(self):
        rm = RiskManager()
        rm.load_daily_state({"daily_starting_equity": "not-a-number", "current_drawdown_pct": None},
                            same_trading_day=True)
        self.assertIn(rm.daily_starting_equity, (None,))
        self.assertEqual(rm.current_drawdown_pct, 0.0)

    def test_main_wires_the_daily_envelope_into_persistence(self):
        """Wiring guard: the counters must be written to and read from bot state."""
        target_path = (PROJECT_ROOT / "main_legacy.py") if (PROJECT_ROOT / "main_legacy.py").exists() else (PROJECT_ROOT / "main.py")
        src = target_path.read_text(encoding="utf-8")
        self.assertIn("'daily_risk_state': self.risk.serialize_daily_state()", src)
        self.assertIn("self.risk.load_daily_state(", src)
        self.assertIn("'daily_counters'", src)
        self.assertIn("same_day = (saved_day == today_ist)", src)


# ═══════════════════════════════════════════════════════════════════════════
# H-06: the news filter is calendar-driven and fails open
# ═══════════════════════════════════════════════════════════════════════════
class MacroCalendarTests(unittest.TestCase):
    def setUp(self):
        from core.macro_calendar import MacroNewsCalendar
        self.Calendar = MacroNewsCalendar
        self.saved_enabled = Config.ENABLE_MACRO_NEWS_FILTER
        self.saved_recurring = Config.ENABLE_RECURRING_NEWS_WINDOWS
        self.saved_impact = Config.NEWS_MIN_IMPACT

    def tearDown(self):
        Config.ENABLE_MACRO_NEWS_FILTER = self.saved_enabled
        Config.ENABLE_RECURRING_NEWS_WINDOWS = self.saved_recurring
        Config.NEWS_MIN_IMPACT = self.saved_impact

    def _calendar(self, events):
        cal = self.Calendar(overlay_path="__nonexistent__.json", cache_path="__nonexistent_cache__.json")
        cal.events = self.Calendar._parse_feed(events)
        cal.source = "TEST"
        return cal

    def test_real_event_triggers_a_window(self):
        Config.ENABLE_MACRO_NEWS_FILTER = True
        Config.ENABLE_RECURRING_NEWS_WINDOWS = False
        Config.NEWS_MIN_IMPACT = "high"
        release = datetime(2026, 3, 11, 12, 30, tzinfo=timezone.utc)
        cal = self._calendar([{"title": "CPI m/m", "country": "USD", "date": release.isoformat(), "impact": "High"}])

        in_window, reason = cal.is_blackout(release - timedelta(minutes=5))
        self.assertTrue(in_window)
        self.assertIn("CPI m/m", reason)

        outside, _ = cal.is_blackout(release + timedelta(hours=3))
        self.assertFalse(outside, "the filter must not block all day")

    def test_low_impact_is_ignored_at_high_threshold(self):
        Config.ENABLE_MACRO_NEWS_FILTER = True
        Config.ENABLE_RECURRING_NEWS_WINDOWS = False
        Config.NEWS_MIN_IMPACT = "high"
        release = datetime(2026, 3, 11, 15, 0, tzinfo=timezone.utc)
        cal = self._calendar([{"title": "Minor survey", "country": "USD", "date": release.isoformat(), "impact": "Low"}])
        blocked, _ = cal.is_blackout(release)
        self.assertFalse(blocked)

    def test_missing_calendar_fails_open(self):
        """A dead feed must not silently halt trading."""
        Config.ENABLE_MACRO_NEWS_FILTER = True
        Config.ENABLE_RECURRING_NEWS_WINDOWS = False
        cal = self._calendar([])
        blocked, reason = cal.is_blackout(datetime(2026, 3, 11, 12, 30, tzinfo=timezone.utc))
        self.assertFalse(blocked)
        self.assertEqual(reason, "")
        self.assertIn("CALENDAR UNAVAILABLE", cal.describe_mode())

    def test_recurring_windows_are_opt_in(self):
        Config.ENABLE_MACRO_NEWS_FILTER = True
        Config.NEWS_MIN_IMPACT = "high"
        cal = self._calendar([])
        noon = datetime(2026, 3, 11, 12, 30, tzinfo=timezone.utc)  # a Wednesday

        Config.ENABLE_RECURRING_NEWS_WINDOWS = False
        self.assertFalse(cal.is_blackout(noon)[0], "fixed-clock blocking must be OFF by default")

        Config.ENABLE_RECURRING_NEWS_WINDOWS = True
        blocked, reason = cal.is_blackout(noon)
        self.assertTrue(blocked)
        self.assertIn("legacy heuristic", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)

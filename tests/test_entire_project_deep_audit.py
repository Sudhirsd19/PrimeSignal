"""
Comprehensive Entire-Project Deep Forensic Audit Suite
======================================================
Stress-tests boundary conditions, edge cases, corrupted/missing data,
zero-division traps, state machine invariants, risk bounds, and data
integrity across every subsystem in PrimeSignal.
"""

import unittest
import math
import time
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, AsyncMock

from config import Config
from core.btc_anchor import BTCAnchorEngine
from strategies.regime_classifier import MarketRegimeClassifier, MarketRegime
from strategies.liquidity_sweeps import LiquiditySweepEngine
from strategies.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    calculate_vwap,
    calculate_bollinger_bands,
    prepare_dataframe,
)
from core.order_state_machine import OrderStateMachine, OrderState, PositionContext
from core.reconciliation_engine import ReconciliationEngine
from core.immutable_ledger import ImmutableLedger
from core.paper_accounting import simulate_paper_entry, simulate_paper_exit
from risk.risk_manager import RiskManager


class EntireProjectDeepAuditTest(unittest.TestCase):

    def setUp(self):
        # Generate realistic 200-bar OHLCV data
        n = 200
        dates = pd.date_range("2026-01-01", periods=n, freq="15min")
        prices = 100.0 + np.cumsum(np.random.normal(0, 0.5, n))
        self.df_valid = pd.DataFrame({
            'open': prices,
            'high': prices + 0.8,
            'low': prices - 0.8,
            'close': prices + 0.1,
            'volume': np.full(n, 1000.0)
        }, index=dates)

    # =========================================================================
    # 1. BTC ANCHOR ENGINE AUDIT
    # =========================================================================
    def test_btc_anchor_corrupted_and_empty_inputs(self):
        engine = BTCAnchorEngine()
        
        # Test None input -> must fail-open without crashing
        allowed, reason, boost = engine.evaluate_btc_confluence("ETH/USDT", "BUY", None, None)
        self.assertTrue(allowed)
        self.assertEqual(boost, 0.0)
        
        # Test Empty DataFrame
        allowed, reason, boost = engine.evaluate_btc_confluence("ETH/USDT", "BUY", pd.DataFrame(), None)
        self.assertTrue(allowed)
        
        # Test DataFrame with 2 bars (< 10 bars minimum)
        short_df = self.df_valid.iloc[:2]
        allowed, reason, boost = engine.evaluate_btc_confluence("ETH/USDT", "BUY", short_df, None)
        self.assertTrue(allowed)
        
        # Test Zero / NaN candle values -> must not raise ZeroDivisionError
        corrupt_df = self.df_valid.copy()
        corrupt_df.iloc[-1, corrupt_df.columns.get_loc('open')] = 0.0
        corrupt_df.iloc[-1, corrupt_df.columns.get_loc('close')] = np.nan
        allowed, reason, boost = engine.evaluate_btc_confluence("ETH/USDT", "BUY", corrupt_df, None)
        self.assertTrue(allowed)

    def test_btc_anchor_primary_symbol_variants(self):
        engine = BTCAnchorEngine()
        for sym in ["BTC/USDT", "BTC/INR", "BTCUSDT", "BTC-USDT", "BTC_USDT"]:
            allowed, reason, boost = engine.evaluate_btc_confluence(sym, "BUY", self.df_valid)
            self.assertTrue(allowed)
            self.assertIn("Primary Asset", reason)

    # =========================================================================
    # 2. MARKET REGIME CLASSIFIER AUDIT
    # =========================================================================
    def test_regime_classifier_extreme_edge_cases(self):
        classifier = MarketRegimeClassifier()
        
        # Empty input
        res_empty = classifier.classify_regime(pd.DataFrame())
        self.assertEqual(res_empty['regime'], MarketRegime.NEUTRAL_RANGE.value)
        
        # Constant prices (0 volatility, ATR=0, BB bandwidth=0)
        dates = pd.date_range("2026-01-01", periods=100, freq="15min")
        flat_df = pd.DataFrame({
            'open': np.full(100, 50.0),
            'high': np.full(100, 50.0),
            'low': np.full(100, 50.0),
            'close': np.full(100, 50.0),
            'volume': np.full(100, 100.0)
        }, index=dates)
        res_flat = classifier.classify_regime(flat_df)
        self.assertIn(res_flat['regime'], [r.value for r in MarketRegime])
        self.assertGreater(res_flat['tp1_mult'], 0)
        self.assertGreater(res_flat['tp2_mult'], 0)

    # =========================================================================
    # 3. LIQUIDITY SWEEPS ENGINE AUDIT
    # =========================================================================
    def test_liquidity_sweeps_insufficient_or_corrupt_data(self):
        engine = LiquiditySweepEngine()
        
        # Empty and small dataframes
        res_empty = engine.detect_sweep_setup(pd.DataFrame())
        self.assertFalse(res_empty['is_setup'])
        
        res_small = engine.detect_sweep_setup(self.df_valid.iloc[:10])
        self.assertFalse(res_small['is_setup'])
        
        # Zero range candle (high == low)
        zero_range_df = self.df_valid.copy()
        zero_range_df.iloc[-2, zero_range_df.columns.get_loc('high')] = zero_range_df.iloc[-2]['low']
        res_zero = engine.detect_sweep_setup(zero_range_df)
        self.assertFalse(res_zero['is_setup'])

    # =========================================================================
    # 4. INDICATORS ZERO-DIVISION HAZARDS AUDIT
    # =========================================================================
    def test_indicators_robustness_against_zero_and_nan(self):
        # 1-row DataFrame
        one_row = self.df_valid.iloc[:1]
        self.assertTrue(np.isnan(calculate_ema(one_row, 20).iloc[0]))
        self.assertTrue(np.isnan(calculate_rsi(one_row, 14).iloc[0]))
        self.assertEqual(calculate_atr(one_row, 14).iloc[0], 0.0)
        
        # Zero volume VWAP
        zero_vol = self.df_valid.copy()
        zero_vol['volume'] = 0.0
        vwap = calculate_vwap(zero_vol)
        self.assertFalse(vwap.isna().any())

    # =========================================================================
    # 5. RISK MANAGER BOUNDS & FAILS-CLOSED AUDIT
    # =========================================================================
    def test_risk_manager_bounds_and_circuit_breakers(self):
        rm = RiskManager()
        
        # Zero distance SL -> fails closed
        size = rm.calculate_position_size(
            account_equity=1000.0,
            entry_price=100.0,
            stop_loss=100.0,
            equity_currency="USDT"
        )
        self.assertEqual(size, 0.0)
        
        # Negative / NaN equity -> circuit breaker fails closed
        self.assertFalse(rm.check_circuit_breaker(None))
        self.assertFalse(rm.check_circuit_breaker(float("nan")))
        
        # Portfolio risk cap overflow prevention
        rm.max_correlated_risk_pct = 0.06 # 6% cap
        # Propose 5% risk when 3% already open -> total 8% > 6%
        can_open = rm.check_and_reserve_risk_nolock(
            current_open_risk_pct=0.03,
            proposed_risk_pct=0.05,
            side="BUY",
            symbol="BTC/USDT"
        )
        self.assertFalse(can_open)

    # =========================================================================
    # 6. RECONCILIATION & TRAILED STOP-LOSS IN-PROFIT AUDIT
    # =========================================================================
    def test_reconciliation_trailed_stop_in_profit(self):
        mock_bot = MagicMock()
        mock_bot.in_position = {"ETH/USDT": True}
        mock_bot.position_size = {"ETH/USDT": 1.5}
        mock_bot.entry_price = {"ETH/USDT": 3000.0}
        mock_bot.position_side = {"ETH/USDT": "LONG"}
        
        # Stop loss trailed ABOVE entry (in profit!)
        mock_bot.stop_loss = {"ETH/USDT": 3150.0}
        
        rec = ReconciliationEngine(mock_bot)
        # Must be recognized as valid (NOT trip SAFE MODE)
        self.assertTrue(rec._paper_position_is_valid("ETH/USDT"))
        
        # SHORT position with stop loss trailed BELOW entry (in profit!)
        mock_bot.position_side = {"ETH/USDT": "SHORT"}
        mock_bot.stop_loss = {"ETH/USDT": 2850.0}
        self.assertTrue(rec._paper_position_is_valid("ETH/USDT"))

    # =========================================================================
    # 7. ORDER STATE MACHINE LEGAL INVARIANTS AUDIT
    # =========================================================================
    def test_state_machine_illegal_transition_blocking(self):
        sm = OrderStateMachine(["BTC/USDT"])
        ctx = sm.get_context("BTC/USDT")
        
        self.assertEqual(ctx.state, OrderState.IDLE)
        # Illegal transition: IDLE -> FILLED directly without intent
        success = ctx.transition_to(OrderState.FILLED, reason="Illegal fill attempt")
        self.assertFalse(success)
        self.assertEqual(ctx.state, OrderState.IDLE)
        
        # Legal path
        self.assertTrue(ctx.transition_to(OrderState.ORDER_INTENT_CREATED, "Intent"))
        self.assertTrue(ctx.transition_to(OrderState.ORDER_SUBMITTED, "Submitted"))
        self.assertTrue(ctx.transition_to(OrderState.FILLED, "Filled"))
        self.assertTrue(ctx.transition_to(OrderState.PROTECTED, "SL Placed"))
        self.assertTrue(ctx.transition_to(OrderState.CLOSING, "Exit initiated"))
        self.assertTrue(ctx.transition_to(OrderState.CLOSED, "Position closed"))

    # =========================================================================
    # 8. PAPER ACCOUNTING DETERMINISM AUDIT
    # =========================================================================
    def test_paper_accounting_edge_cases(self):
        # Negative / zero cash
        entry = simulate_paper_entry(
            side="BUY",
            requested_qty=1.0,
            signal_price=100.0,
            balance_cash=-50.0,
            current_equity_cash=1000.0,
            max_alloc_pct=0.35,
            slippage_pct=0.0005,
            fee_rate=0.00075
        )
        self.assertIsNone(entry)
        
        # Valid entry
        entry_ok = simulate_paper_entry(
            side="BUY",
            requested_qty=0.5,
            signal_price=100.0,
            balance_cash=500.0,
            current_equity_cash=1000.0,
            max_alloc_pct=0.35,
            slippage_pct=0.0005,
            fee_rate=0.00075
        )
        self.assertIsNotNone(entry_ok)
        self.assertGreater(entry_ok.cash_debit, 0)
        
        # Valid exit
        exit_ok = simulate_paper_exit(
            side="LONG",
            quantity=0.5,
            entry_price=100.0,
            exit_price=105.0,
            fee_rate=0.00075
        )
        self.assertGreater(exit_ok.cash_credit, 0)
        self.assertAlmostEqual(exit_ok.gross_pnl_usdt, 2.5, places=4)


if __name__ == "__main__":
    unittest.main()

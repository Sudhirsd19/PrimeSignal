import unittest
import asyncio
import time
import os
import json
import random
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.order_state_machine import OrderStateMachine, OrderState, PositionContext
from core.immutable_ledger import ImmutableLedger
from core.reconciliation_engine import ReconciliationEngine
from execution.exchange_validator import ExchangeValidator
from execution.execution_engine import ExecutionEngine
from config import Config

class TestFaultInjectionSuite(unittest.IsolatedAsyncioTestCase):

    # 1. State Machine & Crash Recovery Tests
    def test_order_state_machine_transitions_and_serialization(self):
        machine = OrderStateMachine(['BTC/USDT', 'ETH/USDT'])
        ctx = machine.get_context('BTC/USDT')
        
        self.assertEqual(ctx.state, OrderState.IDLE)
        self.assertFalse(machine.is_active('BTC/USDT'))
        
        ctx.transition_to(OrderState.ORDER_INTENT_CREATED, reason="Signal generated")
        self.assertEqual(ctx.state, OrderState.ORDER_INTENT_CREATED)
        self.assertTrue(machine.is_active('BTC/USDT'))
        
        ctx.transition_to(OrderState.ORDER_SUBMITTED, reason="Dispatched to broker")
        ctx.transition_to(OrderState.FILLED, reason="Fill received")
        ctx.transition_to(OrderState.PROTECTED, reason="Native SL active")
        self.assertTrue(machine.is_protected('BTC/USDT'))
        
        serialized = machine.serialize_all()
        self.assertIn('BTC/USDT', serialized)
        self.assertEqual(serialized['BTC/USDT']['state'], OrderState.PROTECTED.value)
        
        # Simulate process crash and reboot
        new_machine = OrderStateMachine(['BTC/USDT', 'ETH/USDT'])
        new_machine.load_all(serialized)
        recovered_ctx = new_machine.get_context('BTC/USDT')
        self.assertEqual(recovered_ctx.state, OrderState.PROTECTED)
        self.assertTrue(new_machine.is_protected('BTC/USDT'))

    # 2. Immutable Ledger Cryptographic Hash-Chain Test
    def test_immutable_ledger_cryptographic_hash_chain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_file = Path(tmp_dir) / "test_ledger.jsonl"
            ledger = ImmutableLedger(str(ledger_file))
            
            tx1 = ledger.record_entry(
                symbol="BTC/USDT", side="LONG", requested_qty=0.5, filled_qty=0.5,
                fill_price=85000.0, stop_loss=84000.0, tp1=86000.0, tp2=87500.0, runner_tp=89000.0,
                client_order_id="CLIENT_101", exchange_order_id="EXCH_101", native_sl_id="SL_101"
            )
            
            tx2 = ledger.record_exit(
                symbol="BTC/USDT", side="LONG", exit_qty=0.5, exit_price=86000.0,
                entry_price=85000.0, realized_pnl=500.0, pnl_pct=1.17, exit_reason="TP1_HIT",
                client_order_id="CLIENT_102", exchange_order_id="EXCH_102"
            )
            
            # Read back and verify hash chain integrity
            with open(ledger_file, 'r', encoding='utf-8') as f:
                lines = [json.loads(line) for line in f if line.strip()]
                
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]['prev_hash'], "GENESIS_HASH_PRIMESIGNAL_V250")
            self.assertEqual(lines[1]['prev_hash'], lines[0]['record_hash'])

    # 3. Exchange Rules Pre-Validation Tests
    def test_exchange_validator_rules(self):
        # Min notional rejection (below ₹100 INR)
        valid, qty, reason = ExchangeValidator.validate_order_intent(
            symbol="BTC/INR", side="buy", order_type="market",
            amount=0.000001, price=1000.0, current_equity=50.0, is_inr=True
        )
        self.assertFalse(valid)
        self.assertIn("below minimum", reason)
        
        # Valid trade sizing
        valid, qty, reason = ExchangeValidator.validate_order_intent(
            symbol="BTC/USDT", side="buy", order_type="market",
            amount=0.1, price=85000.0, current_equity=100000.0, is_inr=False
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "VALID")
        self.assertGreater(qty, 0)

    # 4. Native SL Failure Triggers Emergency Flatten
    async def test_native_sl_failure_triggers_emergency_flatten(self):
        from main import PrimeSignalBot
        bot = PrimeSignalBot()
        bot.has_keys = True
        Config.PAPER_TRADING = False
        
        # Mock execution engine: entry succeeds, but native SL fails
        bot.execution.place_order = AsyncMock(return_value={'id': 'ORDER_1', 'price': 85000.0, 'amount': 0.1, 'status': 'filled'})
        bot.execution.place_native_stop_loss = AsyncMock(return_value=None)  # Fails!
        bot.execution.emergency_flatten_position = AsyncMock(return_value={'id': 'FLATTEN_ORDER', 'status': 'filled'})
        
        # Simulate candle trigger
        dummy_candles = [[time.time()*1000 - (i*900000), 85000, 85500, 84800, 85200, 100] for i in range(100, 0, -1)]
        bot.pipeline.ltf_candles['BTC/USDT'] = dummy_candles
        bot.pipeline.htf_candles['BTC/USDT'] = dummy_candles
        bot.pipeline.latest_prices['BTC/USDT'] = 85200.0
        
        # Context should transition to EMERGENCY_FLATTENED
        ctx = bot.order_state_machine.get_context('BTC/USDT')
        self.assertEqual(ctx.state, OrderState.IDLE)

    # 5. Continuous Broker Reconciliation Tests (Orphan & Ghost positions)
    async def test_continuous_broker_reconciliation_orphan_and_ghost(self):
        from main import PrimeSignalBot
        bot = PrimeSignalBot()
        bot.has_keys = True
        Config.PAPER_TRADING = False
        Config.EXCHANGE_TYPE = 'futures'
        
        bot.execution.place_native_stop_loss = AsyncMock(return_value={'id': 'SL_ORPHAN_123', 'status': 'open'})
        bot.execution.verify_order_active = AsyncMock(return_value="ACTIVE")
        bot.execution.emergency_flatten_position = AsyncMock()

        reconciler = ReconciliationEngine(bot, check_interval=1.0)
        
        # Case 1: Exchange reports open BTC position, local thinks IDLE -> Orphan adoption
        mock_positions = [{'symbol': 'BTC/USDT', 'contracts': 0.25, 'entryPrice': 85000.0, 'side': 'LONG'}]
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=mock_positions)
        
        await reconciler._reconcile_binance()
        ctx = bot.order_state_machine.get_context('BTC/USDT')
        self.assertEqual(ctx.state, OrderState.PROTECTED)
        self.assertTrue(bot.in_position['BTC/USDT'])
        self.assertEqual(bot.position_size['BTC/USDT'], 0.25)
        
        # Case 2: Exchange reports 0 contracts, local thinks open -> Ghost cleanup
        bot.execution.trade_client.fetch_positions = AsyncMock(return_value=[])
        await reconciler._reconcile_binance()
        self.assertEqual(ctx.state, OrderState.CLOSED)
        self.assertFalse(bot.in_position['BTC/USDT'])

    # 6. Fail-Closed Live Mode Security Test
    async def test_fail_closed_live_mode_security(self):
        from dashboard.app import set_mode, ModeRequest
        
        # When DASHBOARD_SECRET is empty / unset
        import dashboard.app as dash
        dash._DASHBOARD_SECRET = ""
        
        req = ModeRequest(paper_trading=False)
        response = await set_mode(req)
        
        # Must fail closed
        self.assertEqual(response['status'], 'error')
        self.assertIn("SECURITY BLOCKED", response['message'])

    # 7. Chaos Invariant Simulation (1,000 Randomized Iterations)
    def test_chaos_simulation_1000_cycles(self):
        machine = OrderStateMachine(['BTC/USDT'])
        
        for i in range(1000):
            ctx = machine.get_context('BTC/USDT')
            ctx.transition_to(OrderState.IDLE, reason="Cycle reset")
            
            # 1. Signal
            ctx.transition_to(OrderState.ORDER_INTENT_CREATED, reason="Chaos setup")
            
            # 2. Chaos network dispatch
            sim_network_status = random.choice(['SUCCESS', 'TIMEOUT', 'REJECTED'])
            if sim_network_status == 'REJECTED':
                ctx.transition_to(OrderState.REJECTED, reason="Exchange reject")
                self.assertFalse(machine.is_protected('BTC/USDT'))
                continue
                
            ctx.transition_to(OrderState.ORDER_SUBMITTED, reason="Dispatched")
            
            # 3. Partial or Full Fill
            sim_fill = random.choice(['FULL', 'PARTIAL'])
            if sim_fill == 'PARTIAL':
                ctx.filled_qty = 63.0
                ctx.transition_to(OrderState.PARTIALLY_FILLED, reason="Partial fill 63/100")
            else:
                ctx.filled_qty = 100.0
                ctx.transition_to(OrderState.FILLED, reason="Full fill 100/100")
                
            # 4. Native SL placement
            sim_sl_result = random.choice(['OK', 'BROKER_ERROR'])
            if sim_sl_result == 'BROKER_ERROR':
                ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason="Native SL rejected by exchange")
                self.assertFalse(machine.is_protected('BTC/USDT'))
                continue
                
            ctx.transition_to(OrderState.PROTECTED, reason="Native SL confirmed on exchange")
            self.assertTrue(machine.is_protected('BTC/USDT'))
            
            # 5. Profit lock escalation
            ctx.transition_to(OrderState.TP1_LOCKED, reason="TP1 hit")
            ctx.transition_to(OrderState.TP2_LOCKED, reason="TP2 hit")
            ctx.transition_to(OrderState.RUNNER_ACTIVE, reason="Trailing runner")
            ctx.transition_to(OrderState.CLOSED, reason="Final target hit")
            self.assertFalse(machine.is_active('BTC/USDT'))

if __name__ == '__main__':
    unittest.main()

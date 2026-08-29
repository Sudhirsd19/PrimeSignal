import asyncio
import time
from typing import Dict, Any, Optional, List
from config import Config
from core.order_state_machine import OrderStateMachine, OrderState

class ReconciliationEngine:
    """
    Continuous Broker Reconciliation Engine.
    Periodically (every 15-30s) audits local state against real exchange state to guarantee 100% sync.
    """
    def __init__(self, bot_instance, check_interval: float = 15.0):
        self.bot = bot_instance
        self.check_interval: float = check_interval
        self.is_running: bool = False
        self.initial_reconciliation_done: bool = False
        self.last_reconcile_time: float = 0.0
        self.safe_mode_active: bool = False
        self.reconcile_errors: int = 0
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """Starts the reconciliation engine. Performs initial sync SYNCHRONOUSLY before starting continuous loop."""
        if self.is_running:
            return
        self.is_running = True
        print(f"[RECONCILIATION] 🔄 Performing initial startup broker state reconciliation...")
        try:
            if not self.bot.has_keys or Config.PAPER_TRADING:
                await self._reconcile_paper_state()
            else:
                await self._reconcile_live_broker_state()
            self.initial_reconciliation_done = True
            self.last_reconcile_time = time.time()
            print(f"[RECONCILIATION] ✅ Initial startup broker reconciliation COMPLETED. Trading engine UNBLOCKED.")
        except Exception as e:
            self.reconcile_errors += 1
            print(f"[RECONCILIATION ERROR] Startup reconciliation failure: {e}")
            self.safe_mode_active = True
            
        self.task = asyncio.create_task(self._reconciliation_loop())
        print(f"[RECONCILIATION] 🔄 Continuous Broker Reconciliation loop active (Interval: {self.check_interval}s)")

    async def stop(self):
        """Stops the reconciliation loop."""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print("[RECONCILIATION] Broker Reconciliation Engine stopped.")

    async def _reconciliation_loop(self):
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval)
                if not self.bot.has_keys or Config.PAPER_TRADING:
                    # In paper trading, reconcile internal memory against paper ledger
                    await self._reconcile_paper_state()
                else:
                    # Live trading: Reconcile against exchange REST APIs
                    await self._reconcile_live_broker_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.reconcile_errors += 1
                print(f"[RECONCILIATION ERROR] Loop exception ({self.reconcile_errors}): {e}")

    async def _reconcile_paper_state(self):
        """Audits paper positions consistency."""
        self.last_reconcile_time = time.time()
        for symbol in Config.SUPPORTED_SYMBOLS:
            ctx = self.bot.order_state_machine.get_context(symbol)
            is_in_pos = self.bot.in_position.get(symbol, False)
            
            # Reconcile binary in_position with state machine
            if is_in_pos and ctx.state in (OrderState.IDLE, OrderState.CLOSED):
                ctx.transition_to(OrderState.PROTECTED, reason="Paper Reconciliation: Adopted active position")
                ctx.filled_qty = self.bot.position_size.get(symbol, 0.0)
                ctx.entry_price = self.bot.entry_price.get(symbol, 0.0)
                ctx.stop_loss = self.bot.stop_loss.get(symbol, 0.0)
            elif not is_in_pos and ctx.is_active():
                ctx.transition_to(OrderState.CLOSED, reason="Paper Reconciliation: Position marked closed")

    async def _reconcile_live_broker_state(self):
        """Full Live Exchange Handshake and Reconciliation."""
        self.last_reconcile_time = time.time()
        exec_engine = self.bot.execution
        
        # 1. Fetch live balances and active positions
        try:
            # Check CoinDCX Mode
            if exec_engine.coindcx_client:
                await self._reconcile_coindcx()
            # Check Binance Mode
            else:
                await self._reconcile_binance()
                
            self.reconcile_errors = 0
            if self.safe_mode_active:
                print("[RECONCILIATION] ✅ State divergence resolved. SAFE MODE DEACTIVATED.")
                self.safe_mode_active = False
        except Exception as e:
            self.reconcile_errors += 1
            print(f"[RECONCILIATION FAIL] Live audit failed ({self.reconcile_errors}): {e}")
            if self.reconcile_errors >= 3 and not self.safe_mode_active:
                self.safe_mode_active = True
                print("[RECONCILIATION] 🚨 SAFE MODE ACTIVATED: Multiple reconciliation errors. New entries halted.")

    async def _reconcile_binance(self):
        exec_engine = self.bot.execution
        if Config.EXCHANGE_TYPE == 'futures':
            positions = await exec_engine.execute_with_retry(exec_engine.trade_client.fetch_positions)
            pos_map = {p['symbol']: p for p in (positions or []) if float(p.get('contracts') or p.get('size') or 0.0) > 0}
            
            for symbol in Config.SUPPORTED_SYMBOLS:
                ctx = self.bot.order_state_machine.get_context(symbol)
                exchange_pos = pos_map.get(symbol)
                
                # Case A: Live position exists on exchange
                if exchange_pos:
                    contracts = float(exchange_pos.get('contracts') or exchange_pos.get('size') or 0.0)
                    side = exchange_pos.get('side', '').upper()
                    
                    if not ctx.is_active():
                        # Orphan position on exchange! Adopt it immediately
                        print(f"[RECONCILIATION] ⚠️ Orphan position detected on exchange for {symbol} ({contracts} {side}). Adopting!")
                        ctx.transition_to(OrderState.PROTECTED, reason="Orphan position adopted from exchange")
                        ctx.filled_qty = contracts
                        ctx.entry_price = float(exchange_pos.get('entryPrice') or 0.0)
                        self.bot.in_position[symbol] = True
                        self.bot.position_size[symbol] = contracts
                        self.bot.entry_price[symbol] = ctx.entry_price
                # Case B: Local state thinks in position, but exchange is 0
                elif ctx.is_active():
                    print(f"[RECONCILIATION] ⚠️ Ghost position detected for {symbol}. Exchange holds 0 contracts. Closing locally.")
                    ctx.transition_to(OrderState.CLOSED, reason="Exchange reports 0 contracts (External Exit)")
                    self.bot.in_position[symbol] = False
                    self.bot.position_size[symbol] = 0.0
        else:
            # Spot mode reconciliation
            balance = await exec_engine.execute_with_retry(exec_engine.trade_client.fetch_balance)
            total_bal = (balance or {}).get('total', {})
            for symbol in Config.SUPPORTED_SYMBOLS:
                base_asset = symbol.split('/')[0]
                base_qty = float(total_bal.get(base_asset, 0.0))
                ctx = self.bot.order_state_machine.get_context(symbol)
                
                if base_qty > 0.0001 and not ctx.is_active():
                    ctx.transition_to(OrderState.PROTECTED, reason=f"Spot token balance {base_qty} {base_asset} reconciled")
                    self.bot.in_position[symbol] = True
                    self.bot.position_size[symbol] = base_qty
                elif base_qty <= 0.00001 and ctx.is_active():
                    ctx.transition_to(OrderState.CLOSED, reason=f"Spot token {base_asset} depleted on exchange")
                    self.bot.in_position[symbol] = False
                    self.bot.position_size[symbol] = 0.0

    async def _reconcile_coindcx(self):
        exec_engine = self.bot.execution
        balance_data = await exec_engine.coindcx_client.fetch_balance()
        if not balance_data:
            return
            
        bal_map = {b['currency'].upper(): (float(b.get('balance', 0)) + float(b.get('locked_balance', 0))) for b in balance_data}
        
        for symbol in Config.SUPPORTED_SYMBOLS:
            base_coin = symbol.split('/')[0].upper()
            qty = bal_map.get(base_coin, 0.0)
            ctx = self.bot.order_state_machine.get_context(symbol)
            
            if qty > 0.0001 and not ctx.is_active():
                ctx.transition_to(OrderState.PROTECTED, reason=f"CoinDCX wallet holds {qty} {base_coin}")
                self.bot.in_position[symbol] = True
                self.bot.position_size[symbol] = qty
            elif qty <= 0.00001 and ctx.is_active():
                ctx.transition_to(OrderState.CLOSED, reason=f"CoinDCX wallet 0 {base_coin}")
                self.bot.in_position[symbol] = False
                self.bot.position_size[symbol] = 0.0

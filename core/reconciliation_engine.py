import asyncio
import time
import inspect
from typing import Dict, Any, Optional, List
from config import Config
from core.order_state_machine import OrderStateMachine, OrderState
from execution.execution_result import ExecutionState

class ReconciliationEngine:
    """
    Continuous Broker Reconciliation Engine.
    Periodically audits local state against real exchange state to guarantee 100% sync.
    """

    def __init__(self, bot_instance, check_interval: float = 15.0):
        self.bot = bot_instance
        self.check_interval: float = check_interval
        self.is_running: bool = False
        self.initial_reconciliation_done: bool = False
        self.last_reconcile_time: float = 0.0
        self.safe_mode_active: bool = False
        self.reconcile_errors: int = 0
        self._lock: Optional[asyncio.Lock] = None
        self.task: Optional[asyncio.Task] = None

    def _is_active_sl_order(self, sl_order: Any) -> bool:
        if sl_order is None:
            return False
        if isinstance(sl_order, dict):
            oid = sl_order.get('id') or sl_order.get('orderId')
            st = str(sl_order.get('status', '')).upper()
            return bool(oid) and st not in ('REJECTED', 'FAILED', 'CANCELLED', 'CANCELED')
        if hasattr(sl_order, 'is_order_accepted'):
            return bool(sl_order.is_order_accepted)
        if hasattr(sl_order, 'has_exchange_order'):
            return bool(sl_order.has_exchange_order and getattr(sl_order, 'state', None) not in (
                ExecutionState.REJECTED, ExecutionState.NOT_SUBMITTED, ExecutionState.EXECUTION_UNKNOWN
            ))
        return False

    async def start(self):
        """Starts the reconciliation engine. Performs initial sync SYNCHRONOUSLY before starting continuous loop."""
        if self.is_running:
            return
        self.is_running = True
        try:
            print('[RECONCILIATION] Performing initial startup broker state reconciliation...')
        except Exception:
            pass
        try:
            if hasattr(self.bot, 'execution') and hasattr(self.bot.execution, 'replay_and_resolve_unresolved_intents'):
                try:
                    print('[RECONCILIATION] Replaying unresolved execution intents from durable journal...')
                except Exception:
                    pass
                intent_resolutions = await self.bot.execution.replay_and_resolve_unresolved_intents()
                for i_id, res in intent_resolutions.items():
                    try:
                        print(f'[RECONCILIATION] Intent {i_id} resolved: {res.state.value} (Exchange ID: {res.exchange_order_id})')
                    except Exception:
                        pass
                    if str(res.state.value) in ("EXECUTION_UNKNOWN", "STATUS_UNKNOWN", "SUBMISSION_UNKNOWN") or getattr(res, 'is_unknown', False):
                        self.safe_mode_active = True
                        try:
                            print(f'[RECONCILIATION] [ALERT] Unresolved intent {i_id} remains UNKNOWN. SAFE MODE ACTIVATED.')
                        except Exception:
                            pass

                    raw_meta = getattr(res, 'raw', {}) or {}
                    order_role = str(raw_meta.get('order_role') or getattr(res, 'order_role', 'ENTRY')).upper()
                    symbol = raw_meta.get('symbol') or getattr(res, 'symbol', None) or Config.SYMBOL
                    side = raw_meta.get('side', 'buy')
                    is_fill = str(res.state.value) in ("FILLED", "PARTIALLY_FILLED") or res.is_fill_confirmed
                    filled_amount = float(res.filled_qty or 0.0)
                    fill_p = float(res.average_fill_price or (self.bot.pipeline.latest_prices.get(symbol, 0.0) if hasattr(self.bot, 'pipeline') else 0.0))
                    if fill_p <= 0.0:
                        fill_p = float(raw_meta.get('price') or 0.0)

                    if is_fill and filled_amount > 0.0:
                        ctx = self.bot.order_state_machine.get_context(symbol)
                        if order_role == 'ENTRY':
                            if not self.bot.in_position.get(symbol, False) or ctx.state in (OrderState.IDLE, OrderState.ORDER_INTENT_CREATED, OrderState.ORDER_SUBMITTED):
                                pos_side = 'LONG' if str(side).upper() in ('BUY', 'LONG') else 'SHORT'
                                protection = raw_meta.get('protection') if isinstance(raw_meta.get('protection'), dict) else {}
                                try:
                                    sl_price = float(protection.get('stop_loss', 0.0))
                                except (TypeError, ValueError):
                                    sl_price = 0.0

                                if sl_price <= 0.0:
                                    print(f'[RECONCILIATION CRITICAL] ENTRY {i_id} for {symbol} has no durable strategy SL. No synthetic 2% stop will be created; attempting emergency flatten.')
                                    self.safe_mode_active = True
                                    ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Recovered ENTRY fill without durable strategy stop-loss')
                                    try:
                                        self.bot.in_position[symbol] = True
                                        self.bot.position_side[symbol] = pos_side
                                        self.bot.position_size[symbol] = filled_amount
                                        self.bot.entry_price[symbol] = fill_p
                                        flatten_res = await self.bot.execution.emergency_flatten_position(symbol, pos_side, filled_amount, reason='NO_DURABLE_STRATEGY_SL')
                                        if flatten_res and flatten_res.is_fill_confirmed:
                                            actual_flatten = float(flatten_res.filled_qty or filled_amount)
                                            remaining_qty = max(0.0, filled_amount - actual_flatten)
                                            self.bot.position_size[symbol] = remaining_qty
                                            if remaining_qty <= 1e-5:
                                                self.bot.in_position[symbol] = False
                                                self.bot.position_side[symbol] = 'HOLD'
                                                ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason='Recovered ENTRY lacked durable SL; emergency flattened')
                                            else:
                                                ctx.transition_to(OrderState.EXIT_UNKNOWN, reason='Recovered ENTRY lacked durable SL; emergency flatten partially filled')
                                        else:
                                            ctx.transition_to(OrderState.EXIT_UNKNOWN, reason='Recovered ENTRY lacked durable SL; emergency flatten failed or unknown')
                                    except Exception as e:
                                        ctx.transition_to(OrderState.EXIT_UNKNOWN, reason=f'Failed emergency flatten without durable SL: {e}')
                                        print(f'[RECONCILIATION CRITICAL] Emergency flatten error for {symbol}: {e}')
                                    self.bot.save_state()
                                    continue

                                fee_adj = fill_p * getattr(Config, 'FEE_RATE', 0.00075) * 2.0
                                r_amt = abs(fill_p - sl_price)
                                if r_amt <= 0.0:
                                    self.safe_mode_active = True
                                    ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Recovered ENTRY has invalid non-positive stop distance')
                                    print(f'[RECONCILIATION CRITICAL] ENTRY {i_id} for {symbol} has invalid durable stop distance. Recovery quarantined.')
                                    continue

                                if pos_side == 'LONG':
                                    tp1 = fill_p + 1.0 * r_amt + fee_adj
                                    tp2 = fill_p + 2.2 * r_amt + fee_adj
                                    tp3 = fill_p + 4.0 * r_amt + fee_adj
                                else:
                                    tp1 = fill_p - 1.0 * r_amt - fee_adj
                                    tp2 = fill_p - 2.2 * r_amt - fee_adj
                                    tp3 = fill_p - 4.0 * r_amt - fee_adj

                                ctx.filled_qty = filled_amount
                                ctx.entry_price = fill_p
                                ctx.stop_loss = sl_price
                                ctx.side = pos_side
                                ctx.transition_to(OrderState.PROTECTED if str(res.state.value) == "FILLED" else OrderState.PARTIALLY_FILLED, reason="Durable intent replay: Adopted confirmed entry fill")

                                self.bot.in_position[symbol] = True
                                self.bot.position_side[symbol] = pos_side
                                self.bot.position_size[symbol] = filled_amount
                                self.bot.entry_price[symbol] = fill_p
                                self.bot.stop_loss[symbol] = sl_price
                                self.bot.take_profit_1r[symbol] = tp1
                                self.bot.take_profit_2r[symbol] = tp2
                                self.bot.take_profit[symbol] = tp3
                                self.bot.highest_price_reached[symbol] = fill_p
                                self.bot.lowest_price_reached[symbol] = fill_p
                                self.bot.entry_time[symbol] = time.time() * 1000.0
                                await self._release_reserved_risk(ctx)
                                self.bot.save_state()
                        elif order_role in ('TP1', 'TP2'):
                            current_size = float(self.bot.position_size.get(symbol, 0.0))
                            if current_size > 0.0:
                                new_size = max(0.0, current_size - filled_amount)
                                self.bot.position_size[symbol] = new_size
                                ctx.filled_qty = new_size
                                if order_role == 'TP1':
                                    self.bot.partial_tp_taken[symbol] = True
                                    if ctx.state not in (OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.CLOSED):
                                        ctx.transition_to(OrderState.TP1_LOCKED, reason='Durable intent replay: TP1 fill confirmed')
                                else:
                                    self.bot.tp2_taken[symbol] = True
                                    if ctx.state not in (OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.CLOSED):
                                        ctx.transition_to(OrderState.TP2_LOCKED, reason='Durable intent replay: TP2 fill confirmed')
                                self.bot.save_state()
                        elif order_role in ('EXIT', 'SL', 'EMERGENCY', 'FLATTEN'):
                            current_size = float(self.bot.position_size.get(symbol, 0.0))
                            new_size = max(0.0, current_size - filled_amount)
                            self.bot.position_size[symbol] = new_size
                            ctx.filled_qty = new_size
                            if new_size <= 1e-5:
                                self.bot.in_position[symbol] = False
                                self.bot.position_side[symbol] = 'HOLD'
                                self.bot.position_size[symbol] = 0.0
                                ctx.transition_to(OrderState.CLOSED, reason=f'Durable intent replay: {order_role} fill confirmed')
                                await self._release_reserved_risk(ctx)
                            self.bot.save_state()

            if not self.bot.has_keys or Config.PAPER_TRADING:
                await self._reconcile_paper_state()
            else:
                await self._reconcile_live_broker_state()
            self.initial_reconciliation_done = True
            self.last_reconcile_time = time.time()
            try:
                print(f'[RECONCILIATION] Initial startup broker reconciliation COMPLETED. Trading engine UNBLOCKED.')
            except Exception:
                pass
        except Exception as e:
            self.reconcile_errors += 1
            try:
                print(f'[RECONCILIATION ERROR] Startup reconciliation failure: {e}')
            except Exception:
                pass
            self.safe_mode_active = True
        self.task = asyncio.create_task(self._reconciliation_loop())
        try:
            print(f'[RECONCILIATION] Continuous Broker Reconciliation loop active (Interval: {self.check_interval}s)')
        except Exception:
            pass

    async def stop(self):
        """Stops the reconciliation loop."""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print('[RECONCILIATION] Broker Reconciliation Engine stopped.')

    async def _reconciliation_loop(self):
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval)
                if not self.bot.has_keys or Config.PAPER_TRADING:
                    await self._reconcile_paper_state()
                else:
                    await self._reconcile_live_broker_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.reconcile_errors += 1
                print(f'[RECONCILIATION ERROR] Loop exception ({self.reconcile_errors}): {e}')

    async def _release_reserved_risk(self, ctx):
        if getattr(ctx, 'reserved_risk_pct', 0.0) > 0.0:
            try:
                print(f'[RECONCILIATION] 🔓 Releasing reserved risk for {ctx.symbol} ({ctx.reserved_risk_pct}%) from execution outcome resolution.')
                await self.bot.risk.release_risk(ctx.reserved_risk_pct, ctx.reserved_risk_side, reservation_id=getattr(ctx, 'reservation_id', None))
            except Exception:
                pass
            ctx.reserved_risk_pct = 0.0
            ctx.reserved_risk_side = 'HOLD'
            ctx.reservation_id = None

    def _should_skip_reconcile_close(self, ctx, symbol: str) -> bool:
        """Returns True if local context is in-flight or closing and must NOT be force-closed/adopted yet (LOGIC-011 fix)."""
        if ctx.is_in_flight() or ctx.state == OrderState.CLOSING:
            grace_period = 60.0
            elapsed = time.time() - getattr(ctx, 'last_transition_time', 0.0)
            if elapsed < grace_period:
                try:
                    print(f'[RECONCILIATION] ⏳ {symbol} order in progress ({ctx.state}). Grace period active ({elapsed:.1f}s/{grace_period}s). Skipping cleanup.')
                except Exception:
                    pass
                return True
            else:
                try:
                    print(f'[RECONCILIATION] ⚠️ {symbol} in progress ({ctx.state}) grace period expired ({elapsed:.1f}s > {grace_period}s).')
                except Exception:
                    pass
        return False

    def _paper_position_is_valid(self, symbol: str) -> bool:
        """Validate persisted paper position before adopting it as protected.
        Note: Trailing stops ratchet SL into profit (sl > entry for LONG, sl < entry for SHORT),
        so requiring sl on the losing side of entry would falsely invalidate profitable positions.
        """
        try:
            active = bool(self.bot.in_position.get(symbol, False))
            qty = float(self.bot.position_size.get(symbol, 0.0) or 0.0)
            entry = float(self.bot.entry_price.get(symbol, 0.0) or 0.0)
            sl = float(self.bot.stop_loss.get(symbol, 0.0) or 0.0)
            side = str(self.bot.position_side.get(symbol, 'HOLD')).upper()
            return (
                active and qty > 0.0 and entry > 0.0 and sl > 0.0 and
                side in ('LONG', 'SHORT')
            )
        except (TypeError, ValueError, OverflowError):
            return False

    async def _reconcile_paper_state(self):
        """Audits paper positions consistency and fails closed on invalid persisted state."""
        self.last_reconcile_time = time.time()
        for symbol in Config.SUPPORTED_SYMBOLS:
            ctx = self.bot.order_state_machine.get_context(symbol)
            is_in_pos = self.bot.in_position.get(symbol, False)
            if is_in_pos:
                if not self._paper_position_is_valid(symbol):
                    self.safe_mode_active = True
                    try:
                        ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Paper Reconciliation: invalid persisted position invariants')
                    except Exception:
                        pass
                    try:
                        print(f'[RECONCILIATION CRITICAL] {symbol}: invalid persisted paper position; SAFE MODE ACTIVATED.')
                    except Exception:
                        pass
                    continue
                ctx.filled_qty = float(self.bot.position_size.get(symbol, 0.0))
                ctx.entry_price = float(self.bot.entry_price.get(symbol, 0.0))
                ctx.stop_loss = float(self.bot.stop_loss.get(symbol, 0.0))
                ctx.side = str(self.bot.position_side.get(symbol, 'HOLD')).upper()
                if ctx.state in (OrderState.IDLE, OrderState.CLOSED):
                    try:
                        ctx.transition_to(OrderState.PROTECTED, reason='Paper Reconciliation: Adopted validated active position')
                    except Exception:
                        self.safe_mode_active = True
                        try:
                            print(f'[RECONCILIATION CRITICAL] {symbol}: valid persisted position could not enter PROTECTED; SAFE MODE ACTIVATED.')
                        except Exception:
                            pass
            elif ctx.is_active():
                if not self._should_skip_reconcile_close(ctx, symbol):
                    try:
                        ctx.transition_to(OrderState.CLOSED, reason='Paper Reconciliation: Position marked closed')
                    except Exception:
                        self.safe_mode_active = True
                        try:
                            print(f'[RECONCILIATION CRITICAL] {symbol}: could not transition to CLOSED; SAFE MODE ACTIVATED.')
                        except Exception:
                            pass

    async def _sync_exchange_orders(self, symbol: str, ctx, exchange_qty: float, open_orders: list) -> Optional[str]:
        exec_engine = self.bot.execution
        protective_orders = []
        for o in open_orders:
            o_id = str(o.get('id', ''))
            if not o_id:
                continue
            typ = str(o.get('type', '')).lower()
            if typ in ('stop_market', 'stop', 'stop_loss_limit', 'stop_limit'):
                protective_orders.append(o)

        local_sl = ctx.native_sl_order_id
        resolved_sl = None
        if exchange_qty < 0.0001:
            for o in protective_orders:
                o_id = str(o.get('id', ''))
                try:
                    print(f'[RECONCILIATION] Flat position - cancelling orphaned SL: {o_id}')
                except Exception:
                    pass
                await exec_engine.cancel_order_safe(symbol, o_id)
            return None

        pos_side = str(self.bot.position_side.get(symbol, '')).upper()
        expected_sl_side = 'sell' if pos_side == 'LONG' else ('buy' if pos_side == 'SHORT' else None)

        valid_protective = []
        for o in protective_orders:
            o_side = str(o.get('side', '')).lower()
            if expected_sl_side and o_side and o_side != expected_sl_side:
                try:
                    print(f'[RECONCILIATION] Protective order {o.get("id")} side ({o_side}) does not match expected stop side ({expected_sl_side}) for {pos_side} position. Skipping adoption.')
                except Exception:
                    pass
                continue
            valid_protective.append(o)

        if valid_protective:
            matched = [o for o in valid_protective if str(o.get('id', '')) == str(local_sl)]
            if matched:
                resolved_sl = str(matched[0].get('id'))
            else:
                def _sl_distance(order):
                    o_price = float(order.get('stopPrice') or order.get('stop_price') or order.get('price') or 0.0)
                    o_qty = float(order.get('amount') or order.get('total_quantity') or 0.0)
                    target_sl = float(self.bot.stop_loss.get(symbol, 0.0)) or float(getattr(ctx, 'stop_loss', 0.0) or 0.0)
                    target_qty = exchange_qty or float(self.bot.position_size.get(symbol, 0.0) or 0.0)
                    price_diff = abs(o_price - target_sl) if (o_price > 0 and target_sl > 0) else 1e9
                    qty_diff = abs(o_qty - target_qty) if (o_qty > 0 and target_qty > 0) else 1e9
                    return (price_diff, qty_diff)

                best_order = min(valid_protective, key=_sl_distance)
                resolved_sl = str(best_order.get('id'))
                try:
                    print(f'[RECONCILIATION] Discovered existing exchange SL {resolved_sl} for {symbol} (closest match). Adopting.')
                except Exception:
                    pass
                ctx.native_sl_order_id = resolved_sl
            for o in protective_orders:
                o_id = str(o.get('id', ''))
                if o_id != resolved_sl:
                    try:
                        print(f'[RECONCILIATION] Cancelling duplicate protective order: {o_id}')
                    except Exception:
                        pass
                    await exec_engine.cancel_order_safe(symbol, o_id)
        return resolved_sl

    async def _reconcile_live_broker_state(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._lock.locked():
            return
        async with self._lock:
            await self._reconcile_live_broker_state_inner()

    async def _reconcile_live_broker_state_inner(self):
        """Full Live Exchange Handshake and Reconciliation."""
        self.last_reconcile_time = time.time()
        exec_engine = self.bot.execution
        try:
            if exec_engine.coindcx_client:
                await self._reconcile_coindcx()
            else:
                await self._reconcile_binance()
            self.reconcile_errors = 0
            if self.safe_mode_active:
                has_unknown = any(ctx.state in (OrderState.EXECUTION_UNKNOWN, OrderState.EXIT_UNKNOWN) for ctx in self.bot.order_state_machine.contexts.values())
                has_unresolved = bool(hasattr(self.bot.execution, 'intent_journal') and self.bot.execution.intent_journal.unresolved())
                if not has_unknown and not has_unresolved:
                    print('[RECONCILIATION] ✅ State divergence resolved. SAFE MODE DEACTIVATED.')
                    self.safe_mode_active = False
        except Exception as e:
            self.reconcile_errors += 1
            print(f'[RECONCILIATION FAIL] Live audit failed ({self.reconcile_errors}): {e}')
            if self.reconcile_errors >= 3 and (not self.safe_mode_active):
                self.safe_mode_active = True
                print('[RECONCILIATION] 🚨 SAFE MODE ACTIVATED: Multiple reconciliation errors. New entries halted.')

    async def _reconcile_binance(self):
        exec_engine = self.bot.execution
        try:
            self._global_binance_orders = await exec_engine.execute_with_retry(exec_engine.trade_client.fetch_open_orders)
        except Exception as e:
            print(f"[RECONCILIATION ERROR] Binance fetch_open_orders failed with exception: {e}")
            self._global_binance_orders = None
        if self._global_binance_orders is None:
            self.safe_mode_active = True
            print("[RECONCILIATION] [ALERT] SAFE MODE ACTIVATED: Binance fetch_open_orders failed/unreachable. Aborting pass.")
            return

        if Config.EXCHANGE_TYPE == 'futures':
            try:
                positions = await exec_engine.execute_with_retry(exec_engine.trade_client.fetch_positions)
            except Exception as e:
                print(f'[RECONCILIATION ERROR] Binance fetch_positions failed: {e}')
                self.safe_mode_active = True
                print('[RECONCILIATION] [ALERT] SAFE MODE ACTIVATED: Binance fetch_positions failed/unreachable. Aborting pass.')
                return
            if positions is None:
                self.safe_mode_active = True
                print('[RECONCILIATION] [ALERT] SAFE MODE ACTIVATED: Binance fetch_positions returned None. Aborting pass.')
                return

            pos_map = {p['symbol']: p for p in positions if isinstance(p, dict) and p.get('symbol')}
            for symbol in Config.SUPPORTED_SYMBOLS:
                ctx = self.bot.order_state_machine.get_context(symbol)
                exchange_pos = pos_map.get(symbol)
                contracts = float(exchange_pos.get('contracts', 0.0)) if exchange_pos else 0.0
                symbol_orders = [o for o in self._global_binance_orders if o.get('symbol') == symbol]
                resolved_sl = await self._sync_exchange_orders(symbol, ctx, contracts, symbol_orders)
                if resolved_sl:
                    ctx.native_sl_order_id = resolved_sl

                if contracts > 1e-05 and exchange_pos is not None:
                    should_adopt = not self.bot.in_position.get(symbol, False)
                    if should_adopt:
                        side = str(exchange_pos.get('side', '')).upper()
                        entry_p = float(exchange_pos.get('entryPrice', 0.0))
                        if entry_p <= 0.0:
                            entry_p = float(self.bot.pipeline.latest_prices.get(symbol, 0.0))
                        pos_side = side if side in ('LONG', 'SHORT') else 'LONG'
                        local_sl = float(self.bot.stop_loss.get(symbol, 0.0))
                        if local_sl <= 0.0:
                            print(f'[RECONCILIATION CRITICAL] Orphan futures position {symbol} has no durable strategy SL. No synthetic 2% stop; attempting emergency flatten.')
                            self.safe_mode_active = True
                            ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Orphan position without durable strategy stop-loss')
                            try:
                                self.bot.in_position[symbol] = True
                                self.bot.position_side[symbol] = pos_side
                                self.bot.position_size[symbol] = contracts
                                self.bot.entry_price[symbol] = entry_p
                                flatten_res = await exec_engine.emergency_flatten_position(symbol, pos_side, contracts, reason='ORPHAN_NO_DURABLE_STRATEGY_SL')
                                if flatten_res and flatten_res.is_fill_confirmed:
                                    actual_flatten = float(flatten_res.filled_qty or contracts)
                                    remaining_qty = max(0.0, contracts - actual_flatten)
                                    self.bot.position_size[symbol] = remaining_qty
                                    if remaining_qty <= 1e-5:
                                        self.bot.in_position[symbol] = False
                                        self.bot.position_side[symbol] = 'HOLD'
                                        ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason='Orphan position lacked durable strategy SL; emergency flattened')
                                    else:
                                        ctx.transition_to(OrderState.EXIT_UNKNOWN, reason='Orphan position lacked durable strategy SL; emergency flatten partially filled')
                                else:
                                    ctx.transition_to(OrderState.EXIT_UNKNOWN, reason='Orphan position lacked durable strategy SL; emergency flatten failed or unknown')
                            except Exception as e:
                                ctx.transition_to(OrderState.EXIT_UNKNOWN, reason=f'Orphan emergency flatten error: {e}')
                                print(f'[RECONCILIATION CRITICAL] Emergency flatten error for orphan {symbol}: {e}')
                            self.bot.save_state()
                            continue

                        self.bot.in_position[symbol] = True
                        self.bot.position_side[symbol] = pos_side
                        self.bot.position_size[symbol] = contracts
                        self.bot.entry_price[symbol] = entry_p
                        sl_price = local_sl
                        self.bot.stop_loss[symbol] = sl_price
                        r_amt = abs(entry_p - sl_price)
                        if r_amt <= 0.0:
                            self.safe_mode_active = True
                            ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Orphan position has invalid stop distance')
                            self.bot.save_state()
                            continue
                        fee_adj = entry_p * Config.FEE_RATE * 2.0
                        if self.bot.position_side[symbol] == 'LONG':
                            self.bot.take_profit_1r[symbol] = entry_p + 1.0 * r_amt + fee_adj
                            self.bot.take_profit_2r[symbol] = entry_p + 2.2 * r_amt + fee_adj
                            self.bot.take_profit[symbol] = entry_p + 4.0 * r_amt + fee_adj
                        else:
                            self.bot.take_profit_1r[symbol] = entry_p - 1.0 * r_amt - fee_adj
                            self.bot.take_profit_2r[symbol] = entry_p - 2.2 * r_amt - fee_adj
                            self.bot.take_profit[symbol] = entry_p - 4.0 * r_amt - fee_adj
                        self.bot.highest_price_reached[symbol] = entry_p
                        self.bot.lowest_price_reached[symbol] = entry_p
                        self.bot.entry_time[symbol] = time.time() * 1000.0
                        ctx.filled_qty = contracts
                        ctx.entry_price = entry_p
                        ctx.stop_loss = sl_price
                        ctx.side = self.bot.position_side[symbol]
                        if getattr(Config, 'USE_NATIVE_EXCHANGE_SL', True) and self.bot.has_keys and (not Config.PAPER_TRADING):
                            try:
                                sl_side = 'sell' if self.bot.position_side[symbol] == 'LONG' else 'buy'
                                sl_order = await exec_engine.place_native_stop_loss(symbol, sl_side, contracts, sl_price)
                                if self._is_active_sl_order(sl_order):
                                    ctx.native_sl_order_id = str(sl_order['id']) if isinstance(sl_order, dict) else str(sl_order.exchange_order_id)
                                    ctx.transition_to(OrderState.PROTECTED, reason='Adopted position with confirmed Native SL')
                                    print(f'[RECONCILIATION] ✅ Native SL placed for adopted position {symbol} @ {sl_price}')
                                    await self._release_reserved_risk(ctx)
                                else:
                                    ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Failed native SL on adopted position')
                                    self.safe_mode_active = True
                            except Exception as e:
                                ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason=f'Recon Native SL Error: {e}')
                                self.safe_mode_active = True
                        else:
                            ctx.transition_to(OrderState.PROTECTED, reason='Virtual SL active for adopted position')
                            await self._release_reserved_risk(ctx)
                        self.bot.save_state()
                    else:
                        local_qty = self.bot.position_size.get(symbol, 0.0)
                        if abs(contracts - local_qty) > 1e-05:
                            if contracts > local_qty:
                                self.bot.position_size[symbol] = contracts
                                self.bot.save_state()
                            elif contracts < local_qty:
                                self.bot.position_size[symbol] = contracts
                                self.bot.save_state()

                        if getattr(Config, 'USE_NATIVE_EXCHANGE_SL', True) and self.bot.has_keys and (not Config.PAPER_TRADING):
                            if not resolved_sl:
                                sl_side = 'sell' if self.bot.position_side[symbol] == 'LONG' else 'buy'
                                sl_price = float(self.bot.stop_loss.get(symbol, 0.0))
                                if sl_price <= 0.0:
                                    print(f'[RECONCILIATION CRITICAL] Open futures position {symbol} has no durable strategy SL. No synthetic 2% stop; attempting emergency flatten.')
                                    self.safe_mode_active = True
                                    ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Open futures position without durable strategy stop-loss')
                                    try:
                                        flatten_res = await exec_engine.emergency_flatten_position(symbol, self.bot.position_side[symbol], contracts, reason='UNPROTECTED_NO_DURABLE_STRATEGY_SL')
                                        if flatten_res and flatten_res.is_fill_confirmed:
                                            actual_flatten = float(flatten_res.filled_qty or contracts)
                                            remaining_qty = max(0.0, contracts - actual_flatten)
                                            self.bot.position_size[symbol] = remaining_qty
                                            if remaining_qty <= 1e-5:
                                                self.bot.in_position[symbol] = False
                                                self.bot.position_side[symbol] = 'HOLD'
                                                ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason='Unprotected futures position lacked durable SL; emergency flattened')
                                                await self._release_reserved_risk(ctx)
                                            else:
                                                ctx.transition_to(OrderState.EXIT_UNKNOWN, reason='Unprotected futures position emergency flatten partially filled')
                                        else:
                                            ctx.transition_to(OrderState.EXIT_UNKNOWN, reason='Unprotected futures position emergency flatten failed or unknown')
                                    except Exception as e:
                                        ctx.transition_to(OrderState.EXIT_UNKNOWN, reason=f'Unprotected futures emergency flatten error: {e}')
                                    self.bot.save_state()
                                    continue
                                try:
                                    sl_order = await exec_engine.place_native_stop_loss(symbol, sl_side, contracts, sl_price)
                                    if self._is_active_sl_order(sl_order):
                                        ctx.native_sl_order_id = str(sl_order['id']) if isinstance(sl_order, dict) else str(sl_order.exchange_order_id)
                                        if ctx.state not in (OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING):
                                            ctx.transition_to(OrderState.PROTECTED, reason='Reconciled and placed replacement Native SL')
                                            await self._release_reserved_risk(ctx)
                                    else:
                                        flatten_res = await exec_engine.emergency_flatten_position(symbol, self.bot.position_side[symbol], contracts, reason='UNPROTECTED_FUTURES_POSITION')
                                        if flatten_res and flatten_res.is_fill_confirmed:
                                            ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason='Emergency flattened unprotected futures position')
                                            self.bot.in_position[symbol] = False
                                            self.bot.position_size[symbol] = 0.0
                                            await self._release_reserved_risk(ctx)
                                        else:
                                            ctx.transition_to(OrderState.EXIT_UNKNOWN, reason='Open futures position lacking SL could not be flattened')
                                            self.safe_mode_active = True
                                except Exception as e:
                                    ctx.transition_to(OrderState.EXIT_UNKNOWN, reason=f'Re-protection error: {e}')
                                    self.safe_mode_active = True
                                self.bot.save_state()
                            else:
                                if ctx.state not in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING):
                                    ctx.transition_to(OrderState.PROTECTED, reason='Reconciled active position with verified Native SL')
                                    await self._release_reserved_risk(ctx)
                                    self.bot.save_state()
                        else:
                            if ctx.state not in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.PARTIALLY_FILLED):
                                ctx.transition_to(OrderState.PROTECTED, reason='Virtual SL active for position')
                                await self._release_reserved_risk(ctx)
                                self.bot.save_state()
                else:
                    is_locally_open = self.bot.in_position.get(symbol, False)
                    if is_locally_open and (not self._should_skip_reconcile_close(ctx, symbol)):
                        print(f'[RECONCILIATION] Ghost position detected for {symbol} (Local says OPEN, Exchange says FLAT). Cleaning up local state.')
                        ctx.transition_to(OrderState.CLOSED, reason='Reconciliation: Position flat on exchange')
                        self.bot.in_position[symbol] = False
                        self.bot.position_side[symbol] = 'HOLD'
                        self.bot.position_size[symbol] = 0.0
                        await self._release_reserved_risk(ctx)
                        self.bot.entry_price[symbol] = 0.0
                        self.bot.stop_loss[symbol] = 0.0
                        self.bot.take_profit[symbol] = 0.0
                        self.bot.take_profit_1r[symbol] = 0.0
                        self.bot.take_profit_2r[symbol] = 0.0
                        self.bot.save_state()
        else:
            balance = await exec_engine.execute_with_retry(exec_engine.trade_client.fetch_balance)
            total_bal = (balance or {}).get('total', {})
            for symbol in Config.SUPPORTED_SYMBOLS:
                base_asset = symbol.split('/')[0]
                base_qty = float(total_bal.get(base_asset, 0.0))
                ctx = self.bot.order_state_machine.get_context(symbol)
                symbol_orders = [o for o in getattr(self, '_global_binance_orders', []) if o.get('symbol') == symbol]
                resolved_sl = await self._sync_exchange_orders(symbol, ctx, base_qty, symbol_orders)
                if resolved_sl:
                    ctx.native_sl_order_id = resolved_sl
                if self.bot.in_position.get(symbol, False):
                    expected_size = float(self.bot.position_size.get(symbol, 0.0))
                    if base_qty < (expected_size - 1e-5):
                        print(f'[RECONCILIATION] [WARNING] Spot balance for {symbol} ({base_qty}) is less than expected bot size ({expected_size}). External transfer or manual sell? Quarantining.')
                        ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='Spot balance deficit detected')
                        self.bot.save_state()
                        continue
                    if ctx.native_sl_order_id and self.bot.has_keys and not Config.PAPER_TRADING:
                        sl_filled = await exec_engine.check_order_filled(symbol, ctx.native_sl_order_id)
                        if sl_filled:
                            ctx.transition_to(OrderState.CLOSED, reason=f'Spot SL filled on exchange')
                            self.bot.in_position[symbol] = False
                            self.bot.position_side[symbol] = 'HOLD'
                            self.bot.position_size[symbol] = 0.0
                            await self._release_reserved_risk(ctx)
                            self.bot.entry_price[symbol] = 0.0
                            self.bot.stop_loss[symbol] = 0.0
                            self.bot.take_profit[symbol] = 0.0
                            self.bot.take_profit_1r[symbol] = 0.0
                            self.bot.take_profit_2r[symbol] = 0.0
                            self.bot.save_state()
                            ctx.native_sl_order_id = None
                        else:
                            sl_status = await exec_engine.verify_order_active(symbol, ctx.native_sl_order_id)
                            if sl_status == 'UNKNOWN':
                                ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='SL state UNKNOWN')
                                self.bot.save_state()
                            elif sl_status == 'INACTIVE':
                                ctx.native_sl_order_id = None
                                sl_order = await exec_engine.place_native_stop_loss(symbol, 'sell', self.bot.position_size[symbol], self.bot.stop_loss[symbol])
                                if self._is_active_sl_order(sl_order):
                                    ctx.native_sl_order_id = str(sl_order['id']) if isinstance(sl_order, dict) else str(sl_order.exchange_order_id)
                                    ctx.transition_to(OrderState.PROTECTED, reason='SL re-protected')
                            else:
                                if ctx.state not in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.PARTIALLY_FILLED):
                                    ctx.transition_to(OrderState.PROTECTED, reason='SL verified active')
                    else:
                        if ctx.state not in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.PARTIALLY_FILLED):
                            ctx.transition_to(OrderState.PROTECTED, reason='Spot position active without Native SL')

    async def _reconcile_coindcx(self):
        exec_engine = self.bot.execution
        balance_data = await exec_engine.coindcx_client.fetch_balance()
        if not balance_data:
            return
        bal_map: dict[str, float] = {}
        if isinstance(balance_data, dict):
            total_dict = balance_data.get('total')
            if isinstance(total_dict, dict):
                for curr, val in total_dict.items():
                    try:
                        bal_map[str(curr).upper()] = float(val or 0.0)
                    except (ValueError, TypeError):
                        pass
            elif 'balances' in balance_data and isinstance(balance_data['balances'], list):
                for b in balance_data['balances']:
                    if isinstance(b, dict) and 'currency' in b:
                        curr = str(b['currency']).upper()
                        bal_map[curr] = float(b.get('balance', 0) or 0) + float(b.get('locked_balance', 0) or 0)
        elif isinstance(balance_data, list):
            for b in balance_data:
                if isinstance(b, dict) and 'currency' in b:
                    curr = str(b['currency']).upper()
                    bal_map[curr] = float(b.get('balance', 0) or 0) + float(b.get('locked_balance', 0) or 0)
        # Point 6 Fix: Fetch active exchange orders from CoinDCX for protective order sync
        cdx_orders_raw = []
        if hasattr(exec_engine, "coindcx_client") and exec_engine.coindcx_client is not None:
            try:
                fetch_fn = getattr(exec_engine.coindcx_client, "fetch_active_orders", None)
                if callable(fetch_fn):
                    call_res = fetch_fn()
                    if inspect.isawaitable(call_res):
                        cdx_orders_raw = await call_res
                    elif isinstance(call_res, list):
                        cdx_orders_raw = call_res
            except Exception as e:
                print(f"[RECONCILIATION] CoinDCX fetch_active_orders error: {e}")
                cdx_orders_raw = []

        cdx_orders = cdx_orders_raw or []

        for symbol in Config.SUPPORTED_SYMBOLS:
            base_coin = symbol.split('/')[0].upper()
            market_name = symbol.replace('/', '').upper()
            qty = bal_map.get(base_coin, 0.0)
            ctx = self.bot.order_state_machine.get_context(symbol)

            # Map CoinDCX active orders for this symbol to canonical protective order structure
            symbol_orders = []
            for o in cdx_orders:
                o_mkt = str(o.get('market') or o.get('symbol') or '').upper()
                if o_mkt == market_name or o_mkt == symbol.replace('/', ''):
                    canon_order = {
                        'id': str(o.get('id') or o.get('order_id') or ''),
                        'symbol': symbol,
                        'side': str(o.get('side', '')).lower(),
                        'type': str(o.get('order_type', '')).lower(),
                        'stop_price': float(o.get('stop_price') or o.get('price_per_unit') or 0.0),
                        'price': float(o.get('price_per_unit') or 0.0),
                        'amount': float(o.get('total_quantity') or o.get('amount') or 0.0),
                    }
                    symbol_orders.append(canon_order)

            resolved_sl = await self._sync_exchange_orders(symbol, ctx, qty, symbol_orders)
            if resolved_sl:
                ctx.native_sl_order_id = resolved_sl

            if self.bot.in_position.get(symbol, False):
                expected_size = float(self.bot.position_size.get(symbol, 0.0))
                if qty < (expected_size - 1e-5):
                    print(f'[RECONCILIATION] [WARNING] CoinDCX Spot balance for {symbol} ({qty}) is less than expected bot size ({expected_size}). Quarantining.')
                    ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='CoinDCX Spot balance deficit')
                    self.bot.save_state()
                    continue

                if ctx.native_sl_order_id and self.bot.has_keys and not Config.PAPER_TRADING:
                    sl_status = await exec_engine.verify_order_active(symbol, ctx.native_sl_order_id)
                    if sl_status == 'UNKNOWN':
                        ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason='CoinDCX SL state UNKNOWN')
                        self.bot.save_state()
                    elif sl_status == 'INACTIVE':
                        ctx.native_sl_order_id = None
                        sl_order = await exec_engine.place_native_stop_loss(symbol, 'sell', self.bot.position_size[symbol], self.bot.stop_loss[symbol])
                        if self._is_active_sl_order(sl_order):
                            ctx.native_sl_order_id = str(sl_order['id']) if isinstance(sl_order, dict) else str(sl_order.exchange_order_id)
                            ctx.transition_to(OrderState.PROTECTED, reason='CoinDCX SL re-protected')
                    else:
                        if ctx.state not in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.PARTIALLY_FILLED):
                            ctx.transition_to(OrderState.PROTECTED, reason='CoinDCX SL verified active')
                else:
                    if ctx.state not in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.PARTIALLY_FILLED):
                        ctx.transition_to(OrderState.PROTECTED, reason='CoinDCX position active')

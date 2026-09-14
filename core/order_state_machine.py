import time
from enum import Enum
from typing import Any, Optional, Dict, List

class OrderState(str, Enum):
    IDLE = "IDLE"
    ORDER_INTENT_CREATED = "ORDER_INTENT_CREATED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    EXECUTION_UNKNOWN = "EXECUTION_UNKNOWN"
    EXIT_UNKNOWN = "EXIT_UNKNOWN"
    ORDER_ACK = "ORDER_ACK"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    SL_PLACEMENT_PENDING = "SL_PLACEMENT_PENDING"
    PROTECTED = "PROTECTED"
    TP1_LOCKED = "TP1_LOCKED"
    TP2_LOCKED = "TP2_LOCKED"
    RUNNER_ACTIVE = "RUNNER_ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    EMERGENCY_FLATTENED = "EMERGENCY_FLATTENED"
    REJECTED = "REJECTED"

class PositionContext:
    _LEGAL_TRANSITIONS = {
        OrderState.IDLE: {OrderState.ORDER_INTENT_CREATED, OrderState.PROTECTED, OrderState.CLOSED},
        OrderState.ORDER_INTENT_CREATED: {OrderState.ORDER_SUBMITTED, OrderState.EXECUTION_UNKNOWN, OrderState.REJECTED, OrderState.CLOSED},
        OrderState.ORDER_SUBMITTED: {OrderState.ORDER_ACK, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.EXECUTION_UNKNOWN, OrderState.REJECTED, OrderState.CLOSED},
        OrderState.ORDER_ACK: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.EXECUTION_UNKNOWN, OrderState.REJECTED, OrderState.CLOSED},
        OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.SL_PLACEMENT_PENDING, OrderState.PROTECTED, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.EXECUTION_UNKNOWN, OrderState.CLOSED},
        OrderState.FILLED: {OrderState.SL_PLACEMENT_PENDING, OrderState.PROTECTED, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.CLOSED, OrderState.EXECUTION_UNKNOWN},
        OrderState.SL_PLACEMENT_PENDING: {OrderState.PROTECTED, OrderState.EXECUTION_UNKNOWN, OrderState.EXIT_UNKNOWN, OrderState.CLOSING, OrderState.CLOSED},
        OrderState.PROTECTED: {OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.EXECUTION_UNKNOWN, OrderState.CLOSED},
        OrderState.TP1_LOCKED: {OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.EXECUTION_UNKNOWN, OrderState.CLOSED},
        OrderState.TP2_LOCKED: {OrderState.RUNNER_ACTIVE, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.EXECUTION_UNKNOWN, OrderState.CLOSED},
        OrderState.RUNNER_ACTIVE: {OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.EXECUTION_UNKNOWN, OrderState.CLOSED},
        OrderState.CLOSING: {OrderState.CLOSED, OrderState.EXIT_UNKNOWN, OrderState.EXECUTION_UNKNOWN, OrderState.EMERGENCY_FLATTENED},
        OrderState.EXECUTION_UNKNOWN: {OrderState.ORDER_ACK, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.SL_PLACEMENT_PENDING, OrderState.PROTECTED, OrderState.CLOSING, OrderState.EXIT_UNKNOWN, OrderState.CLOSED, OrderState.REJECTED},
        OrderState.EXIT_UNKNOWN: {OrderState.CLOSING, OrderState.CLOSED, OrderState.EMERGENCY_FLATTENED, OrderState.EXECUTION_UNKNOWN},
        OrderState.REJECTED: {OrderState.IDLE, OrderState.ORDER_INTENT_CREATED},
        OrderState.CLOSED: {OrderState.IDLE, OrderState.ORDER_INTENT_CREATED},
        OrderState.EMERGENCY_FLATTENED: {OrderState.IDLE, OrderState.ORDER_INTENT_CREATED},
    }

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.state = OrderState.IDLE
        self.side = "HOLD"
        self.requested_qty = 0.0
        self.filled_qty = 0.0
        self.remaining_qty = 0.0
        self.entry_price = 0.0
        self.fill_avg_price = 0.0
        self.stop_loss = 0.0
        self.take_profit_1r = 0.0
        self.take_profit_2r = 0.0
        self.take_profit_runner = 0.0
        self.trailing_stop = 0.0
        self.entry_order_id = None
        self.client_order_id = None
        self.intent_id = None
        self.execution_state = "NOT_SUBMITTED"
        self.exit_order_id = None
        self.exit_client_order_id = None
        self.native_sl_order_id = None
        self.native_tp1_order_id = None
        self.native_tp2_order_id = None
        self.created_at = 0.0
        self.filled_at = 0.0
        self.closed_at = 0.0
        self.last_transition_time = time.time()
        self.setup_mode = "STRICT"
        self.zone_id = None
        self.exit_reason = None
        self.realized_pnl = 0.0
        self.history = []
        self.reserved_risk_pct = 0.0
        self.reserved_risk_side = "HOLD"
        self.reservation_id = None

    def is_active(self):
        return self.state not in (OrderState.IDLE, OrderState.CLOSED, OrderState.REJECTED, OrderState.EMERGENCY_FLATTENED)
    def is_protected(self):
        return self.state in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE)
    def is_in_flight(self):
        return self.state in (OrderState.ORDER_INTENT_CREATED, OrderState.ORDER_SUBMITTED, OrderState.EXECUTION_UNKNOWN, OrderState.EXIT_UNKNOWN, OrderState.SL_PLACEMENT_PENDING)

    def transition_to(self, new_state: OrderState, reason: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        prev_state = self.state
        try:
            new_state = OrderState(new_state)
        except (TypeError, ValueError):
            print(f"[STATE MACHINE] [{self.symbol}] INVALID target state: {new_state}")
            return False
        if new_state == prev_state:
            return True
        allowed = self._LEGAL_TRANSITIONS.get(prev_state, set())
        if new_state not in allowed:
            print(f"[STATE MACHINE] [{self.symbol}] BLOCKED invalid transition {prev_state.value} -> {new_state.value} ({reason})")
            self.history.append({'timestamp': time.time(), 'from_state': prev_state.value, 'to_state': new_state.value, 'reason': reason, 'blocked': True, 'filled_qty': self.filled_qty, 'entry_price': self.entry_price, 'stop_loss': self.stop_loss, 'metadata': metadata or {}})
            if len(self.history) > 50:
                self.history.pop(0)
            return False

        # P0 safety invariant: a context may enter PROTECTED only when it
        # actually contains a usable position and a directional stop-loss.
        if new_state == OrderState.PROTECTED:
            protection_error = None
            if self.filled_qty <= 0.0:
                protection_error = "filled_qty must be > 0"
            elif self.entry_price <= 0.0:
                protection_error = "entry_price must be > 0"
            elif self.side not in ("LONG", "SHORT"):
                protection_error = "side must be LONG or SHORT"
            elif self.stop_loss <= 0.0:
                protection_error = "stop_loss must be > 0"
            elif self.side == "LONG" and self.stop_loss >= self.entry_price:
                protection_error = "LONG stop_loss must be below entry_price"
            elif self.side == "SHORT" and self.stop_loss <= self.entry_price:
                protection_error = "SHORT stop_loss must be above entry_price"

            if protection_error:
                fail_reason = "PROTECTED invariant failed: " + protection_error
                now = time.time()
                self.history.append({
                    'timestamp': now,
                    'from_state': prev_state.value,
                    'to_state': OrderState.PROTECTED.value,
                    'reason': reason,
                    'blocked': True,
                    'invariant_failure': fail_reason,
                    'filled_qty': self.filled_qty,
                    'entry_price': self.entry_price,
                    'stop_loss': self.stop_loss,
                    'metadata': metadata or {},
                })
                if len(self.history) > 50:
                    self.history.pop(0)
                self.state = OrderState.EXECUTION_UNKNOWN
                self.last_transition_time = now
                self.execution_state = "EXECUTION_UNKNOWN"
                print(f"[STATE MACHINE] [{self.symbol}] FAIL-CLOSED PROTECTION BLOCK: {fail_reason}")
                return False

        self.state = new_state
        self.last_transition_time = time.time()
        self.history.append({'timestamp': self.last_transition_time, 'from_state': prev_state.value, 'to_state': new_state.value, 'reason': reason, 'blocked': False, 'filled_qty': self.filled_qty, 'entry_price': self.entry_price, 'stop_loss': self.stop_loss, 'metadata': metadata or {}})
        if len(self.history) > 50:
            self.history.pop(0)
        print(f"[STATE MACHINE] [{self.symbol}] {prev_state.value} -> {new_state.value} ({reason})")
        return True

    def to_dict(self):
        return {
            'symbol': self.symbol, 'state': self.state.value, 'side': self.side,
            'requested_qty': self.requested_qty, 'filled_qty': self.filled_qty, 'remaining_qty': self.remaining_qty,
            'entry_price': self.entry_price, 'fill_avg_price': self.fill_avg_price, 'stop_loss': self.stop_loss,
            'take_profit_1r': self.take_profit_1r, 'take_profit_2r': self.take_profit_2r, 'take_profit_runner': self.take_profit_runner,
            'trailing_stop': self.trailing_stop, 'entry_order_id': self.entry_order_id, 'client_order_id': self.client_order_id,
            'intent_id': self.intent_id, 'execution_state': self.execution_state, 'exit_order_id': self.exit_order_id,
            'exit_client_order_id': self.exit_client_order_id, 'native_sl_order_id': self.native_sl_order_id,
            'native_tp1_order_id': self.native_tp1_order_id, 'native_tp2_order_id': self.native_tp2_order_id,
            'created_at': self.created_at, 'filled_at': self.filled_at, 'closed_at': self.closed_at,
            'setup_mode': self.setup_mode, 'zone_id': self.zone_id, 'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl, 'last_transition_time': self.last_transition_time,
            'reserved_risk_pct': self.reserved_risk_pct, 'reserved_risk_side': self.reserved_risk_side,
            'reservation_id': self.reservation_id, 'history': self.history
        }

    @classmethod
    def from_dict(cls, data):
        ctx = cls(symbol=data.get('symbol', 'BTC/USDT'))
        try: ctx.state = OrderState(data.get('state', OrderState.IDLE.value))
        except (ValueError, TypeError): ctx.state = OrderState.IDLE
        numeric = {'requested_qty','filled_qty','remaining_qty','entry_price','fill_avg_price','stop_loss','take_profit_1r','take_profit_2r','take_profit_runner','trailing_stop','created_at','filled_at','closed_at','realized_pnl','last_transition_time','reserved_risk_pct'}
        defaults = {'side':'HOLD','execution_state':'NOT_SUBMITTED','setup_mode':'STRICT','reserved_risk_side':'HOLD'}
        for attr, default in defaults.items(): setattr(ctx, attr, data.get(attr, default))
        for attr in numeric:
            try: setattr(ctx, attr, float(data.get(attr, getattr(ctx, attr, 0.0))))
            except (TypeError, ValueError): pass
        for attr in ('entry_order_id','client_order_id','intent_id','exit_order_id','exit_client_order_id','native_sl_order_id','native_tp1_order_id','native_tp2_order_id','zone_id','exit_reason','reservation_id'):
            setattr(ctx, attr, data.get(attr, getattr(ctx, attr, None)))
        ctx.history = data.get('history', []) if isinstance(data.get('history', []), list) else []
        return ctx

class OrderStateMachine:
    def __init__(self, supported_symbols: List[str]):
        self.contexts = {sym: PositionContext(sym) for sym in supported_symbols}
    def get_context(self, symbol: str):
        if symbol not in self.contexts: self.contexts[symbol] = PositionContext(symbol)
        return self.contexts[symbol]
    def is_active(self, symbol: str): return self.get_context(symbol).is_active()
    def is_protected(self, symbol: str): return self.get_context(symbol).is_protected()
    def serialize_all(self): return {sym: ctx.to_dict() for sym, ctx in self.contexts.items()}
    def load_all(self, state_dict):
        if not isinstance(state_dict, dict): return
        for sym, data in state_dict.items():
            if isinstance(data, dict): self.contexts[sym] = PositionContext.from_dict(data)

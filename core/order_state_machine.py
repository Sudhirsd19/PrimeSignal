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
    def __init__(self, symbol: str):
        self.symbol: str = symbol
        self.state: OrderState = OrderState.IDLE
        self.side: str = "HOLD"  # LONG / SHORT / HOLD
        
        # Order and execution quantities
        self.requested_qty: float = 0.0
        self.filled_qty: float = 0.0
        self.remaining_qty: float = 0.0
        self.entry_price: float = 0.0
        self.fill_avg_price: float = 0.0
        
        # Protection and targets
        self.stop_loss: float = 0.0
        self.take_profit_1r: float = 0.0
        self.take_profit_2r: float = 0.0
        self.take_profit_runner: float = 0.0
        self.trailing_stop: float = 0.0
        
        # Native Exchange Order IDs
        self.entry_order_id: Optional[str] = None
        self.client_order_id: Optional[str] = None
        self.intent_id: Optional[str] = None
        self.execution_state: str = "NOT_SUBMITTED"
        self.exit_order_id: Optional[str] = None
        self.exit_client_order_id: Optional[str] = None
        self.native_sl_order_id: Optional[str] = None
        self.native_tp1_order_id: Optional[str] = None
        self.native_tp2_order_id: Optional[str] = None
        
        # Timestamps and diagnostics
        self.created_at: float = 0.0
        self.filled_at: float = 0.0
        self.closed_at: float = 0.0
        self.last_transition_time: float = time.time()
        self.setup_mode: str = "STRICT"
        self.zone_id: Optional[str] = None
        self.exit_reason: Optional[str] = None
        self.realized_pnl: float = 0.0
        self.history: List[Dict[str, Any]] = []
        
        # Risk tracking for EXECUTION_UNKNOWN exposure leak fix
        self.reserved_risk_pct: float = 0.0
        self.reserved_risk_side: str = "HOLD"
        self.reservation_id: Optional[str] = None

    def is_active(self) -> bool:
        """Returns True if position is currently active and not closed/idle/rejected."""
        return self.state not in (OrderState.IDLE, OrderState.CLOSED, OrderState.REJECTED, OrderState.EMERGENCY_FLATTENED)

    def is_protected(self) -> bool:
        """Returns True if position has an active verified stop loss."""
        return self.state in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE)

    def is_in_flight(self) -> bool:
        """Returns True if an order is being submitted or its outcome is unknown. Reconciliation must not interfere."""
        return self.state in (OrderState.ORDER_INTENT_CREATED, OrderState.ORDER_SUBMITTED, OrderState.EXECUTION_UNKNOWN, OrderState.EXIT_UNKNOWN, OrderState.SL_PLACEMENT_PENDING)

    def transition_to(self, new_state: OrderState, reason: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Transitions position context to a new state and logs transition history.
        """
        prev_state = self.state
        self.state = new_state
        self.last_transition_time = time.time()
        
        record = {
            'timestamp': self.last_transition_time,
            'from_state': prev_state.value if isinstance(prev_state, OrderState) else str(prev_state),
            'to_state': new_state.value if isinstance(new_state, OrderState) else str(new_state),
            'reason': reason,
            'filled_qty': self.filled_qty,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'metadata': metadata or {}
        }
        self.history.append(record)
        if len(self.history) > 50:
            self.history.pop(0)
            
        print(f"[STATE MACHINE] [{self.symbol}] {prev_state.value} -> {new_state.value} ({reason})")
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'state': self.state.value,
            'side': self.side,
            'requested_qty': self.requested_qty,
            'filled_qty': self.filled_qty,
            'remaining_qty': self.remaining_qty,
            'entry_price': self.entry_price,
            'fill_avg_price': self.fill_avg_price,
            'stop_loss': self.stop_loss,
            'take_profit_1r': self.take_profit_1r,
            'take_profit_2r': self.take_profit_2r,
            'take_profit_runner': self.take_profit_runner,
            'trailing_stop': self.trailing_stop,
            'entry_order_id': self.entry_order_id,
            'client_order_id': self.client_order_id,
            'intent_id': self.intent_id,
            'execution_state': self.execution_state,
            'exit_order_id': self.exit_order_id,
            'exit_client_order_id': self.exit_client_order_id,
            'native_sl_order_id': self.native_sl_order_id,
            'created_at': self.created_at,
            'filled_at': self.filled_at,
            'closed_at': self.closed_at,
            'setup_mode': self.setup_mode,
            'zone_id': self.zone_id,
            'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl,
            'last_transition_time': self.last_transition_time,
            'reserved_risk_pct': self.reserved_risk_pct,
            'reserved_risk_side': self.reserved_risk_side,
            'reservation_id': self.reservation_id,
            'history': self.history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionContext':
        ctx = cls(symbol=data.get('symbol', 'BTC/USDT'))
        ctx.state = OrderState(data.get('state', OrderState.IDLE.value))
        ctx.side = data.get('side', 'HOLD')
        ctx.requested_qty = float(data.get('requested_qty', 0.0))
        ctx.filled_qty = float(data.get('filled_qty', 0.0))
        ctx.remaining_qty = float(data.get('remaining_qty', 0.0))
        ctx.entry_price = float(data.get('entry_price', 0.0))
        ctx.fill_avg_price = float(data.get('fill_avg_price', 0.0))
        ctx.stop_loss = float(data.get('stop_loss', 0.0))
        ctx.take_profit_1r = float(data.get('take_profit_1r', 0.0))
        ctx.take_profit_2r = float(data.get('take_profit_2r', 0.0))
        ctx.take_profit_runner = float(data.get('take_profit_runner', 0.0))
        ctx.trailing_stop = float(data.get('trailing_stop', 0.0))
        ctx.entry_order_id = data.get('entry_order_id')
        ctx.client_order_id = data.get('client_order_id')
        ctx.intent_id = data.get('intent_id')
        ctx.execution_state = data.get('execution_state', 'NOT_SUBMITTED')
        ctx.exit_order_id = data.get('exit_order_id')
        ctx.exit_client_order_id = data.get('exit_client_order_id')
        ctx.native_sl_order_id = data.get('native_sl_order_id')
        ctx.created_at = float(data.get('created_at', 0.0))
        ctx.filled_at = float(data.get('filled_at', 0.0))
        ctx.closed_at = float(data.get('closed_at', 0.0))
        ctx.setup_mode = data.get('setup_mode', 'STRICT')
        ctx.zone_id = data.get('zone_id')
        ctx.exit_reason = data.get('exit_reason')
        ctx.realized_pnl = float(data.get('realized_pnl', 0.0))
        ctx.last_transition_time = float(data.get('last_transition_time', ctx.last_transition_time))
        ctx.reserved_risk_pct = float(data.get('reserved_risk_pct', 0.0))
        ctx.reserved_risk_side = data.get('reserved_risk_side', 'HOLD')
        ctx.reservation_id = data.get('reservation_id')
        ctx.history = data.get('history', [])
        return ctx


class OrderStateMachine:
    """Manages multi-symbol granular order lifecycle contexts."""
    def __init__(self, supported_symbols: List[str]):
        self.contexts: Dict[str, PositionContext] = {sym: PositionContext(sym) for sym in supported_symbols}

    def get_context(self, symbol: str) -> PositionContext:
        if symbol not in self.contexts:
            self.contexts[symbol] = PositionContext(symbol)
        return self.contexts[symbol]

    def is_active(self, symbol: str) -> bool:
        state = self.get_context(symbol).state
        return state not in (OrderState.IDLE, OrderState.CLOSED, OrderState.REJECTED, OrderState.EMERGENCY_FLATTENED)

    def is_protected(self, symbol: str) -> bool:
        state = self.get_context(symbol).state
        return state in (OrderState.PROTECTED, OrderState.TP1_LOCKED, OrderState.TP2_LOCKED, OrderState.RUNNER_ACTIVE)

    def serialize_all(self) -> Dict[str, Any]:
        return {sym: ctx.to_dict() for sym, ctx in self.contexts.items()}

    def load_all(self, state_dict: Dict[str, Any]):
        for sym, data in state_dict.items():
            if isinstance(data, dict):
                self.contexts[sym] = PositionContext.from_dict(data)

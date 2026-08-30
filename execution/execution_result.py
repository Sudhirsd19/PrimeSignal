"""Explicit execution outcomes used by the live order adapter.

The trading engine must not use a truthy exchange acknowledgement as proof of
execution.  This module keeps the result vocabulary small and deliberately
represents unresolved outcomes instead of coercing them to rejection or zero.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional


class ExecutionState(str, Enum):
    NOT_SUBMITTED = "NOT_SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    ALREADY_CANCELLED = "ALREADY_CANCELLED"
    REJECTED = "REJECTED"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    STATUS_UNKNOWN = "STATUS_UNKNOWN"
    EXECUTION_UNKNOWN = "EXECUTION_UNKNOWN"
    CANCEL_UNKNOWN = "CANCEL_UNKNOWN"


_FILLED_STATES = {ExecutionState.FILLED, ExecutionState.PARTIALLY_FILLED}
_UNKNOWN_STATES = {
    ExecutionState.SUBMISSION_UNKNOWN,
    ExecutionState.STATUS_UNKNOWN,
    ExecutionState.EXECUTION_UNKNOWN,
    ExecutionState.CANCEL_UNKNOWN,
}


def _first_number(value: Any, names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        try:
            candidate = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
            if candidate is not None and candidate != "":
                return max(0.0, float(candidate))
        except (TypeError, ValueError):
            continue
    return default


@dataclass
class ExecutionResult:
    state: ExecutionState
    requested_qty: float = 0.0
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    average_fill_price: Optional[float] = None
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    intent_id: Optional[str] = None
    venue: Optional[str] = None
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fill_confirmed(self) -> bool:
        return self.state in _FILLED_STATES and self.filled_qty > 0.0

    @property
    def is_full_fill(self) -> bool:
        return self.state == ExecutionState.FILLED

    @property
    def is_partial_fill(self) -> bool:
        return self.state == ExecutionState.PARTIALLY_FILLED

    @property
    def is_unknown(self) -> bool:
        return self.state in _UNKNOWN_STATES

    @property
    def has_exchange_order(self) -> bool:
        return bool(self.exchange_order_id)

    def get(self, key: str, default: Any = None) -> Any:
        """Small mapping compatibility layer for existing read-only callers."""
        values = {
            "id": self.exchange_order_id,
            "clientOrderId": self.client_order_id,
            "client_order_id": self.client_order_id,
            "status": self.state.value.lower(),
            "amount": self.requested_qty,
            "filled": self.filled_qty,
            "filled_quantity": self.filled_qty,
            "remaining": self.remaining_qty,
            "remaining_quantity": self.remaining_qty,
            "price": self.average_fill_price,
            "average": self.average_fill_price,
        }
        return values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    @property
    def is_order_accepted(self) -> bool:
        """Returns True if the order was acknowledged by the exchange with an exchange order id in a valid non-error state."""
        return self.has_exchange_order and self.state in (
            ExecutionState.ACCEPTED,
            ExecutionState.FILLED,
            ExecutionState.PARTIALLY_FILLED,
        )

    def __bool__(self) -> bool:
        # Truthy when the order was successfully executed (fill confirmed) or accepted on exchange with an order ID.
        # Unknown, rejected, already cancelled, or unsubmitted outcomes evaluate to False.
        if self.is_unknown or self.state in (ExecutionState.NOT_SUBMITTED, ExecutionState.REJECTED):
            return False
        return self.is_fill_confirmed or self.is_order_accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_state": self.state.value,
            "requested_qty": self.requested_qty,
            "filled_qty": self.filled_qty,
            "remaining_qty": self.remaining_qty,
            "average_fill_price": self.average_fill_price,
            "exchange_order_id": self.exchange_order_id,
            "client_order_id": self.client_order_id,
            "intent_id": self.intent_id,
            "venue": self.venue,
            "error": self.error,
            "raw": self.raw,
        }

    @classmethod
    def from_exchange(
        cls,
        order: Any,
        requested_qty: float = 0.0,
        client_order_id: Optional[str] = None,
        intent_id: Optional[str] = None,
        venue: Optional[str] = None,
    ) -> "ExecutionResult":
        if isinstance(order, cls):
            if client_order_id and not order.client_order_id:
                order.client_order_id = client_order_id
            if intent_id and not order.intent_id:
                order.intent_id = intent_id
            if venue and not order.venue:
                order.venue = venue
            return order

        if not isinstance(order, dict):
            return cls(
                state=ExecutionState.EXECUTION_UNKNOWN,
                requested_qty=requested_qty or 0.0,
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue=venue,
                error="Exchange returned a non-mapping result",
            )

        status = str(order.get("status") or "").lower()
        requested = _first_number(order, ("requested_qty", "amount", "total_quantity"), requested_qty or 0.0)
        filled = _first_number(order, ("filled", "filled_quantity", "executedQty", "deal_quantity", "quantity"), 0.0)
        remaining = _first_number(order, ("remaining", "remaining_quantity"), max(0.0, requested - filled))
        if status in ("filled", "closed", "completed") and filled <= 0.0:
            # Only a terminal filled status permits amount to represent filled quantity.
            filled = requested
            remaining = 0.0
        elif filled > 0.0 and remaining <= 0.0 and requested > filled:
            remaining = requested - filled

        if status in ("filled", "closed", "completed") and filled > 0.0:
            state = ExecutionState.FILLED if remaining <= 1e-12 else ExecutionState.PARTIALLY_FILLED
        elif status in ("partially_filled", "partial", "partially-filled") or (filled > 0.0 and remaining > 0.0):
            state = ExecutionState.PARTIALLY_FILLED
        elif status in ("cancelled", "canceled", "expired"):
            state = ExecutionState.CANCELLED
        elif status in ("rejected", "failed", "error"):
            state = ExecutionState.REJECTED
        elif status in ("open", "new", "accepted", "pending", "untriggered", "active", ""):
            state = ExecutionState.ACCEPTED
        else:
            state = ExecutionState.STATUS_UNKNOWN

        avg = _first_number(order, ("average", "avg_price", "price_per_unit", "price"), 0.0)
        return cls(
            state=state,
            requested_qty=requested,
            filled_qty=filled,
            remaining_qty=remaining,
            average_fill_price=avg or None,
            exchange_order_id=str(order.get("id")) if order.get("id") is not None else None,
            client_order_id=(order.get("clientOrderId") or order.get("client_order_id") or client_order_id),
            intent_id=intent_id,
            venue=venue,
            raw=dict(order),
        )


def coerce_execution_result(
    value: Any,
    requested_qty: float = 0.0,
    client_order_id: Optional[str] = None,
    intent_id: Optional[str] = None,
    venue: Optional[str] = None,
) -> ExecutionResult:
    return ExecutionResult.from_exchange(
        value,
        requested_qty=requested_qty,
        client_order_id=client_order_id,
        intent_id=intent_id,
        venue=venue,
    )


class ExecutionIntentJournal:
    """Minimal durable write-ahead journal for exchange order intents."""

    _lock = threading.Lock()

    def __init__(self, path: Optional[str | os.PathLike[str]] = None):
        self.path = Path(path or os.getenv("PRIMESIGNAL_INTENT_JOURNAL", "data/execution_intents.jsonl"))

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def create(
        self,
        *,
        intent_id: str,
        client_order_id: str,
        venue: str,
        account_mode: str,
        symbol: str,
        side: str,
        requested_qty: float,
        order_role: str,
        price: Optional[float],
    ) -> None:
        if intent_id in self.latest():
            return
        self.append({
            "event": "INTENT_CREATED",
            "intent_id": intent_id,
            "client_order_id": client_order_id,
            "venue": venue,
            "account_mode": account_mode,
            "symbol": symbol,
            "side": side,
            "requested_qty": requested_qty,
            "order_role": order_role,
            "price": price,
            "created_at": time.time(),
            "state": "ORDER_INTENT_CREATED",
            # The intent UUID also serves as the durable reservation identity
            # unless a caller supplies a separate reservation record.
            "reservation_id": intent_id,
        })

    def result(self, result: ExecutionResult) -> None:
        self.append({
            "event": "INTENT_RESULT",
            "intent_id": result.intent_id,
            "client_order_id": result.client_order_id,
            "exchange_order_id": result.exchange_order_id,
            "state": result.state.value,
            "requested_qty": result.requested_qty,
            "filled_qty": result.filled_qty,
            "remaining_qty": result.remaining_qty,
            "average_fill_price": result.average_fill_price,
            "recorded_at": time.time(),
        })

    def latest(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        latest: dict[str, dict[str, Any]] = {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = record.get("intent_id")
                    if key:
                        # Keep the original intent metadata (symbol, side,
                        # account mode, role) when a later result event is
                        # appended.  Restart recovery needs both.
                        merged = dict(latest.get(key, {}))
                        merged.update(record)
                        latest[key] = merged
        except (OSError, ValueError, TypeError):
            return latest
        return latest

    def unresolved(self) -> list[dict[str, Any]]:
        unresolved_states = {
            ExecutionState.SUBMISSION_UNKNOWN.value,
            ExecutionState.STATUS_UNKNOWN.value,
            ExecutionState.EXECUTION_UNKNOWN.value,
        }
        return [record for record in self.latest().values() if record.get("state") in unresolved_states or record.get("state") == "ORDER_INTENT_CREATED"]


def new_intent_id() -> str:
    return uuid.uuid4().hex

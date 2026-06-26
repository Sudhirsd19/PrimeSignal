"""
Structured JSON trade decision logger for backtesting review.
Writes every trading decision as a JSON line to logs/trade_decisions.jsonl.
"""
import json
import os
from datetime import datetime, timezone
import uuid

from core.firebase_manager import FirebaseManager


class TradeLogger:
    """Appends structured JSON records for every trading decision."""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "trade_decisions.jsonl")

    # ── core writer ──────────────────────────────────────────────────────
    def _write(self, event_type: str, data: dict):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        
        # Write to Firebase if available
        firebase = FirebaseManager()
        if firebase.is_connected:
            try:
                def sanitize(obj):
                    if isinstance(obj, dict):
                        return {k: sanitize(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [sanitize(v) for v in obj]
                    elif hasattr(obj, "item"):  # numpy float/int
                        return obj.item()
                    elif str(type(obj)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
                        return str(obj)
                    return obj
                
                safe_entry = sanitize(entry)
                # Add to trade_logs collection with a generated ID or auto-id
                firebase.db.collection("trade_logs").add(safe_entry)
            except Exception as e:
                print(f"[FIREBASE] Failed to write log: {e}")

        # Always write to local JSONL as backup
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            print(f"[TRADE_LOG] Failed to write log entry: {e}")

    # ── signal events ────────────────────────────────────────────────────
    def log_signal_generated(self, symbol: str, signal: str, score: float,
                             mode: str, metadata: dict):
        """A strategy signal was generated (before any filters)."""
        self._write("SIGNAL_GENERATED", {
            "symbol": symbol,
            "signal": signal,
            "score": score,
            "mode": mode,
            "setup_type": metadata.get("setup_type"),
            "zone_id": metadata.get("zone_id"),
            "stop_loss": metadata.get("stop_loss"),
            "take_profit": metadata.get("take_profit"),
            "reason": metadata.get("reason"),
            "debug_checks": metadata.get("debug_checks"),
        })

    def log_signal_filtered(self, symbol: str, signal: str, reason: str,
                            filter_name: str = ""):
        """A signal was blocked by a filter / safety check."""
        self._write("SIGNAL_FILTERED", {
            "symbol": symbol,
            "signal": signal,
            "filter": filter_name,
            "reason": reason,
        })

    # ── trade events ─────────────────────────────────────────────────────
    def log_trade_executed(self, symbol: str, side: str, size: float,
                           entry_price: float, sl: float, tp: float,
                           order_id: str = "", mode: str = ""):
        """An order was sent to the exchange and filled."""
        self._write("TRADE_EXECUTED", {
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "order_id": order_id,
            "mode": mode,
        })

    def log_trade_exited(self, symbol: str, side: str, entry_price: float,
                         exit_price: float, pnl_pct: float, pnl_usdt: float,
                         reason: str = "", order_id: str = ""):
        """A position was closed."""
        self._write("TRADE_EXITED", {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usdt": pnl_usdt,
            "reason": reason,
            "order_id": order_id,
        })

    def log_partial_tp(self, symbol: str, tp_level: str, size_closed: float,
                       price: float):
        """A partial take-profit was executed (TP1 / TP2)."""
        self._write("PARTIAL_TP", {
            "symbol": symbol,
            "tp_level": tp_level,
            "size_closed": size_closed,
            "price": price,
        })

    # ── risk events ──────────────────────────────────────────────────────
    def log_risk_event(self, event_name: str, details: dict):
        """Circuit breaker, kill switch, cluster loss, etc."""
        self._write("RISK_EVENT", {
            "event_name": event_name,
            **details,
        })

    def log_order_fill(self, symbol: str, order_id: str, status: str,
                       fill_price: float = 0.0, fill_amount: float = 0.0,
                       elapsed_ms: int = 0):
        """Order fill confirmation result."""
        self._write("ORDER_FILL", {
            "symbol": symbol,
            "order_id": order_id,
            "status": status,
            "fill_price": fill_price,
            "fill_amount": fill_amount,
            "elapsed_ms": elapsed_ms,
        })

    def log_kill_switch(self, trigger_source: str, positions_closed: int):
        """Emergency kill switch was triggered."""
        self._write("KILL_SWITCH", {
            "trigger_source": trigger_source,
            "positions_closed": positions_closed,
        })

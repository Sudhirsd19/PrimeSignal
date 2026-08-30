import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

STRATEGY_VERSION = "PrimeSignal-v2.5.0-Institutional"

class ImmutableLedger:
    """
    Append-only cryptographically verifiable trade ledger for audit trails and post-mortem analysis.
    """
    def __init__(self, ledger_path: str = "data/immutable_trade_ledger.jsonl"):
        self.ledger_file = Path(ledger_path)
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_record_hash: str = "GENESIS_HASH_PRIMESIGNAL_V250"
        self._init_hash_chain()

    def _init_hash_chain(self):
        """Reads the last record hash to preserve blockchain-style tamper-evidence."""
        if self.ledger_file.exists():
            try:
                with open(self.ledger_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if 'record_hash' in record:
                                self.last_record_hash = record['record_hash']
            except Exception as e:
                print(f"[LEDGER] Warning reading hash chain: {e}")

    def record_entry(self, symbol: str, side: str, requested_qty: float, filled_qty: float,
                     fill_price: float, stop_loss: float, tp1: float, tp2: float, runner_tp: float,
                     client_order_id: str, exchange_order_id: Optional[str] = None,
                     native_sl_id: Optional[str] = None, config_hash: str = "") -> str:
        """Records position creation in the immutable ledger."""
        timestamp = time.time()
        record_id = f"TX_IN_{symbol.replace('/', '')}_{int(timestamp * 1000)}"
        
        raw_payload = f"{self.last_record_hash}|{record_id}|{symbol}|{side}|{filled_qty}|{fill_price}|{stop_loss}|{timestamp}"
        record_hash = hashlib.sha256(raw_payload.encode()).hexdigest()
        
        entry_record = {
            'record_id': record_id,
            'event_type': 'POSITION_OPENED',
            'timestamp': timestamp,
            'iso_time': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(timestamp)),
            'symbol': symbol,
            'side': side,
            'strategy_version': STRATEGY_VERSION,
            'config_hash': config_hash,
            'requested_qty': requested_qty,
            'filled_qty': filled_qty,
            'fill_price': fill_price,
            'stop_loss': stop_loss,
            'tp1_price': tp1,
            'tp2_price': tp2,
            'runner_tp': runner_tp,
            'client_order_id': client_order_id,
            'exchange_order_id': exchange_order_id,
            'native_sl_order_id': native_sl_id,
            'prev_hash': self.last_record_hash,
            'record_hash': record_hash
        }
        
        self._append(entry_record)
        self.last_record_hash = record_hash
        return record_id

    def record_exit(self, symbol: str, side: str, exit_qty: float, exit_price: float,
                    entry_price: float, realized_pnl: float, pnl_pct: float,
                    exit_reason: str, client_order_id: Optional[str] = None,
                    exchange_order_id: Optional[str] = None) -> str:
        """Records position closure or partial exit in the immutable ledger."""
        timestamp = time.time()
        record_id = f"TX_OUT_{symbol.replace('/', '')}_{int(timestamp * 1000)}"
        
        raw_payload = f"{self.last_record_hash}|{record_id}|{symbol}|{exit_qty}|{exit_price}|{realized_pnl}|{exit_reason}|{timestamp}"
        record_hash = hashlib.sha256(raw_payload.encode()).hexdigest()
        
        exit_record = {
            'record_id': record_id,
            'event_type': 'POSITION_EXITED',
            'timestamp': timestamp,
            'iso_time': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(timestamp)),
            'symbol': symbol,
            'side': side,
            'strategy_version': STRATEGY_VERSION,
            'exit_qty': exit_qty,
            'exit_price': exit_price,
            'entry_price': entry_price,
            'realized_pnl': realized_pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'client_order_id': client_order_id,
            'exchange_order_id': exchange_order_id,
            'prev_hash': self.last_record_hash,
            'record_hash': record_hash
        }
        
        self._append(exit_record)
        self.last_record_hash = record_hash
        return record_id

    def _append(self, record: Dict[str, Any]):
        try:
            with open(self.ledger_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                import os
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[LEDGER ERROR] Failed to write ledger record: {e}")

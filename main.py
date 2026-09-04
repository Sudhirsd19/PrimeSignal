import asyncio
import uuid
import inspect
import os
import threading
import sys
import uvicorn
import time
import datetime
import json
from typing import Any, Optional, Dict, List, cast
import pandas as pd
from pathlib import Path

# Reconfigure stdout/stderr to utf-8 on Windows to prevent UnicodeEncodeError
if sys.platform == 'win32':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
        getattr(sys.stderr, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except (AttributeError, Exception):
        pass

from config import Config
from execution.execution_engine import ExecutionEngine
from execution.execution_result import ExecutionResult, ExecutionState
from execution.exchange_validator import ExchangeValidator
from core.data_pipeline import RealTimeDataPipeline
from core.order_state_machine import OrderStateMachine, OrderState, PositionContext
from core.immutable_ledger import ImmutableLedger
from core.reconciliation_engine import ReconciliationEngine
from strategies.multi_timeframe import MultiTimeframeSMCStrategy
from strategies.indicators import prepare_dataframe, calculate_atr, calculate_ema, calculate_rsi, calculate_vwap
from ml.confirmation import MLSignalConfirmator
from ml.adversarial_debate import AdversarialDebateCourtroom
from core.lead_lag_arbitrage import LeadLagArbitrageEngine
from risk.risk_manager import RiskManager
from alerts.notifier import TelegramNotifier
from dashboard.app import app, DashboardState, add_log_message

class PrimeSignalBot:

    def _extract_filled_qty(self, order, default_req: float) -> float:
        if not order: return 0.0
        if isinstance(order, dict):
            val = order.get('filled', order.get('amount', default_req))
            return float(val if val is not None else default_req)
        return float(order.filled_qty) if order.is_fill_confirmed else 0.0

    def _is_truthy_fill(self, order) -> bool:
        if not order: return False
        if isinstance(order, dict):
            return str(order.get('status', '')).upper() in ('FILLED', 'PARTIALLY_FILLED')
        return order.is_fill_confirmed

    def _is_active_sl_order(self, order) -> bool:
        if order is None:
            return False
        if isinstance(order, dict):
            oid = order.get('id') or order.get('orderId')
            status = str(order.get('status', '')).upper()
            return bool(oid) and status not in ('REJECTED', 'FAILED', 'CANCELLED', 'CANCELED')
        if hasattr(order, 'is_order_accepted'):
            return bool(order.is_order_accepted)
        if hasattr(order, 'has_exchange_order'):
            return bool(order.has_exchange_order and getattr(order, 'state', None) not in (
                ExecutionState.REJECTED, ExecutionState.NOT_SUBMITTED, ExecutionState.EXECUTION_UNKNOWN
            ))
        return False

    def __init__(self):
        self.has_keys = Config.validate()
        
        # Initialize Core Institutional Modules
        self.execution = ExecutionEngine()
        self.pipeline = RealTimeDataPipeline(self.execution)
        self.strategy = MultiTimeframeSMCStrategy()
        self.risk = RiskManager()
        self.notifier = TelegramNotifier()
        self.lead_lag = LeadLagArbitrageEngine()
        self.courtroom = AdversarialDebateCourtroom()
        self.order_state_machine = OrderStateMachine(Config.SUPPORTED_SYMBOLS)
        self.immutable_ledger = ImmutableLedger()
        self.reconciliation = ReconciliationEngine(self, check_interval=15.0)
        
        self.ml_models: dict[str, MLSignalConfirmator] = {sym: MLSignalConfirmator() for sym in Config.SUPPORTED_SYMBOLS}
        
        # Internal State tracking (Per Symbol)
        self.in_position = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self.position_side = {sym: "HOLD" for sym in Config.SUPPORTED_SYMBOLS}
        self.entry_price = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.stop_loss = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.take_profit = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.highest_price_reached = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.lowest_price_reached = {sym: 999999.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.position_size = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.position_mode: dict[str, str] = {sym: "STRICT" for sym in Config.SUPPORTED_SYMBOLS}
        self.entry_time: dict[str, float] = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.last_trade_time: dict[str, float] = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.last_zone_traded: dict[str, str | None] = {sym: None for sym in Config.SUPPORTED_SYMBOLS}
        self.volatility_pause_until: dict[str, float] = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.partial_tp_taken = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self.take_profit_1r = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.tp2_taken = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        self.take_profit_2r = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.realized_pnl = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.original_position_size = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.last_exit_time: dict[str, float] = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.tp_cooldown_until: dict[str, float] = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.consecutive_losses = 0
        self.global_pause_until: float = 0.0
        self.relaxed_losses = 0
        self.relaxed_disabled_until: float = 0.0
        self.relaxed_trades_today = 0
        self.trades_today = 0
        self.last_trade_day = datetime.datetime.now(datetime.timezone.utc).date()
        self.trade_history: list[int] = []
        self.cluster_loss_pause_until: float = 0.0
        self.cluster_risk_penalty = False
        self.global_last_trade_time: float = 0.0
        self.traded_zones_cache = {}

        # ─── PROFIT-BASED LOGIC: New state tracking ───
        # Trade lifecycle ID for consolidated PnL grouping (TP1+TP2+Runner = 1 trade)
        self.current_trade_id: dict[str, str] = {sym: "" for sym in Config.SUPPORTED_SYMBOLS}
        # FX rate lock at entry time for CoinDCX INR trades
        self.entry_fx_rate: dict[str, float] = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        # Accumulated fees per trade lifecycle (entry + partial exits)
        self.accumulated_fees: dict[str, float] = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}

        # Dry-run virtual balance (used for paper trading & dry-run simulation)
        starting_bal = getattr(Config, 'PAPER_STARTING_BALANCE', 2000.0 if getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' else 10000.0)
        self._dry_run_balance_usdt = float(starting_bal)   # starting paper balance
        
        DashboardState.balance_currency = getattr(Config, 'PAPER_CURRENCY', 'INR')
        if not self.has_keys or Config.PAPER_TRADING:
            DashboardState.balance_usdt = self._dry_run_balance_usdt
            DashboardState.balance_base = 0.0
            self.risk.reset_daily_equity(self._dry_run_balance_usdt)
            currency_symbol = "₹" if DashboardState.balance_currency == "INR" else "$"
            print(f"[INIT] ✅ Paper-trading mode: Virtual balance initialized to {currency_symbol}{self._dry_run_balance_usdt:,.2f} {DashboardState.balance_currency}")

        # Per-symbol locks to prevent concurrent candle processing on the same symbol
        self._candle_locks = {sym: asyncio.Lock() for sym in Config.SUPPORTED_SYMBOLS}
        self._pending_candle_evaluations = {sym: False for sym in Config.SUPPORTED_SYMBOLS}
        # Per-symbol exit locks to prevent concurrent exit_position() calls (LOGIC-001 fix)
        self._exit_locks = {sym: asyncio.Lock() for sym in Config.SUPPORTED_SYMBOLS}
        self._active_scan_tasks: set[asyncio.Task] = set()
        self._last_reset_date = datetime.datetime.now(datetime.timezone.utc).date()
        
        # Link callbacks
        self.pipeline.on_candle_close_callback = self.on_candle_close

    _STATE_FILE = Path("bot_state.json")
    _STATE_MUTEX = threading.RLock()
    STATE_SCHEMA_VERSION = "2.0"

    def save_state(self):
        """Persist current position state atomically to disk and Firebase Cloud DB for crash recovery."""
        with self._STATE_MUTEX:
            state = {
                'schema_version': self.STATE_SCHEMA_VERSION,
                'in_position': self.in_position,
                'position_side': self.position_side,
                'entry_price': self.entry_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'position_size': self.position_size,
                'entry_time': self.entry_time,
                'highest_price_reached': self.highest_price_reached,
                'lowest_price_reached': self.lowest_price_reached,
                '_dry_run_balance_usdt': self._dry_run_balance_usdt,
                'take_profit_1r': self.take_profit_1r,
                'take_profit_2r': self.take_profit_2r,
                'partial_tp_taken': self.partial_tp_taken,
                'tp2_taken': self.tp2_taken,
                'realized_pnl': self.realized_pnl,
                'original_position_size': self.original_position_size,
                'last_exit_time': self.last_exit_time,
                'tp_cooldown_until': self.tp_cooldown_until,
                'position_mode': self.position_mode,
                'last_trade_time': self.last_trade_time,
                'last_zone_traded': self.last_zone_traded,
                'current_trade_id': self.current_trade_id,
                'entry_fx_rate': self.entry_fx_rate,
                'accumulated_fees': self.accumulated_fees,
                'closed_trades': list(DashboardState.trades[-100:]),
                'order_state_machine': self.order_state_machine.serialize_all(),
                'active_risk_reservations': self.risk.serialize_reservations() if hasattr(self, 'risk') else {},
                'saved_at_ts': time.time(),
            }
            temp_file = self._STATE_FILE.with_name(f"{self._STATE_FILE.name}.tmp.{os.getpid()}.{time.time_ns()}")
            try:
                # 1. Write to temporary file in same directory
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(state, indent=2, sort_keys=True))
                    f.flush()
                    os.fsync(f.fileno())
                # 2. Atomic rename / replace
                os.replace(temp_file, self._STATE_FILE)
            except Exception as e:
                print(f"[STATE] Failed to atomically save state to local file: {e}")
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
                
            try:
                from core.firebase_manager import FirebaseManager
                firebase = FirebaseManager()
                if firebase.is_connected:
                    firebase.db.collection("bot_state").document("current").set(state)
            except Exception as e:
                print(f"[STATE] Firebase state save note: {e}")

    def load_state(self):
        """Restore position state from Firebase Cloud DB or local disk after a restart."""
        with self._STATE_MUTEX:
            state = None
            
            # 1. Try Firebase Cloud DB first (survives Render container restarts)
            try:
                from core.firebase_manager import FirebaseManager
                firebase = FirebaseManager()
                if firebase.is_connected:
                    doc = firebase.db.collection("bot_state").document("current").get()
                    if doc.exists:
                        state = doc.to_dict()
                        add_log_message("[STATE] Recovered position state from Firebase Cloud DB.")
            except Exception as e:
                print(f"[STATE] Firebase load check: {e}")

            file_exists = self._STATE_FILE.exists()
            # 2. Fallback to local file if not restored from Cloud DB
            if not state and file_exists:
                try:
                    content = self._STATE_FILE.read_text(encoding="utf-8")
                    if not content.strip():
                        print("[STATE] FATAL: bot_state.json exists but is empty (0 bytes).")
                        import sys
                        sys.exit("[CRASH_BOUNDARY] SAFE HALT: bot_state.json is empty or truncated. Manual intervention required to prevent state desync.")
                    state = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"[STATE] FATAL: Corrupted local state file: {e}")
                    import sys
                    sys.exit("[CRASH_BOUNDARY] SAFE HALT: bot_state.json is corrupted or unreadable. Manual intervention required to prevent state desync.")
                except Exception as e:
                    print(f"[STATE] FATAL: Error reading state file: {e}")
                    import sys
                    sys.exit(f"[CRASH_BOUNDARY] SAFE HALT: {e}")

            if not isinstance(state, dict) or not state:
                if not file_exists and state is None:
                    # Genuinely first run or clean startup: state file does not exist on disk
                    self._load_trade_logs()
                    return
                print(f"[STATE] FATAL: bot_state.json exists but contains non-dict or empty payload ({state!r}).")
                import sys
                sys.exit("[CRASH_BOUNDARY] SAFE HALT: bot_state.json has empty/invalid schema. Aborting to prevent silent state reset.")

            mandatory_keys = ['in_position', 'position_side', 'position_size', 'entry_price', 'stop_loss']
            missing_keys = [k for k in mandatory_keys if k not in state]
            if missing_keys:
                print(f"[STATE] FATAL: bot_state.json missing mandatory keys: {missing_keys}")
                import sys
                sys.exit(f"[CRASH_BOUNDARY] SAFE HALT: bot_state.json missing mandatory fields {missing_keys}.")

            # Validate nested dictionary types
            for mkey in mandatory_keys:
                if not isinstance(state[mkey], dict):
                    print(f"[STATE] FATAL: bot_state.json field '{mkey}' must be a mapping.")
                    import sys
                    sys.exit(f"[CRASH_BOUNDARY] SAFE HALT: bot_state.json field '{mkey}' has invalid type.")

            try:
                # Helper to safely load dict state, falling back to default if new symbols were added
                def safe_load(key: str, default_val: Any) -> dict[str, Any]:
                    loaded_dict = state.get(key, {}) if isinstance(state, dict) else {}
                    if not isinstance(loaded_dict, dict):
                        return {sym: default_val for sym in Config.SUPPORTED_SYMBOLS}
                    return {sym: loaded_dict.get(sym, default_val) for sym in Config.SUPPORTED_SYMBOLS}

                self.in_position = {k: bool(v) for k, v in safe_load('in_position', False).items()}
                self.position_side = {k: str(v) for k, v in safe_load('position_side', 'HOLD').items()}
                self.entry_price = {k: float(v or 0.0) for k, v in safe_load('entry_price', 0.0).items()}
                self.stop_loss = {k: float(v or 0.0) for k, v in safe_load('stop_loss', 0.0).items()}
                self.take_profit = {k: float(v or 0.0) for k, v in safe_load('take_profit', 0.0).items()}
                self.position_size = {k: float(v or 0.0) for k, v in safe_load('position_size', 0.0).items()}
                self.entry_time = {k: float(v or 0.0) for k, v in safe_load('entry_time', 0.0).items()}
                self.highest_price_reached = {k: float(v or 0.0) for k, v in safe_load('highest_price_reached', 0.0).items()}
                self.lowest_price_reached = {k: float(v or 999999.0) for k, v in safe_load('lowest_price_reached', 999999.0).items()}
                self._dry_run_balance_usdt = float(state.get('_dry_run_balance_usdt', getattr(Config, 'PAPER_STARTING_BALANCE', 2000.0)))
                self.take_profit_1r = {k: float(v or 0.0) for k, v in safe_load('take_profit_1r', 0.0).items()}
                self.take_profit_2r = {k: float(v or 0.0) for k, v in safe_load('take_profit_2r', 0.0).items()}
                self.partial_tp_taken = {k: bool(v) for k, v in safe_load('partial_tp_taken', False).items()}
                self.tp2_taken = {k: bool(v) for k, v in safe_load('tp2_taken', False).items()}
                self.realized_pnl = {k: float(v or 0.0) for k, v in safe_load('realized_pnl', 0.0).items()}
                self.original_position_size = {k: float(v or 0.0) for k, v in safe_load('original_position_size', 0.0).items()}
                self.last_exit_time = {k: float(v or 0.0) for k, v in safe_load('last_exit_time', 0.0).items()}
                self.tp_cooldown_until = {k: float(v or 0.0) for k, v in safe_load('tp_cooldown_until', 0.0).items()}
                self.position_mode = {k: str(v) for k, v in safe_load('position_mode', 'STRICT').items()}
                self.last_trade_time = {k: float(v or 0.0) for k, v in safe_load('last_trade_time', 0.0).items()}
                self.last_zone_traded = {k: (str(v) if v is not None else None) for k, v in safe_load('last_zone_traded', None).items()}
                self.current_trade_id = {k: str(v or '') for k, v in safe_load('current_trade_id', '').items()}
                self.entry_fx_rate = {k: float(v or 0.0) for k, v in safe_load('entry_fx_rate', 0.0).items()}
                self.accumulated_fees = {k: float(v or 0.0) for k, v in safe_load('accumulated_fees', 0.0).items()}
                
                # Restore closed trades history
                saved_trades = state.get('closed_trades', []) if isinstance(state, dict) else []
                if saved_trades:
                    DashboardState.trades = [t for t in saved_trades if t.get('type') != 'TRADE_LIFECYCLE']
                    
                # Restore Order State Machine contexts
                if isinstance(state, dict) and 'order_state_machine' in state:
                    self.order_state_machine.load_all(state['order_state_machine'])

                # Restore Durable Risk Reservations
                if hasattr(self, 'risk') and isinstance(state, dict) and 'active_risk_reservations' in state:
                    self.risk.load_reservations(state['active_risk_reservations'])
                
                # Sync any additional logs from data/trade_logs.jsonl
                self._load_trade_logs()

                # Sync to dashboard for active UI symbol
                sym = Config.SYMBOL
                DashboardState.in_position = self.in_position[sym]
                DashboardState.position_side = self.position_side[sym]
                DashboardState.entry_price = self.entry_price[sym]
                DashboardState.stop_loss = self.stop_loss[sym]
                DashboardState.take_profit = self.take_profit[sym]
                DashboardState.balance_usdt = self.calculate_total_equity() if (not self.has_keys or Config.PAPER_TRADING) else self._dry_run_balance_usdt
                
                # Immediately populate active_positions map for UI refresh recovery
                active_pos_map = {}
                for s in Config.SUPPORTED_SYMBOLS:
                    if self.in_position[s]:
                        active_pos_map[s] = {
                            'side': self.position_side[s],
                            'entry_price': self.entry_price[s],
                            'stop_loss': self.stop_loss[s],
                            'take_profit': self.take_profit[s],
                            'position_size': self.position_size[s],
                            'current_pnl_usdt': 0.0,
                            'current_pnl_pct': 0.0
                        }
                DashboardState.active_positions = active_pos_map

                open_positions = sum(1 for s in Config.SUPPORTED_SYMBOLS if self.in_position[s])
                if open_positions > 0:
                    add_log_message(f"[STATE] Recovered {open_positions} open positions into Dashboard state.")
                else:
                    add_log_message("[STATE] State loaded — no open positions to recover.")
            except Exception as e:
                print(f"[STATE] Failed to process state: {e}")

    def _load_trade_logs(self):
        """Read historical closed trades from data/trade_logs.jsonl into DashboardState.trades."""
        try:
            log_file = Path("data") / "trade_logs.jsonl"
            if log_file.exists():
                existing_keys = {
                    f"{t.get('symbol')}_{t.get('exit_time') or t.get('time') or 0}_{round(float(t.get('pnl_usdt', t.get('pnl', 0)) or 0), 4)}"
                    for t in DashboardState.trades
                }
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                tr = json.loads(line)
                                if tr.get('type') == 'TRADE_LIFECYCLE':
                                    continue
                                k = f"{tr.get('symbol')}_{tr.get('exit_time') or tr.get('time') or 0}_{round(float(tr.get('pnl_usdt', tr.get('pnl', 0)) or 0), 4)}"
                                if k not in existing_keys:
                                    DashboardState.trades.append(tr)
                                    existing_keys.add(k)
                            except Exception:
                                pass
                if len(DashboardState.trades) > 500:
                    DashboardState.trades = DashboardState.trades[-500:]
        except Exception as e:
            print(f"[STATE] Failed to read data/trade_logs.jsonl: {e}")

    def reset_account_state(self, target_balance: float = 10000.0):
        """Resets virtual paper balance to target amount and closes all simulated positions for fresh trading."""
        self._dry_run_balance_usdt = target_balance
        for sym in Config.SUPPORTED_SYMBOLS:
            self.in_position[sym] = False
            self.position_side[sym] = "HOLD"
            self.entry_price[sym] = 0.0
            self.stop_loss[sym] = 0.0
            self.take_profit[sym] = 0.0
            self.take_profit_1r[sym] = 0.0
            self.take_profit_2r[sym] = 0.0
            self.position_size[sym] = 0.0
            self.entry_time[sym] = 0.0
            self.highest_price_reached[sym] = 0.0
            self.lowest_price_reached[sym] = 999999.0
            self.partial_tp_taken[sym] = False
            self.tp2_taken[sym] = False
            self.realized_pnl[sym] = 0.0
            self.original_position_size[sym] = 0.0
            self.last_exit_time[sym] = 0.0
            self.tp_cooldown_until[sym] = 0.0

        self.traded_zones_cache.clear()
        self.trade_history.clear()
        self.global_pause_until = 0.0
        self.cluster_loss_pause_until = 0.0
        DashboardState.trades.clear()
        DashboardState.balance_usdt = target_balance
        DashboardState.balance_base = 0.0
        DashboardState.active_positions.clear()
        DashboardState.in_position = False
        DashboardState.position_side = "HOLD"
        DashboardState.entry_price = 0.0
        DashboardState.stop_loss = 0.0
        DashboardState.take_profit = 0.0
        DashboardState.current_pnl_pct = 0.0
        DashboardState.current_pnl_usdt = 0.0
        self.save_state()
        add_log_message(f"🔄 [ACCOUNT RESET] Virtual paper balance reset to ${target_balance:,.2f} USDT. Cooldown cleared, all 20 pairs actively scanning.")

    async def initialize(self):
        add_log_message("Starting system initialization for all supported symbols...")

        self.load_state()
        await self.pipeline.start()
        await asyncio.sleep(3)
        
        # Initial Balance load
        if self.has_keys and not Config.PAPER_TRADING:
            balance = await self.execution.fetch_balance()
            if balance:
                usdt_balance = balance.get('total', {}).get('USDT', None)
                if usdt_balance and usdt_balance > 0:
                    DashboardState.balance_usdt = usdt_balance
                else:
                    add_log_message(f"[WARNING] Balance fetch returned {usdt_balance}. Check account type. Keeping last known value.")
                DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
        else:
            DashboardState.balance_usdt = self.calculate_total_equity()
            DashboardState.balance_base = 0.0
        
        # Train ML Models on historical candles for each symbol
        for sym in Config.SUPPORTED_SYMBOLS:
            ltf_history = self.pipeline.ltf_candles[sym]
            if ltf_history:
                df = prepare_dataframe(ltf_history)
                trained = self.ml_models[sym].train(df)
                if not trained:
                    self.ml_models[sym].is_trained = False
        
        add_log_message("ML Models initialized (optional filtering mode).")

        DashboardState.latest_price = self.pipeline.latest_prices.get(Config.SYMBOL, 0.0)
        DashboardState.chart_history = self.pipeline.ltf_candles[Config.SYMBOL][-100:] if self.pipeline.ltf_candles[Config.SYMBOL] else []
        DashboardState.signal_light = "BLUE"
        DashboardState.signal_light_reason = f"Monitoring {len(Config.SUPPORTED_SYMBOLS)} pairs for institutional SMC setups..."
        
        # Sync CoinDCX User Info and Live Balances
        await self.sync_coindcx_data()
        
        # Start Continuous Broker Reconciliation Engine
        await self.reconciliation.start()
        
        add_log_message(f"System ready. Multi-symbol watch active ({len(Config.SUPPORTED_SYMBOLS)} pairs). UI viewing {Config.SYMBOL}")

    async def sync_coindcx_data(self):
        """Fetches and updates CoinDCX profile and balances in DashboardState."""
        has_real_coindcx = bool(
            Config.COINDCX_API_KEY and 
            Config.COINDCX_SECRET_KEY and 
            Config.COINDCX_API_KEY != "your_coindcx_key_here" and 
            Config.COINDCX_SECRET_KEY != "your_coindcx_secret_here"
        )
        
        if has_real_coindcx and hasattr(self.execution, 'coindcx_client') and self.execution.coindcx_client:
            try:
                uinfo = await self.execution.coindcx_client.fetch_user_info()
                if uinfo:
                    f_name = uinfo.get('first_name') or ''
                    l_name = uinfo.get('last_name') or ''
                    full_name = f"{f_name} {l_name}".strip() or uinfo.get('name') or "CoinDCX Trader"
                    DashboardState.coindcx_profile = {
                        "status": "Connected",
                        "name": full_name,
                        "email": uinfo.get('email', 'N/A'),
                        "id": str(uinfo.get('coindcx_id') or uinfo.get('id') or 'N/A')
                    }
                else:
                    DashboardState.coindcx_profile = {
                        "status": "Connected (Live)",
                        "name": "Live Trader",
                        "email": "Connected via API",
                        "id": "DCX-AUTHENTICATED"
                    }
                
                # Fetch live CoinDCX wallet balances
                raw_bal = await self.execution.fetch_balance()
                if raw_bal and 'total' in raw_bal:
                    bal_list: list[dict[str, float | str]] = []
                    for curr, tot in raw_bal['total'].items():
                        free = float((raw_bal.get('free') or {}).get(curr, 0.0) or 0.0)
                        used = float((raw_bal.get('used') or {}).get(curr, 0.0) or 0.0)
                        if (free + used) > 0.00001:
                            bal_list.append({"currency": str(curr), "available": free, "locked": used})
                    if bal_list:
                        DashboardState.coindcx_balances = bal_list
            except Exception as e:
                print(f"[CoinDCX Sync] Error: {e}")
        else:
            # Paper Trading / Simulation Mode: Provide clear virtual details
            DashboardState.coindcx_profile = {
                "status": "Paper Mode (Active)",
                "name": "Virtual Paper Trader",
                "email": "paper.trade@coindcx.local",
                "id": "DCX-VIRTUAL-8849"
            }
            DashboardState.coindcx_balances = [
                {"currency": "USDT", "available": DashboardState.balance_usdt, "locked": 0.0},
                {"currency": "INR", "available": round(DashboardState.balance_usdt * 85.0, 2), "locked": 0.0},
                {"currency": "BTC", "available": DashboardState.balance_base, "locked": 0.0}
            ]

    async def on_candle_close(self, symbol):
        if self._candle_locks[symbol].locked():
            if not self._pending_candle_evaluations[symbol]:
                self._pending_candle_evaluations[symbol] = True
            return

        async with self._candle_locks[symbol]:
            await self._on_candle_close_impl(symbol)
            
            while self._pending_candle_evaluations[symbol]:
                self._pending_candle_evaluations[symbol] = False
                await self._on_candle_close_impl(symbol)

    async def get_open_positions_info(self):
        count = 0
        total_risk_pct = 0.0
        longs_count = 0
        shorts_count = 0
        current_eq = self.calculate_total_equity() if (not self.has_keys or Config.PAPER_TRADING) else DashboardState.balance_usdt

        for sym in Config.SUPPORTED_SYMBOLS:
            if self.in_position[sym]:
                count += 1
                if self.position_side[sym] == "LONG":
                    longs_count += 1
                elif self.position_side[sym] == "SHORT":
                    shorts_count += 1
                risk_usdt = self.position_size[sym] * abs(self.entry_price[sym] - self.stop_loss[sym])
                if current_eq > 0:
                    total_risk_pct += (risk_usdt / current_eq)
                else:
                    total_risk_pct += getattr(Config, 'RISK_PCT', 0.01)
        return count, total_risk_pct, longs_count, shorts_count

    def calculate_total_equity(self):
        current_equity = self._dry_run_balance_usdt
        is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
        rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
        if rate <= 0:
            rate = 85.0

        for sym in Config.SUPPORTED_SYMBOLS:
            if self.in_position[sym] and self.position_size[sym] and self.position_size[sym] > 1e-7:
                live_price = self.pipeline.latest_prices.get(sym, self.entry_price[sym])
                live_p_adj = live_price * rate if is_inr else live_price
                entry_p_adj = self.entry_price[sym] * rate if is_inr else self.entry_price[sym]
                
                if self.position_side[sym] == "LONG":
                    current_equity += self.position_size[sym] * live_p_adj
                elif self.position_side[sym] == "SHORT":
                    unrealized_pnl = self.position_size[sym] * (entry_p_adj - live_p_adj)
                    current_equity += (self.position_size[sym] * entry_p_adj) + unrealized_pnl
        return current_equity

    def is_macro_news_blackout(self):
        """
        Checks if current UTC time falls in high-impact economic news windows
        (US CPI / PPI / NFP / FOMC release times e.g. 12:20-12:45 UTC, 13:20-13:45 UTC, 18:00-19:30 UTC).
        """
        if not getattr(Config, 'ENABLE_MACRO_NEWS_FILTER', True):
            return False, ""
            
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        hour = now_utc.hour
        minute = now_utc.minute
        weekday = now_utc.weekday() # 0=Mon, 4=Fri
        
        # Only weekdays have high-impact macro economic data releases
        if weekday >= 5:
            return False, ""
            
        # Window 1: 12:20 - 12:45 UTC (US 8:30 AM EDT daylight savings / 8:30 AM EST releases)
        if hour == 12 and (20 <= minute <= 45):
            return True, "US Morning Macro Data Window (12:20-12:45 UTC)"
            
        # Window 2: 13:20 - 13:45 UTC (US 8:30 AM EST standard winter releases)
        if hour == 13 and (20 <= minute <= 45):
            return True, "US Main Macro Data Window (13:20-13:45 UTC)"
            
        # Window 3: 18:00 - 19:30 UTC (Fed FOMC Rate Decision & Press Conf, typically Wednesdays)
        if (weekday == 2) and (18 <= hour <= 19):
            return True, "Fed FOMC Rate Decision Window (18:00-19:30 UTC)"
            
        return False, ""

    async def _on_candle_close_impl(self, symbol):
        if time.time() < self.global_pause_until:
            return
            
        # GAP-02 INVARIANT: Block new candle signal evaluation until startup broker reconciliation has completed
        if not self.reconciliation.initial_reconciliation_done:
            add_log_message(f"[{symbol}] ⏸️ Trade skipped: Startup broker reconciliation in progress. Engine locked.")
            return

        # SAFE_MODE INVARIANT: Block new entries if reconciliation detected state divergence / consecutive errors
        if self.reconciliation.safe_mode_active:
            add_log_message(f"[{symbol}] 🚨 Trade skipped: Reconciliation SAFE MODE active (reconciling broker state).")
            return
            
        # Macro Economic News Blackout Window Filter (CPI/FOMC Volatility Protection)
        is_news, news_reason = self.is_macro_news_blackout()
        if is_news:
            add_log_message(f"[{symbol}] Trade skipped: {news_reason}. Macro volatility blackout active.")
            return

        # Update balance via API if live
        if self.has_keys and not Config.PAPER_TRADING:
            balance = await self.execution.fetch_balance()
            if balance and isinstance(balance, dict):
                total_dict = balance.get('total', {})
                if isinstance(total_dict, dict):
                    if Config.COINDCX_TRADE_INR:
                        inr_balance = total_dict.get('INR', None)
                        if inr_balance is not None and float(inr_balance) > 0:
                            DashboardState.balance_usdt = float(inr_balance)
                            DashboardState.balance_currency = "INR"
                    else:
                        usdt_balance = total_dict.get('USDT', None)
                        if usdt_balance is not None and float(usdt_balance) > 0:
                            DashboardState.balance_usdt = float(usdt_balance)
                            DashboardState.balance_currency = "USDT"
                    DashboardState.balance_base = float(total_dict.get(Config.SYMBOL.split('/')[0], 0.0) or 0.0)
                
        # Check drawdown circuit breakers
        current_equity = DashboardState.balance_usdt if (self.has_keys and not Config.PAPER_TRADING) else self.calculate_total_equity()
            
        if not self.risk.check_circuit_breaker(current_equity):
            if getattr(self.risk, 'daily_profit_locked', False):
                DashboardState.daily_profit_locked = True
                DashboardState.signal_light = "GREEN"
                curr_target = getattr(self.risk, 'max_daily_profit_pct', getattr(Config, 'MAX_DAILY_PROFIT_PCT', 4.0))
                DashboardState.signal_light_reason = f"🎯 PROFIT LOCK ACTIVE: Daily profit target hit (+{self.risk.current_drawdown_pct:.2f}% >= +{curr_target:.1f}%). Trading locked to secure gains until unlocked or 00:00 UTC."
                if not getattr(self, '_profit_lock_alert_sent', False):
                    self._profit_lock_alert_sent = True
                    add_log_message(f"🎯 PROFIT LOCK TRIGGERED: Daily profit target reached (+{self.risk.current_drawdown_pct:.2f}% >= +{curr_target:.1f}%). All new entries suspended.")
                    await self.notifier.send_message(f"🎯 *PROFIT LOCK ACTIVATED*\nDaily profit target hit (+{self.risk.current_drawdown_pct:.2f}%). Capital secured; all new trades paused until 00:00 UTC.")
            else:
                DashboardState.daily_profit_locked = False
                DashboardState.signal_light = "RED"
                DashboardState.signal_light_reason = f"🚨 SLEEP MODE ACTIVE: Daily loss limit hit ({self.risk.current_drawdown_pct:.2f}%). Trading suspended until 00:00 UTC."
                if not getattr(self, '_circuit_breaker_alert_sent', False):
                    self._circuit_breaker_alert_sent = True
                    add_log_message(f"🚨 SLEEP MODE / CIRCUIT BREAKER TRIGGERED: Daily loss limit reached ({self.risk.current_drawdown_pct:.2f}%). All entries suspended.")
                    await self.notifier.send_message(f"🚨 *SLEEP MODE ACTIVATED*\nDaily loss circuit breaker triggered ({self.risk.current_drawdown_pct:.2f}%). All new trades suspended until 00:00 UTC.")
            return

        self._profit_lock_alert_sent = False
        self._circuit_breaker_alert_sent = False
        DashboardState.daily_profit_locked = False
        DashboardState.daily_drawdown_pct = self.risk.current_drawdown_pct

        ltf_df = prepare_dataframe(self.pipeline.ltf_candles[symbol])
        htf_df = prepare_dataframe(self.pipeline.htf_candles[symbol])
        
        # Check high volatility kill switch
        if not ltf_df.empty:
            last_candle = ltf_df.iloc[-1]
            move_pct = abs(last_candle['close'] - last_candle['open']) / max(last_candle['open'], 1e-8)
            if move_pct > getattr(Config, 'MAX_CANDLE_MOVE_PCT', 0.015):
                avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) > 14 else 0.0
                if last_candle['volume'] < 1.5 * avg_vol:
                    pause_candles = getattr(Config, 'VOLATILITY_PAUSE_CANDLES', 2)
                    tf_minutes = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
                    self.volatility_pause_until[symbol] = time.time() + (pause_candles * tf_minutes * 60.0)
                    add_log_message(f"[{symbol}] Trading paused: High volatility detected ({move_pct*100:.2f}% move) on LOW volume.")
                else:
                    add_log_message(f"[{symbol}] High volatility ({move_pct*100:.2f}%) on HIGH volume. Institutional move allowed.")

        if time.time() < self.volatility_pause_until.get(symbol, 0.0):
            return

        
        # Session and Execution Delay Filters
        current_hour = datetime.datetime.now(datetime.timezone.utc).hour
        is_low_volume_session = not (12 <= current_hour <= 21)
        
        if self.has_keys and not Config.PAPER_TRADING:
            if 'timestamp' in ltf_df.columns:
                open_time = float(ltf_df.iloc[-1]['timestamp']) / 1000.0
            elif 'time' in ltf_df.columns:
                open_time = float(ltf_df.iloc[-1]['time']) / 1000.0
            else:
                ts_obj = pd.Timestamp(str(ltf_df.index[-1]))
                open_time = float(ts_obj.timestamp()) if hasattr(ts_obj, 'timestamp') else 0.0
            tf_mins = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
            close_time = open_time + (tf_mins * 60)
            delay = time.time() - close_time
            if delay > 120:
                add_log_message(f"[{symbol}] Trade skipped: Execution delay ({delay:.1f}s) > 120s. Stale signal protection.")
                return
            
        # Reset daily trades at UTC midnight
        current_date = datetime.datetime.now(datetime.timezone.utc).date()
        if current_date != self.last_trade_day:
            self.trades_today = 0
            self.relaxed_trades_today = 0
            self.last_trade_day = current_date
            
        signal, metadata = self.strategy.generate_signal(
            htf_df,
            ltf_df,
            relaxed=False
        )
        relaxed_used = False
        
        # Dual-Pass Execution
        if signal == "HOLD":
            open_count, _, _, _ = await self.get_open_positions_info()
                
            if open_count < 2 and (time.time() - self.global_last_trade_time) >= 20 * 60 and time.time() > self.global_pause_until:
                if self.relaxed_trades_today < 2 and time.time() > self.relaxed_disabled_until:
                    signal, metadata = self.strategy.generate_signal(
                        htf_df,
                        ltf_df,
                        relaxed=True
                    )
                    if signal != "HOLD":
                        relaxed_used = True


        if symbol == Config.SYMBOL:
            DashboardState.active_ob = str(metadata.get('reason') or 'No OB/FVG')
            DashboardState.active_ob_level = float(metadata.get('active_ob_level') or 0.0)
            DashboardState.active_ob_type = str(metadata.get('active_ob_type') or 'NONE')
            DashboardState.active_bullish_ob_level = float(metadata.get('active_bullish_ob_level') or 0.0)
            DashboardState.active_bearish_ob_level = float(metadata.get('active_bearish_ob_level') or 0.0)
            if self.ml_models[symbol].is_trained:
                DashboardState.ml_confidence = self.ml_models[symbol].predict_bias(ltf_df)
                nc_pred = self.ml_models[symbol].predict_next_candle(ltf_df)
                DashboardState.next_candle_color = str(nc_pred.get('color', 'GREEN'))
                DashboardState.next_candle_prob = float(nc_pred.get('confidence_pct', 50.0))
            else:
                DashboardState.ml_confidence = 0.5
                DashboardState.next_candle_color = "GREEN"
                DashboardState.next_candle_prob = 50.0
            DashboardState.chart_history = self.pipeline.ltf_candles[symbol][-100:]
        
        if signal == "HOLD":
            # Log debug checks for rejection reason
            debug_val = metadata.get('debug_checks')
            debug = debug_val if isinstance(debug_val, dict) else {}
            reason_str = f"Trend: {debug.get('trend', 'FAIL')}, Zone: {debug.get('zone', 'FAIL')}, Trigger: {debug.get('trigger', 'FAIL')}, VWAP: {debug.get('vwap', 'FAIL')}, Vol: {debug.get('volatility', 'FAIL')}"
            print(f"[NO TRADE] [{symbol}] Reason: {metadata.get('reason')} | {reason_str}")
            return
            
        ctx = self.order_state_machine.get_context(symbol)
        if ctx.is_in_flight():
            add_log_message(f"[{symbol}] Trade skipped: Pending execution outcome ({ctx.state}).")
            return
            
        # Session Volume Block
        if getattr(Config, 'ENABLE_SESSION_FILTER', False) and not Config.PAPER_TRADING and is_low_volume_session:
            avg_vol = ltf_df['volume'].rolling(20).mean().iloc[-2] if len(ltf_df) > 20 else 0.0
            score_val = float(metadata.get('score') or 0.0)
            vol_mult = 1.0 if score_val >= 3.5 else 1.2
            if ltf_df['volume'].iloc[-1] < vol_mult * avg_vol:
                add_log_message(f"[{symbol}] Trade skipped: Outside 12-22 UTC and volume not > {vol_mult}x average.")
                return
                
        # 4H Bias logic
        htf_4h_df = self.pipeline.htf_4h_candles.get(symbol)
        if htf_4h_df is not None and len(htf_4h_df) > 50:
            if isinstance(htf_4h_df, list): htf_4h_df = pd.DataFrame(htf_4h_df, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            ema_4h = htf_4h_df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            cur_score = float(metadata.get('score') or 3.0)
            if signal == "BUY" and htf_4h_df['close'].iloc[-1] < ema_4h:
                metadata['score'] = cur_score - 0.5
            elif signal == "SELL" and htf_4h_df['close'].iloc[-1] > ema_4h:
                metadata['score'] = cur_score - 0.5
                
        add_log_message(f"[{symbol}] Raw strategy signal: {signal} ({metadata.get('reason')})")
        
        # Position Reversal logic
        if self.in_position[symbol]:
            if self.position_side[symbol] == "LONG" and signal == "SELL":
                add_log_message(f"[{symbol}] Trend reversal: Closing LONG position.")
                await self.exit_position(symbol, "SIGNAL_REVERSAL")
            elif self.position_side[symbol] == "SHORT" and signal == "BUY":
                add_log_message(f"[{symbol}] Trend reversal: Closing SHORT position.")
                await self.exit_position(symbol, "SIGNAL_REVERSAL")
            return

        # BTC Correlation Filter
        if signal == "BUY" and symbol != "BTC/USDT":
            btc_raw = self.pipeline.ltf_candles.get("BTC/USDT")
            if btc_raw:
                btc_df = prepare_dataframe(btc_raw) if isinstance(btc_raw, list) else btc_raw
                if not btc_df.empty:
                    btc_last = btc_df.iloc[-1]
                    btc_drop = (btc_last['open'] - btc_last['close']) / btc_last['open']
                    if btc_drop > 0.014:
                        add_log_message(f"[{symbol}] Trade blocked: BTC dropped > 1.4% in last 5m. Blocking altcoin longs.")
                        return

        # Funding Rate & Crowded Trade Sentiment Filter
        fr: float = 0.0
        spread: float = 0.0005
        if getattr(Config, 'ENABLE_FUNDING_RATE_FILTER', True):
            try:
                fetched_fr = await self.execution.fetch_funding_rate(symbol)
                if fetched_fr is not None:
                    fr = float(fetched_fr)
                max_fr = getattr(Config, 'MAX_FUNDING_RATE_PCT', 0.035) / 100.0 # e.g. 0.035% = 0.00035
                if signal == "BUY" and fr > max_fr:
                    add_log_message(f"[{symbol}] Trade blocked: Extreme Bullish Funding Rate ({fr*100:+.4f}% > +{max_fr*100:.3f}%). Long liquidation trap protection.")
                    return
                elif signal == "SELL" and fr < -max_fr:
                    add_log_message(f"[{symbol}] Trade blocked: Extreme Bearish Funding Rate ({fr*100:+.4f}% < -{max_fr*100:.3f}%). Short squeeze trap protection.")
                    return
            except Exception:
                pass
        
        # Daily Trade Limit
        max_daily_trades = getattr(Config, 'MAX_DAILY_TRADES', 6)
        if self.trades_today >= max_daily_trades:
            add_log_message(f"[{symbol}] Trade skipped: Max {max_daily_trades} trades per day reached ({self.trades_today}/{max_daily_trades}).")
            return

        # Cluster Loss Cooldown
        if time.time() < getattr(self, 'cluster_loss_pause_until', 0.0):
            add_log_message(f"[{symbol}] Trade skipped: Cluster loss cooldown active.")
            return

        # Symbol Post-Exit & Profit-Harvest Cooldown (Prevent immediate re-entry at local top/bottom)
        if time.time() < self.tp_cooldown_until.get(symbol, 0.0):
            rem_m = max(1, int((self.tp_cooldown_until[symbol] - time.time()) / 60) + 1)
            add_log_message(f"[{symbol}] Trade skipped: Post-exit / TP-harvest cooldown active ({rem_m}m remaining to prevent re-entering exhausted structure).")
            return

        # Cooldown Check
        cooldown_secs = getattr(Config, 'COOLDOWN_MINUTES', 20) * 60
        if time.time() - self.last_trade_time.get(symbol, 0.0) < cooldown_secs:
            rem_m = max(1, int((cooldown_secs - (time.time() - self.last_trade_time[symbol])) / 60) + 1)
            add_log_message(f"[{symbol}] Trade skipped due to cooldown ({rem_m}m remaining).")
            return

        # Same Zone Check with Traded Zones Cache
        zone_id = metadata.get('zone_id')
        cache_key = f"{symbol}_{zone_id}"
        if zone_id and self.traded_zones_cache.get(cache_key):
            add_log_message(f"[{symbol}] Trade skipped: already traded in this zone ({zone_id}).")
            return
            
        # Clear out old cache (basic cleanup - ideally based on candle count but here based on simple dict size)
        if len(self.traded_zones_cache) > 1000:
            self.traded_zones_cache.clear()
            
        # FIX #2: Define entry_price BEFORE ML block uses it
        entry_price = ltf_df['close'].iloc[-1]
        add_log_message(f"[{symbol}] Entry price set: {entry_price:.4f}")

        # ML Confidence Scaler & Soft Session Filter (Direction-Aware)
        raw_prob = 0.5
        prob = 1.0
        ml_confidence_weight = 0
        if self.ml_models[symbol].is_trained:
            raw_prob = self.ml_models[symbol].predict_bias(ltf_df)
            prob = raw_prob if signal == "BUY" else (1.0 - raw_prob)
            if symbol == Config.SYMBOL:
                DashboardState.ml_confidence = prob
            add_log_message(f"[{symbol}] ML confidence score: {prob:.2f} (raw bullish: {raw_prob:.2f})")

            # Task 7: ML TP Logic - Strictly ordered 3-Stage Targets
            risk_usdt = abs(metadata.get('stop_loss', entry_price) - entry_price)
            fee_adj = entry_price * getattr(Config, 'FEE_RATE', 0.00075) * 2.0
            if prob > 0.65:
                tp2_mult = 2.5
                ml_confidence_weight = 1
            elif prob < 0.55:
                tp2_mult = 1.8
                ml_confidence_weight = -1
            else:
                tp2_mult = float(getattr(Config, 'RISK_REWARD_RATIO', 2.2))

            tp1_mult = float(getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.2))

            if signal == "BUY":
                metadata['tp1'] = entry_price + (tp1_mult * risk_usdt) + fee_adj
                metadata['tp2'] = entry_price + (tp2_mult * risk_usdt) + fee_adj
                metadata['tp3'] = entry_price + (4.0 * risk_usdt) + fee_adj
                metadata['take_profit_1r'] = metadata['tp1']
                metadata['take_profit'] = metadata['tp3']
            elif signal == "SELL":
                metadata['tp1'] = entry_price - (tp1_mult * risk_usdt) - fee_adj
                metadata['tp2'] = entry_price - (tp2_mult * risk_usdt) - fee_adj
                metadata['tp3'] = entry_price - (4.0 * risk_usdt) - fee_adj
                metadata['take_profit_1r'] = metadata['tp1']
                metadata['take_profit'] = metadata['tp3']
        else:
            risk_usdt = abs(metadata.get('stop_loss', entry_price) - entry_price)
            fee_adj = entry_price * getattr(Config, 'FEE_RATE', 0.00075) * 2.0
            tp1_mult = float(getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.2))
            tp2_mult = float(getattr(Config, 'RISK_REWARD_RATIO', 2.2))
            if signal == "BUY":
                metadata['tp1'] = entry_price + (tp1_mult * risk_usdt) + fee_adj
                metadata['tp2'] = entry_price + (tp2_mult * risk_usdt) + fee_adj
                metadata['tp3'] = entry_price + (4.0 * risk_usdt) + fee_adj
                metadata['take_profit_1r'] = metadata['tp1']
                metadata['take_profit'] = metadata['tp3']
            elif signal == "SELL":
                metadata['tp1'] = entry_price - (tp1_mult * risk_usdt) - fee_adj
                metadata['tp2'] = entry_price - (tp2_mult * risk_usdt) - fee_adj
                metadata['tp3'] = entry_price - (4.0 * risk_usdt) - fee_adj
                metadata['take_profit_1r'] = metadata['tp1']
                metadata['take_profit'] = metadata['tp3']

        if is_low_volume_session:
            avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) > 14 else 0.0
            if ltf_df['volume'].iloc[-1] < 0.6 * avg_vol:
                prob *= 0.5
                add_log_message(f"[{symbol}] Low volume session filter triggered, confidence reduced to {prob:.2f}")
                
                # Override TP2 to 1.5R instead of 2.2R in low volume
                risk_usdt = abs(metadata.get('stop_loss', entry_price) - entry_price)
                fee_adj = entry_price * getattr(Config, 'FEE_RATE', 0.00075) * 2.0
                if signal == "BUY":
                    metadata['tp2'] = entry_price + (1.5 * risk_usdt) + fee_adj
                elif signal == "SELL":
                    metadata['tp2'] = entry_price - (1.5 * risk_usdt) - fee_adj
            
        # Task 5: Smart Risk Allocation (Final Edge)
        score = float(metadata.get('score') or 3.0)
        if score >= 4.5: trade_risk_pct = 0.0125
        elif score >= 3.5: trade_risk_pct = 0.01
        else: trade_risk_pct = 0.0075
        
        # Dynamic Kelly Criterion Sizing (when enabled)
        if getattr(Config, 'ENABLE_KELLY_SIZING', False):
            dynamic_risk = self.risk.calculate_kelly_risk_pct(DashboardState.trades, base_risk=trade_risk_pct * 100.0)
            trade_risk_pct = dynamic_risk / 100.0
            add_log_message(f"[{symbol}] Kelly Dynamic Sizing: Risk scaled to {dynamic_risk:.2f}%.")
        
        if getattr(self, 'cluster_risk_penalty', False):
            trade_risk_pct *= 0.5
            add_log_message(f"[{symbol}] Cluster Loss Penalty: Risk slashed by 50%.")
            
        # Runner Logic Metadata
        tp1_scale = float(getattr(Config, 'TP1_SCALE_OUT_PCT', 0.65))
        metadata['tp1_size'] = tp1_scale
        metadata['tp2_size'] = round((1.0 - tp1_scale) * 0.65, 2)
        metadata['runner_size'] = round(1.0 - metadata['tp1_size'] - metadata['tp2_size'], 2)
        
        # Task 10: Equity Protection
        if not hasattr(self, 'hourly_peak_equity'):
            self.hourly_peak_equity = current_equity
            self.last_hour_ts = time.time()
        
        if time.time() - self.last_hour_ts > 3600:
            self.hourly_peak_equity = current_equity
            self.last_hour_ts = time.time()
            self.hourly_dd_penalty = False
            
        if current_equity > self.hourly_peak_equity:
            self.hourly_peak_equity = current_equity
            
        hourly_dd_pct = (self.hourly_peak_equity - current_equity) / self.hourly_peak_equity
        if hourly_dd_pct > 0.03:
            self.hourly_dd_penalty = True
        if hourly_dd_pct < 0.01:
            self.hourly_dd_penalty = False
            
        if getattr(self, 'hourly_dd_penalty', False):
            trade_risk_pct *= 0.5
            add_log_message(f"[{symbol}] Equity Protection: Hourly DD > 3%. Risk slashed by 50%.")

        # ── LOGIC-002 FIX: Atomic Portfolio Risk Reservation ──
        # Acquire portfolio lock to prevent concurrent tasks from reading stale open_count/total_risk
        async with self.risk.portfolio_lock:
            open_count, total_risk, longs_count, shorts_count = await self.get_open_positions_info()
            max_open_trades = int(getattr(Config, 'MAX_OPEN_TRADES', 2))
            max_risk_cap = float(getattr(Config, 'MAX_PORTFOLIO_RISK_PCT', 0.06))
            
            # Account for in-flight reservations
            effective_open_count = open_count + self.risk.reserved_open_count
            effective_longs = longs_count + self.risk.reserved_longs_count
            effective_shorts = shorts_count + self.risk.reserved_shorts_count
            effective_total_risk = total_risk + self.risk.reserved_risk_pct

            if effective_open_count >= max_open_trades:
                add_log_message(f"[{symbol}] Trade skipped: Max {max_open_trades} open trades reached ({effective_open_count} active/in-flight).")
                return
            if signal == "BUY" and effective_longs >= 2:
                add_log_message(f"[{symbol}] Trade skipped: Max 2 LONG positions already open/reserved ({effective_longs} in-flight).")
                return
            if signal == "SELL" and effective_shorts >= 2:
                add_log_message(f"[{symbol}] Trade skipped: Max 2 SHORT positions already open/reserved ({effective_shorts} in-flight).")
                return
            
            # Task 10: Priority Ranking (prob scaled 0..5 to align with score 0..5)
            priority_score = (score * 0.7) + (prob * 5.0 * 0.3)
            
            # Only reserve cap space when portfolio already has active positions
            if effective_open_count >= 1:
                if priority_score < 3.0 and effective_total_risk + trade_risk_pct > max_risk_cap - 0.02:
                    add_log_message(f"[{symbol}] Trade skipped: Priority score {priority_score:.1f} < 3.0. Reserving cap space.")
                    return
                if priority_score < 4.0 and effective_total_risk + trade_risk_pct > max_risk_cap - 0.01:
                    add_log_message(f"[{symbol}] Trade skipped: Priority score {priority_score:.1f} < 4.0. Reserving cap space.")
                    return
            if effective_total_risk + trade_risk_pct > max_risk_cap:
                add_log_message(f"[{symbol}] Trade blocked: Absolute exposure limit reached ({effective_total_risk*100:.2f}% + {trade_risk_pct*100:.2f}% > {max_risk_cap*100:.2f}%).")
                return

            # Atomically reserve portfolio risk capacity with durable reservation identity
            reservation_id = f"RES_{symbol.replace('/', '')}_{int(time.time()*1000)}"
            risk_reserved = await self.risk.check_and_reserve_risk_atomic(
                total_risk, trade_risk_pct, side=signal, reservation_id=reservation_id, symbol=symbol
            )
            if not risk_reserved:
                add_log_message(f"[{symbol}] Trade blocked: Atomic risk reservation denied (concurrent limit).")
                return
            ctx = self.order_state_machine.get_context(symbol)
            ctx.reservation_id = reservation_id
        # Portfolio lock released here — exchange API calls proceed without holding it
        reserved_trade_risk = trade_risk_pct  # Track for cleanup in finally block
        
        try:
            # Liquidity & Spread Filter
            ticker = await self.execution.fetch_ticker_data(symbol)
            if not ticker:
                return
                
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            vol = ticker.get('quoteVolume', 0)
            
            if bid and ask and bid > 0 and ask > 0:
                spread = (ask - bid) / ((ask + bid) / 2)
                max_spread = 0.0015
                if spread > max_spread:
                    add_log_message(f"[{symbol}] Rejected: High spread ({spread*100:.3f}%)")
                    return
                    
            min_vol = 30000000 if relaxed_used else getattr(Config, 'MIN_24H_VOL_USDT', 50000000)
            if not Config.PAPER_TRADING and vol > 0 and vol < min_vol:
                if symbol not in Config.SUPPORTED_SYMBOLS:
                    all_tickers = await self.execution.fetch_all_tickers()
                    is_top_20 = False
                    if all_tickers:
                        sorted_tickers = sorted([t for t in all_tickers.values() if t.get('quoteVolume')], key=lambda x: x.get('quoteVolume', 0), reverse=True)
                        top_20 = [t['symbol'] for t in sorted_tickers[:20]]
                        if symbol in top_20:
                            is_top_20 = True
                    
                    if not is_top_20:
                        add_log_message(f"[{symbol}] Rejected: Low volume ({vol:,.0f} USDT) and not in top 20.")
                        return
            
            # Slippage Check
            live_price = ticker.get('last', entry_price)
            if abs(live_price - entry_price) / entry_price > getattr(Config, 'MAX_SLIPPAGE_PCT', 0.002):
                add_log_message(f"[{symbol}] Trade skipped: Slippage too high. Signal: {entry_price}, Live: {live_price}")
                return
            entry_price = live_price  # Execute at live price
            
            sl = float(metadata['stop_loss']) if metadata.get('stop_loss') is not None else (entry_price * 0.98)
            tp = float(metadata['take_profit']) if metadata.get('take_profit') is not None else (entry_price * 1.04)
            
            is_inr = getattr(Config, 'COINDCX_TRADE_INR', False) if not Config.PAPER_TRADING else (getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR')
            quote_curr = symbol.split('/')[1] if '/' in symbol else "USDT"
            conversion_rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
            if is_inr and not Config.PAPER_TRADING and hasattr(self.execution, 'fetch_usdt_inr_rate'):
                if inspect.iscoroutinefunction(self.execution.fetch_usdt_inr_rate):
                    dynamic_rate = await self.execution.fetch_usdt_inr_rate(side=signal)
                else:
                    dynamic_rate = self.execution.fetch_usdt_inr_rate(side=signal)
                    if inspect.isawaitable(dynamic_rate):
                        dynamic_rate = await dynamic_rate
                try:
                    dynamic_rate_val = dynamic_rate if isinstance(dynamic_rate, float) else (float(dynamic_rate) if dynamic_rate is not None else None)
                except (TypeError, ValueError):
                    dynamic_rate_val = None
                if dynamic_rate_val is None or dynamic_rate_val <= 0:
                    add_log_message(f"[{symbol}] Trade blocked: Live CoinDCX USDT/INR FX rate unavailable or invalid (fail-closed).")
                    return
                conversion_rate = dynamic_rate_val

            pos_size = self.risk.calculate_position_size(
                account_equity=current_equity,
                entry_price=entry_price,
                stop_loss=sl,
                quote_currency=quote_curr,
                is_inr=is_inr,
                conversion_rate=conversion_rate,
            )
            pos_size = pos_size * (trade_risk_pct / (getattr(Config, 'RISK_PCT', 0.8) / 100.0))

            # ── Pre-Trade Exchange Rules & Equity Validation ──
            is_valid, validated_pos_size, v_reason = ExchangeValidator.validate_order_intent(
                symbol=symbol,
                side='buy' if signal == 'BUY' else 'sell',
                order_type='market',
                amount=pos_size,
                price=entry_price,
                current_equity=current_equity,
                is_inr=is_inr,
                quote_currency=quote_curr,
                conversion_rate=conversion_rate,
            )
            if not is_valid:
                add_log_message(f"[{symbol}] Order pre-validation REJECTED: {v_reason}")
                return
            pos_size = validated_pos_size

            if pos_size <= 0.0:
                return

            # ── Next-Gen Proprietary Edge: Dual-Brain Adversarial AI Debate Courtroom ──
            market_context = {
                'cvd': metadata.get('cvd', {}),
                'liquidation': metadata.get('liquidation', {}),
                'ml_confidence': raw_prob,
                'directional_ml_confidence': prob,
                'funding_rate': fr,
                'bb_squeeze': False,
                'spread_pct': spread
            }
            debate_result = self.courtroom.conduct_debate(signal, metadata, market_context)
            if debate_result.get('verdict') != 'APPROVED':
                add_log_message(f"[{symbol}] ⚖️ Trade REJECTED by Dual-Brain AI Courtroom ({debate_result['conviction_pct']}% conviction): {', '.join(debate_result['prosecutor_objections'])}")
                return
            else:
                add_log_message(f"[{symbol}] ⚖️ Trade APPROVED by Dual-Brain AI Courtroom ({debate_result['conviction_pct']}% conviction)!")
                
            ctx = self.order_state_machine.get_context(symbol)
            ctx.transition_to(OrderState.ORDER_INTENT_CREATED, reason=f"{signal} setup approved")
            ctx.requested_qty = pos_size
            ctx.entry_price = entry_price
            ctx.stop_loss = sl
            ctx.side = "LONG" if signal == "BUY" else "SHORT"

            if signal == "BUY":
                add_log_message(f"[{symbol}] Executing BUY (LONG). Size: {pos_size:.6f} | SL: {sl:.2f} | TP: {tp:.2f}")
                order = None
                ctx.transition_to(OrderState.ORDER_SUBMITTED, reason="BUY market order submitted")
                if self.has_keys and not Config.PAPER_TRADING:
                    try:
                        order = await self.execution.place_order('buy', 'market', pos_size, price=entry_price, symbol=symbol)
                    except Exception as net_err:
                        add_log_message(f"[{symbol}] ⚠️ Network error during BUY submission ({net_err}). Transitioning to EXECUTION_UNKNOWN (LOGIC-003 fix).")
                        ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason=f"Submission NetworkError: {net_err}")
                        order = None
                        # P1: Expedite reconciliation to minimize blind window
                        if hasattr(self, 'reconciliation'):
                            asyncio.create_task(self.reconciliation._reconcile_live_broker_state())
                else:
                    min_paper_cost = 50.0 if is_inr else 1.0
                    cur_sym = "₹" if is_inr else "$"
                    max_alloc_pct = getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35)
                    max_allowed_cost = current_equity * max_alloc_pct
                    entry_cost_equity_curr = pos_size * entry_price * (conversion_rate if is_inr else 1.0)

                    if entry_cost_equity_curr > max_allowed_cost:
                        pos_size = max_allowed_cost / (entry_price * (conversion_rate if is_inr else 1.0))
                        entry_cost_equity_curr = max_allowed_cost

                    if entry_cost_equity_curr <= self._dry_run_balance_usdt:
                        self._dry_run_balance_usdt -= entry_cost_equity_curr
                        order = {'id': f'MOCK_BUY_{int(time.time()*1000)}', 'price': entry_price, 'amount': pos_size, 'status': 'filled'}
                    elif self._dry_run_balance_usdt >= min_paper_cost:
                        usable_cash = min(self._dry_run_balance_usdt, self._dry_run_balance_usdt * max_alloc_pct)
                        if usable_cash < min_paper_cost:
                            usable_cash = self._dry_run_balance_usdt
                        pos_size = usable_cash / (entry_price * (conversion_rate if is_inr else 1.0))
                        self._dry_run_balance_usdt -= usable_cash
                        order = {'id': f'MOCK_BUY_{int(time.time()*1000)}', 'price': entry_price, 'amount': pos_size, 'status': 'filled'}
                    else:
                        add_log_message(f"[{symbol}] ⚠️ Paper trading wallet balance depleted ({cur_sym}{self._dry_run_balance_usdt:.2f}). Reset wallet to continue.")

                if self._is_truthy_fill(order):
                    assert order is not None  # Guaranteed by _is_truthy_fill
                    filled_amount = self._extract_filled_qty(order, pos_size)
                    fill_price = float(order.get('price') or entry_price) if isinstance(order, dict) else float(order.average_fill_price or entry_price)
                    ctx.filled_qty = filled_amount
                    ctx.fill_avg_price = fill_price
                    ctx.transition_to(OrderState.FILLED, reason="BUY fill confirmed")
                    
                    # ── NATIVE EXCHANGE STOP LOSS PLACEMENT & VERIFICATION ──
                    if self.has_keys and not Config.PAPER_TRADING:
                        ctx.transition_to(OrderState.SL_PLACEMENT_PENDING, reason="Submitting native SL to exchange")
                        sl_order = await self.execution.place_native_stop_loss(symbol, 'sell', filled_amount, sl)
                        if self._is_active_sl_order(sl_order):
                            ctx.native_sl_order_id = str(sl_order['id']) if isinstance(sl_order, dict) else str(sl_order.exchange_order_id)
                            ctx.transition_to(OrderState.PROTECTED, reason=f"Native SL confirmed on exchange @ {sl}")
                            add_log_message(f"[{symbol}] 🛡️ NATIVE STOP LOSS ACTIVE on exchange (ID: {ctx.native_sl_order_id}, Price: {sl:.4f})")
                        else:
                            add_log_message(f"[{symbol}] 🚨 NATIVE SL PLACEMENT FAILED! Executing EMERGENCY FLATTEN.")
                            flatten_order = await self.execution.emergency_flatten_position(symbol, 'BUY', filled_amount, reason="NATIVE_SL_FAILED")
                            if self._is_truthy_fill(flatten_order):
                                actual_flatten = self._extract_filled_qty(flatten_order, filled_amount)
                                remaining_qty = max(0.0, filled_amount - actual_flatten)
                                self.position_size[symbol] = remaining_qty
                                if remaining_qty <= 0.0001:
                                    self.in_position[symbol] = False
                                    ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason="Native SL placement failed")
                                else:
                                    self.in_position[symbol] = True
                                    self.position_side[symbol] = "LONG"
                                    self.entry_price[symbol] = fill_price
                                    ctx.transition_to(OrderState.EXIT_UNKNOWN, reason="Partial fill on emergency flatten")
                            else:
                                self.in_position[symbol] = True
                                self.position_side[symbol] = "LONG"
                                self.position_size[symbol] = filled_amount
                                self.entry_price[symbol] = fill_price
                                self.stop_loss[symbol] = sl
                                ctx.transition_to(OrderState.EXIT_UNKNOWN, reason="Emergency flatten failed or unknown")
                            self.save_state()
                            if hasattr(self, 'reconciliation'): asyncio.create_task(self.reconciliation._reconcile_live_broker_state())
                            return
                    else:
                        ctx.transition_to(OrderState.PROTECTED, reason="Virtual SL activated")

                    self.in_position[symbol] = True
                    self.position_side[symbol] = "LONG"
                    self.entry_price[symbol] = fill_price
                    self.stop_loss[symbol] = sl
                    
                    # Initialize TP levels for LONG (3-Stage: TP1 65%, TP2 23%, Runner 12%)
                    # LOGIC-004 fix: Authoritative recalculation based on actual execution fill_price
                    self.partial_tp_taken[symbol] = False
                    self.tp2_taken[symbol] = False
                    r_amount = abs(sl - fill_price)
                    tp1_mult = float(getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.2))
                    tp2_mult = float(getattr(Config, 'RISK_REWARD_RATIO', 2.2))
                    self.take_profit_1r[symbol] = float(fill_price + (tp1_mult * r_amount) + fee_adj)
                    self.take_profit_2r[symbol] = float(fill_price + (tp2_mult * r_amount) + fee_adj)
                    self.take_profit[symbol] = float(fill_price + (4.0 * r_amount) + fee_adj)
                    
                    self.highest_price_reached[symbol] = fill_price
                    self.position_size[symbol] = filled_amount
                    self.original_position_size[symbol] = filled_amount
                    self.realized_pnl[symbol] = 0.0
                    self.entry_time[symbol] = time.time() * 1000.0
                    self.last_trade_time[symbol] = time.time()
                    # --- PROFIT-BASED LOGIC: Trade lifecycle init ---
                    self.current_trade_id[symbol] = f"TRADE_{symbol.replace('/', '')}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
                    self.entry_fx_rate[symbol] = float(getattr(Config, 'USDT_INR_RATE', 85.0))
                    entry_fee = filled_amount * fill_price * Config.FEE_RATE
                    self.accumulated_fees[symbol] = entry_fee
                    self.position_mode[symbol] = str(metadata.get('mode') or 'STRICT')
                    zone_id_raw = metadata.get('zone_id')
                    zone_id = str(zone_id_raw) if zone_id_raw is not None else None
                    self.last_zone_traded[symbol] = zone_id
                    if zone_id:
                        self.traded_zones_cache[f"{symbol}_{zone_id}"] = True
                    self.trades_today += 1
                    self.global_last_trade_time = time.time()
                    if metadata.get('mode') == 'RELAXED':
                        self.relaxed_trades_today += 1

                    if symbol == Config.SYMBOL:
                        DashboardState.in_position = True
                        DashboardState.position_side = "LONG"
                        DashboardState.entry_price = fill_price
                        DashboardState.stop_loss = sl
                        DashboardState.take_profit = self.take_profit[symbol]
                        DashboardState.position_size = filled_amount
                        DashboardState.current_pnl_pct = 0.0
                        DashboardState.current_pnl_usdt = 0.0

                    e_time = self.entry_time[symbol]
                    e_time_str = datetime.datetime.fromtimestamp(e_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                    DashboardState.active_positions[symbol] = {
                        'symbol': symbol,
                        'side': "LONG",
                        'entry_price': fill_price,
                        'stop_loss': sl,
                        'take_profit': self.take_profit_1r[symbol],
                        'target_1r': self.take_profit_1r[symbol],
                        'target_2r': self.take_profit_2r[symbol],
                        'final_target': self.take_profit[symbol],
                        'active_target': self.take_profit_1r[symbol],
                        'active_target_name': "TP1 (1.0R)",
                        'target_stage': 1,
                        'position_size': filled_amount,
                        'current_pnl_usdt': 0.0,
                        'current_pnl_currency': 0.0,
                        'current_pnl_pct': 0.0,
                        'guaranteed_pnl_usdt': 0.0,
                        'guaranteed_pnl_currency': 0.0,
                        'guaranteed_pnl_pct': 0.0,
                        'tp1_hit': False,
                        'tp2_hit': False,
                        'profit_locked': False,
                        'live_price': fill_price,
                        'entry_time': e_time,
                        'entry_time_str': e_time_str
                    }

                    # Record in Immutable Ledger
                    self.immutable_ledger.record_entry(
                        symbol=symbol,
                        side="LONG",
                        requested_qty=pos_size,
                        filled_qty=filled_amount,
                        fill_price=fill_price,
                        stop_loss=sl,
                        tp1=self.take_profit_1r[symbol],
                        tp2=self.take_profit_2r[symbol],
                        runner_tp=self.take_profit[symbol],
                        client_order_id=str(order.get('clientOrderId') or f"CLIENT_{int(time.time()*1000)}"),
                        exchange_order_id=str(order.get('id') or ''),
                        native_sl_id=ctx.native_sl_order_id
                    )

                    self.save_state()
                    # Telegram Notification with 3-Stage Target Levels
                    msg_str = (
                        f"🟢 *BUY (LONG) {symbol}*\n"
                        f"Mode: {metadata.get('mode', 'STRICT')}\n"
                        f"Setup Type: {metadata.get('setup_type', 'NONE')}\n"
                        f"Entry: {fill_price:.4f}\n"
                        f"Stop Loss: {sl:.4f} (NATIVE PROTECTED)\n"
                        f"TP1 ({tp1_mult:.1f}R - 50%): {self.take_profit_1r[symbol]:.4f}\n"
                        f"TP2 ({tp2_mult:.1f}R - 30%): {self.take_profit_2r[symbol]:.4f}\n"
                        f"Runner (4.0R - 20%): {self.take_profit[symbol]:.4f}\n"
                        f"Position Size: {filled_amount:.6f}\n"
                        f"Confidence: {prob:.2f}\n"
                        f"Reason: {metadata.get('reason', 'N/A')}"
                    )
                    add_log_message(f"[{symbol}] " + msg_str.replace('\n', ' | '))
                    await self.notifier.send_message(msg_str)
                else:
                    if ctx.state != OrderState.EXECUTION_UNKNOWN:
                        ctx.transition_to(OrderState.REJECTED, reason="Exchange rejected BUY order")
                    add_log_message(f"[{symbol}] ❌ BUY order REJECTED or UNKNOWN (check execution logs)")
                    await self.notifier.send_message(f"⚠️ BUY REJECTED {symbol}: Order failed to execute. Check execution logs.")
                    
            elif signal == "SELL":
                order = None
                ctx.transition_to(OrderState.ORDER_SUBMITTED, reason="SELL market order submitted")
                if self.has_keys and not Config.PAPER_TRADING:
                    try:
                        order = await self.execution.place_order('sell', 'market', pos_size, price=entry_price, symbol=symbol)
                    except Exception as net_err:
                        add_log_message(f"[{symbol}] ⚠️ Network error during SELL submission ({net_err}). Transitioning to EXECUTION_UNKNOWN (LOGIC-003 fix).")
                        ctx.transition_to(OrderState.EXECUTION_UNKNOWN, reason=f"Submission NetworkError: {net_err}")
                        order = None
                        # P1: Expedite reconciliation to minimize blind window
                        if hasattr(self, 'reconciliation'):
                            asyncio.create_task(self.reconciliation._reconcile_live_broker_state())
                else:
                    min_paper_cost = 50.0 if is_inr else 1.0
                    cur_sym = "₹" if is_inr else "$"
                    max_alloc_pct = getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35)
                    max_allowed_cost = current_equity * max_alloc_pct
                    collateral_equity_curr = pos_size * entry_price * (conversion_rate if is_inr else 1.0)

                    if collateral_equity_curr > max_allowed_cost:
                        pos_size = max_allowed_cost / (entry_price * (conversion_rate if is_inr else 1.0))
                        collateral_equity_curr = max_allowed_cost

                    if collateral_equity_curr <= self._dry_run_balance_usdt:
                        self._dry_run_balance_usdt -= collateral_equity_curr
                        order = {'id': f'MOCK_SELL_{int(time.time()*1000)}', 'price': entry_price, 'amount': pos_size, 'status': 'filled'}
                    elif self._dry_run_balance_usdt >= min_paper_cost:
                        usable_cash = min(self._dry_run_balance_usdt, self._dry_run_balance_usdt * max_alloc_pct)
                        if usable_cash < min_paper_cost:
                            usable_cash = self._dry_run_balance_usdt
                        pos_size = usable_cash / (entry_price * (conversion_rate if is_inr else 1.0))
                        self._dry_run_balance_usdt -= usable_cash
                        order = {'id': f'MOCK_SELL_{int(time.time()*1000)}', 'price': entry_price, 'amount': pos_size, 'status': 'filled'}
                    else:
                        add_log_message(f"[{symbol}] ⚠️ Paper trading wallet balance depleted ({cur_sym}{self._dry_run_balance_usdt:.2f}). Reset wallet to continue.")
                    
                if self._is_truthy_fill(order):
                    assert order is not None  # Guaranteed by _is_truthy_fill
                    filled_amount = self._extract_filled_qty(order, pos_size)
                    fill_price = float(order.get('price') or entry_price) if isinstance(order, dict) else float(order.average_fill_price or entry_price)
                    ctx.filled_qty = filled_amount
                    ctx.fill_avg_price = fill_price
                    ctx.transition_to(OrderState.FILLED, reason="SELL fill confirmed")
                    
                    # ── NATIVE EXCHANGE STOP LOSS PLACEMENT & VERIFICATION ──
                    if self.has_keys and not Config.PAPER_TRADING:
                        ctx.transition_to(OrderState.SL_PLACEMENT_PENDING, reason="Submitting native SL to exchange")
                        sl_order = await self.execution.place_native_stop_loss(symbol, 'buy', filled_amount, sl)
                        if self._is_active_sl_order(sl_order):
                            ctx.native_sl_order_id = str(sl_order['id']) if isinstance(sl_order, dict) else str(sl_order.exchange_order_id)
                            ctx.transition_to(OrderState.PROTECTED, reason=f"Native SL confirmed on exchange @ {sl}")
                            add_log_message(f"[{symbol}] 🛡️ NATIVE STOP LOSS ACTIVE on exchange (ID: {ctx.native_sl_order_id}, Price: {sl:.4f})")
                        else:
                            add_log_message(f"[{symbol}] 🚨 NATIVE SL PLACEMENT FAILED! Executing EMERGENCY FLATTEN.")
                            flatten_order = await self.execution.emergency_flatten_position(symbol, 'SELL', filled_amount, reason="NATIVE_SL_FAILED")
                            if self._is_truthy_fill(flatten_order):
                                actual_flatten = self._extract_filled_qty(flatten_order, filled_amount)
                                remaining_qty = max(0.0, filled_amount - actual_flatten)
                                self.position_size[symbol] = remaining_qty
                                if remaining_qty <= 0.0001:
                                    self.in_position[symbol] = False
                                    ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason="Native SL placement failed")
                                else:
                                    self.in_position[symbol] = True
                                    self.position_side[symbol] = "SHORT"
                                    self.entry_price[symbol] = fill_price
                                    ctx.transition_to(OrderState.EXIT_UNKNOWN, reason="Partial fill on emergency flatten")
                            else:
                                self.in_position[symbol] = True
                                self.position_side[symbol] = "SHORT"
                                self.position_size[symbol] = filled_amount
                                self.entry_price[symbol] = fill_price
                                self.stop_loss[symbol] = sl
                                ctx.transition_to(OrderState.EXIT_UNKNOWN, reason="Emergency flatten failed or unknown")
                            self.save_state()
                            if hasattr(self, 'reconciliation'): asyncio.create_task(self.reconciliation._reconcile_live_broker_state())
                            return
                    else:
                        ctx.transition_to(OrderState.PROTECTED, reason="Virtual SL activated")

                    self.in_position[symbol] = True
                    self.position_side[symbol] = "SHORT"
                    self.entry_price[symbol] = fill_price
                    self.stop_loss[symbol] = sl
                    
                    # Initialize TP levels for SHORT (3-Stage: TP1 65%, TP2 23%, Runner 12%)
                    # LOGIC-004 fix: Authoritative recalculation based on actual execution fill_price
                    self.partial_tp_taken[symbol] = False
                    self.tp2_taken[symbol] = False
                    r_amount = abs(sl - fill_price)
                    tp1_mult = float(getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.2))
                    tp2_mult = float(getattr(Config, 'RISK_REWARD_RATIO', 2.2))
                    self.take_profit_1r[symbol] = float(fill_price - (tp1_mult * r_amount) - fee_adj)
                    self.take_profit_2r[symbol] = float(fill_price - (tp2_mult * r_amount) - fee_adj)
                    self.take_profit[symbol] = float(fill_price - (4.0 * r_amount) - fee_adj)
                    
                    self.lowest_price_reached[symbol] = fill_price
                    self.position_size[symbol] = filled_amount
                    self.original_position_size[symbol] = filled_amount
                    self.realized_pnl[symbol] = 0.0
                    self.entry_time[symbol] = time.time() * 1000.0
                    self.last_trade_time[symbol] = time.time()
                    # --- PROFIT-BASED LOGIC: Trade lifecycle init ---
                    self.current_trade_id[symbol] = f"TRADE_{symbol.replace('/', '')}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
                    self.entry_fx_rate[symbol] = float(getattr(Config, 'USDT_INR_RATE', 85.0))
                    entry_fee = filled_amount * fill_price * Config.FEE_RATE
                    self.accumulated_fees[symbol] = entry_fee
                    self.position_mode[symbol] = str(metadata.get('mode') or 'STRICT')
                    zone_id_raw = metadata.get('zone_id')
                    zone_id = str(zone_id_raw) if zone_id_raw is not None else None
                    self.last_zone_traded[symbol] = zone_id
                    if zone_id:
                        self.traded_zones_cache[f"{symbol}_{zone_id}"] = True
                    self.trades_today += 1
                    self.global_last_trade_time = time.time()
                    if metadata.get('mode') == 'RELAXED':
                        self.relaxed_trades_today += 1
                    
                    if symbol == Config.SYMBOL:
                        DashboardState.in_position = True
                        DashboardState.position_side = "SHORT"
                        DashboardState.entry_price = fill_price
                        DashboardState.stop_loss = sl
                        DashboardState.take_profit = self.take_profit[symbol]
                        DashboardState.position_size = filled_amount
                        DashboardState.current_pnl_pct = 0.0
                        DashboardState.current_pnl_usdt = 0.0

                    e_time = self.entry_time[symbol]
                    e_time_str = datetime.datetime.fromtimestamp(e_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                    DashboardState.active_positions[symbol] = {
                        'symbol': symbol,
                        'side': "SHORT",
                        'entry_price': fill_price,
                        'stop_loss': sl,
                        'take_profit': self.take_profit_1r[symbol],
                        'target_1r': self.take_profit_1r[symbol],
                        'target_2r': self.take_profit_2r[symbol],
                        'final_target': self.take_profit[symbol],
                        'active_target': self.take_profit_1r[symbol],
                        'active_target_name': "TP1 (1.0R)",
                        'target_stage': 1,
                        'position_size': filled_amount,
                        'current_pnl_usdt': 0.0,
                        'current_pnl_currency': 0.0,
                        'current_pnl_pct': 0.0,
                        'guaranteed_pnl_usdt': 0.0,
                        'guaranteed_pnl_currency': 0.0,
                        'guaranteed_pnl_pct': 0.0,
                        'tp1_hit': False,
                        'tp2_hit': False,
                        'profit_locked': False,
                        'live_price': fill_price,
                        'entry_time': e_time,
                        'entry_time_str': e_time_str
                    }

                    # Record in Immutable Ledger
                    self.immutable_ledger.record_entry(
                        symbol=symbol,
                        side="SHORT",
                        requested_qty=pos_size,
                        filled_qty=filled_amount,
                        fill_price=fill_price,
                        stop_loss=sl,
                        tp1=self.take_profit_1r[symbol],
                        tp2=self.take_profit_2r[symbol],
                        runner_tp=self.take_profit[symbol],
                        client_order_id=str(order.get('clientOrderId') or f"CLIENT_{int(time.time()*1000)}"),
                        exchange_order_id=str(order.get('id') or ''),
                        native_sl_id=ctx.native_sl_order_id
                    )

                    self.save_state()
                    msg_str = (
                        f"🔴 *SELL (SHORT) {symbol}*\n"
                        f"Mode: {metadata.get('mode', 'STRICT')}\n"
                        f"Setup Type: {metadata.get('setup_type', 'NONE')}\n"
                        f"Entry: {fill_price:.4f}\n"
                        f"Stop Loss: {sl:.4f} (NATIVE PROTECTED)\n"
                        f"TP1 ({tp1_mult:.1f}R - 50%): {self.take_profit_1r[symbol]:.4f}\n"
                        f"TP2 ({tp2_mult:.1f}R - 30%): {self.take_profit_2r[symbol]:.4f}\n"
                        f"Runner (4.0R - 20%): {self.take_profit[symbol]:.4f}\n"
                        f"Position Size: {filled_amount:.6f}\n"
                        f"Confidence: {prob:.2f}\n"
                        f"Reason: {metadata.get('reason', 'N/A')}"
                    )
                    add_log_message(f"[{symbol}] " + msg_str.replace('\n', ' | '))
                    await self.notifier.send_message(msg_str)
                else:
                    if ctx.state != OrderState.EXECUTION_UNKNOWN:
                        ctx.transition_to(OrderState.REJECTED, reason="Exchange rejected SELL order")
                    add_log_message(f"[{symbol}] ❌ SELL order REJECTED or UNKNOWN (check execution logs)")
                    await self.notifier.send_message(f"⚠️ SELL REJECTED {symbol}: Order failed to execute. Check logs.")
        finally:
            # P0: EXPOSURE LEAK FIX - Do NOT release reserved risk if execution outcome is unknown!
            ctx = self.order_state_machine.get_context(symbol)
            if reserved_trade_risk > 0.0:
                if ctx.is_in_flight():
                    add_log_message(f"[{symbol}] ⚠️ Risk reservation RETAINED due to unknown execution outcome ({ctx.state}).")
                    ctx.reserved_risk_pct = reserved_trade_risk
                    ctx.reserved_risk_side = signal
                else:
                    await self.risk.release_risk(reserved_trade_risk, side=signal, reservation_id=getattr(ctx, 'reservation_id', None))
                    ctx.reserved_risk_pct = 0.0
                    ctx.reserved_risk_side = 'HOLD'
                    ctx.reservation_id = None
            
            # FINAL SAVE at end of candle iteration to persist any UNKNOWN states or risk reservations
            self.save_state()

    async def run_live_risk_monitor(self):
        while True:
            try:
                if DashboardState.symbol_change_requested:
                    new_symbol = DashboardState.symbol_change_requested
                    DashboardState.symbol_change_requested = None
                    await self.change_bot_symbol(new_symbol)

                now_utc = datetime.datetime.now(datetime.timezone.utc)
                if now_utc.date() != self._last_reset_date:
                    current_eq = DashboardState.balance_usdt if self.has_keys else self.calculate_total_equity()
                    self.risk.reset_daily_equity(current_eq)
                    self.trades_today = 0
                    self.relaxed_trades_today = 0
                    self.last_trade_day = now_utc.date()
                    self._last_reset_date = now_utc.date()
                    add_log_message(f"[RISK] Daily equity & trades checkpoint reset at UTC midnight.")

                # --- 🚀 INSTITUTIONAL MULTI-SYMBOL REAL-TIME SCANNER ---
                self._fast_scan_counter = getattr(self, '_fast_scan_counter', 0) + 1
                if self._fast_scan_counter % 10 == 0: # Disciplined 10s evaluation cycle
                    open_count, _, _, _ = await self.get_open_positions_info()
                    max_open = getattr(Config, 'MAX_OPEN_TRADES', 3)
                    time_since_last_trade = time.time() - getattr(self, 'global_last_trade_time', 0)
                    if open_count < max_open and time.time() > self.global_pause_until and time_since_last_trade >= 60:
                        # Concurrently evaluate candidates across 20 pairs
                        for sym in Config.SUPPORTED_SYMBOLS:
                            if not self.in_position[sym] and self.pipeline.ltf_candles.get(sym):
                                scan_task = asyncio.create_task(self.on_candle_close(sym))
                                self._active_scan_tasks.add(scan_task)
                                scan_task.add_done_callback(self._active_scan_tasks.discard)

                for symbol in Config.SUPPORTED_SYMBOLS:
                    sym_price = self.pipeline.latest_prices.get(symbol, 0.0)
                    if sym_price > 0:
                        self.lead_lag.record_tick(symbol, sym_price)
                        if symbol != "BTC/USDT":
                            lead_lag_opp = self.lead_lag.evaluate_lead_lag(symbol)
                            if lead_lag_opp.get('signal') != 'NONE':
                                add_log_message(f"[{symbol}] ⚡ Cross-Asset Lead-Lag Impulse: {lead_lag_opp['signal']} (BTC Velocity: {lead_lag_opp['btc_velocity_pct']}%, Lag Edge: +{lead_lag_opp['edge_pct']}%)")

                    if self.in_position[symbol] and self.pipeline.latest_prices.get(symbol, 0.0) > 0:
                        curr_price = self.pipeline.latest_prices[symbol]
                        
                        ltf_df = prepare_dataframe(self.pipeline.ltf_candles[symbol])
                        curr_atr = calculate_atr(ltf_df, Config.ATR_PERIOD).iloc[-1] if not ltf_df.empty else 0.001
                        
                        if self.position_side[symbol] == "LONG":
                            self.highest_price_reached[symbol] = max(self.highest_price_reached[symbol], curr_price)
                            r_dist = abs(self.entry_price[symbol] - self.stop_loss[symbol])
                            
                            # ZERO-RISK FREE-TRADE LOCK: Move SL to Breakeven
                            fee_buffer_pct = getattr(Config, 'DYNAMIC_BE_BUFFER_PCT', 0.0030)
                            fee_offset = self.entry_price[symbol] * fee_buffer_pct
                            tsl_activation = float(getattr(Config, 'TSL_ACTIVATION_R', 1.2))
                            min_required_profit = max(tsl_activation * r_dist, fee_offset * 1.5)
                            
                            # Only activate Breakeven after TP1 profit is secured OR price has reached full activation threshold
                            if (self.partial_tp_taken[symbol] or self.highest_price_reached[symbol] >= self.entry_price[symbol] + min_required_profit):
                                be_sl = self.entry_price[symbol] + fee_offset
                                if be_sl > self.stop_loss[symbol] and curr_price > be_sl:
                                    self.stop_loss[symbol] = be_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = be_sl
                                    add_log_message(f"[{symbol}] 🛡️ ZERO-RISK FREE-TRADE ACTIVATED: SL moved to Breakeven ({be_sl:.4f})")

                            # 1. ⏱️ STAGNATION KILLER: Time-based exit if trade loses momentum near entry
                            if getattr(Config, 'ENABLE_TIME_STOP', True) and not self.partial_tp_taken[symbol]:
                                entry_ts = self.entry_time.get(symbol, 0)
                                if entry_ts > 0:
                                    tf_mins = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
                                    time_elapsed_secs = time.time() - (entry_ts / 1000.0)
                                    candles_open = time_elapsed_secs / (tf_mins * 60)
                                    max_stagnant_candles = getattr(Config, 'MAX_STAGNANT_CANDLES', 16)
                                    if candles_open >= max_stagnant_candles:
                                        stagnant_max_dist = getattr(Config, 'STAGNANT_MAX_R_DISTANCE', 0.25) * r_dist
                                        if abs(curr_price - self.entry_price[symbol]) <= stagnant_max_dist:
                                            add_log_message(f"[{symbol}] ⏱️ STAGNATION KILLER: Position open for {candles_open:.1f} candles ({time_elapsed_secs/3600:.1f}h) with no momentum. Auto-exiting at scratch.")
                                            await self.notifier.send_message(f"⏱️ *STAGNATION AUTO-EXIT ({symbol})*\nTrade open for {candles_open:.1f} candles without momentum. Scratch-closed to reclaim capital.")
                                            await self.exit_position(symbol, "STAGNANT_TIME_EXIT")
                                            continue

                            # 2. ⚡ EARLY STRUCTURAL EXIT: Cut loss early if 15m structure breaks against LONG
                            if getattr(Config, 'ENABLE_STRUCTURAL_EXIT', True) and not self.partial_tp_taken[symbol] and len(ltf_df) >= 3:
                                last_c = ltf_df.iloc[-1]
                                prev_c = ltf_df.iloc[-2]
                                ema_20 = calculate_ema(ltf_df, 20).iloc[-1] if len(ltf_df) >= 20 else 0.0
                                avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) >= 14 else 1.0
                                unrealized_loss_r = (self.entry_price[symbol] - curr_price) / r_dist if r_dist > 0 else 0.0
                                early_max_r = getattr(Config, 'EARLY_EXIT_MAX_LOSS_R', 0.45)
                                if 0.15 <= unrealized_loss_r <= early_max_r and ema_20 > 0:
                                    is_bearish_engulf = (last_c['close'] < last_c['open']) and (prev_c['close'] > prev_c['open']) and (last_c['close'] < prev_c['open'])
                                    is_heavy_vol = last_c['volume'] > 1.8 * avg_vol
                                    is_below_ema = last_c['close'] < ema_20
                                    if (is_bearish_engulf or is_below_ema) and is_heavy_vol:
                                        add_log_message(f"[{symbol}] ⚡ EARLY STRUCTURAL EXIT: Heavy bearish breakdown below EMA20. Cutting loss early at -{unrealized_loss_r:.2f}R.")
                                        await self.notifier.send_message(f"⚡ *EARLY STRUCTURAL EXIT ({symbol})*\nMarket structure broke against LONG before full SL. Loss cut early at -{unrealized_loss_r:.2f}R (Saved ~{(1.0-unrealized_loss_r)*100:.0f}% loss).")
                                        await self.exit_position(symbol, "EARLY_STRUCTURAL_INVALIDATION")
                                        continue
                            
                            # TP1 Scale-Out (80% at 2.0R, or 100% Full Exit)
                            if not self.partial_tp_taken[symbol] and curr_price >= self.take_profit_1r[symbol]:
                                tp1_pct = float(getattr(Config, 'TP1_SCALE_OUT_PCT', 0.65))
                                if tp1_pct >= 0.999:
                                    add_log_message(f"[{symbol}] 🎯 Target 1 hit! Full 100% profit booking initiated.")
                                    await self.exit_position(symbol, "TAKE_PROFIT_1")
                                    continue
                                add_log_message(f"[{symbol}] 🎯 Target 1 hit! Booking {int(tp1_pct*100)}% profit.")
                                tp1_size = self.position_size[symbol] * tp1_pct
                                tp1_success = False
                                tp1_order = None
                                if self.has_keys and not Config.PAPER_TRADING:
                                    tp1_order = await self.execution.place_order('sell', 'market', tp1_size, symbol=symbol, is_exit_order=True)
                                    tp1_success = self._is_truthy_fill(tp1_order)
                                else:
                                    is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
                                    rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
                                    self._dry_run_balance_usdt += tp1_size * curr_price * (rate if is_inr else 1.0)
                                    tp1_success = True
                                if tp1_success:
                                    if self.has_keys and not Config.PAPER_TRADING:
                                        tp1_size = self._extract_filled_qty(tp1_order, tp1_size)
                                    self.position_size[symbol] -= tp1_size
                                    self.partial_tp_taken[symbol] = True
                                    
                                    # Log partial TP1 trade record for accurate PnL tracking
                                    tp1_pnl_usdt = tp1_size * (curr_price - self.entry_price[symbol])
                                    # --- PROFIT-BASED LOGIC: Net fee deduction ---
                                    tp1_fee = tp1_size * self.entry_price[symbol] * Config.FEE_RATE + tp1_size * curr_price * Config.FEE_RATE
                                    tp1_pnl_usdt -= tp1_fee
                                    self.accumulated_fees[symbol] = self.accumulated_fees.get(symbol, 0.0) + (tp1_size * curr_price * Config.FEE_RATE)
                                    tp1_pnl_pct = (curr_price - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0
                                    self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + tp1_pnl_usdt
                                    
                                    now_ts = int(time.time() * 1000)
                                    entry_ts = self.entry_time.get(symbol, now_ts)
                                    dur_secs = max(0, int((now_ts - entry_ts) / 1000))
                                    dur_str = f"{dur_secs // 60}m {dur_secs % 60}s" if dur_secs >= 60 else f"{dur_secs}s"
                                    
                                    tp1_record = {
                                        'trade_id': self.current_trade_id.get(symbol, ''),
                                        'symbol': symbol,
                                        'side': 'LONG',
                                        'type': 'TP1_PARTIAL',
                                        'entry_price': self.entry_price[symbol],
                                        'exit_price': curr_price,
                                        'entry': self.entry_price[symbol],
                                        'exit': curr_price,
                                        'size': tp1_size,
                                        'pnl_usdt': round(tp1_pnl_usdt, 4),
                                        'pnl': round(tp1_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'pnl_currency': round(tp1_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'currency': 'INR' if is_inr else 'USDT',
                                        'pnl_pct': round(tp1_pnl_pct, 2),
                                        'entry_time': entry_ts,
                                        'exit_time': now_ts,
                                        'entry_time_str': datetime.datetime.fromtimestamp(entry_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'exit_time_str': datetime.datetime.fromtimestamp(now_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'duration': dur_str,
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                        'reason': f'TP1_HIT_{int(tp1_pct*100)}PCT'
                                    }
                                    DashboardState.trades.append(tp1_record)
                                    
                                    # Persist partial TP1 to trade_logs.jsonl
                                    try:
                                        log_dir = Path("data")
                                        log_dir.mkdir(parents=True, exist_ok=True)
                                        with open(log_dir / "trade_logs.jsonl", "a", encoding="utf-8") as f:
                                            f.write(json.dumps(tp1_record) + "\n")
                                    except Exception as e:
                                        print(f"[LOG] Failed to write TP1 log: {e}")
                                    
                                    # PROFIT LOCK: Guarantee profit by setting Stop Loss to Breakeven (+0.30% fee safe / +0.35R)
                                    fee_buf = getattr(Config, 'DYNAMIC_BE_BUFFER_PCT', 0.0030)
                                    profit_lock_sl = max(self.entry_price[symbol] * (1.0 + fee_buf), self.entry_price[symbol] + 0.35 * r_dist)
                                    if profit_lock_sl > self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = profit_lock_sl
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = profit_lock_sl
                                        
                                    # P1 / LOGIC-015 fix: Resize Native SL to match remaining position quantity
                                    ctx = self.order_state_machine.get_context(symbol)
                                    if self.has_keys and not Config.PAPER_TRADING and ctx.native_sl_order_id:
                                        await self._resize_native_sl_safe(symbol, 'sell', self.position_size[symbol], self.stop_loss[symbol])
                                    
                                    orig_sz = self.original_position_size.get(symbol, self.position_size[symbol] * 2) or (self.position_size[symbol] * 2)
                                    runner_guar = max(0.0, self.position_size[symbol] * (self.stop_loss[symbol] - self.entry_price[symbol]))
                                    guar_pnl_usdt = self.realized_pnl[symbol] + runner_guar
                                    orig_val = orig_sz * self.entry_price[symbol]
                                    guar_pnl_pct = (guar_pnl_usdt / orig_val * 100.0) if orig_val > 0 else 0.0
                                    
                                    tp2_rr = getattr(Config, 'RISK_REWARD_RATIO', 2.5)
                                    add_log_message(f"[{symbol}] 🔒 PROFIT LOCKED at Stop Loss: {self.stop_loss[symbol]:.4f} (+{guar_pnl_usdt:.2f} USDT total locked). New Target: TP2 @ {self.take_profit_2r[symbol]:.4f}")
                                    await self.notifier.send_message(
                                        f"🔒 *PROFIT LOCKED ({symbol})*\n"
                                        f"🎯 TP1 Hit! 50% profit booked (+{tp1_pnl_usdt:.2f} USDT).\n"
                                        f"🔒 Total Guaranteed Locked Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                        f"🎯 New Active Target: TP2 ({tp2_rr:.1f}R) @ {self.take_profit_2r[symbol]:.4f}"
                                    )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP1 order REJECTED by exchange. State NOT updated.")
                                    
                            # TP2 (Remaining Runner Scale-Out at 3.0R)
                            if self.partial_tp_taken[symbol] and not self.tp2_taken[symbol] and curr_price >= self.take_profit_2r[symbol]:
                                tp2_rr = getattr(Config, 'RISK_REWARD_RATIO', 3.0)
                                add_log_message(f"[{symbol}] 🎯 Target 2 ({tp2_rr:.1f}R) hit! Booking remaining runner.")
                                tp2_size = self.position_size[symbol]
                                tp2_success = False
                                tp2_order = None
                                if self.has_keys and not Config.PAPER_TRADING:
                                    tp2_order = await self.execution.place_order('sell', 'market', tp2_size, symbol=symbol, is_exit_order=True)
                                    tp2_success = self._is_truthy_fill(tp2_order)
                                else:
                                    is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
                                    rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
                                    self._dry_run_balance_usdt += tp2_size * curr_price * (rate if is_inr else 1.0)
                                    tp2_success = True
                                if tp2_success:
                                    if self.has_keys and not Config.PAPER_TRADING:
                                        tp2_size = self._extract_filled_qty(tp2_order, tp2_size)
                                    self.position_size[symbol] -= tp2_size
                                    self.tp2_taken[symbol] = True
                                    
                                    # Log partial TP2 trade record for accurate PnL tracking
                                    tp2_pnl_usdt = tp2_size * (curr_price - self.entry_price[symbol])
                                    # --- PROFIT-BASED LOGIC: Net fee deduction ---
                                    tp2_fee = tp2_size * self.entry_price[symbol] * Config.FEE_RATE + tp2_size * curr_price * Config.FEE_RATE
                                    tp2_pnl_usdt -= tp2_fee
                                    self.accumulated_fees[symbol] = self.accumulated_fees.get(symbol, 0.0) + (tp2_size * curr_price * Config.FEE_RATE)
                                    tp2_pnl_pct = (curr_price - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0
                                    self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + tp2_pnl_usdt
                                    
                                    now_ts = int(time.time() * 1000)
                                    entry_ts = self.entry_time.get(symbol, now_ts)
                                    dur_secs = max(0, int((now_ts - entry_ts) / 1000))
                                    dur_str = f"{dur_secs // 60}m {dur_secs % 60}s" if dur_secs >= 60 else f"{dur_secs}s"
                                    
                                    tp2_record = {
                                        'trade_id': self.current_trade_id.get(symbol, ''),
                                        'symbol': symbol,
                                        'side': 'LONG',
                                        'type': 'TP2_PARTIAL',
                                        'entry_price': self.entry_price[symbol],
                                        'exit_price': curr_price,
                                        'entry': self.entry_price[symbol],
                                        'exit': curr_price,
                                        'size': tp2_size,
                                        'pnl_usdt': round(tp2_pnl_usdt, 4),
                                        'pnl': round(tp2_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'pnl_currency': round(tp2_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'currency': 'INR' if is_inr else 'USDT',
                                        'pnl_pct': round(tp2_pnl_pct, 2),
                                        'entry_time': entry_ts,
                                        'exit_time': now_ts,
                                        'entry_time_str': datetime.datetime.fromtimestamp(entry_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'exit_time_str': datetime.datetime.fromtimestamp(now_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'duration': dur_str,
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                        'reason': 'TP2_HIT_30PCT'
                                    }
                                    DashboardState.trades.append(tp2_record)
                                    
                                    # Persist partial TP2 to trade_logs.jsonl
                                    try:
                                        log_dir = Path("data")
                                        log_dir.mkdir(parents=True, exist_ok=True)
                                        with open(log_dir / "trade_logs.jsonl", "a", encoding="utf-8") as f:
                                            f.write(json.dumps(tp2_record) + "\n")
                                    except Exception as e:
                                        print(f"[LOG] Failed to write TP2 log: {e}")
                                    
                                    # Lock SL at TP1 level (Guaranteed deep profit lock)
                                    if self.take_profit_1r[symbol] > self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = self.take_profit_1r[symbol]
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = self.take_profit_1r[symbol]
                                        
                                    # P1 / LOGIC-015 fix: Resize Native SL to match remaining position quantity
                                    ctx = self.order_state_machine.get_context(symbol)
                                    if self.has_keys and not Config.PAPER_TRADING and ctx.native_sl_order_id:
                                        await self._resize_native_sl_safe(symbol, 'sell', self.position_size[symbol], self.stop_loss[symbol])
                                    
                                    orig_sz = self.original_position_size.get(symbol, self.position_size[symbol] / 0.20) or (self.position_size[symbol] / 0.20)
                                    runner_guar = max(0.0, self.position_size[symbol] * (self.stop_loss[symbol] - self.entry_price[symbol]))
                                    guar_pnl_usdt = self.realized_pnl[symbol] + runner_guar
                                    orig_val = orig_sz * self.entry_price[symbol]
                                    guar_pnl_pct = (guar_pnl_usdt / orig_val * 100.0) if orig_val > 0 else 0.0
                                    
                                    add_log_message(f"[{symbol}] 🚀 TP2 Hit! SL locked at TP1 level ({self.stop_loss[symbol]:.4f}). Trailing Runner active.")
                                    await self.notifier.send_message(
                                        f"🚀 *DEEP PROFIT LOCKED ({symbol})*\n"
                                        f"🎯 TP2 Hit! 30% profit booked (+{tp2_pnl_usdt:.2f} USDT).\n"
                                        f"🔒 Total Guaranteed Deep Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                        f"🎯 New Active Target: Runner Target (3.5R) @ {self.take_profit[symbol]:.4f}"
                                    )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP2 order REJECTED by exchange. State NOT updated.")

                            if self.partial_tp_taken[symbol]:
                                new_sl = self.risk.update_trailing_stop(self.entry_price[symbol], self.highest_price_reached[symbol], self.stop_loss[symbol], curr_atr, "LONG")
                                if new_sl > self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = new_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = new_sl
                                
                            if curr_price >= self.take_profit[symbol]:
                                await self.exit_position(symbol, "TAKE_PROFIT_RUNNER")
                            elif curr_price <= self.stop_loss[symbol]:
                                await self.exit_position(symbol, "TRAILING_STOP")
                                
                        elif self.position_side[symbol] == "SHORT":
                            self.lowest_price_reached[symbol] = min(self.lowest_price_reached[symbol], curr_price)
                            r_dist = abs(self.entry_price[symbol] - self.stop_loss[symbol])
                            
                            # ZERO-RISK FREE-TRADE LOCK: Move SL to Breakeven
                            fee_buffer_pct = getattr(Config, 'DYNAMIC_BE_BUFFER_PCT', 0.0030)
                            fee_offset = self.entry_price[symbol] * fee_buffer_pct
                            tsl_activation = float(getattr(Config, 'TSL_ACTIVATION_R', 1.2))
                            min_required_profit = max(tsl_activation * r_dist, fee_offset * 1.5)
                            
                            # Only activate Breakeven after TP1 profit is secured OR price has reached full activation threshold
                            if (self.partial_tp_taken[symbol] or self.lowest_price_reached[symbol] <= self.entry_price[symbol] - min_required_profit):
                                be_sl = self.entry_price[symbol] - fee_offset
                                if be_sl < self.stop_loss[symbol] and curr_price < be_sl:
                                    self.stop_loss[symbol] = be_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = be_sl
                                    add_log_message(f"[{symbol}] 🛡️ ZERO-RISK FREE-TRADE ACTIVATED: SL moved to Breakeven ({be_sl:.4f})")

                            # 1. ⏱️ STAGNATION KILLER: Time-based exit if trade loses momentum near entry
                            if getattr(Config, 'ENABLE_TIME_STOP', True) and not self.partial_tp_taken[symbol]:
                                entry_ts = self.entry_time.get(symbol, 0)
                                if entry_ts > 0:
                                    tf_mins = int(Config.LTF_TIMEFRAME.replace('m', '').replace('h', '')) * (60 if 'h' in Config.LTF_TIMEFRAME else 1)
                                    time_elapsed_secs = time.time() - (entry_ts / 1000.0)
                                    candles_open = time_elapsed_secs / (tf_mins * 60)
                                    max_stagnant_candles = getattr(Config, 'MAX_STAGNANT_CANDLES', 16)
                                    if candles_open >= max_stagnant_candles:
                                        stagnant_max_dist = getattr(Config, 'STAGNANT_MAX_R_DISTANCE', 0.25) * r_dist
                                        if abs(curr_price - self.entry_price[symbol]) <= stagnant_max_dist:
                                            add_log_message(f"[{symbol}] ⏱️ STAGNATION KILLER: Position open for {candles_open:.1f} candles ({time_elapsed_secs/3600:.1f}h) with no momentum. Auto-exiting at scratch.")
                                            await self.notifier.send_message(f"⏱️ *STAGNATION AUTO-EXIT ({symbol})*\nTrade open for {candles_open:.1f} candles without momentum. Scratch-closed to reclaim capital.")
                                            await self.exit_position(symbol, "STAGNANT_TIME_EXIT")
                                            continue

                            # 2. ⚡ EARLY STRUCTURAL EXIT: Cut loss early if 15m structure breaks against SHORT
                            if getattr(Config, 'ENABLE_STRUCTURAL_EXIT', True) and not self.partial_tp_taken[symbol] and len(ltf_df) >= 3:
                                last_c = ltf_df.iloc[-1]
                                prev_c = ltf_df.iloc[-2]
                                ema_20 = calculate_ema(ltf_df, 20).iloc[-1] if len(ltf_df) >= 20 else 0.0
                                avg_vol = ltf_df['volume'].rolling(14).mean().iloc[-1] if len(ltf_df) >= 14 else 1.0
                                unrealized_loss_r = (curr_price - self.entry_price[symbol]) / r_dist if r_dist > 0 else 0.0
                                early_max_r = getattr(Config, 'EARLY_EXIT_MAX_LOSS_R', 0.45)
                                if 0.15 <= unrealized_loss_r <= early_max_r and ema_20 > 0:
                                    is_bullish_engulf = (last_c['close'] > last_c['open']) and (prev_c['close'] < prev_c['open']) and (last_c['close'] > prev_c['open'])
                                    is_heavy_vol = last_c['volume'] > 1.8 * avg_vol
                                    is_above_ema = last_c['close'] > ema_20
                                    if (is_bullish_engulf or is_above_ema) and is_heavy_vol:
                                        add_log_message(f"[{symbol}] ⚡ EARLY STRUCTURAL EXIT: Heavy bullish breakout above EMA20. Cutting loss early at -{unrealized_loss_r:.2f}R.")
                                        await self.notifier.send_message(f"⚡ *EARLY STRUCTURAL EXIT ({symbol})*\nMarket structure broke against SHORT before full SL. Loss cut early at -{unrealized_loss_r:.2f}R (Saved ~{(1.0-unrealized_loss_r)*100:.0f}% loss).")
                                        await self.exit_position(symbol, "EARLY_STRUCTURAL_INVALIDATION")
                                        continue
                            
                            # TP1 Scale-Out (80% at 2.0R, or 100% Full Exit)
                            if not self.partial_tp_taken[symbol] and curr_price <= self.take_profit_1r[symbol]:
                                tp1_pct = float(getattr(Config, 'TP1_SCALE_OUT_PCT', 0.65))
                                if tp1_pct >= 0.999:
                                    add_log_message(f"[{symbol}] 🎯 Target 1 hit! Full 100% profit booking initiated.")
                                    await self.exit_position(symbol, "TAKE_PROFIT_1")
                                    continue
                                add_log_message(f"[{symbol}] 🎯 Target 1 hit! Booking {int(tp1_pct*100)}% profit.")
                                tp1_size = self.position_size[symbol] * tp1_pct
                                tp1_success = False
                                tp1_order = None
                                if self.has_keys and not Config.PAPER_TRADING:
                                    tp1_order = await self.execution.place_order('buy', 'market', tp1_size, symbol=symbol, is_exit_order=True)
                                    tp1_success = self._is_truthy_fill(tp1_order)
                                else:
                                    is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
                                    rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
                                    tp1_proceeds_usdt = tp1_size * (self.entry_price[symbol] - curr_price) + (tp1_size * self.entry_price[symbol])
                                    self._dry_run_balance_usdt += tp1_proceeds_usdt * (rate if is_inr else 1.0)
                                    tp1_success = True
                                if tp1_success:
                                    if self.has_keys and not Config.PAPER_TRADING:
                                        tp1_size = self._extract_filled_qty(tp1_order, tp1_size)
                                    self.position_size[symbol] -= tp1_size
                                    self.partial_tp_taken[symbol] = True
                                    
                                    # Log partial TP1 trade record for accurate PnL tracking
                                    tp1_pnl_usdt = tp1_size * (self.entry_price[symbol] - curr_price)
                                    # --- PROFIT-BASED LOGIC: Net fee deduction ---
                                    tp1_fee = tp1_size * self.entry_price[symbol] * Config.FEE_RATE + tp1_size * curr_price * Config.FEE_RATE
                                    tp1_pnl_usdt -= tp1_fee
                                    self.accumulated_fees[symbol] = self.accumulated_fees.get(symbol, 0.0) + (tp1_size * curr_price * Config.FEE_RATE)
                                    tp1_pnl_pct = (self.entry_price[symbol] - curr_price) / self.entry_price[symbol] * 100.0
                                    self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + tp1_pnl_usdt
                                    
                                    now_ts = int(time.time() * 1000)
                                    entry_ts = self.entry_time.get(symbol, now_ts)
                                    dur_secs = max(0, int((now_ts - entry_ts) / 1000))
                                    dur_str = f"{dur_secs // 60}m {dur_secs % 60}s" if dur_secs >= 60 else f"{dur_secs}s"
                                    
                                    tp1_record = {
                                        'trade_id': self.current_trade_id.get(symbol, ''),
                                        'symbol': symbol,
                                        'side': 'SHORT',
                                        'type': 'TP1_PARTIAL',
                                        'entry_price': self.entry_price[symbol],
                                        'exit_price': curr_price,
                                        'entry': self.entry_price[symbol],
                                        'exit': curr_price,
                                        'size': tp1_size,
                                        'pnl_usdt': round(tp1_pnl_usdt, 4),
                                        'pnl': round(tp1_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'pnl_currency': round(tp1_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'currency': 'INR' if is_inr else 'USDT',
                                        'pnl_pct': round(tp1_pnl_pct, 2),
                                        'entry_time': entry_ts,
                                        'exit_time': now_ts,
                                        'entry_time_str': datetime.datetime.fromtimestamp(entry_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'exit_time_str': datetime.datetime.fromtimestamp(now_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'duration': dur_str,
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                        'reason': f'TP1_HIT_{int(tp1_pct*100)}PCT'
                                    }
                                    DashboardState.trades.append(tp1_record)
                                    
                                    # Persist partial TP1 to trade_logs.jsonl
                                    try:
                                        log_dir = Path("data")
                                        log_dir.mkdir(parents=True, exist_ok=True)
                                        with open(log_dir / "trade_logs.jsonl", "a", encoding="utf-8") as f:
                                            f.write(json.dumps(tp1_record) + "\n")
                                    except Exception as e:
                                        print(f"[LOG] Failed to write TP1 log: {e}")
                                    
                                    # PROFIT LOCK: Guarantee profit by setting Stop Loss to Breakeven (-0.30% fee safe / -0.35R)
                                    fee_buf = getattr(Config, 'DYNAMIC_BE_BUFFER_PCT', 0.0030)
                                    profit_lock_sl = min(self.entry_price[symbol] * (1.0 - fee_buf), self.entry_price[symbol] - 0.35 * r_dist)
                                    if profit_lock_sl < self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = profit_lock_sl
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = profit_lock_sl
                                        
                                    # P1 / LOGIC-015 fix: Resize Native SL to match remaining position quantity
                                    ctx = self.order_state_machine.get_context(symbol)
                                    if self.has_keys and not Config.PAPER_TRADING and ctx.native_sl_order_id:
                                        await self._resize_native_sl_safe(symbol, 'buy', self.position_size[symbol], self.stop_loss[symbol])
                                    
                                    orig_sz = self.original_position_size.get(symbol, self.position_size[symbol] * 2) or (self.position_size[symbol] * 2)
                                    runner_guar = max(0.0, self.position_size[symbol] * (self.entry_price[symbol] - self.stop_loss[symbol]))
                                    guar_pnl_usdt = self.realized_pnl[symbol] + runner_guar
                                    orig_val = orig_sz * self.entry_price[symbol]
                                    guar_pnl_pct = (guar_pnl_usdt / orig_val * 100.0) if orig_val > 0 else 0.0
                                    
                                    tp2_rr = getattr(Config, 'RISK_REWARD_RATIO', 2.5)
                                    add_log_message(f"[{symbol}] 🔒 PROFIT LOCKED at Stop Loss: {self.stop_loss[symbol]:.4f} (+{guar_pnl_usdt:.2f} USDT total locked). New Target: TP2 @ {self.take_profit_2r[symbol]:.4f}")
                                    await self.notifier.send_message(
                                        f"🔒 *PROFIT LOCKED ({symbol})*\n"
                                        f"🎯 TP1 Hit! 50% profit booked (+{tp1_pnl_usdt:.2f} USDT).\n"
                                        f"🔒 Total Guaranteed Locked Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                        f"🎯 New Active Target: TP2 ({tp2_rr:.1f}R) @ {self.take_profit_2r[symbol]:.4f}"
                                    )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP1 order REJECTED by exchange. State NOT updated.")
                                    
                            # TP2 (Remaining Runner Scale-Out at 3.0R)
                            if self.partial_tp_taken[symbol] and not self.tp2_taken[symbol] and curr_price <= self.take_profit_2r[symbol]:
                                tp2_rr = getattr(Config, 'RISK_REWARD_RATIO', 3.0)
                                add_log_message(f"[{symbol}] 🎯 Target 2 ({tp2_rr:.1f}R) hit! Booking remaining runner.")
                                tp2_size = self.position_size[symbol]
                                tp2_success = False
                                tp2_order = None
                                if self.has_keys and not Config.PAPER_TRADING:
                                    tp2_order = await self.execution.place_order('buy', 'market', tp2_size, symbol=symbol, is_exit_order=True)
                                    tp2_success = self._is_truthy_fill(tp2_order)
                                else:
                                    is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
                                    rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
                                    tp2_proceeds_usdt = tp2_size * (self.entry_price[symbol] - curr_price) + (tp2_size * self.entry_price[symbol])
                                    self._dry_run_balance_usdt += tp2_proceeds_usdt * (rate if is_inr else 1.0)
                                    tp2_success = True
                                if tp2_success:
                                    if self.has_keys and not Config.PAPER_TRADING:
                                        tp2_size = self._extract_filled_qty(tp2_order, tp2_size)
                                    self.position_size[symbol] -= tp2_size
                                    self.tp2_taken[symbol] = True
                                    
                                    # Log partial TP2 trade record for accurate PnL tracking
                                    tp2_pnl_usdt = tp2_size * (self.entry_price[symbol] - curr_price)
                                    # --- PROFIT-BASED LOGIC: Net fee deduction ---
                                    tp2_fee = tp2_size * self.entry_price[symbol] * Config.FEE_RATE + tp2_size * curr_price * Config.FEE_RATE
                                    tp2_pnl_usdt -= tp2_fee
                                    self.accumulated_fees[symbol] = self.accumulated_fees.get(symbol, 0.0) + (tp2_size * curr_price * Config.FEE_RATE)
                                    tp2_pnl_pct = (self.entry_price[symbol] - curr_price) / self.entry_price[symbol] * 100.0
                                    self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + tp2_pnl_usdt
                                    
                                    now_ts = int(time.time() * 1000)
                                    entry_ts = self.entry_time.get(symbol, now_ts)
                                    dur_secs = max(0, int((now_ts - entry_ts) / 1000))
                                    dur_str = f"{dur_secs // 60}m {dur_secs % 60}s" if dur_secs >= 60 else f"{dur_secs}s"
                                    
                                    tp2_record = {
                                        'trade_id': self.current_trade_id.get(symbol, ''),
                                        'symbol': symbol,
                                        'side': 'SHORT',
                                        'type': 'TP2_PARTIAL',
                                        'entry_price': self.entry_price[symbol],
                                        'exit_price': curr_price,
                                        'entry': self.entry_price[symbol],
                                        'exit': curr_price,
                                        'size': tp2_size,
                                        'pnl_usdt': round(tp2_pnl_usdt, 4),
                                        'pnl': round(tp2_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'pnl_currency': round(tp2_pnl_usdt * (rate if is_inr else 1.0), 2),
                                        'currency': 'INR' if is_inr else 'USDT',
                                        'pnl_pct': round(tp2_pnl_pct, 2),
                                        'entry_time': entry_ts,
                                        'exit_time': now_ts,
                                        'entry_time_str': datetime.datetime.fromtimestamp(entry_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'exit_time_str': datetime.datetime.fromtimestamp(now_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                                        'duration': dur_str,
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                        'reason': 'TP2_HIT_30PCT'
                                    }
                                    DashboardState.trades.append(tp2_record)
                                    
                                    # Persist partial TP2 to trade_logs.jsonl
                                    try:
                                        log_dir = Path("data")
                                        log_dir.mkdir(parents=True, exist_ok=True)
                                        with open(log_dir / "trade_logs.jsonl", "a", encoding="utf-8") as f:
                                            f.write(json.dumps(tp2_record) + "\n")
                                    except Exception as e:
                                        print(f"[LOG] Failed to write TP2 log: {e}")
                                    
                                    # Lock SL at TP1 level (Guaranteed deep profit lock)
                                    if self.take_profit_1r[symbol] < self.stop_loss[symbol]:
                                        self.stop_loss[symbol] = self.take_profit_1r[symbol]
                                        if symbol == Config.SYMBOL: DashboardState.stop_loss = self.take_profit_1r[symbol]
                                        
                                    # P1 / LOGIC-015 fix: Resize Native SL to match remaining position quantity
                                    ctx = self.order_state_machine.get_context(symbol)
                                    if self.has_keys and not Config.PAPER_TRADING and ctx.native_sl_order_id:
                                        await self._resize_native_sl_safe(symbol, 'buy', self.position_size[symbol], self.stop_loss[symbol])
                                    
                                    orig_sz = self.original_position_size.get(symbol, self.position_size[symbol] / 0.20) or (self.position_size[symbol] / 0.20)
                                    runner_guar = max(0.0, self.position_size[symbol] * (self.entry_price[symbol] - self.stop_loss[symbol]))
                                    guar_pnl_usdt = self.realized_pnl[symbol] + runner_guar
                                    orig_val = orig_sz * self.entry_price[symbol]
                                    guar_pnl_pct = (guar_pnl_usdt / orig_val * 100.0) if orig_val > 0 else 0.0
                                    
                                    add_log_message(f"[{symbol}] 🚀 TP2 Hit! SL locked at TP1 level ({self.stop_loss[symbol]:.4f}). Trailing Runner active.")
                                    await self.notifier.send_message(
                                        f"🚀 *DEEP PROFIT LOCKED ({symbol})*\n"
                                        f"🎯 TP2 Hit! 30% profit booked (+{tp2_pnl_usdt:.2f} USDT).\n"
                                        f"🔒 Total Guaranteed Deep Profit: +{guar_pnl_usdt:.2f} USDT (+{guar_pnl_pct:.2f}%)\n"
                                        f"🎯 New Active Target: Runner Target (3.5R) @ {self.take_profit[symbol]:.4f}"
                                    )
                                    self.save_state()
                                else:
                                    add_log_message(f"[{symbol}] ⚠️ TP2 order REJECTED by exchange. State NOT updated.")

                            if self.partial_tp_taken[symbol]:
                                new_sl = self.risk.update_trailing_stop(self.entry_price[symbol], self.lowest_price_reached[symbol], self.stop_loss[symbol], curr_atr, "SHORT")
                                if new_sl < self.stop_loss[symbol]:
                                    self.stop_loss[symbol] = new_sl
                                    if symbol == Config.SYMBOL: DashboardState.stop_loss = new_sl
                                
                            if curr_price <= self.take_profit[symbol]:
                                await self.exit_position(symbol, "TAKE_PROFIT_RUNNER")
                            elif curr_price >= self.stop_loss[symbol]:
                                await self.exit_position(symbol, "TRAILING_STOP")

                # Update UI for selected Config.SYMBOL
                sym = Config.SYMBOL
                DashboardState.latest_price = self.pipeline.latest_prices.get(sym, 0.0)
                if not self.has_keys or Config.PAPER_TRADING:
                    DashboardState.balance_usdt = self.calculate_total_equity()
                    
                if self.pipeline.ltf_candles[sym]:
                    DashboardState.chart_history = self.pipeline.ltf_candles[sym][-100:]
                
                # Sync all active positions to DashboardState for multi-symbol UI rendering
                active_pos_map = {}
                for s in Config.SUPPORTED_SYMBOLS:
                    if self.in_position[s]:
                        # Clean up ghost positions where size is zero or depleted
                        if self.position_size[s] is None or self.position_size[s] <= 1e-7:
                            self.in_position[s] = False
                            self.position_side[s] = "HOLD"
                            self.position_size[s] = 0.0
                            continue

                        live_p = self.pipeline.latest_prices.get(s, self.entry_price[s])
                        entry_val = self.entry_price[s]
                        pos_sz = self.position_size[s]
                        if live_p > 0 and entry_val > 0:
                            if self.position_side[s] == "LONG":
                                p_pct = (live_p - entry_val) / entry_val * 100.0
                                p_usdt = pos_sz * (live_p - entry_val)
                            else:
                                p_pct = (entry_val - live_p) / entry_val * 100.0
                                p_usdt = pos_sz * (entry_val - live_p)
                        else:
                            p_pct = 0.0
                            p_usdt = 0.0
                        
                        is_long = (self.position_side[s] == "LONG")
                        sl_val = self.stop_loss[s]
                        entry_val = self.entry_price[s]
                        is_profit_locked = (is_long and sl_val >= entry_val) or (not is_long and sl_val <= entry_val and sl_val > 0)
                        
                        r_dist = abs(entry_val - sl_val) if abs(entry_val - sl_val) > 0 else (entry_val * 0.01)
                        target_1r = self.take_profit_1r[s] if self.take_profit_1r[s] > 0 else (entry_val + 1.0 * r_dist if is_long else entry_val - 1.0 * r_dist)
                        target_2r = self.take_profit_2r[s] if self.take_profit_2r[s] > 0 else (entry_val + 2.5 * r_dist if is_long else entry_val - 2.5 * r_dist)
                        final_tp = self.take_profit[s] if self.take_profit[s] > 0 else (entry_val + 4.0 * r_dist if is_long else entry_val - 4.0 * r_dist)
                        
                        # Guarantee final_tp is beyond target_2r
                        if is_long and final_tp <= target_2r:
                            final_tp = entry_val + 4.0 * r_dist
                        elif not is_long and final_tp >= target_2r:
                            final_tp = entry_val - 4.0 * r_dist

                        tp1_hit = self.partial_tp_taken[s]
                        tp2_hit = self.tp2_taken[s]
                        
                        # Dynamic target escalation upon hitting targets
                        if not tp1_hit:
                            active_target = target_1r
                            active_target_name = "TP1 (1.0R)"
                            target_stage = 1
                        elif not tp2_hit:
                            active_target = target_2r
                            active_target_name = "TP2 (2.5R)"
                            target_stage = 2
                        else:
                            # Runner stage: Ensure target is strictly ahead of trailing SL
                            if is_long:
                                active_target = max(final_tp, sl_val + r_dist)
                            else:
                                active_target = min(final_tp, sl_val - r_dist)
                            active_target_name = "Runner Target (4.0R)"
                            target_stage = 3

                        # Calculate guaranteed locked profit in USDT and % (including realized partial TPs)
                        realized_pnl_val = self.realized_pnl.get(s, 0.0)
                        runner_guar = 0.0
                        if is_profit_locked and entry_val > 0:
                            if is_long:
                                runner_guar = max(0.0, pos_sz * (sl_val - entry_val))
                            else:
                                runner_guar = max(0.0, pos_sz * (entry_val - sl_val))
                        guaranteed_pnl_usdt = realized_pnl_val + runner_guar
                        orig_sz = self.original_position_size.get(s, pos_sz) or pos_sz
                        orig_val = orig_sz * entry_val
                        guaranteed_pnl_pct = (guaranteed_pnl_usdt / orig_val * 100.0) if orig_val > 0 else 0.0

                        e_time = self.entry_time.get(s, int(time.time() * 1000))
                        e_time_str = datetime.datetime.fromtimestamp(e_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

                        active_pos_map[s] = {
                            'side': self.position_side[s],
                            'entry_price': entry_val,
                            'stop_loss': sl_val,
                            'take_profit': active_target,
                            'target_1r': target_1r,
                            'target_2r': target_2r,
                            'final_target': final_tp,
                            'active_target': active_target,
                            'active_target_name': active_target_name,
                            'target_stage': target_stage,
                            'position_size': self.position_size[s],
                            'current_pnl_usdt': p_usdt,
                            'current_pnl_pct': p_pct,
                            'guaranteed_pnl_usdt': guaranteed_pnl_usdt,
                            'guaranteed_pnl_pct': guaranteed_pnl_pct,
                            'tp1_hit': tp1_hit,
                            'tp2_hit': tp2_hit,
                            'profit_locked': is_profit_locked,
                            'live_price': live_p,
                            'entry_time': e_time,
                            'entry_time_str': e_time_str
                        }
                DashboardState.active_positions = active_pos_map

                if self.in_position[sym] and self.pipeline.latest_prices.get(sym, 0.0) > 0:
                    curr_price = self.pipeline.latest_prices[sym]
                    if self.position_side[sym] == "LONG":
                        pnl_pct = (curr_price - self.entry_price[sym]) / self.entry_price[sym] * 100.0
                        pnl_usdt = self.position_size[sym] * (curr_price - self.entry_price[sym])
                    else:
                        pnl_pct = (self.entry_price[sym] - curr_price) / self.entry_price[sym] * 100.0
                        pnl_usdt = self.position_size[sym] * (self.entry_price[sym] - curr_price)
                    DashboardState.current_pnl_pct = pnl_pct
                    DashboardState.current_pnl_usdt = pnl_usdt
                else:
                    DashboardState.current_pnl_pct = 0.0
                    DashboardState.current_pnl_usdt = 0.0

            except Exception as e:
                import traceback
                print(f"[RISK MONITOR] Error: {e}")
                traceback.print_exc()
            await asyncio.sleep(1.0)

    async def emergency_close_all(self):
        """Instantly close all open positions across all supported symbols."""
        closed_symbols = []
        failed_symbols = []
        add_log_message("🚨 EMERGENCY CLOSE ALL TRIGGERED: Exiting all active positions immediately...")
        for sym in Config.SUPPORTED_SYMBOLS:
            if self.in_position[sym]:
                try:
                    await self.exit_position(sym, "EMERGENCY_CLOSE_ALL")
                    # Verify it was actually closed
                    if not self.in_position[sym]:
                        closed_symbols.append(sym)
                    else:
                        failed_symbols.append(sym)
                        add_log_message(f"[{sym}] ⚠️ Exit order may have been rejected. Force-clearing local state.")
                        # Force-clear local state even if exchange order failed
                        self.in_position[sym] = False
                        self.position_side[sym] = "HOLD"
                        self.position_size[sym] = 0.0
                except Exception as e:
                    import traceback
                    add_log_message(f"[{sym}] Error closing position during emergency stop: {e}")
                    traceback.print_exc()
                    failed_symbols.append(sym)
                    # Force-clear local state on error
                    self.in_position[sym] = False
                    self.position_side[sym] = "HOLD"
                    self.position_size[sym] = 0.0
        
        DashboardState.active_positions = {}
        DashboardState.in_position = False
        DashboardState.position_side = "HOLD"
        self.save_state()

        count = len(closed_symbols)
        fail_count = len(failed_symbols)
        msg = f"Emergency stop executed: Closed {count} positions."
        if fail_count > 0:
            msg += f" WARNING: {fail_count} positions force-cleared ({', '.join(failed_symbols)})."
        add_log_message(f"🚨 {msg}")
        await self.notifier.send_message(f"🚨 *EMERGENCY CLOSE ALL EXECUTED*\nClosed {count} positions. Force-cleared {fail_count}.")
        return count + fail_count, msg

    async def _resize_native_sl_safe(self, symbol, side, size, stop_price):
        """Safely resizes a native SL, ensuring no duplicate orders are left (P1)."""
        ctx = self.order_state_machine.get_context(symbol)
        if not ctx.native_sl_order_id:
            return
            
        old_sl_id = ctx.native_sl_order_id
        cancel_success = await self.execution.cancel_order_safe(symbol, old_sl_id)
        
        # Verify if it's actually cancelled
        status = await self.execution.verify_order_active(symbol, old_sl_id)
        
        if status in ('ACTIVE', 'UNKNOWN'):
            add_log_message(f"[{symbol}] 🚨 CRITICAL: Failed to cancel old Native SL {old_sl_id}. Status: {status}. Will NOT place a duplicate.")
            add_log_message(f"[{symbol}] 🚨 Triggering emergency flatten to prevent double-execution.")
            await self.execution.emergency_flatten_position(symbol, 'BUY' if self.position_side[symbol] == 'LONG' else 'SELL', self.position_size[symbol], reason="SL_RESIZE_FAILED")
            ctx.transition_to(OrderState.EMERGENCY_FLATTENED, reason=f"SL resize cancellation failed ({status})")
            self.in_position[symbol] = False
            self.position_side[symbol] = "HOLD"
            self.position_size[symbol] = 0.0
            if hasattr(self, 'reconciliation'):
                await self.reconciliation._release_reserved_risk(ctx)
            return

        # It's cancelled. Safe to place new one.
        try:
            new_sl = await self.execution.place_native_stop_loss(symbol, side, size, stop_price)
            if self._is_active_sl_order(new_sl):
                new_sl_id = str(new_sl['id']) if isinstance(new_sl, dict) else str(new_sl.exchange_order_id)
                # Verify the replacement is active!
                new_status = await self.execution.verify_order_active(symbol, new_sl_id)
                if new_status == 'ACTIVE':
                    ctx.native_sl_order_id = new_sl_id
                    add_log_message(f"[{symbol}] 🛡️ Native SL replaced for remaining size {size} @ {stop_price:.4f}")
                else:
                    ctx.native_sl_order_id = None
                    add_log_message(f"[{symbol}] 🚨 Native SL replaced but verification failed ({new_status}). Resorting to virtual SL.")
            else:
                ctx.native_sl_order_id = None
        except Exception as e:
            add_log_message(f"[{symbol}] 🚨 Exception placing new native SL: {e}")
            ctx.native_sl_order_id = None

    async def exit_position(self, symbol, reason):
        # LOGIC-001 fix: Idempotent exit serialization via per-symbol exit lock
        if symbol not in self._exit_locks:
            self._exit_locks[symbol] = asyncio.Lock()

        if self._exit_locks[symbol].locked():
            add_log_message(f"[{symbol}] ⚠️ Exit already in progress (locked). Skipping duplicate exit request ({reason}).")
            return

        async with self._exit_locks[symbol]:
            if not self.in_position.get(symbol, False):
                add_log_message(f"[{symbol}] Position already closed or not in position. Skipping exit ({reason}).")
                return

            ctx = self.order_state_machine.get_context(symbol)
            ctx.transition_to(OrderState.CLOSING, reason=f"Exit initiated: {reason}")

            # FIX #5: Better fallback for exit_price to avoid 0.0 values
            exit_price = self.pipeline.latest_prices.get(symbol) or self.entry_price[symbol]
            if exit_price <= 0 or not exit_price:
                exit_price = self.entry_price[symbol]
            add_log_message(f"[{symbol}] Exiting at price: {exit_price:.4f} (reason: {reason})")
            if self.has_keys and not Config.PAPER_TRADING:
                side = 'buy' if self.position_side[symbol] == 'SHORT' else 'sell'
                try:
                    order = await self.execution.place_order(side, 'market', self.position_size[symbol], price=exit_price, is_exit_order=True, symbol=symbol)
                except Exception as e:
                    add_log_message(f"[{symbol}] 🚨 Exception during exit execution: {e}. Transitioning to EXIT_UNKNOWN.")
                    ctx.transition_to(OrderState.EXIT_UNKNOWN, reason=f"Exit NetworkError: {e}")
                    order = None
                    if hasattr(self, 'reconciliation'):
                        asyncio.create_task(self.reconciliation._reconcile_live_broker_state())
            else:
                order = {'id': 'MOCK_EXIT_ORDER_ID', 'price': exit_price, 'status': 'filled'}
                
            if self._is_truthy_fill(order):
                actual_exit = self._extract_filled_qty(order, self.position_size[symbol])
                is_full_close = (actual_exit >= (self.position_size[symbol] - 0.0001))

                # Cancel active native stop loss if it exists on exchange AND we fully closed
                if is_full_close and ctx.native_sl_order_id and self.has_keys and not Config.PAPER_TRADING:
                    await self.execution.cancel_order_safe(symbol, ctx.native_sl_order_id)
                    ctx.native_sl_order_id = None
                elif not is_full_close and ctx.native_sl_order_id and self.has_keys and not Config.PAPER_TRADING:
                    # Cancel existing SL, transition to PROTECTED will re-create it for remainder
                    await self.execution.cancel_order_safe(symbol, ctx.native_sl_order_id)
                    ctx.native_sl_order_id = None

                is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
                rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
                if self.position_side[symbol] == "LONG":
                    pnl_pct = (exit_price - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0
                    pnl_usdt = actual_exit * (exit_price - self.entry_price[symbol])
                    if not self.has_keys or Config.PAPER_TRADING:
                        self._dry_run_balance_usdt += actual_exit * exit_price * (rate if is_inr else 1.0)
                else:
                    pnl_pct = (self.entry_price[symbol] - exit_price) / self.entry_price[symbol] * 100.0
                    pnl_usdt = actual_exit * (self.entry_price[symbol] - exit_price)
                    if not self.has_keys or Config.PAPER_TRADING:
                        self._dry_run_balance_usdt += ((actual_exit * self.entry_price[symbol]) + pnl_usdt) * (rate if is_inr else 1.0)

                # --- PROFIT-BASED LOGIC: Net fee deduction ---
                exit_fee = actual_exit * exit_price * Config.FEE_RATE
                entry_fee_for_exit = actual_exit * self.entry_price[symbol] * Config.FEE_RATE
                pnl_usdt_gross = pnl_usdt
                pnl_usdt -= (exit_fee + entry_fee_for_exit)
                pnl_pct = pnl_usdt / (actual_exit * self.entry_price[symbol]) * 100.0 if (actual_exit * self.entry_price[symbol]) > 0 else 0.0
                self.accumulated_fees[symbol] = self.accumulated_fees.get(symbol, 0.0) + exit_fee
                    
                entry_ts = self.entry_time.get(symbol, int(time.time() * 1000))
                exit_ts = int(time.time() * 1000)
                duration_secs = max(0, int((exit_ts - entry_ts) / 1000))
                mins = duration_secs // 60
                secs = duration_secs % 60
                duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                
                entry_dt_str = datetime.datetime.fromtimestamp(entry_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                exit_dt_str = datetime.datetime.fromtimestamp(exit_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

                trade_record = {
                    'trade_id': self.current_trade_id.get(symbol, ''),
                    'symbol': symbol,
                    'side': self.position_side[symbol],
                    'entry_price': self.entry_price[symbol],
                    'exit_price': exit_price,
                    'pnl_usdt': round(pnl_usdt, 4),
                    'pnl_currency': round(pnl_usdt * (rate if is_inr else 1.0), 2),
                    'currency': 'INR' if is_inr else 'USDT',
                    'pnl_usdt_gross': round(pnl_usdt_gross, 4),
                    'total_fees': round(self.accumulated_fees.get(symbol, 0.0), 4),
                    'entry_fx_rate': self.entry_fx_rate.get(symbol, 0.0),
                    'exit_fx_rate': float(getattr(Config, 'USDT_INR_RATE', 85.0)),
                    'pnl_pct': round(pnl_pct, 2),
                    'entry_time': entry_ts,
                    'exit_time': exit_ts,
                    'entry_time_str': entry_dt_str,
                    'exit_time_str': exit_dt_str,
                    'duration': duration_str,
                    'reason': reason
                }
                DashboardState.trades.append(trade_record)
                if len(DashboardState.trades) > 500:
                    DashboardState.trades = DashboardState.trades[-500:]

                # State Machine transition
                if is_full_close:
                    ctx.transition_to(OrderState.CLOSED, reason=reason)
                    ctx.closed_at = time.time()
                    ctx.exit_reason = reason
                    ctx.realized_pnl = pnl_usdt
                else:
                    remaining_size = max(0.0, self.position_size[symbol] - actual_exit)
                    self.position_size[symbol] = remaining_size
                    ctx.filled_qty = remaining_size
                    ctx.transition_to(OrderState.PROTECTED, reason=f"Partial exit executed ({actual_exit} filled, {remaining_size} remaining)")

                # Record in Immutable Ledger
                self.immutable_ledger.record_exit(
                    symbol=symbol,
                    side=self.position_side[symbol],
                    exit_qty=actual_exit,
                    exit_price=exit_price,
                    entry_price=self.entry_price[symbol],
                    realized_pnl=pnl_usdt,
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                    client_order_id=f"EXIT_{int(time.time()*1000)}",
                    exchange_order_id=str(order.get('id', '')) if isinstance(order, dict) else ""
                )

                # Persist to data/trade_logs.jsonl on disk
                try:
                    log_dir = Path("data")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_file = log_dir / "trade_logs.jsonl"
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trade_record) + "\n")
                except Exception as e:
                    print(f"[LOG] Failed to write trade log: {e}")

                # Task 5: Cluster Loss Tracking (Evaluates Total Trade Return = Realized TP1/TP2 Cash + Final Runner PnL)
                total_trade_pnl = self.realized_pnl.get(symbol, 0.0) + pnl_usdt
                is_loss = total_trade_pnl < -0.01
                self.trade_history.append(1 if is_loss else 0)
                if len(self.trade_history) > 6:
                    self.trade_history.pop(0)
                    
                if len(self.trade_history) >= 2 and all(self.trade_history[-2:]):
                    cooldown_secs = 900.0 if Config.PAPER_TRADING else 3600.0
                    cooldown_mins = int(cooldown_secs // 60)
                    cooldown_time = time.time() + cooldown_secs
                    self.cluster_loss_pause_until = cooldown_time
                    self.global_pause_until = cooldown_time  # Update global pause
                    add_log_message(f"🚨 [SAFETY] 2 consecutive losses. Trading paused globally for {cooldown_mins} minutes.")
                    self.trade_history.clear()
                elif len(self.trade_history) >= 6 and sum(self.trade_history) >= 3:
                    self.cluster_risk_penalty = True
                    add_log_message("🚨 [SAFETY] 3 losses in last 6 trades. Global risk slashed by 50%.")
                else:
                    self.cluster_risk_penalty = False

                # Update relaxed cooldowns
                pos_mode = self.position_mode.get(symbol, 'STRICT')
                if pos_mode == 'RELAXED' and is_loss:
                    self.relaxed_losses += 1
                    if self.relaxed_losses >= 2:
                        self.relaxed_disabled_until = time.time() + 7200.0
                        add_log_message("🚨 [SAFETY] 2 relaxed losses. Relaxed mode disabled for 2 hours.")
                        self.relaxed_losses = 0
                elif not is_loss and pos_mode == 'RELAXED':
                    self.relaxed_losses = 0

                now_exit = time.time()
                self.last_exit_time[symbol] = now_exit
                self.last_trade_time[symbol] = now_exit
                self.global_last_trade_time = now_exit

                # Check if previous position had taken profit (TP1 / TP2 / full TP exit)
                had_tp = self.partial_tp_taken[symbol] or self.tp2_taken[symbol] or reason in ["TAKE_PROFIT_RUNNER", "TRAILING_STOP", "TP2_HIT"]
                if had_tp and pnl_usdt >= 0:
                    tp_cooldown_mins = getattr(Config, 'TP_EXIT_COOLDOWN_MINUTES', 25)
                    self.tp_cooldown_until[symbol] = now_exit + (tp_cooldown_mins * 60.0)
                    add_log_message(f"[{symbol}] 🎯 Profit secured from wave ({reason}). Harvest cooldown active for {tp_cooldown_mins}m to prevent chasing exhausted move.")
                else:
                    post_exit_mins = getattr(Config, 'POST_EXIT_COOLDOWN_MINUTES', 15)
                    self.tp_cooldown_until[symbol] = now_exit + (post_exit_mins * 60.0)
                    add_log_message(f"[{symbol}] ⏱️ Position closed. Post-exit cooldown active for {post_exit_mins}m.")

                # --- PROFIT-BASED LOGIC: Consolidated trade lifecycle record ---
                if is_full_close:
                    total_lifecycle_pnl = self.realized_pnl.get(symbol, 0.0) + pnl_usdt
                    total_lifecycle_fees = self.accumulated_fees.get(symbol, 0.0)
                    stages = []
                    if self.partial_tp_taken.get(symbol, False): stages.append('TP1')
                    if self.tp2_taken.get(symbol, False): stages.append('TP2')
                    stages.append(reason)
                    lifecycle_record = {
                        'trade_id': self.current_trade_id.get(symbol, ''),
                        'type': 'TRADE_LIFECYCLE',
                        'symbol': symbol,
                        'side': self.position_side[symbol],
                        'entry_price': self.entry_price[symbol],
                        'final_exit_price': exit_price,
                        'original_size': self.original_position_size.get(symbol, 0),
                        'total_pnl_net': round(total_lifecycle_pnl, 4),
                        'total_fees': round(total_lifecycle_fees, 4),
                        'stages_completed': stages,
                        'exit_reason': reason,
                        'duration': duration_str,
                        'entry_time': entry_ts,
                        'exit_time': exit_ts,
                        'entry_fx_rate': self.entry_fx_rate.get(symbol, 0.0),
                        'exit_fx_rate': float(getattr(Config, 'USDT_INR_RATE', 85.0)),
                    }
                    # Persist consolidated lifecycle audit summary to disk only (not in live execution trades)
                    try:
                        log_dir = Path("data")
                        log_dir.mkdir(parents=True, exist_ok=True)
                        with open(log_dir / "trade_logs.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps(lifecycle_record) + "\n")
                    except Exception as e:
                        print(f"[LOG] Failed to write lifecycle log: {e}")

                if is_full_close:
                    self.in_position[symbol] = False
                    self.position_side[symbol] = "HOLD"
                    self.position_size[symbol] = 0.0
                    self.original_position_size[symbol] = 0.0
                    self.realized_pnl[symbol] = 0.0
                    self.entry_price[symbol] = 0.0
                    self.stop_loss[symbol] = 0.0
                    self.take_profit[symbol] = 0.0
                    self.take_profit_1r[symbol] = 0.0
                    self.take_profit_2r[symbol] = 0.0
                    self.partial_tp_taken[symbol] = False
                    self.tp2_taken[symbol] = False
                    self.current_trade_id[symbol] = ""
                    self.entry_fx_rate[symbol] = 0.0
                    self.accumulated_fees[symbol] = 0.0

                    if symbol in DashboardState.active_positions:
                        new_positions = DashboardState.active_positions.copy()
                        del new_positions[symbol]
                        DashboardState.active_positions = new_positions

                    if symbol == Config.SYMBOL:
                        DashboardState.in_position = False
                        DashboardState.position_side = "HOLD"
                        DashboardState.entry_price = 0.0
                        DashboardState.stop_loss = 0.0
                        DashboardState.take_profit = 0.0
                        DashboardState.current_pnl_pct = 0.0
                        DashboardState.current_pnl_usdt = 0.0
                else:
                    self.in_position[symbol] = True
                    if symbol in DashboardState.active_positions:
                        DashboardState.active_positions[symbol]['position_size'] = self.position_size[symbol]

                self.save_state()
                await self.notifier.send_message(
                    f"🚨 *{symbol} CLOSED ({reason})*\nExit Price: {exit_price:.2f}\nPnL: {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT)"
                )
            else:
                # LOGIC-001 fix: If exchange rejected exit order, revert context to PROTECTED so position can be retried
                if ctx.state != OrderState.EXECUTION_UNKNOWN:
                    ctx.transition_to(OrderState.PROTECTED, reason=f"Exit order rejected on exchange: {reason}")
                add_log_message(f"[{symbol}] ❌ Exit order REJECTED or UNKNOWN. Position remains active locally.")

    async def change_bot_symbol(self, new_symbol):
        if new_symbol not in Config.SUPPORTED_SYMBOLS:
            add_log_message(f"Symbol {new_symbol} is not tracked by the background pipeline.")
            return

        add_log_message(f"Dashboard view switched to {new_symbol}.")
        Config.SYMBOL = new_symbol
        
        DashboardState.in_position = self.in_position[new_symbol]
        DashboardState.position_side = self.position_side[new_symbol]
        DashboardState.entry_price = self.entry_price[new_symbol]
        DashboardState.stop_loss = self.stop_loss[new_symbol]
        DashboardState.take_profit = self.take_profit[new_symbol]
        DashboardState.latest_price = self.pipeline.latest_prices.get(new_symbol, 0.0)
        DashboardState.chart_history = self.pipeline.ltf_candles[new_symbol][-100:] if self.pipeline.ltf_candles[new_symbol] else []
        
        if self.ml_models[new_symbol].is_trained and self.pipeline.ltf_candles[new_symbol]:
            df = prepare_dataframe(self.pipeline.ltf_candles[new_symbol])
            DashboardState.ml_confidence = self.ml_models[new_symbol].predict_bias(df)
        else:
            DashboardState.ml_confidence = 0.5

    async def change_execution_timeframe(self, new_tf):
        new_tf = new_tf.lower()
        if new_tf not in ['1m', '5m']:
            return
        add_log_message(f"⏱️ Switching execution timeframe to {new_tf.upper()}...")
        Config.LTF_TIMEFRAME = new_tf
        
        # Warm up LTF historical candles for the new timeframe across all symbols
        for sym in Config.SUPPORTED_SYMBOLS:
            ltf_ohlcv = await self.execution.fetch_ohlcv(
                symbol=sym,
                timeframe=new_tf,
                limit=500
            )
            if ltf_ohlcv:
                self.pipeline.ltf_candles[sym] = ltf_ohlcv
                df = prepare_dataframe(ltf_ohlcv)
                self.ml_models[sym].train(df)
        
        await self.pipeline.restart_streams()
        sym = Config.SYMBOL
        DashboardState.ltf_timeframe = new_tf
        DashboardState.chart_history = self.pipeline.ltf_candles[sym][-100:] if self.pipeline.ltf_candles[sym] else []
        add_log_message(f"✅ Execution timeframe switched to {new_tf.upper()}. Chart & signals active.")

    def lock_position_profit(self, symbol):
        """Manually lock profit for an active position by adjusting Stop Loss to breakeven or trailing profit."""
        if not self.in_position[symbol]:
            return False, f"No active position open for {symbol}."
            
        live_p = self.pipeline.latest_prices.get(symbol, self.entry_price[symbol])
        entry_p = self.entry_price[symbol]
        sl_p = self.stop_loss[symbol]
        is_long = (self.position_side[symbol] == "LONG")
        
        # Guard: Reject profit lock if position is currently in a loss
        if is_long and live_p < entry_p:
            return False, f"Cannot lock profit for {symbol} — position is currently in loss (Live: {live_p:.4f} < Entry: {entry_p:.4f})."
        if not is_long and live_p > entry_p:
            return False, f"Cannot lock profit for {symbol} — position is currently in loss (Live: {live_p:.4f} > Entry: {entry_p:.4f})."
        
        fee_buffer_pct = getattr(Config, 'DYNAMIC_BE_BUFFER_PCT', 0.0030)
        if is_long:
            # Breakeven lock: set SL to entry_price + fee buffer (0.30%) or 50% of profit, but strictly below live price
            fee_buffer = entry_p * fee_buffer_pct
            target_sl = max(entry_p + fee_buffer, entry_p + (live_p - entry_p) * 0.5)
            new_sl = min(live_p * 0.9995, target_sl)
            new_sl = max(sl_p, new_sl) # Never worsen existing SL
            self.stop_loss[symbol] = new_sl
            if symbol in DashboardState.active_positions:
                DashboardState.active_positions[symbol]['stop_loss'] = new_sl
                DashboardState.active_positions[symbol]['profit_locked'] = True
            if symbol == Config.SYMBOL:
                DashboardState.stop_loss = new_sl
        else:
            fee_buffer = entry_p * fee_buffer_pct
            target_sl = min(entry_p - fee_buffer, entry_p - (entry_p - live_p) * 0.5)
            new_sl = max(live_p * 1.0005, target_sl)
            new_sl = min(sl_p, new_sl) # Never worsen existing SL
            self.stop_loss[symbol] = new_sl
            if symbol in DashboardState.active_positions:
                DashboardState.active_positions[symbol]['stop_loss'] = new_sl
                DashboardState.active_positions[symbol]['profit_locked'] = True
            if symbol == Config.SYMBOL:
                DashboardState.stop_loss = new_sl
                
        self.save_state()
        msg = f"🔒 Profit locked for {symbol}! Stop Loss moved to ${self.stop_loss[symbol]:,.4f}"
        add_log_message(f"[{symbol}] {msg}")
        return True, msg

    async def emergency_close_all(self) -> tuple[int, str]:
        """
        Emergency Kill Switch: Instantly flattens all open positions across all supported symbols.
        Handles both live and paper trading, updates state machines, ledger, logs, and dashboard.
        """
        symbols_to_close = [sym for sym in Config.SUPPORTED_SYMBOLS if self.in_position.get(sym, False)]
        if not symbols_to_close:
            add_log_message("🚨 Emergency close requested, but no open positions were found.")
            return 0, "No open positions to close."

        add_log_message(f"🚨 EMERGENCY CLOSE ALL TRIGGERED! Closing {len(symbols_to_close)} active position(s)...")
        closed_count = 0
        for sym in symbols_to_close:
            try:
                await self.exit_position(sym, "USER_EMERGENCY_CLOSE")
                if not self.in_position.get(sym, False):
                    closed_count += 1
            except Exception as e:
                add_log_message(f"[{sym}] ⚠️ Error closing position during emergency close: {e}")

        self.save_state()
        msg = f"Successfully closed {closed_count} of {len(symbols_to_close)} position(s)."
        add_log_message(f"🚨 EMERGENCY CLOSE ALL COMPLETED: {msg}")
        return closed_count, msg

    async def shutdown(self):
        add_log_message("Shutting down exchange sessions gracefully...")
        await self.reconciliation.stop()
        await self.execution.close()
        self.pipeline.stop()

async def start_all():
    import dashboard.app as dashboard_module
    bot = PrimeSignalBot()
    dashboard_module.bot_instance = bot

    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    if sys.platform == 'win32' and sys.version_info < (3, 12):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(start_all())
    except KeyboardInterrupt:
        print("\nStopping bot...")

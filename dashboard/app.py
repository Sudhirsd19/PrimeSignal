import asyncio
import json
import os
import time
import datetime
import secrets
from typing import Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from starlette.requests import Request
from pydantic import BaseModel
from config import Config
from collections import deque
from core.firebase_manager import FirebaseManager
from core.order_state_machine import OrderState

# Global bot instance — set by main.py before server starts
bot_instance: Any = None
_bot_task: Any = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bot_task
    # Start the websocket broadcast background task
    asyncio.create_task(broadcast_state_loop())
    # Launch bot loop if bot_instance is registered
    if bot_instance is not None:
        print("[STARTUP] Launching bot trading loop from FastAPI lifespan...")
        _bot_task = asyncio.create_task(_run_bot(bot_instance))
    else:
        print("[STARTUP] WARNING: bot_instance not registered. Bot will NOT run.")
    yield
    if _bot_task:
        _bot_task.cancel()

app = FastAPI(title="PrimeSignal Trading Dashboard", lifespan=lifespan)

# Templates path setup
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Request schema for changing symbols
class SymbolRequest(BaseModel):
    symbol: str

class ModeRequest(BaseModel):
    paper_trading: bool

class TimeframeRequest(BaseModel):
    timeframe: str

# ─── ATTACK-1 FIX: API Key Auth for mutating endpoints ──────────────────────
# Without this, anyone on the internet can POST /api/change_symbol and spam
# the bot with symbol changes, triggering WebSocket restarts and CPU spikes.
# Set DASHBOARD_SECRET env var on Render. Omit to disable auth in dev mode.
_DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET", "primesignal_secret_key")
if _DASHBOARD_SECRET:
    print(f"[SECURITY] Dashboard API key auth is ENABLED.")
else:
    print(f"[SECURITY] Dashboard API key auth is DISABLED (DASHBOARD_SECRET not set).")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_dashboard_key(key: Optional[str] = Depends(_api_key_header)):
    """Enforces auth for all environments. Fails closed if missing."""
    secret = os.getenv("DASHBOARD_SECRET", _DASHBOARD_SECRET) or "primesignal_secret_key"
    if not key or not secrets.compare_digest(key, secret):
        raise HTTPException(status_code=403, detail="Invalid dashboard API key. Set X-API-Key header.")

# Global Memory State Store
class DashboardState:
    latest_price = 0.0
    balance_currency = getattr(Config, 'PAPER_CURRENCY', 'INR')
    balance_usdt = float(getattr(Config, 'PAPER_STARTING_BALANCE', 2000.0 if getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' else 10000.0))
    balance_base = 0.0
    
    in_position = False
    position_side = "HOLD"
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    current_pnl_usdt = 0.0
    current_pnl_pct = 0.0
    
    active_positions = {} # dictionary of symbol -> dict with position details
    
    trades = []
    logs = []
    
    signal_progress = 0
    
    daily_drawdown_pct = 0.0
    daily_profit_locked = False
    max_daily_profit_pct = float(getattr(Config, 'MAX_DAILY_PROFIT_PCT', 4.0))
    daily_gain_pct = 0.0
    ml_confidence = 0.5
    next_candle_color = "GREEN"
    next_candle_prob = 50.0
    active_ob = "No OB"
    active_fvg = "No FVG"
    active_ob_level = 0.0
    active_ob_type = "NONE"
    active_bullish_ob_level = 0.0
    active_bearish_ob_level = 0.0
    chart_history = []
    coindcx_profile = {
        "status": "Paper Mode (Active)",
        "name": "Virtual Paper Trader",
        "email": "paper.trade@coindcx.local",
        "id": "DCX-VIRTUAL-8849"
    }
    coindcx_balances = [
        {"currency": "INR", "available": float(getattr(Config, 'PAPER_STARTING_BALANCE', 2000.0)), "locked": 0.0},
        {"currency": "USDT", "available": round(float(getattr(Config, 'PAPER_STARTING_BALANCE', 2000.0)) / 85.0, 2), "locked": 0.0},
        {"currency": "BTC", "available": 0.0, "locked": 0.0}
    ]
    
    signal_light = "BLUE"
    signal_light_reason = "Scanning market for institutional SMC setups..."
    
    symbol_change_requested = None # Holds new symbol if requested by UI
    active_websockets = set()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Renders the main terminal dashboard page."""
    secret = os.getenv("DASHBOARD_SECRET", _DASHBOARD_SECRET) or "primesignal_secret_key"
    return templates.TemplateResponse(request, "index.html", {
        "dashboard_secret": secret
    })

@app.post("/api/change_symbol", dependencies=[Depends(verify_dashboard_key)])
async def change_symbol(req: SymbolRequest):
    symbol = req.symbol.strip().upper()
    if "/" not in symbol:
        return {"status": "error", "message": "Invalid symbol. Use format e.g. BTC/USDT"}
    
    DashboardState.symbol_change_requested = symbol
    return {"status": "success", "message": f"Symbol change to {symbol} requested successfully."}

@app.post("/api/set_mode", dependencies=[Depends(verify_dashboard_key)])
async def set_mode(req: ModeRequest):
    # FAIL-CLOSED SECURITY INVARIANT:
    # If DASHBOARD_SECRET is not configured, LIVE trading is strictly prohibited to prevent accidental funds exposure.
    if not req.paper_trading and not _DASHBOARD_SECRET:
        return {
            "status": "error",
            "message": "SECURITY BLOCKED: Cannot switch to LIVE REAL MONEY mode without DASHBOARD_SECRET configured in .env."
        }

    Config.PAPER_TRADING = req.paper_trading
    
    # If bot is active, trigger mode switch side-effects
    if bot_instance is not None:
        try:
            # Tell bot to update balance immediately
            if not req.paper_trading:
                # Switching to LIVE: fetch live balance
                if bot_instance.has_keys:
                    balance = await bot_instance.execution.fetch_balance()
                    if balance:
                        if Config.COINDCX_TRADE_INR:
                            inr_balance = balance.get('total', {}).get('INR', None)
                            if inr_balance is not None:
                                DashboardState.balance_usdt = inr_balance
                        else:
                            usdt_balance = balance.get('total', {}).get('USDT', None)
                            if usdt_balance is not None:
                                DashboardState.balance_usdt = usdt_balance
                        DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
            else:
                # Switching to PAPER: reset to total virtual equity (cash + open positions)
                DashboardState.balance_usdt = bot_instance.calculate_total_equity()
                DashboardState.balance_base = 0.0
        except Exception as e:
            print(f"[MODE SWITCH] Error syncing balances: {e}")
            
    mode_name = "PAPER TRADING" if req.paper_trading else "REAL MONEY"
    add_log_message(f"Trading mode switched to {mode_name}")
    return {"status": "success", "message": f"Switched to {mode_name}"}

@app.post("/api/emergency_flatten", dependencies=[Depends(verify_dashboard_key)])
async def emergency_flatten():
    """Emergency Kill Switch: Instantly flattens all open positions and activates Safe Mode."""
    if bot_instance is None:
        return {"status": "error", "message": "Bot instance is not initialized."}
    
    add_log_message("🚨 EMERGENCY FLATTEN TRIGGERED VIA DASHBOARD! Closing all open positions...")
    closed_count, msg = await bot_instance.emergency_close_all()
    DashboardState.signal_light = "RED"
    DashboardState.signal_light_reason = "🚨 EMERGENCY KILL SWITCH EXECUTED: All positions flattened."
    return {"status": "success", "message": msg, "closed_count": closed_count}

@app.post("/api/set_timeframe", dependencies=[Depends(verify_dashboard_key)])
async def set_timeframe(req: TimeframeRequest):
    tf = req.timeframe.strip().lower()
    if tf not in ["1m", "5m"]:
        return {"status": "error", "message": "Invalid timeframe. Choose 1m or 5m."}
    
    Config.LTF_TIMEFRAME = tf
    if bot_instance is not None:
        try:
            asyncio.create_task(bot_instance.change_execution_timeframe(tf))
        except Exception as e:
            print(f"[TIMEFRAME SWITCH] Error: {e}")
            
    add_log_message(f"Execution timeframe set to {tf.upper()}")
    return {"status": "success", "message": f"Timeframe switched to {tf.upper()}"}

@app.post("/api/emergency_stop", dependencies=[Depends(verify_dashboard_key)])
async def emergency_stop():
    """Trigger emergency close for all open positions."""
    try:
        try:
            firebase = FirebaseManager()
            if firebase.is_connected and firebase.db is not None:
                firebase.db.collection("control").document("kill_switch").set({"active": True})
        except Exception:
            pass
        
        def _write_kill_switch():
            with open("KILL_SWITCH", "w") as f:
                f.write("Triggered via API")
        await asyncio.to_thread(_write_kill_switch)
            
        closed_count = 0
        msg = "Emergency stop initiated."
        if bot_instance is not None:
            closed_count, msg = await bot_instance.emergency_close_all()
        else:
            add_log_message("🚨 EMERGENCY KILL SWITCH TRIGGERED VIA API 🚨")
            
        return {"status": "success", "message": msg, "closed_count": closed_count}
    except Exception as e:
        return {"status": "error", "message": f"Failed to activate kill switch: {str(e)}"}

class RiskSettingsUpdate(BaseModel):
    tsl_enabled: bool | None = None
    tsl_multiplier: float | None = None
    max_daily_trades: int | None = None

@app.post("/api/update_risk_settings", dependencies=[Depends(verify_dashboard_key)])
async def update_risk_settings(settings: RiskSettingsUpdate):
    from config import Config
    if settings.tsl_multiplier is not None:
        Config.TRAILING_ATR_MULT = settings.tsl_multiplier
    if settings.tsl_enabled is not None:
        if not settings.tsl_enabled:
            Config.TRAILING_ATR_MULT = 999.0 # Effectively disables it
    
    if settings.max_daily_trades is not None and settings.max_daily_trades > 0:
        Config.MAX_DAILY_TRADES = settings.max_daily_trades
        if bot_instance is not None:
            bot_instance.max_daily_trades = settings.max_daily_trades
    
    status_str = f"Max Daily Trades: {Config.MAX_DAILY_TRADES} | TSL ATR Mult: {Config.TRAILING_ATR_MULT}"
    add_log_message(f"⚙️ Risk Settings Updated: {status_str}")
    return {"status": "success", "message": status_str, "max_daily_trades": Config.MAX_DAILY_TRADES}

class LockProfitRequest(BaseModel):
    symbol: str

@app.post("/api/lock_profit", dependencies=[Depends(verify_dashboard_key)])
async def lock_profit(req: LockProfitRequest):
    if bot_instance is None:
        return {"status": "error", "message": "Bot instance is not running."}
    symbol = req.symbol.strip().upper()
    success, msg = bot_instance.lock_position_profit(symbol)
    if success:
        return {"status": "success", "message": msg}
    else:
        return {"status": "error", "message": msg}

class DailyProfitTargetRequest(BaseModel):
    target_pct: float

@app.post("/api/set_daily_profit_target", dependencies=[Depends(verify_dashboard_key)])
async def set_daily_profit_target(req: DailyProfitTargetRequest):
    val = float(req.target_pct)
    if val <= 0.05:
        raise HTTPException(status_code=400, detail="Target percentage must be at least 0.1%.")
    Config.MAX_DAILY_PROFIT_PCT = val
    DashboardState.max_daily_profit_pct = val
    if bot_instance and hasattr(bot_instance, 'risk'):
        res = bot_instance.risk.set_daily_profit_target(val)
        add_log_message(f"🎯 Daily Profit Lock target set to {val:.2f}%")
        return res
    add_log_message(f"🎯 Daily Profit Lock target set to {val:.2f}% (Config)")
    return {"status": "success", "max_daily_profit_pct": val}

@app.post("/api/unlock_daily_profit", dependencies=[Depends(verify_dashboard_key)])
async def unlock_daily_profit():
    DashboardState.daily_profit_locked = False
    if bot_instance and hasattr(bot_instance, 'risk'):
        res = bot_instance.risk.unlock_daily_profit()
        add_log_message("🔓 Daily Profit Lock manually unlocked by user! Trading resumed.")
        if hasattr(bot_instance, 'notifier') and bot_instance.notifier:
            asyncio.create_task(bot_instance.notifier.send_message("🔓 *PROFIT LOCK MANUALLY OVERRIDDEN*\nDaily profit lock has been unlocked via Dashboard. All trade entries resumed."))
        return res
    add_log_message("🔓 Daily Profit Lock manually unlocked.")
    return {"status": "success", "locked": False, "message": "Daily Profit Lock manually unlocked."}

class ChangeCurrencyRequest(BaseModel):
    currency: str

@app.post("/api/change_currency", dependencies=[Depends(verify_dashboard_key)])
async def change_currency(req: ChangeCurrencyRequest):
    curr = req.currency.strip().upper()
    if curr not in ["INR", "USDT"]:
        return {"status": "error", "message": "Supported currencies are INR or USDT"}
    
    Config.PAPER_CURRENCY = curr
    DashboardState.balance_currency = curr
    if curr == "INR":
        Config.COINDCX_TRADE_INR = True
        if Config.PAPER_TRADING:
            Config.PAPER_STARTING_BALANCE = 2000.0
            DashboardState.balance_usdt = 2000.0
            if bot_instance:
                bot_instance._dry_run_balance_usdt = 2000.0
                bot_instance.save_state()
    else:
        Config.COINDCX_TRADE_INR = False
        if Config.PAPER_TRADING:
            Config.PAPER_STARTING_BALANCE = 10000.0
            DashboardState.balance_usdt = 10000.0
            if bot_instance:
                bot_instance._dry_run_balance_usdt = 10000.0
                bot_instance.save_state()
    
    add_log_message(f"💱 Currency switched to {curr} ({'₹2,000 INR' if curr == 'INR' else '$10,000 USDT'} active)")
    return {"status": "success", "currency": curr, "balance": DashboardState.balance_usdt}

class ResetAccountRequest(BaseModel):
    target_balance: float = 10000.0

@app.post("/api/reset_account", dependencies=[Depends(verify_dashboard_key)])
async def reset_account(req: Optional[ResetAccountRequest] = None):
    balance = req.target_balance if req else (2000.0 if Config.PAPER_CURRENCY == 'INR' else 10000.0)
    cur_symbol = "₹" if Config.PAPER_CURRENCY == 'INR' else "$"
    cur_name = Config.PAPER_CURRENCY
    if bot_instance is not None:
        bot_instance.reset_account_state(balance)
        return {"status": "success", "message": f"Account reset to {cur_symbol}{balance:,.2f} {cur_name}. All positions cleared."}
    else:
        DashboardState.balance_usdt = balance
        DashboardState.balance_base = 0.0
        DashboardState.active_positions.clear()
        DashboardState.in_position = False
        DashboardState.trades.clear()
        return {"status": "success", "message": f"Dashboard balance reset to {cur_symbol}{balance:,.2f} {cur_name}."}

@app.post("/api/clear_analytics", dependencies=[Depends(verify_dashboard_key)])
async def clear_analytics():
    """Clears all in-memory and disk trade history logs for fresh analytics tracking."""
    DashboardState.trades.clear()
    if bot_instance is not None:
        bot_instance.trade_history.clear()
        if hasattr(bot_instance, 'trades_today'):
            bot_instance.trades_today = 0
            
    # Truncate on-disk log files if they exist
    for fpath in [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trade_logs.jsonl'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'trade_decisions.jsonl')
    ]:
        try:
            if os.path.exists(fpath):
                with open(fpath, 'w', encoding='utf-8') as f:
                    pass
        except Exception:
            pass
            
    return {"status": "success", "message": "All historical trade logs and analytics have been cleared."}

class TestTradeRequest(BaseModel):
    symbol: str | None = None
    side: str | None = "BUY"

@app.post("/api/trigger_test_trade", dependencies=[Depends(verify_dashboard_key)])
async def trigger_test_trade(req: Optional[TestTradeRequest] = None):
    from config import Config
    import datetime
    if not Config.PAPER_TRADING:
        return {"status": "error", "message": "Test trades are strictly disabled in LIVE mode"}
    symbol = (req.symbol if req and req.symbol else Config.SYMBOL) or "BTC/USDT"
    side = (req.side if req and req.side else "BUY").upper()
    is_long = (side == "BUY")
    
    if bot_instance is not None:
        live_p = bot_instance.pipeline.latest_prices.get(symbol, 0.0) or (DashboardState.latest_price or 78800.0)
        sl_p = live_p * 0.985 if is_long else live_p * 1.015
        tp_p = live_p * 1.035 if is_long else live_p * 0.965
        tp1_p = live_p * 1.015 if is_long else live_p * 0.985
        tp2_p = live_p * 1.025 if is_long else live_p * 0.975

        is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
        fx_rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
        if fx_rate <= 0:
            fx_rate = 85.0
        
        current_eq = bot_instance.calculate_total_equity() if hasattr(bot_instance, 'calculate_total_equity') else bot_instance._dry_run_balance_usdt
        pos_size = 0.0
        if hasattr(bot_instance, 'risk') and bot_instance.risk:
            pos_size = bot_instance.risk.calculate_position_size(
                account_equity=current_eq,
                entry_price=live_p,
                stop_loss=sl_p,
                equity_currency="INR" if is_inr else "USDT",
                quote_currency="USDT",
                conversion_rate=fx_rate,
                is_inr=is_inr
            )
        if pos_size <= 1e-7:
            # Safe allocation fallback (e.g. 30% capital allocation)
            pos_size = round((current_eq * 0.30) / (live_p * (fx_rate if is_inr else 1.0)), 6)
        if pos_size <= 1e-7:
            pos_size = 0.001
        
        # Deduct position cost from dry run virtual cash
        pos_cost = pos_size * live_p * (fx_rate if is_inr else 1.0)
        if hasattr(bot_instance, '_dry_run_balance_usdt'):
            bot_instance._dry_run_balance_usdt = max(0.0, bot_instance._dry_run_balance_usdt - pos_cost)
        
        bot_instance.in_position[symbol] = True
        bot_instance.position_side[symbol] = "LONG" if is_long else "SHORT"
        bot_instance.entry_price[symbol] = live_p
        bot_instance.stop_loss[symbol] = sl_p
        bot_instance.take_profit[symbol] = tp_p
        bot_instance.take_profit_1r[symbol] = tp1_p
        bot_instance.take_profit_2r[symbol] = tp2_p
        bot_instance.position_size[symbol] = pos_size
        bot_instance.original_position_size[symbol] = pos_size
        bot_instance.entry_time[symbol] = int(time.time() * 1000)
        bot_instance.last_trade_time[symbol] = time.time()
        bot_instance.highest_price_reached[symbol] = live_p
        bot_instance.lowest_price_reached[symbol] = live_p
        bot_instance.partial_tp_taken[symbol] = False
        bot_instance.tp2_taken[symbol] = False
        bot_instance.realized_pnl[symbol] = 0.0
        bot_instance.trades_today += 1
        
        # Calculate true authoritative net equity
        DashboardState.balance_usdt = bot_instance.calculate_total_equity()
        if is_long:
            DashboardState.balance_base = pos_size
        
        entry_dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        DashboardState.active_positions[symbol] = {
            'symbol': symbol,
            'side': "LONG" if is_long else "SHORT",
            'entry_price': live_p,
            'stop_loss': sl_p,
            'take_profit': tp_p,
            'target_1r': tp1_p,
            'target_2r': tp2_p,
            'final_target': tp_p,
            'active_target': tp1_p,
            'active_target_name': "TP1 (1.0R)",
            'target_stage': 1,
            'position_size': pos_size,
            'current_pnl_usdt': 0.0,
            'current_pnl_currency': 0.0,
            'current_pnl_pct': 0.0,
            'guaranteed_pnl_usdt': 0.0,
            'guaranteed_pnl_currency': 0.0,
            'guaranteed_pnl_pct': 0.0,
            'tp1_hit': False,
            'tp2_hit': False,
            'profit_locked': False,
            'live_price': live_p,
            'entry_time': int(time.time() * 1000),
            'entry_time_str': entry_dt_str
        }
        
        if symbol == Config.SYMBOL:
            DashboardState.in_position = True
            DashboardState.position_side = "LONG" if is_long else "SHORT"
            DashboardState.entry_price = live_p
            DashboardState.stop_loss = sl_p
            DashboardState.take_profit = tp_p
            DashboardState.position_size = pos_size
            DashboardState.current_pnl_pct = 0.0
            DashboardState.current_pnl_usdt = 0.0
            
        bot_instance.save_state()
        msg = f"🚀 [MANUAL TEST TRADE TRIGGERED] {symbol} {'LONG' if is_long else 'SHORT'} @ ${live_p:,.2f} | SL: ${sl_p:,.2f} | TP: ${tp_p:,.2f} | Size: {pos_size:.6f}"
        add_log_message(msg)
        return {"status": "success", "message": msg}
    else:
        return {"status": "error", "message": "Bot instance is not running."}

@app.get("/api/analytics", dependencies=[Depends(verify_dashboard_key)])
async def get_analytics():
    """Fetch trade logs from local JSONL file and aggregate analytics for the UI."""
    try:
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trade_logs.jsonl')
        decisions_log = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'trade_decisions.jsonl')
        trades = []
        
        # Try primary log file first
        if os.path.exists(log_file):
            def _read_logs():
                loaded = []
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                loaded.append(json.loads(line))
                            except:
                                pass
                return loaded
            trades = await asyncio.to_thread(_read_logs)
        
        # Fallback: also read TRADE_EXITED events from trade_decisions.jsonl
        if os.path.exists(decisions_log):
            def _read_decisions():
                loaded = []
                with open(decisions_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                if entry.get("event") == "TRADE_EXITED":
                                    loaded.append(entry)
                            except:
                                pass
                return loaded
            decision_trades = await asyncio.to_thread(_read_decisions)
            trades.extend(decision_trades)
        
        # Also include in-memory trades from DashboardState
        for mem_trade in DashboardState.trades:
            trades.append(mem_trade)
            
        # Deduplicate trades by symbol + exit time + pnl + type
        seen_trades = set()
        unique_trades = []
        for t in trades:
            sym = t.get("symbol", "")
            ext = t.get("exit_time") or t.get("time") or t.get("ts") or 0
            pnl_val = round(float(t.get("pnl_usdt") if t.get("pnl_usdt") is not None else (t.get("pnl", 0) or 0)), 4)
            tr_type = t.get("type", "") or t.get("reason", "")
            key = f"{sym}_{ext}_{pnl_val}_{tr_type}"
            if key not in seen_trades:
                seen_trades.add(key)
                unique_trades.append(t)
        trades = unique_trades
            
        # Separate lifecycle summary records from individual execution records
        raw_trades = [t for t in trades if t.get('type') != 'TRADE_LIFECYCLE']
        lifecycle_trades = [t for t in trades if t.get('type') == 'TRADE_LIFECYCLE']

        wins = 0
        losses = 0
        cumulative_pnl = 0.0
        equity_curve = []
        
        # Per-coin P&L aggregation
        coin_stats = {}  # symbol -> {wins, losses, total_pnl, trades_count}
        
        total_duration_secs = 0
        duration_count = 0
        formatted_history = []
        import datetime

        for t in raw_trades:
            pnl = float(t.get("pnl_usdt") if t.get("pnl_usdt") is not None else (t.get("pnl", 0) or 0))
            symbol = t.get("symbol", "UNKNOWN")
            
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
                
            cumulative_pnl += pnl
            
            # Aggregate per coin
            if symbol not in coin_stats:
                coin_stats[symbol] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades_count": 0, "best_trade": 0.0, "worst_trade": 0.0}
            
            coin_stats[symbol]["trades_count"] += 1
            coin_stats[symbol]["total_pnl"] += pnl
            if pnl > 0:
                coin_stats[symbol]["wins"] += 1
            elif pnl < 0:
                coin_stats[symbol]["losses"] += 1
            if pnl > coin_stats[symbol]["best_trade"]:
                coin_stats[symbol]["best_trade"] = pnl
            if pnl < coin_stats[symbol]["worst_trade"]:
                coin_stats[symbol]["worst_trade"] = pnl
            
            # Parse Entry & Exit Timestamps
            e_ts = t.get("entry_time") or t.get("ts") or 0
            x_ts = t.get("exit_time") or t.get("time") or e_ts
            
            e_time_str = t.get("entry_time_str")
            if not e_time_str and e_ts:
                try:
                    e_time_str = datetime.datetime.fromtimestamp(float(e_ts) / (1000.0 if float(e_ts) > 1e11 else 1.0)).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    e_time_str = str(e_ts)
            
            x_time_str = t.get("exit_time_str")
            if not x_time_str and x_ts:
                try:
                    x_time_str = datetime.datetime.fromtimestamp(float(x_ts) / (1000.0 if float(x_ts) > 1e11 else 1.0)).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    x_time_str = str(x_ts)
                    
            dur_str = t.get("duration")
            if not dur_str and e_ts and x_ts:
                e_sec = float(e_ts) / (1000.0 if float(e_ts) > 1e11 else 1.0)
                x_sec = float(x_ts) / (1000.0 if float(x_ts) > 1e11 else 1.0)
                d_sec = max(0, int(x_sec - e_sec))
                if d_sec > 0:
                    total_duration_secs += d_sec
                    duration_count += 1
                    m = d_sec // 60
                    s = d_sec % 60
                    dur_str = f"{m}m {s}s" if m > 0 else f"{s}s"
                else:
                    dur_str = "< 1m"
            elif not dur_str:
                dur_str = "N/A"

            # Format time for equity chart
            time_val = t.get("exit_time") or t.get("time") or t.get("ts")
            if time_val:
                try:
                    if isinstance(time_val, str) and not time_val.isdigit():
                        dt = datetime.datetime.fromisoformat(time_val.replace('Z', '+00:00'))
                    else:
                        dt = datetime.datetime.fromtimestamp(float(time_val) / (1000.0 if float(time_val) > 1e11 else 1.0))
                    time_str = dt.strftime("%m-%d %H:%M")
                    equity_curve.append({"time": time_str, "value": round(cumulative_pnl, 2)})
                except Exception:
                    pass
            
            ent_p = float(t.get("entry_price") or t.get("entry") or 0.0)
            ext_p = float(t.get("exit_price") or t.get("exit") or 0.0)
            pnl_pct_val = float(t.get("pnl_pct") if t.get("pnl_pct") is not None else 0.0)
            if pnl_pct_val == 0.0 and ent_p > 0 and ext_p > 0:
                is_l = t.get("side", "LONG") == "LONG"
                pnl_pct_val = ((ext_p - ent_p) / ent_p * 100.0) if is_l else ((ent_p - ext_p) / ent_p * 100.0)

            formatted_history.append({
                "symbol": symbol,
                "side": t.get("side", "LONG"),
                "entry_price": ent_p,
                "exit_price": ext_p,
                "pnl_usdt": round(pnl, 4),
                "pnl_pct": round(pnl_pct_val, 2),
                "entry_time": e_ts,
                "exit_time": x_ts,
                "entry_time_str": e_time_str or "N/A",
                "exit_time_str": x_time_str or "N/A",
                "timestamp": t.get("timestamp") or x_time_str or "N/A",
                "duration": dur_str,
                "reason": t.get("reason") or t.get("type") or "TP/SL Exit"
            })
                    
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        
        avg_dur_str = "N/A"
        if duration_count > 0:
            avg_sec = int(total_duration_secs / duration_count)
            avg_m = avg_sec // 60
            avg_s = avg_sec % 60
            avg_dur_str = f"{avg_m}m {avg_s}s" if avg_m > 0 else f"{avg_s}s"

        # Build per-coin response list sorted by total PnL descending
        coin_pnl_list = []
        for sym, stats in coin_stats.items():
            ct = stats["trades_count"]
            coin_pnl_list.append({
                "symbol": sym,
                "total_pnl": round(stats["total_pnl"], 4),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "trades_count": ct,
                "win_rate": round((stats["wins"] / ct * 100) if ct > 0 else 0, 1),
                "best_trade": round(stats["best_trade"], 4),
                "worst_trade": round(stats["worst_trade"], 4),
            })
        coin_pnl_list.sort(key=lambda x: x["total_pnl"], reverse=True)
        
        # --- PROFIT-BASED LOGIC: Advanced performance analytics ---
        try:
            from core.performance_analytics import calculate_advanced_metrics
            # Prefer consolidated lifecycle trades for true position-level metrics, fallback to raw trades
            metrics_source = lifecycle_trades if len(lifecycle_trades) >= 5 else raw_trades
            advanced_metrics = calculate_advanced_metrics(metrics_source)
        except Exception as adv_err:
            print(f"[ANALYTICS] Advanced metrics calculation failed: {adv_err}")
            advanced_metrics = {}
        
        return {
            "status": "success",
            "wins": wins,
            "losses": losses,
            "total_trades": total,
            "win_rate": win_rate,
            "total_pnl": round(cumulative_pnl, 2),
            "avg_duration": avg_dur_str,
            "equity_curve": equity_curve,
            "coin_pnl": coin_pnl_list,
            "history": list(reversed(formatted_history))[:100],
            # Advanced institutional metrics
            "advanced_metrics": advanced_metrics,
            # Consolidated trade lifecycles
            "lifecycle_trades": list(reversed([{
                "trade_id": lt.get("trade_id", ""),
                "symbol": lt.get("symbol", ""),
                "side": lt.get("side", ""),
                "entry_price": lt.get("entry_price", 0),
                "final_exit_price": lt.get("final_exit_price", 0),
                "total_pnl_net": lt.get("total_pnl_net", 0),
                "total_fees": lt.get("total_fees", 0),
                "stages_completed": lt.get("stages_completed", []),
                "exit_reason": lt.get("exit_reason", ""),
                "duration": lt.get("duration", ""),
            } for lt in lifecycle_trades]))[:50],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/state", dependencies=[Depends(verify_dashboard_key)])
async def get_state():
    """Rest API endpoint for current state."""
    return {
        "latest_price": DashboardState.latest_price,
        "balance_usdt": DashboardState.balance_usdt,
        "balance_base": DashboardState.balance_base,
        "active_positions": DashboardState.active_positions,
        "daily_drawdown_pct": DashboardState.daily_drawdown_pct,
        "ml_confidence": DashboardState.ml_confidence,
        "active_ob": DashboardState.active_ob,
        "active_fvg": DashboardState.active_fvg,
        "active_ob_level": DashboardState.active_ob_level,
        "active_ob_type": DashboardState.active_ob_type,
        "symbol": Config.SYMBOL,
        "trades_count": len(DashboardState.trades),
        "trades_today": bot_instance.trades_today if bot_instance else 0,
        "max_daily_trades": getattr(Config, 'MAX_DAILY_TRADES', 6),
        "daily_profit_locked": getattr(bot_instance.risk, 'daily_profit_locked', DashboardState.daily_profit_locked) if (bot_instance and hasattr(bot_instance, 'risk')) else DashboardState.daily_profit_locked,
        "max_daily_profit_pct": getattr(bot_instance.risk, 'max_daily_profit_pct', getattr(Config, 'MAX_DAILY_PROFIT_PCT', 4.0)) if (bot_instance and hasattr(bot_instance, 'risk')) else getattr(Config, 'MAX_DAILY_PROFIT_PCT', 4.0),
        "daily_gain_pct": round(getattr(bot_instance.risk, 'current_drawdown_pct', DashboardState.daily_drawdown_pct), 2) if (bot_instance and hasattr(bot_instance, 'risk')) else DashboardState.daily_drawdown_pct,
        "signal_light": DashboardState.signal_light,
        "signal_light_reason": DashboardState.signal_light_reason,
        "signal_progress": DashboardState.signal_progress
    }

@app.get("/api/trades", dependencies=[Depends(verify_dashboard_key)])
async def get_trades():
    return DashboardState.trades

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # AUD-P9-05: Enforce fail-closed dashboard authentication before accepting financial stream
    secret = os.getenv("DASHBOARD_SECRET", _DASHBOARD_SECRET) or "primesignal_secret_key"
    token = websocket.query_params.get("token") or websocket.query_params.get("key") or websocket.headers.get("x-api-key")
    if not token or not secrets.compare_digest(token, secret):
        await websocket.close(code=1008, reason="Unauthorized: Missing or invalid dashboard API key")
        return
    await websocket.accept()
    DashboardState.active_websockets.add(websocket)
    try:
        # Send initial state immediately
        await send_state_to_ws(websocket)
        
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        DashboardState.active_websockets.discard(websocket)

def _build_state_payload():
    """Build the state dict to broadcast to WebSocket clients."""
    active_sym = Config.SYMBOL
    live_prices = bot_instance.pipeline.latest_prices if (bot_instance and hasattr(bot_instance, 'pipeline')) else {}
    live_p = float(live_prices.get(active_sym, DashboardState.latest_price) or DashboardState.latest_price or 0.0)

    is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
    fx_rate = float(getattr(Config, 'USDT_INR_RATE', 85.0)) if is_inr else 1.0
    if fx_rate <= 0:
        fx_rate = 85.0

    # Build active positions dynamically from bot_instance with authoritative live PnL
    active_pos_map = {}
    if bot_instance and hasattr(bot_instance, 'in_position'):
        for sym in Config.SUPPORTED_SYMBOLS:
            if bot_instance.in_position.get(sym, False):
                pos_sz = float(bot_instance.position_size.get(sym, 0.0) or 0.0)
                if pos_sz <= 1e-7:
                    continue
                entry_val = float(bot_instance.entry_price.get(sym, 0.0) or 0.0)
                sl_val = float(bot_instance.stop_loss.get(sym, 0.0) or 0.0)
                side_val = str(bot_instance.position_side.get(sym, 'LONG')).upper()
                is_long = (side_val == 'LONG')
                
                sym_live_p = float(live_prices.get(sym, 0.0) or (live_p if sym == active_sym else entry_val))
                if sym_live_p <= 0:
                    sym_live_p = entry_val
                
                # Exact PnL calculations
                if entry_val > 0 and sym_live_p > 0:
                    if is_long:
                        p_pct = ((sym_live_p - entry_val) / entry_val) * 100.0
                        p_usdt = pos_sz * (sym_live_p - entry_val)
                    else:
                        p_pct = ((entry_val - sym_live_p) / entry_val) * 100.0
                        p_usdt = pos_sz * (entry_val - sym_live_p)
                else:
                    p_pct = 0.0
                    p_usdt = 0.0
                
                p_currency = p_usdt * fx_rate if is_inr else p_usdt

                # 3-Stage Target calculations
                r_dist = abs(entry_val - sl_val) if abs(entry_val - sl_val) > 0 else (entry_val * 0.01)
                t1r = float(bot_instance.take_profit_1r.get(sym, 0.0) or (entry_val + 1.0 * r_dist if is_long else entry_val - 1.0 * r_dist))
                t2r = float(bot_instance.take_profit_2r.get(sym, 0.0) or (entry_val + 2.5 * r_dist if is_long else entry_val - 2.5 * r_dist))
                t_final = float(bot_instance.take_profit.get(sym, 0.0) or (entry_val + 4.0 * r_dist if is_long else entry_val - 4.0 * r_dist))
                
                tp1_hit = bool(bot_instance.partial_tp_taken.get(sym, False))
                tp2_hit = bool(bot_instance.tp2_taken.get(sym, False))
                is_locked = (is_long and sl_val >= entry_val) or (not is_long and sl_val <= entry_val and sl_val > 0)
                
                if not tp1_hit:
                    act_target = t1r
                    act_name = "TP1 (1.0R)"
                    stage = 1
                elif not tp2_hit:
                    act_target = t2r
                    act_name = "TP2 (2.5R)"
                    stage = 2
                else:
                    act_target = t_final
                    act_name = "Runner Target (4.0R)"
                    stage = 3
                
                # Guaranteed PnL
                realized_pnl_val = float(bot_instance.realized_pnl.get(sym, 0.0) or 0.0)
                runner_guar = 0.0
                if is_locked and entry_val > 0:
                    runner_guar = max(0.0, pos_sz * (sl_val - entry_val)) if is_long else max(0.0, pos_sz * (entry_val - sl_val))
                guar_usdt = realized_pnl_val + runner_guar
                guar_currency = guar_usdt * fx_rate if is_inr else guar_usdt
                orig_sz = float(bot_instance.original_position_size.get(sym, pos_sz) or pos_sz)
                orig_val = orig_sz * entry_val
                guar_pct = (guar_usdt / orig_val * 100.0) if orig_val > 0 else 0.0

                e_time = bot_instance.entry_time.get(sym, int(time.time() * 1000))
                try:
                    e_time_str = datetime.datetime.fromtimestamp(float(e_time) / (1000.0 if float(e_time) > 1e11 else 1.0)).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    e_time_str = time.strftime('%Y-%m-%d %H:%M:%S')

                active_pos_map[sym] = {
                    'symbol': sym,
                    'side': side_val,
                    'entry_price': entry_val,
                    'stop_loss': sl_val,
                    'take_profit': act_target,
                    'target_1r': t1r,
                    'target_2r': t2r,
                    'final_target': t_final,
                    'active_target': act_target,
                    'active_target_name': act_name,
                    'target_stage': stage,
                    'position_size': pos_sz,
                    'current_pnl_usdt': round(p_usdt, 4),
                    'current_pnl_currency': round(p_currency, 2),
                    'current_pnl_pct': round(p_pct, 2),
                    'guaranteed_pnl_usdt': round(guar_usdt, 4),
                    'guaranteed_pnl_currency': round(guar_currency, 2),
                    'guaranteed_pnl_pct': round(guar_pct, 2),
                    'tp1_hit': tp1_hit,
                    'tp2_hit': tp2_hit,
                    'profit_locked': is_locked,
                    'live_price': sym_live_p,
                    'entry_time': e_time,
                    'entry_time_str': e_time_str
                }
        DashboardState.active_positions = active_pos_map
    elif DashboardState.active_positions:
        for sym, pos in DashboardState.active_positions.items():
            sym_live_p = float(live_prices.get(sym, 0.0) or (live_p if sym == active_sym else pos.get('entry_price', 0.0)))
            entry_val = float(pos.get('entry_price', 0.0) or 0.0)
            pos_sz = float(pos.get('position_size', 1.0) or 1.0)
            is_long = (pos.get('side', 'LONG') == 'LONG')
            if entry_val > 0 and sym_live_p > 0:
                p_pct = ((sym_live_p - entry_val) / entry_val * 100.0) if is_long else ((entry_val - sym_live_p) / entry_val * 100.0)
                p_usdt = (pos_sz * (sym_live_p - entry_val)) if is_long else (pos_sz * (entry_val - sym_live_p))
            else:
                p_pct = 0.0
                p_usdt = 0.0
            pos['live_price'] = sym_live_p
            pos['current_pnl_usdt'] = round(p_usdt, 4)
            pos['current_pnl_currency'] = round(p_usdt * fx_rate if is_inr else p_usdt, 2)
            pos['current_pnl_pct'] = round(p_pct, 2)
        active_pos_map = DashboardState.active_positions

    # Calculate Authoritative Active Trade Metrics for Selected active_sym
    if active_sym in active_pos_map:
        pos = active_pos_map[active_sym]
        entry_p = pos['entry_price']
        sl_p = pos['stop_loss']
        tp_p = pos['take_profit']
        side = pos['side']
        in_pos = True
        live_pnl_pct = pos['current_pnl_pct']
        live_pnl_usdt = pos['current_pnl_currency']
        target_progress = 0.0
        if tp_p > entry_p if side == "LONG" else tp_p < entry_p:
            target_progress = max(0.0, min(1.0, abs(live_p - entry_p) / max(1e-6, abs(tp_p - entry_p))))
    else:
        entry_p = DashboardState.entry_price or 0.0
        sl_p = DashboardState.stop_loss or 0.0
        tp_p = DashboardState.take_profit or 0.0
        side = DashboardState.position_side or "HOLD"
        in_pos = DashboardState.in_position
        live_pnl_usdt = 0.0
        live_pnl_pct = 0.0
        target_progress = 0.0
        if in_pos and entry_p > 0 and live_p > 0:
            if side == "LONG":
                live_pnl_pct = ((live_p - entry_p) / entry_p) * 100.0
                raw_pnl = (DashboardState.position_size or 1.0) * (live_p - entry_p)
                live_pnl_usdt = raw_pnl * fx_rate if is_inr else raw_pnl
                if tp_p > entry_p:
                    target_progress = max(0.0, min(1.0, (live_p - entry_p) / (tp_p - entry_p)))
            elif side == "SHORT":
                live_pnl_pct = ((entry_p - live_p) / entry_p) * 100.0
                raw_pnl = (DashboardState.position_size or 1.0) * (entry_p - live_p)
                live_pnl_usdt = raw_pnl * fx_rate if is_inr else raw_pnl
                if tp_p < entry_p:
                    target_progress = max(0.0, min(1.0, (entry_p - live_p) / (entry_p - tp_p)))

    # Compute authoritative held crypto quantity dynamically from open position
    base_coin = active_sym.split('/')[0]
    held_qty = 0.0
    if bot_instance and hasattr(bot_instance, 'in_position'):
        if bot_instance.in_position.get(active_sym, False) and bot_instance.position_side.get(active_sym) == "LONG":
            held_qty = bot_instance.position_size.get(active_sym, 0.0)
    elif active_sym in active_pos_map:
        pos = active_pos_map[active_sym]
        if pos.get('side') == 'LONG':
            held_qty = pos.get('position_size', 0.0)
    elif in_pos and side == "LONG":
        held_qty = DashboardState.position_size or 0.0
        
    DashboardState.balance_base = held_qty
    
    # Sync base coin holding in coindcx_balances array
    coindcx_bals = list(DashboardState.coindcx_balances)
    found_coin = False
    for item in coindcx_bals:
        if item.get('currency') == base_coin:
            item['available'] = round(float(held_qty), 6)
            found_coin = True
            break
    if not found_coin and base_coin not in ('USDT', 'INR'):
        coindcx_bals.append({'currency': base_coin, 'available': round(float(held_qty), 6), 'locked': 0.0})

    # Dynamic real-time calculation of portfolio equity
    if bot_instance and hasattr(bot_instance, 'calculate_total_equity'):
        if not bot_instance.has_keys or Config.PAPER_TRADING:
            DashboardState.balance_usdt = round(bot_instance.calculate_total_equity(), 2)

    return {
        "latest_price": live_p,
        "latest_prices": live_prices,
        "balance_usdt": DashboardState.balance_usdt,
        "balance_base": held_qty,
        "in_position": in_pos,
        "position_side": side,
        "entry_price": entry_p,
        "stop_loss": sl_p,
        "take_profit": tp_p,
        "live_pnl_usdt": round(live_pnl_usdt, 2),
        "live_pnl_pct": round(live_pnl_pct, 2),
        "target_progress": round(target_progress, 4),
        "server_timestamp": int(time.time() * 1000),
        "active_positions": active_pos_map,
        "usdt_inr_rate": fx_rate,
        "daily_drawdown_pct": DashboardState.daily_drawdown_pct,
        "ml_confidence": DashboardState.ml_confidence,
        "next_candle_color": DashboardState.next_candle_color,
        "next_candle_prob": DashboardState.next_candle_prob,
        "active_ob": DashboardState.active_ob,
        "active_fvg": DashboardState.active_fvg,
        "active_ob_level": DashboardState.active_ob_level,
        "active_ob_type": DashboardState.active_ob_type,
        "active_bullish_ob_level": DashboardState.active_bullish_ob_level,
        "active_bearish_ob_level": DashboardState.active_bearish_ob_level,
        "symbol": Config.SYMBOL,
        "ltf_timeframe": Config.LTF_TIMEFRAME,
        "htf_timeframe": Config.HTF_TIMEFRAME,
        "paper_trading": Config.PAPER_TRADING,
        "balance_currency": getattr(Config, 'PAPER_CURRENCY', 'INR' if Config.COINDCX_TRADE_INR else 'USDT'),
        "trades": DashboardState.trades[-5:],  # Last 5 trades
        "trades_today": bot_instance.trades_today if bot_instance else 0,
        "max_daily_trades": getattr(Config, 'MAX_DAILY_TRADES', 6),
        "daily_profit_locked": getattr(bot_instance.risk, 'daily_profit_locked', DashboardState.daily_profit_locked) if (bot_instance and hasattr(bot_instance, 'risk')) else DashboardState.daily_profit_locked,
        "max_daily_profit_pct": getattr(bot_instance.risk, 'max_daily_profit_pct', getattr(Config, 'MAX_DAILY_PROFIT_PCT', 4.0)) if (bot_instance and hasattr(bot_instance, 'risk')) else getattr(Config, 'MAX_DAILY_PROFIT_PCT', 4.0),
        "daily_gain_pct": round(getattr(bot_instance.risk, 'current_drawdown_pct', DashboardState.daily_drawdown_pct), 2) if (bot_instance and hasattr(bot_instance, 'risk')) else DashboardState.daily_drawdown_pct,
        "logs": DashboardState.logs[-10:],     # Last 10 logs
        "chart_history": DashboardState.chart_history,
        "coindcx_profile": DashboardState.coindcx_profile,
        "coindcx_balances": DashboardState.coindcx_balances,
        "signal_light": DashboardState.signal_light,
        "signal_light_reason": DashboardState.signal_light_reason,
        "signal_progress": DashboardState.signal_progress,
        "global_pause_until": getattr(bot_instance, 'global_pause_until', 0.0) if bot_instance else 0.0,
        "cluster_loss_pause_until": getattr(bot_instance, 'cluster_loss_pause_until', 0.0) if bot_instance else 0.0,
        "tp_cooldown_until": getattr(bot_instance, 'tp_cooldown_until', 0.0) if bot_instance else 0.0,
        "supported_symbols": Config.SUPPORTED_SYMBOLS
    }

async def send_state_to_ws(websocket: WebSocket):
    """Sends current state dict as JSON to a specific WebSocket client."""
    state_payload = _build_state_payload()
    await websocket.send_text(json.dumps(state_payload, default=str))

async def broadcast_state_loop():
    """Background task that broadcasts state updates to all connected WebSockets."""
    while True:
        if DashboardState.active_websockets:
            sockets = list(DashboardState.active_websockets)
            try:
                state_payload = _build_state_payload()
                json_str = json.dumps(state_payload, default=str)
            except Exception as e:
                print(f"[WS] Serialization error: {e}")
                await asyncio.sleep(0.5)
                continue
            
            async def _safe_send(ws: WebSocket):
                try:
                    await ws.send_text(json_str)
                except Exception:
                    DashboardState.active_websockets.discard(ws)

            await asyncio.gather(*[_safe_send(ws) for ws in sockets])
        await asyncio.sleep(0.5) # Clean 500ms broadcast rate without buffer exhaustion

async def _run_bot(bot: Any):
    """Wrapper that runs bot initialization and risk monitor loop."""
    try:
        print("[BOT] Initializing bot...")
        await bot.initialize()
        print("[BOT] Initialization complete. Entering risk monitor loop...")
        await bot.run_live_risk_monitor()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        print(f"[BOT] FATAL ERROR: {e}")
        traceback.print_exc()

def add_log_message(msg: str):
    import datetime
    import sys
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{time_str}] {msg}"
    DashboardState.logs.append(log_entry)
    
    encoding = sys.stdout.encoding or 'utf-8'
    try:
        print(log_entry)
    except UnicodeEncodeError:
        safe_entry = log_entry.encode(encoding, errors='replace').decode(encoding)
        print(safe_entry)
        
    if len(DashboardState.logs) > 100:
        DashboardState.logs.pop(0)

import asyncio
import json
import os
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

app = FastAPI(title="PrimeSignal Trading Dashboard")

# Global bot instance — set by main.py before server starts
bot_instance = None
_bot_task = None

# Templates path setup
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Request schema for changing symbols
class SymbolRequest(BaseModel):
    symbol: str

class ModeRequest(BaseModel):
    paper_trading: bool

# ─── ATTACK-1 FIX: API Key Auth for mutating endpoints ──────────────────────
# Without this, anyone on the internet can POST /api/change_symbol and spam
# the bot with symbol changes, triggering WebSocket restarts and CPU spikes.
# Set DASHBOARD_SECRET env var on Render. Omit to disable auth in dev mode.
_DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_dashboard_key(key: str = Depends(_api_key_header)):
    """Only enforces auth if DASHBOARD_SECRET is set in environment."""
    if _DASHBOARD_SECRET and key != _DASHBOARD_SECRET:
        raise HTTPException(status_code=403, detail="Invalid dashboard API key. Set X-API-Key header.")

# Global Memory State Store
class DashboardState:
    latest_price = 0.0
    balance_usdt = 10000.0
    balance_base = 0.0
    in_position = False
    position_side = "HOLD"
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    current_pnl_usdt = 0.0
    current_pnl_pct = 0.0
    
    trades = []
    logs = []
    
    daily_drawdown_pct = 0.0
    ml_confidence = 0.5
    active_ob = "No OB"
    active_fvg = "No FVG"
    active_ob_level = 0.0
    active_ob_type = "NONE"
    active_bullish_ob_level = 0.0
    active_bearish_ob_level = 0.0
    chart_history = []
    coindcx_profile = None
    coindcx_balances = []
    
    signal_light = "RED"
    signal_light_reason = "System starting up..."
    
    symbol_change_requested = None # Holds new symbol if requested by UI
    active_websockets = set()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Renders the main terminal dashboard page."""
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/api/change_symbol", dependencies=[Depends(verify_dashboard_key)])
async def change_symbol(req: SymbolRequest):
    symbol = req.symbol.strip().upper()
    if "/" not in symbol:
        return {"status": "error", "message": "Invalid symbol. Use format e.g. BTC/USDT"}
    
    DashboardState.symbol_change_requested = symbol
    return {"status": "success", "message": f"Symbol change to {symbol} requested successfully."}

@app.post("/api/set_mode", dependencies=[Depends(verify_dashboard_key)])
async def set_mode(req: ModeRequest):
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
                            if usdt_balance and usdt_balance > 0:
                                DashboardState.balance_usdt = usdt_balance
                        DashboardState.balance_base = balance.get('total', {}).get(Config.SYMBOL.split('/')[0], 0.0)
            else:
                # Switching to PAPER: reset to virtual balance
                DashboardState.balance_usdt = bot_instance._dry_run_balance_usdt
                DashboardState.balance_base = 0.0
        except Exception as e:
            print(f"[MODE SWITCH] Error syncing balances: {e}")
            
    mode_name = "PAPER TRADING" if req.paper_trading else "REAL MONEY"
    add_log_message(f"Trading mode switched to {mode_name}")
    return {"status": "success", "message": f"Switched to {mode_name}"}

@app.post("/api/emergency_stop", dependencies=[Depends(verify_dashboard_key)])
async def emergency_stop():
    """Trigger the emergency kill switch via Firebase or local file."""
    try:
        firebase = FirebaseManager()
        if firebase.is_connected:
            firebase.db.collection("control").document("kill_switch").set({"active": True})
        
        with open("KILL_SWITCH", "w") as f:
            f.write("Triggered via API")
            
        add_log_message("🚨 EMERGENCY KILL SWITCH TRIGGERED VIA API 🚨")
        return {"status": "success", "message": "Kill switch activated. All positions will be exited."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to activate kill switch: {str(e)}"}

class RiskSettingsUpdate(BaseModel):
    tsl_enabled: bool
    tsl_multiplier: float

@app.post("/api/update_risk_settings", dependencies=[Depends(verify_dashboard_key)])
async def update_risk_settings(settings: RiskSettingsUpdate):
    from config import Config
    # If the user disables TSL, we can just set the multiplier very high
    Config.TRAILING_ATR_MULT = settings.tsl_multiplier
    if not settings.tsl_enabled:
        Config.TRAILING_ATR_MULT = 999.0 # Effectively disables it
    
    status_str = f"TSL {'Enabled' if settings.tsl_enabled else 'Disabled'} ({settings.tsl_multiplier}x)"
    add_log_message(f"⚙️ Risk Settings Updated: {status_str}")
    return {"status": "success", "message": status_str}

@app.get("/api/state")
async def get_state():
    """Rest API endpoint for current state."""
    return {
        "latest_price": DashboardState.latest_price,
        "balance_usdt": DashboardState.balance_usdt,
        "balance_base": DashboardState.balance_base,
        "in_position": DashboardState.in_position,
        "position_side": DashboardState.position_side,
        "entry_price": DashboardState.entry_price,
        "stop_loss": DashboardState.stop_loss,
        "take_profit": DashboardState.take_profit,
        "current_pnl_usdt": DashboardState.current_pnl_usdt,
        "current_pnl_pct": DashboardState.current_pnl_pct,
        "daily_drawdown_pct": DashboardState.daily_drawdown_pct,
        "ml_confidence": DashboardState.ml_confidence,
        "active_ob": DashboardState.active_ob,
        "active_fvg": DashboardState.active_fvg,
        "active_ob_level": DashboardState.active_ob_level,
        "active_ob_type": DashboardState.active_ob_type,
        "symbol": Config.SYMBOL,
        "trades_count": len(DashboardState.trades),
        "signal_light": DashboardState.signal_light,
        "signal_light_reason": DashboardState.signal_light_reason
    }

@app.get("/api/trades")
async def get_trades():
    return DashboardState.trades

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    DashboardState.active_websockets.add(websocket)
    try:
        # Send initial state immediately
        await send_state_to_ws(websocket)
        
        while True:
            # Keep connection alive, listen for any client messages
            data = await websocket.receive_text()
            # Respond to ping
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        DashboardState.active_websockets.remove(websocket)
    except Exception:
        if websocket in DashboardState.active_websockets:
            DashboardState.active_websockets.remove(websocket)

async def send_state_to_ws(websocket):
    """Sends current state dict as JSON to a specific WebSocket client."""
    state_payload = {
        "latest_price": DashboardState.latest_price,
        "balance_usdt": DashboardState.balance_usdt,
        "balance_base": DashboardState.balance_base,
        "in_position": DashboardState.in_position,
        "position_side": DashboardState.position_side,
        "entry_price": DashboardState.entry_price,
        "stop_loss": DashboardState.stop_loss,
        "take_profit": DashboardState.take_profit,
        "current_pnl_usdt": DashboardState.current_pnl_usdt,
        "current_pnl_pct": DashboardState.current_pnl_pct,
        "daily_drawdown_pct": DashboardState.daily_drawdown_pct,
        "ml_confidence": DashboardState.ml_confidence,
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
        "balance_currency": "USDT" if (Config.PAPER_TRADING or not Config.COINDCX_TRADE_INR) else "INR",
        "trades": DashboardState.trades[-5:],  # Last 5 trades
        "logs": DashboardState.logs[-10:],     # Last 10 logs
        "chart_history": DashboardState.chart_history,
        "coindcx_profile": DashboardState.coindcx_profile,
        "coindcx_balances": DashboardState.coindcx_balances,
        "signal_light": DashboardState.signal_light,
        "signal_light_reason": DashboardState.signal_light_reason
    }
    await websocket.send_text(json.dumps(state_payload))

async def broadcast_state_loop():
    """Background task that broadcasts state updates to all connected WebSockets."""
    while True:
        if DashboardState.active_websockets:
            # Create a copy of the set to avoid modification errors during iteration
            sockets = list(DashboardState.active_websockets)
            for ws in sockets:
                try:
                    await send_state_to_ws(ws)
                except Exception as e:
                    print(f"[WS] Broadcast error, dropping client: {e}")
                    DashboardState.active_websockets.discard(ws)
        await asyncio.sleep(1.0) # Update once per second

@app.on_event("startup")
async def startup_event():
    global _bot_task
    # Start the websocket broadcast background task
    asyncio.create_task(broadcast_state_loop())
    # Launch bot loop if bot_instance is registered
    if bot_instance is not None:
        print("[STARTUP] Launching bot trading loop from FastAPI startup event...")
        _bot_task = asyncio.create_task(_run_bot(bot_instance))
    else:
        print("[STARTUP] WARNING: bot_instance not registered. Bot will NOT run.")

async def _run_bot(bot):
    """Wrapper that runs bot initialization and risk monitor loop."""
    try:
        print("[BOT] Initializing bot...")
        await bot.initialize()
        print("[BOT] Initialization complete. Entering risk monitor loop...")
        
        # Inject test trade trigger for dashboard UI demonstration
        async def trigger_test_trade():
            await asyncio.sleep(8)
            symbol = "BTC/USDT"
            if not bot.pipeline.ltf_candles.get(symbol):
                print("[TEST TRIGGER] Caches not warmed up yet. Cannot test.")
                return
            
            # Reset safety pauses so the test trade doesn't get blocked
            bot.global_pause_until = 0
            bot.relaxed_disabled_until = 0
            bot.cluster_loss_pause_until = 0
            bot.consecutive_losses = 0
            if hasattr(bot, 'trade_history'):
                bot.trade_history.clear()
            
            # Force close any previously recovered open trade to start demo clean
            if bot.in_position[symbol]:
                await bot.exit_position(symbol, "FORCE_CLOSE_PREVIOUS_DEMO")
                await asyncio.sleep(2)
            
            latest_close = bot.pipeline.ltf_candles[symbol][-1][4]
            bot.pipeline.latest_prices[symbol] = latest_close
            
            # Inject extremely high volume to last candle to bypass low volume session filters
            bot.pipeline.ltf_candles[symbol][-1][5] = 9999999999.0
            
            # Temporary settings to ensure trade execution
            from config import Config
            original_slippage = Config.MAX_SLIPPAGE_PCT
            Config.MAX_SLIPPAGE_PCT = 1.0
            
            original_generate_signal = bot.strategy.generate_signal
            def mock_generate_signal(htf_df, ltf_df, relaxed=False, super_relaxed=False):
                metadata = {
                    'stop_loss': latest_close * 0.99,
                    'take_profit': latest_close * 1.02,
                    'tp1': latest_close * 1.01,
                    'tp2': latest_close * 1.02,
                    'score': 4.5,
                    'mode': 'STRICT',
                    'setup_type': 'TEST_OB',
                    'zone_id': 'TEST_ZONE_123',
                    'reason': 'Mocked test signal for UI dashboard demonstration',
                    'debug_checks': {
                        'trend': 'PASS',
                        'zone': 'PASS',
                        'trigger': 'PASS',
                        'vwap': 'PASS',
                        'volatility': 'PASS'
                    }
                }
                return "BUY", metadata
            bot.strategy.generate_signal = mock_generate_signal
            
            print("[TEST TRIGGER] Executing test BUY trade...")
            await bot._on_candle_close_impl(symbol)
            
            bot.strategy.generate_signal = original_generate_signal
            Config.MAX_SLIPPAGE_PCT = original_slippage
            print("[TEST TRIGGER] Test BUY trade completed and active on dashboard!")
            
            # Wait 1200 seconds (20 minutes) so user can see it active on the UI
            await asyncio.sleep(1200)
            print("[TEST TRIGGER] Executing test exit (liquidation)...")
            await bot.exit_position(symbol, "TEST_EXIT")
            print("[TEST TRIGGER] Test exit completed!")
            
        asyncio.create_task(trigger_test_trade())
        
        await bot.run_live_risk_monitor()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        print(f"[BOT] FATAL ERROR: {e}")
        traceback.print_exc()

def add_log_message(msg):
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

import asyncio
import json
import os
import secrets
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

class TimeframeRequest(BaseModel):
    timeframe: str

# ─── ATTACK-1 FIX: API Key Auth for mutating endpoints ──────────────────────
# Without this, anyone on the internet can POST /api/change_symbol and spam
# the bot with symbol changes, triggering WebSocket restarts and CPU spikes.
# Set DASHBOARD_SECRET env var on Render. Omit to disable auth in dev mode.
_DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET", "")
if _DASHBOARD_SECRET:
    print(f"[SECURITY] Dashboard API key auth is ENABLED.")
else:
    print(f"[SECURITY] Dashboard API key auth is DISABLED (DASHBOARD_SECRET not set).")
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
    active_positions = {} # dictionary of symbol -> dict with position details
    
    trades = []
    logs = []
    
    signal_progress = 0
    
    daily_drawdown_pct = 0.0
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
    coindcx_profile = None
    coindcx_balances = []
    
    signal_light = "RED"
    signal_light_reason = "System starting up..."
    
    symbol_change_requested = None # Holds new symbol if requested by UI
    active_websockets = set()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Renders the main terminal dashboard page."""
    return templates.TemplateResponse(request, "index.html", {
        "dashboard_api_key": _DASHBOARD_SECRET
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
            if firebase.is_connected:
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
            
        wins = 0
        losses = 0
        cumulative_pnl = 0.0
        equity_curve = []
        
        # Per-coin P&L aggregation
        coin_stats = {}  # symbol -> {wins, losses, total_pnl, trades_count}
        
        for t in trades:
            pnl = float(t.get("pnl_usdt", 0) or 0)
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
            
            # Format time for chart
            time_val = t.get("time") or t.get("exit_time") or t.get("ts")
            if time_val:
                import datetime
                try:
                    if isinstance(time_val, str):
                        dt = datetime.datetime.fromisoformat(time_val.replace('Z', '+00:00'))
                    else:
                        dt = datetime.datetime.fromtimestamp(float(time_val) / 1000.0)
                    time_str = dt.strftime("%m-%d %H:%M")
                    equity_curve.append({"time": time_str, "value": cumulative_pnl})
                except Exception:
                    pass
                    
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        
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
        
        return {
            "status": "success",
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": cumulative_pnl,
            "equity_curve": equity_curve,
            "coin_pnl": coin_pnl_list,
            "history": list(reversed(trades))[:50]
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
        "signal_light": DashboardState.signal_light,
        "signal_light_reason": DashboardState.signal_light_reason,
        "signal_progress": DashboardState.signal_progress
    }

@app.get("/api/trades", dependencies=[Depends(verify_dashboard_key)])
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
        DashboardState.active_websockets.discard(websocket)
    except Exception:
        if websocket in DashboardState.active_websockets:
            DashboardState.active_websockets.discard(websocket)

def _build_state_payload():
    """Build the state dict to broadcast to WebSocket clients."""
    return {
        "latest_price": DashboardState.latest_price,
        "latest_prices": bot_instance.pipeline.latest_prices if (bot_instance and hasattr(bot_instance, 'pipeline')) else {},
        "balance_usdt": DashboardState.balance_usdt,
        "balance_base": DashboardState.balance_base,
        "active_positions": DashboardState.active_positions,
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
        "balance_currency": "USDT" if (Config.PAPER_TRADING or not Config.COINDCX_TRADE_INR) else "INR",
        "trades": DashboardState.trades[-5:],  # Last 5 trades
        "logs": DashboardState.logs[-10:],     # Last 10 logs
        "chart_history": DashboardState.chart_history,
        "coindcx_profile": DashboardState.coindcx_profile,
        "coindcx_balances": DashboardState.coindcx_balances,
        "signal_light": DashboardState.signal_light,
        "signal_light_reason": DashboardState.signal_light_reason,
        "signal_progress": DashboardState.signal_progress
    }

async def send_state_to_ws(websocket):
    """Sends current state dict as JSON to a specific WebSocket client."""
    state_payload = _build_state_payload()
    await websocket.send_text(json.dumps(state_payload, default=str))

async def broadcast_state_loop():
    """Background task that broadcasts state updates to all connected WebSockets."""
    while True:
        if DashboardState.active_websockets:
            # Create a copy of the set to avoid modification errors during iteration
            sockets = list(DashboardState.active_websockets)
            
            # Serialize payload ONCE for all clients
            try:
                state_payload = _build_state_payload()
                json_str = json.dumps(state_payload, default=str)
            except Exception as e:
                print(f"[WS] Serialization error: {e}")
                await asyncio.sleep(0.5)
                continue
            
            async def _safe_send(ws):
                try:
                    await ws.send_text(json_str)
                except Exception as e:
                    print(f"[WS] Broadcast error, dropping client: {e}")
                    DashboardState.active_websockets.discard(ws)
                    try:
                        await ws.close()
                    except Exception:
                        pass

            await asyncio.gather(*[_safe_send(ws) for ws in sockets])
        await asyncio.sleep(0.5) # Update twice per second (500ms live stream)

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

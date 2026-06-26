import asyncio
import json
import os
from fastapi.testclient import TestClient
from dashboard.app import app, DashboardState
import websockets
import time
import threading
import uvicorn

def run_tests():
    client = TestClient(app)
    results = []

    # 1. Test Dashboard UI Load
    res = client.get("/")
    if res.status_code == 200 and b"<html" in res.content.lower():
        results.append("PASS: UI Load (GET /)")
    else:
        results.append("FAIL: UI Load (GET /)")

    # 2. Test API State
    res = client.get("/api/state")
    if res.status_code == 200:
        data = res.json()
        if "latest_price" in data and "active_positions" in data:
            results.append("PASS: API State (GET /api/state)")
        else:
            results.append("FAIL: API State (GET /api/state) - Missing fields")
    else:
        results.append(f"FAIL: API State (GET /api/state) - Status {res.status_code}")

    # 3. Test API Analytics
    res = client.get("/api/analytics")
    if res.status_code == 200:
        data = res.json()
        if "wins" in data and "win_rate" in data:
            results.append("PASS: API Analytics (GET /api/analytics)")
        else:
            results.append("FAIL: API Analytics (GET /api/analytics) - Missing fields")
    else:
        results.append(f"FAIL: API Analytics (GET /api/analytics) - Status {res.status_code}")

    # 4. Test Emergency Stop
    res = client.post("/api/emergency_stop")
    if res.status_code == 200:
        if os.path.exists("KILL_SWITCH"):
            results.append("PASS: Emergency Stop (POST /api/emergency_stop) - File created")
            os.remove("KILL_SWITCH") # Cleanup
        else:
            results.append("FAIL: Emergency Stop (POST /api/emergency_stop) - File not created")
    else:
         results.append(f"FAIL: Emergency Stop (POST /api/emergency_stop) - Status {res.status_code}")

    # 5. Test Mode Switch
    res = client.post("/api/set_mode", json={"paper_trading": True})
    if res.status_code == 200:
        results.append("PASS: Set Mode (POST /api/set_mode)")
    else:
        results.append(f"FAIL: Set Mode (POST /api/set_mode) - Status {res.status_code}")

    # 6. Test Symbol Change
    res = client.post("/api/change_symbol", json={"symbol": "ETH/USDT"})
    if res.status_code == 200 and DashboardState.symbol_change_requested == "ETH/USDT":
        results.append("PASS: Change Symbol (POST /api/change_symbol)")
    else:
        results.append(f"FAIL: Change Symbol (POST /api/change_symbol) - Status {res.status_code}")

    # 7. Mock Trade Addition to check state
    DashboardState.trades.append({
        "symbol": "BTC/USDT",
        "entry_price": 50000,
        "exit_price": 51000,
        "pnl_usdt": 100,
        "condition": "Test Condition",
        "time": int(time.time() * 1000)
    })
    res = client.get("/api/trades")
    if res.status_code == 200 and len(res.json()) > 0:
        results.append("PASS: Trade Logging & Fetch (GET /api/trades)")
    else:
         results.append("FAIL: Trade Logging & Fetch (GET /api/trades)")

    print("\n--- DASHBOARD TESTING RESULTS ---")
    for r in results:
        print(r)
    print("---------------------------------")

if __name__ == "__main__":
    run_tests()

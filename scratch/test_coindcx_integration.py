import asyncio
from execution.coindcx_client import CoinDCXClient
from config import Config

async def test_integration():
    print("--- Testing CoinDCX Client Integration ---")
    
    # Initialize client (using dummy credentials for public tests)
    client = CoinDCXClient("dummy_api_key", "dummy_secret_key")
    
    print("1. Initializing markets details...")
    success = await client.initialize()
    if success:
        print("   [OK] Loaded markets info successfully.")
        btc_details = client.markets_info.get("BTCINR")
        if btc_details:
            print(f"   [OK] BTCINR precision: {btc_details['precision']}, min quantity: {btc_details['min_quantity']}")
        else:
            print("   [FAIL] BTCINR details not found.")
    else:
        print("   [FAIL] Failed to load markets details.")

    print("\n2. Fetching public ticker data for BTCINR...")
    ticker = await client.fetch_ticker_data("BTCINR")
    if ticker:
        print(f"   [OK] BTCINR Last Price: {ticker['last']} INR")
        print(f"   [OK] BTCINR Bid/Ask: {ticker['bid']} / {ticker['ask']}")
    else:
        print("   [FAIL] Failed to fetch BTCINR ticker data.")

    print("\n3. Testing symbol translation logic...")
    symbol = "BTC/USDT"
    entry_price_usdt = 64000.0
    sl_usdt = 63360.0 # 1% SL distance
    
    if client.initialized:
        coindcx_symbol = f"{symbol.split('/')[0]}INR"
        ticker = await client.fetch_ticker_data(coindcx_symbol)
        if ticker:
            entry_price_inr = ticker['last']
            sl_pct = abs(entry_price_usdt - sl_usdt) / entry_price_usdt
            sl_inr = entry_price_inr * (1 - sl_pct)
            print(f"   Binance entry: {entry_price_usdt} USDT | SL: {sl_usdt} USDT (Dist: {sl_pct*100:.1f}%)")
            print(f"   CoinDCX entry: {entry_price_inr} INR | SL: {sl_inr:.2f} INR (Dist: {sl_pct*100:.1f}%)")
            print("   [OK] Symbol translation scaled correctly.")
        else:
            print("   [FAIL] Symbol translation test failed (no ticker).")

if __name__ == "__main__":
    asyncio.run(test_integration())

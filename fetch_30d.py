import ccxt
import json
import time
from datetime import datetime, timedelta

def fetch_data(symbol, timeframe, days):
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Calculate start time
    start_time = datetime.utcnow() - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)
    end_time = int(datetime.utcnow().timestamp() * 1000)
    
    all_ohlcv = []
    
    print(f"Fetching {symbol} {timeframe} for last {days} days (from {start_time})...")
    
    while since < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
                
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            
            # Print progress safely to avoid too much spam
            if len(all_ohlcv) % 5000 == 0:
                print(f"  Fetched {len(all_ohlcv)} candles...")
                
            time.sleep(0.1) # Respect rate limits
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
            
    # De-duplicate just in case
    unique_dict = {x[0]: x for x in all_ohlcv}
    all_ohlcv = list(unique_dict.values())
    all_ohlcv.sort(key=lambda x: x[0])
    
    print(f"Completed! Total {timeframe} candles: {len(all_ohlcv)}")
    return all_ohlcv

if __name__ == "__main__":
    from config import Config
    
    symbol = Config.SYMBOL
    htf = Config.HTF_TIMEFRAME
    ltf = Config.LTF_TIMEFRAME
    
    # Fetch 30 days of data
    days = 30
    
    htf_data = fetch_data(symbol, htf, days)
    ltf_data = fetch_data(symbol, ltf, days)
    
    with open("htf_data.json", "w") as f:
        json.dump(htf_data, f)
        
    with open("ltf_data.json", "w") as f:
        json.dump(ltf_data, f)
        
    print("Saved to htf_data.json and ltf_data.json")

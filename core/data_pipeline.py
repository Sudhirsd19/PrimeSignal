import asyncio
import json
import websockets
import pandas as pd
from config import Config

class RealTimeDataPipeline:
    def __init__(self, execution_engine):
        self.execution = execution_engine
        
        # In-memory OHLCV caches
        self.ltf_candles = []  # list of [timestamp, open, high, low, close, volume]
        self.htf_candles = []
        
        # Live status
        self.latest_price = 0.0
        self.websocket_active = False
        self.websocket_task = None
        self.current_websocket = None
        
        # Callback for new candle close events
        self.on_candle_close_callback = None

    async def initialize_history(self):
        """
        Warm up caches with historical data from the exchange.
        """
        print("[DATA] Warming up historical candle caches...")
        
        # Fetch HTF history
        htf_ohlcv = await self.execution.fetch_ohlcv(
            symbol=Config.SYMBOL, 
            timeframe=Config.HTF_TIMEFRAME, 
            limit=Config.TREND_EMA + 50
        )
        if htf_ohlcv:
            self.htf_candles = htf_ohlcv
            print(f"  Loaded {len(self.htf_candles)} historical HTF ({Config.HTF_TIMEFRAME}) candles.")
            
        # Fetch LTF history — 500 bars needed:
        # • ML training needs 200+ clean samples after NaN warmup rows (~40 rows) are dropped
        # • SMC OB lookback scans last 50 bars; more history = more structure context
        ltf_ohlcv = await self.execution.fetch_ohlcv(
            symbol=Config.SYMBOL,
            timeframe=Config.LTF_TIMEFRAME,
            limit=500
        )
        if ltf_ohlcv:
            self.ltf_candles = ltf_ohlcv
            print(f"  Loaded {len(self.ltf_candles)} historical LTF ({Config.LTF_TIMEFRAME}) candles.")

    async def start(self):
        """
        Starts the real-time websocket connection to Binance public feed.
        """
        await self.initialize_history()
        
        # Binance stream symbol must be lowercase and without '/'
        stream_symbol = Config.SYMBOL.replace('/', '').lower()
        
        # Stream URLs for both timeframes
        ltf_stream = f"{stream_symbol}@kline_{Config.LTF_TIMEFRAME}"
        htf_stream = f"{stream_symbol}@kline_{Config.HTF_TIMEFRAME}"
        
        url = f"wss://stream.binance.com:9443/ws/{ltf_stream}/{htf_stream}"
        
        self.websocket_task = asyncio.create_task(self._websocket_loop(url))

    async def _websocket_loop(self, url):
        self.websocket_active = True
        print(f"[DATA] Connecting to Binance WebSocket feed: {url}")
        
        retry_delay = 2.0
        while self.websocket_active:
            try:
                async with websockets.connect(url) as websocket:
                    self.current_websocket = websocket
                    print("[DATA] WebSocket Connected successfully!")
                    retry_delay = 2.0  # Reset retry delay
                    
                    async for message in websocket:
                        if not self.websocket_active:
                            break
                        data = json.loads(message)
                        event_type = data.get('e')
                        
                        if event_type == 'kline':
                            kline = data['k']
                            timeframe = kline['i']
                            
                            # Parse kline details
                            candle = [
                                kline['t'],                  # Start time
                                float(kline['o']),           # Open
                                float(kline['h']),           # High
                                float(kline['l']),           # Low
                                float(kline['c']),           # Close
                                float(kline['v'])            # Volume
                            ]
                            is_closed = kline['x']
                            
                            if timeframe == Config.LTF_TIMEFRAME:
                                self.latest_price = candle[4]
                                self._update_candle_cache(self.ltf_candles, candle, is_closed)
                                
                                # If a lower-timeframe candle just closed, trigger strategy evaluation
                                if is_closed and self.on_candle_close_callback:
                                    task = asyncio.create_task(self.on_candle_close_callback())
                                    # Attach error handler to prevent silent failures
                                    task.add_done_callback(lambda t: self._handle_callback_exception(t))
                                    
                            elif timeframe == Config.HTF_TIMEFRAME:
                                self._update_candle_cache(self.htf_candles, candle, is_closed)
                                
            except websockets.exceptions.ConnectionClosed:
                print(f"[DATA] WebSocket disconnected. Reconnecting in {retry_delay:.1f}s...")
            except asyncio.CancelledError:
                print("[DATA] WebSocket loop task cancelled.")
                raise
            except Exception as e:
                print(f"[DATA] WebSocket error: {e}. Reconnecting in {retry_delay:.1f}s...")
                
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60.0)

    def _update_candle_cache(self, cache_list, new_candle, is_closed):
        """
        Updates the candle cache list:
        - If the new candle's timestamp matches the last candle in cache, we update it.
        - If it's a new timestamp, we append it.
        - If is_closed is True, we lock it in. If it is False, we keep it mutable.
        """
        if not cache_list:
            cache_list.append(new_candle)
            return

        # Check if timestamp aligns with last candle
        if new_candle[0] == cache_list[-1][0]:
            # Update current live candle
            cache_list[-1] = new_candle
        else:
            # Append new candle
            cache_list.append(new_candle)
            
        # Keep cache length bounded to prevent memory issues
        if len(cache_list) > 1000:
            cache_list.pop(0)

    def _handle_callback_exception(self, task):
        """Handle exceptions from async callback task to prevent silent failures."""
        if task.cancelled():
            print("[DATA] Candle close callback was cancelled")
        else:
            try:
                task.result()  # This will raise if an exception occurred
            except Exception as e:
                import traceback
                print(f"[ERROR] Exception in on_candle_close callback: {e}")
                traceback.print_exc()

    def stop(self):
        self.websocket_active = False
        if self.current_websocket:
            asyncio.create_task(self.current_websocket.close())
            self.current_websocket = None
        if self.websocket_task:
            self.websocket_task.cancel()
            self.websocket_task = None
        print("[DATA] WebSocket pipeline stopped.")

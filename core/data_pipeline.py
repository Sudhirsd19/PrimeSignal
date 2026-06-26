import asyncio
import json
import websockets
import pandas as pd
from config import Config

class RealTimeDataPipeline:
    def __init__(self, execution_engine):
        self.execution = execution_engine
        
        # In-memory OHLCV caches keyed by symbol
        self.ltf_candles = {sym: [] for sym in Config.SUPPORTED_SYMBOLS}
        self.htf_candles = {sym: [] for sym in Config.SUPPORTED_SYMBOLS}
        self.htf_4h_candles = {sym: [] for sym in Config.SUPPORTED_SYMBOLS}
        
        # Live status
        self.latest_prices = {sym: 0.0 for sym in Config.SUPPORTED_SYMBOLS}
        self.websocket_active = False
        self.websocket_task = None
        self.current_websocket = None
        
        # Callback for new candle close events
        self.on_candle_close_callback = None

        # Feature 6: Track last candle time per symbol/timeframe for gap healing
        self._last_candle_ts = {
            'ltf': {sym: 0 for sym in Config.SUPPORTED_SYMBOLS},
            'htf': {sym: 0 for sym in Config.SUPPORTED_SYMBOLS},
            '4h':  {sym: 0 for sym in Config.SUPPORTED_SYMBOLS},
        }

    async def initialize_history(self):
        """
        Warm up caches with historical data from the exchange.
        """
        print("[DATA] Warming up historical candle caches for all symbols...")
        for symbol in Config.SUPPORTED_SYMBOLS:
            # Fetch HTF history
            htf_ohlcv = await self.execution.fetch_ohlcv(
                symbol=symbol, 
                timeframe=Config.HTF_TIMEFRAME, 
                limit=Config.TREND_EMA + 50
            )
            if htf_ohlcv is not None:
                self.htf_candles[symbol] = htf_ohlcv
            else:
                print(f"ERROR: Failed to fetch historical data (HTF) for {symbol}")
                
            # Fetch 4H history
            htf_4h_ohlcv = await self.execution.fetch_ohlcv(
                symbol=symbol, 
                timeframe='4h', 
                limit=50
            )
            if htf_4h_ohlcv is not None:
                self.htf_4h_candles[symbol] = htf_4h_ohlcv
            else:
                print(f"ERROR: Failed to fetch historical data (4H) for {symbol}")
                
            # Fetch LTF history — 500 bars needed:
            # • ML training needs 200+ clean samples after NaN warmup rows (~40 rows) are dropped
            # • SMC OB lookback scans last 50 bars; more history = more structure context
            ltf_ohlcv = await self.execution.fetch_ohlcv(
                symbol=symbol,
                timeframe=Config.LTF_TIMEFRAME,
                limit=500
            )
            if ltf_ohlcv is not None:
                self.ltf_candles[symbol] = ltf_ohlcv
            else:
                print(f"ERROR: Failed to fetch historical data (LTF) for {symbol}")
        print("[DATA] Historical caches warmed up.")

    async def start(self):
        """
        Starts the real-time websocket connection to Binance public feed.
        """
        await self.initialize_history()
        
        streams = []
        for symbol in Config.SUPPORTED_SYMBOLS:
            stream_symbol = symbol.replace('/', '').lower()
            streams.append(f"{stream_symbol}@kline_{Config.LTF_TIMEFRAME}")
            streams.append(f"{stream_symbol}@kline_{Config.HTF_TIMEFRAME}")
            streams.append(f"{stream_symbol}@kline_4h")
            
        streams_joined = '/'.join(streams)
        url = f"wss://stream.binance.com:9443/stream?streams={streams_joined}"
        
        self.websocket_task = asyncio.create_task(self._websocket_loop(url))

    async def _websocket_loop(self, url):
        self.websocket_active = True
        print(f"[DATA] Connecting to Binance WebSocket feed...")
        
        retry_delay = 2.0
        while self.websocket_active:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as websocket:
                    self.current_websocket = websocket
                    print("[DATA] WebSocket Connected successfully!")
                    retry_delay = 2.0  # Reset retry delay

                    # Feature 6: Heal any gaps from disconnection
                    await self._heal_candle_gaps()
                    
                    async for message in websocket:
                        if not self.websocket_active:
                            break
                        data = json.loads(message)
                        
                        if 'data' in data:
                            kline_data = data['data']
                        else:
                            # fallback for single stream connection
                            kline_data = data

                        event_type = kline_data.get('e')
                        
                        if event_type == 'kline':
                            kline = kline_data['k']
                            timeframe = kline['i']
                            symbol_raw = kline['s'] # e.g. 'BTCUSDT'
                            
                            # Map back to SUPPORTED_SYMBOLS
                            symbol = next((s for s in Config.SUPPORTED_SYMBOLS if s.replace('/', '') == symbol_raw), None)
                            if not symbol:
                                continue
                            
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
                                self.latest_prices[symbol] = candle[4]
                                self._update_candle_cache(self.ltf_candles[symbol], candle, is_closed)
                                if is_closed:
                                    self._last_candle_ts['ltf'][symbol] = candle[0]
                                
                                # If a lower-timeframe candle just closed, trigger strategy evaluation
                                if is_closed and self.on_candle_close_callback:
                                    task = asyncio.create_task(self.on_candle_close_callback(symbol))
                                    # Attach error handler to prevent silent failures
                                    task.add_done_callback(lambda t: self._handle_callback_exception(t))
                                    
                            elif timeframe == Config.HTF_TIMEFRAME:
                                self._update_candle_cache(self.htf_candles[symbol], candle, is_closed)
                                if is_closed:
                                    self._last_candle_ts['htf'][symbol] = candle[0]
                            elif timeframe == '4h':
                                self._update_candle_cache(self.htf_4h_candles[symbol], candle, is_closed)
                                if is_closed:
                                    self._last_candle_ts['4h'][symbol] = candle[0]
                                
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

    # ── Feature 6: WebSocket reconnection gap healing ────────────────────
    async def _heal_candle_gaps(self):
        """Re-fetches missing candles after a WebSocket disconnect."""
        healed = 0
        for symbol in Config.SUPPORTED_SYMBOLS:
            # Heal LTF gaps
            last_ts = self._last_candle_ts['ltf'].get(symbol, 0)
            if last_ts > 0 and self.ltf_candles[symbol]:
                fetched = await self._fetch_since(symbol, Config.LTF_TIMEFRAME, last_ts)
                if fetched:
                    merged = self._merge_candles(self.ltf_candles[symbol], fetched)
                    self.ltf_candles[symbol] = merged
                    healed += len(fetched)

            # Heal HTF gaps
            last_ts = self._last_candle_ts['htf'].get(symbol, 0)
            if last_ts > 0 and self.htf_candles[symbol]:
                fetched = await self._fetch_since(symbol, Config.HTF_TIMEFRAME, last_ts)
                if fetched:
                    merged = self._merge_candles(self.htf_candles[symbol], fetched)
                    self.htf_candles[symbol] = merged
                    healed += len(fetched)

            # Heal 4H gaps
            last_ts = self._last_candle_ts['4h'].get(symbol, 0)
            if last_ts > 0 and self.htf_4h_candles[symbol]:
                fetched = await self._fetch_since(symbol, '4h', last_ts)
                if fetched:
                    merged = self._merge_candles(self.htf_4h_candles[symbol], fetched)
                    self.htf_4h_candles[symbol] = merged
                    healed += len(fetched)

        if healed > 0:
            print(f"[DATA] Gap healing complete: {healed} candles recovered across all symbols.")

    async def _fetch_since(self, symbol, timeframe, since_ts):
        """Fetches candles from exchange since a given timestamp."""
        try:
            ohlcv = await self.execution.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, limit=200
            )
            if ohlcv:
                # Only return candles after the last known timestamp
                return [c for c in ohlcv if c[0] > since_ts]
        except Exception as e:
            print(f"[DATA] Gap heal fetch failed for {symbol} {timeframe}: {e}")
        return []

    @staticmethod
    def _merge_candles(existing: list, new_candles: list) -> list:
        """Merges new candles into existing cache, deduplicating by timestamp."""
        existing_ts = {c[0] for c in existing}
        merged = list(existing)
        for c in new_candles:
            if c[0] not in existing_ts:
                merged.append(c)
        merged.sort(key=lambda c: c[0])
        # Keep bounded
        if len(merged) > 1000:
            merged = merged[-1000:]
        return merged

import asyncio
import json
import time
import websockets
import pandas as pd
from config import Config

def parse_timeframe_to_minutes(tf_str: str) -> int:
    """Safely converts timeframe string (e.g. '1m', '5m', '15m', '1h', '4h', '1d') to minutes."""
    if not tf_str:
        return 15
    tf = tf_str.lower().strip()
    if tf.endswith('m'):
        return int(tf[:-1])
    elif tf.endswith('h'):
        return int(tf[:-1]) * 60
    elif tf.endswith('d'):
        return int(tf[:-1]) * 1440
    try:
        return int(tf)
    except ValueError:
        return 15

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
                
            # Fetch LTF history.
            # Deep history on low timeframes (1m/5m) ensures 10,000+ bars for robust ML training.
            # Persistent disk caching ensures instant startup by loading cached history and fetching only delta.
            target_bars = self.get_target_ltf_bars(Config.LTF_TIMEFRAME)
            ltf_ohlcv = await self._fetch_ohlcv_paged(
                symbol=symbol,
                timeframe=Config.LTF_TIMEFRAME,
                total_bars=target_bars,
            )
            if ltf_ohlcv:
                self.ltf_candles[symbol] = ltf_ohlcv
                print(f"[DATA] {symbol}: {len(ltf_ohlcv)} LTF bars warmed up ({Config.LTF_TIMEFRAME}).")
            else:
                print(f"ERROR: Failed to fetch historical data (LTF) for {symbol}")
        print("[DATA] Historical caches warmed up.")

    @staticmethod
    def get_target_ltf_bars(timeframe: str) -> int:
        """Determines target historical candle count based on timeframe depth requirements."""
        tf = (timeframe or '').strip().lower()
        if tf in ('1m', '5m'):
            return int(getattr(Config, 'LTF_HISTORY_BARS_DEEP', 10000))
        return int(getattr(Config, 'LTF_HISTORY_BARS', 2000))

    async def _fetch_ohlcv_paged(self, symbol: str, timeframe: str, total_bars: int) -> list:
        """Fetches up to `total_bars` of recent history, paginating forward with persistent disk caching.

        Loads cached candles from data/candles_cache_{clean_symbol}_{timeframe}.json, fetches only
        the delta since the newest cached candle, and saves the updated history back to disk.
        """
        if total_bars <= 0:
            return []
        try:
            tf_mins = parse_timeframe_to_minutes(timeframe)
        except Exception:
            tf_mins = 15
        tf_ms = tf_mins * 60 * 1000

        from pathlib import Path
        clean_sym = symbol.replace('/', '_').lower()
        cache_path = Path("data") / f"candles_cache_{clean_sym}_{timeframe.lower()}.json"

        collected: dict[int, list] = {}

        # 1. Load existing disk cache if present
        try:
            if cache_path.exists():
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached_data, list):
                    for candle in cached_data:
                        if isinstance(candle, (list, tuple)) and len(candle) >= 6 and candle[0] is not None:
                            collected[int(candle[0])] = list(candle)
        except Exception as e:
            print(f"[DATA] Note: could not load candle cache for {symbol} ({e})")

        now_ms = int(time.time() * 1000)
        page_limit = min(1000, total_bars)

        if collected:
            latest_cached_ts = max(collected.keys())
            since = latest_cached_ts + tf_ms
        else:
            since = now_ms - (total_bars * tf_ms)

        if since < now_ms:
            max_requests = max(2, (total_bars // page_limit) + 3)
            for _ in range(max_requests):
                try:
                    batch = await self.execution.fetch_ohlcv(
                        symbol=symbol, timeframe=timeframe, limit=page_limit, since=since
                    )
                except TypeError:
                    batch = await self.execution.fetch_ohlcv(
                        symbol=symbol, timeframe=timeframe, limit=page_limit
                    )
                except Exception as exc:
                    print(f"[DATA] Paged fetch error for {symbol} {timeframe}: {exc}")
                    break

                if not batch:
                    break

                for candle in batch:
                    if candle and candle[0] is not None:
                        collected[int(candle[0])] = list(candle)

                newest_ts = int(batch[-1][0])
                next_since = newest_ts + tf_ms
                if len(collected) >= total_bars or next_since >= now_ms:
                    break
                if next_since <= since:
                    break
                since = next_since

        ordered = [collected[ts] for ts in sorted(collected.keys())]
        result = ordered[-total_bars:]

        # 2. Persist updated candles back to disk cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result), encoding="utf-8")
        except Exception as e:
            print(f"[DATA] Note: could not save candle cache for {symbol} ({e})")

        return result

    async def start(self):
        """
        Starts the real-time websocket connection to Binance public feed.
        """
        await self.initialize_history()
        
        streams = []
        for symbol in Config.SUPPORTED_SYMBOLS:
            stream_symbol = symbol.replace('/', '').lower()
            streams.append(f"{stream_symbol}@miniTicker")
            streams.append(f"{stream_symbol}@kline_{Config.LTF_TIMEFRAME}")
            streams.append(f"{stream_symbol}@kline_{Config.HTF_TIMEFRAME}")
            streams.append(f"{stream_symbol}@kline_4h")
            
        streams_joined = '/'.join(streams)
        url = f"wss://stream.binance.com:9443/stream?streams={streams_joined}"
        
        self.websocket_task = asyncio.create_task(self._websocket_loop(url))

    async def refresh_ltf_history(self):
        """Concurrent warmup of LTF candle caches for all symbols using asyncio.gather.
        Guarantees all-or-nothing atomicity: self.ltf_candles is ONLY updated if ALL symbols
        are successfully fetched with adequate candle history. If any symbol fails, raises
        RuntimeError to trigger clean rollback in change_execution_timeframe().
        """
        print(f"[DATA] Refreshing LTF candle caches for all symbols on {Config.LTF_TIMEFRAME}...")
        target_bars = self.get_target_ltf_bars(Config.LTF_TIMEFRAME)
        new_candles = {}
        failed_symbols = []
        lock = asyncio.Lock()

        async def _fetch_one(symbol):
            try:
                ltf_ohlcv = await self._fetch_ohlcv_paged(
                    symbol=symbol,
                    timeframe=Config.LTF_TIMEFRAME,
                    total_bars=target_bars,
                )
                if ltf_ohlcv and len(ltf_ohlcv) >= 10:
                    async with lock:
                        new_candles[symbol] = ltf_ohlcv
                    print(f"[DATA] {symbol}: {len(ltf_ohlcv)} LTF bars warmed up ({Config.LTF_TIMEFRAME}).")
                else:
                    bars_count = len(ltf_ohlcv) if ltf_ohlcv else 0
                    async with lock:
                        failed_symbols.append(f"{symbol} (insufficient bars: {bars_count})")
                    print(f"[DATA] Insufficient LTF bars for {symbol} ({bars_count} < 10) on {Config.LTF_TIMEFRAME}.")
            except Exception as e:
                async with lock:
                    failed_symbols.append(f"{symbol} ({e})")
                print(f"[DATA] Error refreshing {symbol} {Config.LTF_TIMEFRAME}: {e}")

        await asyncio.gather(*[_fetch_one(s) for s in Config.SUPPORTED_SYMBOLS])

        if failed_symbols:
            err_msg = f"Failed to refresh LTF history atomically for: {', '.join(failed_symbols)}"
            print(f"[DATA] ❌ {err_msg}. Aborting cache update to prevent mixed-cache state.")
            raise RuntimeError(err_msg)

        # Apply atomically to all symbols
        for sym, candles in new_candles.items():
            self.ltf_candles[sym] = candles

        print(f"[DATA] Historical LTF caches atomically refreshed for {Config.LTF_TIMEFRAME}.")

    async def restart_streams(self):
        """Restarts the websocket connection with updated LTF streams."""
        if self.websocket_task and not self.websocket_task.done():
            self.websocket_active = False
            if self.current_websocket:
                try:
                    await self.current_websocket.close()
                except Exception:
                    pass
            self.websocket_task.cancel()
            await asyncio.sleep(0.2)

        # Refresh only LTF candles concurrently across all symbols
        await self.refresh_ltf_history()

        streams = []
        for symbol in Config.SUPPORTED_SYMBOLS:
            stream_symbol = symbol.replace('/', '').lower()
            streams.append(f"{stream_symbol}@miniTicker")
            streams.append(f"{stream_symbol}@kline_{Config.LTF_TIMEFRAME}")
            streams.append(f"{stream_symbol}@kline_{Config.HTF_TIMEFRAME}")
            streams.append(f"{stream_symbol}@kline_4h")
            
        streams_joined = '/'.join(streams)
        url = f"wss://stream.binance.com:9443/stream?streams={streams_joined}"
        self.websocket_task = asyncio.create_task(self._websocket_loop(url))

    async def _websocket_loop(self, url):
        self.websocket_active = True
        print(f"[DATA] Connecting to Binance WebSocket feed (Klines + Real-time MiniTicker)...")
        
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
                        
                        # Handle ultra-fast real-time MiniTicker ticks
                        if event_type in ('24hrMiniTicker', 'miniTicker'):
                            symbol_raw = kline_data.get('s')
                            symbol = next((s for s in Config.SUPPORTED_SYMBOLS if s.replace('/', '') == symbol_raw), None)
                            if symbol:
                                live_c = float(kline_data.get('c', 0.0))
                                if live_c > 0:
                                    self.latest_prices[symbol] = live_c
                                    if symbol == Config.SYMBOL:
                                        from dashboard.app import DashboardState
                                        DashboardState.latest_price = live_c
                            continue
                        
                        if event_type == 'kline':
                            kline = kline_data['k']
                            timeframe = kline['i']
                            is_closed = bool(kline.get('x', False))
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
                            # Always update latest_prices for any incoming kline event
                            self.latest_prices[symbol] = candle[4]
                            if symbol == Config.SYMBOL:
                                from dashboard.app import DashboardState
                                DashboardState.latest_price = candle[4]
                            
                            if timeframe == Config.LTF_TIMEFRAME:
                                # Exact Timestamp-based Gap Detection & Serialized Ingestion (P1 Invariant)
                                last_ts = self._last_candle_ts['ltf'].get(symbol, 0)
                                tf_mins = parse_timeframe_to_minutes(Config.LTF_TIMEFRAME)
                                tf_ms = tf_mins * 60 * 1000
                                if last_ts > 0 and (candle[0] - last_ts) > (tf_ms * 1.5):
                                    print(f"[DATA] ⚠️ Sequence gap detected on {symbol}! (Expected: {last_ts + tf_ms}, Got: {candle[0]}). Serializing backfill...")
                                    try:
                                        backfilled = await self.execution.fetch_ohlcv(symbol=symbol, timeframe=Config.LTF_TIMEFRAME, limit=100)
                                        if backfilled:
                                            new_candles = [c for c in backfilled if c[0] > last_ts]
                                            self.ltf_candles[symbol].extend(new_candles)
                                            max_bars = self.get_target_ltf_bars(Config.LTF_TIMEFRAME)
                                            self.ltf_candles[symbol] = self.ltf_candles[symbol][-max_bars:]
                                    except Exception as e:
                                        print(f"[DATA] ⛔ Backfill failed on {symbol}: {e}. Skipping this candle to maintain continuous stream.")
                                        continue
                                
                                self._update_candle_cache(self.ltf_candles[symbol], candle, is_closed)
                                if is_closed:
                                    self._last_candle_ts['ltf'][symbol] = int(candle[0])
                                
                                # If a lower-timeframe candle just closed, trigger strategy evaluation
                                if is_closed and self.on_candle_close_callback:
                                    task = asyncio.create_task(self.on_candle_close_callback(symbol))
                                    # Attach error handler to prevent silent failures
                                    task.add_done_callback(lambda t: self._handle_callback_exception(t))
                                    
                            elif timeframe == Config.HTF_TIMEFRAME:
                                self._update_candle_cache(self.htf_candles[symbol], candle, is_closed)
                                if is_closed:
                                    self._last_candle_ts['htf'][symbol] = int(candle[0])
                            elif timeframe == '4h':
                                self._update_candle_cache(self.htf_4h_candles[symbol], candle, is_closed)
                                if is_closed:
                                    self._last_candle_ts['4h'][symbol] = int(candle[0])
                                
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
        elif new_candle[0] > cache_list[-1][0]:
            # Append new candle (normal case — newer timestamp)
            cache_list.append(new_candle)
        else:
            # Out-of-order candle — insert at correct chronological position
            for i in range(len(cache_list) - 1, -1, -1):
                if cache_list[i][0] == new_candle[0]:
                    cache_list[i] = new_candle
                    break
                elif cache_list[i][0] < new_candle[0]:
                    cache_list.insert(i + 1, new_candle)
                    break
            else:
                # Candle is older than all entries — prepend it
                cache_list.insert(0, new_candle)
            
        # Keep cache length bounded to prevent memory issues
        max_cache_len = max(1000, self.get_target_ltf_bars(getattr(Config, 'LTF_TIMEFRAME', '15m')))
        if len(cache_list) > max_cache_len:
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
        max_bars = max(1000, RealTimeDataPipeline.get_target_ltf_bars(getattr(Config, 'LTF_TIMEFRAME', '15m')))
        if len(merged) > max_bars:
            merged = merged[-max_bars:]
        return merged

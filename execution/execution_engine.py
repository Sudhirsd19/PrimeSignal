import ccxt.async_support as ccxt
import asyncio
import inspect
import time
from typing import Any, cast, Optional
from config import Config
from execution.execution_result import (
    ExecutionIntentJournal,
    ExecutionResult,
    ExecutionState,
    coerce_execution_result,
    new_intent_id,
)


class ExecutionEngine:
    def __init__(self, intent_journal_path=None):
        is_futures = Config.EXCHANGE_TYPE == 'futures'

        # 1. Public client (for data — always spot for Binance public streams)
        self.public_client = ccxt.binance()

        # 2. Trading client (for private actions)
        options: dict[str, Any] = {}
        if Config.API_KEY and Config.API_KEY != "your_api_key_here":
            options['apiKey'] = Config.API_KEY
        if Config.SECRET_KEY and Config.SECRET_KEY != "your_api_secret_here":
            options['secret'] = Config.SECRET_KEY

        # Feature 2: Futures support — set defaultType to 'future' for USDT-M
        if is_futures:
            options['options'] = {'defaultType': 'future'}
            self.trade_client = ccxt.binance(cast(Any, options))
            print(f"[EXECUTION] Binance USDT-M Futures mode enabled (Leverage: {Config.FUTURES_LEVERAGE}x, Margin: {Config.FUTURES_MARGIN_MODE})")
        else:
            self.trade_client = ccxt.binance(cast(Any, options))

        # 3. CoinDCX Integration (spot only — CoinDCX has no standard futures API)
        from execution.coindcx_client import CoinDCXClient
        self.coindcx_client: CoinDCXClient | None = None
        if getattr(Config, 'TRADING_VENUE', 'BINANCE') == 'COINDCX':
            if is_futures:
                raise ValueError("[EXECUTION CRITICAL] CoinDCX venue selected but EXCHANGE_TYPE='futures'. CoinDCX does not support futures via API.")
            if Config.COINDCX_API_KEY and Config.COINDCX_API_KEY != "your_coindcx_key_here":
                self.coindcx_client = CoinDCXClient(Config.COINDCX_API_KEY, Config.COINDCX_SECRET_KEY)
                print("[EXECUTION] Explicit CoinDCX venue active. Client integrated successfully.")
            elif not Config.PAPER_TRADING:
                raise ValueError("[EXECUTION CRITICAL] TRADING_VENUE is set to COINDCX for live trading, but valid credentials were not found.")
        else:
            print("[EXECUTION] Explicit Binance venue active.")

        if Config.USE_TESTNET:
            self.trade_client.set_sandbox_mode(True)
            print("[EXECUTION] Sandbox mode enabled (Binance Testnet)")
        else:
            print("[EXECUTION] WARNING: Live mainnet enabled. Operating with real funds.")

        self._tickers_cache = {}
        self._tickers_cache_time = 0.0
        self._futures_initialized = False
        self.intent_journal = ExecutionIntentJournal(intent_journal_path)
        if self.coindcx_client:
            self.coindcx_client.intent_journal = cast(Any, self.intent_journal)

    def prepare_order_intent(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        order_role: str = "ENTRY",
        intent_id: str | None = None,
        client_order_id: str | None = None,
        is_exit_order: bool = False,
    ) -> tuple[str, str]:
        """Create and durably record one economic intent before submission."""
        intent_id = intent_id or new_intent_id()
        client_order_id = client_order_id or f"PS_{intent_id[:24].upper()}"
        venue = "COINDCX" if self.coindcx_client else "BINANCE"
        account_mode = "futures" if Config.EXCHANGE_TYPE == "futures" else "spot"
        self.intent_journal.create(
            intent_id=intent_id,
            client_order_id=client_order_id,
            venue=venue,
            account_mode=account_mode,
            symbol=symbol,
            side=side.lower(),
            requested_qty=amount,
            order_role=order_role,
            price=price,
        )
        return intent_id, client_order_id

    def unresolved_intents(self) -> list[dict[str, Any]]:
        """Return durable intents whose exchange outcome still needs resolution."""
        return self.intent_journal.unresolved()

    async def _init_futures(self, symbol=None):
        """One-time setup for futures: load markets, set leverage and margin mode."""
        if self._futures_initialized or Config.EXCHANGE_TYPE != 'futures':
            return
        sym = symbol or Config.SYMBOL
        try:
            await self.trade_client.load_markets()
            # Set margin mode (isolated / cross)
            try:
                await self.trade_client.set_margin_mode(Config.FUTURES_MARGIN_MODE, sym)
                print(f"[FUTURES] Margin mode set to {Config.FUTURES_MARGIN_MODE.upper()} for {sym}")
            except Exception as e:
                err_msg = str(e).lower()
                # Binance returns -4046 / "No need to change margin type" if already in desired mode
                if "no need to change margin type" in err_msg or "-4046" in err_msg:
                    print(f"[FUTURES] Margin mode already set to {Config.FUTURES_MARGIN_MODE.upper()} for {sym}")
                else:
                    raise RuntimeError(f"Failed to set futures margin mode to {Config.FUTURES_MARGIN_MODE} on {sym}: {e}")

            # Set leverage
            try:
                await self.trade_client.set_leverage(Config.FUTURES_LEVERAGE, sym)
                print(f"[FUTURES] Leverage set to {Config.FUTURES_LEVERAGE}x for {sym}")
            except Exception as e:
                raise RuntimeError(f"Failed to set futures leverage to {Config.FUTURES_LEVERAGE}x on {sym}: {e}")

            self._futures_initialized = True
        except Exception as e:
            self._futures_initialized = False
            print(f"[FUTURES CRITICAL] Error initializing futures settings: {e}")
            raise

    async def close(self):
        await self.public_client.close()
        await self.trade_client.close()
        if self.coindcx_client:
            await self.coindcx_client.close()

    async def fetch_balance(self):
        """Fetch balances with automatic retry."""
        if self.coindcx_client:
            return await self.coindcx_client.fetch_balance()
        return await self.execute_with_retry(self.trade_client.fetch_balance)

    async def fetch_current_price(self, symbol=None):
        """Fetch last price from public ticker."""
        if symbol is None:
            symbol = Config.SYMBOL
        ticker = await self.execute_with_retry(self.public_client.fetch_ticker, symbol)
        if ticker:
            return ticker['last']
        return None

    async def fetch_ticker_data(self, symbol=None):
        """Fetch full ticker data including bid, ask, and quoteVolume."""
        if symbol is None:
            symbol = Config.SYMBOL
        return await self.execute_with_retry(self.public_client.fetch_ticker, symbol)

    async def fetch_all_tickers(self):
        """Fetch all tickers with a 60-second cache to find top volume symbols."""
        now = time.time()
        if now - self._tickers_cache_time > 60 or not self._tickers_cache:
            tickers = await self.execute_with_retry(self.public_client.fetch_tickers)
            if tickers:
                self._tickers_cache = tickers
                self._tickers_cache_time = now
        return self._tickers_cache

    async def fetch_ohlcv(self, symbol=None, timeframe=None, limit=100):
        """Fetch historical candlestick data (OHLCV) from public client."""
        if symbol is None:
            symbol = Config.SYMBOL
        if timeframe is None:
            timeframe = Config.LTF_TIMEFRAME
        return await self.execute_with_retry(self.public_client.fetch_ohlcv, symbol, timeframe, None, limit)

    async def fetch_funding_rate(self, symbol=None):
        """
        Fetch real-time 8h funding rate for a symbol (e.g. 0.0001 = 0.01%).
        Uses 60-second cache to minimize network calls.
        """
        if symbol is None:
            symbol = Config.SYMBOL
        
        if not hasattr(self, '_funding_rates_cache'):
            self._funding_rates_cache = {}
            self._funding_rates_cache_time = {}

        now = time.time()
        if symbol in self._funding_rates_cache and (now - self._funding_rates_cache_time.get(symbol, 0)) < 60:
            return self._funding_rates_cache[symbol]

        try:
            import aiohttp
            raw_symbol = symbol.replace('/', '').upper()
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={raw_symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=4.0)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        last_funding_rate = float(data.get('lastFundingRate', 0.0))
                        self._funding_rates_cache[symbol] = last_funding_rate
                        self._funding_rates_cache_time[symbol] = now
                        return last_funding_rate
                    else:
                        print(f"[EXECUTION WARNING] Binance funding rate API returned status {resp.status}")
                        return None
        except Exception as e:
            print(f"[EXECUTION WARNING] Error fetching funding rate: {e}")
            return None
            
        return None

    # ── Feature 1: Order fill confirmation ───────────────────────────────
    async def wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: float = 30.0,
        requested_qty: float = 0.0,
        client_order_id: str | None = None,
        intent_id: str | None = None,
    ) -> ExecutionResult:
        """Poll authoritative status without converting timeout to a fill."""
        start = time.time()
        poll_interval = 0.5  # start fast, then slow down
        while time.time() - start < timeout:
            try:
                order = await self.trade_client.fetch_order(order_id, symbol)
                result = ExecutionResult.from_exchange(
                    order,
                    requested_qty=requested_qty,
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue="BINANCE",
                )
                if result.is_fill_confirmed:
                    elapsed = int((time.time() - start) * 1000)
                    print(f"[FILL] Order {order_id} {result.state.value} in {elapsed}ms. Avg price: {result.average_fill_price}")
                    return result
                if result.state in (ExecutionState.CANCELLED, ExecutionState.REJECTED):
                    print(f"[FILL] Order {order_id} ended with status: {result.state.value}")
                    return result
            except Exception as e:
                print(f"[FILL] Error polling order {order_id}: {e}")
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 3.0)  # gradual backoff

        print(f"[FILL] Order {order_id} TIMED OUT after {timeout}s — status unknown")
        return ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            requested_qty=requested_qty or 0.0,
            exchange_order_id=order_id,
            client_order_id=client_order_id,
            intent_id=intent_id,
            venue="BINANCE",
            error="fill polling timed out or status remained unavailable",
        )

    async def place_order(self, side, order_type, amount, price=None,
                          max_slippage_pct=0.005, symbol=None,
                          is_exit_order=False, confirm_fill=True,
                          order_role="ENTRY", candle_ts=None,
                          intent_id=None, client_order_id=None):
        """
        Routes orders with slippage checks, retry logic, and fill confirmation.

        is_exit_order (bool): If True, bypasses slippage guard. Exit orders
            MUST always execute regardless of slippage.
        confirm_fill (bool): If True, polls order status until filled/cancelled.
        order_role (str): "ENTRY", "TP1", "TP2", "SL", "EMERGENCY"
        candle_ts (int/float): Timestamp of candle that generated the signal
        """
        if symbol is None:
            symbol = Config.SYMBOL

        try:
            intent_id, client_order_id = self.prepare_order_intent(
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                order_role=order_role,
                intent_id=intent_id,
                client_order_id=client_order_id,
                is_exit_order=is_exit_order,
            )
        except Exception as e:
            return ExecutionResult(
                state=ExecutionState.NOT_SUBMITTED,
                requested_qty=float(amount or 0.0),
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue="COINDCX" if self.coindcx_client else "BINANCE",
                error=f"intent journal failed: {e}",
            )

        # Initialize futures settings on first order if applicable
        if Config.EXCHANGE_TYPE == 'futures':
            try:
                await self._init_futures(symbol)
            except Exception as e:
                print(f"[EXECUTION CRITICAL] Order rejected: Futures configuration failed: {e}")
                return ExecutionResult(
                    state=ExecutionState.REJECTED,
                    requested_qty=float(amount or 0.0),
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue="BINANCE",
                    error=f"Futures leverage/margin configuration failed: {e}"
                )

        if self.coindcx_client:
            coindcx_symbol = symbol
            if Config.COINDCX_TRADE_INR:
                target = symbol.split('/')[0]
                coindcx_symbol = f"{target}/INR"
            order = await self.coindcx_client.place_order(
                side, order_type, amount, price, symbol=coindcx_symbol,
                client_order_id=client_order_id, intent_id=intent_id,
            )
            result = coerce_execution_result(
                order, requested_qty=amount,
                client_order_id=client_order_id, intent_id=intent_id,
                venue="COINDCX",
            )
            if result.has_exchange_order and confirm_fill and result.state == ExecutionState.ACCEPTED:
                result = await self.coindcx_client.wait_for_fill(
                    str(result.exchange_order_id),
                    requested_qty=amount,
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                )
            self.intent_journal.result(result)
            return result

        # 0. Enforce exchange LOT_SIZE precision
        try:
            if not self.trade_client.markets:
                await self.trade_client.load_markets()
            amount_prec = self.trade_client.amount_to_precision(symbol, amount)
            precise_amount = float(amount_prec) if amount_prec is not None else float(amount)
            if precise_amount != amount:
                print(f"[EXECUTION] Quantity rounded for exchange precision: {amount} → {precise_amount}")
            amount = precise_amount
        except Exception as e:
            print(f"[EXECUTION] WARNING: Could not apply precision rounding ({e}). Using raw amount.")

        # Check minimum order size
        try:
            markets = self.trade_client.markets
            if markets and symbol in markets:
                min_amount = markets[symbol].get('limits', {}).get('amount', {}).get('min', 0) or 0
                if amount < min_amount:
                    print(f"[EXECUTION] Order rejected: Amount {amount:.8f} is below minimum {min_amount} for {symbol}")
                    return ExecutionResult(
                        state=ExecutionState.REJECTED,
                        requested_qty=float(amount),
                        client_order_id=client_order_id,
                        intent_id=intent_id,
                        venue="BINANCE",
                        error="below exchange minimum",
                    )
        except Exception as e:
            print(f"[EXECUTION] WARNING: Could not check minimum order size ({e}). Proceeding anyway.")

        # 1. Slippage check for market ENTRY orders only
        if order_type.upper() == "MARKET" and not is_exit_order:
            ticker = await self.execute_with_retry(self.public_client.fetch_ticker, symbol)
            if not ticker:
                print("[EXECUTION] Order aborted: Unable to fetch live price ticker for slippage check.")
                return ExecutionResult(
                    state=ExecutionState.NOT_SUBMITTED,
                    requested_qty=float(amount),
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue="BINANCE",
                    error="ticker unavailable",
                )

            current_price = ticker['last']
            if price is not None:
                if side.upper() == "BUY":
                    slippage = (current_price - price) / price
                    if slippage > max_slippage_pct:
                        print(f"[EXECUTION] Order aborted: Slippage ({slippage*100:.2f}%) exceeds max ({max_slippage_pct*100:.2f}%).")
                        return ExecutionResult(
                            state=ExecutionState.NOT_SUBMITTED,
                            requested_qty=float(amount),
                            client_order_id=client_order_id,
                            intent_id=intent_id,
                            venue="BINANCE",
                            error="slippage limit exceeded",
                        )
                elif side.upper() == "SELL":
                    slippage = (price - current_price) / price
                    if slippage > max_slippage_pct:
                        print(f"[EXECUTION] Order aborted: Slippage ({slippage*100:.2f}%) exceeds max ({max_slippage_pct*100:.2f}%).")
                        return ExecutionResult(
                            state=ExecutionState.NOT_SUBMITTED,
                            requested_qty=float(amount),
                            client_order_id=client_order_id,
                            intent_id=intent_id,
                            venue="BINANCE",
                            error="slippage limit exceeded",
                        )

        params: dict[str, Any] = {'clientOrderId': client_order_id}
        if is_exit_order and Config.EXCHANGE_TYPE == 'futures':
            params['reduceOnly'] = True
        
        fn = None
        args = [symbol, amount]

        if order_type.upper() == "MARKET":
            fn = self.trade_client.create_market_order
            args = [symbol, side.lower(), amount, params]
        elif order_type.upper() == "LIMIT":
            if price is None:
                print("[EXECUTION] Order error: Limit orders require a price.")
                return ExecutionResult(
                    state=ExecutionState.NOT_SUBMITTED,
                    requested_qty=float(amount),
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue="BINANCE",
                    error="limit order requires price",
                )
            fn = self.trade_client.create_order
            args = [symbol, 'limit', side.lower(), amount, price, params]

        if fn is None:
            fn = self.trade_client.create_order
            args = [symbol, order_type.lower(), side.lower(), amount, price, params]

        print(f"[EXECUTION] Sending {order_type.upper()} {side.upper()} order for {amount} {symbol} (Client ID: {client_order_id})...")
        try:
            # A create-order POST is an economic mutation.  Once it has been
            # sent, a timeout is ambiguous; do not blindly send it again.
            # Resolve the original clientOrderId instead.
            order = await self._execute_mutation_once(fn, *args)
        except Exception as e:
            resolved = await self._resolve_binance_order(symbol, client_order_id, intent_id, amount)
            if resolved is not None:
                self.intent_journal.result(resolved)
                return resolved
            result = ExecutionResult(
                state=ExecutionState.EXECUTION_UNKNOWN,
                requested_qty=float(amount),
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue="BINANCE",
                error=str(e),
            )
            self.intent_journal.result(result)
            return result

        result = coerce_execution_result(
            order,
            requested_qty=amount,
            client_order_id=client_order_id,
            intent_id=intent_id,
            venue="BINANCE",
        )
        if result.has_exchange_order:
            print(f"[EXECUTION] Order acknowledged! ID: {result.exchange_order_id}, State: {result.state.value}")
        if confirm_fill and result.has_exchange_order and result.state == ExecutionState.ACCEPTED:
            result = await self.wait_for_fill(
                str(result.exchange_order_id), symbol, timeout=30.0,
                requested_qty=amount, client_order_id=client_order_id,
                intent_id=intent_id,
            )
        self.intent_journal.result(result)
        return result

    async def _resolve_binance_order(
        self,
        symbol: str,
        client_order_id: str,
        intent_id: str | None,
        requested_qty: float,
    ) -> ExecutionResult | None:
        """Resolve an ambiguous Binance submission by the original client ID."""
        candidates: dict[str, dict[str, Any]] = {}
        for method_name in ("fetch_open_orders", "fetch_orders"):
            method = getattr(self.trade_client, method_name, None)
            if method is None:
                continue
            try:
                orders = await method(symbol)
            except Exception:
                continue
            if not orders:
                continue
            for order in orders:
                if not isinstance(order, dict):
                    continue
                info = order.get("info") if isinstance(order.get("info"), dict) else {}
                candidate_id = order.get("clientOrderId") or order.get("client_order_id") or info.get("clientOrderId")
                if candidate_id and str(candidate_id) == client_order_id:
                    exchange_id = order.get("id") or order.get("orderId") or candidate_id
                    candidates[str(exchange_id)] = order
        if len(candidates) == 1:
            return ExecutionResult.from_exchange(
                next(iter(candidates.values())), requested_qty=requested_qty,
                client_order_id=client_order_id, intent_id=intent_id,
                venue="BINANCE",
            )
        return None

    async def _execute_mutation_once(self, func, *args):
        """Invoke one exchange mutation without retrying an ambiguous POST."""
        result = func(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def execute_with_retry(self, func, *args, retries=3, delay=1.0, **kwargs):
        """
        Executes a CCXT call with exponential backoff retry logic.
        """
        for attempt in range(1, retries + 1):
            try:
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except ccxt.InsufficientFunds as e:
                print(f"[EXECUTION] Trade failed (Insufficient Funds): {e}")
                break
            except ccxt.InvalidOrder as e:
                # P1 Idempotency Fix: Duplicate order ID means the previous attempt succeeded!
                err_str = str(e).lower()
                if 'duplicate' in err_str or 'client order id' in err_str or 'already exists' in err_str:
                    print(f"[EXECUTION] Duplicate clientOrderId detected ({e}). Assuming order succeeded!")
                    if len(args) > 3 and isinstance(args[-1], dict) and 'clientOrderId' in args[-1]:
                        # Optional: Fetch order status here, but for now just raise NetworkError to trigger EXECUTION_UNKNOWN
                        raise ccxt.NetworkError(f"Duplicate clientOrderId indicates execution success but state unknown: {e}")
                print(f"[EXECUTION] Trade failed (Invalid Order): {e}")
                break
            except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.RequestTimeout) as e:
                if attempt == retries:
                    print(f"[EXECUTION] API failed after {retries} attempts. Final Error: {e}")
                    raise ccxt.NetworkError(f"API Failed completely: {e}") # Force exception
                sleep_time = delay * (2 ** (attempt - 1))
                print(f"[EXECUTION] API error ({e}). Retrying in {sleep_time:.1f}s (Attempt {attempt}/{retries})...")
                await asyncio.sleep(sleep_time)
            except ccxt.BaseError as e:
                print(f"[EXECUTION] CCXT Error (not retryable): {type(e).__name__}: {e}")
                raise
            except Exception as e:
                import traceback
                print(f"[EXECUTION] CRITICAL: Unexpected exception in {func.__name__}:")
                print(f"[EXECUTION]   Error type: {type(e).__name__}")
                print(f"[EXECUTION]   Error message: {e}")
                traceback.print_exc()
                raise
                break
        return None

    async def fetch_coindcx_user_info(self):
        """Fetch CoinDCX user profile information."""
        if self.coindcx_client:
            return await self.coindcx_client.fetch_user_info()
        return None

    async def place_native_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float, candle_ts: float = 0.0, intent_id: str | None = None, client_order_id: str | None = None) -> ExecutionResult:
        """
        Submits an exchange-native protective Stop Loss order.
        For LONG positions, places a SELL STOP_MARKET / STOP_LOSS.
        For SHORT positions, places a BUY STOP_MARKET / STOP_LOSS.
        """
        try:
            intent_id, client_order_id = self.prepare_order_intent(
                symbol=symbol, side=side, order_type="stop_loss", amount=amount,
                price=stop_price, order_role="SL", intent_id=intent_id,
                client_order_id=client_order_id,
            )
        except Exception as e:
            return ExecutionResult(
                state=ExecutionState.NOT_SUBMITTED, requested_qty=amount or 0.0,
                client_order_id=client_order_id, intent_id=intent_id,
                venue="COINDCX" if self.coindcx_client else "BINANCE", error=str(e),
            )

        if self.coindcx_client:
            coindcx_symbol = symbol
            if Config.COINDCX_TRADE_INR:
                target = symbol.split('/')[0]
                coindcx_symbol = f"{target}/INR"
            result = await self.coindcx_client.place_stop_loss(
                side=side,
                amount=amount,
                stop_price=stop_price,
                symbol=coindcx_symbol,
                client_order_id=client_order_id,
                intent_id=intent_id,
            )
            result = coerce_execution_result(result, requested_qty=amount, client_order_id=client_order_id, intent_id=intent_id, venue="COINDCX")
            self.intent_journal.result(result)
            return result

        # CCXT / Binance implementation
        try:
            if not self.trade_client.markets:
                await self.trade_client.load_markets()
            amount_prec = self.trade_client.amount_to_precision(symbol, amount)
            amount = float(amount_prec) if amount_prec is not None else amount
            price_prec = self.trade_client.price_to_precision(symbol, stop_price)
            stop_price = float(price_prec) if price_prec is not None else stop_price

            params: dict[str, Any] = {
                'stopPrice': stop_price,
                'clientOrderId': client_order_id
            }

            if Config.EXCHANGE_TYPE == 'futures':
                params['reduceOnly'] = True
                order_type = 'STOP_MARKET'
            else:
                params['stopLimitPrice'] = stop_price * (0.995 if side.lower() == 'sell' else 1.005)
                order_type = 'STOP_LOSS_LIMIT'

            print(f"[NATIVE SL] Submitting {order_type} {side.upper()} order for {amount} {symbol} @ {stop_price}...")
            sl_order = await self._execute_mutation_once(
                self.trade_client.create_order,
                symbol, order_type.lower(), side.lower(), amount, stop_price, params
            )
            sl_order = coerce_execution_result(
                sl_order,
                requested_qty=amount,
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue="BINANCE",
            )
            if sl_order.has_exchange_order:
                print(f"[NATIVE SL] ✅ Active on exchange! ID: {sl_order['id']}, Stop: {stop_price}")
                self.intent_journal.result(sl_order)
                return sl_order
        except Exception as e:
            print(f"[NATIVE SL ERROR] Failed to place exchange stop loss: {e}")
            resolved = await self._resolve_binance_order(symbol, client_order_id, intent_id, amount)
            if resolved is not None:
                self.intent_journal.result(resolved)
                return resolved
        result = ExecutionResult(
            state=ExecutionState.EXECUTION_UNKNOWN,
            requested_qty=amount or 0.0,
            client_order_id=client_order_id,
            intent_id=intent_id,
            venue="BINANCE",
            error="native stop-loss response did not contain an exchange order id",
        )
        self.intent_journal.result(result)
        return result

    async def check_order_filled(self, symbol: str, order_id: str) -> bool:
        """Checks if a specific order was filled."""
        if not order_id: return False
        if self.coindcx_client:
            try:
                status_data = await self.coindcx_client.fetch_order_status(order_id)
                if status_data and status_data.get('status') in ('filled', 'closed'):
                    return True
            except: pass
            return False
        
        try:
            order = await self.execute_with_retry(self.trade_client.fetch_order, order_id, symbol)
            if order and order.get('status') in ('filled', 'closed'):
                return True
        except: pass
        return False

    async def verify_order_active(self, symbol: str, order_id: str) -> str:
        """Verifies if an order is actively resting on the exchange."""
        if not order_id:
            return 'INACTIVE'
        if self.coindcx_client:
            try:
                status_data = await self.coindcx_client.fetch_order_status(order_id)
                if status_data is None:
                    return 'UNKNOWN'
                if status_data.get('status') in ('open', 'active', 'untriggered', 'pending'):
                    return 'ACTIVE'
                if status_data.get('status') in ('cancelled', 'canceled', 'rejected', 'expired', 'closed', 'filled'):
                    return 'INACTIVE'
                return 'UNKNOWN'
            except Exception as e:
                print(f"[ORDER VERIFY] CoinDCX Error checking order {order_id}: {e}")
                return 'UNKNOWN'

        try:
            order = await self.execute_with_retry(self.trade_client.fetch_order, order_id, symbol)
            if order is None:
                return 'UNKNOWN'
            if order.get('status') in ('open', 'untriggered', 'pending', 'new', 'active'):
                return 'ACTIVE'
            if order.get('status') in ('closed', 'filled', 'canceled', 'cancelled', 'rejected', 'expired'):
                return 'INACTIVE'
            return 'UNKNOWN'
        except ccxt.OrderNotFound:
            return 'INACTIVE'
        except Exception as e:
            print(f"[ORDER VERIFY] Error checking order {order_id}: {e}")
            return 'UNKNOWN'

    async def cancel_order_safe(self, symbol: str, order_id: str) -> ExecutionResult:
        """Safely cancels an order without throwing unhandled exceptions."""
        if not order_id:
            return ExecutionResult(state=ExecutionState.ALREADY_CANCELLED, venue="BINANCE")
        if self.coindcx_client:
            try:
                raw = await self.coindcx_client.cancel_order(order_id)
                result = coerce_execution_result(raw, venue="COINDCX")
                result.exchange_order_id = order_id
                if result.state != ExecutionState.CANCELLED:
                    return result
                status_data = await self.coindcx_client.fetch_order_status(order_id)
                if status_data is None:
                    result.state = ExecutionState.CANCEL_UNKNOWN
                elif status_data.get('status') not in ('cancelled', 'canceled', 'rejected', 'expired', 'closed'):
                    result.state = ExecutionState.CANCEL_UNKNOWN
                return result
            except Exception as e:
                print(f"[EXECUTION] Warning cancelling CoinDCX order {order_id}: {e}")
                return ExecutionResult(state=ExecutionState.CANCEL_UNKNOWN, exchange_order_id=order_id, venue="COINDCX", error=str(e))

        try:
            await self.execute_with_retry(self.trade_client.cancel_order, order_id, symbol)
            status = await self.trade_client.fetch_order(order_id, symbol)
            if status is None:
                return ExecutionResult(state=ExecutionState.CANCEL_UNKNOWN, exchange_order_id=order_id, venue="BINANCE")
            status_name = (status.get('status') or '').lower()
            if status_name in ('canceled', 'cancelled', 'closed', 'expired', 'rejected'):
                print(f"[EXECUTION] Successfully cancelled order {order_id}")
                return ExecutionResult(state=ExecutionState.CANCELLED, exchange_order_id=order_id, venue="BINANCE", raw=dict(status))
            return ExecutionResult(state=ExecutionState.CANCEL_UNKNOWN, exchange_order_id=order_id, venue="BINANCE", raw=dict(status))
        except ccxt.OrderNotFound:
            return ExecutionResult(state=ExecutionState.ALREADY_CANCELLED, exchange_order_id=order_id, venue="BINANCE")
        except Exception as e:
            print(f"[EXECUTION] Warning cancelling order {order_id}: {e}")
            return ExecutionResult(state=ExecutionState.CANCEL_UNKNOWN, exchange_order_id=order_id, venue="BINANCE", error=str(e))

    async def emergency_flatten_position(self, symbol: str, side: str, amount: float, reason: str = "EMERGENCY") -> ExecutionResult:
        """Force-closes a position immediately at market to avoid unprotected risk."""
        print(f"[EMERGENCY FLATTEN] 🚨 Flattening {side.upper()} {amount} {symbol} | Reason: {reason}")
        exit_side = 'sell' if side.upper() in ('BUY', 'LONG') else 'buy'
        return await self.place_order(
            side=exit_side,
            order_type='market',
            amount=amount,
            symbol=symbol,
            is_exit_order=True,
            order_role="EMERGENCY"
        )

    async def fetch_usdt_inr_rate(self, side: str = "BUY") -> float | None:
        """Dynamic FX rate provider delegating to CoinDCXClient."""
        if self.coindcx_client:
            return await self.coindcx_client.fetch_usdt_inr_rate(side=side)
        return None

    async def reconcile_intent_on_exchange(self, intent: dict[str, Any]) -> Optional[ExecutionResult]:
        """Queries exchange to authoritatively reconcile an in-flight or ambiguous intent."""
        symbol = intent.get('symbol')
        side = intent.get('side', 'buy')
        qty = float(intent.get('requested_qty', 0.0))
        client_order_id = intent.get('client_order_id')
        intent_id = intent.get('intent_id')
        venue = intent.get('venue', 'BINANCE')

        if str(venue).upper() == 'COINDCX' and self.coindcx_client:
            return await self.coindcx_client._reconcile_ambiguous_order(
                symbol or 'BTCINR', side, qty,
                created_after_ts=int((intent.get('created_at', time.time()) - 30) * 1000),
                client_order_id=client_order_id,
                intent_id=intent_id
            )

        # Binance reconciliation by targeted clientOrderId, open orders, and broader history
        try:
            if not self.trade_client or not symbol:
                return ExecutionResult(
                    state=ExecutionState.EXECUTION_UNKNOWN,
                    requested_qty=qty,
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue=venue,
                    error="trade_client or symbol not initialized"
                )

            # 1. Targeted lookup by clientOrderId if supported
            if client_order_id and hasattr(self.trade_client, 'fetch_order'):
                try:
                    if inspect.iscoroutinefunction(self.trade_client.fetch_order):
                        order = await self.trade_client.fetch_order(
                            id="",
                            symbol=symbol,
                            params={'origClientOrderId': client_order_id}
                        )
                    else:
                        order = self.trade_client.fetch_order(
                            id="",
                            symbol=symbol,
                            params={'origClientOrderId': client_order_id}
                        )
                    if order and isinstance(order, dict):
                        return ExecutionResult.from_exchange(
                            order, requested_qty=qty, client_order_id=client_order_id,
                            intent_id=intent_id, venue=venue
                        )
                except ccxt.OrderNotFound:
                    return ExecutionResult(
                        state=ExecutionState.REJECTED,
                        requested_qty=qty,
                        client_order_id=client_order_id,
                        intent_id=intent_id,
                        venue=venue,
                        error="Authoritatively verified absent on exchange (OrderNotFound)"
                    )
                except Exception as e:
                    # Targeted query failed; fall through to order list inspection
                    pass

            # 2. Check open orders
            open_orders = await self.execute_with_retry(self.trade_client.fetch_open_orders, symbol)
            for o in (open_orders or []):
                o_cid = o.get('clientOrderId') or (o.get('info') or {}).get('clientOrderId')
                if o_cid and client_order_id and str(o_cid) == str(client_order_id):
                    return ExecutionResult.from_exchange(o, requested_qty=qty, client_order_id=client_order_id, intent_id=intent_id, venue=venue)
            
            # 3. Check closed/filled orders (100 limit)
            closed_orders = await self.execute_with_retry(self.trade_client.fetch_closed_orders, symbol, limit=100)
            for o in (closed_orders or []):
                o_cid = o.get('clientOrderId') or (o.get('info') or {}).get('clientOrderId')
                if o_cid and client_order_id and str(o_cid) == str(client_order_id):
                    return ExecutionResult.from_exchange(o, requested_qty=qty, client_order_id=client_order_id, intent_id=intent_id, venue=venue)

            # If not authoritatively absent via OrderNotFound and not found in order lists, quarantine in EXECUTION_UNKNOWN
            return ExecutionResult(
                state=ExecutionState.EXECUTION_UNKNOWN,
                requested_qty=qty,
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue=venue,
                error="Intent could not be authoritatively verified on exchange; quarantined in SAFE MODE"
            )
        except Exception as e:
            print(f"[RECONCILE INTENT] Error querying exchange for intent {intent_id}: {e}")
            return ExecutionResult(
                state=ExecutionState.EXECUTION_UNKNOWN,
                requested_qty=qty,
                client_order_id=client_order_id,
                intent_id=intent_id,
                venue=venue,
                error=str(e)
            )

    async def replay_and_resolve_unresolved_intents(self) -> dict[str, ExecutionResult]:
        """
        Phase 5 Invariant: Replays and resolves all unresolved intents from journal on startup.
        """
        unresolved = self.intent_journal.unresolved()
        resolutions: dict[str, ExecutionResult] = {}
        if not unresolved:
            return resolutions

        print(f"[REPLAY] Found {len(unresolved)} unresolved intent(s) in journal. Reconciling with exchange...")
        for intent in unresolved:
            intent_id = str(intent.get('intent_id') or '')
            if not intent_id:
                continue
            client_order_id = intent.get('client_order_id')
            qty = float(intent.get('requested_qty', 0.0))
            venue = intent.get('venue', 'BINANCE')
            symbol = intent.get('symbol', Config.SYMBOL)
            order_role = intent.get('order_role', 'ENTRY')
            side = intent.get('side', 'buy')
            price = intent.get('price')
            
            discovered = await self.reconcile_intent_on_exchange(intent)
            if discovered is not None and not discovered.is_unknown:
                if not discovered.raw: discovered.raw = {}
                discovered.raw['order_role'] = order_role
                discovered.raw['symbol'] = symbol
                discovered.raw['side'] = side
                discovered.raw['price'] = price
                discovered.intent_id = intent_id
                discovered.client_order_id = client_order_id
                discovered.venue = venue
                print(f"[REPLAY] Intent {intent_id} ({client_order_id}) resolved on exchange: {discovered.state.value} (Filled: {discovered.filled_qty})")
                self.intent_journal.result(discovered)
                resolutions[intent_id] = discovered
            elif discovered is not None and discovered.is_unknown:
                if not discovered.raw: discovered.raw = {}
                discovered.raw['order_role'] = order_role
                discovered.raw['symbol'] = symbol
                discovered.raw['side'] = side
                discovered.raw['price'] = price
                discovered.intent_id = intent_id
                discovered.client_order_id = client_order_id
                discovered.venue = venue
                print(f"[REPLAY] Intent {intent_id} ({client_order_id}) remains UNKNOWN due to exchange query error; quarantined.")
                self.intent_journal.result(discovered)
                resolutions[intent_id] = discovered
            else:
                print(f"[REPLAY] Intent {intent_id} ({client_order_id}) verified absent on exchange. Marking REJECTED.")
                rejected_res = ExecutionResult(
                    state=ExecutionState.REJECTED,
                    requested_qty=qty,
                    client_order_id=client_order_id,
                    intent_id=intent_id,
                    venue=venue,
                    error="Unplaced or rejected before restart; verified absent on exchange",
                    raw={'order_role': order_role, 'symbol': symbol, 'side': side, 'price': price}
                )
                self.intent_journal.result(rejected_res)
                resolutions[intent_id] = rejected_res

        return resolutions

import math
import os
import time
import asyncio
from typing import Dict, Any, Optional
from config import Config

class RiskManager:
    def __init__(self):
        self.daily_starting_equity = None
        self.current_drawdown_pct = 0.0
        self.reserved_risk_pct = 0.0 # Atomic risk tracker for pending orders
        self.reserved_open_count = 0
        self.reserved_longs_count = 0
        self.reserved_shorts_count = 0
        self.active_reservations: dict[str, dict[str, Any]] = {}
        # Canonical decimal representation: 1.2% = 0.012
        raw_cap = float(getattr(Config, 'MAX_CORRELATED_RISK_PCT', 0.012))
        self.max_correlated_risk_pct = raw_cap / 100.0 if raw_cap > 0.2 else raw_cap
        self._lock = asyncio.Lock()
        # Portfolio-level lock for atomic risk reservation across concurrent entry tasks
        self.portfolio_lock = asyncio.Lock()

    def _recalculate_reserved_totals(self):
        """Derives current reservation metrics from active durable reservations."""
        total_risk = 0.0
        open_count = 0
        longs = 0
        shorts = 0
        for res in self.active_reservations.values():
            if res.get('state') == 'ACTIVE':
                total_risk += float(res.get('risk_pct', 0.0))
                open_count += 1
                side = str(res.get('side', '')).upper()
                if side in ("BUY", "LONG"):
                    longs += 1
                elif side in ("SELL", "SHORT"):
                    shorts += 1
        self.reserved_risk_pct = max(0.0, total_risk)
        self.reserved_open_count = max(0, open_count)
        self.reserved_longs_count = max(0, longs)
        self.reserved_shorts_count = max(0, shorts)

    async def check_and_reserve_risk_atomic(self, current_open_risk_pct: float, proposed_risk_pct: float, side: str = "BUY", reservation_id: str | None = None, symbol: str = "") -> bool:
        """
        Atomically checks and commits durable reservation in a single critical section (P0 Invariant):
        CurrentRisk + ReservedRisk + ProposedRisk <= max_correlated_risk_pct
        """
        async with self._lock:
            self._recalculate_reserved_totals()
            total_projected_risk = current_open_risk_pct + self.reserved_risk_pct + proposed_risk_pct
            if total_projected_risk > (self.max_correlated_risk_pct + 0.0001):
                try:
                    print(f"[RISK] ⛔ Portfolio risk cap breach prevented! Projected: {total_projected_risk*100:.2f}%, Limit: {self.max_correlated_risk_pct*100:.2f}%")
                except Exception:
                    print(f"[RISK] [BLOCK] Portfolio risk cap breach prevented! Projected: {total_projected_risk*100:.2f}%, Limit: {self.max_correlated_risk_pct*100:.2f}%")
                return False
            
            res_id = reservation_id or f"RES_{symbol.replace('/', '')}_{int(time.time()*1000)}"
            self.active_reservations[res_id] = {
                'reservation_id': res_id,
                'symbol': symbol,
                'side': side.upper(),
                'risk_pct': float(proposed_risk_pct),
                'timestamp': time.time(),
                'state': 'ACTIVE'
            }
            self._recalculate_reserved_totals()
            return True

    async def can_open_trade_atomic(self, current_open_risk_pct: float, proposed_risk_pct: float, side: str = "BUY", reservation_id: str | None = None, symbol: str = "") -> bool:
        """Compatibility wrapper for atomic check and reserve."""
        return await self.check_and_reserve_risk_atomic(current_open_risk_pct, proposed_risk_pct, side=side, reservation_id=reservation_id, symbol=symbol)

    async def release_risk(self, risk_pct: float, side: str = "BUY", reservation_id: str | None = None):
        """Releases reserved risk idempotently by reservation ID or scalar value."""
        async with self._lock:
            if reservation_id and reservation_id in self.active_reservations:
                self.active_reservations[reservation_id]['state'] = 'RELEASED'
                del self.active_reservations[reservation_id]
            else:
                # Fallback to releasing first matching anonymous reservation
                matched_id = None
                for r_id, r_data in list(self.active_reservations.items()):
                    if r_data.get('state') == 'ACTIVE' and abs(float(r_data.get('risk_pct', 0.0)) - float(risk_pct)) < 1e-6:
                        matched_id = r_id
                        break
                if matched_id:
                    del self.active_reservations[matched_id]
            self._recalculate_reserved_totals()

    def serialize_reservations(self) -> dict[str, dict[str, Any]]:
        """Serializes active risk reservations for persistent bot state storage."""
        return {
            r_id: dict(data)
            for r_id, data in self.active_reservations.items()
            if data.get('state') == 'ACTIVE'
        }

    def load_reservations(self, reservations_data: dict[str, Any] | None):
        """Restores durable risk reservations after process restart."""
        self.active_reservations.clear()
        if isinstance(reservations_data, dict):
            for r_id, data in reservations_data.items():
                if isinstance(data, dict) and data.get('state', 'ACTIVE') == 'ACTIVE':
                    self.active_reservations[str(r_id)] = {
                        'reservation_id': str(r_id),
                        'symbol': str(data.get('symbol', '')),
                        'side': str(data.get('side', 'BUY')).upper(),
                        'risk_pct': float(data.get('risk_pct', 0.0)),
                        'timestamp': float(data.get('timestamp', time.time())),
                        'state': 'ACTIVE'
                    }
        self._recalculate_reserved_totals()

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        quote_currency: str = "USDT",
        equity_currency: str | None = None,
        conversion_rate: float | None = None,
        is_inr: bool | None = None,
    ) -> float:
        """
        Calculates position size dynamically based on stop-loss distance and account risk percentage,
        strictly enforcing currency unit isolation between Account Equity and Quote/Price currencies.
        """
        if account_equity is None or entry_price is None or stop_loss is None:
            return 0.0
        if math.isnan(account_equity) or math.isnan(entry_price) or math.isnan(stop_loss):
            return 0.0
        if account_equity <= 0 or entry_price <= 0:
            return 0.0

        # Determine currencies and conversion rate
        if is_inr is None:
            is_inr = (equity_currency == 'INR') or getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
        
        effective_equity_curr = "INR" if is_inr else (equity_currency or "USDT")
        rate = float(conversion_rate or getattr(Config, 'USDT_INR_RATE', 85.0))
        if rate <= 0:
            rate = 85.0

        # Convert entry_price and stop_loss into the Account Equity currency
        # If equity is INR and price is in USDT (standard Binance stream price):
        if effective_equity_curr == "INR" and str(quote_currency).upper() in ("USDT", "USD"):
            entry_price_equity_curr = entry_price * rate
            stop_loss_equity_curr = stop_loss * rate
        elif effective_equity_curr == "USDT" and str(quote_currency).upper() == "INR":
            entry_price_equity_curr = entry_price / rate
            stop_loss_equity_curr = stop_loss / rate
        else:
            # Currencies match (e.g. USDT equity & USDT price, or INR equity & INR price)
            entry_price_equity_curr = entry_price
            stop_loss_equity_curr = stop_loss

        # 1. Calculate budget to risk (Normalize and cap max risk relative to currency)
        base_risk = account_equity * (Config.RISK_PCT / 100.0)
        curr_mult = rate if effective_equity_curr == "INR" else 1.0
        max_single_trade_risk = float(os.getenv("MAX_SINGLE_TRADE_RISK_USDT", "25.0")) * curr_mult
        threshold_equity = 1000.0 * curr_mult
        trade_risk = min(base_risk, max_single_trade_risk) if account_equity >= threshold_equity else base_risk

        # 2. Calculate stop distance in Account Equity currency
        stop_distance = abs(entry_price_equity_curr - stop_loss_equity_curr)
        if stop_distance <= 0 or math.isnan(stop_distance) or math.isinf(stop_distance):
            print(f"⚠️ RISK MANAGER WARNING: Stop distance is non-positive or invalid ({stop_distance})")
            print(f"   Entry Price: {entry_price}, Stop Loss: {stop_loss}")
            print(f"   Failing closed: returning 0.0 position size")
            return 0.0
            
        # 3. Calculate position size in base asset units
        position_size = trade_risk / stop_distance
        
        # 4. Limit check: Position cost in Equity currency must be <= total equity * max_allocation * leverage
        is_futures = getattr(Config, 'EXCHANGE_TYPE', 'spot') == 'futures'
        leverage = getattr(Config, 'FUTURES_LEVERAGE', 1.0) if is_futures else 1.0
        max_alloc = getattr(Config, 'MAX_TRADE_ALLOCATION', getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35))
        if max_alloc > 1.0:
            max_alloc = max_alloc / 100.0 # Fail-safe normalization (e.g. 35 -> 0.35)
        
        max_position_value = account_equity * max_alloc * leverage
        
        # Ensure minimum notional threshold can still be met for small accounts (e.g. 100 INR / $10 USDT)
        min_notional = 100.0 if effective_equity_curr == "INR" else 10.0
        if account_equity >= min_notional and max_position_value < min_notional:
            max_position_value = min(account_equity * 0.95, min_notional * 1.1)

        position_value_equity_curr = position_size * entry_price_equity_curr
        
        if position_value_equity_curr > max_position_value:
            if entry_price_equity_curr <= 0: return 0.0
            position_size = (max_position_value * 0.999) / entry_price_equity_curr
            print(f"[RISK] Position size capped at {max_alloc*100:.0f}% max capital allocation: {position_size:.6f} ({effective_equity_curr} {position_size*entry_price_equity_curr:.2f})")
            
        return round(position_size, 6)

    def check_circuit_breaker(self, current_equity: float) -> bool:
        """
        Implements daily max loss limits. Stops bot if drawdown limit is hit.
        """
        if current_equity is None or math.isnan(current_equity):
            print("🚨 CIRCUIT BREAKER WARNING: Invalid equity value received. Halting trading.")
            return False

        if self.daily_starting_equity is None or math.isnan(self.daily_starting_equity):
            self.daily_starting_equity = current_equity
            return True
            
        pnl = current_equity - self.daily_starting_equity
        if self.daily_starting_equity <= 0:
            self.current_drawdown_pct = 0.0
        else:
            self.current_drawdown_pct = (pnl / self.daily_starting_equity) * 100.0
        
        # Check if max drawdown is exceeded (drawdown is negative PnL)
        if self.current_drawdown_pct <= -Config.MAX_DAILY_LOSS_PCT:
            msg = f"🚨 CIRCUIT BREAKER TRIGGERED: Daily loss limit hit ({self.current_drawdown_pct:.2f}%). Trading suspended."
            try:
                print(msg)
            except UnicodeEncodeError:
                import sys
                enc = sys.stdout.encoding or 'utf-8'
                print(msg.encode(enc, errors='replace').decode(enc))
            return False
            
        return True

    def reset_daily_equity(self, current_equity: float):
        """Reset starting balance for the day (run at UTC midnight)."""
        self.daily_starting_equity = current_equity
        self.current_drawdown_pct = 0.0
        eq_val = current_equity if (current_equity is not None and not math.isnan(current_equity)) else 0.0
        print(f"[RISK] Daily equity checkpoint reset to {eq_val:.2f} USDT")

    def update_trailing_stop(self, entry_price: float, extreme_price: float, stop_loss: float, curr_atr: float, position_side: str = "LONG") -> float:
        """
        Calculates ATR-based trailing stop loss adjustments.
        """
        if stop_loss is None or curr_atr is None or extreme_price is None:
            return stop_loss if stop_loss is not None else 0.0
        if math.isnan(stop_loss) or math.isnan(curr_atr) or math.isnan(extreme_price):
            return stop_loss

        # Trailing offset = multiple of the current ATR
        trailing_offset = curr_atr * Config.TRAILING_ATR_MULT

        if position_side.upper() == "LONG":
            new_stop = extreme_price - trailing_offset
            return new_stop if new_stop > stop_loss else stop_loss

        elif position_side.upper() == "SHORT":
            new_stop = extreme_price + trailing_offset
            return new_stop if new_stop < stop_loss else stop_loss

        return stop_loss

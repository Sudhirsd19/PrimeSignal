import math
import os
import asyncio
from config import Config

class RiskManager:
    def __init__(self):
        self.daily_starting_equity = None
        self.current_drawdown_pct = 0.0
        self.reserved_risk_pct = 0.0 # Atomic risk tracker for pending orders
        self.reserved_open_count = 0
        self.reserved_longs_count = 0
        self.reserved_shorts_count = 0
        # Canonical decimal representation: 1.2% = 0.012
        raw_cap = float(getattr(Config, 'MAX_CORRELATED_RISK_PCT', 0.012))
        self.max_correlated_risk_pct = raw_cap / 100.0 if raw_cap > 0.2 else raw_cap
        self._lock = asyncio.Lock()
        # Portfolio-level lock for atomic risk reservation across concurrent entry tasks
        self.portfolio_lock = asyncio.Lock()

    async def check_and_reserve_risk_atomic(self, current_open_risk_pct: float, proposed_risk_pct: float, side: str = "BUY") -> bool:
        """
        Atomically checks and commits reservation in a single critical section (P0 Invariant):
        CurrentRisk + ReservedRisk + ProposedRisk <= max_correlated_risk_pct
        """
        async with self._lock:
            total_projected_risk = current_open_risk_pct + self.reserved_risk_pct + proposed_risk_pct
            if total_projected_risk > (self.max_correlated_risk_pct + 0.0001):
                try:
                    print(f"[RISK] ⛔ Portfolio risk cap breach prevented! Projected: {total_projected_risk*100:.2f}%, Limit: {self.max_correlated_risk_pct*100:.2f}%")
                except Exception:
                    print(f"[RISK] [BLOCK] Portfolio risk cap breach prevented! Projected: {total_projected_risk*100:.2f}%, Limit: {self.max_correlated_risk_pct*100:.2f}%")
                return False
            self.reserved_risk_pct += proposed_risk_pct
            self.reserved_open_count += 1
            if side.upper() in ("BUY", "LONG"):
                self.reserved_longs_count += 1
            elif side.upper() in ("SELL", "SHORT"):
                self.reserved_shorts_count += 1
            return True

    async def can_open_trade_atomic(self, current_open_risk_pct: float, proposed_risk_pct: float, side: str = "BUY") -> bool:
        """Compatibility wrapper for atomic check and reserve."""
        return await self.check_and_reserve_risk_atomic(current_open_risk_pct, proposed_risk_pct, side=side)

    async def release_risk(self, risk_pct: float, side: str = "BUY"):
        """Releases reserved risk when order is confirmed filled or cancelled."""
        async with self._lock:
            self.reserved_risk_pct = max(0.0, self.reserved_risk_pct - risk_pct)
            self.reserved_open_count = max(0, self.reserved_open_count - 1)
            if side.upper() in ("BUY", "LONG"):
                self.reserved_longs_count = max(0, self.reserved_longs_count - 1)
            elif side.upper() in ("SELL", "SHORT"):
                self.reserved_shorts_count = max(0, self.reserved_shorts_count - 1)

    def calculate_position_size(self, account_equity: float, entry_price: float, stop_loss: float) -> float:
        """
        Calculates position size dynamically based on stop-loss distance and account risk percentage.
        """
        if account_equity is None or entry_price is None or stop_loss is None:
            return 0.0
        if math.isnan(account_equity) or math.isnan(entry_price) or math.isnan(stop_loss):
            return 0.0
        if account_equity <= 0 or entry_price <= 0:
            return 0.0

        # 1. Calculate budget to risk (Normalize and cap max risk relative to currency)
        base_risk = account_equity * (Config.RISK_PCT / 100.0)
        is_inr = getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
        curr_mult = 85.0 if is_inr else 1.0
        max_single_trade_risk = float(os.getenv("MAX_SINGLE_TRADE_RISK_USDT", "25.0")) * curr_mult
        threshold_equity = 1000.0 * curr_mult
        trade_risk = min(base_risk, max_single_trade_risk) if account_equity >= threshold_equity else base_risk

        # 2. Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0 or math.isnan(stop_distance):
            print(f"⚠️ RISK MANAGER WARNING: Stop distance is zero or negative ({stop_distance})")
            print(f"   Entry Price: {entry_price}, Stop Loss: {stop_loss}")
            print(f"   Falling back to default trade size: {Config.TRADE_AMOUNT}")
            return Config.TRADE_AMOUNT
            
        # 3. Calculate position size
        position_size = trade_risk / stop_distance
        
        # 4. Limit check: Position cost in USDT must be <= total equity * max_allocation * leverage
        is_futures = getattr(Config, 'EXCHANGE_TYPE', 'spot') == 'futures'
        leverage = getattr(Config, 'FUTURES_LEVERAGE', 1.0) if is_futures else 1.0
        max_alloc = getattr(Config, 'MAX_TRADE_ALLOCATION', getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35))
        if max_alloc > 1.0:
            max_alloc = max_alloc / 100.0 # Fail-safe normalization (e.g. 35 -> 0.35)
        
        max_position_value = account_equity * max_alloc * leverage
        
        # Ensure minimum notional threshold can still be met for small accounts (e.g. 1000 INR / $10 USDT)
        is_inr = getattr(Config, 'COINDCX_TRADE_INR', False) and not Config.PAPER_TRADING
        min_notional = 100.0 if is_inr else 10.0
        if account_equity >= min_notional and max_position_value < min_notional:
            max_position_value = min(account_equity * 0.95, min_notional * 1.1)

        position_value_usdt = position_size * entry_price
        
        if position_value_usdt > max_position_value:
            if entry_price <= 0: return 0.0
            position_size = (max_position_value * 0.999) / entry_price
            print(f"[RISK] Position size capped at {max_alloc*100:.0f}% max capital allocation: {position_size:.6f} (${position_size*entry_price:.2f})")
            
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

import math
from config import Config

class RiskManager:
    def __init__(self):
        self.daily_starting_equity = None
        self.current_drawdown_pct = 0.0

    def calculate_position_size(self, account_equity, entry_price, stop_loss):
        """
        Calculates position size dynamically based on stop-loss distance and account risk percentage.
        """
        if account_equity is None or entry_price is None or stop_loss is None:
            return 0.0
        if math.isnan(account_equity) or math.isnan(entry_price) or math.isnan(stop_loss):
            return 0.0
        if account_equity <= 0 or entry_price <= 0:
            return 0.0

        # 1. Calculate USDT budget to risk
        usdt_risk = account_equity * (Config.RISK_PCT / 100.0)

        # 2. Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0 or math.isnan(stop_distance):
            print(f"⚠️ RISK MANAGER WARNING: Stop distance is zero or negative ({stop_distance})")
            print(f"   Entry Price: {entry_price}, Stop Loss: {stop_loss}")
            print(f"   Falling back to default trade size: {Config.TRADE_AMOUNT}")
            return Config.TRADE_AMOUNT
            
        # 3. Calculate position size
        position_size = usdt_risk / stop_distance
        
        # 4. Limit check: Position cost in USDT must be <= total equity * leverage
        is_futures = getattr(Config, 'EXCHANGE_TYPE', 'spot') == 'futures'
        leverage = getattr(Config, 'FUTURES_LEVERAGE', 1.0) if is_futures else 1.0
        max_position_value = account_equity * leverage
        position_value_usdt = position_size * entry_price
        
        if position_value_usdt > max_position_value:
            if entry_price <= 0: return 0.0
            position_size = (max_position_value * 0.999) / entry_price
            print(f"[RISK] Position size capped at maximum account capacity ({leverage}x): {position_size:.6f}")
            
        return round(position_size, 6)

    def check_circuit_breaker(self, current_equity):
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

    def reset_daily_equity(self, current_equity):
        """Reset starting balance for the day (run at UTC midnight)."""
        self.daily_starting_equity = current_equity
        self.current_drawdown_pct = 0.0
        eq_val = current_equity if (current_equity is not None and not math.isnan(current_equity)) else 0.0
        print(f"[RISK] Daily equity checkpoint reset to {eq_val:.2f} USDT")

    def update_trailing_stop(self, entry_price, extreme_price, stop_loss, curr_atr, position_side="LONG"):
        """
        Calculates ATR-based trailing stop loss adjustments.
        """
        if stop_loss is None or curr_atr is None or extreme_price is None:
            return stop_loss
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

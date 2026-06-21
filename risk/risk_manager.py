from config import Config

class RiskManager:
    def __init__(self):
        self.daily_starting_equity = None
        self.current_drawdown_pct = 0.0

    def calculate_position_size(self, account_equity, entry_price, stop_loss):
        """
        Calculates position size dynamically based on stop-loss distance and account risk percentage.
        Formula:
            USDT Risk = Equity * Risk_Percent
            Size = USDT Risk / (Entry - StopLoss)
        """
        if account_equity <= 0:
            return 0.0

        # 1. Calculate USDT budget to risk
        usdt_risk = account_equity * (Config.RISK_PCT / 100.0)

        # 2. Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)

        # FIX #6: Add diagnostic logging for zero-distance fallback
        if stop_distance <= 0:
            print(f"⚠️ RISK MANAGER WARNING: Stop distance is zero or negative")
            print(f"   Entry Price: {entry_price}")
            print(f"   Stop Loss: {stop_loss}")
            print(f"   Falling back to default trade size: {Config.TRADE_AMOUNT}")
            return Config.TRADE_AMOUNT
            
        # 3. Calculate position size (amount in crypto asset, e.g. BTC)
        position_size = usdt_risk / stop_distance
        
        # 4. Limit check: Spot trading does not allow borrowing / leverage
        # Position cost in USDT must be <= total equity
        position_value_usdt = position_size * entry_price
        
        if position_value_usdt > account_equity:
            # De-leverage size to maximum possible cash balance with a 0.1% safety buffer to prevent rounding issues
            position_size = (account_equity * 0.999) / entry_price
            print(f"[RISK] Position size capped at maximum cash balance: {position_size:.6f}")
            
        # Dynamic check for minimum/maximum transaction sizes
        return round(position_size, 6)

    def check_circuit_breaker(self, current_equity):
        """
        Implements daily max loss limits. Stops bot if drawdown limit is hit.
        """
        if self.daily_starting_equity is None:
            self.daily_starting_equity = current_equity
            return True
            
        pnl = current_equity - self.daily_starting_equity
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
        print(f"[RISK] Daily equity checkpoint reset to {current_equity:.2f} USDT")

    def update_trailing_stop(self, entry_price, extreme_price, stop_loss, curr_atr, position_side="LONG"):
        """
        Calculates ATR-based trailing stop loss adjustments.

        Args:
            entry_price   : Original entry price of the position.
            extreme_price : For LONG → highest price reached since entry.
                            For SHORT → lowest price reached since entry.
            stop_loss     : Current stop loss level.
            curr_atr      : Current Average True Range.
            position_side : "LONG" or "SHORT"

        Returns:
            New stop loss value (moves only in profit direction, never against it).
        """
        if stop_loss is None or curr_atr is None:
            return stop_loss

        # Trailing offset = multiple of the current ATR
        trailing_offset = curr_atr * Config.TRAILING_ATR_MULT

        if position_side.upper() == "LONG":
            # New stop = extreme high minus trailing offset
            # Only valid to move stop UP (lock in more profit)
            new_stop = extreme_price - trailing_offset
            return new_stop if new_stop > stop_loss else stop_loss

        elif position_side.upper() == "SHORT":
            # New stop = extreme low plus trailing offset
            # Only valid to move stop DOWN (lock in more profit on a short)
            new_stop = extreme_price + trailing_offset
            return new_stop if new_stop < stop_loss else stop_loss

        return stop_loss

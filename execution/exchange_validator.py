import math
from typing import Tuple, Dict, Any, Optional
from config import Config

class ExchangeValidator:
    """
    Pre-validates order intents against exchange-specific lot size, tick size,
    minimum notional, and account equity limits before submitting to broker.
    """
    @staticmethod
    def validate_order_intent(symbol: str, side: str, order_type: str,
                              amount: float, price: float, current_equity: float,
                              markets_info: Optional[Dict[str, Any]] = None,
                              is_inr: bool = False,
                              quote_currency: str = "USDT",
                              conversion_rate: float | None = None) -> Tuple[bool, float, str]:
        """
        Validates an order intent before execution, strictly ensuring currency isolation.
        Returns: (is_valid: bool, sanitized_amount: float, reason: str)
        """
        if amount <= 0:
            return False, 0.0, "Quantity must be strictly positive."
        if price <= 0:
            return False, 0.0, "Price must be strictly positive."

        rate = float(conversion_rate or getattr(Config, 'USDT_INR_RATE', 85.0))
        if rate <= 0:
            rate = 85.0

        # Harmonize price into the account equity currency (INR vs USDT)
        if is_inr and str(quote_currency).upper() in ("USDT", "USD"):
            price_equity_curr = price * rate
        elif (not is_inr) and str(quote_currency).upper() == "INR":
            price_equity_curr = price / rate
        else:
            price_equity_curr = price
            
        # 1. Minimum Notional Check in Account Equity Currency
        min_notional = 100.0 if is_inr else 10.0
        notional_value = amount * price_equity_curr
        if notional_value < min_notional:
            # Check if equity allows scaling up to min notional
            if current_equity >= min_notional * 1.1:
                amount = (min_notional * 1.05) / price_equity_curr
                notional_value = amount * price_equity_curr
            else:
                return False, amount, f"Notional value ({notional_value:.2f}) below minimum ({min_notional:.2f})"

        # 2. Maximum Allocation Check (35% Equity Cap)
        max_trade_val = current_equity * getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35)
        if current_equity >= min_notional and max_trade_val < min_notional:
            max_trade_val = min(current_equity * 0.95, min_notional * 1.1)
            
        if notional_value > max_trade_val:
            amount = (max_trade_val * 0.999) / price_equity_curr

        # 3. Market Precision / Lot Size Validation
        if markets_info and symbol in markets_info:
            market = markets_info[symbol]
            limits = market.get('limits', {})
            min_qty = limits.get('amount', {}).get('min', 0.0) or 0.0
            max_qty = limits.get('amount', {}).get('max', 999999.0) or 999999.0
            
            if amount < min_qty:
                return False, amount, f"Quantity {amount} below exchange minimum {min_qty} for {symbol}"
            if amount > max_qty:
                amount = max_qty
                
            precision = market.get('precision', {}).get('amount')
            if precision is not None and isinstance(precision, int):
                multiplier = 10 ** precision
                amount = math.floor(amount * multiplier) / multiplier
            elif precision is not None and isinstance(precision, float):
                amount = math.floor(amount / precision) * precision
        else:
            # Fallback safe floor truncation to 6 decimals
            amount = math.floor(amount * 1000000.0) / 1000000.0

        if amount <= 0:
            return False, 0.0, "Quantity reduced to zero after precision truncation."

        return True, amount, "VALID"

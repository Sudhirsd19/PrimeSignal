"""Deterministic paper-trading cash accounting helpers.

All prices and PnL are expressed in USDT.  ``conversion_rate`` converts the
USDT cash flow into the configured paper wallet currency (for example INR).
"""

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class PaperEntryResult:
    quantity: float
    fill_price: float
    fee_usdt: float
    cash_debit: float


@dataclass(frozen=True)
class PaperExitResult:
    gross_pnl_usdt: float
    exit_fee_usdt: float
    cash_credit: float


def simulate_paper_entry(
    side: str,
    requested_qty: float,
    signal_price: float,
    balance_cash: float,
    current_equity_cash: float,
    max_alloc_pct: float,
    slippage_pct: float,
    fee_rate: float,
    conversion_rate: float = 1.0,
    min_paper_cash: float = 1.0,
) -> Optional[PaperEntryResult]:
    """Return a fill and exact wallet debit, or ``None`` when no fill is affordable."""
    side = str(side).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    values = (requested_qty, signal_price, balance_cash, current_equity_cash,
              max_alloc_pct, slippage_pct, fee_rate, conversion_rate, min_paper_cash)
    if not all(isfinite(float(v)) for v in values):
        return None
    if requested_qty <= 0 or signal_price <= 0 or balance_cash <= 0 or current_equity_cash <= 0:
        return None
    if conversion_rate <= 0 or fee_rate < 0 or max_alloc_pct <= 0 or slippage_pct < 0:
        return None

    if side == "BUY":
        fill_price = signal_price * (1.0 + slippage_pct)
    else:
        fill_price = signal_price * (1.0 - slippage_pct)
    if fill_price <= 0 or not isfinite(fill_price):
        return None

    max_cash = current_equity_cash * max_alloc_pct
    if max_cash < min_paper_cash and balance_cash >= min_paper_cash:
        target_cash = balance_cash
    else:
        target_cash = min(balance_cash, max_cash)
    if target_cash < min_paper_cash:
        return None

    cash_per_unit = fill_price * conversion_rate * (1.0 + fee_rate)
    quantity = min(requested_qty, target_cash / cash_per_unit)
    if quantity <= 0 or not isfinite(quantity):
        return None

    notional_usdt = quantity * fill_price
    fee_usdt = notional_usdt * fee_rate
    cash_debit = (notional_usdt + fee_usdt) * conversion_rate
    if cash_debit <= 0 or cash_debit > balance_cash + 1e-9:
        return None

    return PaperEntryResult(
        quantity=float(quantity),
        fill_price=float(fill_price),
        fee_usdt=float(fee_usdt),
        cash_debit=float(cash_debit),
    )


def simulate_paper_exit(
    side: str,
    quantity: float,
    entry_price: float,
    exit_price: float,
    fee_rate: float,
    conversion_rate: float = 1.0,
) -> PaperExitResult:
    """Return gross PnL, exit fee and exact wallet credit for a paper close."""
    side = str(side).upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    values = (quantity, entry_price, exit_price, fee_rate, conversion_rate)
    if not all(isfinite(float(v)) for v in values):
        raise ValueError("paper exit inputs must be finite")
    if quantity < 0 or entry_price <= 0 or exit_price <= 0 or fee_rate < 0 or conversion_rate <= 0:
        raise ValueError("invalid paper exit inputs")

    if side == "LONG":
        gross_pnl_usdt = quantity * (exit_price - entry_price)
        gross_cash_return_usdt = quantity * exit_price
    else:
        gross_pnl_usdt = quantity * (entry_price - exit_price)
        gross_cash_return_usdt = quantity * entry_price + gross_pnl_usdt

    exit_fee_usdt = quantity * exit_price * fee_rate
    cash_credit = (gross_cash_return_usdt - exit_fee_usdt) * conversion_rate
    return PaperExitResult(
        gross_pnl_usdt=float(gross_pnl_usdt),
        exit_fee_usdt=float(exit_fee_usdt),
        cash_credit=float(cash_credit),
    )


def net_leg_pnl_pct(net_pnl_usdt: float, quantity: float, entry_price: float) -> float:
    """Return net PnL percentage on the leg's entry notional."""
    denominator = quantity * entry_price
    if denominator <= 0 or not isfinite(float(net_pnl_usdt)):
        return 0.0
    return float(net_pnl_usdt) / denominator * 100.0

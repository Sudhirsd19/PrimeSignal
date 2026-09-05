"""
PrimeSignal Advanced Performance Analytics Module
Calculates institutional-grade metrics from trade history:
- Profit Factor, Sharpe Ratio, Sortino Ratio
- Max Drawdown, Avg Win/Loss, Expectancy
- Consecutive Win/Loss streaks
"""
import math
from typing import Any


def calculate_advanced_metrics(trades: list[dict[str, Any]], initial_capital: float = 10000.0) -> dict[str, Any]:
    """Calculates institutional-grade performance metrics from trade history."""
    if not trades:
        return _empty_metrics()

    pnl_values: list[float] = []
    for t in trades:
        pnl = t.get("total_pnl_net") if t.get("total_pnl_net") is not None else (t.get("pnl_usdt") if t.get("pnl_usdt") is not None else t.get("pnl", 0))
        pnl_values.append(float(pnl or 0))

    if not pnl_values:
        return _empty_metrics()

    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]
    breakevens = [p for p in pnl_values if p == 0]

    total_trades = len(pnl_values)
    total_wins = len(wins)
    total_losses = len(losses)
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

    # Profit Factor = Sum(Gains) / Sum(Losses)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # Average Win / Average Loss / Win-Loss Ratio
    avg_win = (gross_profit / total_wins) if total_wins > 0 else 0.0
    avg_loss = (gross_loss / total_losses) if total_losses > 0 else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)

    # Expectancy = (Win% x Avg Win) - (Loss% x Avg Loss)
    win_pct = total_wins / total_trades if total_trades > 0 else 0
    loss_pct = total_losses / total_trades if total_trades > 0 else 0
    expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)

    # P2-1 FIX: Equity curve for drawdown calculation, anchored to starting capital baseline at t=0
    # to ensure early/initial losses are accurately captured rather than zeroed out.
    current_equity = float(initial_capital)
    equity_curve: list[float] = [current_equity]
    for p in pnl_values:
        current_equity += p
        equity_curve.append(current_equity)

    max_drawdown, max_drawdown_pct = _calculate_max_drawdown(equity_curve)
    sharpe_ratio = _calculate_sharpe_ratio(pnl_values, annualization_factor=math.sqrt(6 * 365))
    sortino_ratio = _calculate_sortino_ratio(pnl_values, annualization_factor=math.sqrt(6 * 365))
    max_consec_wins, max_consec_losses, current_streak, current_streak_type = _calculate_streaks(pnl_values)

    best_trade = max(pnl_values) if pnl_values else 0.0
    worst_trade = min(pnl_values) if pnl_values else 0.0
    net_pnl = sum(pnl_values)

    return {
        "total_trades": total_trades, "total_wins": total_wins, "total_losses": total_losses,
        "total_breakevens": len(breakevens), "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if not math.isinf(profit_factor) else "inf",
        "gross_profit": round(gross_profit, 4), "gross_loss": round(gross_loss, 4),
        "net_pnl": round(net_pnl, 4), "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4),
        "win_loss_ratio": round(win_loss_ratio, 3) if not math.isinf(win_loss_ratio) else "inf",
        "expectancy": round(expectancy, 4), "best_trade": round(best_trade, 4),
        "worst_trade": round(worst_trade, 4), "max_drawdown": round(max_drawdown, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 2), "sharpe_ratio": round(sharpe_ratio, 3),
        "sortino_ratio": round(sortino_ratio, 3) if not math.isinf(sortino_ratio) else "inf",
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses, "current_streak": current_streak,
        "current_streak_type": current_streak_type,
    }


def _empty_metrics() -> dict[str, Any]:
    """Returns zeroed metrics when no trades exist."""
    return {
        "total_trades": 0, "total_wins": 0, "total_losses": 0, "total_breakevens": 0,
        "win_rate": 0.0, "profit_factor": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        "net_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "win_loss_ratio": 0.0,
        "expectancy": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
        "max_drawdown": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        "current_streak": 0, "current_streak_type": "none",
    }


def _calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Max peak-to-trough drawdown from equity curve."""
    if not equity_curve:
        return 0.0, 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        drawdown = peak - val
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = (drawdown / peak * 100.0) if peak > 0 else 0.0
    return max_dd, max_dd_pct


def _calculate_sharpe_ratio(returns: list[float], annualization_factor: float = 1.0) -> float:
    """Sharpe Ratio = Mean(Returns) / Std(Returns) x Annualization Factor"""
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    if std_dev == 0:
        return 0.0
    return (mean_return / std_dev) * annualization_factor


def _calculate_sortino_ratio(returns: list[float], annualization_factor: float = 1.0) -> float:
    """Sortino Ratio = Mean(Returns) / Downside Std x Annualization Factor"""
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    downside_returns = [r for r in returns if r < 0]
    if not downside_returns:
        return float("inf") if mean_return > 0 else 0.0
    downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
    downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0
    if downside_std == 0:
        return 0.0
    return (mean_return / downside_std) * annualization_factor


def _calculate_streaks(pnl_values: list[float]) -> tuple[int, int, int, str]:
    """Calculates max consecutive wins/losses and current streak."""
    max_wins = 0
    max_losses = 0
    current = 0
    current_type = "none"
    for pnl in pnl_values:
        if pnl > 0:
            if current_type == "win":
                current += 1
            else:
                current = 1
                current_type = "win"
            max_wins = max(max_wins, current)
        elif pnl < 0:
            if current_type == "loss":
                current += 1
            else:
                current = 1
                current_type = "loss"
            max_losses = max(max_losses, current)
    return max_wins, max_losses, current, current_type

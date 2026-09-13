"""
PrimeSignal Walk-Forward Harness
================================

H-05 FIX — `optimize_winrate.py` and `find_profitable_edge.py` grid-searched
parameters over a single window and then reported the best in-sample
combination as the "edge". That is textbook curve fitting: with ~24,000
candidate configurations, the top in-sample result is a statistical artefact,
and two of those winners had already been hardcoded into Config.

Every sweep now runs through this harness, which:

  * splits the data into an in-sample (tuning) window and a held-out
    out-of-sample (validation) window per pair,
  * evaluates EVERY candidate on BOTH windows,
  * ranks by out-of-sample performance, never by in-sample,
  * reports the multiplicity (how many configs were tried) and the
    in-sample -> out-of-sample degradation, because that gap is the
    overfitting tell.

The harness reuses the single live-parity BacktestEngine, so a sweep can never
again "test" logic that differs from what the bot executes.
"""

from __future__ import annotations

from typing import Any, Callable

# NOTE: this is a library used by optimize_winrate.py / find_profitable_edge.py.
# Run those from the project root (they bootstrap sys.path); this file is not a
# standalone entry point.
from config import Config
from risk.risk_manager import RiskManager
from backtester.backtester import BacktestEngine


def split_pair(ltf: list, train_fraction: float = 0.70):
    """Splits a candle list into (in-sample, out-of-sample)."""
    split = int(len(ltf) * train_fraction)
    return ltf[:split], ltf[split:]


def _fmt_pf(pf) -> str:
    try:
        if pf == float("inf"):
            return "inf"
        return f"{float(pf):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def run_grid(
    strategy_factory: Callable[[], Any],
    datasets: list[tuple[str, list, list]],
    param_grid: list[dict[str, Any]],
    initial_balance: float = 1000.0,
    ml_factory: Callable[[], Any] | None = None,
    train_fraction: float = 0.70,
    min_is_trades: int = 5,
) -> list[dict[str, Any]]:
    """Evaluates the full grid across every dataset and aggregates IS vs OOS.

    Aggregation is trade-count weighted, and a configuration must clear
    `min_is_trades` in-sample before it is even considered — otherwise the
    ranking is dominated by configs that simply never traded.
    """
    rows: list[dict[str, Any]] = []
    total = len(param_grid)
    for idx, params in enumerate(param_grid, 1):
        agg = {
            "overrides": dict(params),
            "is_trades": 0, "is_wins": 0, "is_net": 0.0, "is_start": 0.0, "is_dd": 0.0,
            "is_gross_win": 0.0, "is_gross_loss": 0.0, "is_fees": 0.0,
            "oos_trades": 0, "oos_wins": 0, "oos_net": 0.0, "oos_start": 0.0, "oos_dd": 0.0,
            "oos_gross_win": 0.0, "oos_gross_loss": 0.0, "oos_fees": 0.0,
        }
        for _, htf, ltf in datasets:
            train, test = split_pair(ltf, train_fraction)
            for label, candles, key in (("is", train, "is"), ("oos", test, "oos")):
                if len(candles) < 200:
                    continue
                ml = ml_factory() if ml_factory else None
                engine = BacktestEngine(
                    strategy=strategy_factory(),
                    risk_manager=RiskManager(),
                    ml_confirmator=ml,
                    overrides=params,
                )
                res = engine.run(htf, candles, initial_balance=initial_balance)
                if not res:
                    continue
                agg[f"{key}_trades"] += res["total_trades"]
                agg[f"{key}_wins"] += res["wins"]
                agg[f"{key}_net"] += res["net_profit"]
                agg[f"{key}_start"] += res["initial_balance"]
                agg[f"{key}_fees"] += res["total_fees"]
                agg[f"{key}_gross_win"] += sum(t["pnl_usdt"] for t in res["trade_list"] if t["pnl_usdt"] > 0)
                agg[f"{key}_gross_loss"] += abs(sum(t["pnl_usdt"] for t in res["trade_list"] if t["pnl_usdt"] < 0))
                agg[f"{key}_dd"] = min(agg[f"{key}_dd"], res["max_drawdown_pct"])
        agg["is_win_rate"] = (agg["is_wins"] / agg["is_trades"] * 100.0) if agg["is_trades"] else 0.0
        agg["oos_win_rate"] = (agg["oos_wins"] / agg["oos_trades"] * 100.0) if agg["oos_trades"] else 0.0
        agg["is_return_pct"] = (agg["is_net"] / agg["is_start"] * 100.0) if agg["is_start"] else 0.0
        agg["oos_return_pct"] = (agg["oos_net"] / agg["oos_start"] * 100.0) if agg["oos_start"] else 0.0
        agg["is_pf"] = _profit_factor_from_agg(agg, "is")
        agg["oos_pf"] = _profit_factor_from_agg(agg, "oos")
        agg["degradation_pct"] = agg["is_return_pct"] - agg["oos_return_pct"]
        rows.append(agg)
        if idx % 5 == 0 or idx == total:
            print(f"    evaluated {idx}/{total} configurations...")

    rows = [r for r in rows if r["is_trades"] >= min_is_trades]
    # RANK BY OUT-OF-SAMPLE. Never by in-sample.
    rows.sort(key=lambda r: (r["oos_net"], r["oos_trades"]), reverse=True)
    return rows


def _profit_factor_from_agg(agg: dict, key: str) -> float:
    """Profit factor from accumulated gross win / gross loss (trade weighted)."""
    gross_win = float(agg.get(f"{key}_gross_win", 0.0))
    gross_loss = float(agg.get(f"{key}_gross_loss", 0.0))
    if gross_loss > 0:
        return gross_win / gross_loss
    return float("inf") if gross_win > 0 else 0.0


def print_grid(rows: list[dict[str, Any]], param_keys: list[str], label: str, top: int = 12) -> None:
    """Prints an IS vs OOS table and an explicit overfitting verdict."""
    print("\n" + "=" * 100)
    print(f"  {label}")
    print("=" * 100)
    if not rows:
        print("  No configuration produced enough in-sample trades to be evaluated.")
        return

    head = " | ".join(f"{k[:18]:<18}" for k in param_keys)
    print(f"  {head} | {'IS trd':>6} | {'IS win%':>7} | {'IS ret%':>8} | {'OOS trd':>7} | {'OOS win%':>8} | {'OOS ret%':>8} | {'decay':>7}")
    print("  " + "-" * 96)
    for r in rows[:top]:
        vals = " | ".join(f"{str(r['overrides'].get(k))[:18]:<18}" for k in param_keys)
        print(f"  {vals} | {r['is_trades']:>6} | {r['is_win_rate']:>6.1f}% | {r['is_return_pct']:>+7.2f}% | "
              f"{r['oos_trades']:>7} | {r['oos_win_rate']:>7.1f}% | {r['oos_return_pct']:>+7.2f}% | {r['degradation_pct']:>+6.2f}%")

    print("  " + "-" * 96)
    print(f"  Configurations that cleared the in-sample trade floor: {len(rows)}")
    print("  Ranked by OUT-OF-SAMPLE net PnL. In-sample columns are shown only to expose decay.")
    best = rows[0]
    print(f"\n  BEST OUT-OF-SAMPLE CONFIG : {best['overrides']}")
    print(f"    in-sample  : {best['is_trades']} trades, {best['is_win_rate']:.1f}% win, {best['is_return_pct']:+.2f}% return")
    print(f"    out-sample : {best['oos_trades']} trades, {best['oos_win_rate']:.1f}% win, {best['oos_return_pct']:+.2f}% return")
    print(f"    degradation: {best['degradation_pct']:+.2f} percentage points of return lost out-of-sample")

    print("\n  OVERFITTING VERDICT")
    print("  " + "-" * 96)
    if best["oos_trades"] < 20:
        print("    NOT ENOUGH OUT-OF-SAMPLE TRADES to draw any conclusion.")
    elif best["oos_return_pct"] <= 0:
        print("    FAIL — even the best out-of-sample configuration loses money.")
        print("    Nothing in this grid should be promoted into Config.")
    elif best["degradation_pct"] > max(5.0, abs(best["is_return_pct"]) * 0.5):
        print("    SUSPECT — out-of-sample return is profitable but collapsed by more than half")
        print("    versus in-sample. Strong overfitting signal; treat with suspicion.")
    else:
        print("    PLAUSIBLE — the ranking holds out-of-sample. Still confirm on a")
        print("    different market regime before promoting anything into Config.")
    print("  " + "=" * 96)


def grid_over(param_space: dict[str, list], fixed: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Builds the cartesian product of a parameter space as a list of override dicts."""
    from itertools import product

    keys = list(param_space.keys())
    combos = list(product(*(param_space[k] for k in keys)))
    out = []
    for combo in combos:
        params = dict(fixed or {})
        params.update(dict(zip(keys, combo)))
        out.append(params)
    return out

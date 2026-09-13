"""
PrimeSignal Parameter Sweep (walk-forward)
==========================================

H-05 FIX. The original script swept 336 exit/sizing combinations across ONE
window, sorted them by WIN RATE, and printed the top 10 as "highest win rate
configurations (80% target)". Problems:

  * sorting by win rate ignores profit factor, expectancy and drawdown — a
    config that wins 80% of the time at 0.2R and loses 20% at 3R is a losing
    system that looks great by win rate alone;
  * there was no held-out window, so the winners were in-sample curve fits;
  * it ran a bespoke simulator with different exit logic from the shipped bot.

This version sweeps the REAL Config knobs through the shared live-parity engine
and the shared walk-forward harness, which ranks strictly by OUT-OF-SAMPLE
performance and prints the in-sample -> out-of-sample decay (the overfitting
tell). A configuration is never promoted into Config automatically.
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402
from strategies.multi_timeframe import MultiTimeframeSMCStrategy  # noqa: E402
from backtester.walkforward import grid_over, run_grid, print_grid  # noqa: E402

# Kept small on purpose: every extra dimension multiplies the number of
# in-sample coincidences. 3 x 3 x 3 = 27 configurations, not 336.
PARAM_SPACE = {
    "TSL_ACTIVATION_R": [0.3, 0.55, 0.8],
    "MIN_RISK_REWARD_RATIO": [1.0, 1.5, 2.0],
    "RISK_REWARD_RATIO": [2.0, 2.5, 3.0],
}
PARAM_KEYS = list(PARAM_SPACE.keys())


def load_datasets() -> list[tuple[str, list, list]]:
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "LTC/USDT", "LINK/USDT"]
    datasets: list[tuple[str, list, list]] = []
    for pair in pairs:
        clean = pair.replace("/", "_")
        for ltf_name, htf_name in (
            (f"{clean}_15m_30d.json", f"{clean}_1h_30d.json"),
            (f"{clean}_15m_1600_2w.json", f"{clean}_1h_600_2w.json"),
        ):
            ltf_path = os.path.join(PROJECT_ROOT, "data", ltf_name)
            htf_path = os.path.join(PROJECT_ROOT, "data", htf_name)
            if os.path.exists(ltf_path) and os.path.exists(htf_path):
                with open(ltf_path, "r", encoding="utf-8") as f:
                    ltf = json.load(f)
                with open(htf_path, "r", encoding="utf-8") as f:
                    htf = json.load(f)
                datasets.append((pair, htf, ltf))
                break
    return datasets


def main() -> int:
    print("=" * 100)
    print("  PRIMESIGNAL WALK-FORWARD PARAMETER SWEEP")
    print("=" * 100)
    print(f"  Risk config hash : {Config.get_risk_config_hash()[:16]}...")
    print(f"  Venue / shorts   : {Config.EXCHANGE_TYPE} / {'ENABLED' if Config.venue_supports_short() else 'DISABLED'}")
    print(f"  In-sample        : first 70% of each pair's window")
    print(f"  Out-of-sample    : last 30% (ranked on this only)")

    datasets = load_datasets()
    if not datasets:
        print("\n  ERROR: no data found under data/.")
        print("  Fetch it first:  python fetch_backtest_data.py --days 30")
        return 2

    print(f"  Datasets         : {len(datasets)} pairs")
    grid = grid_over(PARAM_SPACE)
    print(f"  Configurations   : {len(grid)}")
    print("-" * 100)

    rows = run_grid(
        strategy_factory=MultiTimeframeSMCStrategy,
        datasets=datasets,
        param_grid=grid,
        initial_balance=1000.0,
    )
    print_grid(rows, PARAM_KEYS, "WALK-FORWARD SWEEP RESULTS", top=12)

    out_path = os.path.join(PROJECT_ROOT, "walkforward_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"param_space": PARAM_SPACE, "results": rows}, f, indent=2, default=str)
        print(f"\n  Full grid written to {os.path.basename(out_path)} (inspect before changing any Config value).")
    except OSError as exc:
        print(f"\n  Could not write results file: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

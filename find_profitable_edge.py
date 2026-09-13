"""
PrimeSignal Filter Edge Finder (walk-forward)
=============================================

H-05 FIX. The original script pulled 8 pairs, grid-searched
ADX x TP1 x TP2 x VWAP-alignment x volume-filter over a single window, sorted
purely by net PnL, and printed the top 15 as "profitable edges". It then
re-implemented its own entry/exit logic (EMA cross + rejection candles, 50/30/20
scaling, BE at 1.0R) which does not match the shipped bot at all — so whatever
it "found" was never evidence about the product.

This version tests the ACTUAL Config filter switches that the live bot reads,
through the shared live-parity engine and the shared walk-forward harness.
It answers the only question that matters: does a given filter improve
OUT-OF-SAMPLE results?

Nothing is written back into Config — promotion stays a human decision.
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

# Real toggles the live engine honours, one dimension at a time plus the
# two most consequential interactions.
BASE = {"ADX_MIN_THRESHOLD": 25.0}

PARAM_SPACE = {
    "ADX_MIN_THRESHOLD": [20.0, 25.0, 30.0],
    "ENABLE_WEEKEND_FILTER": [True, False],
    "ENABLE_BB_SQUEEZE_FILTER": [True, False],
    "ENABLE_STRUCTURAL_EXIT": [True, False],
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
    print("  PRIMESIGNAL FILTER EDGE FINDER (walk-forward, live-parity engine)")
    print("=" * 100)
    print(f"  Risk config hash : {Config.get_risk_config_hash()[:16]}...")
    print(f"  Venue / shorts   : {Config.EXCHANGE_TYPE} / {'ENABLED' if Config.venue_supports_short() else 'DISABLED'}")
    print("  Method           : in-sample = first 70%, out-of-sample = last 30%")
    print("  Ranking          : OUT-OF-SAMPLE net PnL only")

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
    print_grid(rows, PARAM_KEYS, "FILTER CONFIGURATION RESULTS", top=16)

    if rows:
        shipped_defaults = {
            "ADX_MIN_THRESHOLD": 25.0,
            "ENABLE_WEEKEND_FILTER": True,
            "ENABLE_BB_SQUEEZE_FILTER": True,
            "ENABLE_STRUCTURAL_EXIT": False,
        }
        baseline = next((r for r in rows if r["overrides"] == shipped_defaults), None)
        print("\n  HOW TO READ THIS")
        print("  " + "-" * 96)
        print("    Compare each row against the shipped defaults:")
        print(f"      ADX_MIN_THRESHOLD={Config.ADX_MIN_THRESHOLD}, "
              f"ENABLE_WEEKEND_FILTER={Config.ENABLE_WEEKEND_FILTER}, "
              f"ENABLE_BB_SQUEEZE_FILTER={Config.ENABLE_BB_SQUEEZE_FILTER}, "
              f"ENABLE_STRUCTURAL_EXIT={Config.ENABLE_STRUCTURAL_EXIT}")
        if baseline:
            print(f"    Shipped-default row out-of-sample: "
                  f"{baseline['oos_trades']} trades, {baseline['oos_win_rate']:.1f}% win, "
                  f"{baseline['oos_return_pct']:+.2f}% return")
        print("    A filter only earns its place if it improves the OUT-OF-SAMPLE row.")

    out_path = os.path.join(PROJECT_ROOT, "edge_finder_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"param_space": PARAM_SPACE, "results": rows}, f, indent=2, default=str)
        print(f"\n  Full grid written to {os.path.basename(out_path)}.")
    except OSError as exc:
        print(f"\n  Could not write results file: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

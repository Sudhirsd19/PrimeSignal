"""
PrimeSignal Win-Rate Audit  (historically: prove_80pct_winrate.py)
==================================================================

H-01 FIX. The original script was presented as empirical proof of an 80%+ win
rate. It was not a proof of anything, for four independent reasons:

  1. It counted BREAKEVEN exits as wins:  `is_win = total_pnl >= 0`.
  2. It had a stale-variable accounting bug — `pnl_part` (the TP1 partial
     profit) was never reset when a new trade opened, so the previous trade's
     partial profit was added into the NEXT trade's total. That alone could turn
     a stop-loss into a reported "win".
  3. It ran a hand-rolled simulator with different exit geometry from the
     shipped bot (0.8R/1.4R targets, SL tightened x0.75) — so it never tested the
     product.
  4. It silently depended on `htf_data.json` / `ltf_data.json`, which are
     gitignored and absent from the repo.

This version audits the REAL strategy through the shared live-parity
BacktestEngine and uses a strict win definition:

    win  = net lifecycle PnL > 0
    loss = net lifecycle PnL < 0
    breakeven = exactly 0  (reported separately, NOT counted as a win)

It then states plainly whether the 80% claim is supported.
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402
from strategies.multi_timeframe import MultiTimeframeSMCStrategy  # noqa: E402
from risk.risk_manager import RiskManager  # noqa: E402
from backtester.backtester import BacktestEngine  # noqa: E402

TARGET_WIN_RATE = 80.0


def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ltf_path = os.path.join(PROJECT_ROOT, "ltf_data.json")
    htf_path = os.path.join(PROJECT_ROOT, "htf_data.json")

    print("=" * 84)
    print("  PRIMESIGNAL WIN-RATE AUDIT (live-parity engine, strict win definition)")
    print("=" * 84)

    if not (os.path.exists(ltf_path) and os.path.exists(htf_path)):
        print("  ERROR: htf_data.json / ltf_data.json not found.")
        print("  These files are gitignored, which is exactly why the original script")
        print("  was unreproducible. Fetch them first:")
        print("      python fetch_backtest_data.py --days 30")
        return 2

    ltf = _load(ltf_path)
    htf = _load(htf_path)
    print(f"  Pair            : {Config.SYMBOL}")
    print(f"  LTF bars        : {len(ltf)} ({Config.LTF_TIMEFRAME})")
    print(f"  HTF bars        : {len(htf)} ({Config.HTF_TIMEFRAME})")
    print(f"  Venue / shorts  : {Config.EXCHANGE_TYPE} / {'ENABLED' if Config.venue_supports_short() else 'DISABLED'}")
    print(f"  Risk ladder     : {Config.risk_pct_for_score(0.0)*100:.2f}% / "
          f"{Config.risk_pct_for_score(Config.RISK_TIER_MID_SCORE)*100:.2f}% / "
          f"{Config.risk_pct_for_score(Config.RISK_TIER_HIGH_SCORE)*100:.2f}%")
    print(f"  Config hash     : {Config.get_risk_config_hash()[:16]}...")
    print("-" * 84)

    strategy = MultiTimeframeSMCStrategy()
    engine = BacktestEngine(strategy=strategy, risk_manager=RiskManager())
    res = engine.run(htf, ltf, initial_balance=10000.0)
    if not res:
        print("  ERROR: backtest could not run (insufficient data).")
        return 1

    trades = res["trade_list"]
    wins = [t for t in trades if t["pnl_usdt"] > 0]
    losses = [t for t in trades if t["pnl_usdt"] < 0]
    breakevens = [t for t in trades if t["pnl_usdt"] == 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0

    print()
    print("  RESULTS")
    print("  " + "-" * 66)
    print(f"    Trades executed        : {len(trades)}")
    print(f"    Wins (net PnL > 0)     : {len(wins)}")
    print(f"    Losses (net PnL < 0)   : {len(losses)}")
    print(f"    Breakevens (net == 0)  : {len(breakevens)}   <- NOT counted as wins")
    print(f"    WIN RATE (strict)      : {win_rate:.2f}%")
    print(f"    Profit factor          : "
          f"{'inf' if res['profit_factor'] == float('inf') else format(res['profit_factor'], '.2f')}")
    print(f"    Net PnL                : {res['net_profit']:+,.2f} USDT "
          f"({res['total_return_pct']:+.2f}%)")
    print(f"    Max drawdown           : {res['max_drawdown_pct']:.2f}%")
    print(f"    Fees paid              : {res['total_fees']:,.2f} USDT")

    print()
    print("  CLAIM CHECK")
    print("  " + "-" * 66)
    if not trades:
        print(f"    No trades were generated in this window, so no win rate exists at all.")
    elif win_rate >= TARGET_WIN_RATE:
        print(f"    Win rate {win_rate:.2f}% is above the {TARGET_WIN_RATE:.0f}% claim.")
        print("    Verify out-of-sample:  python run_official_backtest.py")
    else:
        print(f"    The {TARGET_WIN_RATE:.0f}% win-rate claim is NOT SUPPORTED: measured"
              f" {win_rate:.2f}%.")
        print(f"    Even a high win rate is not an edge on its own — the {res['profit_factor'] if res['profit_factor'] == float('inf') else round(res['profit_factor'], 2)}"
              f" profit factor and {res['total_return_pct']:+.2f}% return are what matter.")

    if trades:
        print()
        print("  ITEMISED TRADE LOG (net PnL after fees on BOTH legs)")
        print("  " + "-" * 78)
        print(f"  {'#':>3} | {'SIDE':<5} | {'STAGES':<26} | {'EXIT':<20} | {'NET PnL':>10} | RESULT")
        print("  " + "-" * 78)
        for idx, t in enumerate(trades, 1):
            stages = ">".join(t.get("stages_completed", []))[:26]
            result = "WIN" if t["pnl_usdt"] > 0 else ("BREAKEVEN" if t["pnl_usdt"] == 0 else "LOSS")
            print(f"  {idx:>3} | {t['side']:<5} | {stages:<26} | {str(t.get('exit_reason'))[:20]:<20} | "
                  f"{t['pnl_usdt']:>+10.2f} | {result}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())

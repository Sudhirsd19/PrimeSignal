"""
PrimeSignal Official Backtest
=============================

C-03 FIX — this runner was completely broken. It read `res['win_rate_trades']`
and `res['total_net_profit']`, neither of which BacktestEngine.calculate_metrics()
has ever returned, so it raised KeyError on the first symbol. It also:

  * averaged per-pair win rates without weighting by trade count (H-05),
  * reported a single in-sample window with no holdout (H-05),
  * never stated which configuration or data window produced the numbers (H-02).

It now:
  * fetches missing data via fetch_backtest_data.py (or tells you exactly how),
  * runs the live-parity BacktestEngine,
  * reports IN-SAMPLE and OUT-OF-SAMPLE (last 30%) results side by side,
  * aggregates wins/losses over actual trade counts,
  * prints the pinned window and risk-config hash from backtest_manifest.json.
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

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "backtest_manifest.json")
OOS_FRACTION = 0.30


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def resolve_pair_files(pair: str, manifest: dict, ltf_tf: str, htf_tf: str):
    """Locate committed/fetched data for a pair, newest layout first."""
    clean = pair.replace("/", "_")
    candidates = []
    entry = (manifest.get("pairs") or {}).get(pair) or {}
    if entry.get("status") == "ok":
        candidates.append((os.path.join(PROJECT_ROOT, entry["ltf_file"]), os.path.join(PROJECT_ROOT, entry["htf_file"])))
    days_tag = f"{manifest.get('days', 30):g}d"
    candidates.append((os.path.join(DATA_DIR, f"{clean}_{ltf_tf}_{days_tag}.json"), os.path.join(DATA_DIR, f"{clean}_{htf_tf}_{days_tag}.json")))
    candidates.append((os.path.join(DATA_DIR, f"{clean}_{ltf_tf}_30d.json"), os.path.join(DATA_DIR, f"{clean}_{htf_tf}_30d.json")))
    candidates.append((os.path.join(DATA_DIR, f"{clean}_15m_1600_2w.json"), os.path.join(DATA_DIR, f"{clean}_1h_600_2w.json")))
    for ltf_path, htf_path in candidates:
        if os.path.exists(ltf_path) and os.path.exists(htf_path):
            return ltf_path, htf_path
    return None, None


def aggregate(results: list[dict]) -> dict:
    """Portfolio aggregation weighted by actual trade counts (H-05 fix)."""
    trades = [t for r in results for t in r.get("trade_list", [])]
    start = sum(r["initial_balance"] for r in results)
    final = sum(r["final_balance"] for r in results)
    wins = [t for t in trades if t["pnl_usdt"] > 0]
    losses = [t for t in trades if t["pnl_usdt"] < 0]
    be = [t for t in trades if t["pnl_usdt"] == 0]
    gross_p = sum(t["pnl_usdt"] for t in wins)
    gross_l = abs(sum(t["pnl_usdt"] for t in losses))
    return {
        "pairs": len(results),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakevens": len(be),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "start": start,
        "final": final,
        "net_profit": final - start,
        "return_pct": ((final - start) / start * 100.0) if start else 0.0,
        "profit_factor": (gross_p / gross_l) if gross_l > 0 else (float("inf") if gross_p > 0 else 0.0),
        "fees": sum(t.get("total_fees", 0.0) for t in trades),
        "max_drawdown_pct": min((r["max_drawdown_pct"] for r in results), default=0.0),
    }


def print_table(title: str, agg: dict):
    pf = agg["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"\n  {title}")
    print("  " + "-" * 66)
    print(f"    Pairs tested          : {agg['pairs']}")
    print(f"    Trades                : {agg['trades']}  ({agg['wins']}W / {agg['losses']}L / {agg['breakevens']}BE)")
    print(f"    Win rate              : {agg['win_rate']:.2f}%   (wins / all trades, breakevens excluded)")
    print(f"    Net profit            : {agg['net_profit']:+,.2f} USDT on {agg['start']:,.2f} start")
    print(f"    Return                : {agg['return_pct']:+.2f}%")
    print(f"    Profit factor         : {pf_str}")
    print(f"    Total fees paid       : {agg['fees']:,.2f} USDT")
    print(f"    Worst pair drawdown   : {agg['max_drawdown_pct']:.2f}%")


def main():
    ltf_tf = Config.LTF_TIMEFRAME
    htf_tf = Config.HTF_TIMEFRAME
    manifest = load_manifest()

    print("=" * 84)
    print("  PRIMESIGNAL OFFICIAL BACKTEST (live-parity engine)")
    print("=" * 84)
    print(f"  Risk config hash : {Config.get_risk_config_hash()[:16]}...")
    print(f"  Venue            : {Config.TRADING_VENUE} / {Config.EXCHANGE_TYPE} "
          f"(shorts {'ENABLED' if Config.venue_supports_short() else 'DISABLED'})")
    print(f"  Fee rate         : {Config.FEE_RATE*100:.4f}% per side")
    print(f"  Risk ladder      : "
          f"{Config.risk_pct_for_score(0.0)*100:.2f}% / "
          f"{Config.risk_pct_for_score(Config.RISK_TIER_MID_SCORE)*100:.2f}% / "
          f"{Config.risk_pct_for_score(Config.RISK_TIER_HIGH_SCORE)*100:.2f}%")
    if manifest.get("window"):
        print(f"  Data window      : {manifest['window']['start_utc']} -> {manifest['window']['end_utc']}")
    else:
        print("  Data window      : UNKNOWN (no backtest_manifest.json — results are not reproducible)")
        print("                     run:  python fetch_backtest_data.py")

    pairs = list((manifest.get("pairs") or {}).keys()) or Config.SUPPORTED_SYMBOLS
    print(f"  Pairs            : {len(pairs)}")
    print("=" * 84)

    # ── load ────────────────────────────────────────────────────────────────
    loaded: list[tuple[str, list, list]] = []
    missing: list[str] = []
    for pair in pairs:
        ltf_path, htf_path = resolve_pair_files(pair, manifest, ltf_tf, htf_tf)
        if not ltf_path:
            missing.append(pair)
            continue
        with open(ltf_path, "r", encoding="utf-8") as f:
            ltf = json.load(f)
        with open(htf_path, "r", encoding="utf-8") as f:
            htf = json.load(f)
        loaded.append((pair, ltf, htf))

    if not loaded:
        print("\n  ERROR: no cached data found in data/ and no pinned manifest.")
        print("  Fetch it first (writes backtest_manifest.json + data/*.json):")
        print("      python fetch_backtest_data.py --days 30")
        print("  Then re-run this script.")
        return 2

    if missing:
        print(f"\n  NOTE: {len(missing)} pair(s) had no local data and were skipped: {', '.join(missing[:8])}")

    # ── run ─────────────────────────────────────────────────────────────────
    strategy = MultiTimeframeSMCStrategy()
    risk_mgr = RiskManager()
    engine = BacktestEngine(strategy=strategy, risk_manager=risk_mgr)

    full_results: list[dict] = []
    in_sample: list[dict] = []
    out_sample: list[dict] = []

    print(f"\n  {'PAIR':<12} | {'TRADES':>6} | {'WIN%':>6} | {'RETURN%':>8} | {'MAXDD%':>7} | {'PF':>6}")
    print("  " + "-" * 62)
    for pair, ltf, htf in loaded:
        res = engine.run(htf, ltf, initial_balance=1000.0)
        if not res:
            continue
        full_results.append(res)
        pf = res["profit_factor"]
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"  {pair:<12} | {res['total_trades']:>6} | {res['win_rate']:>5.1f}% | "
              f"{res['total_return_pct']:>+7.2f}% | {res['max_drawdown_pct']:>6.2f}% | {pf_str:>6}")

        # H-05 FIX: split each pair into in-sample (first 70%) and
        # out-of-sample (last 30%) so a regime-dependent result is visible.
        split = int(len(ltf) * (1.0 - OOS_FRACTION))
        if split > 200 and (len(ltf) - split) > 100:
            is_res = engine.run(htf, ltf[:split], initial_balance=1000.0)
            oos_res = engine.run(htf, ltf[split:], initial_balance=1000.0)
            if is_res:
                in_sample.append(is_res)
            if oos_res:
                out_sample.append(oos_res)

    if not full_results:
        print("\n  No results produced (insufficient data per pair).")
        return 1

    agg_full = aggregate(full_results)
    print_table("FULL WINDOW (in-sample + out-of-sample, informational only)", agg_full)

    if in_sample and out_sample:
        agg_is = aggregate(in_sample)
        agg_oos = aggregate(out_sample)
        print_table("IN-SAMPLE (first 70% of each window)", agg_is)
        print_table("OUT-OF-SAMPLE (last 30% of each window) — THE ONLY ONE THAT COUNTS", agg_oos)

        print("\n  VERDICT")
        print("  " + "-" * 66)
        oos_pf = agg_oos["profit_factor"]
        oos_pf_val = float("inf") if oos_pf == float("inf") else oos_pf
        if agg_oos["trades"] < 30:
            print("    INCONCLUSIVE — fewer than 30 out-of-sample trades. Not a sample size.")
        elif agg_oos["net_profit"] > 0 and oos_pf_val > 1.0:
            print("    PASS — out-of-sample window is net profitable with PF > 1.0.")
            print("    Still verify across a different market regime before sizing up.")
        else:
            print("    FAIL — the strategy does not hold up out-of-sample.")
            print("    Do NOT treat the full-window numbers above as evidence of edge.")

        print("\n  Note: drawdown is reported per pair; a portfolio-level drawdown would be")
        print("  larger. Slippage is modelled at 50% of MAX_SLIPPAGE_PCT and fees at FEE_RATE.")
    else:
        print("\n  Out-of-sample split skipped (not enough bars per pair).")

    print("\n" + "=" * 84)
    print("  Reproduction: python fetch_backtest_data.py  ->  python run_official_backtest.py")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())

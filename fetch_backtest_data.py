"""
PrimeSignal Backtest Data Fetcher
=================================

H-02 FIX — the repository contained a dozen "performance proof" scripts that all
depended on `data/*.json` and `htf_data.json` / `ltf_data.json`, none of which are
in the repo (they are gitignored). Nobody could reproduce a single reported
number, and the scripts simply crashed on a fresh checkout.

This script produces exactly the files those runners expect AND writes a
`backtest_manifest.json` that pins:

  * the exact UTC start/end of every fetched window
  * the pairs, timeframes and bar counts actually retrieved
  * the canonical risk-config hash (Config.get_risk_config_hash())

With the manifest you can re-fetch the identical window later:

    python fetch_backtest_data.py --end 2026-09-13T00:00:00Z
    python run_official_backtest.py

Stdlib only (urllib), so it works before any dependency is installed.

Outputs
-------
    data/<PAIR>_15m_<days>d.json      execution-frame candles
    data/<PAIR>_1h_<days>d.json       trend-frame candles
    ltf_data.json / htf_data.json     BTC copy for legacy scripts
    backtest_manifest.json            reproducibility record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "backtest_manifest.json")

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}


def _http_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": "PrimeSignal/2.6"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_klines(pair: str, interval: str, start_ms: int, end_ms: int, pause: float = 0.12) -> list:
    """Paginates Binance klines forward from start_ms to end_ms (1000/request cap)."""
    raw_symbol = pair.replace("/", "").upper()
    step = INTERVAL_MS[interval]
    out: dict[int, list] = {}
    cursor = start_ms
    guard = 0
    while cursor < end_ms and guard < 200:
        guard += 1
        url = (
            f"{BINANCE_KLINES}?symbol={raw_symbol}&interval={interval}"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        try:
            rows = _http_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            print(f"    ! {pair} {interval}: {type(exc).__name__}: {exc}")
            break
        if not rows:
            break
        for r in rows:
            out[int(r[0])] = [int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
        newest = int(rows[-1][0])
        nxt = newest + step
        if nxt <= cursor:
            break
        cursor = nxt
        if len(rows) < 1000:
            break
        time.sleep(pause)
    return [out[k] for k in sorted(out)]


def main():
    ap = argparse.ArgumentParser(description="Fetch reproducible PrimeSignal backtest data")
    ap.add_argument("--days", type=float, default=30.0, help="how many days of history (default 30)")
    ap.add_argument("--end", type=str, default=None, help="pin the window end (ISO-8601 UTC, e.g. 2026-09-13T00:00:00Z)")
    ap.add_argument("--pairs", type=str, default=None, help="comma-separated pairs (default: Config.SUPPORTED_SYMBOLS)")
    ap.add_argument("--ltf", type=str, default=None, help="execution timeframe (default: Config.LTF_TIMEFRAME)")
    ap.add_argument("--htf", type=str, default=None, help="trend timeframe (default: Config.HTF_TIMEFRAME)")
    args = ap.parse_args()

    ltf_tf = args.ltf or Config.LTF_TIMEFRAME
    htf_tf = args.htf or Config.HTF_TIMEFRAME
    pairs = [p.strip() for p in (args.pairs.split(",") if args.pairs else Config.SUPPORTED_SYMBOLS) if p.strip()]

    if args.end:
        end_dt = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)
    # Align the window end to a bar boundary so the window is stable across runs.
    end_ms = int(end_dt.timestamp() * 1000) // INTERVAL_MS["15m"] * INTERVAL_MS["15m"]
    start_ms = end_ms - int(args.days * 24 * 60 * 60 * 1000)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 84)
    print("  PrimeSignal Reproducible Data Fetch")
    print("=" * 84)
    print(f"  Window   : {datetime.fromtimestamp(start_ms/1000, timezone.utc).isoformat()} "
          f"-> {datetime.fromtimestamp(end_ms/1000, timezone.utc).isoformat()}")
    print(f"  Pairs    : {len(pairs)}")
    print(f"  Frames   : LTF {ltf_tf} | HTF {htf_tf}")
    print("-" * 84)

    days_tag = f"{args.days:g}d"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start_utc": datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat(),
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
        "timeframes": {"ltf": ltf_tf, "htf": htf_tf},
        "days": args.days,
        "risk_config_hash": Config.get_risk_config_hash(),
        "fee_rate": Config.FEE_RATE,
        "worst_case_slippage_pct": Config.MAX_SLIPPAGE_PCT,
        "venue": {
            "trading_venue": Config.TRADING_VENUE,
            "exchange_type": Config.EXCHANGE_TYPE,
            "supports_short": Config.venue_supports_short(),
        },
        "pairs": {},
        "source": BINANCE_KLINES,
    }

    ok_pairs = 0
    for pair in pairs:
        clean = pair.replace("/", "_")
        ltf = fetch_klines(pair, ltf_tf, start_ms, end_ms)
        htf = fetch_klines(pair, htf_tf, start_ms, end_ms)
        if not ltf or not htf:
            print(f"  {pair:<12} SKIPPED (no data returned)")
            manifest["pairs"][pair] = {"status": "no_data"}
            continue

        ltf_path = os.path.join(DATA_DIR, f"{clean}_{ltf_tf}_{days_tag}.json")
        htf_path = os.path.join(DATA_DIR, f"{clean}_{htf_tf}_{days_tag}.json")
        with open(ltf_path, "w", encoding="utf-8") as f:
            json.dump(ltf, f)
        with open(htf_path, "w", encoding="utf-8") as f:
            json.dump(htf, f)

        manifest["pairs"][pair] = {
            "status": "ok",
            "ltf_file": os.path.relpath(ltf_path, PROJECT_ROOT),
            "htf_file": os.path.relpath(htf_path, PROJECT_ROOT),
            "ltf_bars": len(ltf),
            "htf_bars": len(htf),
            "ltf_first_utc": datetime.fromtimestamp(ltf[0][0] / 1000, timezone.utc).isoformat(),
            "ltf_last_utc": datetime.fromtimestamp(ltf[-1][0] / 1000, timezone.utc).isoformat(),
            "sha256_ltf": hashlib.sha256(json.dumps(ltf).encode()).hexdigest()[:16],
        }
        ok_pairs += 1
        print(f"  {pair:<12} ltf={len(ltf):<5} htf={len(htf):<5} -> {os.path.basename(ltf_path)}")

        # Legacy aliases for backtest.py / prove_80pct_winrate.py / optimize_winrate.py
        if pair == "BTC/USDT":
            with open(os.path.join(PROJECT_ROOT, "ltf_data.json"), "w", encoding="utf-8") as f:
                json.dump(ltf, f)
            with open(os.path.join(PROJECT_ROOT, "htf_data.json"), "w", encoding="utf-8") as f:
                json.dump(htf, f)

    print("-" * 84)
    print(f"  {ok_pairs}/{len(pairs)} pairs written to data/")
    if ok_pairs == 0:
        # Fail-closed. A manifest is a *reproducibility record*: publishing one for a
        # window that contains zero usable bars would make a later run look pinned and
        # reproducible when it is not, and the old code printed "Manifest: ..." even
        # when nothing was fetched. Remove any stale record and exit non-zero.
        if os.path.exists(MANIFEST_PATH):
            try:
                os.remove(MANIFEST_PATH)
            except OSError:
                pass
        print("\n  ERROR: no data fetched. Check network access to api.binance.com.")
        print("  backtest_manifest.json was NOT written — there is nothing to pin.")
        return 1

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: backtest_manifest.json (config hash {manifest['risk_config_hash'][:16]}...)")
    print("  Re-run any backtest with:  python run_official_backtest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

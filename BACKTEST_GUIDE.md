# PrimeSignal Canonical Backtesting & Verification Guide

## 1. Canonical Backtesting Engine
The sole institutional ground-truth backtest engine for PrimeSignal is:
* **Engine Implementation:** [`backtester/backtester.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/PrimeSignal/backtester/backtester.py) (`METRICS_VERSION = "2.6-live-parity"`)
* **Official CLI Runner:** [`run_official_backtest.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/PrimeSignal/run_official_backtest.py)

---

## 2. Live-Parity Guarantees
The canonical engine strictly mirrors the live execution path to eliminate backtest-live discrepancy:

| Live Control | Canonical Backtest Parity (`backtester/backtester.py`) |
| :--- | :--- |
| **Risk Ladder** | Directly queries `Config.risk_pct_for_score(score)` based on conviction score. |
| **Scale-Out Exits** | Models exact TP1 (`TP1_SCALE_OUT_PCT`), TP2 (`TP2_REMAINING_SCALE_PCT`), and 4.0R runner. |
| **Breakeven & Trailing** | Enforces `DYNAMIC_BE_BUFFER_PCT (+0.35R floor)` and ATR trailing stops activated only post-TP1. |
| **Daily Drawdown Breakers**| Enforces `MAX_DAILY_LOSS_PCT`, `ENABLE_DAILY_PROFIT_LOCK`, `MAX_DAILY_TRADES`, and consecutive loss limits. |
| **Execution Costs** | Applies exchange `FEE_RATE` across entry and each scale-out tranche, plus slippage penalties. |
| **Holdout Separation** | Enforces out-of-sample holdout validation (`OOS_FRACTION = 0.30`) with trade-weighted metric aggregation. |
| **Win Definition** | A trade is counted as a WIN strictly if net lifecycle PnL > 0 after all fees. Breakeven exits are categorized separately. |

---

## 3. How to Run the Canonical Backtest
To execute the official institutional backtest:
```bash
# In-sample + Out-of-sample (30% holdout) over pinned window:
python run_official_backtest.py --days 30
```

---

## 4. Legacy Script Status
Root-level one-off scripts (`run_30d_backtest.py`, `backtest_1month.py`, `backtest_portfolio.py`, etc.) represent historical point-in-time explorations. For forensic audits and production certification, always refer to [`run_official_backtest.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/PrimeSignal/run_official_backtest.py).

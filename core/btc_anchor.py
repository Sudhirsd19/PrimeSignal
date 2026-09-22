"""
BTC Master Correlation Anchor Engine
====================================
Evaluates cross-asset directional beta and macro/micro momentum of Bitcoin (BTC)
to protect altcoin trades from market-wide flash flushes, false breakouts, and macro divergence.

Institutional Principle:
Altcoins exhibit 0.75-0.90 correlation to BTC on lower timeframes.
Taking an altcoin LONG when BTC is in an active micro-flush or HTF downtrend
is the single largest driver of retail stop-outs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Any
import pandas as pd
import numpy as np

from strategies.indicators import calculate_ema, calculate_rsi, calculate_vwap, prepare_dataframe


class BTCAnchorEngine:
    """Institutional Cross-Asset Correlation Engine anchoring altcoins to BTC."""

    def __init__(
        self,
        flash_drop_threshold: float = 0.006,   # 0.6% single-candle drop
        flash_pump_threshold: float = 0.006,   # 0.6% single-candle pump
        rolling_3bar_drop_threshold: float = 0.010,  # 1.0% over 3 candles (45m)
        rolling_3bar_pump_threshold: float = 0.010,  # 1.0% over 3 candles (45m)
    ):
        self.flash_drop_threshold = flash_drop_threshold
        self.flash_pump_threshold = flash_pump_threshold
        self.rolling_3bar_drop_threshold = rolling_3bar_drop_threshold
        self.rolling_3bar_pump_threshold = rolling_3bar_pump_threshold

    def evaluate_btc_confluence(
        self,
        symbol: str,
        signal: str,
        btc_ltf_data: Any,
        btc_htf_data: Optional[Any] = None,
        eval_closed_only: bool = True,
    ) -> Tuple[bool, str, float]:
        """
        Evaluates whether an altcoin setup aligns with current BTC macro/micro dynamics.

        Args:
            symbol: Target symbol (e.g. 'ETH/USDT', 'SOL/USDT', 'BTC/USDT').
            signal: 'BUY' or 'SELL'.
            btc_ltf_data: BTC 15m candles (DataFrame or list of dicts/lists).
            btc_htf_data: BTC 1h candles (DataFrame or list of dicts/lists, optional).
            eval_closed_only: If True, evaluates completed closed candles for full strategy parity.

        Returns:
            Tuple of (allowed: bool, reason: str, score_adjustment: float)
        """
        clean_sym = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
        if clean_sym.startswith("BTC") and not clean_sym.startswith("BTCST"):
            # Self-reference: BTC doesn't filter itself via correlation
            return True, f"Primary Asset ({symbol})", 0.0

        if signal not in ("BUY", "SELL"):
            return True, "No directional signal", 0.0

        # Prepare LTF DataFrame
        if btc_ltf_data is None:
            return True, "BTC LTF data unavailable (fail-open)", 0.0

        btc_ltf_df = prepare_dataframe(btc_ltf_data) if isinstance(btc_ltf_data, list) else btc_ltf_data
        if not isinstance(btc_ltf_df, pd.DataFrame) or len(btc_ltf_df) < 10:
            return True, "Insufficient BTC LTF candles (fail-open)", 0.0

        # ── 1. Micro Flash Flush / Pump Guard (15m frame) ──
        # Flash drop/pump circuit breaker inspects the latest live candle (iloc[-1])
        # as well as the last closed candle (iloc[-2]) if eval_closed_only to catch sudden crashes in real time.
        check_candles = [btc_ltf_df.iloc[-1]]
        if eval_closed_only and len(btc_ltf_df) >= 2:
            check_candles.append(btc_ltf_df.iloc[-2])

        for last_candle in check_candles:
            c_open = float(last_candle.get('open', 0.0))
            c_close = float(last_candle.get('close', 0.0))

            if c_open > 0:
                single_bar_ret = (c_close - c_open) / c_open
                
                # Flash Drop check
                if signal == "BUY" and single_bar_ret <= -self.flash_drop_threshold:
                    drop_pct = abs(single_bar_ret) * 100.0
                    return False, f"BTC Flash Flush active (-{drop_pct:.2f}% in 15m): Altcoin longs blocked", 0.0

                # Flash Pump check (for shorts)
                if signal == "SELL" and single_bar_ret >= self.flash_pump_threshold:
                    pump_pct = single_bar_ret * 100.0
                    return False, f"BTC Flash Surge active (+{pump_pct:.2f}% in 15m): Altcoin shorts blocked", 0.0

        # ── 2. Rolling Multi-Candle Momentum (3-bar / 45m) ──
        ltf_eval_idx = -2 if (eval_closed_only and len(btc_ltf_df) >= 2) else -1
        c_close = float(btc_ltf_df['close'].iloc[ltf_eval_idx])
        lookback_bars = 4 if ltf_eval_idx == -1 else 5
        if len(btc_ltf_df) >= lookback_bars:
            p_3bars_ago_idx = -lookback_bars
            p_3bars_ago = float(btc_ltf_df['close'].iloc[p_3bars_ago_idx])
            if p_3bars_ago > 0:
                rolling_ret = (c_close - p_3bars_ago) / p_3bars_ago
                if signal == "BUY" and rolling_ret <= -self.rolling_3bar_drop_threshold:
                    drop_pct = abs(rolling_ret) * 100.0
                    return False, f"BTC Sustained Selloff (-{drop_pct:.2f}% over 45m): Altcoin longs blocked", 0.0
                if signal == "SELL" and rolling_ret >= self.rolling_3bar_pump_threshold:
                    pump_pct = rolling_ret * 100.0
                    return False, f"BTC Sustained Rally (+{pump_pct:.2f}% over 45m): Altcoin shorts blocked", 0.0

        # ── 3. HTF Macro Confluence (1h frame) ──
        # Evaluate on the last closed 1H candle to prevent intra-bar oscillation
        btc_htf_trend = "NEUTRAL"
        if btc_htf_data is not None:
            btc_htf_df = prepare_dataframe(btc_htf_data) if isinstance(btc_htf_data, list) else btc_htf_data
            if isinstance(btc_htf_df, pd.DataFrame) and len(btc_htf_df) >= 50:
                htf_eval_idx = -2 if len(btc_htf_df) >= 2 else -1
                htf_close = float(btc_htf_df['close'].iloc[htf_eval_idx])
                htf_ema50 = float(calculate_ema(btc_htf_df, 50).iloc[htf_eval_idx])
                htf_ema200 = float(calculate_ema(btc_htf_df, min(200, len(btc_htf_df))).iloc[htf_eval_idx]) if len(btc_htf_df) >= 100 else htf_ema50

                if htf_close > htf_ema50 and htf_ema50 >= htf_ema200:
                    btc_htf_trend = "BULLISH"
                elif htf_close < htf_ema50 and htf_ema50 <= htf_ema200:
                    btc_htf_trend = "BEARISH"

        # LTF RSI on last closed candle
        ltf_rsi_series = calculate_rsi(btc_ltf_df, 14)
        ltf_rsi_idx = -2 if len(btc_ltf_df) >= 2 else -1
        btc_ltf_rsi = float(ltf_rsi_series.iloc[ltf_rsi_idx]) if not ltf_rsi_series.empty and not math.isnan(ltf_rsi_series.iloc[ltf_rsi_idx]) else 50.0

        # Hard Macro Trend Filter: Never buy altcoins during a confirmed BTC 1h Downtrend
        if signal == "BUY" and btc_htf_trend == "BEARISH":
            return False, f"BTC HTF Downtrend (1h Trend: BEARISH, RSI: {btc_ltf_rsi:.1f}): Altcoin long blocked", 0.0

        if signal == "SELL" and btc_htf_trend == "BULLISH":
            return False, f"BTC HTF Uptrend (1h Trend: BULLISH, RSI: {btc_ltf_rsi:.1f}): Altcoin short blocked", 0.0

        # Confluence Score Boost for fully aligned trades
        if signal == "BUY" and btc_htf_trend == "BULLISH" and btc_ltf_rsi >= 50.0:
            return True, f"Strong BTC Bullish Confluence (1h Trend: BULL, RSI: {btc_ltf_rsi:.1f})", 0.5

        if signal == "SELL" and btc_htf_trend == "BEARISH" and btc_ltf_rsi <= 50.0:
            return True, f"Strong BTC Bearish Confluence (1h Trend: BEAR, RSI: {btc_ltf_rsi:.1f})", 0.5

        return True, f"BTC Neutral/Aligned (Trend: {btc_htf_trend}, RSI: {btc_ltf_rsi:.1f})", 0.0

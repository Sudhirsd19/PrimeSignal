import time
from collections import deque
from config import Config

class LeadLagArbitrageEngine:
    """
    Cross-Asset Lead-Lag Latency Momentum Arbitrage Engine.
    Monitors high-frequency BTC velocity to front-run delayed altcoin propagation waves.
    """
    def __init__(self):
        # Ring buffers storing (timestamp_seconds, price)
        self.price_history = {sym: deque(maxlen=120) for sym in Config.SUPPORTED_SYMBOLS}
        self.last_arbitrage_signal_time = {sym: 0 for sym in Config.SUPPORTED_SYMBOLS}

    def record_tick(self, symbol: str, price: float):
        """Record live tick with current timestamp."""
        if price and price > 0:
            now = time.time()
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=120)
            self.price_history[symbol].append((now, price))

    def evaluate_lead_lag(self, alt_symbol: str) -> dict:
        """
        Evaluates whether a high-velocity BTC move provides a front-running opportunity on the altcoin.
        """
        if not getattr(Config, 'ENABLE_LEAD_LAG_ARBITRAGE', True):
            return {'signal': 'NONE', 'btc_velocity_pct': 0.0, 'alt_velocity_pct': 0.0, 'edge_pct': 0.0}

        if alt_symbol == "BTC/USDT":
            return {'signal': 'NONE', 'btc_velocity_pct': 0.0, 'alt_velocity_pct': 0.0, 'edge_pct': 0.0}

        btc_ticks = self.price_history.get("BTC/USDT", deque())
        alt_ticks = self.price_history.get(alt_symbol, deque())

        if len(btc_ticks) < 10 or len(alt_ticks) < 10:
            return {'signal': 'NONE', 'btc_velocity_pct': 0.0, 'alt_velocity_pct': 0.0, 'edge_pct': 0.0}

        now = time.time()
        # Look back 30 seconds
        window = 30.0

        # Find BTC price 30s ago
        btc_now = btc_ticks[-1][1]
        btc_past = next((p for t, p in btc_ticks if now - t <= window), btc_ticks[0][1])
        btc_velocity = (btc_now - btc_past) / btc_past if btc_past > 0 else 0.0

        # Find Altcoin price 30s ago
        alt_now = alt_ticks[-1][1]
        alt_past = next((p for t, p in alt_ticks if now - t <= window), alt_ticks[0][1])
        alt_velocity = (alt_now - alt_past) / alt_past if alt_past > 0 else 0.0

        impulse_thresh = getattr(Config, 'BTC_IMPULSE_VELOCITY_PCT', 0.0030) # 0.30% in 30s
        lag_max_thresh = getattr(Config, 'ALT_LAG_MAX_REACTION_PCT', 0.0008) # 0.08%

        signal = 'NONE'
        edge_pct = 0.0

        # Cooldown check (minimum 5 mins between lead-lag signals on same pair)
        if now - self.last_arbitrage_signal_time.get(alt_symbol, 0) > 300:
            # Bullish Front-run: BTC surged violently (> +0.30%), but Alt is lagging (< +0.08%)
            if btc_velocity >= impulse_thresh and alt_velocity <= lag_max_thresh:
                signal = 'BULLISH_LEAD_LAG_ARBITRAGE'
                edge_pct = (btc_velocity - alt_velocity) * 100.0
                self.last_arbitrage_signal_time[alt_symbol] = now

            # Bearish Front-run: BTC dumped violently (< -0.30%), but Alt has not yet dumped (> -0.08%)
            elif btc_velocity <= -impulse_thresh and alt_velocity >= -lag_max_thresh:
                signal = 'BEARISH_LEAD_LAG_ARBITRAGE'
                edge_pct = (alt_velocity - btc_velocity) * 100.0
                self.last_arbitrage_signal_time[alt_symbol] = now

        return {
            'signal': signal,
            'btc_velocity_pct': round(btc_velocity * 100.0, 3),
            'alt_velocity_pct': round(alt_velocity * 100.0, 3),
            'edge_pct': round(edge_pct, 2)
        }

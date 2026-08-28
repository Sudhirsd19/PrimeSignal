import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from config import Config
from strategies.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap

# Shared feature column list — single source of truth for all methods
FEATURE_COLS = [
    'rsi', 'atr_pct', 'ema_ratio', 'vwap_dist', 'vol_ratio',
    'body_ratio', 'wick_ratio', 'close_position',
    'volume_delta', 'price_momentum', 'hour_sin', 'hour_cos'
]


class MLSignalConfirmator:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.is_trained = False

    @staticmethod
    def _compute_features(data):
        """
        Computes all 12 features on a DataFrame in-place and returns the feature column names.
        Expects `data` to already have OHLCV columns and a DatetimeIndex.
        """
        # Core indicators
        data['rsi'] = calculate_rsi(data, Config.RSI_PERIOD)
        atr = calculate_atr(data, Config.ATR_PERIOD)
        data['atr_pct'] = atr / data['close']

        ema_short = calculate_ema(data, Config.SHORT_EMA)
        ema_long = calculate_ema(data, Config.LONG_EMA)
        data['ema_ratio'] = ema_short / ema_long

        vwap = calculate_vwap(data)
        data['vwap_dist'] = (data['close'] - vwap) / vwap

        # Volume relative to its 20-period average
        data['vol_ratio'] = data['volume'] / data['volume'].rolling(20).mean().replace(0, 1e-9)

        # Candle structure features
        candle_range = data['high'] - data['low']
        body = abs(data['close'] - data['open'])
        data['body_ratio'] = body / candle_range.replace(0, 1e-9)

        # Wick ratio: lower wick relative to range (buying pressure indicator)
        lower_wick = data[['close', 'open']].min(axis=1) - data['low']
        data['wick_ratio'] = lower_wick / candle_range.replace(0, 1e-9)

        # Close position within the range (0 = closed at low, 1 = closed at high)
        data['close_position'] = (data['close'] - data['low']) / candle_range.replace(0, 1e-9)

        # Volume delta: volume change vs prior candle
        data['volume_delta'] = data['volume'].pct_change().fillna(0).clip(-5, 5)

        # Price momentum: 5-bar rate of change
        data['price_momentum'] = data['close'].pct_change(5).fillna(0).clip(-0.1, 0.1)

        # Cyclical time-of-day encoding (captures session effects)
        hour = data.index.hour + data.index.minute / 60.0
        data['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

        return FEATURE_COLS

    def prepare_features(self, df):
        """
        Creates technical and structural features for the ML model.
        Returns:
            X: pandas DataFrame of features
            y: pandas Series of binary labels (1 = price goes up in next 5 bars, 0 = otherwise)
        """
        # Copy to avoid side-effects
        data = df.copy()

        feature_cols = self._compute_features(data)

        # Drop rows where indicators are not fully computed yet
        data.dropna(subset=feature_cols, inplace=True)

        # Define target label: Did price go up over the next 5 candles?
        future_lookahead = 5
        data['future_return'] = data['close'].shift(-future_lookahead) / data['close'] - 1.0

        # Binary target: 1 if positive return, 0 if flat/negative return
        data['target'] = (data['future_return'] > 0.001).astype(int)

        # Drop rows where future_return is NaN (the last future_lookahead rows)
        # NOTE: Cannot dropna on 'target' because NaN > 0.001 → False → 0 (never NaN)
        clean_data = data.dropna(subset=['future_return'])

        X = clean_data[feature_cols]
        y = clean_data['target']

        return X, y

    def train(self, df):
        """
        Trains the GradientBoosting model on historical data.
        """
        if len(df) < 200:
            print("WARNING: Insufficient data to train ML model. Needs at least 200 candles.")
            return False

        try:
            print(f"[ML] Preparing training features from {len(df)} candles (12-feature GradientBoosting)...")
            X, y = self.prepare_features(df)

            if len(X) < 100:
                print("WARNING: Too few clean data rows after feature extraction.")
                return False

            print(f"[ML] Training GradientBoosting model on {len(X)} samples...")
            self.model.fit(X, y)
            self.is_trained = True

            # Print simple feature importances
            raw_importances = getattr(self.model, "feature_importances_", [])
            importances = [float(imp) for imp in raw_importances]
            print("[ML] Trained Successfully! Feature Importances:")
            sorted_feats = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
            for name, imp in sorted_feats:
                print(f"  {name}: {imp:.3f}")
            return True
        except Exception as e:
            print(f"ERROR training ML model: {e}")
            return False

    def _extract_feature_row(self, df):
        """
        Extracts a single feature row for the last completed candle (index -2).
        Returns a DataFrame with one row ready for model prediction.
        """
        data = df.copy()
        feature_cols = self._compute_features(data)

        row = {}
        for col in feature_cols:
            val = data[col].iloc[-1]
            row[col] = val if not (isinstance(val, float) and np.isnan(val)) else 0.0

        return pd.DataFrame([row])

    def confirm_signal(self, df, signal_type):
        """
        Confirms if the generated signal is validated by ML prediction.

        Args:
            df: Current DataFrame
            signal_type: "BUY" or "SELL"

        Returns:
            confirmed: bool (True to execute, False to block)
            probability: float (ML model confidence score)
        """
        if not self.is_trained or len(df) < 50:
            return False, 0.0

        try:
            feature_row = self._extract_feature_row(df)

            # Predict probability of price going up (target = 1)
            prob_up = self.model.predict_proba(feature_row)[0][1]

            # Confirmation thresholds
            if signal_type == "BUY":
                # For buy signal, we want high probability of price going up
                confirmed = prob_up >= Config.ML_CONFIRMATION_THRESHOLD
                return confirmed, prob_up
            elif signal_type == "SELL":
                # For sell signal, we want high probability of price going down (low prob of going up)
                prob_down = 1.0 - prob_up
                confirmed = prob_down >= Config.ML_CONFIRMATION_THRESHOLD
                return confirmed, prob_down

            return False, 0.5
        except Exception as e:
            print(f"WARNING: Error running ML signal confirmation, blocking trade for safety: {e}")
            return False, 0.0

    def predict_bias(self, df):
        """
        Predicts the bullish probability of the last completed candle (index -2).
        Returns a float between 0.0 and 1.0 representing the bullish bias.
        """
        if not self.is_trained or len(df) < 50:
            return 0.5

        try:
            feature_row = self._extract_feature_row(df)
            return float(self.model.predict_proba(feature_row)[0][1])
        except Exception as e:
            print(f"WARNING: Error predicting ML bias: {e}")
            return 0.5

    def predict_next_candle(self, df):
        """
        Predicts the color and probability of the UPCOMING 5m candle.
        Returns:
            dict: {'color': 'GREEN' or 'RED', 'confidence_pct': float, 'bullish_prob': float}
        """
        if len(df) < 20:
            return {'color': 'GREEN', 'confidence_pct': 50.0, 'bullish_prob': 50.0}

        try:
            # 1. Base ML model probability if trained
            ml_prob = self.predict_bias(df)

            # 2. Immediate candle momentum & order flow
            last = df.iloc[-1]
            body_return = (last['close'] - last['open']) / last['open'] if last['open'] > 0 else 0.0
            candle_range = last['high'] - last['low']
            lower_wick = min(last['close'], last['open']) - last['low']
            upper_wick = last['high'] - max(last['close'], last['open'])

            wick_bias = 0.0
            if candle_range > 0:
                wick_bias = (lower_wick - upper_wick) / candle_range

            # Composite momentum score (scaled between -0.20 and +0.20)
            mom_factor = np.clip(body_return * 40.0 + wick_bias * 0.15, -0.20, 0.20)

            # Final Bullish Probability
            final_prob = np.clip(ml_prob + mom_factor, 0.15, 0.85)
            if np.isnan(final_prob):
                final_prob = 0.50

            color = "GREEN" if final_prob >= 0.50 else "RED"
            confidence = final_prob if color == "GREEN" else (1.0 - final_prob)

            return {
                'color': color,
                'confidence_pct': round(float(confidence * 100), 1),
                'bullish_prob': round(float(final_prob * 100), 1)
            }
        except Exception as e:
            print(f"WARNING: Error predicting next candle color: {e}")
            return {'color': 'GREEN', 'confidence_pct': 50.0, 'bullish_prob': 50.0}

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import Config
from strategies.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap

class MLSignalConfirmator:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False

    def prepare_features(self, df):
        """
        Creates technical and structural features for the ML model.
        Returns:
            X: pandas DataFrame of features
            y: pandas Series of binary labels (1 = price goes up in next 5 bars, 0 = otherwise)
        """
        # Copy to avoid side-effects
        data = df.copy()
        
        # 1. Calculate features
        data['rsi'] = calculate_rsi(data, Config.RSI_PERIOD)
        atr = calculate_atr(data, Config.ATR_PERIOD)
        data['atr_pct'] = atr / data['close']  # Normalize ATR as percentage of price
        
        ema_short = calculate_ema(data, Config.SHORT_EMA)
        ema_long = calculate_ema(data, Config.LONG_EMA)
        data['ema_ratio'] = ema_short / ema_long
        
        vwap = calculate_vwap(data)
        data['vwap_dist'] = (data['close'] - vwap) / vwap
        
        # Volume relative to its 20-period average
        data['vol_ratio'] = data['volume'] / data['volume'].rolling(20).mean().replace(0, 1e-9)
        
        # Feature columns
        feature_cols = ['rsi', 'atr_pct', 'ema_ratio', 'vwap_dist', 'vol_ratio']
        
        # Drop rows where indicators are not fully computed yet
        data.dropna(subset=feature_cols, inplace=True)
        
        # 2. Define target label: Did price go up over the next 5 candles?
        future_lookahead = 5
        data['future_return'] = data['close'].shift(-future_lookahead) / data['close'] - 1.0
        
        # Binary target: 1 if positive return, 0 if flat/negative return
        data['target'] = (data['future_return'] > 0.001).astype(int)
        
        # Drop rows where target is NaN (the last future_lookahead rows)
        clean_data = data.dropna(subset=['target'])
        
        X = clean_data[feature_cols]
        y = clean_data['target']
        
        return X, y

    def train(self, df):
        """
        Trains the RandomForest model on historical data.
        """
        if len(df) < 200:
            print("WARNING: Insufficient data to train ML model. Needs at least 200 candles.")
            return False
            
        try:
            print(f"[ML] Preparing training features from {len(df)} candles...")
            X, y = self.prepare_features(df)
            
            if len(X) < 100:
                print("WARNING: Too few clean data rows after feature extraction.")
                return False
                
            print(f"[ML] Training Random Forest model on {len(X)} samples...")
            self.model.fit(X, y)
            self.is_trained = True
            
            # Print simple feature importances
            importances = self.model.feature_importances_
            print("[ML] Trained Successfully! Feature Importances:")
            for name, imp in zip(X.columns, importances):
                print(f"  {name}: {imp:.3f}")
            return True
        except Exception as e:
            print(f"ERROR training ML model: {e}")
            return False

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
        if not self.is_trained:
            # Fail-safe: Block trades if model is not yet trained.
            # This prevents unfiltered entries during warm-up.
            print("WARNING: ML model not yet trained. Blocking signal as a safety measure.")
            return False, 0.0
            
        try:
            # Extract latest completed candle (index -2) features
            data = df.copy()
            data['rsi'] = calculate_rsi(data, Config.RSI_PERIOD)
            atr = calculate_atr(data, Config.ATR_PERIOD)
            data['atr_pct'] = atr / data['close']
            
            ema_short = calculate_ema(data, Config.SHORT_EMA)
            ema_long = calculate_ema(data, Config.LONG_EMA)
            data['ema_ratio'] = ema_short / ema_long
            
            vwap = calculate_vwap(data)
            data['vwap_dist'] = (data['close'] - vwap) / vwap
            data['vol_ratio'] = data['volume'] / data['volume'].rolling(20).mean().replace(0, 1e-9)
            
            # Get feature row for last completed candle
            feature_row = pd.DataFrame([{
                'rsi': data['rsi'].iloc[-2],
                'atr_pct': data['atr_pct'].iloc[-2],
                'ema_ratio': data['ema_ratio'].iloc[-2],
                'vwap_dist': data['vwap_dist'].iloc[-2],
                'vol_ratio': data['vol_ratio'].iloc[-2]
            }])
            
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
            data = df.copy()
            data['rsi'] = calculate_rsi(data, Config.RSI_PERIOD)
            atr = calculate_atr(data, Config.ATR_PERIOD)
            data['atr_pct'] = atr / data['close']
            
            ema_short = calculate_ema(data, Config.SHORT_EMA)
            ema_long = calculate_ema(data, Config.LONG_EMA)
            data['ema_ratio'] = ema_short / ema_long
            
            vwap = calculate_vwap(data)
            data['vwap_dist'] = (data['close'] - vwap) / vwap
            data['vol_ratio'] = data['volume'] / data['volume'].rolling(20).mean().replace(0, 1e-9)
            
            feature_row = pd.DataFrame([{
                'rsi': data['rsi'].iloc[-2],
                'atr_pct': data['atr_pct'].iloc[-2],
                'ema_ratio': data['ema_ratio'].iloc[-2],
                'vwap_dist': data['vwap_dist'].iloc[-2],
                'vol_ratio': data['vol_ratio'].iloc[-2]
            }])
            
            return float(self.model.predict_proba(feature_row)[0][1])
        except Exception as e:
            print(f"WARNING: Error predicting ML bias: {e}")
            return 0.5


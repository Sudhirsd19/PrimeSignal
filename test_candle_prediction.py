import json
import os
import sys

if sys.platform == 'win32':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
        getattr(sys.stderr, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except (AttributeError, Exception):
        pass

sys.path.insert(0, os.path.dirname(__file__))

from ml.confirmation import MLSignalConfirmator
from strategies.indicators import prepare_dataframe

def test_candle_prediction():
    print("=" * 70, flush=True)
    print("TESTING CANDLE PREDICTION ENGINE (ml/confirmation.py)", flush=True)
    print("=" * 70, flush=True)

    with open("ltf_data.json", "r") as f:
        raw_candles = json.load(f)

    df_full = prepare_dataframe(raw_candles)
    print(f"Total candles loaded: {len(df_full)}", flush=True)

    # Train on first 500 candles
    df_train = df_full.iloc[:500]
    df_test = df_full.iloc[500:800] # Test on next 300 candles

    model = MLSignalConfirmator()
    trained = model.train(df_train)
    print(f"ML Model Trained: {trained}", flush=True)

    total_tested = 0
    correct_count = 0
    high_conf_total = 0
    high_conf_correct = 0

    green_pred = 0
    red_pred = 0
    green_actual = 0
    red_actual = 0

    print("\nEvaluating out-of-sample next-candle forecasts...", flush=True)

    for k in range(50, len(df_test) - 1):
        window_df = df_test.iloc[max(0, k-60):k+1]
        
        pred = model.predict_next_candle(window_df)
        pred_color = str(pred['color'])
        conf = float(pred['confidence_pct'])

        next_candle = df_test.iloc[k+1]
        actual_color = "GREEN" if next_candle['close'] >= next_candle['open'] else "RED"

        total_tested += 1
        if pred_color == "GREEN": green_pred += 1
        else: red_pred += 1

        if actual_color == "GREEN": green_actual += 1
        else: red_actual += 1

        is_correct = (pred_color == actual_color)
        if is_correct:
            correct_count += 1

        if conf >= 60.0:
            high_conf_total += 1
            if is_correct:
                high_conf_correct += 1

        if total_tested <= 10:
            print(f"  Bar {k+1}: Predicted={pred_color} ({conf:.1f}%) | Actual={actual_color} | Open={next_candle['open']:.2f} Close={next_candle['close']:.2f} | {'✓ MATCH' if is_correct else '✗ MISS'}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("CANDLE PREDICTION ACCURACY SUMMARY", flush=True)
    print("=" * 70, flush=True)
    acc = (correct_count / total_tested * 100) if total_tested > 0 else 0
    high_acc = (high_conf_correct / high_conf_total * 100) if high_conf_total > 0 else 0

    print(f"Total Next Candles Tested: {total_tested}", flush=True)
    print(f"Overall Prediction Accuracy: {acc:.2f}% ({correct_count}/{total_tested})", flush=True)
    print(f"High-Confidence (>60%) Accuracy: {high_acc:.2f}% ({high_conf_correct}/{high_conf_total})", flush=True)
    print(f"Forecast Distribution: GREEN={green_pred}, RED={red_pred}", flush=True)
    print(f"Actual Distribution  : GREEN={green_actual}, RED={red_actual}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    test_candle_prediction()

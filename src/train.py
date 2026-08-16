import os
import sys
import pandas as pd
import joblib
sys.path.append('.')
from src.models.model import prep, FEATURES, TARGETS, get_split, train, evaluate

if __name__ == "__main__":
    df = prep('data/processed/features.csv')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_split(df)

    models = train(x_train, y_train)

    os.makedirs('models', exist_ok=True)
    for target, model in models.items():
        joblib.dump(model, f'models/{target}.joblib')
    print(f"Saved {len(models)} models to models/")

    print("\n--- Val Results ---")
    val_results = evaluate(models, x_val, y_val)
    for target, metrics in val_results.items():
        print(f"{target}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    print("\n--- Test Results ---")
    test_results = evaluate(models, x_test, y_test)
    for target, metrics in test_results.items():
        print(f"{target}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    print("\n--- Feature Importances (prob_A model) ---")
    importance = pd.Series(models['prob_A'].feature_importances_, index=FEATURES)
    print(importance.sort_values(ascending=False))
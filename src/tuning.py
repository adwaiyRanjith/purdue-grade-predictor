import xgboost as xgb
import sys
sys.path.append('.')
import pandas as pd
from itertools import product
from src.models.model import prep, get_split, evaluate, TARGETS, FEATURES

grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'random_state': [42]
}

if __name__ == "__main__":
    df = prep('data/processed/features.csv')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_split(df)
    k = list(grid.keys())
    v = list(grid.values())
    
    best_rmse = float('inf')
    best_params = None

    for combo in product(*v):
        params = dict(zip(k, combo))
        models = {}
        for t in TARGETS:
            model = xgb.XGBRegressor(**params)
            model.fit(x_train, y_train[t])
            models[t] = model
        results = evaluate(models, x_val, y_val)
        avg_rmse = sum(results[t]['rmse'] for t in TARGETS) / len(TARGETS)
        
        print(f"params={params}, avg_val_rmse={avg_rmse:.4f}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    print(f"\nBest params: {best_params}")
    print(f"Best avg val RMSE: {best_rmse:.4f}")

        

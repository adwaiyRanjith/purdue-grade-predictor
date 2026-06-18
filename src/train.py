import sys
import pandas as pd
sys.path.append('.')
from src.models.model import prep, FEATURES, get_split, train, evaluate

if __name__ == "__main__":
    df = prep('data/processed/features.csv')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_split(df)
    model = train(x_train, y_train)
    rmse, mae = evaluate(model, x_val, y_val)
    print(f"Val RMSE: {rmse:.4f}")
    print(f"Val MAE: {mae:.4f}")
    print(f"Baseline RMSE: 0.4342")
    print(f"Baseline MAE: 0.3350")
    test_rmse, test_mae = evaluate(model, x_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    print(importance.sort_values(ascending=False))
    print(f"Test null prof_hist_mean: {x_test['prof_hist_mean'].isnull().sum()}")
    print(f"Test total rows: {len(x_test)}")

    print(f"Val null prof_hist_mean: {x_val['prof_hist_mean'].isnull().sum()} / {len(x_val)} = {x_val['prof_hist_mean'].isnull().mean():.2%}")
    print(f"Test null prof_hist_mean: {x_test['prof_hist_mean'].isnull().sum()} / {len(x_test)} = {x_test['prof_hist_mean'].isnull().mean():.2%}")

    test_df = df[df['Academic Period'].isin([202510, 202520, 202610])].copy()
    test_df['pred'] = model.predict(x_test)
    for period in sorted(test_df['Academic Period'].unique()):
        sub = test_df[test_df['Academic Period'] == period]
        rmse = ((sub['pred'] - sub['avg_gpa']) ** 2).mean() ** 0.5
        print(f"{period}: n={len(sub)}, RMSE={rmse:.4f}")

    null_mask = x_test['prof_hist_mean'].isnull()
    for name, mask in [('null prof_hist', null_mask), ('has prof_hist', ~null_mask)]:
        sub_x, sub_y = x_test[mask], y_test[mask]
        pred = model.predict(sub_x)
        rmse = ((pred - sub_y) ** 2).mean() ** 0.5
        print(f"{name}: n={mask.sum()}, RMSE={rmse:.4f}")
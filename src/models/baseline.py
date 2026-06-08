import pandas as pd

def baseline(df: pd.DataFrame) -> tuple:
    df = df.dropna(subset=['course_hist_mean'])
    predictions = df['course_hist_mean']
    actual = df['avg_gpa']

    rmse = ((predictions - actual) **2).mean() **0.5

    mae = abs(predictions - actual).mean()
    return (rmse, mae)

if __name__ == "__main__":
    df = pd.read_csv('data/processed/features.csv')
    rmse, mae = baseline(df)
    print(f"Baseline RMSE: {rmse:.4f}")
    print(f"Baseline MAE: {mae:.4f}")
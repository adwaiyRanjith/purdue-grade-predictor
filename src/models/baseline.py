import pandas as pd


BUCKETS = ['A', 'B', 'C', 'DF', 'W']

def baseline(df: pd.DataFrame) -> dict:
    results = {}
    for b in BUCKETS:
        df = df.dropna(subset=[f'course_hist_rate_{b}'])
        predictions = df[f'course_hist_rate_{b}']
        actual = df[f'prob_{b}']
        rmse = ((predictions - actual) ** 2).mean() ** 0.5
        mae = abs(predictions - actual).mean()
        results[b] = {'rmse': rmse, 'mae': mae}
    return results

if __name__ == "__main__":
    df = pd.read_csv('data/processed/features.csv')
    results = baseline(df)
    for bucket, metrics in results.items():
        print(f"prob_{bucket}: rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}")
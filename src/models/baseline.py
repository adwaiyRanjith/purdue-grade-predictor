import pandas as pd

def baseline(df: pd.DataFrame) -> tuple:
    df = df.dropna(subset=['course_hist_mean'])
    predictions = df['course_hist_mean']
    actual = df['avg_gpa']

    rmse = ((predictions - actual) **2).mean() **0.5

    mae = abs(predictions - actual).mean()
    return (rmse, mae)

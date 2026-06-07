import pandas as pd

def split(df: pd.DataFrame):
    train = df[df['Academic Period'] < 202410]
    val = df[df['Academic Period'].isin([202410,202420])]
    test = df[df['Academic Period'].isin([202510,202520, 202610])]
    return train, val, test

if __name__ == "__main__":
    df = pd.read_csv('data/processed/features.csv')
    train, val, test = split(df)
    print(f"Train: {train.shape}")
    print(f"Val: {val.shape}")
    print(f"Test: {test.shape}")
    print(f"Train periods: {sorted(train['Academic Period'].unique())}")
    print(f"Val periods: {sorted(val['Academic Period'].unique())}")
    print(f"Test periods: {sorted(test['Academic Period'].unique())}")
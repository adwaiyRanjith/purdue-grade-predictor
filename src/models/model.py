import xgboost as xgb
import pandas as pd
import sys
sys.path.append('.')
from data.data_split import split



def prep(path: str):
    df = pd.read_csv(path)
    df['is_covid'] = df['is_covid'].astype(int)
    df['semester_type'] = df['semester_type'].map({'Fall': 0, 'Spring': 1, 'Summer': 2})
    return df

FEATURES = [
    'bayesian_rate_A', 'bayesian_rate_B', 'bayesian_rate_C', 'bayesian_rate_DF', 'bayesian_rate_W',
    'course_hist_rate_A', 'course_hist_rate_B', 'course_hist_rate_C', 'course_hist_rate_DF', 'course_hist_rate_W',
    'prof_hist_rate_A', 'prof_hist_rate_B', 'prof_hist_rate_C', 'prof_hist_rate_DF', 'prof_hist_rate_W',
    'dept_hist_rate_A', 'dept_hist_rate_B', 'dept_hist_rate_C', 'dept_hist_rate_DF', 'dept_hist_rate_W',
    'semester_type', 'course_level', 'is_covid', 'prof_experience', 'course_experience'
]

TARGETS = ['prob_A', 'prob_B', 'prob_C', 'prob_DF', 'prob_W']



def get_split(df: pd.DataFrame):
    train, val, test = split(df)
    x_train, y_train = train[FEATURES], train[TARGETS]
    x_val, y_val = val[FEATURES], val[TARGETS]
    x_test, y_test = test[FEATURES], test[TARGETS]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def train(x, y):
    models = {}
    for t in TARGETS:
        model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=42)
        model.fit(x, y[t])
        models[t] = model
    return models

def evaluate(models, x, y):
    results = {}
    for t, m in models.items():
        predictions = m.predict(x)
        rmse = ((predictions - y[t]) ** 2).mean() ** 0.5
        mae = abs(predictions - y[t]).mean()
        results[t] = {'rmse': rmse, 'mae': mae}
    return results

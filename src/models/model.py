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
    'course_hist_mean',
    'prof_hist_mean',
    'bayesian_prof_mean',
    'semester_type',
    'course_level',
    'is_covid',
    'prof_experience'
]

def get_split(df: pd.DataFrame):
    train, val, test = split(df)
    x_train, y_train = train[FEATURES], train['avg_gpa']
    x_val, y_val = val[FEATURES], val['avg_gpa']
    x_test, y_test = test[FEATURES], test['avg_gpa']
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def train(x, y):
    model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=42)
    output = model.fit(x,y)
    return output

def evaluate(model, x, y) -> tuple:
    predictions = model.predict(x)
    rmse = ((predictions - y) ** 2).mean() ** 0.5
    mae = abs(predictions - y).mean()
    return (rmse, mae)

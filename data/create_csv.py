import sys
sys.path.append('.')
import pandas as pd
import os
from data.feature_engineering import (
    calc_avg_gpa, calc_course_hist_mean, calc_prof_hist_mean,
    calc_semester_type, calc_course_level, calc_is_covid, calc_prof_exp
)
from data.data_loader import load_all_data

def create_features_csv():
    df = load_all_data()
    
    df['avg_gpa'] = calc_avg_gpa(df)
    df = df.dropna(subset=['avg_gpa'])
    df = df[df['avg_gpa'] > 0.0]
    
    df['course_hist_mean'] = calc_course_hist_mean(df)
    df['prof_hist_mean'] = calc_prof_hist_mean(df)
    df['semester_type'] = calc_semester_type(df)
    df['course_level'] = calc_course_level(df)
    df['is_covid'] = calc_is_covid(df)
    df['prof_experience'] = calc_prof_exp(df)
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/features.csv', index=False)
    print(f"Saved {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.columns.tolist())

if __name__ == "__main__":
    create_features_csv()
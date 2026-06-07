
import pandas as pd
import sys
sys.path.append('.')
from data.data_loader import load_all_data
import numpy as np

def calc_avg_gpa(df: pd.DataFrame) -> pd.Series:
    grade_scale = {
        'A+': 4.0, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'D-': 0.7,
        'F': 0.0
    }
    grades = df[list(grade_scale.keys())].fillna(0)
    numerator = sum(grades[col] * points for col, points in grade_scale.items())
    denominator = grades.sum(axis=1)

    return numerator / denominator.replace(0, float('nan'))

def calc_course_hist_mean(df: pd.DataFrame) -> pd.Series:
    sem_means = df.groupby(['Course Number', 'Academic Period'])['avg_gpa'].mean()
    sem_means = sem_means.reset_index().sort_values('Academic Period')
    sem_means['course_hist_mean'] = sem_means.groupby('Course Number')['avg_gpa'].transform(lambda x: x.shift().expanding().mean())

    result = df.merge(
        sem_means[['Course Number', 'Academic Period', 'course_hist_mean']], 
        on=['Course Number', 'Academic Period'], 
        how='left'
    )
    result.index = df.index
    return result['course_hist_mean']

def calc_prof_hist_mean(df: pd.DataFrame) -> pd.Series:
    sem_means = df.groupby(['Instructor', 'Academic Period'])['avg_gpa'].mean()
    sem_means = sem_means.reset_index().sort_values('Academic Period')
    sem_means['prof_hist_mean'] = sem_means.groupby('Instructor')['avg_gpa'].transform(lambda x: x.shift().expanding().mean())

    result = df.merge(
        sem_means[['Instructor', 'Academic Period', 'prof_hist_mean']], 
        on=['Instructor', 'Academic Period'], 
        how='left'
    )
    result.index = df.index
    return result['prof_hist_mean']

def calc_semester_type(df: pd.DataFrame) -> pd.Series:
    return df['Academic Period Desc'].str.split().str[0]

def calc_course_level(df: pd.DataFrame) -> pd.Series:
    def level(num):
        if num < 30000:
            return 1
        elif num < 50000:
            return 2
        else:
            return 3
    return df['Course Number'].apply(level)

def calc_is_covid(df: pd.DataFrame) -> pd.Series:
    return df['Academic Period'].isin([202020, 202030, 202110, 202120])

def calc_prof_exp(df: pd.DataFrame) -> pd.Series:
    sem_counts = df.groupby(['Instructor', 'Academic Period']).size().reset_index()
    sem_counts = sem_counts.sort_values('Academic Period')
    sem_counts['prof_experience'] = sem_counts.groupby('Instructor').cumcount()
    
    result = df.merge(
        sem_counts[['Instructor', 'Academic Period', 'prof_experience']],
        on=['Instructor', 'Academic Period'],
        how='left'
    )
    result.index = df.index
    return result['prof_experience']


if __name__ == "__main__":
    df = load_all_data()
    df['avg_gpa'] = calc_avg_gpa(df)
    df = df.dropna(subset=['avg_gpa'])
    df = df[df['avg_gpa'] > 0.0]
    
    df['course_hist_mean'] = calc_course_hist_mean(df)
    df['prof_hist_mean'] = calc_prof_hist_mean(df)
    df['semester_type'] = calc_semester_type(df)
    df['course_level'] = calc_course_level(df)
    df['is_covid'] = calc_is_covid(df)
    
    print(df['semester_type'].value_counts())
    print(df['course_level'].value_counts())
    print(df['is_covid'].value_counts())
    print(df[['avg_gpa', 'course_hist_mean', 'prof_hist_mean']].describe())
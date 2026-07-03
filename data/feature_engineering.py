
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


def calc_grade_buckets(df: pd.DataFrame) -> pd.DataFrame:
    grade_cols = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F', 'W']
    grades = df[grade_cols].fillna(0)
    demoninator = grades.sum(axis=1)
    prob_A = grades[['A+', 'A', 'A-']].sum(axis=1) / demoninator
    prob_B = grades[['B+', 'B', 'B-']].sum(axis=1) / demoninator
    prob_C = grades[['C+', 'C', 'C-']].sum(axis=1) / demoninator
    prob_DF = grades[['D+', 'D', 'D-', 'F']].sum(axis=1) / demoninator
    prob_W = grades['W'] / demoninator

    return pd.DataFrame({'prob_A': prob_A, 'prob_B': prob_B, 'prob_C': prob_C, 'prob_DF': prob_DF, 'prob_W': prob_W}, index=df.index)

def calc_dept_hist_rate(df: pd.DataFrame, target_col) -> pd.Series:
    sem_means = df.groupby(['Subject', 'Academic Period'])[target_col].mean()
    sem_means = sem_means.reset_index().sort_values('Academic Period')
    sem_means['dept_hist_rate'] = sem_means.groupby('Subject')[target_col].transform(lambda x: x.shift().expanding().mean())

    result = df.merge(
        sem_means[['Subject', 'Academic Period', 'dept_hist_rate']], 
        on=['Subject', 'Academic Period'], 
        how='left'
    )
    result.index = df.index
    return result['dept_hist_rate']

def calc_course_hist_rate(df: pd.DataFrame, target_col) -> pd.Series:
    sem_means = df.groupby(['Course Number', 'Academic Period'])[target_col].mean()
    sem_means = sem_means.reset_index().sort_values('Academic Period')
    sem_means['course_hist_rate'] = sem_means.groupby('Course Number')[target_col].transform(lambda x: x.shift().expanding().mean())

    result = df.merge(
        sem_means[['Course Number', 'Academic Period', 'course_hist_rate']], 
        on=['Course Number', 'Academic Period'], 
        how='left'
    )
    result.index = df.index
    return result['course_hist_rate']

def calc_prof_hist_rate(df: pd.DataFrame, target_col) -> pd.Series:
    sem_means = df.groupby(['Instructor', 'Academic Period'])[target_col].mean()
    sem_means = sem_means.reset_index().sort_values('Academic Period')
    sem_means['prof_hist_rate'] = sem_means.groupby('Instructor')[target_col].transform(lambda x: x.shift().expanding().mean())

    result = df.merge(
        sem_means[['Instructor', 'Academic Period', 'prof_hist_rate']], 
        on=['Instructor', 'Academic Period'], 
        how='left'
    )
    result.index = df.index
    return result['prof_hist_rate']

def calc_course_exp(df: pd.DataFrame) -> pd.Series:
    sem_counts = df.groupby(['Course Number', 'Academic Period']).size().reset_index()
    sem_counts = sem_counts.sort_values('Academic Period')
    sem_counts['course_experience'] = sem_counts.groupby('Course Number').cumcount()
    
    result = df.merge(
        sem_counts[['Course Number', 'Academic Period', 'course_experience']],
        on=['Course Number', 'Academic Period'],
        how='left'
    )
    result.index = df.index
    return result['course_experience']

def calc_bayesian_rate(df: pd.DataFrame, bucket: str, k_p: float = 5, k_c: float = 5) -> pd.Series:
    course_rate = df[f'course_hist_rate_{bucket}']
    dept_rate = df[f'dept_hist_rate_{bucket}']
    n_c = df['course_experience']
    weight = n_c / (n_c + k_c)
    blended_course = weight * course_rate + (1 - weight) * dept_rate

    prof_rate = df[f'prof_hist_rate_{bucket}'].fillna(blended_course)
    n_p = df['prof_experience']
    weight2 = n_p / (n_p + k_p)
    blended_prof = weight2 * prof_rate + (1 - weight2) * blended_course
    global_mean = df[f'prob_{bucket}'].mean()
    blended_prof = blended_prof.fillna(global_mean)
    return blended_prof

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

    buckets = calc_grade_buckets(df)
    total = buckets.sum(axis=1)
    print(buckets.describe())
    print(f"\nBucket sum check: min={total.min():.4f}, max={total.max():.4f}, mean={total.mean():.4f}")
    df = pd.concat([df, buckets], axis =1)
    print('prob_A' in df.columns)
    print(df.columns.tolist())
    df['dept_hist_rate_A'] = calc_dept_hist_rate(df, 'prob_A')
    df['dept_hist_rate_B'] = calc_dept_hist_rate(df, 'prob_B')
    df['dept_hist_rate_C'] = calc_dept_hist_rate(df, 'prob_C')
    df['dept_hist_rate_DF'] = calc_dept_hist_rate(df, 'prob_DF')
    df['dept_hist_rate_W'] = calc_dept_hist_rate(df, 'prob_W')
    dept_cols = ['dept_hist_rate_A', 'dept_hist_rate_B', 'dept_hist_rate_C', 'dept_hist_rate_DF', 'dept_hist_rate_W']
    print(df[dept_cols].describe())

    df['course_hist_rate_A'] = calc_course_hist_rate(df, 'prob_A')
    df['course_hist_rate_B'] = calc_course_hist_rate(df, 'prob_B')
    df['course_hist_rate_C'] = calc_course_hist_rate(df, 'prob_C')
    df['course_hist_rate_DF'] = calc_course_hist_rate(df, 'prob_DF')
    df['course_hist_rate_W'] = calc_course_hist_rate(df, 'prob_W')
    course_cols = ['course_hist_rate_A', 'course_hist_rate_B', 'course_hist_rate_C', 'course_hist_rate_DF', 'course_hist_rate_W']
    print(df[course_cols].describe())

    df['prof_hist_rate_A'] = calc_prof_hist_rate(df, 'prob_A')
    df['prof_hist_rate_B'] = calc_prof_hist_rate(df, 'prob_B')
    df['prof_hist_rate_C'] = calc_prof_hist_rate(df, 'prob_C')
    df['prof_hist_rate_DF'] = calc_prof_hist_rate(df, 'prob_DF')
    df['prof_hist_rate_W'] = calc_prof_hist_rate(df, 'prob_W')
    prof_cols = ['prof_hist_rate_A', 'prof_hist_rate_B', 'prof_hist_rate_C', 'prof_hist_rate_DF', 'prof_hist_rate_W']
    print(df[prof_cols].describe())

    df['prof_experience'] = calc_prof_exp(df)
    df['course_experience'] = calc_course_exp(df)
    df['bayesian_rate_A'] = calc_bayesian_rate(df, 'A')
    for bucket in ['A', 'B', 'C', 'DF', 'W']:
        df[f'bayesian_rate_{bucket}'] = calc_bayesian_rate(df, bucket)
    bayesian_cols = [f'bayesian_rate_{b}' for b in ['A', 'B', 'C', 'DF', 'W']]
    print(df[bayesian_cols].describe())
    print(df[bayesian_cols].isnull().sum())
    
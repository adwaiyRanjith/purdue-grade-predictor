import sys
sys.path.append('.')
import os
import pandas as pd
import joblib

BUCKETS = ['A', 'B', 'C', 'DF', 'W']


def _semester_weighted_state(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Average prob_* per semester first, then across semesters, so a course/professor/
    department with more sections in a given semester isn't overweighted relative to one
    with fewer. Matches the semester-level-then-average pattern used in training
    (data/feature_engineering.py's calc_*_hist_rate functions), just without the .shift()
    since serving wants history inclusive of the latest known semester, not exclusive of it.
    """
    prob_cols = [f'prob_{b}' for b in BUCKETS]
    sem_means = df.groupby(keys + ['Academic Period'])[prob_cols].mean().reset_index()
    state = sem_means.groupby(keys)[prob_cols].mean()
    state['experience'] = sem_means.groupby(keys).size()
    state.columns = [f'rate_{b}' for b in BUCKETS] + ['experience']
    return state


def build_state_tables(df: pd.DataFrame):
    """Precomputed 'current state' for serving: one row per course, professor, and
    department, holding historical bucket rates and an experience count.

    This deliberately does NOT precompute bayesian_rate_*. That blend depends on which
    specific course a professor is being asked about (a professor's Bayesian rate for
    a course they've taught many times looks different from the same professor's rate
    for a course they've never taught), so it has to be assembled at request time from
    course_state + prof_state + dept_state using shrink_toward from feature_engineering.py
    — see api/main.py.
    """
    course_state = _semester_weighted_state(df, ['Subject', 'Course Number'])
    prof_state = _semester_weighted_state(df, ['Instructor'])
    dept_state = _semester_weighted_state(df, ['Subject']).drop(columns='experience')
    return course_state, prof_state, dept_state


if __name__ == "__main__":
    df = pd.read_csv('data/processed/features.csv')

    course_state, prof_state, dept_state = build_state_tables(df)

    os.makedirs('data/processed/state', exist_ok=True)
    joblib.dump(course_state, 'data/processed/state/course_state.joblib')
    joblib.dump(prof_state, 'data/processed/state/prof_state.joblib')
    joblib.dump(dept_state, 'data/processed/state/dept_state.joblib')

    print(f"course_state: {len(course_state)} (Subject, Course Number) pairs")
    print(f"prof_state: {len(prof_state)} instructors")
    print(f"dept_state: {len(dept_state)} departments")

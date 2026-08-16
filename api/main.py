import sys
sys.path.append('.')
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse
from data.feature_engineering import shrink_toward, course_level_from_number
from src.models.model import FEATURES, TARGETS, SEMESTER_TYPE_MAP

BUCKETS = ['A', 'B', 'C', 'DF', 'W']
K_C = 5
K_P = 5

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for target in TARGETS:
        artifacts[target] = joblib.load(f'models/{target}.joblib')
    artifacts['course_state'] = joblib.load('data/processed/state/course_state.joblib')
    artifacts['prof_state'] = joblib.load('data/processed/state/prof_state.joblib')
    artifacts['dept_state'] = joblib.load('data/processed/state/dept_state.joblib')
    yield
    artifacts.clear()


app = FastAPI(title="Purdue Grade Predictor API", lifespan=lifespan)


def _course_row(subject: str, course_number: int):
    key = (subject, float(course_number))
    course_state = artifacts['course_state']
    if key in course_state.index:
        return course_state.loc[key], True
    return None, False


def _prof_row(instructor: str):
    prof_state = artifacts['prof_state']
    if instructor in prof_state.index:
        return prof_state.loc[instructor], True
    return None, False


def _dept_row(subject: str):
    dept_state = artifacts['dept_state']
    if subject not in dept_state.index:
        raise HTTPException(status_code=404, detail=f"Unknown subject '{subject}'")
    return dept_state.loc[subject]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    dept_row = _dept_row(req.subject)
    course_row, course_known = _course_row(req.subject, req.course_number)
    prof_row, professor_known = _prof_row(req.instructor)

    course_experience = float(course_row['experience']) if course_known else 0.0
    prof_experience = float(prof_row['experience']) if professor_known else 0.0

    features = {
        'semester_type': SEMESTER_TYPE_MAP[req.semester_type],
        'course_level': course_level_from_number(req.course_number),
        'is_covid': 0,
        'prof_experience': prof_experience,
        'course_experience': course_experience,
    }

    for bucket in BUCKETS:
        dept_rate = dept_row[f'rate_{bucket}']
        course_rate = course_row[f'rate_{bucket}'] if course_known else dept_rate
        blended_course = shrink_toward(course_rate, course_experience, dept_rate, K_C)

        prof_rate = prof_row[f'rate_{bucket}'] if professor_known else blended_course
        bayesian_rate = shrink_toward(prof_rate, prof_experience, blended_course, K_P)

        features[f'course_hist_rate_{bucket}'] = course_rate
        features[f'prof_hist_rate_{bucket}'] = prof_rate
        features[f'dept_hist_rate_{bucket}'] = dept_rate
        features[f'bayesian_rate_{bucket}'] = bayesian_rate

    row = pd.DataFrame([features])[FEATURES]

    raw = {bucket: max(float(artifacts[f'prob_{bucket}'].predict(row)[0]), 0.0) for bucket in BUCKETS}
    total = sum(raw.values())
    normalized = {b: (v / total if total > 0 else 1 / len(BUCKETS)) for b, v in raw.items()}

    return PredictResponse(
        prob_A=normalized['A'],
        prob_B=normalized['B'],
        prob_C=normalized['C'],
        prob_DF=normalized['DF'],
        prob_W=normalized['W'],
        course_known=course_known,
        professor_known=professor_known,
    )


@app.get("/courses")
def list_courses():
    return [{"subject": s, "course_number": int(c)} for s, c in artifacts['course_state'].index]


@app.get("/professors")
def list_professors():
    return list(artifacts['prof_state'].index)

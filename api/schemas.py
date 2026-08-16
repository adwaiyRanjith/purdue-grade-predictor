from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    subject: str = Field(..., description="Course subject/department code, e.g. 'CS'")
    course_number: int = Field(..., description="Course number, e.g. 18000")
    instructor: str = Field(..., description="Instructor name as it appears in Purdue records")
    semester_type: Literal["Fall", "Spring", "Summer"]


class PredictResponse(BaseModel):
    prob_A: float
    prob_B: float
    prob_C: float
    prob_DF: float
    prob_W: float
    course_known: bool = Field(..., description="Whether this (subject, course_number) has prior history")
    professor_known: bool = Field(..., description="Whether this instructor has prior history")

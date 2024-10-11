from __future__ import annotations
from typing import List, Optional

from pydantic import BaseModel

# Circular reference issue
# https://stackoverflow.com/questions/76724501/fastapi-many-to-many-relationship-multiple-models-and-schemas-circular-depende
# Create a Pydantic Model for the database Model above


# Schemas for Topic
class Topic(BaseModel):
    id: int
    title: str
    # questions: List[Question] = []

    class Config:
        orm_mode = True

class TopicCreate(BaseModel):
    title: str

# Schemas for Question
class QuestionCreate(BaseModel):
    question_number: int
    description: str
    difficulty: int
    # paper_id: int
    topics_str: List[str] = []  # list of string of topics
    
class QuestionUpdate(BaseModel):
    id: int
    question_number: int
    description: str
    difficulty: int
    # paper_id: int
    topics_str: List[str] = []  # list of string of topics

class Question(BaseModel):
    id: int
    question_number: int
    description: str 
    difficulty: int
    paper_id: int
    topics: List[Topic] = []  # list of string of topics

    class Config:
        orm_mode = True


# Schemas for Paper
#NOTE: by right should have all these validation enforced when returning from API, but comment out for now
class Paper(BaseModel):
    id: int
    title: str
    description: Optional[str] = None #NOTE: should not be none, update frontend to parse this information
    module: Optional[str] = None #NOTE: should not be none, update frontend to parse this information
    year: Optional[int] = None #NOTE: should not be none, update frontend to parse this information
    overall_difficulty: Optional[float] = None #can be none, input comes later
    questions: List[Question] = []
    statistics: Optional[Statistic] = None # can be none, input comes later
    learning_outcomes: List[LearningOutcome] = [] # can be none, input comes later

    class Config:
        orm_mode = True

class PaperCreate(BaseModel):
    title: str
    description: Optional[str] = None #NOTE: should not be none, update frontend to parse this information
    module: Optional[str] = None #NOTE: should not be none, update frontend to parse this information
    year: Optional[int] = None #NOTE: should not be none, update frontend to parse this information
    overall_difficulty: Optional[float] = None #can be none, input comes later
    questions: List[Question] = []
    statistics: Optional[Statistic] = None # can be none, input comes later
    learning_outcomes: List[LearningOutcome] = [] # can be empty, input comes later



# NOTE: These schemas are not used for now, just here to prep for future use
class Statistic(BaseModel):
    id: int
    normalised_average_marks: float
    normalised_mean_marks: float
    normalised_median_marks: float
    normalised_min_marks: float
    normalised_max_marks: float

    class Config:
        orm_mode = True

class LearningOutcome(BaseModel):
    id: int
    title: str
    description: str
    papers: List[Paper] = []

    class Config:
        orm_mode = True




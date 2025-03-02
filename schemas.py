"""
Schemas.py is used to define the Pydantic models for the database models.
Pydantic models are used for data validation and serialization. They ensure that the data being sent to and received from the API conforms to the expected structure.

This file serves several purposes:
1. **Data Validation**: Pydantic models validate the data to ensure it meets the required types and constraints before it is processed by the application.
2. **Data Serialization**: Pydantic models can convert data between different formats, such as converting database models to JSON for API responses.
3. **API Integration**: When using frameworks like FastAPI, Pydantic models are used to define the request and response schemas for API endpoints.
4. **Type Safety**: Pydantic models provide type hints and autocompletion in IDEs, making the code more readable and easier to maintain.

Below are the Pydantic models corresponding to the SQLAlchemy models defined in models.py.
These models are used to validate and serialize data for the Topic and Question entities.

# Circular reference issue
# https://stackoverflow.com/questions/76724501/fastapi-many-to-many-relationship-multiple-models-and-schemas-circular-depende
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

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

# class for generation of question prompt
class GenerateQuestion(BaseModel):
    prompt: str



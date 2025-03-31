"""
Schemas.py is used to define the Pydantic models for the database models.
Pydantic models are used for data validation and serialization. They ensure that the data being sent to and received from the API conforms to the expected structure.

This file serves several purposes:
1. **Data Validation**: Pydantic models validate the data to ensure it meets the required types and constraints before it is processed by the application.
2. **API Integration**: When using frameworks like FastAPI, Pydantic models are used to define the request and response schemas for API endpoints.
3. **Type Safety**: Pydantic models provide type hints and autocompletion in IDEs, making the code more readable and easier to maintain.

Below are the Pydantic models corresponding to the SQLAlchemy models defined in models.py.
These models are used to validate and serialize data for the Topic and Question entities.

Note that validation are handled at the schema ONLY for request's schema to avoid duplicate validation.


# Circular reference issue
# https://stackoverflow.com/questions/76724501/fastapi-many-to-many-relationship-multiple-models-and-schemas-circular-depende
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


"""BASE MODELS
Defines schemas that mirror database tables:
- Includes fields from database tables
- Enables ORM mode for database operations
"""


class Topic(BaseModel):
    id: int
    title: str

    class Config:
        orm_mode = True

class TopicCreate(BaseModel):
    title: str

# Schemas for Question
class QuestionCreate(BaseModel):
    question_number: int
    description: str
    mark: Optional[int] = 0 # Make it required in the future
    difficulty: int
    # paper_id: int
    topics_str: List[str] = []  # list of string of topics
    
class QuestionUpdate(BaseModel):
    id: int
    question_number: int
    description: str
    mark: Optional[int] = 0 # Make it required in the future
    difficulty: int
    # paper_id: int
    topics_str: List[str] = []  # list of string of topics

class Question(BaseModel):
    id: int
    question_number: int
    description: str
    mark: int
    difficulty: int
    paper_id: int
    topics: List[Topic] = []


class Paper(BaseModel):
    id: int
    title: str
    description: Optional[str] = None  # NOTE: Nullable for this iteration
    course_id: Optional[int] = None #NOTE: should not be none, update frontend to parse this information
    module: Optional[str] = None  # NOTE: Nullable for this iteration
    year: Optional[int] = None  # NOTE: Nullable for this iteration
    overall_difficulty: Optional[float] = None  # NOTE: Nullable for this iteration
    questions: List[Question] = []
    statistics: Optional[Statistic] = None  # NOTE: Nullable for this iteration
    learning_outcomes: List[LearningOutcome]  # NOTE: Nullable for this iteration
    student_scores: Optional[List[List[int]]] = [] # can be empty, input comes later

    class Config:
        orm_mode = True


# NOTE: Schemas for Statustic and LearningOutcome are not in used for this first iteration.
class PaperCreate(BaseModel):
    title: str
    description: Optional[str] = None #NOTE: should not be none, update frontend to parse this information
    course_id: Optional[int] = None #NOTE: should not be none, update frontend to parse this information
    module: Optional[str] = None #NOTE: should not be none, update frontend to parse this information
    year: Optional[int] = None #NOTE: should not be none, update frontend to parse this information
    overall_difficulty: Optional[float] = None #can be none, input comes later
    questions: List[Question] = []
    statistics: Optional[Statistic] = None # can be none, input comes later
    learning_outcomes: List[LearningOutcome] = [] # can be empty, input comes later
    student_scores: Optional[List[List[int]]] = [] # can be empty, input comes later

class PaperUpdate(BaseModel):
    id: int
    title: Optional[str]
    questions: List[QuestionUpdate]

# Schemas for Course
class Course(BaseModel):
    id: int
    title: str
    papers: List[Paper] = []

    class Config:
        orm_mode = True

class CourseCreate(BaseModel):
    title: str
    papers: List[Paper] = []


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


"""CREATE/UPDATE MODELS
Defines schemas for creating and updating database records:
- Create models specify required fields for new records
- Update models include ID and modifiable fields
"""


class TopicCreate(BaseModel):
    title: str = Field(..., min_length=1)


class QuestionCreate(BaseModel):
    question_number: int
    description: str
    mark: int
    difficulty: int
    topics_str: List[str]  # list of string of topics


class QuestionUpdate(BaseModel):
    id: int
    question_number: int
    description: str
    mark: int
    difficulty: int
    topics_str: List[str] = []  # list of string of topics

    class Config:
        orm_mode = True


class PaperCreate(BaseModel):
    title: str
    description: Optional[str] = None  # NOTE: Nullable for this iteration
    module: Optional[str] = None  # NOTE: Nullable for this iteration
    year: Optional[int] = None  # NOTE: Nullable for this iteration
    overall_difficulty: Optional[float] = None  # NOTE: Nullable for this iteration
    questions: List[Question]
    statistics: Optional[Statistic] = None  # NOTE: Nullable for this iteration
    learning_outcomes: List[LearningOutcome] = []  # NOTE: Nullable for this iteration


"""API REQUEST/RESPONSE SCHEMA
Defines the expected format and validation rules for API communication:
- Request schemas validate incoming data (fields, types, constraints)
- Response schemas structure outgoing data
"""


# for generate of question using gpt
class GenerateQuestionGPTRequest(BaseModel):
    prompt: str = Field(..., min_length=1)  # Non-empty prompt


# for quick generation of questions feature
class TopicMark(BaseModel):
    topic_id: int = Field(..., gt=0)  # Positive ID
    max_allocated_marks: int = Field(..., gt=0)  # Positive marks


class QuickGenerateQuestionsRequest(BaseModel):
    topics: List[TopicMark] = Field(..., min_items=1)  # At least one topic


class QuestionCreateRequest(BaseModel):
    description: str = Field(..., min_length=1)  # Non-empty description
    mark: int = Field(..., gt=0)  # Positive marks
    difficulty: int = Field(..., ge=1, le=5)  # Difficulty between 1-5
    topics: List[str] = Field(..., min_items=1)  # at least one topic


class PaperCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)  # Non-empty title
    questions: List[QuestionCreateRequest] = Field(..., min_items=1)  # At least one question


# this is the format of the parsed paper from GPT API
class PaperParseResponse(BaseModel):
    title: str
    questions: List[dict]
    all_topics: List[str]


# this is the format of the generated question from GPT API
class GenerateQuestionGPTResponse(BaseModel):
    description: str
    topics: list[str]
    mark: int
    difficulty: int

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
from typing import List, Optional, Any, Dict
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
        from_attributes = True

# 21 oct
class OptionOverview(BaseModel):
    interpretation: str
    likely_misunderstanding: str


class Question(BaseModel):
    id: int
    question_number: int
    type: str
    description: str
    answer: Optional[str] = None
    mark: int
    difficulty: int
    paper_id: int
    topics: List[Topic] = []
    level: Optional[str] = None
    overview: Optional[Dict[str, OptionOverview]] = {}
    student_choice_statistics: Optional[Dict[str, float]] = None
    grade_statistics: Optional[Dict[str, float]] = None
    quartile_statistics: Optional[Dict[str, Any]] = None
    individual_choice_statistics: Optional[Dict[str, float]] = None

    class Config:
        from_attributes = True


class Paper(BaseModel):
    id: int
    title: str
    description: Optional[str] = None  # NOTE: Nullable for this iteration
    module_code: Optional[str] = None
    module_name: Optional[str] = None
    academic_year: Optional[str] = None
    semester: Optional[str] = None  # NOTE: Nullable for this iteration
    overall_difficulty: Optional[float] = None  # NOTE: Nullable for this iteration
    questions: List[Question] = []
    statistics: Optional[Statistic] = None  # NOTE: Nullable for this iteration
    learning_outcomes: List[LearningOutcome]  # NOTE: Nullable for this iteration
    dashboard_data: Optional[Dashboard] = None

    class Config:
        from_attributes = True


# NOTE: Schemas for Statustic and LearningOutcome are not in used for this first iteration.
class Statistic(BaseModel):
    id: int
    normalised_average_marks: float
    normalised_mean_marks: float
    normalised_median_marks: float
    normalised_min_marks: float
    normalised_max_marks: float

    class Config:
        from_attributes = True


class LearningOutcome(BaseModel):
    id: int
    title: str
    description: str
    papers: List[Paper] = []

    class Config:
        from_attributes = True


"""CREATE/UPDATE MODELS
Defines schemas for creating and updating database records:
- Create models specify required fields for new records
- Update models include ID and modifiable fields
"""


class TopicCreate(BaseModel):
    title: str = Field(..., min_length=1)


class QuestionCreate(BaseModel):
    question_number: int
    type: str = "MCQ"
    description: str
    mark: int
    difficulty: int
    topics_str: List[str]  # list of string of topics
    level: str
    overview: Optional[Dict[str, OptionOverview]] = {}


class QuestionUpdate(BaseModel):
    id: int
    question_number: int
    type: str = "MCQ"
    description: str
    mark: int
    difficulty: int
    topics_str: List[str] = []  # list of string of topics
    level: Optional[str] = None
    overview: Optional[Dict[str, OptionOverview]] = {}

    class Config:
        from_attributes = True
        
# 16 oct


class DashboardPayload(BaseModel):
    title: Optional[str] = None
    average_difficulty: Optional[float] = None
    topic_frequencies: Optional[List[Any]] = None
    difficulty_distribution: Optional[List[Any]] = None

    class Config:
        from_attributes = True


class DashboardBase(BaseModel):
    payload: DashboardPayload   # ✅ nested “payload” to match frontend


class DashboardCreate(DashboardBase):
    pass


class Dashboard(DashboardBase):
    id: int
    paper_id: int

    class Config:
        from_attributes = True

class PaperCreate(BaseModel):
    title: str
    description: Optional[str] = None  # NOTE: Nullable for this iteration
    module_code: Optional[str] = None
    module_name: Optional[str] = None
    academic_year: Optional[str] = None
    semester: Optional[str] = None
    overall_difficulty: Optional[float] = None  # NOTE: Nullable for this iteration
    questions: List[QuestionCreateRequest]
    statistics: Optional[Statistic] = None  # NOTE: Nullable for this iteration
    learning_outcomes: List[LearningOutcome] = []  # NOTE: Nullable for this iteration
    dashboard_data: Optional[DashboardCreate] = None


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
    question_number: int
    type: str = "MCQ"
    description: str = Field(..., min_length=1)  # Non-empty description
    mark: int = Field(..., gt=0)  # Positive marks
    difficulty: int = Field(..., ge=1, le=5)  # Difficulty between 1-5
    topics: List[str] = Field(..., min_items=1)  # at least one topic
    level: Optional[str] = None
    overview: Optional[Dict[str, OptionOverview]] = None


class PaperCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)  # Non-empty title
    questions: List[QuestionCreateRequest] = Field(..., min_items=1)  # At least one question


    
# 21 oct 
class ParsedQuestion(BaseModel):
    description: str
    topics: List[str]
    level: str
    difficulty: int
    mark: int
    overview: Optional[Dict[str, OptionOverview]] = None


# this is the format of the parsed paper from GPT API
class PaperParseResponse(BaseModel):
    title: str
    questions: List[dict]
    all_topics: List[str]


class PaperUpdate(BaseModel):
    title: Optional[str] = None
    module_code: Optional[str] = None
    module_name: Optional[str] = None
    academic_year: Optional[str] = None
    semester: Optional[str] = None


# this is the format of the generated question from GPT API
class GenerateQuestionGPTResponse(BaseModel):
    type: str = "MCQ"
    description: str
    topics: list[str]
    mark: int
    difficulty: int
    
# 9 oct
class PaperSummary(BaseModel):
    id: int
    title: str
    module_code: Optional[str] = None
    module_name: Optional[str] = None
    academic_year: Optional[str] = None
    semester: Optional[str] = None

    class Config:
        from_attributes = True
        

class GradeDistributionBase(BaseModel):
    bins: List[str]
    counts: List[int]
    total_students: int
    average_score: float
    standard_deviation: Optional[float] = None


class GradeDistributionCreate(GradeDistributionBase):
    pass


class GradeDistribution(GradeDistributionBase):
    id: int
    paper_id: int

    class Config:
        from_attributes = True
        

# 30 oct
class QuestionChoiceStatistics(BaseModel):
    """Data for a single question's choice statistics."""
    question_id: int
    # e.g., {"A": 0.25, "B": 0.5, "C": 0.1, "D": 0.15}
    statistics: Dict[str, float]
    # e.g. {"correct": 0.6, "incorrect": 0.4}
    grade_statistics: Optional[Dict[str, float]] = None
    quartile_statistics: Optional[Dict[str, Any]] = None
    individual_choice_statistics: Optional[Dict[str, float]] = None
    answer: Optional[str] = None


class StudentChoiceData(BaseModel):
    """The complete payload for a paper's student choices."""
    questions: List[QuestionChoiceStatistics]
    
    
# 23 feb
# 1. Define the expected JSON structure from the AI
class AIAnalysisPayload(BaseModel):
    patterns: List[str]
    distractor_analysis: List[str]
    assessment_quality: List[str]
    cognitive_depth: List[str]
    quartile_analysis: List[str]
    recommendations: List[str]
   
    # Add any other specific fields the AI generates here

# 2. Base schema for the database model


class PaperAnalysisBase(BaseModel):
    analysis_data: AIAnalysisPayload

# 3. Schema for creating a new record


class PaperAnalysisCreate(PaperAnalysisBase):
    pass

# 4. Schema for the FastAPI response


class PaperAnalysis(PaperAnalysisBase):
    id: int
    paper_id: int

    class Config:
        from_attributes = True
        

class TopicInsight(BaseModel):
    topic_name: str
    better_performing_cohort: str
    analysis: str

# 1. The exact structure we expect from Gemini


class PaperComparisonPayload(BaseModel):
    executive_summary: str
    performance_shift: str
    topic_insights: List[TopicInsight]
    cognitive_depth_analysis: str
    actionable_recommendations: List[str]

# 2. Base schema for the DB record


class PaperComparisonBase(BaseModel):
    comparison_data: PaperComparisonPayload

# 3. Schema for creating/updating


class PaperComparisonCreate(PaperComparisonBase):
    pass

# 4. Schema for the FastAPI response (includes IDs)


class PaperComparisonDB(PaperComparisonBase):
    id: int
    baseline_paper_id: int
    comparison_paper_id: int

    class Config:
        from_attributes = True

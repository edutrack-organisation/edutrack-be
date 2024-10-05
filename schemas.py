from __future__ import annotations
from typing import List, Union

from pydantic import BaseModel

# Create a Pydantic Model for the database Model above
class Topic(BaseModel):
    id: int
    title: str
    questions: List[Question] = []

    class Config:
        orm_mode = True

class TopicCreate(BaseModel):
    title: str

class Question(BaseModel):
    id: int
    question_number: int
    description: str
    difficulty: int
    paper_id: int
    paper: Paper
    topics: List[Topic] = []

    class Config:
        orm_mode = True

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

class Paper(BaseModel):
    id: int
    title: str
    description: Union[str, None] = None
    module: str
    year: int
    overall_difficulty: float
    questions: List[Question] = []
    statistics: Statistic
    learning_outcomes: List[LearningOutcome] = []

    class Config:
        orm_mode = True



    
# class ItemBase(BaseModel):
#     title: str
#     description: Union[str, None] = None


# class ItemCreate(ItemBase):
#     pass


# class Item(ItemBase):
#     id: int
#     owner_id: int

#     class Config:
#         orm_mode = True


# class UserBase(BaseModel):
#     email: str


# class UserCreate(UserBase):
#     password: str


# class User(UserBase):
#     id: int
#     is_active: bool
#     items: List[Item] = []

#     class Config:
#         orm_mode = True
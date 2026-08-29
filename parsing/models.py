"""Validated data contracts shared by the parsing pipeline and API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    MCQ = "MCQ"
    MRQ = "MRQ"
    UNKNOWN = "UNKNOWN"


class QuestionOption(BaseModel):
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("label", "text", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ParsedQuestion(BaseModel):
    question_number: str | None = None
    question_type: QuestionType
    description: str = Field(min_length=1)
    options: list[QuestionOption] = Field(default_factory=list)
    correct_answers: list[str] = Field(default_factory=list)
    mark: int = Field(default=0, ge=0)
    difficulty: int = Field(default=0, ge=0, le=5)
    topics: list[str] = Field(default_factory=list)


class ParsedPaper(BaseModel):
    title: str = Field(min_length=1)
    questions: list[ParsedQuestion] = Field(min_length=1)


class ExtractedDocument(BaseModel):
    markdown: str = Field(min_length=1)
    page_count: int = Field(ge=1)

"""Structured exam-question extraction through a local Ollama model."""

from __future__ import annotations

import json
from typing import Any

import httpx
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from parsing.config import PdfParsingSettings
from parsing.errors import (
    DocumentTooLongError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    QuestionExtractionError,
)
from parsing.models import ParsedPaper, ParsedQuestion, QuestionOption, QuestionType


SYSTEM_PROMPT = """You are a faithful university exam transcription system.
Extract questions from Docling Markdown without paraphrasing, solving, summarising, or adding content.

Rules:
1. Preserve question order, wording, mathematical notation, and option wording.
2. Exclude administrative instructions unless they are part of a question.
3. Put only the question stem in the description field; put choices in options.
4. Use MCQ when one option is to be selected, MRQ when multiple options may be selected,
   and UNKNOWN when the document does not make the type clear.
5. Record marks only when explicitly printed. Otherwise use null.
6. Record correct answer labels only when an answer key or explicit answer is present in the document.
   Otherwise return an empty list. Never solve the question.
7. Use null for a missing question number or paper title.
8. Do not infer topics or difficulty.
9. Return every question exactly once as JSON conforming to the supplied schema.
"""


class _ModelQuestion(BaseModel):
    question_number: str | None = None
    question_type: QuestionType
    description: str = Field(min_length=1)
    options: list[QuestionOption] = Field(default_factory=list)
    correct_answers: list[str] = Field(default_factory=list)
    mark: int | None = Field(default=None, ge=0)

    @field_validator("question_number", "description")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("correct_answers")
    @classmethod
    def normalize_answer_labels(cls, values: list[str]) -> list[str]:
        return [value.strip().upper() for value in values]

    @model_validator(mode="after")
    def validate_answer_labels(self) -> "_ModelQuestion":
        labels = [option.label.upper() for option in self.options]
        if len(labels) != len(set(labels)):
            raise ValueError("option labels must be unique within a question")
        unknown_answers = set(self.correct_answers) - set(labels)
        if unknown_answers:
            raise ValueError(f"answer labels do not match options: {sorted(unknown_answers)}")
        if self.question_type == QuestionType.MCQ and len(self.correct_answers) > 1:
            raise ValueError("an MCQ cannot have multiple correct answer labels")
        return self


class _ModelPaper(BaseModel):
    title: str | None = None
    questions: list[_ModelQuestion] = Field(min_length=1)

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class OllamaQuestionExtractor:
    """Ask Ollama for schema-constrained output, then validate it locally."""

    def __init__(
        self,
        settings: PdfParsingSettings,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = AsyncClient(host=self._settings.ollama_host)
        return self._client

    async def extract(self, markdown: str, fallback_title: str) -> ParsedPaper:
        if len(markdown) > self._settings.ollama_max_input_chars:
            raise DocumentTooLongError(
                "Extracted document is too long for the configured model limit "
                f"({len(markdown)} > {self._settings.ollama_max_input_chars} characters)"
            )

        schema = _ModelPaper.model_json_schema()
        user_prompt = (
            "Extract the exam paper from the Markdown below. The required JSON schema is:\n"
            f"{json.dumps(schema, separators=(',', ':'))}\n\n"
            "<document>\n"
            f"{markdown}\n"
            "</document>"
        )

        try:
            response = await self._get_client().chat(
                model=self._settings.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format=schema,
                options={
                    "temperature": 0,
                    "seed": 7,
                    "num_ctx": self._settings.ollama_context_window,
                },
                stream=False,
            )
        except ResponseError as exc:
            if exc.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Ollama model '{self._settings.ollama_model}' is not installed"
                ) from exc
            raise OllamaUnavailableError(f"Ollama request failed: {exc.error}") from exc
        except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as exc:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self._settings.ollama_host}"
            ) from exc

        content = getattr(getattr(response, "message", None), "content", None)
        if not content:
            raise QuestionExtractionError("Ollama returned an empty response")

        try:
            model_paper = _ModelPaper.model_validate_json(content)
        except ValidationError as exc:
            raise QuestionExtractionError(
                f"Ollama returned invalid question data: {exc.errors(include_url=False)}"
            ) from exc

        questions = [self._to_parsed_question(question) for question in model_paper.questions]
        title = model_paper.title or self._fallback_title(fallback_title)
        return ParsedPaper(title=title, questions=questions)

    @staticmethod
    def _to_parsed_question(question: _ModelQuestion) -> ParsedQuestion:
        option_lines = [f"{option.label}. {option.text}" for option in question.options]
        description = "\n".join([question.description, *option_lines])
        return ParsedQuestion(
            question_number=question.question_number,
            question_type=question.question_type,
            description=description,
            options=question.options,
            correct_answers=question.correct_answers,
            mark=question.mark or 0,
            difficulty=0,
            topics=[],
        )

    @staticmethod
    def _fallback_title(filename: str) -> str:
        title = filename.rsplit(".", maxsplit=1)[0].replace("_", " ").strip()
        return title or "Untitled exam paper"


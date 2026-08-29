import json
from types import SimpleNamespace

import httpx
import pytest
from ollama import ResponseError

from parsing.config import PdfParsingSettings
from parsing.errors import (
    DocumentTooLongError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    QuestionExtractionError,
)
from parsing.models import QuestionType
from parsing.ollama_questions import OllamaQuestionExtractor


def make_settings(max_chars: int = 70000) -> PdfParsingSettings:
    return PdfParsingSettings(
        ollama_host="http://localhost:11434",
        ollama_model="qwen2.5:7b-instruct",
        ollama_context_window=32768,
        ollama_max_input_chars=max_chars,
        docling_device="auto",
        docling_timeout_seconds=120,
        max_pdf_pages=100,
        max_upload_bytes=25 * 1024 * 1024,
    )


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    async def chat(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


@pytest.mark.asyncio
async def test_extract_returns_backward_compatible_structured_questions():
    content = json.dumps(
        {
            "title": "CS2105 Final Examination",
            "questions": [
                {
                    "question_number": "1",
                    "question_type": "MCQ",
                    "description": "Which transport protocol is reliable?",
                    "options": [
                        {"label": "A", "text": "UDP"},
                        {"label": "B", "text": "TCP"},
                    ],
                    "correct_answers": [],
                    "mark": 2,
                }
            ],
        }
    )
    client = FakeClient(content)
    extractor = OllamaQuestionExtractor(make_settings(), client=client)

    paper = await extractor.extract("# Exam\n\n1. Which protocol...", "uploaded.pdf")

    assert paper.title == "CS2105 Final Examination"
    assert len(paper.questions) == 1
    question = paper.questions[0]
    assert question.question_type == QuestionType.MCQ
    assert question.description.endswith("A. UDP\nB. TCP")
    assert question.correct_answers == []
    assert question.topics == []
    assert question.difficulty == 0
    assert client.kwargs["format"]["title"] == "_ModelPaper"
    assert client.kwargs["options"]["temperature"] == 0
    assert client.kwargs["options"]["num_ctx"] == 32768


@pytest.mark.asyncio
async def test_extract_uses_filename_when_document_has_no_title():
    content = json.dumps(
        {
            "title": None,
            "questions": [
                {
                    "question_number": None,
                    "question_type": "UNKNOWN",
                    "description": "Explain packet switching.",
                    "options": [],
                    "correct_answers": [],
                    "mark": None,
                }
            ],
        }
    )
    extractor = OllamaQuestionExtractor(make_settings(), client=FakeClient(content))

    paper = await extractor.extract("Explain packet switching.", "cs2105_final_2025.pdf")

    assert paper.title == "cs2105 final 2025"
    assert paper.questions[0].mark == 0


@pytest.mark.asyncio
async def test_extract_rejects_documents_over_local_model_limit():
    extractor = OllamaQuestionExtractor(make_settings(max_chars=5), client=FakeClient("{}"))

    with pytest.raises(DocumentTooLongError):
        await extractor.extract("too long", "paper.pdf")


@pytest.mark.asyncio
async def test_extract_rejects_invalid_model_output():
    extractor = OllamaQuestionExtractor(make_settings(), client=FakeClient('{"questions": []}'))

    with pytest.raises(QuestionExtractionError, match="invalid question data"):
        await extractor.extract("question text", "paper.pdf")


@pytest.mark.asyncio
async def test_extract_rejects_whitespace_only_question_text():
    content = json.dumps(
        {
            "title": "Exam",
            "questions": [
                {
                    "question_number": "1",
                    "question_type": "MCQ",
                    "description": "   ",
                    "options": [],
                    "correct_answers": [],
                    "mark": 1,
                }
            ],
        }
    )
    extractor = OllamaQuestionExtractor(make_settings(), client=FakeClient(content))

    with pytest.raises(QuestionExtractionError, match="invalid question data"):
        await extractor.extract("question text", "paper.pdf")


@pytest.mark.asyncio
async def test_extract_explains_missing_model():
    class MissingModelClient:
        async def chat(self, **kwargs):
            raise ResponseError("model not found", 404)

    extractor = OllamaQuestionExtractor(make_settings(), client=MissingModelClient())

    with pytest.raises(OllamaModelNotFoundError, match="qwen2.5:7b-instruct"):
        await extractor.extract("question text", "paper.pdf")


@pytest.mark.asyncio
async def test_extract_explains_unavailable_ollama():
    class OfflineClient:
        async def chat(self, **kwargs):
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            raise httpx.ConnectError("connection refused", request=request)

    extractor = OllamaQuestionExtractor(make_settings(), client=OfflineClient())

    with pytest.raises(OllamaUnavailableError, match="localhost:11434"):
        await extractor.extract("question text", "paper.pdf")

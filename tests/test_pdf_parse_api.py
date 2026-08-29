import os
from io import BytesIO

os.environ.setdefault("OPEN_AI_API_KEY", "test-key")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from main import parse_paper_pdf
from parsing.errors import OllamaUnavailableError
from parsing.models import ParsedPaper, ParsedQuestion, QuestionType


def pdf_upload(content: bytes = b"%PDF-1.7\nexam") -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename="cs2105_exam.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )


@pytest.mark.asyncio
async def test_parse_endpoint_returns_questions_and_available_topics():
    class FakeParser:
        async def parse(self, path, original_filename):
            assert path.exists()
            assert original_filename == "cs2105_exam.pdf"
            return ParsedPaper(
                title="CS2105 Exam",
                questions=[
                    ParsedQuestion(
                        question_number="1",
                        question_type=QuestionType.MCQ,
                        description="Question\nA. First\nB. Second",
                        mark=1,
                    )
                ],
            )

    response = await parse_paper_pdf(pdf_upload(), parser=FakeParser())

    assert response.title == "CS2105 Exam"
    assert response.questions[0].question_type == QuestionType.MCQ


@pytest.mark.asyncio
async def test_parse_endpoint_rejects_non_pdf_content():
    class ParserMustNotRun:
        async def parse(self, path, original_filename):
            raise AssertionError("parser should not be called")

    with pytest.raises(HTTPException) as raised:
        await parse_paper_pdf(pdf_upload(b"plain text"), parser=ParserMustNotRun())

    assert raised.value.status_code == 400
    assert "signature" in raised.value.detail


@pytest.mark.asyncio
async def test_parse_endpoint_reports_offline_ollama():
    class OfflineParser:
        async def parse(self, path, original_filename):
            raise OllamaUnavailableError("Cannot connect to local Ollama")

    with pytest.raises(HTTPException) as raised:
        await parse_paper_pdf(pdf_upload(), parser=OfflineParser())

    assert raised.value.status_code == 503
    assert raised.value.detail == "Cannot connect to local Ollama"


from pathlib import Path

import pytest

from parsing.models import ExtractedDocument, ParsedPaper, ParsedQuestion, QuestionType
from parsing.service import PdfQuestionParsingService


@pytest.mark.asyncio
async def test_service_passes_docling_text_to_question_extractor():
    class FakePdfExtractor:
        def extract(self, path):
            assert path == Path("/tmp/paper.pdf")
            return ExtractedDocument(markdown="# Exam\nQuestion 1", page_count=1)

    class FakeQuestionExtractor:
        async def extract(self, markdown, fallback_title):
            assert markdown == "# Exam\nQuestion 1"
            assert fallback_title == "original_name.pdf"
            return ParsedPaper(
                title="Exam",
                questions=[
                    ParsedQuestion(
                        question_type=QuestionType.UNKNOWN,
                        description="Question 1",
                    )
                ],
            )

    service = PdfQuestionParsingService(FakePdfExtractor(), FakeQuestionExtractor())

    paper = await service.parse(Path("/tmp/paper.pdf"), "original_name.pdf")

    assert paper.title == "Exam"


"""Application service orchestrating PDF conversion and question extraction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from parsing.models import ExtractedDocument, ParsedPaper


class PdfTextExtractor(Protocol):
    def extract(self, pdf_path: Path) -> ExtractedDocument: ...


class QuestionExtractor(Protocol):
    async def extract(self, markdown: str, fallback_title: str) -> ParsedPaper: ...


class PdfQuestionParsingService:
    """Coordinate CPU-bound Docling work with asynchronous Ollama inference."""

    def __init__(
        self,
        pdf_extractor: PdfTextExtractor,
        question_extractor: QuestionExtractor,
    ) -> None:
        self._pdf_extractor = pdf_extractor
        self._question_extractor = question_extractor

    async def parse(self, pdf_path: Path, original_filename: str) -> ParsedPaper:
        document = await asyncio.to_thread(self._pdf_extractor.extract, pdf_path)
        return await self._question_extractor.extract(
            document.markdown,
            fallback_title=original_filename,
        )


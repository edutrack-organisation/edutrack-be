from pathlib import Path
from types import SimpleNamespace

import pytest

from parsing.config import PdfParsingSettings
from parsing.docling_pdf import DoclingPdfExtractor
from parsing.errors import PdfConversionError


def make_settings() -> PdfParsingSettings:
    return PdfParsingSettings(
        ollama_host="http://localhost:11434",
        ollama_model="qwen2.5:7b-instruct",
        ollama_context_window=32768,
        ollama_max_input_chars=70000,
        docling_device="auto",
        docling_timeout_seconds=120,
        max_pdf_pages=100,
        max_upload_bytes=25 * 1024 * 1024,
    )


class FakeDocument:
    pages = {1: object(), 2: object()}

    def export_to_markdown(self, **kwargs):
        assert kwargs["page_break_placeholder"] == "\n\n<!-- PAGE BREAK -->\n\n"
        return "# CS2105 Final\n\n1. Which protocol is reliable?"


class FakeConverter:
    def convert(self, path, **kwargs):
        assert path == Path("paper.pdf")
        assert kwargs["max_num_pages"] == 100
        assert kwargs["max_file_size"] == 25 * 1024 * 1024
        return SimpleNamespace(document=FakeDocument())


def test_extract_returns_markdown_and_page_count():
    extractor = DoclingPdfExtractor(make_settings(), converter=FakeConverter())

    document = extractor.extract(Path("paper.pdf"))

    assert document.markdown.startswith("# CS2105")
    assert document.page_count == 2


def test_extract_wraps_docling_errors():
    class BrokenConverter:
        def convert(self, *args, **kwargs):
            raise RuntimeError("bad PDF")

    extractor = DoclingPdfExtractor(make_settings(), converter=BrokenConverter())

    with pytest.raises(PdfConversionError, match="bad PDF"):
        extractor.extract(Path("paper.pdf"))


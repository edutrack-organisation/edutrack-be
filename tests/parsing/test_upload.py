from pathlib import Path

import pytest

from parsing.errors import InvalidPdfError, PdfTooLargeError
from parsing.upload import save_validated_pdf


class FakeUpload:
    def __init__(self, content: bytes, content_type: str = "application/pdf"):
        self.content_type = content_type
        self._content = content
        self._position = 0

    async def read(self, size: int = -1) -> bytes:
        if self._position >= len(self._content):
            return b""
        end = len(self._content) if size < 0 else self._position + size
        chunk = self._content[self._position : end]
        self._position = end
        return chunk


@pytest.mark.asyncio
async def test_save_validated_pdf_streams_to_destination(tmp_path: Path):
    destination = tmp_path / "paper.pdf"
    content = b"%PDF-1.7\nexam data"

    byte_count = await save_validated_pdf(
        FakeUpload(content),
        destination,
        max_bytes=1024,
    )

    assert byte_count == len(content)
    assert destination.read_bytes() == content


@pytest.mark.asyncio
async def test_save_validated_pdf_rejects_spoofed_file(tmp_path: Path):
    with pytest.raises(InvalidPdfError, match="signature"):
        await save_validated_pdf(
            FakeUpload(b"not a PDF"),
            tmp_path / "paper.pdf",
            max_bytes=1024,
        )


@pytest.mark.asyncio
async def test_save_validated_pdf_rejects_oversized_file(tmp_path: Path):
    with pytest.raises(PdfTooLargeError, match="upload limit"):
        await save_validated_pdf(
            FakeUpload(b"%PDF-" + b"x" * 20),
            tmp_path / "paper.pdf",
            max_bytes=10,
        )


@pytest.mark.asyncio
async def test_save_validated_pdf_rejects_wrong_content_type(tmp_path: Path):
    with pytest.raises(InvalidPdfError, match="Only PDF"):
        await save_validated_pdf(
            FakeUpload(b"%PDF-1.7", content_type="text/plain"),
            tmp_path / "paper.pdf",
            max_bytes=1024,
        )


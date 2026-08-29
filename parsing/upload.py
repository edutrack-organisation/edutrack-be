"""Safe streaming of uploaded PDF files to temporary storage."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from parsing.errors import InvalidPdfError, PdfTooLargeError


PDF_SIGNATURE = b"%PDF-"
UPLOAD_CHUNK_SIZE = 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


class AsyncUpload(Protocol):
    content_type: str | None

    async def read(self, size: int = -1) -> bytes: ...


async def save_validated_pdf(
    upload: AsyncUpload,
    destination: Path,
    max_bytes: int,
) -> int:
    """Validate type/signature and stream an upload without trusting its filename."""
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidPdfError("Only PDF uploads are supported")

    first_chunk = await upload.read(UPLOAD_CHUNK_SIZE)
    if not first_chunk.startswith(PDF_SIGNATURE):
        raise InvalidPdfError("The uploaded file does not have a valid PDF signature")

    total_bytes = len(first_chunk)
    if total_bytes > max_bytes:
        raise PdfTooLargeError(f"PDF exceeds the {max_bytes // (1024 * 1024)} MB upload limit")

    with destination.open("wb") as output:
        output.write(first_chunk)
        while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise PdfTooLargeError(
                    f"PDF exceeds the {max_bytes // (1024 * 1024)} MB upload limit"
                )
            output.write(chunk)

    return total_bytes


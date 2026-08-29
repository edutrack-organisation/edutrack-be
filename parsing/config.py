"""Configuration for the local PDF parsing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class PdfParsingSettings:
    """Environment-backed settings with safe local defaults."""

    ollama_host: str
    ollama_model: str
    ollama_context_window: int
    ollama_max_input_chars: int
    docling_device: str
    docling_timeout_seconds: float
    max_pdf_pages: int
    max_upload_bytes: int

    @classmethod
    def from_env(cls) -> "PdfParsingSettings":
        max_upload_mb = _positive_int("PDF_MAX_UPLOAD_MB", 25)
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            ollama_context_window=_positive_int("OLLAMA_CONTEXT_WINDOW", 32768),
            ollama_max_input_chars=_positive_int("OLLAMA_MAX_INPUT_CHARS", 70000),
            docling_device=os.getenv("DOCLING_DEVICE", "auto"),
            docling_timeout_seconds=_positive_float("DOCLING_TIMEOUT_SECONDS", 120.0),
            max_pdf_pages=_positive_int("PDF_MAX_PAGES", 100),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
        )

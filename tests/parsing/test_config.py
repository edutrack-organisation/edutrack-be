from parsing.config import PdfParsingSettings


def test_settings_use_local_defaults(monkeypatch):
    for name in (
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
        "OLLAMA_CONTEXT_WINDOW",
        "OLLAMA_MAX_INPUT_CHARS",
        "DOCLING_DEVICE",
        "DOCLING_TIMEOUT_SECONDS",
        "PDF_MAX_PAGES",
        "PDF_MAX_UPLOAD_MB",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = PdfParsingSettings.from_env()

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5:7b-instruct"
    assert settings.ollama_context_window == 32768
    assert settings.max_upload_bytes == 25 * 1024 * 1024


def test_settings_reject_invalid_limits(monkeypatch):
    monkeypatch.setenv("PDF_MAX_PAGES", "0")

    try:
        PdfParsingSettings.from_env()
    except ValueError as exc:
        assert str(exc) == "PDF_MAX_PAGES must be greater than zero"
    else:
        raise AssertionError("Expected invalid page limit to be rejected")

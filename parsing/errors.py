"""Domain exceptions raised by the PDF parsing pipeline."""


class PdfParsingError(Exception):
    """Base exception for expected parsing failures."""


class InvalidPdfError(PdfParsingError):
    """The upload is not a supported PDF."""


class PdfTooLargeError(InvalidPdfError):
    """The uploaded PDF exceeds the configured size limit."""


class PdfConversionError(PdfParsingError):
    """Docling could not convert the PDF into usable text."""


class DocumentTooLongError(PdfParsingError):
    """The extracted document exceeds the configured local-model limit."""


class QuestionExtractionError(PdfParsingError):
    """The model response could not be validated as an exam paper."""


class OllamaUnavailableError(PdfParsingError):
    """The local Ollama service cannot be reached."""


class OllamaModelNotFoundError(PdfParsingError):
    """The configured model has not been installed in Ollama."""

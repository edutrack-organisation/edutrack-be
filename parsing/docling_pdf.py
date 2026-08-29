"""PDF-to-Markdown extraction using Docling."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from parsing.config import PdfParsingSettings
from parsing.errors import PdfConversionError
from parsing.models import ExtractedDocument


class DoclingPdfExtractor:
    """Lazily initialize Docling and convert one local PDF at a time."""

    def __init__(
        self,
        settings: PdfParsingSettings,
        converter: Any | None = None,
    ) -> None:
        self._settings = settings
        self._converter = converter
        # A single local conversion at a time avoids duplicate initialization
        # and excessive memory pressure under concurrent upload requests.
        self._conversion_lock = Lock()

    def _get_converter(self) -> Any:
        if self._converter is None:
            # Docling imports large ML dependencies, so defer them until the
            # first parsing request instead of slowing every API startup.
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions(
                document_timeout=self._settings.docling_timeout_seconds,
                accelerator_options=AcceleratorOptions(device=self._settings.docling_device),
                do_ocr=True,
                do_table_structure=True,
                do_formula_enrichment=False,
                do_picture_description=False,
            )
            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                },
            )
        return self._converter

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """Convert a PDF to page-delimited Markdown for downstream extraction."""
        try:
            with self._conversion_lock:
                result = self._get_converter().convert(
                    pdf_path,
                    raises_on_error=True,
                    max_num_pages=self._settings.max_pdf_pages,
                    max_file_size=self._settings.max_upload_bytes,
                )
            markdown = result.document.export_to_markdown(
                page_break_placeholder="\n\n<!-- PAGE BREAK -->\n\n",
            ).strip()
            page_count = len(result.document.pages)
        except Exception as exc:
            raise PdfConversionError(f"Docling could not convert the PDF: {exc}") from exc

        if not markdown:
            raise PdfConversionError("Docling did not extract any text from the PDF")
        if page_count < 1:
            raise PdfConversionError("Docling reported that the PDF has no pages")

        return ExtractedDocument(markdown=markdown, page_count=page_count)

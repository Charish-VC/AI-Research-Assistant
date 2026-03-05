"""PDF document text extractor.

Uses ``pdfplumber`` for high-quality text extraction with fallback to
``PyPDF2`` when pdfplumber fails on a particular page.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.models import DocumentMetadata, SourceType
from src.pipeline.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """Extract text and metadata from PDF files."""

    source_type = SourceType.PDF

    def extract_text(self, file_path: Path) -> str:
        """Extract text from all pages of a PDF.

        Tries ``pdfplumber`` first for each page, falling back to
        ``PyPDF2`` if a page fails.

        Args:
            file_path: Path to a ``.pdf`` file.

        Returns:
            Concatenated text from all pages.
        """
        pages: list[str] = []

        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(text)
                    else:
                        logger.debug(
                            "pdfplumber returned empty for page %d, trying PyPDF2", i
                        )
                        pages.append(self._extract_page_pypdf2(file_path, i))
        except Exception as exc:
            logger.warning("pdfplumber failed (%s), falling back to PyPDF2", exc)
            pages = self._extract_all_pypdf2(file_path)

        return "\n\n".join(pages)

    def extract_metadata(
        self, file_path: Path, raw_text: str
    ) -> DocumentMetadata:
        """Pull metadata from the PDF info dictionary.

        Args:
            file_path: Path to the PDF.
            raw_text: Previously extracted text (unused here).

        Returns:
            ``DocumentMetadata`` populated from PDF info fields.
        """
        metadata = DocumentMetadata(source=str(file_path))
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(file_path))
            info = reader.metadata
            if info:
                metadata.title = info.title or ""
                metadata.authors = [info.author] if info.author else []
        except Exception as exc:
            logger.debug("Could not read PDF metadata: %s", exc)

        return metadata

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_page_pypdf2(file_path: Path, page_index: int) -> str:
        """Extract text from a single page using PyPDF2."""
        from PyPDF2 import PdfReader

        reader = PdfReader(str(file_path))
        if page_index < len(reader.pages):
            return reader.pages[page_index].extract_text() or ""
        return ""

    @staticmethod
    def _extract_all_pypdf2(file_path: Path) -> list[str]:
        """Extract text from every page using PyPDF2."""
        from PyPDF2 import PdfReader

        reader = PdfReader(str(file_path))
        return [page.extract_text() or "" for page in reader.pages]

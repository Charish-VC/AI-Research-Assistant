"""Plain-text document extractor.

Handles ``.txt`` files with automatic encoding detection via ``chardet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.models import DocumentMetadata, SourceType
from src.pipeline.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class TextExtractor(BaseExtractor):
    """Extract text from plain ``.txt`` files."""

    source_type = SourceType.TXT

    def extract_text(self, file_path: Path) -> str:
        """Read a text file with automatic encoding detection.

        Tries UTF-8 first; falls back to ``chardet`` for encoding
        detection if a ``UnicodeDecodeError`` occurs.

        Args:
            file_path: Path to a ``.txt`` file.

        Returns:
            File content as a string.
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.debug("UTF-8 decode failed for %s, detecting encoding", file_path.name)
            return self._read_with_detection(file_path)

    def extract_metadata(
        self, file_path: Path, raw_text: str
    ) -> DocumentMetadata:
        """Derive basic metadata from a text file.

        Uses the first non-empty line as the title.

        Args:
            file_path: Path to the text file.
            raw_text: Previously extracted text.

        Returns:
            ``DocumentMetadata`` with a title heuristic applied.
        """
        metadata = DocumentMetadata(source=str(file_path))

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if lines:
            # Take the first line (truncated) as the title
            metadata.title = lines[0][:200]

        return metadata

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _read_with_detection(file_path: Path) -> str:
        """Read a file using chardet-detected encoding."""
        import chardet

        raw_bytes = file_path.read_bytes()
        detection = chardet.detect(raw_bytes)
        encoding = detection.get("encoding", "utf-8") or "utf-8"
        logger.info("Detected encoding %s for %s", encoding, file_path.name)
        return raw_bytes.decode(encoding, errors="replace")

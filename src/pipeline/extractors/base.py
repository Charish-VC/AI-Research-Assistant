"""Abstract base class for document text extractors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.models import Document, DocumentMetadata, SourceType

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Interface that every file-type extractor must implement.

    Subclasses override ``extract_text`` and ``extract_metadata`` to handle
    format-specific logic.  The public ``extract`` method orchestrates the
    full extraction and returns a ``Document``.
    """

    source_type: SourceType

    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Return the raw text content of the file.

        Args:
            file_path: Path to the source document.

        Returns:
            Extracted plain-text string.
        """

    def extract_metadata(self, file_path: Path, raw_text: str) -> DocumentMetadata:
        """Optionally extract metadata from the file or its text.

        The default implementation returns an empty ``DocumentMetadata``.
        Subclasses may override for richer extraction.

        Args:
            file_path: Path to the source document.
            raw_text: Previously extracted text.

        Returns:
            Populated ``DocumentMetadata`` instance.
        """
        return DocumentMetadata(source=str(file_path))

    def extract(self, file_path: Path) -> Document:
        """Run the full extraction pipeline for a single file.

        Args:
            file_path: Path to the source document.

        Returns:
            A ``Document`` with text and metadata populated.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or unreadable.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        logger.info("Extracting text from %s (%s)", path.name, self.source_type.value)
        raw_text = self.extract_text(path)

        if not raw_text.strip():
            raise ValueError(f"No text could be extracted from {path.name}")

        metadata = self.extract_metadata(path, raw_text)
        metadata.source = str(path)

        return Document(
            filename=path.name,
            source_type=self.source_type,
            raw_text=raw_text,
            metadata=metadata,
        )

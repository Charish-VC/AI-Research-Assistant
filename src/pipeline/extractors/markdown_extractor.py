"""Markdown document text extractor.

Converts Markdown to plain text via ``markdown-it-py``, preserving heading
structure as metadata.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.models import DocumentMetadata, SourceType
from src.pipeline.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class MarkdownExtractor(BaseExtractor):
    """Extract text and metadata from Markdown files."""

    source_type = SourceType.MARKDOWN

    def extract_text(self, file_path: Path) -> str:
        """Read and convert Markdown to plain text.

        Uses ``markdown-it-py`` to parse the Markdown, then strips all
        markup to yield plain text while preserving paragraph structure.

        Args:
            file_path: Path to a ``.md`` file.

        Returns:
            Plain text with paragraphs separated by double newlines.
        """
        raw_md = file_path.read_text(encoding="utf-8", errors="replace")

        try:
            from markdown_it import MarkdownIt

            md = MarkdownIt()
            html = md.render(raw_md)
            # Strip HTML tags produced by the renderer
            text = re.sub(r"<[^>]+>", "", html)
        except ImportError:
            logger.warning("markdown-it-py not available; using raw Markdown text")
            text = raw_md

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def extract_metadata(
        self, file_path: Path, raw_text: str
    ) -> DocumentMetadata:
        """Extract metadata from Markdown content.

        Pulls the first ``# heading`` as the title and collects all
        headings as potential keywords.

        Args:
            file_path: Path to the Markdown file.
            raw_text: Previously extracted text.

        Returns:
            ``DocumentMetadata`` with title and keywords populated.
        """
        raw_md = file_path.read_text(encoding="utf-8", errors="replace")
        metadata = DocumentMetadata(source=str(file_path))

        # First H1 as the title
        h1_match = re.search(r"^#\s+(.+)$", raw_md, re.MULTILINE)
        if h1_match:
            metadata.title = h1_match.group(1).strip()

        # All headings as keywords
        headings = re.findall(r"^#{1,6}\s+(.+)$", raw_md, re.MULTILINE)
        metadata.keywords = [h.strip() for h in headings]

        return metadata

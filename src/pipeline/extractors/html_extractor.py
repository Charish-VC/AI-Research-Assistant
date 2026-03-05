"""HTML document text extractor.

Uses ``BeautifulSoup`` to strip boilerplate (nav, footer, scripts) and
extract meaningful body text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.models import DocumentMetadata, SourceType
from src.pipeline.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

# Tags whose entire subtree should be removed before text extraction
_STRIP_TAGS = {
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
    "iframe",
    "svg",
}


class HTMLExtractor(BaseExtractor):
    """Extract text and metadata from HTML files."""

    source_type = SourceType.HTML

    def extract_text(self, file_path: Path) -> str:
        """Parse HTML and return cleaned body text.

        Removes navigation, scripts, styles, and other non-content
        elements before extracting text.

        Args:
            file_path: Path to an ``.html`` file.

        Returns:
            Cleaned plain-text content.
        """
        from bs4 import BeautifulSoup

        raw_html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove boilerplate tags
        for tag_name in _STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        text = soup.get_text(separator="\n")

        # Normalise whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_metadata(
        self, file_path: Path, raw_text: str
    ) -> DocumentMetadata:
        """Extract metadata from HTML ``<head>`` elements.

        Reads ``<title>``, ``<meta name="author">``,
        ``<meta name="keywords">``, and ``<meta name="description">``.

        Args:
            file_path: Path to the HTML file.
            raw_text: Previously extracted text (unused).

        Returns:
            ``DocumentMetadata`` populated from HTML head tags.
        """
        from bs4 import BeautifulSoup

        raw_html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "html.parser")
        metadata = DocumentMetadata(source=str(file_path))

        # Title
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata.title = title_tag.string.strip()

        # Meta tags
        author_tag = soup.find("meta", attrs={"name": re.compile(r"author", re.I)})
        if author_tag and author_tag.get("content"):
            metadata.authors = [str(author_tag["content"]).strip()]

        keywords_tag = soup.find("meta", attrs={"name": re.compile(r"keywords", re.I)})
        if keywords_tag and keywords_tag.get("content"):
            metadata.keywords = [
                k.strip() for k in str(keywords_tag["content"]).split(",") if k.strip()
            ]

        return metadata

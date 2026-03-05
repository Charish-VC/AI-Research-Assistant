"""Heuristic metadata extractor.

Enriches ``DocumentMetadata`` by analysing the raw text for patterns
such as title-like headings, author blocks, dates, and keyword phrases.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from src.models import DocumentMetadata

logger = logging.getLogger(__name__)

# Common date patterns (US, ISO, European shorthand)
_DATE_PATTERNS: list[str] = [
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",              # 2024-01-15
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",              # 01/15/2024
    r"\b(\w+\s+\d{1,2},?\s+\d{4})\b",                  # January 15, 2024
    r"\b(\d{1,2}\s+\w+\s+\d{4})\b",                    # 15 January 2024
]

# Pattern for "Author: ..." or "By ..." lines
_AUTHOR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:authors?|by)\s*[:]\s*(.+)", re.IGNORECASE),
    re.compile(r"^(?:by)\s+(.+)$", re.IGNORECASE | re.MULTILINE),
]

# Common academic / technical keywords to detect
_KEYWORD_HINTS: list[str] = [
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "transformer",
    "reinforcement learning",
    "computer vision",
    "large language model",
    "data engineering",
    "RAG",
    "retrieval augmented generation",
    "vector database",
    "embedding",
    "fine-tuning",
    "knowledge graph",
    "attention mechanism",
]


class MetadataExtractor:
    """Extract or refine metadata from raw document text.

    Works heuristically — no LLM calls required.  Enriches any fields
    that were not already populated by the file-type extractor.
    """

    def extract(
        self,
        raw_text: str,
        existing: DocumentMetadata | None = None,
    ) -> DocumentMetadata:
        """Analyse *raw_text* and return enriched metadata.

        Only overwrites fields that are currently empty in *existing*.

        Args:
            raw_text: The raw (or cleaned) text of the document.
            existing: Metadata already extracted by the file-type
                extractor.  ``None`` creates a fresh instance.

        Returns:
            Enriched ``DocumentMetadata``.
        """
        meta = existing.model_copy() if existing else DocumentMetadata()

        if not meta.title:
            meta.title = self._extract_title(raw_text)

        if not meta.authors:
            meta.authors = self._extract_authors(raw_text)

        if not meta.date:
            meta.date = self._extract_date(raw_text)

        if not meta.keywords:
            meta.keywords = self._extract_keywords(raw_text)

        logger.debug(
            "Extracted metadata — title=%r, authors=%d, date=%r, keywords=%d",
            meta.title[:40] if meta.title else "",
            len(meta.authors),
            meta.date,
            len(meta.keywords),
        )
        return meta

    # ── Private extraction helpers ───────────────────────────────────────

    @staticmethod
    def _extract_title(text: str) -> str:
        """Heuristic: first non-trivial line is likely the title."""
        for line in text.splitlines():
            stripped = line.strip()
            # Skip blank lines and very short / very long lines
            if stripped and 3 < len(stripped) < 300:
                # Skip lines that look like metadata markers
                if stripped.lower().startswith(("abstract", "keywords", "author")):
                    continue
                return stripped
        return ""

    @staticmethod
    def _extract_authors(text: str) -> list[str]:
        """Look for explicit author annotations."""
        for pattern in _AUTHOR_PATTERNS:
            match = pattern.search(text[:3000])
            if match:
                raw = match.group(1).strip()
                authors = [a.strip() for a in re.split(r"[,;&]|\band\b", raw) if a.strip()]
                return authors[:10]  # cap at 10
        return []

    @staticmethod
    def _extract_date(text: str) -> str:
        """Find the first date-like string in the text header region."""
        header = text[:2000]  # Search only the first ~2000 characters
        for pattern_str in _DATE_PATTERNS:
            match = re.search(pattern_str, header)
            if match:
                raw_date = match.group(1)
                # Try to normalise to ISO format
                for fmt in (
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%m/%d/%Y",
                    "%m-%d-%Y",
                    "%B %d, %Y",
                    "%B %d %Y",
                    "%d %B %Y",
                ):
                    try:
                        dt = datetime.strptime(raw_date, fmt)
                        return dt.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
                return raw_date  # return raw if we can't parse
        return ""

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Detect known keywords by scanning the full text."""
        text_lower = text.lower()
        found: list[str] = []
        for kw in _KEYWORD_HINTS:
            if kw.lower() in text_lower:
                found.append(kw)
        return found

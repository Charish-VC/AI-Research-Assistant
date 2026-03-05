"""Text cleaning and normalisation utilities.

Removes references, normalises whitespace, and strips boilerplate so
that chunks contain only meaningful content.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaner:
    """Stateless text cleaner — call ``clean()`` to apply all steps."""

    # Regex patterns compiled once at class level
    _URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
    _EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
    _REFERENCE_BLOCK_RE = re.compile(
        r"(?:^|\n)(references|bibliography|works cited)\s*\n[\s\S]*$",
        re.IGNORECASE,
    )
    _INLINE_CITATION_RE = re.compile(r"\[[\d,;\s\-–]+\]")
    _PARENTHETICAL_CITATION_RE = re.compile(
        r"\(\s*(?:[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s*)?(?:,?\s*\d{4})\s*;?\s*)+\)"
    )
    _PAGE_NUMBER_RE = re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$", re.MULTILINE)
    _MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline to *text*.

        Steps (in order):
            1. Unicode normalisation (NFC)
            2. Remove reference / bibliography sections
            3. Remove inline citations  ``[1]``, ``[2,3]``
            4. Remove parenthetical citations  ``(Smith et al., 2020)``
            5. Remove URLs and email addresses
            6. Remove page numbers on their own line
            7. Collapse whitespace

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text ready for chunking.
        """
        if not text:
            return text

        text = self._normalise_unicode(text)
        text = self._remove_reference_sections(text)
        text = self._remove_inline_citations(text)
        text = self._remove_parenthetical_citations(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._remove_page_numbers(text)
        text = self._collapse_whitespace(text)

        logger.debug("Cleaned text length: %d chars", len(text))
        return text.strip()

    # ── Individual cleaning steps ────────────────────────────────────────

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _remove_reference_sections(self, text: str) -> str:
        return self._REFERENCE_BLOCK_RE.sub("", text)

    def _remove_inline_citations(self, text: str) -> str:
        return self._INLINE_CITATION_RE.sub("", text)

    def _remove_parenthetical_citations(self, text: str) -> str:
        return self._PARENTHETICAL_CITATION_RE.sub("", text)

    def _remove_urls(self, text: str) -> str:
        return self._URL_RE.sub("", text)

    def _remove_emails(self, text: str) -> str:
        return self._EMAIL_RE.sub("", text)

    def _remove_page_numbers(self, text: str) -> str:
        return self._PAGE_NUMBER_RE.sub("", text)

    def _collapse_whitespace(self, text: str) -> str:
        text = self._MULTI_WHITESPACE_RE.sub(" ", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
        return text

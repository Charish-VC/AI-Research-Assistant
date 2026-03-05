"""Recursive text chunker with section-awareness.

Splits text into token-limited chunks, preferring to break at paragraph
and section boundaries.  Uses ``tiktoken`` for accurate token counting.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import tiktoken

from src.models import Chunk, DocumentMetadata

logger = logging.getLogger(__name__)

# Separators ordered by preference (split on the first that works)
_SEPARATORS: list[str] = [
    "\n\n",  # paragraph boundary
    "\n",    # line boundary
    ". ",    # sentence boundary
    " ",     # word boundary
]


class RecursiveChunker:
    """Split text into overlapping, token-limited chunks.

    Args:
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        encoding_name: ``tiktoken`` encoding to use for token counting.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = tiktoken.get_encoding(encoding_name)

    # ── Public API ───────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: DocumentMetadata | None = None,
    ) -> list[Chunk]:
        """Split *text* into ``Chunk`` objects.

        Args:
            text: Cleaned text to chunk.
            doc_id: Parent document ID to attach to each chunk.
            metadata: Document-level metadata to propagate.

        Returns:
            Ordered list of ``Chunk`` instances.
        """
        if not text.strip():
            return []

        raw_chunks = self._recursive_split(text, _SEPARATORS)
        merged = self._merge_with_overlap(raw_chunks)

        chunks: list[Chunk] = []
        now = datetime.now(timezone.utc).isoformat()

        for idx, chunk_text in enumerate(merged):
            token_count = len(self._encoder.encode(chunk_text))
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    chunk_index=idx,
                    text=chunk_text,
                    token_count=token_count,
                    metadata=metadata or DocumentMetadata(),
                    source=metadata.source if metadata else "",
                    created_at=now,
                )
            )

        logger.info(
            "Chunked doc %s into %d chunks (size=%d, overlap=%d)",
            doc_id[:8],
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._encoder.encode(text))

    # ── Private splitting logic ──────────────────────────────────────────

    def _recursive_split(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """Recursively split text using progressively finer separators."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if self.count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # If this single part exceeds chunk_size, split it further
                if self.count_tokens(part) > self.chunk_size:
                    result.extend(self._recursive_split(part, remaining_seps))
                else:
                    current = part

        if current:
            result.append(current)

        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Re-merge chunks to respect overlap between consecutive pieces.

        Adds ``chunk_overlap`` tokens from the end of chunk *i* to the
        beginning of chunk *i+1*.
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_tokens = self._encoder.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap :]
            overlap_text = self._encoder.decode(overlap_tokens)

            merged_text = overlap_text + " " + chunks[i]
            # Trim if the merged text is now too large
            while self.count_tokens(merged_text) > self.chunk_size:
                # Drop tokens from the overlap prefix
                words = merged_text.split(" ", 1)
                if len(words) > 1:
                    merged_text = words[1]
                else:
                    break

            merged.append(merged_text)

        return merged

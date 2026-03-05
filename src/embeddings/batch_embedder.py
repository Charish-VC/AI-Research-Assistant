"""Batch embedding wrapper with retry logic and progress tracking.

Wraps any ``BaseEmbedder`` to add batching, exponential backoff retries,
and progress logging.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.models import Chunk, EmbeddedChunk

if TYPE_CHECKING:
    from src.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class BatchEmbedder:
    """Batch-process chunks through an embedder with retry logic.

    Args:
        embedder: An instance of ``BaseEmbedder`` (e.g. ``OllamaEmbedder``).
        batch_size: Number of chunks to embed in one batch.
        max_retries: Maximum retry attempts per batch on failure.
        retry_delay: Initial delay (seconds) before retrying; doubles
            on each subsequent attempt.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.embedder = embedder
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed all chunks with batching, retries, and progress logging.

        Args:
            chunks: List of ``Chunk`` instances whose ``.text`` will be
                embedded.

        Returns:
            List of ``EmbeddedChunk`` instances (same order as input).
        """
        results: list[EmbeddedChunk] = []
        total = len(chunks)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = chunks[start:end]
            batch_num = (start // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            logger.info(
                "Embedding batch %d/%d  (%d chunks)",
                batch_num,
                total_batches,
                len(batch),
            )

            texts = [c.text for c in batch]
            embeddings = await self._embed_with_retry(texts)

            for chunk, embedding in zip(batch, embeddings):
                results.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

        logger.info("Finished embedding %d chunks", total)
        return results

    # ── Private helpers ──────────────────────────────────────────────────

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with exponential backoff on failure.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            Exception: Re-raises the last exception after all retries fail.
        """
        delay = self.retry_delay
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.embedder.embed_batch(texts)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Embedding attempt %d/%d failed: %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} attempts"
        ) from last_exc

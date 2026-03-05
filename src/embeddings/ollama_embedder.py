"""Ollama embedding client for ``nomic-embed-text``.

Calls the local Ollama HTTP API to generate 768-dimensional embeddings.
"""

from __future__ import annotations

import logging

import httpx
import numpy as np

from src.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class OllamaEmbedder(BaseEmbedder):
    """Generate embeddings via the Ollama ``/api/embeddings`` endpoint.

    Args:
        host: Ollama server URL (e.g. ``http://localhost:11434``).
        model: Name of the embedding model.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 120.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._dimension = 768  # nomic-embed-text output dimension
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (768 for nomic-embed-text)."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string via Ollama.

        The resulting vector is L2-normalised so that cosine similarity
        can be computed via inner product.

        Args:
            text: Input text to embed.

        Returns:
            768-dimensional normalised embedding vector.

        Raises:
            httpx.HTTPStatusError: If the Ollama API returns a non-2xx status.
        """
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        embedding = response.json()["embedding"]
        return self._normalise(embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts sequentially.

        Ollama does not natively support batch embedding, so we call
        the single-embed endpoint for each text.

        Args:
            texts: List of input strings.

        Returns:
            List of normalised embedding vectors.
        """
        embeddings: list[list[float]] = []
        for text in texts:
            vec = await self.embed(text)
            embeddings.append(vec)
        return embeddings

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalise(vector: list[float]) -> list[float]:
        """L2-normalise a vector for inner-product cosine similarity."""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

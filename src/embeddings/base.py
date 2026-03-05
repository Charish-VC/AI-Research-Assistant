"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Contract for all embedding implementations.

    Every embedder must support embedding a single text and a batch of
    texts.  Implementations may be local (Ollama) or remote (OpenAI).
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: Input string to embed.

        Returns:
            Embedding vector as a list of floats.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input string.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""

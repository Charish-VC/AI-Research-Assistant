"""Ollama generation client for RAG answer synthesis.

Calls the local Ollama HTTP API to generate natural language answers
grounded in retrieved document chunks.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are a research assistant. Answer the question using \
ONLY the context provided below. If the answer cannot be \
found in the context, say "I cannot find this information \
in the uploaded documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

NO_CONTEXT_ANSWER = (
    "I cannot find this information in the uploaded documents."
)


class OllamaGenerator:
    """Generate answers from retrieved chunks using Ollama's ``/api/generate`` endpoint.

    Args:
        host: Ollama server URL (e.g. ``http://localhost:11434``).
        model: Name of the generation model (e.g. ``llama3``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.generation_model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        """Build a RAG prompt and generate an answer via Ollama.

        Args:
            question: The user's question.
            chunks: List of retrieved chunk dicts, each containing at
                    least a ``text`` key.

        Returns:
            The generated answer string.
        """
        # If no chunks provided, return the no-context answer directly
        if not chunks:
            logger.info("No relevant chunks provided — returning default answer")
            return NO_CONTEXT_ANSWER

        # Build context from chunk texts
        context_parts: list[str] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                context_parts.append(text)

        if not context_parts:
            logger.info("All chunks empty — returning default answer")
            return NO_CONTEXT_ANSWER

        context = "\n\n".join(context_parts)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        logger.info(
            "Generating answer with model=%s, context_chunks=%d",
            self.model,
            len(context_parts),
        )

        try:
            return await self._call_ollama(prompt)
        except httpx.ConnectError as exc:
            logger.error("Cannot connect to Ollama at %s: %s", self.host, exc)
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Ensure Ollama is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama returned error: %s", exc.response.text)
            raise
        except httpx.TimeoutException as exc:
            logger.error("Ollama request timed out: %s", exc)
            raise

    async def _call_ollama(self, prompt: str) -> str:
        """Send a generation request to Ollama and return the response text."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        answer = data.get("response", "").strip()
        logger.info("Generation complete (%d chars)", len(answer))
        return answer

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

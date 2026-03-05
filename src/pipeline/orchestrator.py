"""Pipeline orchestrator — runs the full document‑to‑vector pipeline.

Sequences: extract → clean → chunk → enrich metadata → embed → store.
Tracks status and logs each stage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.config.settings import Settings, get_settings
from src.embeddings.batch_embedder import BatchEmbedder
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.models import (
    Chunk,
    Document,
    PipelineResult,
    ProcessingStatus,
    SourceType,
)
from src.pipeline.chunkers.recursive_chunker import RecursiveChunker
from src.pipeline.cleaners.text_cleaner import TextCleaner
from src.pipeline.extractors.html_extractor import HTMLExtractor
from src.pipeline.extractors.markdown_extractor import MarkdownExtractor
from src.pipeline.extractors.pdf_extractor import PDFExtractor
from src.pipeline.extractors.text_extractor import TextExtractor
from src.pipeline.metadata.metadata_extractor import MetadataExtractor
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

# File extension → extractor mapping
_EXTRACTOR_MAP: dict[str, type] = {
    ".pdf": PDFExtractor,
    ".md": MarkdownExtractor,
    ".markdown": MarkdownExtractor,
    ".html": HTMLExtractor,
    ".htm": HTMLExtractor,
    ".txt": TextExtractor,
}


class PipelineOrchestrator:
    """End-to-end ingestion pipeline: file → vectors in FAISS.

    Usage::

        orch = PipelineOrchestrator()
        result = await orch.run("path/to/paper.pdf")
        print(result.status, result.chunk_count)

    Args:
        settings: Application settings.  Defaults to the singleton.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        # Stage components
        self._cleaner = TextCleaner()
        self._chunker = RecursiveChunker(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        self._metadata_extractor = MetadataExtractor()

        # Embedding + vector store
        self._embedder = OllamaEmbedder(
            host=self._settings.ollama_host,
            model=self._settings.ollama_embedding_model,
            timeout=self._settings.ollama_request_timeout,
        )
        self._batch_embedder = BatchEmbedder(self._embedder)
        self._vector_store = FAISSStore(
            dimension=self._settings.faiss_dimension,
            index_dir=self._settings.faiss_index_path,
        )

        # In-memory document registry for status tracking
        self._results: dict[str, PipelineResult] = {}

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def vector_store(self) -> FAISSStore:
        """Expose the vector store for direct queries."""
        return self._vector_store

    @property
    def embedder(self) -> OllamaEmbedder:
        """Expose the embedder for query embedding."""
        return self._embedder

    def get_result(self, doc_id: str) -> PipelineResult | None:
        """Return the pipeline result for a given document ID."""
        return self._results.get(doc_id)

    def remove_result(self, doc_id: str) -> None:
        """Remove a pipeline result from the in-memory registry."""
        self._results.pop(doc_id, None)

    async def run(self, file_path: str | Path) -> PipelineResult:
        """Execute the full pipeline on a single file.

        Stages:
            1. **Extract** — read text from the file
            2. **Clean** — normalise and strip references
            3. **Chunk** — split into overlapping token-limited pieces
            4. **Metadata** — enrich metadata heuristically
            5. **Embed** — generate vectors via Ollama
            6. **Store** — add to FAISS index and persist

        Args:
            file_path: Path to the document file.

        Returns:
            ``PipelineResult`` with status, chunk count, and timing.
        """
        path = Path(file_path)
        start = time.time()
        doc_id = ""

        try:
            # ── Stage 1: Extract ─────────────────────────────────────
            logger.info("▸ Stage 1/6 — Extracting text from %s", path.name)
            extractor = self._get_extractor(path)
            document: Document = extractor.extract(path)
            doc_id = document.doc_id
            document.status = ProcessingStatus.PROCESSING
            logger.info(
                "  Extracted %d characters from %s",
                len(document.raw_text),
                path.name,
            )

            # ── Stage 2: Clean ───────────────────────────────────────
            logger.info("▸ Stage 2/6 — Cleaning text")
            cleaned_text = self._cleaner.clean(document.raw_text)
            logger.info(
                "  Cleaned: %d → %d chars",
                len(document.raw_text),
                len(cleaned_text),
            )

            # ── Stage 3: Chunk ───────────────────────────────────────
            logger.info("▸ Stage 3/6 — Chunking text")
            chunks: list[Chunk] = self._chunker.chunk(
                cleaned_text,
                doc_id=document.doc_id,
                metadata=document.metadata,
            )
            logger.info("  Generated %d chunks", len(chunks))

            # ── Stage 4: Metadata enrichment ─────────────────────────
            logger.info("▸ Stage 4/6 — Enriching metadata")
            enriched_meta = self._metadata_extractor.extract(
                document.raw_text, document.metadata
            )
            document.metadata = enriched_meta
            for chunk in chunks:
                chunk.metadata = enriched_meta

            # ── Stage 5: Embed ───────────────────────────────────────
            logger.info("▸ Stage 5/6 — Generating embeddings")
            embedded_chunks = await self._batch_embedder.embed_chunks(chunks)

            # ── Stage 6: Store ───────────────────────────────────────
            logger.info("▸ Stage 6/6 — Storing in FAISS")
            added = self._vector_store.add(embedded_chunks)
            self._vector_store.save()

            duration = time.time() - start
            result = PipelineResult(
                doc_id=document.doc_id,
                filename=path.name,
                status=ProcessingStatus.COMPLETED,
                chunk_count=added,
                duration_seconds=round(duration, 2),
            )
            logger.info(
                "✔ Pipeline completed for %s — %d chunks in %.1fs",
                path.name,
                added,
                duration,
            )

        except Exception as exc:
            duration = time.time() - start
            result = PipelineResult(
                doc_id=doc_id or "unknown",
                filename=path.name,
                status=ProcessingStatus.FAILED,
                error=str(exc),
                duration_seconds=round(duration, 2),
            )
            logger.error("✘ Pipeline failed for %s: %s", path.name, exc)

        self._results[result.doc_id] = result
        return result

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ):
        """Embed a query and search the vector store.

        Args:
            query: User query string.
            top_k: Number of results to return.
            threshold: Minimum similarity score.

        Returns:
            List of ``RetrievalResult`` instances.
        """
        k = top_k or self._settings.retrieval_top_k
        thr = threshold if threshold is not None else self._settings.similarity_threshold

        query_embedding = await self._embedder.embed(query)
        return self._vector_store.search(query_embedding, top_k=k, threshold=thr)

    async def close(self) -> None:
        """Release resources (HTTP clients, etc.)."""
        await self._embedder.close()

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_extractor(path: Path):
        """Return the appropriate extractor for a file path."""
        ext = path.suffix.lower()
        extractor_cls = _EXTRACTOR_MAP.get(ext)
        if extractor_cls is None:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(_EXTRACTOR_MAP.keys())}"
            )
        return extractor_cls()

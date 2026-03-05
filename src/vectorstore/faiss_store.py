"""FAISS vector store — index lifecycle, search, and persistence.

Uses ``IndexFlatIP`` (inner-product) for exact cosine similarity search
on L2-normalised vectors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from src.models import Chunk, EmbeddedChunk, RetrievalResult
from src.vectorstore.id_mapper import IDMapper

logger = logging.getLogger(__name__)


class FAISSStore:
    """Manage a FAISS index together with its chunk-ID mapping.

    Args:
        dimension: Embedding vector dimensionality (768 for nomic-embed-text).
        index_dir: Directory for persisting the index and ID map.
    """

    INDEX_FILENAME = "index.faiss"
    MAP_FILENAME = "id_map.json"
    CHUNKS_FILENAME = "chunks.json"
    HASH_FILENAME = "doc_hashes.json"

    def __init__(self, dimension: int = 768, index_dir: str = "data/faiss_index") -> None:
        self.dimension = dimension
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._index_dir / self.INDEX_FILENAME
        self._map_path = self._index_dir / self.MAP_FILENAME
        self._chunks_path = self._index_dir / self.CHUNKS_FILENAME
        self._hash_path = self._index_dir / self.HASH_FILENAME

        # Chunk metadata cache: chunk_id → Chunk
        self._chunks: dict[str, Chunk] = {}

        # Document content hashes: sha256_hex → doc_id
        self._doc_hashes: dict[str, str] = {}

        # Load or create
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._id_mapper = IDMapper(self._map_path)
            self._load_chunks()
            self._load_hashes()
            logger.info(
                "Loaded FAISS index with %d vectors, %d chunks from %s",
                self._index.ntotal,
                len(self._chunks),
                self._index_path,
            )
        else:
            self._index = faiss.IndexFlatIP(dimension)
            self._id_mapper = IDMapper(self._map_path)
            self._load_hashes()  # hashes may exist even without index
            logger.info("Created new FAISS IndexFlatIP (dim=%d)", dimension)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Return the number of vectors currently in the index."""
        return self._index.ntotal

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to the index.

        Args:
            embedded_chunks: List of ``EmbeddedChunk`` instances.

        Returns:
            Number of vectors successfully added.
        """
        if not embedded_chunks:
            return 0

        vectors = np.array(
            [ec.embedding for ec in embedded_chunks], dtype=np.float32
        )

        # Register each chunk in the ID mapper and metadata cache
        for ec in embedded_chunks:
            self._id_mapper.add(ec.chunk.chunk_id)
            self._chunks[ec.chunk.chunk_id] = ec.chunk

        self._index.add(vectors)

        logger.info(
            "Added %d vectors to FAISS index (total: %d)",
            len(embedded_chunks),
            self._index.ntotal,
        )
        return len(embedded_chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        """Search for the most similar chunks to a query vector.

        Args:
            query_embedding: Normalised query embedding vector.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score to include in results.

        Returns:
            Ordered list of ``RetrievalResult`` (highest score first).
        """
        if self._index.ntotal == 0:
            logger.warning("Search called on empty FAISS index")
            return []

        query = np.array([query_embedding], dtype=np.float32)
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or score < threshold:
                continue

            chunk_id = self._id_mapper.get_chunk_id(int(idx))
            if chunk_id is None:
                logger.warning("No chunk_id found for FAISS position %d", idx)
                continue

            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                logger.warning("Chunk metadata not found for %s", chunk_id)
                continue

            results.append(
                RetrievalResult(chunk=chunk, score=float(score), rank=rank)
            )

        return results

    def save(self) -> None:
        """Persist the FAISS index, ID map, chunk metadata, and hashes to disk."""
        faiss.write_index(self._index, str(self._index_path))
        self._id_mapper.save()
        self._save_chunks()
        self._save_hashes()
        logger.info(
            "Saved FAISS index (%d vectors, %d chunks) to %s",
            self._index.ntotal,
            len(self._chunks),
            self._index_path,
        )

    def clear(self) -> None:
        """Remove all vectors and reset the index."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_mapper = IDMapper(self._map_path)
        self._chunks.clear()
        self._doc_hashes.clear()
        # Remove persisted files
        for p in (self._chunks_path, self._hash_path, self._index_path, self._map_path):
            if p.exists():
                p.unlink()
        logger.info("Cleared FAISS index")

    def load_chunk_metadata(self, chunks: list[Chunk]) -> None:
        """Populate the in-memory chunk metadata cache.

        Call this after loading an index from disk to restore the
        ability to return full ``Chunk`` objects from search results.

        Args:
            chunks: List of ``Chunk`` instances to cache.
        """
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        logger.info("Loaded %d chunk metadata entries into cache", len(chunks))

    # ── Document hash management (duplicate detection) ───────────────────

    def has_hash(self, content_hash: str) -> bool:
        """Check if a document with this content hash has been ingested."""
        return content_hash in self._doc_hashes

    def get_doc_id_by_hash(self, content_hash: str) -> str | None:
        """Return the doc_id associated with a content hash, if any."""
        return self._doc_hashes.get(content_hash)

    def add_hash(self, content_hash: str, doc_id: str) -> None:
        """Register a content hash → doc_id mapping."""
        self._doc_hashes[content_hash] = doc_id

    def remove_hash_by_doc_id(self, doc_id: str) -> None:
        """Remove the hash entry for a given doc_id."""
        to_remove = [h for h, d in self._doc_hashes.items() if d == doc_id]
        for h in to_remove:
            del self._doc_hashes[h]

    # ── Document deletion ────────────────────────────────────────────────

    def remove_document(self, doc_id: str) -> int:
        """Remove all vectors for a document and rebuild the FAISS index.

        FAISS IndexFlatIP does not support individual vector removal,
        so we rebuild the index from the remaining vectors.

        Args:
            doc_id: The document ID whose chunks should be removed.

        Returns:
            Number of vectors removed.
        """
        # Find chunk_ids belonging to this document
        chunk_ids_to_remove = {
            cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id
        }
        if not chunk_ids_to_remove:
            return 0

        # Collect positions to keep
        keep_positions: list[int] = []
        keep_chunk_ids: list[str] = []
        for pos in range(self._index.ntotal):
            cid = self._id_mapper.get_chunk_id(pos)
            if cid and cid not in chunk_ids_to_remove:
                keep_positions.append(pos)
                keep_chunk_ids.append(cid)

        # Rebuild index with remaining vectors
        if keep_positions:
            remaining_vectors = np.array(
                [self._index.reconstruct(int(pos)) for pos in keep_positions],
                dtype=np.float32,
            )
            new_index = faiss.IndexFlatIP(self.dimension)
            new_index.add(remaining_vectors)
        else:
            new_index = faiss.IndexFlatIP(self.dimension)

        # Rebuild ID mapper
        new_mapper = IDMapper(self._map_path)
        for cid in keep_chunk_ids:
            new_mapper.add(cid)

        # Remove chunk metadata
        for cid in chunk_ids_to_remove:
            self._chunks.pop(cid, None)

        # Remove document hash
        self.remove_hash_by_doc_id(doc_id)

        removed = len(chunk_ids_to_remove)
        self._index = new_index
        self._id_mapper = new_mapper

        logger.info(
            "Removed %d vectors for doc %s (remaining: %d)",
            removed,
            doc_id[:12],
            self._index.ntotal,
        )
        return removed

    # ── Private persistence helpers ──────────────────────────────────────

    def _save_chunks(self) -> None:
        """Serialize chunk metadata to JSON."""
        data = {cid: chunk.model_dump() for cid, chunk in self._chunks.items()}
        self._chunks_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def _load_chunks(self) -> None:
        """Deserialize chunk metadata from JSON."""
        if not self._chunks_path.exists():
            logger.debug("No chunks.json found at %s", self._chunks_path)
            return
        raw = json.loads(self._chunks_path.read_text(encoding="utf-8"))
        self._chunks = {cid: Chunk.model_validate(data) for cid, data in raw.items()}
        logger.info("Loaded %d chunk metadata entries from %s", len(self._chunks), self._chunks_path)

    def _save_hashes(self) -> None:
        """Persist document content hashes to JSON."""
        self._hash_path.write_text(
            json.dumps(self._doc_hashes, indent=2), encoding="utf-8"
        )

    def _load_hashes(self) -> None:
        """Load document content hashes from JSON."""
        if not self._hash_path.exists():
            return
        self._doc_hashes = json.loads(self._hash_path.read_text(encoding="utf-8"))
        logger.info("Loaded %d document hashes from %s", len(self._doc_hashes), self._hash_path)

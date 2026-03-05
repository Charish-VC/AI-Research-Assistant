"""Bidirectional mapping between FAISS integer positions and chunk IDs.

FAISS indexes vectors by contiguous integer position (0, 1, 2, …).
We need to map those positions back to our string ``chunk_id`` UUIDs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IDMapper:
    """Maintain a two-way map: FAISS position ↔ chunk_id.

    The mapping is persisted to a JSON file alongside the FAISS index.

    Args:
        path: File path for persistence (``id_map.json``).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._pos_to_id: dict[int, str] = {}
        self._id_to_pos: dict[str, int] = {}
        self._next_pos: int = 0
        self._path = path

        if path and path.exists():
            self.load(path)

    # ── Public API ───────────────────────────────────────────────────────

    def add(self, chunk_id: str) -> int:
        """Register a new chunk_id and return its FAISS position.

        Args:
            chunk_id: UUID string for the chunk.

        Returns:
            Integer position assigned in the FAISS index.
        """
        if chunk_id in self._id_to_pos:
            return self._id_to_pos[chunk_id]

        pos = self._next_pos
        self._pos_to_id[pos] = chunk_id
        self._id_to_pos[chunk_id] = pos
        self._next_pos += 1
        return pos

    def get_chunk_id(self, position: int) -> str | None:
        """Return the chunk_id at the given FAISS position."""
        return self._pos_to_id.get(position)

    def get_position(self, chunk_id: str) -> int | None:
        """Return the FAISS position for a given chunk_id."""
        return self._id_to_pos.get(chunk_id)

    @property
    def size(self) -> int:
        """Return the number of mapped entries."""
        return len(self._pos_to_id)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Persist the mapping to a JSON file.

        Args:
            path: Output file path.  Falls back to the path given at
                construction time.
        """
        target = path or self._path
        if not target:
            raise ValueError("No path specified for saving ID map")

        # JSON keys must be strings
        data = {
            "pos_to_id": {str(k): v for k, v in self._pos_to_id.items()},
            "next_pos": self._next_pos,
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved ID map (%d entries) to %s", self.size, target)

    def load(self, path: Path | None = None) -> None:
        """Load the mapping from a JSON file.

        Args:
            path: Input file path.  Falls back to the path given at
                construction time.
        """
        target = path or self._path
        if not target or not target.exists():
            logger.debug("No ID map file to load from %s", target)
            return

        data = json.loads(target.read_text(encoding="utf-8"))
        self._pos_to_id = {int(k): v for k, v in data["pos_to_id"].items()}
        self._id_to_pos = {v: int(k) for k, v in data["pos_to_id"].items()}
        self._next_pos = data.get("next_pos", len(self._pos_to_id))
        logger.info("Loaded ID map (%d entries) from %s", self.size, target)

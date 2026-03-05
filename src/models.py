"""Shared data models used across the pipeline.

These are plain Pydantic models (not ORM or DB models) that flow through
every stage of the pipeline — from ingestion to retrieval.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Supported document source types."""

    PDF = "pdf"
    MARKDOWN = "md"
    HTML = "html"
    TXT = "txt"


class ProcessingStatus(str, Enum):
    """Document lifecycle states."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── Document-level models ───────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    date: str = ""
    source: str = ""
    keywords: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Represents an ingested document with its raw text."""

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    source_type: SourceType
    raw_text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Chunk-level models ──────────────────────────────────────────────────


class Chunk(BaseModel):
    """A single text chunk produced by the chunking stage."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    chunk_index: int
    text: str
    token_count: int = 0
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    source: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Embedding / retrieval models ────────────────────────────────────────


class EmbeddedChunk(BaseModel):
    """A chunk together with its embedding vector."""

    chunk: Chunk
    embedding: list[float]


class RetrievalResult(BaseModel):
    """A single result from a vector similarity search."""

    chunk: Chunk
    score: float
    rank: int


# ── Pipeline status tracking ────────────────────────────────────────────


class PipelineResult(BaseModel):
    """Summary of a pipeline run for one document."""

    doc_id: str
    filename: str
    status: ProcessingStatus
    chunk_count: int = 0
    error: str | None = None
    duration_seconds: float = 0.0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

"""Pydantic request and response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Ingest ───────────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    """Response returned after a document is submitted for ingestion."""

    doc_id: str
    filename: str
    status: str
    message: str


# ── Query ────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Body for the ``POST /query`` endpoint."""

    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score. If omitted, uses SIMILARITY_THRESHOLD from config.",
    )


class ChunkResult(BaseModel):
    """A single retrieved chunk in a query response."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    source: str
    metadata: dict


class QueryResponse(BaseModel):
    """Response for a retrieval query."""

    query: str
    results: list[ChunkResult]
    total: int


# ── Status ───────────────────────────────────────────────────────────────


class PipelineStatusResponse(BaseModel):
    """Status of a pipeline run."""

    doc_id: str
    filename: str
    status: str
    chunk_count: int
    error: str | None = None
    duration_seconds: float


# ── Delete ───────────────────────────────────────────────────────────────


class DeleteResponse(BaseModel):
    """Response for document deletion."""

    doc_id: str
    chunks_removed: int
    message: str


# ── Health ───────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Application health check response."""

    status: str
    version: str
    index_size: int
    ollama_available: bool


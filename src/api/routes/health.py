"""Health check endpoint."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline
from src.api.schemas import HealthResponse
from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
)
async def health_check(
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> HealthResponse:
    """Return application health status, FAISS index size, and
    Ollama availability.
    """
    ollama_ok = await _check_ollama(pipeline)
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        version="0.1.0",
        index_size=pipeline.vector_store.size,
        ollama_available=ollama_ok,
    )


async def _check_ollama(pipeline: PipelineOrchestrator) -> bool:
    """Ping the Ollama API to verify it is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{pipeline.embedder.host}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False

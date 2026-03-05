"""Query endpoint — embed a question and retrieve relevant chunks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline
from src.api.schemas import ChunkResult, QueryRequest, QueryResponse
from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Search the knowledge base",
)
async def query_knowledge_base(
    body: QueryRequest,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> QueryResponse:
    """Embed the query via Ollama and perform a similarity search
    against the FAISS vector store.  Returns the top-k most relevant
    document chunks with their similarity scores.
    """
    results = await pipeline.search(
        query=body.query,
        top_k=body.top_k,
        threshold=body.threshold,  # None → uses SIMILARITY_THRESHOLD from config
    )

    chunks = [
        ChunkResult(
            chunk_id=r.chunk.chunk_id,
            doc_id=r.chunk.doc_id,
            text=r.chunk.text,
            score=round(r.score, 4),
            rank=r.rank,
            source=r.chunk.source,
            metadata=r.chunk.metadata.model_dump(),
        )
        for r in results
    ]

    return QueryResponse(
        query=body.query,
        results=chunks,
        total=len(chunks),
    )

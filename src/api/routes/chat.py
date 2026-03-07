"""Chat endpoint — full RAG pipeline: embed → retrieve → generate."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_pipeline
from src.api.schemas import ChatRequest, ChatResponse, ChatSource
from src.pipeline.generation.generator import OllamaGenerator
from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

# Module-level generator singleton (created on first use)
_generator: OllamaGenerator | None = None


def _get_generator() -> OllamaGenerator:
    """Return a singleton OllamaGenerator instance."""
    global _generator
    if _generator is None:
        _generator = OllamaGenerator()
    return _generator


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question and get an AI-generated answer",
)
async def chat(
    body: ChatRequest,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> ChatResponse:
    """Full RAG pipeline:

    1. Embed the question using the existing embedding pipeline.
    2. Retrieve top-k chunks above the similarity threshold.
    3. Pass the question + chunks to OllamaGenerator for answer synthesis.
    4. Return the generated answer alongside source chunks.
    """
    # Step 1 & 2: Retrieve relevant chunks
    results = await pipeline.search(
        query=body.question,
        top_k=body.top_k,
        threshold=body.threshold,
    )

    # Build chunk dicts for the generator
    chunk_dicts = [
        {"text": r.chunk.text}
        for r in results
    ]

    # Step 3: Generate answer
    generator = _get_generator()
    try:
        answer = await generator.generate(
            question=body.question,
            chunks=chunk_dicts,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {exc}",
        ) from exc

    # Step 4: Build response
    sources = [
        ChatSource(
            chunk_id=r.chunk.chunk_id,
            doc_id=r.chunk.doc_id,
            source=r.chunk.source,
            score=round(r.score, 4),
            text=r.chunk.text,
        )
        for r in results
    ]

    return ChatResponse(
        question=body.question,
        answer=answer,
        sources=sources,
        total_sources=len(sources),
    )

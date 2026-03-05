"""Pipeline status and document management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_pipeline
from src.api.schemas import DeleteResponse, PipelineStatusResponse
from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])


@router.get(
    "/status/{doc_id}",
    response_model=PipelineStatusResponse,
    summary="Get pipeline processing status",
)
async def get_status(
    doc_id: str,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> PipelineStatusResponse:
    """Return the processing status of a previously ingested document."""
    result = pipeline.get_result(doc_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pipeline run found for doc_id '{doc_id}'",
        )

    return PipelineStatusResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        status=result.status.value,
        chunk_count=result.chunk_count,
        error=result.error,
        duration_seconds=result.duration_seconds,
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document and its vectors",
)
async def delete_document(
    document_id: str,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> DeleteResponse:
    """Remove all chunks and vectors for a document from the FAISS index.

    Also removes the content hash so the document can be re-ingested.
    """
    removed = pipeline.vector_store.remove_document(document_id)

    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for document_id '{document_id}'",
        )

    # Persist the updated index
    pipeline.vector_store.save()

    # Clean up in-memory pipeline result
    pipeline.remove_result(document_id)

    return DeleteResponse(
        doc_id=document_id,
        chunks_removed=removed,
        message=f"Removed {removed} chunks and vectors for document {document_id}",
    )

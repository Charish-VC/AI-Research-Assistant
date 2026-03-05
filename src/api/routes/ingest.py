"""Document ingestion endpoint.

Accepts file uploads, saves them to local storage, and kicks off the
processing pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from src.api.dependencies import get_pipeline
from src.api.schemas import IngestResponse
from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingest"])

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process a document",
)
async def ingest_document(
    file: UploadFile,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> IngestResponse:
    """Accept a document upload, persist it locally, and run the
    full processing pipeline (extract → clean → chunk → embed → store).

    Supported file types: PDF, Markdown, HTML, TXT.
    Returns 409 Conflict if the document has already been ingested.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save upload to a temp file, then run the pipeline
    tmp_dir = Path(tempfile.mkdtemp(prefix="airip_"))
    tmp_path = tmp_dir / file.filename

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ── Duplicate detection ──────────────────────────────────────
        content_hash = _compute_file_hash(tmp_path)
        if pipeline.vector_store.has_hash(content_hash):
            existing_doc_id = pipeline.vector_store.get_doc_id_by_hash(content_hash)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document already ingested (duplicate detected). "
                    f"Existing doc_id: {existing_doc_id}"
                ),
            )

        result = await pipeline.run(tmp_path)

        # Register hash on successful ingestion
        if result.status.value == "COMPLETED":
            pipeline.vector_store.add_hash(content_hash, result.doc_id)
            pipeline.vector_store.save()

        return IngestResponse(
            doc_id=result.doc_id,
            filename=result.filename,
            status=result.status.value,
            message=(
                f"Processed {result.chunk_count} chunks in {result.duration_seconds}s"
                if result.status.value == "COMPLETED"
                else f"Pipeline failed: {result.error}"
            ),
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as exc:
        logger.exception("Ingest error for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        # Cleanup temp file
        shutil.rmtree(tmp_dir, ignore_errors=True)


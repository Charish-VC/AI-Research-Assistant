"""Document ingestion endpoint.

Accepts file uploads, saves them to local storage, and kicks off the
processing pipeline. After successful ingestion all AWS side-effects
(S3 upload, DynamoDB record, SQS message) are dispatched as
non-blocking background tasks so they never delay the HTTP response.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status

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


def _aws_background_tasks(
    doc_id: str,
    filename: str,
    source: str,
    chunk_count: int,
    tmp_path: Path,
    tmp_dir: Path,
    chunks_data: list[dict[str, Any]],
    meta_dict: dict[str, Any],
) -> None:
    """Run all 5 AWS operations in order and then clean up the temp file.

    This function is called as a FastAPI BackgroundTask and must never
    raise — all exceptions are caught and logged.
    """
    try:
        from src.aws.dynamodb_client import DynamoDBClient
        from src.aws.s3_client import S3Client
        from src.aws.sqs_client import SQSClient

        s3 = S3Client()
        dynamo = DynamoDBClient()
        sqs = SQSClient()

        # 1. Upload raw document to S3
        s3_raw_key = s3.upload_raw_document(doc_id, filename, tmp_path)
        s3_raw_path = (
            f"s3://{s3._bucket}/{s3_raw_key}" if s3_raw_key else ""
        )

        # 2. Upload chunks JSON to S3
        s3.upload_chunks(doc_id, chunks_data)

        # 3. Upload metadata JSON to S3
        s3.upload_metadata(doc_id, meta_dict)

        # 4. Write DynamoDB record
        dynamo.put_document(
            doc_id=doc_id,
            filename=filename,
            source=source,
            embedding_count=chunk_count,
            status="COMPLETED",
            s3_raw_path=s3_raw_path,
        )

        # 5. Send SQS notification
        sqs.send_message(
            doc_id=doc_id,
            filename=filename,
            s3_raw_path=s3_raw_path,
        )

        logger.info("AWS background tasks completed for doc_id=%s", doc_id)

    except Exception as exc:  # noqa: BLE001
        logger.error("AWS background task error for doc_id=%s: %s", doc_id, exc)
    finally:
        # Cleanup—deferred until after the AWS uploads finish
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process a document",
)
async def ingest_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> IngestResponse:
    """Accept a document upload, persist it locally, and run the
    full processing pipeline (extract → clean → chunk → embed → store).

    Supported file types: PDF, Markdown, HTML, TXT.
    Returns 409 Conflict if the document has already been ingested.
    AWS side-effects run asynchronously and do not block the response.
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

    # Save upload to a temp file
    tmp_dir = Path(tempfile.mkdtemp(prefix="airip_"))
    tmp_path = tmp_dir / file.filename

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ── Duplicate detection ──────────────────────────────────────────
        content_hash = _compute_file_hash(tmp_path)
        if pipeline.vector_store.has_hash(content_hash):
            existing_doc_id = pipeline.vector_store.get_doc_id_by_hash(content_hash)
            shutil.rmtree(tmp_dir, ignore_errors=True)
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

        # ── Schedule non-blocking AWS side-effects ───────────────────────
        if result.status.value == "COMPLETED":
            chunks_data: list[dict[str, Any]] = [
                {
                    "chunk_id": i,
                    "text": getattr(chunk, "text", str(chunk)),
                }
                for i, chunk in enumerate(getattr(result, "chunks", []))
            ]
            meta_dict: dict[str, Any] = {
                "doc_id": result.doc_id,
                "filename": result.filename,
                "chunk_count": result.chunk_count,
                "duration_seconds": result.duration_seconds,
                "content_hash": content_hash,
            }
            background_tasks.add_task(
                _aws_background_tasks,
                doc_id=result.doc_id,
                filename=result.filename,
                source=str(tmp_path),
                chunk_count=result.chunk_count,
                tmp_path=tmp_path,
                tmp_dir=tmp_dir,
                chunks_data=chunks_data,
                meta_dict=meta_dict,
            )
        else:
            # Pipeline failed — clean up immediately
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

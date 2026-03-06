"""S3 client for uploading raw documents, chunks, and metadata.

All public methods are safe to call from background tasks — they catch
all exceptions internally and never propagate to the caller.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import boto3
import boto3.exceptions
from botocore.exceptions import BotoCoreError, ClientError

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class S3Client:
    """Thin wrapper around boto3 S3 for the AI Research Platform."""

    def __init__(self) -> None:
        settings = get_settings()
        self._bucket = settings.s3_bucket_name
        self._client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        logger.info("S3Client initialised (bucket=%s)", self._bucket)

    # ── Public helpers ───────────────────────────────────────────────────

    def upload_raw_document(
        self,
        doc_id: str,
        filename: str,
        file_path: Path,
    ) -> str | None:
        """Upload the original file to s3://<bucket>/raw/<doc_id>/<filename>.

        Returns the S3 key on success, or None on failure.
        """
        key = f"raw/{doc_id}/{filename}"
        try:
            self._client.upload_file(str(file_path), self._bucket, key)
            logger.info("S3 raw upload ok: s3://%s/%s", self._bucket, key)
            return key
        except (BotoCoreError, ClientError, OSError, boto3.exceptions.S3UploadFailedError) as exc:
            logger.error("S3 raw upload failed for doc_id=%s: %s", doc_id, exc)
            return None

    def upload_chunks(
        self,
        doc_id: str,
        chunks_data: list[dict[str, Any]],
    ) -> str | None:
        """Upload chunk JSON to s3://<bucket>/embeddings/<doc_id>/chunks.json.

        Returns the S3 key on success, or None on failure.
        """
        key = f"embeddings/{doc_id}/chunks.json"
        try:
            body = json.dumps(chunks_data, ensure_ascii=False).encode("utf-8")
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            logger.info("S3 chunks upload ok: s3://%s/%s", self._bucket, key)
            return key
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 chunks upload failed for doc_id=%s: %s", doc_id, exc)
            return None

    def upload_metadata(
        self,
        doc_id: str,
        meta_dict: dict[str, Any],
    ) -> str | None:
        """Upload metadata JSON to s3://<bucket>/metadata/<doc_id>/meta.json.

        Returns the S3 key on success, or None on failure.
        """
        key = f"metadata/{doc_id}/meta.json"
        try:
            body = json.dumps(meta_dict, ensure_ascii=False).encode("utf-8")
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            logger.info("S3 metadata upload ok: s3://%s/%s", self._bucket, key)
            return key
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 metadata upload failed for doc_id=%s: %s", doc_id, exc)
            return None

    def object_exists(self, key: str) -> bool:
        """Return True if the given S3 key exists in the bucket."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

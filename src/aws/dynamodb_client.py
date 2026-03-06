"""DynamoDB client for storing per-document metadata records.

Table schema:
    Partition key : doc_id  (String)

If the table does not yet exist it is created on first use with PAY_PER_REQUEST
billing so there are no capacity planning concerns.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class DynamoDBClient:
    """Thin wrapper around boto3 DynamoDB for the AI Research Platform."""

    def __init__(self) -> None:
        settings = get_settings()
        self._table_name = settings.dynamodb_table_name
        resource = boto3.resource(
            "dynamodb",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        self._table = self._get_or_create_table(resource)
        logger.info("DynamoDBClient initialised (table=%s)", self._table_name)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _get_or_create_table(self, resource: Any) -> Any:
        """Return an existing table handle or create a new one."""
        try:
            table = resource.create_table(
                TableName=self._table_name,
                KeySchema=[{"AttributeName": "doc_id", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "doc_id", "AttributeType": "S"}
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
            logger.info("DynamoDB table created: %s", self._table_name)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ResourceInUseException":
                table = resource.Table(self._table_name)
                logger.info("DynamoDB table already exists: %s", self._table_name)
            else:
                raise
        return table

    # ── Public interface ─────────────────────────────────────────────────

    def put_document(
        self,
        doc_id: str,
        filename: str,
        source: str,
        embedding_count: int,
        status: str,
        s3_raw_path: str,
    ) -> bool:
        """Write a document record. Returns True on success, False on failure."""
        item: dict[str, Any] = {
            "doc_id": doc_id,
            "filename": filename,
            "source": source,
            "embedding_count": embedding_count,
            "status": status,
            "s3_raw_path": s3_raw_path,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
        try:
            self._table.put_item(Item=item)
            logger.info("DynamoDB put_item ok: doc_id=%s", doc_id)
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.error("DynamoDB put_item failed for doc_id=%s: %s", doc_id, exc)
            return False

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document record by doc_id. Returns None if not found or on error."""
        try:
            response = self._table.get_item(Key={"doc_id": doc_id})
            item = response.get("Item")
            if item:
                logger.info("DynamoDB get_item ok: doc_id=%s", doc_id)
            else:
                logger.info("DynamoDB get_item: doc_id=%s not found", doc_id)
            return item  # type: ignore[return-value]
        except (BotoCoreError, ClientError) as exc:
            logger.error("DynamoDB get_item failed for doc_id=%s: %s", doc_id, exc)
            return None

    def update_status(self, doc_id: str, new_status: str) -> bool:
        """Update the status field for a document. Returns True on success."""
        try:
            self._table.update_item(
                Key={"doc_id": doc_id},
                UpdateExpression="SET #st = :s, updated_at = :t",
                ExpressionAttributeNames={"#st": "status"},
                ExpressionAttributeValues={
                    ":s": new_status,
                    ":t": int(time.time()),
                },
            )
            logger.info(
                "DynamoDB update_status ok: doc_id=%s status=%s", doc_id, new_status
            )
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.error(
                "DynamoDB update_status failed for doc_id=%s: %s", doc_id, exc
            )
            return False

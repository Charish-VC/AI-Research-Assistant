"""SQS client for queuing document processing notifications.

On init the client resolves the queue URL by name so callers only
need to set SQS_QUEUE_NAME in the environment.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class SQSClient:
    """Thin wrapper around boto3 SQS for the AI Research Platform."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = boto3.client(
            "sqs",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        self._queue_url = self._resolve_queue_url(settings.sqs_queue_name)
        logger.info("SQSClient initialised (queue=%s)", settings.sqs_queue_name)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _resolve_queue_url(self, queue_name: str) -> str:
        """Resolve queue URL from queue name, creating the queue if absent."""
        try:
            response = self._client.get_queue_url(QueueName=queue_name)
            url: str = response["QueueUrl"]
            logger.info("SQS queue URL resolved: %s", url)
            return url
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "AWS.SimpleQueueService.NonExistentQueue":
                logger.info("SQS queue not found — creating: %s", queue_name)
                response = self._client.create_queue(QueueName=queue_name)
                url = response["QueueUrl"]
                logger.info("SQS queue created: %s", url)
                return url
            raise

    # ── Public interface ─────────────────────────────────────────────────

    def send_message(
        self,
        doc_id: str,
        filename: str,
        s3_raw_path: str,
    ) -> str | None:
        """Send a processing notification message.

        Returns the SQS MessageId on success, or None on failure.
        """
        body = json.dumps(
            {"doc_id": doc_id, "filename": filename, "s3_raw_path": s3_raw_path},
            ensure_ascii=False,
        )
        try:
            response = self._client.send_message(
                QueueUrl=self._queue_url,
                MessageBody=body,
            )
            msg_id: str = response["MessageId"]
            logger.info("SQS send_message ok: MessageId=%s doc_id=%s", msg_id, doc_id)
            return msg_id
        except (BotoCoreError, ClientError) as exc:
            logger.error("SQS send_message failed for doc_id=%s: %s", doc_id, exc)
            return None

    def receive_messages(
        self,
        max_messages: int = 1,
        wait_seconds: int = 5,
    ) -> list[dict[str, Any]]:
        """Receive up to *max_messages* messages with long-polling.

        Returns a list of raw SQS message dicts (may be empty).
        """
        try:
            response = self._client.receive_message(
                QueueUrl=self._queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_seconds,
                AttributeNames=["All"],
                MessageAttributeNames=["All"],
            )
            messages: list[dict[str, Any]] = response.get("Messages", [])
            logger.info("SQS receive_messages: got %d message(s)", len(messages))
            return messages
        except (BotoCoreError, ClientError) as exc:
            logger.error("SQS receive_messages failed: %s", exc)
            return []

    def delete_message(self, receipt_handle: str) -> bool:
        """Delete a message by its ReceiptHandle. Returns True on success."""
        try:
            self._client.delete_message(
                QueueUrl=self._queue_url,
                ReceiptHandle=receipt_handle,
            )
            logger.info("SQS delete_message ok")
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.error("SQS delete_message failed: %s", exc)
            return False

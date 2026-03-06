"""AWS Lambda worker for document processing.

Triggered by SQS events (batch size 1).  Each invocation:
  1. Downloads the raw document from S3
  2. Extracts text (PDF via PyPDF2, plain-text for md/txt/html)
  3. Cleans and chunks the text
  4. Generates embeddings via Amazon Bedrock (Titan Embed Text v1)
  5. Stores embeddings JSON back to S3
  6. Updates the DynamoDB processing status

On failure the DynamoDB status is set to FAILED and the error is logged.
The exception is *not* re-raised so SQS will delete the message
(avoids infinite retry loops).
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Optional — only needed for PDF extraction
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None  # type: ignore[assignment,misc]

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Environment ──────────────────────────────────────────────────────────
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "ai-research-assistant-dev")
DYNAMO_TABLE = os.environ.get("DYNAMODB_TABLE_NAME", "ai-research-documents")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Bedrock embedding model
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v1"
BEDROCK_EMBED_DIM = 1536  # Titan v1 output dimension

# Chunking defaults (simplified — no tiktoken in Lambda)
CHUNK_SIZE_CHARS = 3000  # rough character limit per chunk
CHUNK_OVERLAP_CHARS = 400

# ── AWS clients (created once per container via Lambda execution context) ──
s3_client = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
dynamo_table = dynamodb.Table(DYNAMO_TABLE)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ═══════════════════════════════════════════════════════════════════════════
# Lambda handler
# ═══════════════════════════════════════════════════════════════════════════

def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """SQS-triggered Lambda entry point.

    Expects a single SQS record whose body is a JSON object with keys:
      - doc_id   (str)
      - filename (str)
      - s3_raw_path (str)  — full S3 key, e.g. ``raw/<doc_id>/file.pdf``
    """
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        doc_id: str = body["doc_id"]
        filename: str = body["filename"]
        s3_raw_path: str = body["s3_raw_path"]

        logger.info(
            "Processing doc_id=%s filename=%s s3_raw_path=%s",
            doc_id,
            filename,
            s3_raw_path,
        )

        try:
            _process_document(doc_id, filename, s3_raw_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED doc_id=%s: %s", doc_id, exc, exc_info=True)
            _update_status(doc_id, "FAILED", error=str(exc))
            # Do NOT re-raise — let SQS delete the message

    return {"statusCode": 200, "body": "ok"}


# ═══════════════════════════════════════════════════════════════════════════
# Core processing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _process_document(doc_id: str, filename: str, s3_raw_path: str) -> None:
    """Full processing pipeline for a single document."""

    # ── 1. Download from S3 ──────────────────────────────────────────────
    local_dir = Path(f"/tmp/{doc_id}")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename

    # s3_raw_path may be "s3://bucket/key" or just "key"
    s3_key = s3_raw_path
    if s3_key.startswith("s3://"):
        # strip "s3://bucket/"
        s3_key = "/".join(s3_key.split("/")[3:])

    logger.info("Downloading s3://%s/%s → %s", S3_BUCKET, s3_key, local_path)
    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))

    # ── 2. Extract text ──────────────────────────────────────────────────
    logger.info("Extracting text from %s", filename)
    text = _extract_text(local_path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {filename}")
    logger.info("Extracted %d characters", len(text))

    # ── 3. Clean text ────────────────────────────────────────────────────
    text = _clean_text(text)
    logger.info("Cleaned text: %d characters", len(text))

    # ── 4. Chunk ─────────────────────────────────────────────────────────
    chunks = _chunk_text(text)
    logger.info("Generated %d chunks", len(chunks))

    # ── 5. Generate embeddings via Bedrock ────────────────────────────────
    logger.info("Generating embeddings via Bedrock (%s)", BEDROCK_MODEL_ID)
    embedded_chunks = _embed_chunks(chunks, doc_id)
    logger.info("Embedded %d chunks", len(embedded_chunks))

    # ── 6. Upload embeddings to S3 ───────────────────────────────────────
    embeddings_key = f"embeddings/{doc_id}/chunks.json"
    embeddings_body = json.dumps(embedded_chunks, ensure_ascii=False).encode("utf-8")
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=embeddings_key,
        Body=embeddings_body,
        ContentType="application/json",
    )
    logger.info("Uploaded embeddings to s3://%s/%s", S3_BUCKET, embeddings_key)

    # ── 7. Update DynamoDB → COMPLETED ───────────────────────────────────
    _update_status(doc_id, "COMPLETED")
    logger.info("Pipeline completed for doc_id=%s", doc_id)


# ═══════════════════════════════════════════════════════════════════════════
# Text extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_text(file_path: Path) -> str:
    """Extract text from a file based on its extension."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in {".md", ".markdown", ".txt", ".html", ".htm"}:
        return file_path.read_text(encoding="utf-8", errors="replace")
    else:
        # Fallback — try to read as plain text
        return file_path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(file_path: Path) -> str:
    """Extract text from a PDF using PyPDF2."""
    if PdfReader is None:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. "
            "Add it to src/lambda/requirements.txt."
        )
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


# ═══════════════════════════════════════════════════════════════════════════
# Text cleaning (simplified — matches src/pipeline/cleaners/text_cleaner.py)
# ═══════════════════════════════════════════════════════════════════════════

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """Basic text cleaning: remove URLs, collapse whitespace."""
    text = _URL_RE.sub("", text)
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Chunking (simplified recursive split — no tiktoken dependency)
# ═══════════════════════════════════════════════════════════════════════════

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping character-limited chunks."""
    raw = _recursive_split(text, _SEPARATORS, CHUNK_SIZE_CHARS)
    return _merge_with_overlap(raw)


def _recursive_split(text: str, separators: list[str], max_size: int) -> list[str]:
    """Recursively split text using progressively finer separators."""
    if len(text) <= max_size:
        return [text] if text.strip() else []

    if not separators:
        # Hard split at max_size as last resort
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]

    sep = separators[0]
    rest = separators[1:]
    parts = text.split(sep)
    result: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                result.append(current)
            if len(part) > max_size:
                result.extend(_recursive_split(part, rest, max_size))
            else:
                current = part

    if current:
        result.append(current)

    return result


def _merge_with_overlap(chunks: list[str]) -> list[str]:
    """Add character overlap between consecutive chunks."""
    if len(chunks) <= 1:
        return chunks

    merged = [chunks[0]]
    for i in range(1, len(chunks)):
        overlap = chunks[i - 1][-CHUNK_OVERLAP_CHARS:]
        merged.append(overlap + " " + chunks[i])
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Bedrock embeddings
# ═══════════════════════════════════════════════════════════════════════════

def _embed_single(text: str) -> list[float]:
    """Generate an embedding for a single text via Bedrock Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock_runtime.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def _embed_chunks(chunks: list[str], doc_id: str) -> list[dict[str, Any]]:
    """Embed all chunks and return a list of dicts for JSON serialisation."""
    embedded: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(chunks):
        # Bedrock Titan has a 8192-token input limit; truncate if needed
        truncated = chunk_text[:25000]  # ~8k tokens rough safety margin
        embedding = _embed_single(truncated)
        embedded.append(
            {
                "chunk_id": idx,
                "doc_id": doc_id,
                "text": chunk_text,
                "embedding": embedding,
                "embedding_dim": len(embedding),
            }
        )
    return embedded


# ═══════════════════════════════════════════════════════════════════════════
# DynamoDB helpers
# ═══════════════════════════════════════════════════════════════════════════

def _update_status(
    doc_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """Update the processing status in DynamoDB."""
    update_expr = "SET processing_status = :s, updated_at = :t"
    expr_values: dict[str, Any] = {
        ":s": status,
        ":t": int(time.time()),
    }

    if error:
        update_expr += ", error_message = :e"
        expr_values[":e"] = error

    try:
        dynamo_table.update_item(
            Key={"doc_id": doc_id},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_values,
        )
        logger.info("DynamoDB status updated: doc_id=%s status=%s", doc_id, status)
    except (BotoCoreError, ClientError) as exc:
        logger.error(
            "Failed to update DynamoDB status for doc_id=%s: %s", doc_id, exc
        )

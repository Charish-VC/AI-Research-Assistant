"""Centralized application settings via pydantic-settings.

All configuration is driven by environment variables or a .env file.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── Ollama ───────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_request_timeout: float = 120.0

    # ── Generation ───────────────────────────────────────────────────────
    generation_model: str = "llama3"

    # ── OpenAI (optional fallback) ───────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── FAISS ────────────────────────────────────────────────────────────
    faiss_index_path: str = "data/faiss_index"
    faiss_dimension: int = 768

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_top_k: int = 20
    similarity_threshold: float = 0.65

    # ── API ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── AWS ──────────────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "ai-research-assistant-dev"
    dynamodb_table_name: str = "ai-research-documents"
    sqs_queue_name: str = "document-processing-queue"

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def faiss_index_dir(self) -> Path:
        """Resolved path to the FAISS index directory."""
        p = Path(self.faiss_index_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def configure_logging(self) -> None:
        """Configure root logger based on settings."""
        logging.basicConfig(
            level=self.log_level.upper(),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()
    settings.configure_logging()
    return settings

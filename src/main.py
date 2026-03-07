"""FastAPI application entry point.

Creates the app, registers routes, and configures middleware and
lifecycle events.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_pipeline
from src.api.routes import chat, documents, health, ingest, query
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager — startup and shutdown hooks."""
    settings = get_settings()
    settings.configure_logging()
    logger.info("Starting AI Research Intelligence Platform v0.1.0")
    logger.info("Environment: %s", settings.environment)
    logger.info("Ollama host: %s", settings.ollama_host)
    logger.info("FAISS index: %s", settings.faiss_index_path)

    yield

    # Shutdown: close HTTP clients
    pipeline = get_pipeline()
    await pipeline.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="AI Research Intelligence Platform",
    description=(
        "Hybrid AI system for research document ingestion, processing, "
        "and RAG-powered retrieval. Local intelligence layer powered by "
        "FAISS vector search and Ollama embeddings."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(health.router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to docs."""
    return {
        "name": "AI Research Intelligence Platform",
        "version": "0.1.0",
        "docs": "/docs",
    }

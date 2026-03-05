"""FastAPI dependency factories.

Provides singleton instances of core services via ``Depends()``.
"""

from __future__ import annotations

from functools import lru_cache

from src.config.settings import Settings, get_settings
from src.pipeline.orchestrator import PipelineOrchestrator


@lru_cache(maxsize=1)
def get_pipeline() -> PipelineOrchestrator:
    """Return the singleton ``PipelineOrchestrator``.

    The pipeline is created once and reused for all requests.
    """
    settings = get_settings()
    return PipelineOrchestrator(settings)

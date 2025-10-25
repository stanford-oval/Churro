"""Configuration and environment setup for LLM utilities.

Design goals:
 - Minimize side-effects at import time (no unconditional mutation of os.environ).
 - Centralize environment variable access and validation.
 - Provide a lazy initialization entrypoint so tests can override env before setup.

Public API:
 - DEFAULT_TIMEOUT: int constant.
 - get_settings(): returns current snapshot of relevant settings.
 - ensure_initialized(): idempotent lazy initialization of litellm & caching.

Environment variables (all optional unless a specific provider is used):
 - AZURE_API_BASE
 - AZURE_API_VERSION
 - AZURE_OPENAI_API_KEY
 - LOCAL_VLLM_PORT (for locally hosted vLLM servers)
 - VERTEX_AI_LOCATION
"""

from dataclasses import dataclass
from functools import lru_cache

import litellm
from litellm.caching.caching import enable_cache

from churro.config.settings import get_settings as get_churro_settings
from churro.utils.log_utils import logger


DEFAULT_TIMEOUT: int = 60 * 10  # 10 minutes

_initialized: bool = False


@dataclass(frozen=True)
class LLMSettings:
    """Snapshot of environment-driven runtime settings."""

    azure_api_base: str | None
    azure_api_version: str | None
    azure_openai_api_key: str | None
    local_vllm_port: int | None
    vertex_ai_location: str

    @property
    def local_base_url(self) -> str | None:
        """Return the base URL for local vLLM server, or None if not configured."""
        if self.local_vllm_port:
            return f"http://localhost:{self.local_vllm_port}/v1"
        return None


def ensure_initialized() -> None:
    """Idempotently initialize litellm configuration & caching.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _initialized
    if _initialized:
        return
    enable_cache(type="disk")  # type: ignore[arg-type]
    litellm.suppress_debug_info = True
    litellm.drop_params = True
    _initialized = True


@lru_cache(maxsize=1)
def get_settings() -> LLMSettings:
    """Return cached settings snapshot derived from environment variables."""
    base_settings = get_churro_settings()
    azure = base_settings.azure_openai

    azure_api_version = azure.api_version
    if not azure_api_version:
        # Do not force-set; just warn for visibility.
        logger.warning(
            "AZURE_API_VERSION not set; Azure model calls may fail if provider requires explicit version."
        )
    settings = LLMSettings(
        azure_api_base=azure.api_base,
        azure_api_version=azure_api_version,
        azure_openai_api_key=azure.api_key,
        local_vllm_port=base_settings.local.vllm_port,
        vertex_ai_location=base_settings.vertex_ai.location,
    )

    if settings.azure_api_base is None:
        logger.debug("AZURE_API_BASE is not set; skipping Azure-specific validation until used.")
    if settings.local_vllm_port is None:
        logger.debug(
            "LOCAL_VLLM_PORT is not set; local vLLM models will be unavailable until configured."
        )
    return settings


# Gemini safety settings for Vertex requests
GEMINI_SAFETY_SETTINGS: list[dict[str, str]] = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF"},
]

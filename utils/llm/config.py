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
import os
from typing import Optional

from dotenv import load_dotenv
import litellm
from litellm.caching.caching import enable_cache

from utils.log_utils import logger


# Load environment variables from .env file (safe if called multiple times)
load_dotenv()

# Constants
DEFAULT_TIMEOUT: int = 60 * 10  # 10 minutes

_initialized: bool = False


@dataclass(frozen=True)
class LLMSettings:
    """Snapshot of environment-driven runtime settings."""

    azure_api_base: Optional[str]
    azure_api_version: Optional[str]
    azure_openai_api_key: Optional[str]
    local_vllm_port: Optional[int]
    vertex_ai_location: str

    @property
    def local_base_url(self) -> Optional[str]:
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
    azure_api_version = os.getenv("AZURE_API_VERSION")
    if not azure_api_version:
        # Do not force-set; just warn for visibility.
        logger.warning(
            "AZURE_API_VERSION not set; Azure model calls may fail if provider requires explicit version."
        )
    settings = LLMSettings(
        azure_api_base=os.getenv("AZURE_API_BASE"),
        azure_api_version=azure_api_version,
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        local_vllm_port=int(os.getenv("LOCAL_VLLM_PORT", 9000)),
        vertex_ai_location=os.getenv("VERTEX_AI_LOCATION", "us-east5"),
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

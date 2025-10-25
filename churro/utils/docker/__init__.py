"""High-level Docker utilities package.

This package provides structured helpers for:
    * Starting generic Docker containers (``start_container``)
    * Waiting for a readiness log pattern with inactivity timeout semantics
        (``wait_for_readiness`` / ``start_and_wait_ready``)
    * Launching specialized model serving runtimes (currently vLLM via
        ``start_vllm_server``)
    * Conditionally spinning up local vLLM servers for engines described in
        ``llm.models.MODEL_MAP`` (``maybe_start_vllm_server_for_engine``).

Principles:
    * Keep low-level SDK usage encapsulated (see ``sdk.py``) so higher-level
        code can be easily mocked in tests.
    * Avoid side effects at import time (no client construction until needed).
    * Provide precise docstrings and type hints for clarity.

Public API (re-exported):
        - DockerError
        - DockerContainer
        - start_container
        - wait_for_readiness
        - start_and_wait_ready
        - start_vllm_server
        - maybe_start_vllm_server_for_engine
        - get_hf_repo_for_hosted
        - has_at_least_one_vllm
"""

from .container import DockerContainer
from .errors import DockerError
from .operations import start_and_wait_ready, start_container, wait_for_readiness
from .servers import start_vllm_server
from .vllm import (
    get_hf_repo_for_hosted,
    has_at_least_one_vllm,
    maybe_start_vllm_server_for_engine,
)


__all__ = [
    "DockerError",
    "DockerContainer",
    "start_container",
    "wait_for_readiness",
    "start_and_wait_ready",
    "start_vllm_server",
    "maybe_start_vllm_server_for_engine",
    "get_hf_repo_for_hosted",
    "has_at_least_one_vllm",
]

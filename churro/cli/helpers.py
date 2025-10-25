from __future__ import annotations

from collections.abc import Generator
import contextlib
from typing import Any

from churro.utils.docker.vllm import maybe_start_vllm_server_for_engine
from churro.utils.log_utils import logger


@contextlib.contextmanager
def managed_vllm_container(
    *,
    engine: str | None,
    backup_engine: str | None,
    system: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
) -> Generator[Any, None, None]:
    """Start a vLLM container when needed and ensure it stops on exit."""
    container = maybe_start_vllm_server_for_engine(
        engine=engine,
        system=system,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
    )
    backup_container: Any | None = None
    if backup_engine and backup_engine != engine:
        backup_container = maybe_start_vllm_server_for_engine(
            engine=backup_engine,
            system=system,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            log_prefix="[backup]",
        )
    try:
        yield container
    finally:
        if backup_container is not None:
            logger.info("Stopping backup vLLM container...")
            with contextlib.suppress(Exception):
                backup_container.stop()
        if container is not None:
            logger.info("Stopping vLLM container...")
            with contextlib.suppress(Exception):
                container.stop()

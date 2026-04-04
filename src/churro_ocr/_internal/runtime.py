"""Runtime helpers shared by sync wrappers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync[T](awaitable: Coroutine[Any, Any, T]) -> T:
    """Run an awaitable from sync code.

    Raises:
        RuntimeError: If called from a running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "Synchronous churro-ocr APIs cannot be used from an active event loop. Use the async API instead."
    )

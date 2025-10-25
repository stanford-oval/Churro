# Test configuration utilities.
# Ensures the repository root is on sys.path so that 'churro.utils', etc. can be imported
# when running pytest without installing the package.
from __future__ import annotations

import asyncio
from collections.abc import Generator
import gc
from pathlib import Path
import sys

import pytest

from churro.utils.llm.shutdown import shutdown_llm_clients


ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def cleanup_async_clients() -> Generator[None, None, None]:
    """Ensure aiohttp sessions opened via litellm are closed after tests."""
    yield

    async def _close_async_resources() -> None:
        await shutdown_llm_clients()

    try:
        asyncio.run(_close_async_resources())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_close_async_resources())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    policy = asyncio.get_event_loop_policy()
    try:
        loop = policy.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and not loop.is_closed():
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    for obj in gc.get_objects():
        if isinstance(obj, asyncio.AbstractEventLoop) and not obj.is_closed():
            try:
                obj.run_until_complete(obj.shutdown_asyncgens())
            except Exception:
                pass
            try:
                obj.close()
            except Exception:
                pass


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide a fresh event loop per test and ensure it closes cleanly."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        if not loop.is_closed():
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

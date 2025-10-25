"""Shutdown helpers for LLM utilities.

Provides an async function to gracefully close any underlying HTTP client sessions
opened by litellm (which internally uses aiohttp for some providers). This helps
suppress warnings about unclosed ClientSession objects at interpreter shutdown
when our pipelines perform many concurrent requests and exit quickly.

If litellm changes its internal structure, failure to locate the session is
silently ignored.
"""

from __future__ import annotations

import gc
from typing import Any

import litellm

from churro.utils.log_utils import logger


async def _gather_candidate_sessions() -> set[Any]:
    """Return a set of aiohttp.ClientSession-like objects discovered heuristically."""
    sessions: set[Any] = set()
    try:
        import aiohttp
    except Exception:
        return sessions

    # 1. Probe known litellm attributes
    for _name, value in list(litellm.__dict__.items()):
        if isinstance(value, aiohttp.ClientSession):  # type: ignore[attr-defined]
            sessions.add(value)

    # 2. Probe common attribute names on litellm module itself
    for attr in ["_client_session", "client_session", "session"]:
        value = getattr(litellm, attr, None)
        if isinstance(value, aiohttp.ClientSession):  # type: ignore[attr-defined]
            sessions.add(value)

    # 3. OpenAI aiosession accessor (handles old/new patterns)
    try:  # pragma: no cover - optional dependency
        import openai  # type: ignore

        aiosession = getattr(openai, "aiosession", None)
        if aiosession and hasattr(aiosession, "get"):
            get_fn = aiosession.get
            maybe = get_fn()  # returns session or awaitable depending on version
            if hasattr(maybe, "__await__"):
                maybe = await maybe  # type: ignore[assignment]
            if isinstance(maybe, aiohttp.ClientSession):  # type: ignore[attr-defined]
                sessions.add(maybe)
    except Exception:
        pass

    # 4. GC scan as last resort (may find additional leaked sessions)
    try:
        import aiohttp  # type: ignore  # re-import inside scope for mypy

        for obj in gc.get_objects():
            if isinstance(obj, aiohttp.ClientSession) and not obj.closed:  # type: ignore[attr-defined]
                sessions.add(obj)
    except Exception:  # pragma: no cover
        pass

    return sessions


async def shutdown_llm_clients() -> None:
    """Attempt to gracefully close underlying aiohttp ClientSession objects.

    Strategy:
      * Probe litellm module attributes for sessions
      * Probe openai.aiosession (old + new patterns)
      * GC scan fallback
    Safe to call multiple times.
    """
    try:
        sessions = await _gather_candidate_sessions()
        closed = 0
        for sess in sessions:
            try:
                if getattr(sess, "closed", False):
                    continue
                maybe = sess.close()
                if hasattr(maybe, "__await__"):
                    await maybe
                closed += 1
            except Exception as e:  # pragma: no cover
                logger.debug(f"Failed closing session {sess}: {e}")
        if closed:
            logger.debug(f"Closed {closed} aiohttp session(s) during shutdown")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"LLM client shutdown encountered error: {e}")

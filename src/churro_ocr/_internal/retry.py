"""Shared retry helpers for provider API calls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from time import monotonic
from typing import cast

from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 16.0
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 520, 521, 522, 524})
RETRYABLE_EXCEPTION_CLASS_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ClientConnectionError",
        "ClientConnectorError",
        "ClientOSError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "PoolTimeout",
        "RateLimitError",
        "ReadTimeout",
        "RemoteProtocolError",
        "ServiceRequestError",
        "ServiceResponseError",
        "ServerDisconnectedError",
        "WriteTimeout",
    }
)
RETRYABLE_EXCEPTION_MODULE_PREFIXES = ("httpcore", "httpx")

retry_sleep = asyncio.sleep

type RetryPredicate = Callable[[BaseException], bool]


def _coerce_status_code(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _headers_from_candidate(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def get_error_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP-like status code from a provider exception when available."""
    status_code = _coerce_status_code(getattr(exc, "status_code", None))
    if status_code is not None:
        return status_code

    for attribute_name in ("response", "raw_response"):
        response = getattr(exc, attribute_name, None)
        status_code = _coerce_status_code(getattr(response, "status_code", None))
        if status_code is not None:
            return status_code
    return None


def get_error_retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a retry delay from response headers when one is available."""
    headers = _headers_from_candidate(getattr(exc, "headers", None))
    if headers is None:
        for attribute_name in ("response", "raw_response"):
            response = getattr(exc, attribute_name, None)
            headers = _headers_from_candidate(getattr(response, "headers", None))
            if headers is not None:
                break
    if headers is None:
        return None

    retry_after = headers.get("retry-after")
    if retry_after is None:
        retry_after = headers.get("Retry-After")
    if retry_after is None:
        return None
    if not isinstance(retry_after, str | int | float):
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def compute_retry_delay_seconds(
    exc: BaseException,
    *,
    attempt_number: int,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> float:
    """Compute the retry delay for a failed provider request."""
    retry_after = get_error_retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    return min(
        initial_backoff_seconds * (2 ** max(0, attempt_number - 1)),
        max_backoff_seconds,
    )


def _remaining_retry_budget_seconds(
    *,
    started_at: float,
    max_total_seconds: float | None,
) -> float | None:
    if max_total_seconds is None:
        return None
    return max(0.0, max_total_seconds - (monotonic() - started_at))


def is_retryable_api_error(exc: BaseException) -> bool:
    """Return whether a provider exception should be retried."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
        return True

    status_code = get_error_status_code(exc)
    if status_code is not None:
        return status_code in TRANSIENT_STATUS_CODES

    exc_type = exc.__class__
    if exc_type.__name__ in RETRYABLE_EXCEPTION_CLASS_NAMES:
        return True

    module_name = exc_type.__module__
    return any(module_name.startswith(prefix) for prefix in RETRYABLE_EXCEPTION_MODULE_PREFIXES)


def _build_before_sleep_callback(
    *,
    operation_name: str,
    context: str | None,
    max_attempts: int,
) -> Callable[[RetryCallState], None]:
    message_context = f" {context}" if context else ""

    def _before_sleep(retry_state: RetryCallState) -> None:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None and outcome.failed else None
        if exc is None:
            return
        delay_seconds = retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
        logger.warning(
            "Transient %s failure%s (status=%s, attempt=%s/%s); retrying in %.1fs.",
            operation_name,
            message_context,
            get_error_status_code(exc) or "unknown",
            retry_state.attempt_number,
            max_attempts,
            delay_seconds,
        )

    return _before_sleep


async def retry_api_call[T](
    fn: Callable[[], Awaitable[T]],
    *,
    operation_name: str,
    context: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_total_seconds: float | None = None,
    retry_filter: RetryPredicate = is_retryable_api_error,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> T:
    """Execute an async provider request with shared retry behavior."""
    started_at = monotonic()

    def _retryable_within_budget(exc: BaseException) -> bool:
        remaining_budget = _remaining_retry_budget_seconds(
            started_at=started_at,
            max_total_seconds=max_total_seconds,
        )
        if remaining_budget is not None and remaining_budget <= 0:
            return False
        return retry_filter(exc)

    def _retry_wait_seconds(retry_state: RetryCallState) -> float:
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return 0.0
        exc = retry_state.outcome.exception()
        if exc is None:
            return 0.0
        delay_seconds = compute_retry_delay_seconds(
            exc,
            attempt_number=retry_state.attempt_number,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
        remaining_budget = _remaining_retry_budget_seconds(
            started_at=started_at,
            max_total_seconds=max_total_seconds,
        )
        if remaining_budget is None:
            return delay_seconds
        return min(delay_seconds, remaining_budget)

    retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception(_retryable_within_budget),
        wait=_retry_wait_seconds,
        before_sleep=_build_before_sleep_callback(
            operation_name=operation_name,
            context=context,
            max_attempts=max_attempts,
        ),
        sleep=retry_sleep,
    )

    async for attempt in retrying:
        with attempt:
            return await fn()

    raise AssertionError("AsyncRetrying exited without returning or raising.")


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "compute_retry_delay_seconds",
    "get_error_retry_after_seconds",
    "get_error_status_code",
    "is_retryable_api_error",
    "retry_api_call",
    "retry_sleep",
]

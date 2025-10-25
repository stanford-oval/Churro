"""Structured concurrency helpers for running async workloads in parallel."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import ParamSpec, Protocol, TypeVar

from tqdm import tqdm

from .log_utils import logger


class ProgressReporter(Protocol):
    """Lightweight progress reporter abstraction."""

    def start(self, total: int) -> None: ...

    def increment(self) -> None: ...

    def close(self) -> None: ...


class TqdmProgressReporter:
    """Progress reporter backed by tqdm."""

    def __init__(self, desc: str, update_interval: float = 1.0) -> None:
        self._desc = desc
        self._update_interval = update_interval
        self._pbar: tqdm | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tick_handle: asyncio.TimerHandle | None = None

    def start(self, total: int) -> None:
        self._pbar = tqdm(
            total=total,
            desc=self._desc,
            smoothing=0,
            leave=False,
        )
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - defensive guard for sync contexts
            self._loop = None
        if self._loop and self._update_interval > 0:
            self._schedule_tick()

    def _schedule_tick(self) -> None:
        if not self._loop or self._tick_handle is not None or self._pbar is None:
            # Loop missing, already scheduled, or pbar unavailable
            return
        self._tick_handle = self._loop.call_later(self._update_interval, self._tick)

    def _tick(self) -> None:
        self._tick_handle = None
        if self._pbar is None:
            return
        self._pbar.refresh()
        if self._loop and self._update_interval > 0:
            self._schedule_tick()

    def increment(self) -> None:
        if self._pbar is not None:
            self._pbar.update(1)

    def close(self) -> None:
        if self._tick_handle is not None:
            self._tick_handle.cancel()
            self._tick_handle = None
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        self._loop = None


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration for async jobs."""

    max_attempts: int = 1
    timeout: float | None = None
    retry_exceptions: tuple[type[BaseException], ...] = (asyncio.TimeoutError,)
    backoff_seconds: float = 0.0

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        return isinstance(exc, self.retry_exceptions)


P = ParamSpec("P")
T = TypeVar("T")


class ParallelExecutor:
    """Run async callables in parallel with structured error handling."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        retry_policy: RetryPolicy | None = None,
        progress_reporter: ProgressReporter | None = None,
        return_exceptions: bool = False,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._max_concurrency = max_concurrency
        self._retry_policy = retry_policy or RetryPolicy()
        self._progress = progress_reporter
        self._return_exceptions = return_exceptions

    async def map(
        self,
        fn: Callable[..., Awaitable[T]],
        *iterables: Sequence[object],
    ) -> list[T | None]:
        """Execute `fn` across provided iterables with bounded concurrency."""
        if not iterables:
            return []

        lengths = [len(it) for it in iterables]
        if any(length != lengths[0] for length in lengths):
            raise ValueError("All iterables must have the same length.")

        jobs = list(enumerate(zip(*iterables, strict=False)))
        total = len(jobs)
        results: list[T | BaseException | None] = [None] * total

        queue: asyncio.Queue[tuple[int, tuple[object, ...]]] = asyncio.Queue()
        for job in jobs:
            queue.put_nowait((job[0], tuple(job[1])))

        if self._progress:
            self._progress.start(total)

        errors: dict[int, BaseException] = {}

        async def worker() -> None:
            while True:
                try:
                    index, args = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                attempt = 0
                while True:
                    attempt += 1
                    try:
                        coro = fn(*args)
                        if self._retry_policy.timeout is not None:
                            result = await asyncio.wait_for(
                                coro, timeout=self._retry_policy.timeout
                            )
                        else:
                            result = await coro
                        results[index] = result
                        break
                    except Exception as exc:
                        if self._retry_policy.should_retry(exc, attempt):
                            if self._retry_policy.backoff_seconds > 0:
                                await asyncio.sleep(self._retry_policy.backoff_seconds)
                            continue
                        errors[index] = exc
                        if self._return_exceptions:
                            results[index] = exc
                        else:
                            results[index] = None
                        logger.exception(f"Parallel executor job {index} failed", exc_info=exc)
                        break
                queue.task_done()
                if self._progress:
                    self._progress.increment()

        async with asyncio.TaskGroup() as tg:
            for _ in range(min(self._max_concurrency, total)):
                tg.create_task(worker())

        if self._progress:
            self._progress.close()

        if errors and not self._return_exceptions:
            logger.error(
                f"Parallel executor encountered {len(errors)} failed job(s): {list(errors.keys())}."
            )
        return results  # type: ignore[return-value]


async def run_async_in_parallel(
    async_function: Callable[P, Awaitable[T]],
    *iterables: Sequence[object],
    max_concurrency: int,
    timeout: float = 60 * 60 * 1,
    desc: str = "",
) -> list[T | None]:
    """Execute an async function over iterables with bounded concurrency.

    The helper preserves the previous behaviour of retrying requests that time out
    while returning ``None`` for tasks that ultimately fail.
    """
    reporter = TqdmProgressReporter(desc) if desc else None
    retry_policy = RetryPolicy(
        max_attempts=3 if timeout else 1,
        timeout=timeout,
    )
    executor = ParallelExecutor(
        max_concurrency=max_concurrency,
        retry_policy=retry_policy,
        progress_reporter=reporter,
    )
    return await executor.map(async_function, *iterables)

from __future__ import annotations

import asyncio

import pytest

from churro.utils.concurrency import ParallelExecutor, RetryPolicy, run_async_in_parallel


@pytest.mark.asyncio
async def test_parallel_executor_runs_all_jobs() -> None:
    executor = ParallelExecutor(max_concurrency=2)

    async def echo(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    results = await executor.map(echo, [1, 2, 3, 4])
    assert results == [2, 4, 6, 8]


@pytest.mark.asyncio
async def test_parallel_executor_retries_timeout() -> None:
    attempts: dict[int, int] = {}

    async def flaky(value: int) -> int:
        attempts[value] = attempts.get(value, 0) + 1
        if attempts[value] == 1:
            await asyncio.sleep(0.05)
        return value

    executor = ParallelExecutor(
        max_concurrency=1,
        retry_policy=RetryPolicy(max_attempts=2, timeout=0.01),
    )
    results = await executor.map(flaky, [0])
    assert results == [0]
    assert attempts[0] == 2


@pytest.mark.asyncio
async def test_parallel_executor_records_failure() -> None:
    async def boom(_: int) -> int:
        raise ValueError("nope")

    executor = ParallelExecutor(max_concurrency=1)
    results = await executor.map(boom, [1, 2])
    assert results == [None, None]


@pytest.mark.asyncio
async def test_parallel_executor_returns_exceptions() -> None:
    async def sometimes_fail(value: int) -> int:
        if value == 1:
            raise RuntimeError("boom")
        return value

    executor = ParallelExecutor(max_concurrency=2, return_exceptions=True)
    results = await executor.map(sometimes_fail, [0, 1, 2])

    assert results[0] == 0
    assert isinstance(results[1], RuntimeError)
    assert results[2] == 2


@pytest.mark.asyncio
async def test_run_async_in_parallel_wrapper() -> None:
    async def identity(value: int) -> int:
        return value

    results = await run_async_in_parallel(identity, [1, 2, 3], max_concurrency=2, desc="")
    assert results == [1, 2, 3]

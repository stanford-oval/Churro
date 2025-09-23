import asyncio
from typing import Optional

from diskcache import Cache
from PIL import Image
from tqdm.asyncio import tqdm as async_tqdm

from .log_utils import logger


cache: Optional[Cache] = None


async def run_async_in_parallel(
    async_function,
    *iterables,
    max_concurrency: int,
    timeout: float = 60 * 60 * 1,
    desc: str = "",
    progbar_update_interval: float = 1.0,
):
    """Executes an asynchronous function concurrently over multiple iterables with a fixed number of worker tasks.

    This function schedules calls to the provided asynchronous function using a queue and worker tasks.
    Each worker repeatedly retrieves a tuple of arguments from the queue, executes the asynchronous function,
    and if the call does not complete within the specified timeout, it is retried until successful. The progress
    of execution is periodically updated via a progress bar that refreshes at fixed intervals.

    Parameters:
        async_function (Callable): The asynchronous function to be executed. It must accept as many arguments as there are iterables.
        *iterables (Iterable): One or more iterables supplying arguments for async_function. All iterables must have the same length.
        max_concurrency (int): The maximum number of concurrent worker tasks to run.
        timeout (float, optional): The maximum number of seconds to wait for async_function to complete for each call.
                                  Defaults to 3600 seconds (1 hour).
        desc (str, optional): Description text for the progress bar. If empty, the progress bar is disabled.
                              Defaults to an empty string.
        progbar_update_interval (float, optional): The interval (in seconds) at which the progress bar is updated.
                                                   Defaults to 1.0 second.

    Returns:
        list: A list of results obtained from executing async_function with the provided arguments.
              If a task times out or raises an exception, the corresponding result is set to None.

    Raises:
        ValueError: If the provided iterables do not have the same length.
    """
    if not iterables:
        return []

    length = len(iterables[0])
    for it in iterables:
        if len(it) != length:
            raise ValueError("All iterables must have the same length.")

    # Enqueue all jobs as (index, args) pairs.
    queue: asyncio.Queue[tuple[int, tuple]] = asyncio.Queue()
    for index, args in enumerate(zip(*iterables)):
        await queue.put((index, args))

    results: list = [None] * length
    finished_count = 0  # shared progress counter
    pbar = async_tqdm(total=length, smoothing=0, desc=desc, disable=(not desc))

    async def worker():
        nonlocal finished_count
        while True:
            try:
                index, args = await queue.get()
            except asyncio.CancelledError:
                break
            try:
                # Retry until async_function finishes within timeout.
                while True:
                    try:
                        # Wait for the async_function with a timeout.
                        results[index] = await asyncio.wait_for(async_function(*args), timeout)
                        break  # success: exit retry loop
                    except asyncio.TimeoutError:
                        # Log or print a message here if desired.
                        # The task timed out; retry it.
                        continue
            except Exception:
                logger.exception(f"Exception in async worker: {index}")
                results[index] = None
            finally:
                finished_count += 1
                queue.task_done()

    # Refresh task to update the tqdm progress bar exactly once every progbar_update_interval seconds.
    stop_refresh = asyncio.Event()

    async def refresh_progress():
        while not stop_refresh.is_set():
            await asyncio.sleep(progbar_update_interval)
            pbar.n = finished_count
            pbar.refresh()

    # Spawn the worker tasks and the refresh task.
    workers = [asyncio.create_task(worker()) for _ in range(max_concurrency)]
    refresh_task = asyncio.create_task(refresh_progress())

    # Wait until all jobs are processed.
    await queue.join()
    stop_refresh.set()
    await refresh_task  # wait for refresh task to finish

    # Cancel workers and do a final progress update.
    for w in workers:
        w.cancel()
    pbar.n = finished_count
    pbar.refresh()
    pbar.close()

    return results


def resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Resize to fit inside (``max_width``, ``max_height``) maintaining aspect ratio if larger."""
    original_width, original_height = image.size

    # Only scale if the image is larger than the max dimensions
    if original_width <= max_width and original_height <= max_height:
        return image

    scale = min(max_width / original_width, max_height / original_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    return image.resize(new_size, resample=Image.LANCZOS)  # type: ignore

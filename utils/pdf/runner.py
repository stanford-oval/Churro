"""Queued asynchronous PDF processing pipeline.

This pipeline performs three staged activities:
    1. Rasterization of PDF pages in a process pool (CPU bound)
    2. Page analysis & optional splitting/ trimming (async workers + LLM calls)
    3. Deterministic ordered saving of resulting page (or split-page) images
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import os
from typing import Dict, Iterable, List, Sequence

import fitz  # PyMuPDF
from PIL import Image

from utils.llm import extract_tag_from_llm_output, run_llm_async
from utils.log_utils import logger
from utils.pdf.pdfs_to_images import (
    split_double_page,
    trim_image,
)


# ------------------------- Data Structures ------------------------- #


@dataclass(slots=True)
class RasterTask:
    """Represents a single rasterized PDF page awaiting processing."""

    pdf_id: int
    pdf_path: str
    page_index: int  # original page number in PDF
    png_bytes: bytes


@dataclass(slots=True)
class PageSplitGroup:
    """Container for all (optionally split & trimmed) images originating from a single PDF page.

    The saver coroutine buffers these groups per PDF and releases them strictly in ascending
    original page order, assigning monotonically increasing logical split indices at *save time*.
    This guarantees deterministic on-disk ordering even though page processing is performed
    concurrently across multiple workers.
    """

    pdf_id: int
    pdf_path: str
    page_index: int  # original page index inside the PDF
    images: List[Image.Image]


# ------------------------- Raster Producer ------------------------- #


def _raster_batch(pdf_path: str, page_numbers: list[int], zoom: float) -> list[tuple[int, bytes]]:
    doc = fitz.open(pdf_path)
    out: list[tuple[int, bytes]] = []
    try:
        for p in page_numbers:
            try:
                page = doc[p]
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))  # type: ignore
                out.append((p, pix.tobytes("png")))
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to rasterize page {p} of {pdf_path}: {e}")
        return out
    finally:
        doc.close()


async def raster_producer(
    pdf_paths: Sequence[str],
    raster_queue: asyncio.Queue[RasterTask | None],
    dpi: int,
    batch_pages: int,
    max_workers: int,
) -> None:
    loop = asyncio.get_running_loop()
    zoom = dpi / 72.0

    # Always use 'spawn' for safety (avoid forking active async/thread state)
    from multiprocessing import get_context  # type: ignore

    ctx = get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        for pdf_id, pdf_path in enumerate(pdf_paths):
            try:
                doc = fitz.open(pdf_path)
                page_count = doc.page_count
                doc.close()
            except Exception as e:
                logger.error(f"Skipping {pdf_path}: cannot open ({e})")
                continue

            page_numbers = list(range(page_count))
            batches = [
                page_numbers[i : i + batch_pages] for i in range(0, len(page_numbers), batch_pages)
            ]

            for batch in batches:
                # Backpressure: wait if queue full
                while raster_queue.full():
                    await asyncio.sleep(0)
                fut = loop.run_in_executor(pool, _raster_batch, pdf_path, batch, zoom)
                batch_results = await fut
                for page_index, png_bytes in batch_results:
                    await raster_queue.put(
                        RasterTask(
                            pdf_id=pdf_id,
                            pdf_path=pdf_path,
                            page_index=page_index,
                            png_bytes=png_bytes,
                        )
                    )
        # Signal completion to consumers by sending sentinel per worker (handled outside)
    # Producer exits


# ------------------------- Page Processing Workers ------------------------- #


async def page_worker(
    raster_queue: asyncio.Queue[RasterTask | None],
    processed_queue: asyncio.Queue[PageSplitGroup | None],
    engine: str,
    trim: bool,
    llm_concurrency: asyncio.Semaphore,
) -> None:
    while True:
        task = await raster_queue.get()
        if task is None:
            raster_queue.task_done()
            break

        try:
            image = Image.open(BytesIO(task.png_bytes))
        except Exception as e:
            logger.error(
                f"Failed to decode PNG bytes for pdf_id={task.pdf_id} page={task.page_index}: {e}"
            )
            # Emit an empty group so ordering logic can advance; otherwise saver would stall.
            await processed_queue.put(
                PageSplitGroup(
                    pdf_id=task.pdf_id,
                    pdf_path=task.pdf_path,
                    page_index=task.page_index,
                    images=[],
                )
            )
            raster_queue.task_done()
            continue

        # Always call LLM for number of pages
        async with llm_concurrency:
            llm_output = await run_llm_async(
                system_prompt_text=None,
                user_message_text=(
                    "You are given an image of a scanned document. Your task is to decide "
                    "whether the image actually contains ONE page or a TWO-PAGE SPREAD (i.e., two distinct "
                    "adjacent pages captured together).\n\n"
                    "Return ONLY 1 or 2 inside <number_of_pages> tags.\n\n"
                    "Classify as 2 ONLY if there are clearly two separate full pages. Strong indicators of TWO pages: "
                    "(a) visible central gutter or fold between pages, (b) duplicated page headers/footers or page numbers "
                    "appearing twice (left & right), (c) two independent margin/edge boundaries, (d) overall wide aspect ratio "
                    "where each half looks like a normal page.\n"
                    "Keep as 1 if it is a single page that merely has: multiple text columns, sidebars, advertisements, "
                    "tables, fold marks, marginal notes, decorative frames, or partial cropping of a neighboring page edge. "
                    "Multiple columns alone DO NOT mean two pages.\n\n"
                    "Edge cases: If unsure, output 1. If one page is only partially visible (e.g., a sliver of another page), "
                    "still output 1.\n\n"
                    "Format EXACTLY:\n<number_of_pages>\n1 or 2\n</number_of_pages>"
                ),
                user_message_image=image,
                image_detail="low",
                model=engine,
            )
        number_of_pages_str = extract_tag_from_llm_output(llm_output, tags="number_of_pages")
        try:
            if isinstance(number_of_pages_str, list):
                number = int(number_of_pages_str[0]) if number_of_pages_str else 1
            else:
                number = int(number_of_pages_str)
        except Exception:
            number = 1

        splits: list[Image.Image]
        if number == 2:
            splits = split_double_page(image)
        else:
            splits = [image]

        results: list[Image.Image] = []
        if trim:
            for s in splits:
                try:
                    s = await trim_image(s)
                except Exception as e:
                    logger.error(f"Trim failed (pdf_id={task.pdf_id}, page={task.page_index}): {e}")
                results.append(s)
        else:
            results = splits

        # Queue one group representing this original PDF page.
        await processed_queue.put(
            PageSplitGroup(
                pdf_id=task.pdf_id,
                pdf_path=task.pdf_path,
                page_index=task.page_index,
                images=results,
            )
        )
        raster_queue.task_done()


# ------------------------- Saver ------------------------- #


async def saver(
    processed_queue: asyncio.Queue[PageSplitGroup | None],
    output_dir: str,
    splits_counter: Dict[int, int],
) -> None:
    # Ensure per-PDF directories? Keeping flat naming like original.
    os.makedirs(output_dir, exist_ok=True)

    # Per-PDF buffering to enforce ascending original page order during save.
    pending: Dict[int, Dict[int, List[Image.Image]]] = {}
    next_expected: Dict[int, int] = {}

    while True:
        group = await processed_queue.get()
        if group is None:  # type: ignore
            processed_queue.task_done()
            break

        pending.setdefault(group.pdf_id, {})[group.page_index] = group.images
        next_expected.setdefault(group.pdf_id, 0)
        splits_counter.setdefault(group.pdf_id, 0)

        # Flush any now-contiguous pages.
        while next_expected[group.pdf_id] in pending[group.pdf_id]:
            images = pending[group.pdf_id].pop(next_expected[group.pdf_id])
            pdf_stem = os.path.basename(group.pdf_path).replace(".pdf", "").replace(" ", "_")
            for img in images:
                logical_index = splits_counter[group.pdf_id]
                fname = f"{pdf_stem}_page_{logical_index:04d}.png"
                out_path = os.path.join(output_dir, fname)
                try:
                    img.save(out_path, "PNG")
                except Exception as e:  # pragma: no cover
                    logger.error(f"Failed saving {out_path}: {e}")
                splits_counter[group.pdf_id] = logical_index + 1
            next_expected[group.pdf_id] += 1

        processed_queue.task_done()


# ------------------------- Orchestrator ------------------------- #


async def run_pdf_pipeline(
    pdf_paths: Iterable[str],
    output_dir: str,
    engine: str,
    dpi: int = 300,
    batch_pages: int = 8,
    queue_maxsize: int = 64,
    raster_workers: int | None = None,
    page_workers: int | None = None,
    llm_concurrency_limit: int = 64,
    trim: bool = True,
) -> None:
    pdf_paths = [p for p in pdf_paths if os.path.isfile(p) and p.lower().endswith(".pdf")]
    if not pdf_paths:
        logger.warning("No valid PDF files provided to pipeline.")
        return

    os.makedirs(output_dir, exist_ok=True)

    raster_workers = raster_workers or max(1, (os.cpu_count() or 2) // 2)
    page_workers = page_workers or max(2, (os.cpu_count() or 2))

    raster_queue: asyncio.Queue[RasterTask | None] = asyncio.Queue(maxsize=queue_maxsize)
    processed_queue: asyncio.Queue[PageSplitGroup | None] = asyncio.Queue(maxsize=queue_maxsize)

    # Accumulates final number of saved splits per PDF (filled in by saver)
    splits_counter: Dict[int, int] = {}

    llm_semaphore = asyncio.Semaphore(llm_concurrency_limit)

    producer_task = asyncio.create_task(
        raster_producer(
            pdf_paths,
            raster_queue,
            dpi,
            batch_pages,
            raster_workers,
        )
    )

    worker_tasks = [
        asyncio.create_task(
            page_worker(
                raster_queue,
                processed_queue,
                engine,
                trim,
                llm_semaphore,
            )
        )
        for _ in range(page_workers)
    ]

    saver_task = asyncio.create_task(saver(processed_queue, output_dir, splits_counter))

    # Wait for producer completion then signal page workers
    await producer_task
    for _ in worker_tasks:
        await raster_queue.put(None)  # type: ignore

    await raster_queue.join()

    # Signal saver termination
    await processed_queue.put(None)  # type: ignore

    await asyncio.gather(*worker_tasks)
    await processed_queue.join()
    await saver_task

    logger.info(
        f"Pipeline complete. PDFs processed: {len(pdf_paths)} | Total pages: {sum(splits_counter.values())}",
    )

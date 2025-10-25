"""Queued asynchronous PDF processing pipeline with staged concurrency."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import os
from typing import Protocol

import fitz  # PyMuPDF
from PIL import Image

from churro.systems.detect_layout import shutdown_layout_clients, tidy_image_via_layout_detection
from churro.utils.concurrency import ProgressReporter
from churro.utils.llm import extract_tag_from_llm_output, run_llm_async
from churro.utils.llm.shutdown import shutdown_llm_clients
from churro.utils.log_utils import logger
from churro.utils.pdf.pdfs_to_images import split_double_page


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}

DEFAULT_RASTER_DPI = 300
SINGLE_PAGE_ASPECT_RATIO_THRESHOLD = 1.0


def _collect_image_paths(image_dir: str) -> list[str]:
    collected: list[str] = []
    for root, _, files in os.walk(image_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                collected.append(os.path.join(root, file_name))
    return sorted(collected)


@dataclass(slots=True)
class RasterTask:
    pdf_id: int
    pdf_path: str
    page_index: int
    png_bytes: bytes


@dataclass(slots=True)
class PageSplitGroup:
    pdf_id: int
    pdf_path: str
    page_index: int
    images: list[Image.Image]


def _infer_native_dpi(page: fitz.Page) -> int | None:
    images = page.get_images(full=True)
    if not images:
        return None

    best_dpi: float | None = None
    best_area = 0.0
    page_number = getattr(page, "number", getattr(page, "index", -1))

    for image in images:
        xref = image[0]
        width = image[2]
        height = image[3]
        try:
            rects = page.get_image_rects(xref)
        except Exception as exc:  # pragma: no cover - defensive against PyMuPDF edge cases
            logger.debug(
                "Failed to compute image rects for xref=%s on page %s: %s",
                xref,
                page_number,
                exc,
            )
            continue

        for rect in rects:
            area = rect.width * rect.height
            if area <= 0:
                continue

            width_inches = rect.width / 72.0
            height_inches = rect.height / 72.0
            if width_inches <= 0 or height_inches <= 0:
                continue

            x_dpi = width / width_inches
            y_dpi = height / height_inches
            candidate = max(x_dpi, y_dpi)
            if candidate <= 0:
                continue

            if area > best_area:
                best_area = area
                best_dpi = candidate
            elif best_area and abs(area - best_area) < 1e-6:
                if best_dpi is None or candidate > best_dpi:
                    best_dpi = candidate

    if best_dpi is None:
        return None

    return max(int(round(best_dpi)), 1)


def _raster_batch(
    pdf_path: str,
    page_numbers: list[int],
    dpi_override: int | None,
    fallback_dpi: int,
) -> list[tuple[int, bytes]]:
    doc = fitz.open(pdf_path)
    out: list[tuple[int, bytes]] = []
    try:
        for page_number in page_numbers:
            try:
                page = doc[page_number]
                target_dpi = dpi_override
                native_dpi = None
                if target_dpi is None:
                    native_dpi = _infer_native_dpi(page)
                    target_dpi = native_dpi or fallback_dpi
                    if native_dpi is None:
                        logger.debug(
                            "Falling back to %s DPI for %s page %s; no native DPI detected.",
                            fallback_dpi,
                            pdf_path,
                            page_number,
                        )

                zoom = max(target_dpi, 1) / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))  # type: ignore[attr-defined]
                out.append((page_number, pix.tobytes("png")))
            except Exception as exc:  # pragma: no cover
                logger.error(f"Failed to rasterize page {page_number} of {pdf_path}: {exc}")
        return out
    finally:
        doc.close()


class RasterizationStage:
    """CPU-bound PDF rasterization running in a process pool."""

    def __init__(
        self,
        *,
        dpi: int | None,
        batch_pages: int,
        max_workers: int,
        fallback_dpi: int = DEFAULT_RASTER_DPI,
    ) -> None:
        self._dpi_override = dpi
        self._batch_pages = batch_pages
        self._max_workers = max_workers
        self._fallback_dpi = fallback_dpi

    async def produce(
        self,
        pdf_paths: Sequence[str],
        queue: asyncio.Queue[RasterTask | None],
        *,
        start_pdf_id: int = 0,
    ) -> None:
        loop = asyncio.get_running_loop()

        from multiprocessing import get_context  # type: ignore

        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=self._max_workers, mp_context=ctx) as pool:
            for offset, pdf_path in enumerate(pdf_paths):
                pdf_id = start_pdf_id + offset
                try:
                    with fitz.open(pdf_path) as doc:
                        page_numbers = list(range(doc.page_count))
                except Exception as exc:
                    logger.error(f"Skipping {pdf_path}: cannot open ({exc})")
                    continue

                batches = [
                    page_numbers[i : i + self._batch_pages]
                    for i in range(0, len(page_numbers), self._batch_pages)
                ]
                for batch in batches:
                    while queue.full():
                        await asyncio.sleep(0)
                    fut = loop.run_in_executor(
                        pool,
                        _raster_batch,
                        pdf_path,
                        batch,
                        self._dpi_override,
                        self._fallback_dpi,
                    )
                    for page_index, png_bytes in await fut:
                        await queue.put(
                            RasterTask(
                                pdf_id=pdf_id,
                                pdf_path=pdf_path,
                                page_index=page_index,
                                png_bytes=png_bytes,
                            )
                        )


class ImageIngestionStage:
    """Stage that normalizes supplied image files into raster tasks."""

    async def produce(
        self,
        image_paths: Sequence[str],
        queue: asyncio.Queue[RasterTask | None],
        *,
        start_pdf_id: int,
    ) -> None:
        for offset, image_path in enumerate(image_paths):
            while queue.full():
                await asyncio.sleep(0)
            try:
                with Image.open(image_path) as img:
                    buffer = BytesIO()
                    img.convert("RGB").save(buffer, format="PNG")
            except Exception as exc:
                logger.error(f"Skipping {image_path}: cannot open/convert image ({exc})")
                continue
            await queue.put(
                RasterTask(
                    pdf_id=start_pdf_id + offset,
                    pdf_path=image_path,
                    page_index=0,
                    png_bytes=buffer.getvalue(),
                )
            )


class PageSplitterFactory(Protocol):
    def __call__(self, engine: str) -> PageSplitter: ...


class PageTrimmerFactory(Protocol):
    def __call__(self, enable_trim: bool) -> PageTrimmer: ...


class PageSplitter(Protocol):
    async def split(self, image: Image.Image) -> list[Image.Image]: ...


class PageTrimmer(Protocol):
    async def trim(self, image: Image.Image) -> Image.Image: ...


class LLMPageSplitter(PageSplitter):
    """Default splitter that consults an LLM to decide whether to split pages."""

    def __init__(self, *, engine: str) -> None:
        self._engine = engine

    async def split(self, image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        if height <= 0:
            logger.debug(
                f"Skipping LLM page split because image height is non-positive (width={width}, height={height})."
            )
            return [image]

        aspect_ratio = width / height
        if aspect_ratio <= SINGLE_PAGE_ASPECT_RATIO_THRESHOLD:
            logger.debug(
                f"Skipping LLM page split for narrow image (width={width}, height={height}, aspect_ratio={aspect_ratio:.3f})."
            )
            return [image]

        llm_output = await run_llm_async(
            system_prompt_text=None,
            user_message_text=PAGE_SPLIT_PROMPT,
            user_message_image=image,
            image_detail="low",
            model=self._engine,
        )
        number_of_pages = _parse_number_of_pages(llm_output)
        return split_double_page(image) if number_of_pages == 2 else [image]


class LayoutTrimmer(PageTrimmer):
    async def trim(self, image: Image.Image) -> Image.Image:
        return await tidy_image_via_layout_detection(image)


class IdentityTrimmer(PageTrimmer):
    async def trim(self, image: Image.Image) -> Image.Image:
        return image


def default_splitter_factory(engine: str) -> PageSplitter:
    return LLMPageSplitter(engine=engine)


def default_trimmer_factory(enable_trim: bool) -> PageTrimmer:
    return LayoutTrimmer() if enable_trim else IdentityTrimmer()


class PageProcessingStage:
    """Async stage that performs LLM-based page splitting and optional trimming."""

    def __init__(
        self,
        *,
        splitter: PageSplitter,
        trimmer: PageTrimmer,
        concurrency_limit: int,
    ) -> None:
        self._splitter = splitter
        self._trimmer = trimmer
        self._semaphore = asyncio.Semaphore(concurrency_limit)

    def spawn_workers(
        self,
        *,
        count: int,
        raster_queue: asyncio.Queue[RasterTask | None],
        processed_queue: asyncio.Queue[PageSplitGroup | None],
    ) -> list[asyncio.Task[None]]:
        return [
            asyncio.create_task(self._worker(raster_queue, processed_queue)) for _ in range(count)
        ]

    async def _worker(
        self,
        raster_queue: asyncio.Queue[RasterTask | None],
        processed_queue: asyncio.Queue[PageSplitGroup | None],
    ) -> None:
        while True:
            task = await raster_queue.get()
            if task is None:
                raster_queue.task_done()
                break

            try:
                image = Image.open(BytesIO(task.png_bytes))
            except Exception as exc:
                logger.error(
                    f"Failed to decode PNG bytes for pdf_id={getattr(task, 'pdf_id', 'unknown')} page={getattr(task, 'page_index', 'unknown')}: {exc}",
                )
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

            async with self._semaphore:
                try:
                    splits = await self._splitter.split(image)
                except Exception as exc:
                    logger.error(
                        f"Split failed (pdf_id={task.pdf_id}, page={task.page_index}): {exc}",
                    )
                    splits = [image]

            results: list[Image.Image] = []
            for split in splits:
                try:
                    trimmed = await self._trimmer.trim(split)
                except Exception as exc:
                    logger.error(
                        f"Trim failed (pdf_id={task.pdf_id}, page={task.page_index}): {exc}",
                    )
                    trimmed = split
                results.append(trimmed)

            await processed_queue.put(
                PageSplitGroup(
                    pdf_id=task.pdf_id,
                    pdf_path=task.pdf_path,
                    page_index=task.page_index,
                    images=results,
                )
            )
            raster_queue.task_done()


class SavingStage:
    """Saver that writes output images in deterministic order."""

    def __init__(self, output_dir: str, progress: ProgressReporter | None = None) -> None:
        self._output_dir = output_dir
        self._splits_counter: dict[int, int] = {}
        self._progress = progress

    @property
    def splits_counter(self) -> dict[int, int]:
        return self._splits_counter

    async def consume(self, queue: asyncio.Queue[PageSplitGroup | None]) -> None:
        os.makedirs(self._output_dir, exist_ok=True)
        pending: dict[int, dict[int, list[Image.Image]]] = {}
        next_expected: dict[int, int] = {}

        while True:
            group = await queue.get()
            if group is None:
                queue.task_done()
                break

            pending.setdefault(group.pdf_id, {})[group.page_index] = group.images
            next_expected.setdefault(group.pdf_id, 0)
            self._splits_counter.setdefault(group.pdf_id, 0)

            while next_expected[group.pdf_id] in pending[group.pdf_id]:
                images = pending[group.pdf_id].pop(next_expected[group.pdf_id])
                stem, _ = os.path.splitext(os.path.basename(group.pdf_path))
                pdf_stem = stem.replace(" ", "_")

                for image in images:
                    logical_index = self._splits_counter[group.pdf_id]
                    filename = f"{pdf_stem}_page_{logical_index:04d}.png"
                    out_path = os.path.join(self._output_dir, filename)
                    try:
                        image.save(out_path, "PNG")
                    except Exception as exc:  # pragma: no cover
                        logger.error(f"Failed saving {out_path}: {exc}")
                    self._splits_counter[group.pdf_id] = logical_index + 1
                next_expected[group.pdf_id] += 1
                if self._progress:
                    self._progress.increment()

            queue.task_done()


def _parse_number_of_pages(llm_output: str) -> int:
    number_of_pages_str = extract_tag_from_llm_output(llm_output, tags="number_of_pages")
    try:
        if isinstance(number_of_pages_str, list):
            return int(number_of_pages_str[0]) if number_of_pages_str else 1
        return int(number_of_pages_str)
    except Exception:
        return 1


PAGE_SPLIT_PROMPT = (
    "You are given an image of a scanned document. Your task is to decide whether the image "
    "actually contains ONE page or a TWO-PAGE SPREAD (i.e., two distinct adjacent pages captured together).\n\n"
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
)


@dataclass(slots=True)
class PdfPipelineConfig:
    engine: str
    output_dir: str
    dpi: int | None = 300
    batch_pages: int = 8
    queue_maxsize: int = 64
    raster_workers: int | None = None
    page_workers: int | None = None
    llm_concurrency_limit: int = 64
    trim: bool = True
    splitter_factory: PageSplitterFactory | None = None
    trimmer_factory: PageTrimmerFactory | None = None
    fallback_dpi: int = DEFAULT_RASTER_DPI


class PdfPipeline:
    """High-level coordinator for staged PDF/image processing."""

    def __init__(
        self,
        config: PdfPipelineConfig,
        *,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self._config = config
        self._progress = progress_reporter

    async def run(
        self,
        *,
        pdf_paths: Iterable[str],
        image_dir: str | None = None,
        image_paths: Iterable[str] | None = None,
    ) -> None:
        pdf_files = self._dedupe_pdfs(list(pdf_paths))
        image_files = self._collect_images(image_dir, image_paths)

        if not pdf_files and not image_files:
            logger.warning("No valid PDF or image inputs provided to pipeline.")
            return

        os.makedirs(self._config.output_dir, exist_ok=True)

        progress_started = False
        total_units = self._estimate_total_work(pdf_files, image_files)
        if self._progress and total_units:
            self._progress.start(total_units)
            progress_started = True

        raster_workers = self._config.raster_workers or max(1, (os.cpu_count() or 2) // 2)
        page_workers = self._config.page_workers or max(2, os.cpu_count() or 2)

        raster_queue: asyncio.Queue[RasterTask | None] = asyncio.Queue(
            maxsize=self._config.queue_maxsize
        )
        processed_queue: asyncio.Queue[PageSplitGroup | None] = asyncio.Queue(
            maxsize=self._config.queue_maxsize
        )

        raster_stage = RasterizationStage(
            dpi=self._config.dpi,
            batch_pages=self._config.batch_pages,
            max_workers=raster_workers,
            fallback_dpi=self._config.fallback_dpi,
        )
        image_stage = ImageIngestionStage()
        splitter_factory = self._config.splitter_factory or default_splitter_factory
        trimmer_factory = self._config.trimmer_factory or default_trimmer_factory

        splitter = splitter_factory(self._config.engine)
        trimmer = trimmer_factory(self._config.trim)
        processor_stage = PageProcessingStage(
            splitter=splitter,
            trimmer=trimmer,
            concurrency_limit=self._config.llm_concurrency_limit,
        )
        saver_stage = SavingStage(self._config.output_dir, progress=self._progress)

        try:
            producer_tasks: list[asyncio.Task[None]] = []
            if pdf_files:
                producer_tasks.append(
                    asyncio.create_task(
                        raster_stage.produce(pdf_files, raster_queue, start_pdf_id=0)
                    )
                )
            if image_files:
                producer_tasks.append(
                    asyncio.create_task(
                        image_stage.produce(
                            image_files,
                            raster_queue,
                            start_pdf_id=len(pdf_files),
                        )
                    )
                )

            worker_tasks = processor_stage.spawn_workers(
                count=page_workers, raster_queue=raster_queue, processed_queue=processed_queue
            )
            saver_task = asyncio.create_task(saver_stage.consume(processed_queue))

            if producer_tasks:
                await asyncio.gather(*producer_tasks)
            for _ in worker_tasks:
                await raster_queue.put(None)

            await raster_queue.join()
            await processed_queue.put(None)
            await asyncio.gather(*worker_tasks)
            await processed_queue.join()
            await saver_task

            logger.info(
                f"Pipeline complete. PDFs processed: {len(pdf_files)} | Images processed: {len(image_files)} "
                f"| Total pages: {sum(saver_stage.splits_counter.values())}"
            )
        finally:
            if progress_started and self._progress:
                self._progress.close()

    def _estimate_total_work(self, pdf_files: Sequence[str], image_files: Sequence[str]) -> int:
        total_units = len(image_files)
        for path in pdf_files:
            try:
                with fitz.open(path) as doc:
                    total_units += doc.page_count
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(f"Failed to open {path} for progress estimation: {exc}")
        return total_units

    def _dedupe_pdfs(self, pdf_paths: list[str]) -> list[str]:
        if not pdf_paths:
            return []
        pdf_files = [
            path for path in pdf_paths if os.path.isfile(path) and path.lower().endswith(".pdf")
        ]
        if pdf_paths and not pdf_files:
            logger.warning("No valid PDF files found; skipping PDF rasterization stage.")
        seen: set[str] = set()
        unique: list[str] = []
        for path in pdf_files:
            if path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    def _collect_images(
        self,
        image_dir: str | None,
        image_paths: Iterable[str] | None,
    ) -> list[str]:
        image_files: list[str] = []
        if image_dir:
            if os.path.isdir(image_dir):
                image_files.extend(_collect_image_paths(image_dir))
                if not image_files:
                    logger.warning(
                        "Image directory provided but no supported image files were found; skipping image stage."
                    )
            else:
                logger.warning(f"Image directory does not exist or is not a directory: {image_dir}")

        if image_paths:
            for image_path in image_paths:
                if not os.path.isfile(image_path):
                    logger.warning(f"Image path does not exist or is not a file: {image_path}")
                    continue
                ext = os.path.splitext(image_path)[1].lower()
                if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                    logger.warning(
                        f"Unsupported image extension '{ext}' for path {image_path}; skipping."
                    )
                    continue
                image_files.append(image_path)

        seen: set[str] = set()
        unique: list[str] = []
        for path in image_files:
            if path not in seen:
                seen.add(path)
                unique.append(path)
        return unique


async def run_pdf_pipeline(
    pdf_paths: Iterable[str],
    output_dir: str,
    engine: str,
    dpi: int | None = 300,
    batch_pages: int = 8,
    queue_maxsize: int = 64,
    raster_workers: int | None = None,
    page_workers: int | None = None,
    llm_concurrency_limit: int = 64,
    trim: bool = True,
    image_dir: str | None = None,
    image_paths: Iterable[str] | None = None,
    splitter_factory: PageSplitterFactory | None = None,
    trimmer_factory: PageTrimmerFactory | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> None:
    config = PdfPipelineConfig(
        engine=engine,
        output_dir=output_dir,
        dpi=dpi,
        batch_pages=batch_pages,
        queue_maxsize=queue_maxsize,
        raster_workers=raster_workers,
        page_workers=page_workers,
        llm_concurrency_limit=llm_concurrency_limit,
        trim=trim,
        splitter_factory=splitter_factory,
        trimmer_factory=trimmer_factory,
    )
    pipeline = PdfPipeline(config, progress_reporter=progress_reporter)
    try:
        await pipeline.run(pdf_paths=pdf_paths, image_dir=image_dir, image_paths=image_paths)
    finally:
        try:
            await shutdown_layout_clients()
        finally:
            await shutdown_llm_clients()

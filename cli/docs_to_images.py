from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import typer

from churro.utils.concurrency import TqdmProgressReporter
from churro.utils.llm import shutdown_llm_clients
from churro.utils.llm.models import MODEL_MAP
from churro.utils.log_utils import logger
from churro.utils.pdf import run_pdf_pipeline
from churro.utils.pdf.runner import (
    SUPPORTED_IMAGE_EXTENSIONS,
    default_splitter_factory,
    default_trimmer_factory,
)


DEFAULT_EXTENSIONS: tuple[str, ...] = (".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif")
DEFAULT_ENGINE = "gemini-2.5-pro-low"
DEFAULT_PATTERN = "*"
# Fallback DPI used when native page resolution cannot be determined.
DEFAULT_DPI = 300
DEFAULT_BATCH_PAGES = 16
DEFAULT_QUEUE_MAXSIZE = 64
DEFAULT_LLM_CONCURRENCY_LIMIT = 64


@dataclass(slots=True)
class DocsToImagesOptions:
    input_dir: Path | None
    input_file: Path | None
    recursive: bool
    pattern: str
    extensions: Sequence[str]
    output_dir: Path
    engine: str
    dpi: int | None
    batch_pages: int
    queue_maxsize: int
    raster_workers: int | None
    page_workers: int | None
    llm_concurrency_limit: int
    no_trim: bool
    dry_run: bool


def _normalise_extensions(raw_exts: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for ext in raw_exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext in seen:
            continue
        seen.add(ext)
        cleaned.append(ext)
    return cleaned


def _collect_inputs(
    input_dir: Path | None,
    input_file: Path | None,
    recursive: bool,
    pattern: str,
    extensions: list[str],
) -> tuple[list[str], list[str]]:
    pdfs: list[str] = []
    images: list[str] = []

    def _add_path(path: Path) -> None:
        ext = path.suffix.lower()
        if ext == ".pdf":
            pdfs.append(str(path))
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(str(path))
        else:
            logger.warning(f"Skipping unsupported extension '{ext}' for file {path}")

    if input_file:
        if input_file.is_file():
            if input_file.suffix.lower() in extensions:
                _add_path(input_file)
            else:
                logger.warning(
                    f"--input-file {input_file} does not match the provided extensions; ignoring."
                )
        else:
            logger.warning(f"--input-file {input_file} is not a file; ignoring.")

    if input_dir:
        glob_pattern = f"**/{pattern}" if recursive else pattern
        for path in input_dir.glob(glob_pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            _add_path(path)

    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    return _dedupe(pdfs), _dedupe(images)


def _validate_engine(engine: str) -> None:
    if engine not in MODEL_MAP:
        keys_preview = ", ".join(list(MODEL_MAP.keys())[:10])
        logger.error(
            f"Engine '{engine}' not found in MODEL_MAP. Available keys (first 10): {keys_preview} ..."
        )
        raise typer.Exit(code=2)


async def run(options: DocsToImagesOptions) -> int:
    _validate_engine(options.engine)

    extensions = _normalise_extensions(options.extensions)
    if not extensions:
        logger.warning("No valid extensions provided.")
        return 0

    supported_extensions = [
        ext for ext in extensions if ext == ".pdf" or ext in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not supported_extensions:
        logger.warning("No supported extensions remain after filtering.")
        return 0

    pdf_files, image_files = _collect_inputs(
        input_dir=options.input_dir,
        input_file=options.input_file,
        recursive=options.recursive,
        pattern=options.pattern,
        extensions=supported_extensions,
    )

    if options.dry_run:
        for path in pdf_files + image_files:
            logger.info(f"DRY RUN: {path}")
        return 0

    if not pdf_files and not image_files:
        logger.warning("No valid PDF or image inputs provided to pipeline.")
        return 0

    progress = TqdmProgressReporter("docs-to-images")
    try:
        await run_pdf_pipeline(
            pdf_paths=pdf_files,
            output_dir=str(options.output_dir),
            engine=options.engine,
            dpi=options.dpi,
            batch_pages=options.batch_pages,
            queue_maxsize=options.queue_maxsize,
            raster_workers=options.raster_workers,
            page_workers=options.page_workers,
            llm_concurrency_limit=options.llm_concurrency_limit,
            trim=not options.no_trim,
            image_paths=image_files,
            splitter_factory=default_splitter_factory,
            trimmer_factory=default_trimmer_factory,
            progress_reporter=progress,
        )
    finally:
        progress.close()
        await shutdown_llm_clients()
    return 0

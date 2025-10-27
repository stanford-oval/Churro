from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from functools import wraps
import os
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import typer  # type: ignore[import]

from churro.systems.detect_layout import (
    log_total_azure_cost,
    log_total_google_document_ai_cost,
)
from churro.systems.ocr_factory import OCRFactory
from churro.utils.llm import log_total_llm_cost
from churro.utils.log_utils import logger

from . import benchmark, docs_to_images, infer, text_to_historical_doc_xml


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


app = typer.Typer(
    help="Churro OCR command-line interface",
)


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _normalize_suffixes(suffixes: list[str] | None, *, default: Sequence[str]) -> list[str]:
    """Convert bare suffix tokens (e.g. 'pdf') into dotted extensions."""
    if not suffixes:
        return list(default)
    cleaned: list[str] = []
    seen: set[str] = set()
    for suffix in suffixes:
        token = suffix.strip().lower()
        if not token:
            continue
        if token.startswith("."):
            token = token[1:]
        ext = f".{token}"
        if ext in seen:
            continue
        seen.add(ext)
        cleaned.append(ext)
    return cleaned or list(default)


def _synchronous(handler: Callable[_P, Coroutine[Any, Any, _T]]) -> Callable[_P, _T]:
    @wraps(handler)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return asyncio.run(handler(*args, **kwargs))
        except KeyboardInterrupt as err:
            logger.info("Interrupted by user")
            raise typer.Exit(code=130) from err

    return wrapper


@app.command("docs-to-images")
@_synchronous
async def docs_to_images_command(
    input_dir: Path | None = typer.Option(
        None,
        "--input-dir",
        help="Directory containing input files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    input_file: Path | None = typer.Option(
        None,
        "--input-file",
        help="Single file to process.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        help="Recurse into subdirectories when using --input-dir.",
    ),
    suffix: list[str] | None = typer.Option(
        None,
        "--suffix",
        "-s",
        help="File suffix without dot (e.g. pdf, png). Repeat to add more.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Destination directory for output PNG files (created if missing).",
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
    engine: str = typer.Option(
        docs_to_images.DEFAULT_ENGINE,
        "--engine",
        help="Logical model key for page splitting (MODEL_MAP).",
        show_default=True,
    ),
    dpi: int | None = typer.Option(
        None,
        "--dpi",
        help="Rasterization DPI. Defaults to native page DPI when available.",
        show_default=False,
    ),
    batch_pages: int = typer.Option(
        docs_to_images.DEFAULT_BATCH_PAGES,
        "--batch-pages",
        help="Number of PDF pages per raster batch.",
        show_default=True,
    ),
    queue_maxsize: int = typer.Option(
        docs_to_images.DEFAULT_QUEUE_MAXSIZE,
        "--queue-maxsize",
        help="Maximum queue size for raster & processed queues.",
        show_default=True,
    ),
    raster_workers: int | None = typer.Option(
        None,
        "--raster-workers",
        help="Override number of raster process pool workers.",
    ),
    page_workers: int | None = typer.Option(
        None,
        "--page-workers",
        help="Override number of async page processing workers.",
    ),
    llm_concurrency_limit: int = typer.Option(
        docs_to_images.DEFAULT_LLM_CONCURRENCY_LIMIT,
        "--llm-concurrency-limit",
        help="Max simultaneous LLM calls for page splitting.",
        show_default=True,
    ),
    no_trim: bool = typer.Option(
        False,
        "--no-trim",
        help="Disable layout-based margin trimming.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List the PDFs that would be processed then exit without running pipeline.",
    ),
) -> int:
    if (input_dir is None) == (input_file is None):
        raise typer.BadParameter(
            "Provide exactly one of --input-dir or --input-file.",
            param_hint="--input-dir/--input-file",
        )

    options = docs_to_images.DocsToImagesOptions(
        input_dir=input_dir,
        input_file=input_file,
        recursive=recursive,
        pattern=docs_to_images.DEFAULT_PATTERN,
        extensions=_normalize_suffixes(suffix, default=docs_to_images.DEFAULT_EXTENSIONS),
        output_dir=output_dir,
        engine=engine,
        dpi=dpi,
        batch_pages=batch_pages,
        queue_maxsize=queue_maxsize,
        raster_workers=raster_workers,
        page_workers=page_workers,
        llm_concurrency_limit=llm_concurrency_limit,
        no_trim=no_trim,
        dry_run=dry_run,
    )
    result = await docs_to_images.run(options)
    log_total_llm_cost()
    log_total_azure_cost()
    log_total_google_document_ai_cost()
    if result != 0:
        raise typer.Exit(code=result)
    return result


@app.command("infer")
@_synchronous
async def infer_command(
    system: str = typer.Option(
        ...,
        "--system",
        help=f"OCR system identifier. Choices: {', '.join(infer.SYSTEM_CHOICES)}.",
    ),
    engine: str | None = typer.Option(
        None,
        "--engine",
        help="Logical engine key (MODEL_MAP) when the system requires an LLM backend.",
    ),
    backup_engine: str | None = typer.Option(
        None,
        "--backup-engine",
        help="Optional backup engine key (MODEL_MAP) for the OCR system.",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="(vLLM only) Tensor parallel size for launched container.",
        show_default=True,
    ),
    data_parallel_size: int = typer.Option(
        1,
        "--data-parallel-size",
        help="(vLLM only) Data parallel size for launched container.",
        show_default=True,
    ),
    image: Path | None = typer.Option(
        None,
        "--image",
        help="Path to a page image. If you have a PDF, rasterize first.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    image_dir: Path | None = typer.Option(
        None,
        "--image-dir",
        help="Directory containing images to batch transcribe.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    suffix: list[str] | None = typer.Option(
        None,
        "--suffix",
        "-s",
        help="Image suffix without dot (e.g. png). Repeat to add more.",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        help="Recurse into subdirectories when using --image-dir.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Write transcriptions to this directory as <image_basename>.txt (defaults to --image-dir when omitted).",
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip images that already have a corresponding output .txt in --output-dir.",
    ),
    max_concurrency: int = typer.Option(
        infer.DEFAULT_MAX_CONCURRENCY,
        "--max-concurrency",
        help="Max concurrent image requests to vLLM server.",
        show_default=True,
    ),
    strip_xml: bool = typer.Option(
        False,
        "--strip-xml",
        help="Finetuned system outputs HistoricalDocument XML by default. Set this flag to output plain text instead.",
    ),
    output_markdown: bool = typer.Option(
        False,
        "--output-markdown",
        help="Instruct the LLM to output Markdown instead of plain text (LLM system only).",
    ),
    use_improver: bool = typer.Option(
        False,
        "--use-improver",
        help="Post-process OCR text with the LLMImprover.",
    ),
    improver_engine: str = typer.Option(
        infer.DEFAULT_IMPROVER_ENGINE,
        "--improver-engine",
        help="Logical engine key (MODEL_MAP) for the LLM improver.",
        show_default=True,
    ),
    improver_backup_engine: str = typer.Option(
        infer.DEFAULT_IMPROVER_BACKUP_ENGINE,
        "--improver-backup-engine",
        help="Optional backup engine key for the LLM improver.",
        show_default=True,
    ),
    improver_resize: int | None = typer.Option(
        None,
        "--improver-resize",
        help="Resize longest image side to this many pixels before improvement.",
    ),
) -> int:
    if image is None and image_dir is None:
        raise typer.BadParameter(
            "Provide either --image or --image-dir.",
            param_hint="--image/--image-dir",
        )
    if image and image_dir:
        raise typer.BadParameter(
            "Specify only one of --image or --image-dir.",
            param_hint="--image/--image-dir",
        )

    system_key = system.lower()

    resolved_output_dir = output_dir
    if resolved_output_dir is None and image_dir is not None:
        resolved_output_dir = image_dir
        logger.info(
            "No --output-dir provided; using --image-dir as output destination.",
        )

    options = infer.InferOptions(
        system=system_key,
        engine=engine,
        backup_engine=backup_engine,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        image=image,
        image_dir=image_dir,
        pattern=infer.DEFAULT_PATTERN,
        suffixes=_normalize_suffixes(suffix, default=infer.DEFAULT_SUFFIXES),
        recursive=recursive,
        output_dir=resolved_output_dir,
        skip_existing=skip_existing,
        max_concurrency=max_concurrency,
        strip_xml=strip_xml,
        output_markdown=output_markdown,
        use_improver=use_improver,
        improver_engine=improver_engine if use_improver else None,
        improver_backup_engine=improver_backup_engine if use_improver else None,
        improver_resize=improver_resize,
    )
    result = await infer.run(options)
    log_total_llm_cost()
    log_total_azure_cost()
    log_total_google_document_ai_cost()
    if result != 0:
        raise typer.Exit(code=result)
    return result


@app.command("text-to-historical-doc-xml")
@_synchronous
async def text_to_historical_doc_xml_command(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing matched PNG/TXT pairs.",
    ),
    engine: str = typer.Option(
        text_to_historical_doc_xml.DEFAULT_ENGINE,
        "--engine",
        help="Logical model key to use for XML generation.",
        show_default=True,
    ),
    max_concurrency: int = typer.Option(
        text_to_historical_doc_xml.DEFAULT_MAX_CONCURRENCY,
        "--max-concurrency",
        help="Maximum number of concurrent LLM calls.",
        show_default=True,
    ),
    corpus_description: str = typer.Option(
        "",
        "--corpus-description",
        help="Optional corpus description to include in prompts.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Regenerate XML even if output already exists.",
    ),
) -> None:
    await text_to_historical_doc_xml.run_text_to_historical_doc_xml(
        input_dir,
        engine,
        max_concurrency,
        corpus_description,
        overwrite,
    )
    log_total_llm_cost()


@app.command("benchmark")
@_synchronous
async def benchmark_command(
    system: str = typer.Option(
        ...,
        "--system",
        help=f"Specify the system to run. Choices: {', '.join(OCRFactory.get_available_systems())}.",
    ),
    engine: str | None = typer.Option(
        None,
        "--engine",
        help="For LLM baseline, specify the LLM to use.",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Tensor parallel size. Only used for local models.",
        show_default=True,
    ),
    data_parallel_size: int = typer.Option(
        1,
        "--data-parallel-size",
        help="Data parallel size. Only used for local models.",
        show_default=True,
    ),
    resize: int | None = typer.Option(
        None,
        "--resize",
        help="If set, resize large images to fit inside a square of this size (in pixels).",
    ),
    max_concurrency: int = typer.Option(
        50,
        "--max-concurrency",
        help="Maximum number of LLM requests to allow at once.",
        show_default=True,
    ),
    input_size: int = typer.Option(
        0,
        "--input-size",
        help="Number of images to process. 0 means all images.",
        show_default=True,
    ),
    dataset_split: str = typer.Option(
        ...,
        "--dataset-split",
        help="Data split to use (dev or test).",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        help="Offset for the input images.",
        show_default=True,
    ),
) -> int:
    system_key = system.lower()
    dataset_split_value = dataset_split.lower()

    options = benchmark.BenchmarkOptions(
        system=system_key,
        engine=engine,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        resize=resize,
        max_concurrency=max_concurrency,
        input_size=input_size,
        dataset_split=dataset_split_value,
        offset=offset,
    )
    result = await benchmark.run(options)
    log_total_llm_cost()
    log_total_azure_cost()
    log_total_google_document_ai_cost()
    if result != 0:
        raise typer.Exit(code=result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        app(args=list(argv) if argv is not None else None, standalone_mode=False)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)


if __name__ == "__main__":  # pragma: no cover
    app()

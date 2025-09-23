#!/usr/bin/env python
"""Batch convert a directory of PDF files into page PNG images using the queued PDF pipeline.

This is a thin CLI wrapper over ``utils.pdf.run_pdf_pipeline`` that:
  * Recursively (optional) discovers PDF files in an input directory
  * Runs the asynchronous raster + LLM-based split + (optional) trim pipeline
  * Saves output PNGs to the specified output directory (flat naming)

Example:
  pixi run python pdf_dir_to_images.py \
      --input-dir /path/to/pdfs \
      --output-dir workdir/outputs/pages \
      --engine gemini-2.5-flash-noreasoning \
      --dpi 300 --no-trim

For a single file you can also simply pass --input-file.

The engine should be a logical model key from ``utils.llm.models.MODEL_MAP``.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from utils.llm import shutdown_llm_clients
from utils.llm.models import MODEL_MAP
from utils.log_utils import logger
from utils.pdf import run_pdf_pipeline


def collect_pdfs(
    input_dir: Path | None,
    input_file: Path | None,
    recursive: bool,
    pattern: str,
) -> list[str]:
    pdfs: list[str] = []
    if input_file:
        if input_file.is_file() and input_file.suffix.lower() == ".pdf":
            pdfs.append(str(input_file))
        else:
            logger.warning(f"--input-file {input_file} is not a PDF file; ignoring.")
    if input_dir:
        glob_pattern = f"**/{pattern}" if recursive else pattern
        for p in input_dir.glob(glob_pattern):
            if p.is_file() and p.suffix.lower() == ".pdf":
                pdfs.append(str(p))
    # Deduplicate while preserving order
    seen = set()
    ordered: list[str] = []
    for p in pdfs:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert PDFs to page images with splitting & trimming."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=Path, help="Directory containing PDF files.")
    src.add_argument("--input-file", type=Path, help="Single PDF file to process.")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when using --input-dir.",
    )
    p.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for PDFs inside --input-dir (default: *.pdf).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for output PNG files (created if missing).",
    )
    p.add_argument(
        "--engine",
        default="gemini-2.5-flash-noreasoning",
        help="Logical model key for LLM page splitting (must exist in MODEL_MAP).",
    )
    p.add_argument("--dpi", type=int, default=300, help="Rasterization DPI (default 300).")
    p.add_argument(
        "--batch-pages",
        type=int,
        default=8,
        help="Number of PDF pages per raster batch submitted to process pool.",
    )
    p.add_argument(
        "--queue-maxsize",
        type=int,
        default=64,
        help="Maximum queue size for raster & processed queues (memory control).",
    )
    p.add_argument(
        "--raster-workers",
        type=int,
        default=None,
        help="Override number of raster process pool workers (default: half CPUs).",
    )
    p.add_argument(
        "--page-workers",
        type=int,
        default=None,
        help="Override number of async page processing workers (default: CPUs).",
    )
    p.add_argument(
        "--llm-concurrency-limit",
        type=int,
        default=64,
        help="Max simultaneous LLM calls for page splitting.",
    )
    p.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable layout-based margin trimming (faster, keeps raw scans).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the PDFs that would be processed then exit without running pipeline.",
    )
    return p


def validate_engine(engine: str) -> None:
    if engine not in MODEL_MAP:
        logger.error(
            f"Engine '{engine}' not found in MODEL_MAP. Available keys: {list(MODEL_MAP.keys())} ..."
        )
        raise SystemExit(2)


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_engine(args.engine)

    pdfs = collect_pdfs(args.input_dir, args.input_file, args.recursive, args.pattern)
    if not pdfs:
        logger.warning("No PDF files found.")
        return 0

    preview = pdfs[:3]
    logger.info(f"Processing {len(pdfs)} PDF file(s). First {len(preview)}: {preview}")
    if args.dry_run:
        for p in pdfs:
            logger.info(f"DRY RUN: {p}")
        return 0

    # Run pipeline
    async def _run():
        await run_pdf_pipeline(
            pdf_paths=pdfs,
            output_dir=str(args.output_dir),
            engine=args.engine,
            dpi=args.dpi,
            batch_pages=args.batch_pages,
            queue_maxsize=args.queue_maxsize,
            raster_workers=args.raster_workers,
            page_workers=args.page_workers,
            llm_concurrency_limit=args.llm_concurrency_limit,
            trim=not args.no_trim,
        )

    async def _main():
        try:
            await _run()
        finally:
            # Ensure we attempt cleanup within the same event loop.
            try:
                await shutdown_llm_clients()
            except Exception:  # pragma: no cover - defensive
                pass

    asyncio.run(_main())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

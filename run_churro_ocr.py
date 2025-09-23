#!/usr/bin/env python
"""Inference script for the finetuned historical OCR VLM served via local vLLM.

This trimmed version supports ONLY the vLLM engine path (container based). All direct
Hugging Face Transformers model loading / streaming paths have been removed to reduce
duplication with the broader project inference stack.

Capabilities:
    * Run on a single image OR over a directory (recursive optional)
    * Asynchronous parallel requests to the local vLLM server (bounded by --max_concurrency)
    * Natural sort ordering of images for deterministic output ordering
    * Optional per-image .txt output directory, skipping existing outputs

Examples:
    # Single image
    pixi run python run_churro_ocr.py --engine churro --image tests/ahisto_103_84.jpeg \
            --prompt "Transcribe the entirety of this historical document to XML format." --max-new-tokens 20000

    # Directory of images (PNG/JPEG) recursively with concurrency 4
    pixi run python run_churro_ocr.py --engine churro --image-dir path/to/pages --recursive \
            --pattern "*.png" --max_concurrency 16 --output-dir workdir/o/texts
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import re

from PIL import Image

from utils.docker.vllm import has_at_least_one_vllm, maybe_start_vllm_server_for_engine
from utils.llm.core import run_llm_async
from utils.llm.models import MODEL_MAP
from utils.log_utils import logger
from utils.utils import run_async_in_parallel


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def trim_leading_prompt(text: str, prompt: str) -> str:
    """Remove the leading prompt text if the model duplicated it.

    Args:
        text: Generated model output.
        prompt: Prompt that may have been prepended by the model.

    Returns:
        Cleaned text without the duplicated prompt prefix.
    """
    if prompt and text.startswith(prompt):
        return text[len(prompt) :].lstrip("\n ")
    return text


def write_or_print_output(
    img_path: Path,
    text: str,
    output_dir: Path | None,
    skip_existing: bool,
    multi_mode: bool,
) -> None:
    """Write transcription to file or print to stdout.

    Centralizes logic shared by HF and vLLM paths.
    """
    out_path: Path | None = None
    if output_dir:
        out_path = output_dir / (img_path.stem + ".txt")
        if skip_existing and out_path.exists():
            logger.info(f"Skipping existing {out_path.name}")
            return
    if out_path is not None:
        out_path.write_text(text)
        logger.info(f"Wrote {out_path}")
    else:
        if multi_mode:
            header = f"===== {img_path} ====="
            print(header)
        print(text)


def start_vllm_container(args: argparse.Namespace):
    """Start a vLLM container for the provided engine key.

    Raises SystemExit if validation fails.
    """
    if args.engine not in MODEL_MAP or not has_at_least_one_vllm(args.engine):
        logger.error(
            f"--engine '{args.engine}' not found in MODEL_MAP with a vLLM provider variant."
        )
        raise SystemExit(1)
    logger.info(
        f"Starting vLLM engine '{args.engine}' (tp={args.tensor_parallel_size}, dp={args.data_parallel_size})"
    )
    return maybe_start_vllm_server_for_engine(
        engine=args.engine,
        system="finetuned",
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        install_flash_attn=True,
    )


async def run_vllm_inference(
    images: list[Path],
    args: argparse.Namespace,
    multi_mode: bool,
) -> int:
    """Run inference against the local vLLM server."""

    async def _infer_one(img_path: Path) -> str:
        try:
            img = load_image(img_path)
            return await run_llm_async(
                model=args.engine,  # type: ignore[arg-type]
                system_prompt_text=args.prompt,
                user_message_text=None,
                user_message_image=img,
            )
        except Exception as e:  # pragma: no cover - runtime safety
            logger.error(f"Inference failed for {img_path}: {e}")
            return ""

    async def _run_parallel() -> list[str]:
        return await run_async_in_parallel(
            _infer_one,
            images,
            max_concurrency=max(1, args.max_concurrency),
            desc="Running via vLLM",
        )

    outputs = await _run_parallel()
    for img_path, raw_text in zip(images, outputs):
        text = trim_leading_prompt(raw_text, args.prompt)
        write_or_print_output(
            img_path=img_path,
            text=text,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            multi_mode=multi_mode,
        )
    return 0


def collect_images(
    image: Path | None,
    image_dir: Path | None,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    """Collect one or many image paths.

    Deduplicates while preserving order (initially), caller may sort.
    """
    images: list[Path] = []
    if image and image.is_file():
        if image.suffix.lower() in ALLOWED_IMAGE_EXTS:
            images.append(image)
        else:  # pragma: no cover - defensive
            logger.warning(f"--image {image} does not have a supported extension; skipping.")
    if image_dir and image_dir.is_dir():
        glob_pattern = f"**/{pattern}" if recursive else pattern
        for p in image_dir.glob(glob_pattern):
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS:
                images.append(p)
    # Deduplicate
    seen: set[str] = set()
    ordered: list[Path] = []
    for p in images:
        s = str(p)
        if s not in seen:
            seen.add(s)
            ordered.append(p)
    return ordered


def _natural_key(p: Path):
    """Return a tuple key for natural sorting of filenames (page2 before page10)."""
    parts = re.split(r"(\d+)", p.name)
    return tuple(int(s) if s.isdigit() else s.lower() for s in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run OCR inference via a local vLLM-served vision-language model (single image or directory)."
    )
    p.add_argument(
        "--engine",
        required=True,
        help="Logical engine key (MODEL_MAP) to serve via local vLLM container.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="(vLLM only) Tensor parallel size for launched container (ignored with --hf-model).",
    )
    p.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="(vLLM only) Data parallel size for launched container (ignored with --hf-model).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image",
        type=Path,
        help="Path to a page image (JPEG/PNG). If you have a PDF, rasterize first.",
    )
    src.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to batch transcribe.",
    )
    p.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern inside --image-dir (default: *.png). Also accepts other typical image extensions.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when using --image-dir.",
    )
    p.add_argument(
        "--prompt",
        default="Transcribe the entirety of this historical document to XML format.",
        help="System text prompt.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="If set, write each transcription to this directory as <image_basename>.txt.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have a corresponding output .txt in --output-dir.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Max concurrent image requests to vLLM server.",
    )
    return p


## Removed dtype detection (HF path removed).


def load_image(path: str | Path) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


async def main():
    args = build_arg_parser().parse_args()

    # Collect image(s)
    images = collect_images(args.image, args.image_dir, args.pattern, args.recursive)
    # Sort images by natural filename order for deterministic processing
    images = sorted(images, key=_natural_key)
    if not images:
        logger.error("No images found to process.")
        return 1

    multi_mode = len(images) > 1
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    vllm_container = maybe_start_vllm_server_for_engine(
        engine=args.engine,
        system="finetuned",
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
    )

    if multi_mode:
        logger.info(f"Processing {len(images)} image(s)...")

    if args.max_concurrency < 1:
        logger.warning("--max_concurrency < 1 ignored; defaulting to 1")
        args.max_concurrency = 1

    await run_vllm_inference(images, args, multi_mode)

    try:
        vllm_container.stop()  # type: ignore[union-attr]
        logger.info("Stopped vLLM container.")
    except Exception:  # pragma: no cover - defensive
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())

"""Command-line interface for Gemini page boundary detection."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from PIL import Image

from churro.utils.llm import log_total_llm_cost
from churro.utils.log_utils import logger

from ._constants import DEFAULT_MODEL_KEY, MAX_PAGE_REVIEW_ROUNDS
from ._pipeline import save_page_crops
from ._serialization import boxes_to_json_payload
from .detector import run_page_detection


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running the Gemini page boundary detector."""
    parser = argparse.ArgumentParser(
        description="Detect document page boundaries using Gemini.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image containing one or two document pages.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for the annotated image. Defaults to <input>_boxed.png.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_KEY,
        help="Logical model key defined in utils.llm.models.MODEL_MAP.",
    )
    parser.add_argument(
        "--max-review-rounds",
        type=int,
        default=MAX_PAGE_REVIEW_ROUNDS,
        help="Maximum number of Gemini review passes to perform.",
    )
    return parser.parse_args()


async def _async_main(
    image_path: Path,
    output_path: Path,
    model_key: str,
    max_review_rounds: int,
) -> None:
    """Run the detection pipeline and persist annotated outputs asynchronously."""
    with Image.open(image_path) as img:
        original_rgb = img.convert("RGB")
        detection_result = await run_page_detection(
            original_rgb,
            model_key=model_key,
            max_review_rounds=max_review_rounds,
        )
        detection_result.annotated_image.save(output_path)

        crop_paths = save_page_crops(detection_result.crops, output_path)
        for idx, polygon in enumerate(detection_result.polygons_original, start=1):
            left, top, right, bottom = polygon.bounds
            logger.info(
                f"Page {idx} original coords: left={left:.1f}, top={top:.1f},"
                f" right={right:.1f}, bottom={bottom:.1f}",
            )
        for path in crop_paths:
            logger.info(f"Saved page crop: {path}")
        logger.info(f"Final boxes JSON: {boxes_to_json_payload(detection_result.boxes)}")
        logger.info(
            f"Saved annotated image with {len(detection_result.boxes)} page box(es)"
            f" to '{output_path}'",
        )
    log_total_llm_cost()


def main() -> None:
    """CLI entry point for Gemini page boundary detection."""
    args = parse_args()
    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Missing input image: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Input path must be a file: {image_path}")

    default_output = image_path.with_name(f"{image_path.stem}_boxed.png")
    output_path: Path = args.output or default_output

    asyncio.run(_async_main(image_path, output_path, args.model, args.max_review_rounds))


__all__ = ["main"]


if __name__ == "__main__":
    main()

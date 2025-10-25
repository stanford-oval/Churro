"""High-level Gemini page boundary detector interfaces."""

from __future__ import annotations

from PIL import Image

from ._constants import DEFAULT_MODEL_KEY, MAX_PAGE_REVIEW_ROUNDS
from ._image_processing import (
    convert_boxes_to_original_polygons,
    draw_boxes,
    extract_crops,
    prepare_page_image,
)
from ._models import PageDetectionResult
from ._pipeline import run_detection_pipeline


class GeminiPageBoundaryDetector:
    """Runs the Gemini page detection pipeline with optional review rounds."""

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL_KEY,
        max_review_rounds: int = MAX_PAGE_REVIEW_ROUNDS,
    ) -> None:
        self.model_key = model_key
        self.max_review_rounds = max(0, max_review_rounds)

    async def detect(self, image: Image.Image) -> PageDetectionResult:
        processed_image, transform = prepare_page_image(image)
        final_boxes = await run_detection_pipeline(
            processed_image,
            model_key=self.model_key,
            max_review_rounds=self.max_review_rounds,
        )
        annotated = draw_boxes(processed_image, final_boxes)
        polygons_original = convert_boxes_to_original_polygons(final_boxes, transform)
        crops = extract_crops(image, polygons_original)
        return PageDetectionResult(
            crops=crops,
            boxes=final_boxes,
            polygons_aligned=polygons_original,
            polygons_original=polygons_original,
            annotated_image=annotated,
            transform=transform,
        )


async def run_page_detection(
    image: Image.Image,
    model_key: str = DEFAULT_MODEL_KEY,
    max_review_rounds: int = MAX_PAGE_REVIEW_ROUNDS,
) -> PageDetectionResult:
    """Run Gemini page detection and return crops plus metadata."""
    detector = GeminiPageBoundaryDetector(
        model_key=model_key,
        max_review_rounds=max_review_rounds,
    )
    return await detector.detect(image)


__all__ = [
    "GeminiPageBoundaryDetector",
    "run_page_detection",
]

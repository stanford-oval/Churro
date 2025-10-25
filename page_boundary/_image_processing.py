"""Image preprocessing helpers for Gemini page detection."""

from __future__ import annotations

from collections.abc import Iterable

from PIL import Image, ImageDraw, ImageOps

from churro.page import PageObject, extract_polygon_region
from churro.utils.image.transform import adjust_image, resize_image_to_fit

from ._constants import (
    BORDER_FRACTION,
    GUIDELINE_COLOR,
    PAGE_DETECTION_BOX_WIDTH,
    PROCESSED_MAX_DIM,
)
from ._models import PageBox, PageDetectionTransform


def _add_white_border(
    image: Image.Image, fraction: float = BORDER_FRACTION
) -> tuple[Image.Image, int, int]:
    """Add a white border around the image proportional to the input size."""
    if fraction <= 0:
        return image, 0, 0
    border_w = max(1, int(round(image.width * fraction)))
    border_h = max(1, int(round(image.height * fraction)))
    expanded = ImageOps.expand(
        image,
        border=(border_w, border_h, border_w, border_h),
        fill="white",
    )
    return expanded, border_w, border_h


def prepare_page_image(image: Image.Image) -> tuple[Image.Image, PageDetectionTransform]:
    """Normalize an input image so Gemini receives a padded and size-limited RGB copy."""
    original_size = image.size
    rgb_image = image.convert("RGB")
    grayscale = adjust_image(rgb_image, thresholding=True)
    padded, border_w, border_h = _add_white_border(grayscale)
    padded_size = padded.size
    processed = resize_image_to_fit(padded, PROCESSED_MAX_DIM, PROCESSED_MAX_DIM)
    processed_size = processed.size
    scale_x = processed_size[0] / padded_size[0] if padded_size[0] else 1.0
    scale_y = processed_size[1] / padded_size[1] if padded_size[1] else 1.0
    processed_rgb = processed.convert("RGB") if processed.mode != "RGB" else processed
    transform = PageDetectionTransform(
        original_size=original_size,
        border=(border_w, border_h),
        padded_size=padded_size,
        processed_size=processed_size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return processed_rgb, transform


def convert_boxes_to_original_polygons(
    boxes: Iterable[PageBox],
    transform: PageDetectionTransform,
) -> list[PageObject]:
    """Map normalized Gemini bounding boxes back to polygons on the original image."""
    processed_w, processed_h = transform.processed_size
    original_w, original_h = transform.original_size
    border_w, border_h = transform.border
    scale_x = transform.scale_x or 1.0
    scale_y = transform.scale_y or 1.0

    polygons: list[PageObject] = []
    for box in boxes:
        left_proc, top_proc, right_proc, bottom_proc = box.denormalize(processed_w, processed_h)
        left_padded = left_proc / scale_x
        right_padded = right_proc / scale_x
        top_padded = top_proc / scale_y
        bottom_padded = bottom_proc / scale_y

        left_orig = max(0.0, min(original_w, left_padded - border_w))
        right_orig = max(0.0, min(original_w, right_padded - border_w))
        top_orig = max(0.0, min(original_h, top_padded - border_h))
        bottom_orig = max(0.0, min(original_h, bottom_padded - border_h))

        polygons.append(
            PageObject.from_bounds(
                left_orig,
                top_orig,
                right_orig,
                bottom_orig,
                object_id=f"box-{box.page_index}-{len(polygons)}",
            )
        )

    return polygons


def draw_boxes(image: Image.Image, boxes: Iterable[PageBox]) -> Image.Image:
    """Overlay denormalized boxes onto a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for box in boxes:
        left, top, right, bottom = box.denormalize(width, height)
        draw.rectangle(
            [left, top, right, bottom],
            outline=GUIDELINE_COLOR,
            width=PAGE_DETECTION_BOX_WIDTH,
        )
    return annotated


def extract_crops(source: Image.Image, polygons: Iterable[PageObject]) -> list[Image.Image]:
    """Create page crops from the provided polygons."""
    return [extract_polygon_region(source, polygon) for polygon in polygons]


__all__ = [
    "prepare_page_image",
    "convert_boxes_to_original_polygons",
    "draw_boxes",
    "extract_crops",
]

"""Minimal helper utilities for the queued PDF pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ocr.systems.detect_layout import detect_layout
from page.polygon import Polygon
from page.visualization import crop_image_and_page_to_content, extract_polygon_region
from utils.log_utils import logger


__all__ = [
    "find_brightest_line",
    "split_double_page",
    "trim_image",
]


def find_brightest_line(image: Image.Image, margin_ratio: float = 0.45) -> tuple[int, int]:
    """Locate a bright vertical (possibly slanted) separator line.

    A brute-force search scans candidate start/end x-coordinates (sampled every
    few pixels) and accumulates grayscale brightness along the implied line.
    The line with the highest summed intensity—subject to exclusion margins on
    the left/right edges—is returned.

    Args:
        image: Source PIL image (mode is converted internally to ``L``).
        margin_ratio: Fractional horizontal margin to ignore on both sides.

    Returns:
        (best_x0, best_x1) integer coordinates representing the top and bottom
        x positions of the detected line.
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=float)
    h, w = arr.shape

    margin_x_min = int(margin_ratio * w)
    margin_x_max = int((1 - margin_ratio) * w)

    best_x0 = 0
    best_x1 = 0
    max_sum = -1.0

    # Precompute all row indices for faster vector calculations
    y_indices = np.arange(h)

    step_size = 5
    for x0 in range(margin_x_min, margin_x_max, step_size):
        for x1 in range(margin_x_min, margin_x_max, step_size):
            slope = (x1 - x0) / h
            # Compute all x positions and round once using NumPy
            x_positions = x0 + slope * y_indices
            x_rounded = np.round(x_positions).astype(int)

            # Sum brightness in one vector operation
            total_brightness = float(np.sum(arr[y_indices, x_rounded]))

            if total_brightness > max_sum:
                max_sum = total_brightness
                best_x0 = x0
                best_x1 = x1

    return best_x0, best_x1


def split_double_page(image: Image.Image) -> list[Image.Image]:
    """Split a detected double-page scan into left and right page images.

    Uses the brightest vertical separator line (see ``find_brightest_line``)
    plus a small pixel margin to build polygons that are cropped out of the
    original image.
    """
    # Split the image in half
    best_x0, best_x1 = find_brightest_line(image)
    logger.debug(f"best_x0: {best_x0}, best_x1: {best_x1}")

    # Add margin of error to polygon coordinates
    margin = 10  # pixels

    left_polygon = Polygon(
        coordinates=[
            0,
            0,
            best_x0 + margin,
            0,
            best_x1 + margin,
            image.height,
            0,
            image.height,
        ]
    )
    left_image = extract_polygon_region(image=image, polygon=left_polygon)

    right_polygon = Polygon(
        coordinates=[
            best_x0 - margin,
            0,
            best_x1 - margin,
            image.height,
            image.width,
            image.height,
            image.width,
            0,
        ]
    )
    right_image = extract_polygon_region(image=image, polygon=right_polygon)

    return [left_image, right_image]


async def trim_image(image: Image.Image) -> Image.Image:
    """Trim extraneous margins from an image using layout detection.

    Delegates to ``detect_layout`` and ``crop_image_and_page_to_content``; keeps
    identical parameters and margin (30px) as before.
    """
    (_, _, page_with_original_coordinates) = await detect_layout(
        image,
    )
    trimmed_image, _, _ = crop_image_and_page_to_content(
        image,
        page_with_original_coordinates,
        margin=30,  # pixels
    )
    return trimmed_image

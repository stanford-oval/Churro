"""Dataclasses describing Gemini page detection structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from churro.page import PageObject

from ._constants import NORMALIZED_MAX_COORD, NORMALIZED_MIN_COORD


def _clamp_normalized(value: float) -> int:
    clamped = max(NORMALIZED_MIN_COORD, min(NORMALIZED_MAX_COORD, value))
    rounded = int(round(clamped))
    return max(0, min(1000, rounded))


@dataclass
class PageDetectionTransform:
    """Stores geometry needed to map detection outputs back to the original image."""

    original_size: tuple[int, int]
    border: tuple[int, int]
    padded_size: tuple[int, int]
    processed_size: tuple[int, int]
    scale_x: float
    scale_y: float


@dataclass
class PageBox:
    """Normalized Gemini bounding box."""

    page_index: int
    ymin: int
    xmin: int
    ymax: int
    xmax: int

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> PageBox:
        if "page_index" not in payload:
            raise ValueError("Expected 'page_index' key in Gemini response.")
        required_keys = {"left", "top", "right", "bottom"}
        if not required_keys.issubset(payload):
            missing = required_keys - set(payload)
            raise ValueError(
                f"Gemini response must include keys {sorted(required_keys)},"
                f" missing {sorted(missing)}.",
            )

        ymin = _clamp_normalized(float(payload["top"]))
        xmin = _clamp_normalized(float(payload["left"]))
        ymax = _clamp_normalized(float(payload["bottom"]))
        xmax = _clamp_normalized(float(payload["right"]))
        return cls(
            page_index=int(payload["page_index"]),
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
        )

    def denormalize(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert 0-1000 normalized box to pixel coordinates."""
        top = max(0, min(height, int(round(self.ymin * height / 1000))))
        left = max(0, min(width, int(round(self.xmin * width / 1000))))
        bottom = max(0, min(height, int(round(self.ymax * height / 1000))))
        right = max(0, min(width, int(round(self.xmax * width / 1000))))
        return left, top, right, bottom

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "left": self.xmin,
            "top": self.ymin,
            "right": self.xmax,
            "bottom": self.ymax,
        }


@dataclass
class PageDetectionResult:
    """Outputs produced by the Gemini page detection pipeline."""

    crops: list[Image.Image]
    boxes: list[PageBox]
    polygons_aligned: list[PageObject]
    polygons_original: list[PageObject]
    annotated_image: Image.Image
    transform: PageDetectionTransform


__all__ = [
    "PageDetectionTransform",
    "PageBox",
    "PageDetectionResult",
]

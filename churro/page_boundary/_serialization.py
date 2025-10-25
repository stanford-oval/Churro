"""Serialization helpers for Gemini page boundary payloads."""

from __future__ import annotations

from collections.abc import Iterable
import json

from ._models import PageBox


def strip_code_fence(raw: str) -> str:
    """Remove a leading/trailing triple-backtick fence if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_pages_json(raw: str) -> list[PageBox]:
    cleaned = strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode Gemini response as JSON: {exc}") from exc

    pages_data = parsed.get("pages", [])
    if not isinstance(pages_data, list):
        raise ValueError("Gemini response JSON must include a 'pages' list.")
    page_boxes = [PageBox.from_json(item) for item in pages_data]
    return sorted(page_boxes, key=lambda box: box.page_index)


def boxes_to_json_payload(boxes: Iterable[PageBox]) -> str:
    payload = {"pages": [box.to_json_dict() for box in boxes]}
    return json.dumps(payload, ensure_ascii=False)


def boxes_equal(a: Iterable[PageBox], b: Iterable[PageBox]) -> bool:
    list_a = list(a)
    list_b = list(b)
    if len(list_a) != len(list_b):
        return False
    for box_a, box_b in zip(list_a, list_b, strict=False):
        if box_a.page_index != box_b.page_index:
            return False
        coords_a = (box_a.ymin, box_a.xmin, box_a.ymax, box_a.xmax)
        coords_b = (box_b.ymin, box_b.xmin, box_b.ymax, box_b.xmax)
        if coords_a != coords_b:
            return False
    return True


__all__ = [
    "parse_pages_json",
    "boxes_to_json_payload",
    "boxes_equal",
]

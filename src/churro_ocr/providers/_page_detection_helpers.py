"""Shared helpers for page-detection providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from PIL import Image, ImageDraw, ImageOps

from churro_ocr.errors import ConfigurationError, ProviderError
from churro_ocr.page_detection import PageCandidate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from churro_ocr.types import BoundingBox, Polygon

_BORDER_FRACTION = 0.05
_PROCESSED_MAX_DIM = 2500
_PAGE_DETECTION_BOX_WIDTH = 10
_TEXT_BLOCK_DETECTION_BOX_WIDTH = 6
_REVIEW_CROP_MARGIN_FRACTION = 0.12
_TEXT_BLOCK_REVIEW_CROP_MARGIN_FRACTION = 0.22
_REVIEW_EDGE_STOP_DEADBAND = 6
_REVIEW_EDGE_STOP_STABLE_ROUNDS = 2
_REVIEW_EDGE_STOP_OSCILLATION_MAGNITUDE_RATIO_MIN = 0.5
_REVIEW_EDGE_STOP_OSCILLATION_MAGNITUDE_RATIO_MAX = 2.0
_GUIDELINE_COLOR = "#ff3b30"
_SCALE_WITH_BORDER = 1 + (2 * _BORDER_FRACTION)
_NORMALIZED_MIN_COORD = (_BORDER_FRACTION / _SCALE_WITH_BORDER) * 1000
_NORMALIZED_MAX_COORD = ((1 + _BORDER_FRACTION) / _SCALE_WITH_BORDER) * 1000
_EDGE_NAMES = ("left", "top", "right", "bottom")


class _AzurePageLike(Protocol):
    polygon: object
    width: object
    height: object
    page_number: object
    unit: object
    angle: object


class _AzureAnalyzeResultLike(Protocol):
    pages: Sequence[_AzurePageLike] | None


def _configuration_error(message: str) -> ConfigurationError:
    return ConfigurationError(message)


def _provider_error(message: str) -> ProviderError:
    return ProviderError(message)


def _type_error(message: str) -> TypeError:
    return TypeError(message)


def _value_error(message: str) -> ValueError:
    return ValueError(message)


def _full_image_candidate(image: Image.Image) -> PageCandidate:
    return PageCandidate(bbox=(0.0, 0.0, float(image.width), float(image.height)))


def _bbox_from_polygon(
    polygon: Polygon,
) -> BoundingBox:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _normalize_polygon(
    coordinates: Sequence[float] | None,
) -> Polygon:
    if not coordinates or len(coordinates) < 6:
        return ()
    pairs = [
        (float(coordinates[index]), float(coordinates[index + 1]))
        for index in range(0, len(coordinates) - 1, 2)
    ]
    if len(pairs) > 1 and pairs[0] == pairs[-1]:
        pairs.pop()
    return tuple(pairs)


def _clamp_normalized(value: float) -> int:
    clamped = max(_NORMALIZED_MIN_COORD, min(_NORMALIZED_MAX_COORD, value))
    rounded = round(clamped)
    return max(0, min(1000, rounded))


@dataclass(slots=True)
class _PageDetectionTransform:
    original_size: tuple[int, int]
    border: tuple[int, int]
    padded_size: tuple[int, int]
    processed_size: tuple[int, int]
    scale_x: float
    scale_y: float

    def map_box_to_original(self, box: _PageBox) -> BoundingBox:
        processed_width, processed_height = self.processed_size
        original_width, original_height = self.original_size
        border_width, border_height = self.border

        left_processed, top_processed, right_processed, bottom_processed = box.denormalize(
            processed_width,
            processed_height,
        )
        left_padded = left_processed / (self.scale_x or 1.0)
        top_padded = top_processed / (self.scale_y or 1.0)
        right_padded = right_processed / (self.scale_x or 1.0)
        bottom_padded = bottom_processed / (self.scale_y or 1.0)

        left_original = max(0.0, min(original_width, left_padded - border_width))
        top_original = max(0.0, min(original_height, top_padded - border_height))
        right_original = max(0.0, min(original_width, right_padded - border_width))
        bottom_original = max(0.0, min(original_height, bottom_padded - border_height))
        return left_original, top_original, right_original, bottom_original


@dataclass(slots=True)
class _PageBox:
    page_index: int
    ymin: int
    xmin: int
    ymax: int
    xmax: int

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> _PageBox:
        if "page_index" not in payload:
            message = "Expected 'page_index' key in page-detection response."
            raise _value_error(message)
        required_keys = {"left", "top", "right", "bottom"}
        if not required_keys.issubset(payload):
            missing = required_keys - set(payload)
            message = (
                f"Page-detection response must include keys {sorted(required_keys)}, "
                f"missing {sorted(missing)}."
            )
            raise _value_error(message)
        return cls(
            page_index=int(payload["page_index"]),
            ymin=_clamp_normalized(float(payload["top"])),
            xmin=_clamp_normalized(float(payload["left"])),
            ymax=_clamp_normalized(float(payload["bottom"])),
            xmax=_clamp_normalized(float(payload["right"])),
        )

    def denormalize(self, width: int, height: int) -> tuple[int, int, int, int]:
        top = max(0, min(height, round(self.ymin * height / 1000)))
        left = max(0, min(width, round(self.xmin * width / 1000)))
        bottom = max(0, min(height, round(self.ymax * height / 1000)))
        right = max(0, min(width, round(self.xmax * width / 1000)))
        return left, top, right, bottom


EdgeDecisionAction = Literal["expand", "shrink", "no_change"]


@dataclass(slots=True, frozen=True)
class _EdgeReviewDecision:
    action: EdgeDecisionAction
    amount: int


@dataclass(slots=True, frozen=True)
class _BoxReviewDecision:
    page_index: int
    left: _EdgeReviewDecision
    top: _EdgeReviewDecision
    right: _EdgeReviewDecision
    bottom: _EdgeReviewDecision


def _add_white_border(
    image: Image.Image,
    *,
    fraction: float = _BORDER_FRACTION,
) -> tuple[Image.Image, int, int]:
    if fraction <= 0:
        return image, 0, 0
    border_width = max(1, round(image.width * fraction))
    border_height = max(1, round(image.height * fraction))
    expanded = ImageOps.expand(
        image,
        border=(border_width, border_height, border_width, border_height),
        fill="white",
    )
    return expanded, border_width, border_height


def _resize_image_to_fit(image: Image.Image, *, max_dim: int = _PROCESSED_MAX_DIM) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_dim:
        return image
    scale = max_dim / longest_side
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))))


def _prepare_detection_image(image: Image.Image) -> tuple[Image.Image, _PageDetectionTransform]:
    rgb_image = image.convert("RGB")
    bordered, border_width, border_height = _add_white_border(rgb_image)
    processed = _resize_image_to_fit(bordered)
    transform = _PageDetectionTransform(
        original_size=image.size,
        border=(border_width, border_height),
        padded_size=bordered.size,
        processed_size=processed.size,
        scale_x=processed.width / bordered.width if bordered.width else 1.0,
        scale_y=processed.height / bordered.height if bordered.height else 1.0,
    )
    return processed, transform


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_page_boxes_json(output: str) -> list[_PageBox]:
    response_text = _strip_code_fence(output)
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        message = "LLM page detection returned invalid JSON."
        raise _provider_error(message) from exc

    if not isinstance(payload, dict):
        message = "LLM page detection response must be a JSON object."
        raise _provider_error(message)

    pages = payload.get("pages")
    if not isinstance(pages, list):
        message = "LLM page detection response must include a `pages` list."
        raise _provider_error(message)

    boxes: list[_PageBox] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            message = f"LLM page detection entry {page_index} must be an object."
            raise _provider_error(message)
        try:
            boxes.append(_PageBox.from_json(cast("dict[str, Any]", page)))
        except (TypeError, ValueError) as exc:
            message = f"LLM page detection entry {page_index} is invalid: {exc}"
            raise _provider_error(message) from exc
    return sorted(boxes, key=lambda box: box.page_index)


def _build_target_box_from_payload(payload: dict[str, Any], *, target_index: int) -> _PageBox:
    return _PageBox.from_json(
        {
            "page_index": target_index,
            "left": payload["left"],
            "top": payload["top"],
            "right": payload["right"],
            "bottom": payload["bottom"],
        }
    )


def _parse_target_box_json(
    output: str,
    *,
    target_key: str,
    found_key: str,
    error_context: str,
) -> _PageBox | None:
    response_text = _strip_code_fence(output)
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        message = f"{error_context} returned invalid JSON."
        raise _provider_error(message) from exc

    if not isinstance(payload, dict):
        message = f"{error_context} response must be a JSON object."
        raise _provider_error(message)
    payload_dict = cast("dict[str, Any]", payload)

    if {"left", "top", "right", "bottom"}.issubset(payload_dict):
        try:
            return _build_target_box_from_payload(payload_dict, target_index=1)
        except (TypeError, ValueError) as exc:
            message = f"{error_context} bbox is invalid: {exc}"
            raise _provider_error(message) from exc

    raw_target = payload_dict.get(target_key)
    if raw_target is None:
        raw_target = payload_dict.get("bbox")
    if isinstance(raw_target, dict):
        try:
            return _build_target_box_from_payload(cast("dict[str, Any]", raw_target), target_index=1)
        except (TypeError, ValueError) as exc:
            message = f"{error_context} bbox is invalid: {exc}"
            raise _provider_error(message) from exc
    if raw_target is not None:
        message = f"{error_context} response `{target_key}` must be an object or null."
        raise _provider_error(message)

    if (
        payload_dict.get(found_key) is False
        or payload_dict.get("found") is False
        or (target_key in payload_dict and payload_dict[target_key] is None)
        or ("bbox" in payload_dict and payload_dict["bbox"] is None)
    ):
        return None

    message = (
        f"{error_context} response must include a `{target_key}` object "
        f"or explicitly mark `{found_key}` false."
    )
    raise _provider_error(message)


def _parse_text_block_box_json(output: str) -> _PageBox | None:
    return _parse_target_box_json(
        output,
        target_key="block",
        found_key="block_found",
        error_context="LLM text-block localization",
    )


def _parse_edge_review_decision(
    payload: object,
    *,
    edge_name: str,
) -> _EdgeReviewDecision:
    if not isinstance(payload, dict):
        message = f"Review edge '{edge_name}' must be an object."
        raise _type_error(message)
    payload_dict = cast("dict[str, object]", payload)

    raw_action = payload_dict.get("action")
    if raw_action is None:
        raw_action = payload_dict.get("decision")
    if not isinstance(raw_action, str):
        message = f"Review edge '{edge_name}' must include string 'action'."
        raise _type_error(message)
    action = raw_action.strip().lower()
    if action not in {"expand", "shrink", "no_change"}:
        message = f"Review edge '{edge_name}' action must be one of 'expand', 'shrink', 'no_change'."
        raise _value_error(message)
    action_literal = cast("EdgeDecisionAction", action)

    try:
        raw_amount = payload_dict.get("amount")
        amount = 0 if raw_amount is None else round(float(cast("Any", raw_amount)))
    except (TypeError, ValueError) as exc:
        message = f"Review edge '{edge_name}' amount must be numeric."
        raise _value_error(message) from exc
    amount = max(0, min(1000, amount))
    if action_literal == "no_change":
        amount = 0
    return _EdgeReviewDecision(action=action_literal, amount=amount)


def _parse_single_edge_review_decision_json(
    output: str,
) -> tuple[int, str, _EdgeReviewDecision]:
    try:
        payload = json.loads(_strip_code_fence(output))
    except json.JSONDecodeError as exc:
        message = f"Failed to decode edge-review response as JSON: {exc}"
        raise _value_error(message) from exc

    if not isinstance(payload, dict):
        message = "Edge-review response must be a JSON object."
        raise _type_error(message)
    if "page_index" not in payload:
        message = "Edge-review response must include 'page_index'."
        raise _value_error(message)

    raw_edge = payload.get("edge")
    if not isinstance(raw_edge, str):
        message = "Edge-review response must include string 'edge'."
        raise _type_error(message)
    edge_name = raw_edge.strip().lower()
    if edge_name not in _EDGE_NAMES:
        message = "Edge-review response 'edge' must be left/top/right/bottom."
        raise _value_error(message)

    decision_payload = payload.get("decision")
    if not isinstance(decision_payload, dict):
        decision_payload = {
            "action": payload.get("action"),
            "amount": payload.get("amount", 0),
        }

    return (
        int(payload["page_index"]),
        edge_name,
        _parse_edge_review_decision(
            decision_payload,
            edge_name=edge_name,
        ),
    )


def _parse_text_block_edge_review_decision_json(
    output: str,
) -> tuple[str, _EdgeReviewDecision]:
    try:
        payload = json.loads(_strip_code_fence(output))
    except json.JSONDecodeError as exc:
        message = f"Failed to decode text-block edge-review response as JSON: {exc}"
        raise _value_error(message) from exc

    if not isinstance(payload, dict):
        message = "Text-block edge-review response must be a JSON object."
        raise _type_error(message)
    payload_dict = cast("dict[str, object]", payload)

    raw_edge = payload_dict.get("edge")
    if not isinstance(raw_edge, str):
        message = "Text-block edge-review response must include string 'edge'."
        raise _type_error(message)
    edge_name = raw_edge.strip().lower()
    if edge_name not in _EDGE_NAMES:
        message = "Text-block edge-review response 'edge' must be left/top/right/bottom."
        raise _value_error(message)

    decision_payload = payload_dict.get("decision")
    if not isinstance(decision_payload, dict):
        decision_payload = {
            "action": payload_dict.get("action"),
            "amount": payload_dict.get("amount", 0),
        }
    return edge_name, _parse_edge_review_decision(decision_payload, edge_name=edge_name)


def _boxes_equal(left_boxes: Sequence[_PageBox], right_boxes: Sequence[_PageBox]) -> bool:
    if len(left_boxes) != len(right_boxes):
        return False
    for left_box, right_box in zip(left_boxes, right_boxes, strict=False):
        if (
            left_box.page_index != right_box.page_index
            or left_box.xmin != right_box.xmin
            or left_box.ymin != right_box.ymin
            or left_box.xmax != right_box.xmax
            or left_box.ymax != right_box.ymax
        ):
            return False
    return True


def _bbox_to_polygon(
    bbox: BoundingBox,
) -> Polygon:
    left, top, right, bottom = bbox
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def _normalize_pixel_coord(value: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(1000, round(value * 1000 / size)))


def _build_box_review_preview(
    image: Image.Image,
    box: _PageBox,
    *,
    margin_fraction: float = _REVIEW_CROP_MARGIN_FRACTION,
    outline_width: int = _PAGE_DETECTION_BOX_WIDTH,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    left, top, right, bottom = box.denormalize(width, height)

    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    margin_x = max(outline_width * 2, round(box_width * margin_fraction))
    margin_y = max(outline_width * 2, round(box_height * margin_fraction))

    crop_left = max(0, left - margin_x)
    crop_top = max(0, top - margin_y)
    crop_right = min(width, right + margin_x)
    crop_bottom = min(height, bottom + margin_y)

    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    preview = crop.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle(
        [left - crop_left, top - crop_top, right - crop_left, bottom - crop_top],
        outline=_GUIDELINE_COLOR,
        width=outline_width,
    )
    return preview, (crop_left, crop_top, crop_right, crop_bottom)


def _build_edge_strip_review_preview(
    image: Image.Image,
    box: _PageBox,
    edge_name: str,
    *,
    outline_width: int = _PAGE_DETECTION_BOX_WIDTH,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    left, top, right, bottom = box.denormalize(width, height)
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)

    band_half_x = max(outline_width * 3, round(box_width * 0.18))
    band_half_y = max(outline_width * 3, round(box_height * 0.18))
    orthogonal_pad_x = max(outline_width * 2, round(box_width * 0.06))
    orthogonal_pad_y = max(outline_width * 2, round(box_height * 0.06))

    if edge_name == "left":
        x0 = max(0, left - band_half_x)
        x1 = min(width, left + band_half_x)
        y0 = max(0, top - orthogonal_pad_y)
        y1 = min(height, bottom + orthogonal_pad_y)
    elif edge_name == "right":
        x0 = max(0, right - band_half_x)
        x1 = min(width, right + band_half_x)
        y0 = max(0, top - orthogonal_pad_y)
        y1 = min(height, bottom + orthogonal_pad_y)
    elif edge_name == "top":
        x0 = max(0, left - orthogonal_pad_x)
        x1 = min(width, right + orthogonal_pad_x)
        y0 = max(0, top - band_half_y)
        y1 = min(height, top + band_half_y)
    elif edge_name == "bottom":
        x0 = max(0, left - orthogonal_pad_x)
        x1 = min(width, right + orthogonal_pad_x)
        y0 = max(0, bottom - band_half_y)
        y1 = min(height, bottom + band_half_y)
    else:
        message = f"Unsupported edge '{edge_name}'. Expected left/top/right/bottom."
        raise _value_error(message)

    if x0 >= x1 or y0 >= y1:
        message = f"Invalid strip bounds for edge '{edge_name}'."
        raise _value_error(message)
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def _convert_source_box_to_review_crop_box(
    box: _PageBox,
    crop_bounds: tuple[int, int, int, int],
    source_size: tuple[int, int],
) -> _PageBox:
    source_width, source_height = source_size
    crop_left, crop_top, crop_right, crop_bottom = crop_bounds
    crop_width = max(1, crop_right - crop_left)
    crop_height = max(1, crop_bottom - crop_top)
    left, top, right, bottom = box.denormalize(source_width, source_height)
    return _PageBox.from_json(
        {
            "page_index": box.page_index,
            "left": _normalize_pixel_coord(max(0, min(crop_width, left - crop_left)), crop_width),
            "top": _normalize_pixel_coord(max(0, min(crop_height, top - crop_top)), crop_height),
            "right": _normalize_pixel_coord(
                max(0, min(crop_width, right - crop_left)),
                crop_width,
            ),
            "bottom": _normalize_pixel_coord(
                max(0, min(crop_height, bottom - crop_top)),
                crop_height,
            ),
        }
    )


def _map_review_crop_box_to_source_box(
    reviewed_box: _PageBox,
    crop_bounds: tuple[int, int, int, int],
    source_size: tuple[int, int],
    *,
    page_index: int,
) -> _PageBox:
    source_width, source_height = source_size
    crop_left, crop_top, crop_right, crop_bottom = crop_bounds
    crop_width = max(1, crop_right - crop_left)
    crop_height = max(1, crop_bottom - crop_top)
    local_left, local_top, local_right, local_bottom = reviewed_box.denormalize(
        crop_width,
        crop_height,
    )
    return _PageBox.from_json(
        {
            "page_index": page_index,
            "left": _normalize_pixel_coord(
                max(0, min(source_width, crop_left + local_left)),
                source_width,
            ),
            "top": _normalize_pixel_coord(
                max(0, min(source_height, crop_top + local_top)),
                source_height,
            ),
            "right": _normalize_pixel_coord(
                max(0, min(source_width, crop_left + local_right)),
                source_width,
            ),
            "bottom": _normalize_pixel_coord(
                max(0, min(source_height, crop_top + local_bottom)),
                source_height,
            ),
        }
    )


def _merge_instruction_prompts(*parts: str | None) -> str:
    """Merge one or more instruction strings into a single non-empty user prompt."""
    merged_parts = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    if not merged_parts:
        message = "Expected at least one non-empty instruction prompt."
        raise _value_error(message)
    return "\n\n".join(merged_parts)


def _normalize_azure_page_polygon(
    page: _AzurePageLike, *, image: Image.Image
) -> Polygon:
    raw_polygon = getattr(page, "polygon", None)
    polygon = _normalize_polygon(raw_polygon)
    if not polygon:
        return ()

    page_width = float(getattr(page, "width", 0.0) or image.width)
    page_height = float(getattr(page, "height", 0.0) or image.height)
    scale_x = image.width / page_width if page_width else 1.0
    scale_y = image.height / page_height if page_height else 1.0
    return tuple((x * scale_x, y * scale_y) for x, y in polygon)

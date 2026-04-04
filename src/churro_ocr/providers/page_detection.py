"""Built-in page detection backends."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, cast

from PIL import Image, ImageDraw, ImageOps

from churro_ocr._internal.litellm import LiteLLMTransport
from churro_ocr._internal.logging import logger
from churro_ocr._internal.runtime import run_sync
from churro_ocr.errors import ConfigurationError, ProviderError
from churro_ocr.page_detection import PageCandidate, PageDetectionBackend
from churro_ocr.prompts import DEFAULT_BOUNDARY_DETECTION_PROMPT
from churro_ocr.prompts.layout import (
    build_boundary_review_prompt,
    build_text_block_boundary_review_prompt,
    build_text_block_localization_prompt,
)
from churro_ocr.providers.specs import LiteLLMTransportConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

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
LiteLLMTransportLike = LiteLLMTransportConfig | LiteLLMTransport | None


def _full_image_candidate(image: Image.Image) -> PageCandidate:
    return PageCandidate(bbox=(0.0, 0.0, float(image.width), float(image.height)))


def _bbox_from_polygon(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _normalize_polygon(
    coordinates: Sequence[float] | None,
) -> tuple[tuple[float, float], ...]:
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
    rounded = int(round(clamped))
    return max(0, min(1000, rounded))


@dataclass(slots=True)
class _PageDetectionTransform:
    original_size: tuple[int, int]
    border: tuple[int, int]
    padded_size: tuple[int, int]
    processed_size: tuple[int, int]
    scale_x: float
    scale_y: float

    def map_box_to_original(self, box: _PageBox) -> tuple[float, float, float, float]:
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
            raise ValueError("Expected 'page_index' key in page-detection response.")
        required_keys = {"left", "top", "right", "bottom"}
        if not required_keys.issubset(payload):
            missing = required_keys - set(payload)
            raise ValueError(
                f"Page-detection response must include keys {sorted(required_keys)}, "
                f"missing {sorted(missing)}."
            )
        return cls(
            page_index=int(payload["page_index"]),
            ymin=_clamp_normalized(float(payload["top"])),
            xmin=_clamp_normalized(float(payload["left"])),
            ymax=_clamp_normalized(float(payload["bottom"])),
            xmax=_clamp_normalized(float(payload["right"])),
        )

    def denormalize(self, width: int, height: int) -> tuple[int, int, int, int]:
        top = max(0, min(height, int(round(self.ymin * height / 1000))))
        left = max(0, min(width, int(round(self.xmin * width / 1000))))
        bottom = max(0, min(height, int(round(self.ymax * height / 1000))))
        right = max(0, min(width, int(round(self.xmax * width / 1000))))
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
    border_width = max(1, int(round(image.width * fraction)))
    border_height = max(1, int(round(image.height * fraction)))
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
    return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))))


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
        raise ProviderError("LLM page detection returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ProviderError("LLM page detection response must be a JSON object.")

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ProviderError("LLM page detection response must include a `pages` list.")

    boxes: list[_PageBox] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ProviderError(f"LLM page detection entry {page_index} must be an object.")
        try:
            boxes.append(_PageBox.from_json(cast("dict[str, Any]", page)))
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"LLM page detection entry {page_index} is invalid: {exc}") from exc
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
        raise ProviderError(f"{error_context} returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ProviderError(f"{error_context} response must be a JSON object.")
    payload_dict = cast("dict[str, Any]", payload)

    if {"left", "top", "right", "bottom"}.issubset(payload_dict):
        try:
            return _build_target_box_from_payload(payload_dict, target_index=1)
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"{error_context} bbox is invalid: {exc}") from exc

    raw_target = payload_dict.get(target_key)
    if raw_target is None:
        raw_target = payload_dict.get("bbox")
    if isinstance(raw_target, dict):
        try:
            return _build_target_box_from_payload(cast("dict[str, Any]", raw_target), target_index=1)
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"{error_context} bbox is invalid: {exc}") from exc
    if raw_target is not None:
        raise ProviderError(f"{error_context} response `{target_key}` must be an object or null.")

    if (
        payload_dict.get(found_key) is False
        or payload_dict.get("found") is False
        or (target_key in payload_dict and payload_dict[target_key] is None)
        or ("bbox" in payload_dict and payload_dict["bbox"] is None)
    ):
        return None

    raise ProviderError(
        f"{error_context} response must include a `{target_key}` object "
        f"or explicitly mark `{found_key}` false."
    )


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
        raise ValueError(f"Review edge '{edge_name}' must be an object.")
    payload_dict = cast("dict[str, object]", payload)

    raw_action = payload_dict.get("action")
    if raw_action is None:
        raw_action = payload_dict.get("decision")
    if not isinstance(raw_action, str):
        raise ValueError(f"Review edge '{edge_name}' must include string 'action'.")
    action = raw_action.strip().lower()
    if action not in {"expand", "shrink", "no_change"}:
        raise ValueError(f"Review edge '{edge_name}' action must be one of 'expand', 'shrink', 'no_change'.")
    action_literal = cast("EdgeDecisionAction", action)

    try:
        raw_amount = payload_dict.get("amount")
        amount = 0 if raw_amount is None else int(round(float(cast("Any", raw_amount))))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Review edge '{edge_name}' amount must be numeric.") from exc
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
        raise ValueError(f"Failed to decode edge-review response as JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Edge-review response must be a JSON object.")
    if "page_index" not in payload:
        raise ValueError("Edge-review response must include 'page_index'.")

    raw_edge = payload.get("edge")
    if not isinstance(raw_edge, str):
        raise ValueError("Edge-review response must include string 'edge'.")
    edge_name = raw_edge.strip().lower()
    if edge_name not in _EDGE_NAMES:
        raise ValueError("Edge-review response 'edge' must be left/top/right/bottom.")

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
        raise ValueError(f"Failed to decode text-block edge-review response as JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Text-block edge-review response must be a JSON object.")
    payload_dict = cast("dict[str, object]", payload)

    raw_edge = payload_dict.get("edge")
    if not isinstance(raw_edge, str):
        raise ValueError("Text-block edge-review response must include string 'edge'.")
    edge_name = raw_edge.strip().lower()
    if edge_name not in _EDGE_NAMES:
        raise ValueError("Text-block edge-review response 'edge' must be left/top/right/bottom.")

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
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float], ...]:
    left, top, right, bottom = bbox
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def _normalize_pixel_coord(value: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(1000, int(round(value * 1000 / size))))


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
    margin_x = max(outline_width * 2, int(round(box_width * margin_fraction)))
    margin_y = max(outline_width * 2, int(round(box_height * margin_fraction)))

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

    band_half_x = max(outline_width * 3, int(round(box_width * 0.18)))
    band_half_y = max(outline_width * 3, int(round(box_height * 0.18)))
    orthogonal_pad_x = max(outline_width * 2, int(round(box_width * 0.06)))
    orthogonal_pad_y = max(outline_width * 2, int(round(box_height * 0.06)))

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
        raise ValueError(f"Unsupported edge '{edge_name}'. Expected left/top/right/bottom.")

    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Invalid strip bounds for edge '{edge_name}'.")
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
        raise ValueError("Expected at least one non-empty instruction prompt.")
    return "\n\n".join(merged_parts)


async def _complete_page_boxes(
    *,
    model: str,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str | None,
    transport: LiteLLMTransport,
) -> list[_PageBox]:
    messages = transport.prepare_messages(
        system_prompt=None,
        user_prompt=_merge_instruction_prompts(system_prompt, user_prompt),
        images=[image],
    )
    output = await transport.complete_text(
        model=model,
        messages=messages,
        output_json=True,
    )
    logger.info("Initial LLM page-detection response: %s", output)
    return _parse_page_boxes_json(output)


async def _complete_text_block_box(
    *,
    model: str,
    image: Image.Image,
    block_tag: str,
    block_text: str,
    transport: LiteLLMTransport,
) -> _PageBox | None:
    messages = transport.prepare_messages(
        system_prompt=None,
        user_prompt=build_text_block_localization_prompt(
            block_tag=block_tag,
            block_text=block_text,
        ),
        images=[image],
    )
    output = await transport.complete_text(
        model=model,
        messages=messages,
        output_json=True,
    )
    logger.info("Initial LLM text-block-localization response: %s", output)
    return _parse_text_block_box_json(output)


async def _review_single_edge_from_strip(
    *,
    model: str,
    review_image: Image.Image,
    strip_image: Image.Image,
    strip_bounds: tuple[int, int, int, int],
    edge_name: str,
    page_index: int,
    history_steps: int,
    round_index: int,
    transport: LiteLLMTransport,
) -> _EdgeReviewDecision:
    strip_axis_pixels = _strip_axis_size_pixels(strip_bounds, edge_name=edge_name)
    if strip_axis_pixels <= 0:
        raise ValueError(f"Invalid strip axis size for edge '{edge_name}'.")

    prompt = build_boundary_review_prompt(
        edge_name=edge_name,
        page_index=page_index,
        strip_axis=(
            "horizontal (x-axis / strip width)"
            if edge_name in {"left", "right"}
            else "vertical (y-axis / strip height)"
        ),
    )
    messages = transport.prepare_messages(
        system_prompt=None,
        user_prompt=prompt,
        images=[strip_image],
    )
    output = await transport.complete_text(
        model=model,
        messages=messages,
        output_json=True,
    )
    logger.info(
        "Review LLM single-edge response (round=%s, page=%s, edge=%s, history rounds=%s): %s",
        round_index,
        page_index,
        edge_name,
        history_steps,
        output,
    )
    response_page_index, response_edge_name, strip_decision = _parse_single_edge_review_decision_json(output)
    if response_page_index != page_index:
        logger.info(
            "Single-edge review page_index mismatch (expected=%s, got=%s) for edge=%s; using expected.",
            page_index,
            response_page_index,
            edge_name,
        )
    if response_edge_name != edge_name:
        logger.info(
            "Single-edge review edge mismatch (expected=%s, got=%s); using expected edge.",
            edge_name,
            response_edge_name,
        )

    local_axis_pixels = review_image.width if edge_name in {"left", "right"} else review_image.height
    local_amount = _convert_strip_delta_to_local_delta(
        strip_decision.amount,
        strip_axis_pixels=strip_axis_pixels,
        local_axis_pixels=local_axis_pixels,
    )
    return _EdgeReviewDecision(action=strip_decision.action, amount=local_amount)


async def _review_single_text_block_edge_from_strip(
    *,
    model: str,
    review_image: Image.Image,
    strip_image: Image.Image,
    strip_bounds: tuple[int, int, int, int],
    edge_name: str,
    block_tag: str,
    block_text: str,
    history_steps: int,
    round_index: int,
    transport: LiteLLMTransport,
) -> _EdgeReviewDecision:
    strip_axis_pixels = _strip_axis_size_pixels(strip_bounds, edge_name=edge_name)
    if strip_axis_pixels <= 0:
        raise ValueError(f"Invalid strip axis size for edge '{edge_name}'.")

    prompt = build_text_block_boundary_review_prompt(
        edge_name=edge_name,
        block_tag=block_tag,
        block_text=block_text,
        strip_axis=(
            "horizontal (x-axis / strip width)"
            if edge_name in {"left", "right"}
            else "vertical (y-axis / strip height)"
        ),
    )
    messages = transport.prepare_messages(
        system_prompt=None,
        user_prompt=prompt,
        images=[strip_image],
    )
    output = await transport.complete_text(
        model=model,
        messages=messages,
        output_json=True,
    )
    logger.info(
        "Text-block review LLM single-edge response (round=%s, edge=%s, history rounds=%s): %s",
        round_index,
        edge_name,
        history_steps,
        output,
    )
    response_edge_name, strip_decision = _parse_text_block_edge_review_decision_json(output)
    if response_edge_name != edge_name:
        logger.info(
            "Text-block edge-review mismatch (expected=%s, got=%s); using expected edge.",
            edge_name,
            response_edge_name,
        )

    local_axis_pixels = review_image.width if edge_name in {"left", "right"} else review_image.height
    local_amount = _convert_strip_delta_to_local_delta(
        strip_decision.amount,
        strip_axis_pixels=strip_axis_pixels,
        local_axis_pixels=local_axis_pixels,
    )
    return _EdgeReviewDecision(action=strip_decision.action, amount=local_amount)


async def _review_page_box(
    *,
    image: Image.Image,
    current_box: _PageBox,
    history_steps: int,
    round_index: int,
    model: str,
    transport: LiteLLMTransport,
) -> _PageBox:
    review_image, crop_bounds = _build_box_review_preview(image, current_box)
    local_box = _convert_source_box_to_review_crop_box(current_box, crop_bounds, image.size)
    edge_strip_inputs = [
        (edge_name, *_build_edge_strip_review_preview(review_image, local_box, edge_name))
        for edge_name in _EDGE_NAMES
    ]
    edge_results = await asyncio.gather(
        *[
            _review_single_edge_from_strip(
                model=model,
                review_image=review_image,
                strip_image=strip_image,
                strip_bounds=strip_bounds,
                edge_name=edge_name,
                page_index=current_box.page_index,
                history_steps=history_steps,
                round_index=round_index,
                transport=transport,
            )
            for edge_name, strip_image, strip_bounds in edge_strip_inputs
        ],
        return_exceptions=True,
    )

    edge_decisions: dict[str, _EdgeReviewDecision] = {}
    for edge_name, result in zip(_EDGE_NAMES, edge_results, strict=False):
        if isinstance(result, Exception):
            logger.info(
                "Edge-strip review failed for round %s, page %s, edge %s; using no_change: %s",
                round_index,
                current_box.page_index,
                edge_name,
                result,
            )
            edge_decisions[edge_name] = _no_change_edge_review_decision()
            continue
        edge_decisions[edge_name] = result

    reviewed_local_box = _apply_box_review_decision(
        local_box,
        _BoxReviewDecision(
            page_index=current_box.page_index,
            left=edge_decisions["left"],
            top=edge_decisions["top"],
            right=edge_decisions["right"],
            bottom=edge_decisions["bottom"],
        ),
        expected_page_index=current_box.page_index,
    )
    return _map_review_crop_box_to_source_box(
        reviewed_local_box,
        crop_bounds,
        image.size,
        page_index=current_box.page_index,
    )


async def _review_text_block_box(
    *,
    image: Image.Image,
    current_box: _PageBox,
    block_tag: str,
    block_text: str,
    history_steps: int,
    round_index: int,
    model: str,
    transport: LiteLLMTransport,
) -> _PageBox:
    review_image, crop_bounds = _build_box_review_preview(
        image,
        current_box,
        margin_fraction=_TEXT_BLOCK_REVIEW_CROP_MARGIN_FRACTION,
        outline_width=_TEXT_BLOCK_DETECTION_BOX_WIDTH,
    )
    local_box = _convert_source_box_to_review_crop_box(current_box, crop_bounds, image.size)
    edge_strip_inputs = [
        (
            edge_name,
            *_build_edge_strip_review_preview(
                review_image,
                local_box,
                edge_name,
                outline_width=_TEXT_BLOCK_DETECTION_BOX_WIDTH,
            ),
        )
        for edge_name in _EDGE_NAMES
    ]
    edge_results = await asyncio.gather(
        *[
            _review_single_text_block_edge_from_strip(
                model=model,
                review_image=review_image,
                strip_image=strip_image,
                strip_bounds=strip_bounds,
                edge_name=edge_name,
                block_tag=block_tag,
                block_text=block_text,
                history_steps=history_steps,
                round_index=round_index,
                transport=transport,
            )
            for edge_name, strip_image, strip_bounds in edge_strip_inputs
        ],
        return_exceptions=True,
    )

    edge_decisions: dict[str, _EdgeReviewDecision] = {}
    for edge_name, result in zip(_EDGE_NAMES, edge_results, strict=False):
        if isinstance(result, Exception):
            logger.info(
                "Text-block edge-strip review failed for round %s, edge %s; using no_change: %s",
                round_index,
                edge_name,
                result,
            )
            edge_decisions[edge_name] = _no_change_edge_review_decision()
            continue
        edge_decisions[edge_name] = result

    reviewed_local_box = _apply_box_review_decision(
        local_box,
        _BoxReviewDecision(
            page_index=current_box.page_index,
            left=edge_decisions["left"],
            top=edge_decisions["top"],
            right=edge_decisions["right"],
            bottom=edge_decisions["bottom"],
        ),
        expected_page_index=current_box.page_index,
    )
    return _map_review_crop_box_to_source_box(
        reviewed_local_box,
        crop_bounds,
        image.size,
        page_index=current_box.page_index,
    )


async def _run_review_pipeline(
    *,
    initial_boxes: list[_PageBox],
    max_review_rounds: int,
    review_box: Callable[[_PageBox, int, int], Awaitable[_PageBox]],
    subject_name_singular: str,
    subject_name_plural: str,
) -> list[_PageBox]:
    history_boxes: list[list[_PageBox]] = [initial_boxes]
    page_review_states = {box.page_index: _new_page_review_stop_state() for box in initial_boxes}
    final_boxes = initial_boxes

    for round_index in range(max(0, max_review_rounds)):
        previous_boxes = history_boxes[-1]
        active_boxes = [
            box
            for box in previous_boxes
            if not _page_review_is_fully_frozen(
                page_review_states.setdefault(box.page_index, _new_page_review_stop_state())
            )
        ]
        if not active_boxes:
            logger.info(
                "All %s frozen by review stop condition before round %s; stopping reviews.",
                subject_name_plural,
                round_index + 1,
            )
            final_boxes = previous_boxes
            break

        review_results = await asyncio.gather(
            *(review_box(box, len(history_boxes), round_index + 1) for box in active_boxes),
            return_exceptions=True,
        )
        results_by_page = {
            box.page_index: result for box, result in zip(active_boxes, review_results, strict=False)
        }

        reviewed_boxes: list[_PageBox] = []
        for prior_box in previous_boxes:
            page_state = page_review_states.setdefault(
                prior_box.page_index,
                _new_page_review_stop_state(),
            )
            if _page_review_is_fully_frozen(page_state):
                reviewed_boxes.append(prior_box)
                continue

            result = results_by_page.get(prior_box.page_index)
            if isinstance(result, Exception):
                logger.info(
                    "Review round %s %s %s failed, keeping prior box: %s",
                    round_index + 1,
                    subject_name_singular,
                    prior_box.page_index,
                    result,
                )
                reviewed_boxes.append(prior_box)
                continue
            if result is None:
                reviewed_boxes.append(prior_box)
                continue

            reviewed_boxes.append(
                _apply_page_review_stop_condition(
                    prior_box=prior_box,
                    reviewed_box=result,
                    page_state=page_state,
                    round_index=round_index + 1,
                    subject_name=subject_name_singular,
                )
            )

        reviewed_boxes = sorted(reviewed_boxes, key=lambda item: item.page_index)
        if not reviewed_boxes:
            break
        if _boxes_equal(reviewed_boxes, previous_boxes):
            final_boxes = reviewed_boxes
            break

        history_boxes.append(reviewed_boxes)
        final_boxes = reviewed_boxes

    _log_box_history(history_boxes, subject_name=subject_name_singular.title())
    return final_boxes


@dataclass(slots=True)
class LLMPageDetector(PageDetectionBackend):
    """Detect one or more pages via a multimodal LLM prompt."""

    model: str
    system_prompt: str = DEFAULT_BOUNDARY_DETECTION_PROMPT
    prompt_template: str | None = None
    transport: LiteLLMTransportConfig | None = None
    max_review_rounds: int = 0

    async def detect(self, image: Image.Image) -> list[PageCandidate]:
        processed_image, transform = _prepare_detection_image(image)
        transport = LiteLLMTransport(self.transport)
        boxes = await _complete_page_boxes(
            model=self.model,
            image=processed_image,
            system_prompt=self.system_prompt,
            user_prompt=self.prompt_template,
            transport=transport,
        )
        if not boxes:
            return [_full_image_candidate(image)]
        if self.max_review_rounds > 0:

            async def _review_page_candidate(
                box: _PageBox,
                history_steps: int,
                round_index: int,
            ) -> _PageBox:
                return await _review_page_box(
                    image=processed_image,
                    current_box=box,
                    history_steps=history_steps,
                    round_index=round_index,
                    model=self.model,
                    transport=transport,
                )

            boxes = await _run_review_pipeline(
                initial_boxes=boxes,
                max_review_rounds=self.max_review_rounds,
                review_box=_review_page_candidate,
                subject_name_singular="page",
                subject_name_plural="pages",
            )

        candidates: list[PageCandidate] = []
        for page_index, box in enumerate(boxes):
            original_bbox = transform.map_box_to_original(box)
            candidates.append(
                PageCandidate(
                    bbox=original_bbox,
                    polygon=_bbox_to_polygon(original_bbox),
                    metadata={
                        "page_index": page_index,
                        "detector": "llm",
                        "response_page_index": box.page_index,
                    },
                )
            )
        return candidates or [_full_image_candidate(image)]


async def locate_text_block_bbox_with_llm(
    image: Image.Image,
    block_text: str,
    *,
    block_tag: str,
    model: str,
    transport: LiteLLMTransportLike = None,
    max_review_rounds: int = 0,
) -> tuple[float, float, float, float] | None:
    """Locate the tight bbox of a specific rendered text block via a multimodal LLM."""
    normalized_block_text = block_text.strip()
    if not normalized_block_text:
        raise ValueError("block_text must not be blank.")

    normalized_block_tag = block_tag.strip()
    if not normalized_block_tag:
        raise ValueError("block_tag must not be blank.")

    processed_image, transform = _prepare_detection_image(image)
    llm_transport = transport if isinstance(transport, LiteLLMTransport) else LiteLLMTransport(transport)
    box = await _complete_text_block_box(
        model=model,
        image=processed_image,
        block_tag=normalized_block_tag,
        block_text=normalized_block_text,
        transport=llm_transport,
    )
    if box is None:
        logger.info(
            "LLM text-block localization did not find a match for tag=%s.",
            normalized_block_tag,
        )
        return None

    if max_review_rounds > 0:

        async def _review_text_block_candidate(
            review_box: _PageBox,
            history_steps: int,
            round_index: int,
        ) -> _PageBox:
            return await _review_text_block_box(
                image=processed_image,
                current_box=review_box,
                block_tag=normalized_block_tag,
                block_text=normalized_block_text,
                history_steps=history_steps,
                round_index=round_index,
                model=model,
                transport=llm_transport,
            )

        reviewed_boxes = await _run_review_pipeline(
            initial_boxes=[box],
            max_review_rounds=max_review_rounds,
            review_box=_review_text_block_candidate,
            subject_name_singular="text block",
            subject_name_plural="text blocks",
        )
        if not reviewed_boxes:
            return None
        box = reviewed_boxes[0]

    return transform.map_box_to_original(box)


def locate_text_block_bbox_with_llm_sync(
    image: Image.Image,
    block_text: str,
    *,
    block_tag: str,
    model: str,
    transport: LiteLLMTransportLike = None,
    max_review_rounds: int = 0,
) -> tuple[float, float, float, float] | None:
    """Synchronously locate the tight bbox of a specific rendered text block via a multimodal LLM."""
    return run_sync(
        locate_text_block_bbox_with_llm(
            image,
            block_text,
            block_tag=block_tag,
            model=model,
            transport=transport,
            max_review_rounds=max_review_rounds,
        )
    )


@dataclass(slots=True)
class AzurePageDetector(PageDetectionBackend):
    """Detect pages from Azure Document Intelligence page output."""

    endpoint: str
    api_key: str
    model_id: str = "prebuilt-layout"

    async def detect(self, image: Image.Image) -> list[PageCandidate]:
        try:
            from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:  # pragma: no cover - optional extra path
            raise ConfigurationError(
                "Azure page detection requires the 'azure' extra. "
                'Install with `pip install "churro-ocr[azure]"`.'
            ) from exc

        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )
        try:
            poller = await client.begin_analyze_document(
                model_id=self.model_id,
                body=BytesIO(buffer.getvalue()),
                content_type="application/octet-stream",
            )
            result = await poller.result()
        finally:
            await client.close()

        candidates: list[PageCandidate] = []
        for page_index, page in enumerate(result.pages or []):
            polygon = _normalize_azure_page_polygon(page, image=image)
            bbox = _bbox_from_polygon(polygon) if polygon else None
            metadata = {
                "page_index": page_index,
                "page_number": getattr(page, "page_number", page_index + 1),
                "detector": "azure",
            }
            unit = getattr(page, "unit", None)
            if unit is not None:
                metadata["unit"] = str(unit)
            angle = getattr(page, "angle", None)
            if angle is not None:
                metadata["angle"] = float(angle)
            candidates.append(PageCandidate(bbox=bbox, polygon=polygon, metadata=metadata))
        return candidates or [_full_image_candidate(image)]


def _normalize_azure_page_polygon(page: Any, *, image: Image.Image) -> tuple[tuple[float, float], ...]:
    raw_polygon = getattr(page, "polygon", None)
    polygon = _normalize_polygon(raw_polygon)
    if not polygon:
        return ()

    page_width = float(getattr(page, "width", 0.0) or image.width)
    page_height = float(getattr(page, "height", 0.0) or image.height)
    scale_x = image.width / page_width if page_width else 1.0
    scale_y = image.height / page_height if page_height else 1.0
    return tuple((x * scale_x, y * scale_y) for x, y in polygon)


def _apply_box_review_decision(
    current_box: _PageBox,
    decision: _BoxReviewDecision,
    *,
    expected_page_index: int,
) -> _PageBox:
    page_index = expected_page_index
    if decision.page_index != expected_page_index:
        logger.info(
            "Review decision page_index mismatch (expected=%s, got=%s); using expected.",
            expected_page_index,
            decision.page_index,
        )

    left = _apply_edge_decision_to_coordinate(current_box.xmin, decision.left, is_min_edge=True)
    top = _apply_edge_decision_to_coordinate(current_box.ymin, decision.top, is_min_edge=True)
    right = _apply_edge_decision_to_coordinate(current_box.xmax, decision.right, is_min_edge=False)
    bottom = _apply_edge_decision_to_coordinate(
        current_box.ymax,
        decision.bottom,
        is_min_edge=False,
    )

    min_span = 1
    if left >= right:
        center = (left + right) // 2
        left = max(0, center - min_span)
        right = min(1000, center + min_span)
    if top >= bottom:
        center = (top + bottom) // 2
        top = max(0, center - min_span)
        bottom = min(1000, center + min_span)

    return _PageBox.from_json(
        {
            "page_index": page_index,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }
    )


def _no_change_edge_review_decision() -> _EdgeReviewDecision:
    return _EdgeReviewDecision(action="no_change", amount=0)


def _new_page_review_stop_state() -> dict[str, dict[str, int | bool | None]]:
    return {
        edge_name: {
            "frozen": False,
            "stable_rounds": 0,
            "last_sign": None,
            "last_mag": None,
        }
        for edge_name in _EDGE_NAMES
    }


def _page_review_is_fully_frozen(page_state: dict[str, dict[str, int | bool | None]]) -> bool:
    return all(bool(page_state[edge_name]["frozen"]) for edge_name in _EDGE_NAMES)


def _apply_page_review_stop_condition(
    *,
    prior_box: _PageBox,
    reviewed_box: _PageBox,
    page_state: dict[str, dict[str, int | bool | None]],
    round_index: int,
    subject_name: str = "page",
) -> _PageBox:
    prior_coords = _box_to_edge_coords(prior_box)
    reviewed_coords = _box_to_edge_coords(reviewed_box)
    final_coords = dict(reviewed_coords)

    for edge_name in _EDGE_NAMES:
        edge_state = page_state[edge_name]
        prior_value = prior_coords[edge_name]
        candidate_value = reviewed_coords[edge_name]
        delta = candidate_value - prior_value
        magnitude = abs(delta)

        if bool(edge_state["frozen"]):
            final_coords[edge_name] = prior_value
            continue

        if magnitude <= _REVIEW_EDGE_STOP_DEADBAND:
            final_coords[edge_name] = prior_value
            edge_state["stable_rounds"] = int(edge_state["stable_rounds"] or 0) + 1
            if int(edge_state["stable_rounds"]) >= _REVIEW_EDGE_STOP_STABLE_ROUNDS:
                edge_state["frozen"] = True
                logger.info(
                    "Freezing %s %s edge %s after %s stable round(s) (deadband <= %s).",
                    subject_name,
                    prior_box.page_index,
                    edge_name,
                    edge_state["stable_rounds"],
                    _REVIEW_EDGE_STOP_DEADBAND,
                )
            continue

        edge_state["stable_rounds"] = 0
        sign = 1 if delta > 0 else -1
        previous_sign = edge_state["last_sign"]
        previous_magnitude = edge_state["last_mag"]
        if (
            isinstance(previous_sign, int)
            and previous_sign != 0
            and previous_sign != sign
            and isinstance(previous_magnitude, int)
            and previous_magnitude > _REVIEW_EDGE_STOP_DEADBAND
            and _is_oscillating_magnitude(magnitude, previous_magnitude)
        ):
            final_coords[edge_name] = _select_more_expansive_oscillation_coordinate(
                edge_name=edge_name,
                prior_value=prior_value,
                candidate_value=candidate_value,
            )
            edge_state["frozen"] = True
            logger.info(
                "Freezing %s %s edge %s on round %s due to oscillation (prev=%s, current=%s, final=%s).",
                subject_name,
                prior_box.page_index,
                edge_name,
                round_index,
                previous_magnitude,
                magnitude,
                final_coords[edge_name],
            )
            continue

        edge_state["last_sign"] = sign
        edge_state["last_mag"] = magnitude

    return _build_page_box_from_edge_coords(prior_box.page_index, final_coords)


def _is_oscillating_magnitude(current_magnitude: int, previous_magnitude: int) -> bool:
    if current_magnitude <= 0 or previous_magnitude <= 0:
        return False
    ratio = current_magnitude / previous_magnitude if previous_magnitude else 0.0
    return (
        _REVIEW_EDGE_STOP_OSCILLATION_MAGNITUDE_RATIO_MIN
        <= ratio
        <= _REVIEW_EDGE_STOP_OSCILLATION_MAGNITUDE_RATIO_MAX
    )


def _select_more_expansive_oscillation_coordinate(
    *,
    edge_name: str,
    prior_value: int,
    candidate_value: int,
) -> int:
    if edge_name in {"left", "top"}:
        return min(prior_value, candidate_value)
    return max(prior_value, candidate_value)


def _box_to_edge_coords(box: _PageBox) -> dict[str, int]:
    return {
        "left": box.xmin,
        "top": box.ymin,
        "right": box.xmax,
        "bottom": box.ymax,
    }


def _build_page_box_from_edge_coords(page_index: int, coords: dict[str, int]) -> _PageBox:
    left = int(coords["left"])
    top = int(coords["top"])
    right = int(coords["right"])
    bottom = int(coords["bottom"])

    min_span = 1
    if left >= right:
        center = (left + right) // 2
        left = max(0, center - min_span)
        right = min(1000, center + min_span)
    if top >= bottom:
        center = (top + bottom) // 2
        top = max(0, center - min_span)
        bottom = min(1000, center + min_span)

    return _PageBox.from_json(
        {
            "page_index": page_index,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }
    )


def _strip_axis_size_pixels(
    strip_bounds: tuple[int, int, int, int],
    *,
    edge_name: str,
) -> int:
    x0, y0, x1, y1 = strip_bounds
    return (x1 - x0) if edge_name in {"left", "right"} else (y1 - y0)


def _convert_strip_delta_to_local_delta(
    strip_delta_normalized: int,
    *,
    strip_axis_pixels: int,
    local_axis_pixels: int,
) -> int:
    if strip_delta_normalized <= 0 or strip_axis_pixels <= 0 or local_axis_pixels <= 0:
        return 0
    delta_pixels = strip_delta_normalized * strip_axis_pixels / 1000
    local_delta = int(round(delta_pixels * 1000 / local_axis_pixels))
    return max(0, min(1000, local_delta))


def _apply_edge_decision_to_coordinate(
    current_value: int,
    decision: _EdgeReviewDecision,
    *,
    is_min_edge: bool,
) -> int:
    if decision.action == "no_change" or decision.amount <= 0:
        return current_value
    if decision.action == "expand":
        return current_value - decision.amount if is_min_edge else current_value + decision.amount
    if decision.action == "shrink":
        return current_value + decision.amount if is_min_edge else current_value - decision.amount
    return current_value


def _log_box_history(
    history_boxes: Sequence[Sequence[_PageBox]],
    *,
    subject_name: str = "Page",
) -> None:
    per_page_history: dict[int, dict[str, list[int]]] = {}
    for boxes in history_boxes:
        for box in boxes:
            page_history = per_page_history.setdefault(
                box.page_index,
                {"left": [], "top": [], "right": [], "bottom": []},
            )
            page_history["left"].append(box.xmin)
            page_history["top"].append(box.ymin)
            page_history["right"].append(box.xmax)
            page_history["bottom"].append(box.ymax)
    if not per_page_history:
        return
    label_width = max(len(key) for key in ("left", "top", "right", "bottom"))
    for page_index in sorted(per_page_history):
        logger.info("%s %s coordinate history:", subject_name, page_index)
        page_history = per_page_history[page_index]
        for key in ("left", "top", "right", "bottom"):
            logger.info("%s: %s", key.ljust(label_width), " -> ".join(map(str, page_history[key])))


__all__ = [
    "AzurePageDetector",
    "LLMPageDetector",
    "locate_text_block_bbox_with_llm",
    "locate_text_block_bbox_with_llm_sync",
]

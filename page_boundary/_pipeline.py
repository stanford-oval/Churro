"""Core Gemini page boundary detection pipeline primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image

from churro.utils.llm import run_llm_async
from churro.utils.log_utils import logger

from ._constants import (
    DAMPENING_CONSTANT,
    MAX_PAGE_REVIEW_ROUNDS,
    PAGE_DETECTION_PROMPT,
    PAGE_REVIEW_PROMPT_TEMPLATE,
)
from ._image_processing import draw_boxes
from ._models import PageBox
from ._serialization import boxes_equal, parse_pages_json


async def detect_page_boxes(image: Image.Image, model_key: str) -> list[PageBox]:
    """Call Gemini to predict normalized page bounding boxes."""
    response_text = await run_llm_async(
        model=model_key,
        system_prompt_text=PAGE_DETECTION_PROMPT,
        user_message_text=None,
        user_message_image=image,
        image_detail="high",
        output_json=True,
    )
    logger.info(f"Initial Gemini response: {response_text}")
    return parse_pages_json(response_text)


async def review_page_boxes(
    annotated_image: Image.Image,
    history_summary: str,
    history_steps: int,
    model_key: str,
) -> list[PageBox]:
    """Review and optionally refine bounding boxes using Gemini with visual feedback."""
    review_prompt = PAGE_REVIEW_PROMPT_TEMPLATE
    if history_summary:
        history_sections = [
            "Here are the coordinates for the currently drawn bounding boxes:",
            history_summary,
        ]
        review_prompt += "\n\n" + "\n".join(history_sections)
    response_text = await run_llm_async(
        model=model_key,
        system_prompt_text=review_prompt,
        user_message_text=None,
        user_message_image=annotated_image,
        image_detail="high",
        output_json=True,
    )
    logger.info(
        f"Review Gemini response (history rounds={history_steps}): {response_text}",
    )
    return parse_pages_json(response_text)


async def run_detection_pipeline(
    image: Image.Image,
    model_key: str,
    max_review_rounds: int = MAX_PAGE_REVIEW_ROUNDS,
) -> list[PageBox]:
    """Perform initial Gemini detection and iterative review refinement."""
    initial_boxes = await detect_page_boxes(image, model_key=model_key)
    history_boxes: list[list[PageBox]] = [initial_boxes]
    final_boxes = initial_boxes

    for round_idx in range(max(0, max_review_rounds)):
        history_summary = _format_boxes_for_prompt(history_boxes[-1])
        dampening_factor = DAMPENING_CONSTANT ** len(history_boxes)
        annotated_preview = draw_boxes(image, history_boxes[-1])
        reviewed_boxes: list[PageBox] | None = None
        try:
            reviewed_boxes = await review_page_boxes(
                annotated_preview,
                history_summary,
                history_steps=len(history_boxes),
                model_key=model_key,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.info(
                f"Review round {round_idx + 1} failed, stopping reviews: {exc}",
            )
            break

        if reviewed_boxes is None:
            break

        if not reviewed_boxes:
            break

        dampened_boxes = _apply_dampening(history_boxes[-1], reviewed_boxes, dampening_factor)

        if boxes_equal(dampened_boxes, history_boxes[-1]):
            final_boxes = dampened_boxes
            break

        history_boxes.append(dampened_boxes)
        final_boxes = dampened_boxes

    _log_box_history(history_boxes)

    return final_boxes


def save_page_crops(crops: Iterable[Image.Image], output_path: Path) -> list[Path]:
    """Persist each page crop to disk next to the annotated image."""
    saved_paths: list[Path] = []
    for idx, crop in enumerate(crops, start=1):
        crop_path = output_path.with_name(f"{output_path.stem}_page{idx}.png")
        crop.save(crop_path)
        saved_paths.append(crop_path)
    return saved_paths


__all__ = [
    "detect_page_boxes",
    "review_page_boxes",
    "run_detection_pipeline",
    "save_page_crops",
]


def _build_coordinate_history(
    history_boxes: Sequence[Sequence[PageBox]],
) -> dict[int, dict[str, list[int]]]:
    per_page_history: dict[int, dict[str, list[int]]] = {}
    for boxes in history_boxes:
        for box in boxes:
            page = per_page_history.setdefault(
                box.page_index,
                {"left": [], "top": [], "right": [], "bottom": []},
            )
            page["left"].append(box.xmin)
            page["top"].append(box.ymin)
            page["right"].append(box.xmax)
            page["bottom"].append(box.ymax)
    return per_page_history


def _format_boxes_for_prompt(boxes: Sequence[PageBox]) -> str:
    if not boxes:
        return ""
    label_width = max(len(key) for key in ("left", "top", "right", "bottom"))
    lines: list[str] = []
    for box in sorted(boxes, key=lambda item: item.page_index):
        lines.append(f"Page {box.page_index}:")
        lines.append(f"{'left'.ljust(label_width)}: {box.xmin}")
        lines.append(f"{'top'.ljust(label_width)}: {box.ymin}")
        lines.append(f"{'right'.ljust(label_width)}: {box.xmax}")
        lines.append(f"{'bottom'.ljust(label_width)}: {box.ymax}")
    return "\n".join(lines)


def _log_box_history(history_boxes: Sequence[Sequence[PageBox]]) -> None:
    per_page_history = _build_coordinate_history(history_boxes)
    if not per_page_history:
        return
    label_width = max(len(key) for key in ("left", "top", "right", "bottom"))
    for page_index in sorted(per_page_history):
        logger.info(f"Page {page_index} coordinate history:")
        history = per_page_history[page_index]
        for key in ("left", "top", "right", "bottom"):
            formatted = " -> ".join(str(value) for value in history[key])
            logger.info(f"{key.ljust(label_width)}: {formatted}")


def _apply_dampening(
    previous_boxes: Sequence[PageBox],
    updated_boxes: Sequence[PageBox],
    factor: float,
) -> list[PageBox]:
    prev_by_page = {box.page_index: box for box in previous_boxes}
    dampened: list[PageBox] = []
    for box in updated_boxes:
        previous = prev_by_page.get(box.page_index)
        if previous is None:
            dampened.append(box)
            continue
        dampened.append(
            PageBox(
                page_index=box.page_index,
                ymin=_blend_coordinate(previous.ymin, box.ymin, factor),
                xmin=_blend_coordinate(previous.xmin, box.xmin, factor),
                ymax=_blend_coordinate(previous.ymax, box.ymax, factor),
                xmax=_blend_coordinate(previous.xmax, box.xmax, factor),
            )
        )
    return dampened


def _blend_coordinate(previous: int, current: int, factor: float) -> int:
    delta = current - previous
    adjusted = previous + delta * factor
    return max(0, min(1000, int(round(adjusted))))

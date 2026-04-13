"""Built-in page detection backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, cast

from churro_ocr._internal.install import install_command_hint
from churro_ocr._internal.litellm import LiteLLMTransport
from churro_ocr._internal.logging import logger
from churro_ocr._internal.retry import retry_api_call
from churro_ocr._internal.runtime import run_sync
from churro_ocr.page_detection import PageCandidate, PageDetectionBackend
from churro_ocr.prompts import DEFAULT_BOUNDARY_DETECTION_PROMPT
from churro_ocr.prompts.layout import (
    build_boundary_review_prompt,
    build_text_block_boundary_review_prompt,
    build_text_block_localization_prompt,
)
from churro_ocr.providers import _page_detection_helpers as _helpers
from churro_ocr.providers import _page_detection_review as _review
from churro_ocr.providers.specs import LiteLLMTransportConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from PIL import Image

    from churro_ocr.types import BoundingBox

_AzureAnalyzeResultLike = _helpers._AzureAnalyzeResultLike
_bbox_from_polygon = _helpers._bbox_from_polygon
_bbox_to_polygon = _helpers._bbox_to_polygon
_BoxReviewDecision = _helpers._BoxReviewDecision
_boxes_equal = _helpers._boxes_equal
_build_box_review_preview = _helpers._build_box_review_preview
_build_edge_strip_review_preview = _helpers._build_edge_strip_review_preview
_configuration_error = _helpers._configuration_error
_convert_source_box_to_review_crop_box = _helpers._convert_source_box_to_review_crop_box
_EDGE_NAMES = _helpers._EDGE_NAMES
_EdgeReviewDecision = _helpers._EdgeReviewDecision
_full_image_candidate = _helpers._full_image_candidate
_map_review_crop_box_to_source_box = _helpers._map_review_crop_box_to_source_box
_merge_instruction_prompts = _helpers._merge_instruction_prompts
_normalize_azure_page_polygon = _helpers._normalize_azure_page_polygon
_PageBox = _helpers._PageBox
_PAGE_DETECTION_BOX_WIDTH = _helpers._PAGE_DETECTION_BOX_WIDTH
_PageDetectionTransform = _helpers._PageDetectionTransform
_parse_page_boxes_json = _helpers._parse_page_boxes_json
_parse_single_edge_review_decision_json = _helpers._parse_single_edge_review_decision_json
_parse_text_block_box_json = _helpers._parse_text_block_box_json
_parse_text_block_edge_review_decision_json = _helpers._parse_text_block_edge_review_decision_json
_prepare_detection_image = _helpers._prepare_detection_image
_provider_error = _helpers._provider_error
_strip_code_fence = _helpers._strip_code_fence
_TEXT_BLOCK_DETECTION_BOX_WIDTH = _helpers._TEXT_BLOCK_DETECTION_BOX_WIDTH
_TEXT_BLOCK_REVIEW_CROP_MARGIN_FRACTION = _helpers._TEXT_BLOCK_REVIEW_CROP_MARGIN_FRACTION
_type_error = _helpers._type_error
_value_error = _helpers._value_error
_apply_box_review_decision = _review._apply_box_review_decision
_apply_edge_decision_to_coordinate = _review._apply_edge_decision_to_coordinate
_apply_page_review_stop_condition = _review._apply_page_review_stop_condition
_convert_strip_delta_to_local_delta = _review._convert_strip_delta_to_local_delta
_is_oscillating_magnitude = _review._is_oscillating_magnitude
_log_box_history = _review._log_box_history
_new_page_review_stop_state = _review._new_page_review_stop_state
_no_change_edge_review_decision = _review._no_change_edge_review_decision
_page_review_is_fully_frozen = _review._page_review_is_fully_frozen
_select_more_expansive_oscillation_coordinate = _review._select_more_expansive_oscillation_coordinate
_strip_axis_size_pixels = _review._strip_axis_size_pixels
LiteLLMTransportLike = LiteLLMTransportConfig | LiteLLMTransport | None


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
        message = f"Invalid strip axis size for edge '{edge_name}'."
        raise _value_error(message)

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
        message = f"Invalid strip axis size for edge '{edge_name}'."
        raise _value_error(message)

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
    """Detect one or more pages via a multimodal LLM prompt.

    :param model: Multimodal model identifier to query through LiteLLM.
    :param system_prompt: System prompt used for the initial page-box request.
    :param prompt_template: Optional user prompt override for the initial request.
    :param transport: Optional LiteLLM transport config.
    :param max_review_rounds: Number of iterative review rounds used to refine
        the initial page boxes.
    """

    model: str
    system_prompt: str = DEFAULT_BOUNDARY_DETECTION_PROMPT
    prompt_template: str | None = None
    transport: LiteLLMTransportConfig | None = None
    max_review_rounds: int = 0

    async def detect(self, image: Image.Image) -> list[PageCandidate]:
        """Detect page candidates from one image.

        :param image: Source image that may contain one or more visible pages.
        :returns: Detected page candidates in reading order. Falls back to a
            single full-image candidate when no page boxes are returned.
        """
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
) -> BoundingBox | None:
    """Locate the tight bbox of a specific rendered text block via a multimodal LLM.

    :param image: Source page image containing the rendered block.
    :param block_text: Normalized text content of the target block.
    :param block_tag: HDML-style block tag describing the block type.
    :param model: Multimodal model identifier to query through LiteLLM.
    :param transport: Optional LiteLLM transport or transport config.
    :param max_review_rounds: Number of iterative review rounds used to refine
        the initial box.
    :returns: Bounding box in source-image coordinates, or ``None`` when no
        unique matching block can be found.
    :raises ValueError: If ``block_text`` or ``block_tag`` is blank.
    """
    normalized_block_text = block_text.strip()
    if not normalized_block_text:
        message = "block_text must not be blank."
        raise _value_error(message)

    normalized_block_tag = block_tag.strip()
    if not normalized_block_tag:
        message = "block_tag must not be blank."
        raise _value_error(message)

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
) -> BoundingBox | None:
    """Synchronously locate the tight bbox of a specific rendered text block via a multimodal LLM.

    :param image: Source page image containing the rendered block.
    :param block_text: Normalized text content of the target block.
    :param block_tag: HDML-style block tag describing the block type.
    :param model: Multimodal model identifier to query through LiteLLM.
    :param transport: Optional LiteLLM transport or transport config.
    :param max_review_rounds: Number of iterative review rounds used to refine
        the initial box.
    :returns: Bounding box in source-image coordinates, or ``None`` when no
        unique matching block can be found.
    :raises ValueError: If ``block_text`` or ``block_tag`` is blank.
    """
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
    """Detect pages from Azure Document Intelligence page output.

    :param endpoint: Azure Document Intelligence endpoint URL.
    :param api_key: Azure API key for the configured resource.
    :param model_id: Azure model ID used for page analysis.
    """

    endpoint: str
    api_key: str
    model_id: str = "prebuilt-layout"

    async def detect(self, image: Image.Image) -> list[PageCandidate]:
        """Detect page candidates from one image using Azure.

        :param image: Source image to analyze.
        :returns: Detected page candidates in reading order. Falls back to a
            single full-image candidate when Azure returns no pages.
        :raises ConfigurationError: If the optional Azure dependency is not installed.
        """
        try:
            from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:  # pragma: no cover - optional extra path
            message = f"Azure page detection requires the `azure` runtime. {install_command_hint('azure')}"
            raise _configuration_error(message) from exc

        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )
        try:
            image_bytes = buffer.getvalue()

            async def _analyze_document() -> _AzureAnalyzeResultLike:
                poller = await client.begin_analyze_document(
                    model_id=self.model_id,
                    body=BytesIO(image_bytes),
                    content_type="application/octet-stream",
                )
                return cast("_AzureAnalyzeResultLike", await poller.result())

            result = await retry_api_call(
                _analyze_document,
                operation_name="Azure page detection request",
                context=f"for model {self.model_id}",
            )
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


__all__ = [
    "AzurePageDetector",
    "LLMPageDetector",
    "locate_text_block_bbox_with_llm",
    "locate_text_block_bbox_with_llm_sync",
]

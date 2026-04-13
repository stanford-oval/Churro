"""Review-state helpers for iterative page-detection refinement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from churro_ocr._internal.logging import logger
from churro_ocr.providers._page_detection_helpers import (
    _EDGE_NAMES,
    _REVIEW_EDGE_STOP_DEADBAND,
    _REVIEW_EDGE_STOP_OSCILLATION_MAGNITUDE_RATIO_MAX,
    _REVIEW_EDGE_STOP_OSCILLATION_MAGNITUDE_RATIO_MIN,
    _REVIEW_EDGE_STOP_STABLE_ROUNDS,
    _BoxReviewDecision,
    _EdgeReviewDecision,
    _PageBox,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    local_delta = round(delta_pixels * 1000 / local_axis_pixels)
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

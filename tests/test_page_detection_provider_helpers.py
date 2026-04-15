from __future__ import annotations

from typing import Any, cast

import pytest
from PIL import Image

from churro_ocr.errors import ProviderError
from churro_ocr.providers.page_detection import (
    _apply_box_review_decision,
    _apply_edge_decision_to_coordinate,
    _apply_page_review_stop_condition,
    _BoxReviewDecision,
    _build_edge_strip_review_preview,
    _convert_strip_delta_to_local_delta,
    _EdgeReviewDecision,
    _is_oscillating_magnitude,
    _merge_instruction_prompts,
    _new_page_review_stop_state,
    _PageBox,
    _parse_page_boxes_json,
    _parse_single_edge_review_decision_json,
    _parse_text_block_box_json,
    _parse_text_block_edge_review_decision_json,
    _review_page_box,
    _review_single_edge_from_strip,
    _review_single_text_block_edge_from_strip,
    _review_text_block_box,
    _run_review_pipeline,
    _select_more_expansive_oscillation_coordinate,
    _strip_code_fence,
)


def test_strip_code_fence_removes_fenced_wrapper() -> None:
    assert _strip_code_fence('```json\n{"pages": []}\n```') == '{"pages": []}'


def test_parse_page_boxes_json_supports_fenced_json_and_sorts_boxes() -> None:
    boxes = _parse_page_boxes_json(
        """```json
{"pages": [
  {"page_index": 2, "left": 500, "top": 100, "right": 700, "bottom": 900},
  {"page_index": 1, "left": 100, "top": 100, "right": 400, "bottom": 900}
]}
```"""
    )

    assert [box.page_index for box in boxes] == [1, 2]
    assert boxes[0].xmin < boxes[0].xmax


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("[]", "JSON object"),
        ('{"pages": {}}', "`pages` list"),
        ('{"pages": ["bad"]}', "entry 0 must be an object"),
    ],
)
def test_parse_page_boxes_json_rejects_invalid_payloads(payload: str, expected: str) -> None:
    with pytest.raises(ProviderError, match=expected):
        _parse_page_boxes_json(payload)


def test_parse_text_block_box_json_supports_flat_nested_and_not_found_payloads() -> None:
    flat = _parse_text_block_box_json('{"left": 10, "top": 20, "right": 30, "bottom": 40}')
    nested = _parse_text_block_box_json('{"block": {"left": 100, "top": 200, "right": 300, "bottom": 400}}')
    missing = _parse_text_block_box_json('{"block_found": false}')

    assert flat is not None
    assert flat.page_index == 1
    assert nested is not None
    assert nested.page_index == 1
    assert missing is None


def test_parse_text_block_box_json_rejects_invalid_block_payload() -> None:
    with pytest.raises(ProviderError, match="`block` must be an object or null"):
        _parse_text_block_box_json('{"block": 1}')


def test_parse_single_edge_review_decision_json_accepts_nested_and_top_level_payloads() -> None:
    nested = _parse_single_edge_review_decision_json(
        '{"page_index": 3, "edge": "left", "decision": {"action": "expand", "amount": 12}}'
    )
    top_level = _parse_single_edge_review_decision_json(
        '{"page_index": 4, "edge": "top", "action": "no_change", "amount": 99}'
    )

    assert nested == (3, "left", _EdgeReviewDecision(action="expand", amount=12))
    assert top_level == (4, "top", _EdgeReviewDecision(action="no_change", amount=0))


def test_parse_single_edge_review_decision_json_rejects_invalid_edge() -> None:
    with pytest.raises(ValueError, match="left/top/right/bottom"):
        _parse_single_edge_review_decision_json(
            '{"page_index": 1, "edge": "center", "action": "expand", "amount": 1}'
        )


def test_parse_text_block_edge_review_decision_json_validates_payload() -> None:
    edge_name, decision = _parse_text_block_edge_review_decision_json(
        '{"edge": "bottom", "decision": {"action": "shrink", "amount": 5}}'
    )

    assert edge_name == "bottom"
    assert decision == _EdgeReviewDecision(action="shrink", amount=5)


def test_build_edge_strip_review_preview_rejects_unknown_edges() -> None:
    with pytest.raises(ValueError, match="Unsupported edge 'center'"):
        _build_edge_strip_review_preview(
            Image.new("RGB", (40, 40), color="white"),
            _PageBox.from_json({"page_index": 0, "left": 200, "top": 200, "right": 800, "bottom": 800}),
            "center",
        )


def test_merge_instruction_prompts_merges_non_empty_parts_and_rejects_empty_input() -> None:
    assert _merge_instruction_prompts(" first ", None, "second") == "first\n\nsecond"
    with pytest.raises(ValueError, match="at least one non-empty instruction prompt"):
        _merge_instruction_prompts(None, "  ")


def test_apply_box_review_decision_uses_expected_page_index_and_min_span() -> None:
    current_box = _PageBox.from_json({"page_index": 0, "left": 300, "top": 300, "right": 700, "bottom": 700})
    reviewed = _apply_box_review_decision(
        current_box,
        _BoxReviewDecision(
            page_index=99,
            left=_EdgeReviewDecision(action="shrink", amount=500),
            top=_EdgeReviewDecision(action="shrink", amount=500),
            right=_EdgeReviewDecision(action="expand", amount=0),
            bottom=_EdgeReviewDecision(action="expand", amount=0),
        ),
        expected_page_index=1,
    )

    assert reviewed.page_index == 1
    assert reviewed.xmin < reviewed.xmax
    assert reviewed.ymin < reviewed.ymax


def test_apply_page_review_stop_condition_freezes_edges_after_stable_rounds() -> None:
    prior_box = _PageBox.from_json({"page_index": 0, "left": 100, "top": 100, "right": 900, "bottom": 900})
    small_shift = _PageBox.from_json({"page_index": 0, "left": 104, "top": 102, "right": 896, "bottom": 898})
    page_state = _new_page_review_stop_state()

    first = _apply_page_review_stop_condition(
        prior_box=prior_box,
        reviewed_box=small_shift,
        page_state=page_state,
        round_index=1,
    )
    second = _apply_page_review_stop_condition(
        prior_box=first,
        reviewed_box=small_shift,
        page_state=page_state,
        round_index=2,
    )

    assert first == prior_box
    assert second == prior_box
    assert all(bool(page_state[edge_name]["frozen"]) for edge_name in ("left", "top", "right", "bottom"))


def test_page_detection_math_helpers_cover_expansion_and_oscillation_logic() -> None:
    assert _convert_strip_delta_to_local_delta(200, strip_axis_pixels=50, local_axis_pixels=100) == 100
    assert _convert_strip_delta_to_local_delta(0, strip_axis_pixels=50, local_axis_pixels=100) == 0
    assert (
        _apply_edge_decision_to_coordinate(
            100,
            _EdgeReviewDecision(action="expand", amount=10),
            is_min_edge=True,
        )
        == 90
    )
    assert (
        _apply_edge_decision_to_coordinate(
            100,
            _EdgeReviewDecision(action="shrink", amount=10),
            is_min_edge=False,
        )
        == 90
    )
    assert _is_oscillating_magnitude(6, 8) is True
    assert _is_oscillating_magnitude(0, 8) is False
    assert (
        _select_more_expansive_oscillation_coordinate(
            edge_name="left",
            prior_value=100,
            candidate_value=120,
        )
        == 100
    )
    assert (
        _select_more_expansive_oscillation_coordinate(
            edge_name="right",
            prior_value=700,
            candidate_value=680,
        )
        == 700
    )


@pytest.mark.asyncio
async def test_review_single_edge_from_strip_logs_mismatches_and_scales_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_messages: list[str] = []

    class FakeLogger:
        def info(self, message: str, *args: object) -> None:
            log_messages.append(message % args if args else message)

    class FakeTransport:
        def prepare_messages(
            self,
            *,
            system_prompt: str | None,
            user_prompt: str | None,
            images: list[Image.Image],
        ) -> list[dict[str, object]]:
            assert system_prompt is None
            assert user_prompt is not None
            assert images
            assert images[0].size == (20, 60)
            return [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]

        async def complete_text(
            self,
            *,
            model: str,
            messages: list[dict[str, object]],
            output_json: bool,
        ) -> str:
            assert model == "example/model"
            assert output_json is True
            assert messages[0]["role"] == "user"
            return '{"page_index": 99, "edge": "top", "action": "expand", "amount": 500}'

    monkeypatch.setattr("churro_ocr.providers.page_detection.logger", FakeLogger())

    decision = await _review_single_edge_from_strip(
        model="example/model",
        review_image=Image.new("RGB", (200, 100), color="white"),
        strip_image=Image.new("RGB", (20, 60), color="white"),
        strip_bounds=(0, 0, 20, 60),
        edge_name="left",
        page_index=1,
        history_steps=2,
        round_index=3,
        transport=cast("Any", FakeTransport()),
    )

    assert decision == _EdgeReviewDecision(action="expand", amount=50)
    assert any("page_index mismatch" in message for message in log_messages)
    assert any("edge mismatch" in message for message in log_messages)


@pytest.mark.asyncio
async def test_review_single_text_block_edge_from_strip_logs_mismatch_and_scales_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_messages: list[str] = []

    class FakeLogger:
        def info(self, message: str, *args: object) -> None:
            log_messages.append(message % args if args else message)

    class FakeTransport:
        def prepare_messages(
            self,
            *,
            system_prompt: str | None,
            user_prompt: str | None,
            images: list[Image.Image],
        ) -> list[dict[str, object]]:
            assert system_prompt is None
            assert user_prompt is not None
            assert images
            assert images[0].size == (60, 20)
            return [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]

        async def complete_text(
            self,
            *,
            model: str,
            messages: list[dict[str, object]],
            output_json: bool,
        ) -> str:
            assert model == "example/model"
            assert output_json is True
            assert messages[0]["role"] == "user"
            return '{"edge": "left", "action": "shrink", "amount": 500}'

    monkeypatch.setattr("churro_ocr.providers.page_detection.logger", FakeLogger())

    decision = await _review_single_text_block_edge_from_strip(
        model="example/model",
        review_image=Image.new("RGB", (120, 100), color="white"),
        strip_image=Image.new("RGB", (60, 20), color="white"),
        strip_bounds=(0, 0, 60, 20),
        edge_name="top",
        block_tag="Paragraph",
        block_text="Et fuit lux",
        history_steps=1,
        round_index=1,
        transport=cast("Any", FakeTransport()),
    )

    assert decision == _EdgeReviewDecision(action="shrink", amount=100)
    assert any("edge-review mismatch" in message for message in log_messages)


@pytest.mark.asyncio
async def test_review_page_box_falls_back_to_no_change_when_an_edge_review_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_box = _PageBox.from_json({"page_index": 2, "left": 200, "top": 200, "right": 800, "bottom": 800})

    async def _fake_review_single_edge_from_strip(**kwargs: object) -> _EdgeReviewDecision:
        if kwargs["edge_name"] == "left":
            raise RuntimeError("boom")
        return _EdgeReviewDecision(action="no_change", amount=0)

    monkeypatch.setattr(
        "churro_ocr.providers.page_detection._review_single_edge_from_strip",
        _fake_review_single_edge_from_strip,
    )

    reviewed = await _review_page_box(
        image=Image.new("RGB", (400, 400), color="white"),
        current_box=current_box,
        history_steps=1,
        round_index=1,
        model="example/model",
        transport=cast("Any", object()),
    )

    assert reviewed == current_box


@pytest.mark.asyncio
async def test_review_text_block_box_falls_back_to_no_change_when_an_edge_review_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_box = _PageBox.from_json({"page_index": 3, "left": 150, "top": 250, "right": 850, "bottom": 750})

    async def _fake_review_single_text_block_edge_from_strip(**kwargs: object) -> _EdgeReviewDecision:
        if kwargs["edge_name"] == "bottom":
            raise RuntimeError("boom")
        return _EdgeReviewDecision(action="no_change", amount=0)

    monkeypatch.setattr(
        "churro_ocr.providers.page_detection._review_single_text_block_edge_from_strip",
        _fake_review_single_text_block_edge_from_strip,
    )

    reviewed = await _review_text_block_box(
        image=Image.new("RGB", (400, 400), color="white"),
        current_box=current_box,
        block_tag="Paragraph",
        block_text="Et fuit lux",
        history_steps=1,
        round_index=1,
        model="example/model",
        transport=cast("Any", object()),
    )

    assert reviewed == current_box


@pytest.mark.asyncio
async def test_run_review_pipeline_stops_immediately_when_all_pages_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_box = _PageBox.from_json({"page_index": 1, "left": 200, "top": 200, "right": 800, "bottom": 800})
    review_called = {"value": False}

    monkeypatch.setattr(
        "churro_ocr.providers.page_detection._new_page_review_stop_state",
        lambda: {
            edge_name: {
                "frozen": True,
                "stable_rounds": 0,
                "last_sign": None,
                "last_mag": None,
            }
            for edge_name in ("left", "top", "right", "bottom")
        },
    )

    async def _review_box(box: _PageBox, history_steps: int, round_index: int) -> _PageBox:
        del box, history_steps, round_index
        review_called["value"] = True
        message = "review_box should not be called for frozen pages"
        raise AssertionError(message)

    result = await _run_review_pipeline(
        initial_boxes=[initial_box],
        max_review_rounds=2,
        review_box=_review_box,
        subject_name_singular="page",
        subject_name_plural="pages",
    )

    assert result == [initial_box]
    assert review_called["value"] is False


@pytest.mark.asyncio
async def test_run_review_pipeline_preserves_prior_boxes_on_exception_and_none_result() -> None:
    first_box = _PageBox.from_json({"page_index": 1, "left": 100, "top": 100, "right": 400, "bottom": 400})
    second_box = _PageBox.from_json({"page_index": 2, "left": 500, "top": 500, "right": 900, "bottom": 900})

    async def _review_box(box: _PageBox, history_steps: int, round_index: int) -> _PageBox:
        del history_steps, round_index
        if box.page_index == 1:
            raise RuntimeError("boom")
        return cast("Any", None)

    result = await _run_review_pipeline(
        initial_boxes=[first_box, second_box],
        max_review_rounds=1,
        review_box=_review_box,
        subject_name_singular="page",
        subject_name_plural="pages",
    )

    assert result == [first_box, second_box]

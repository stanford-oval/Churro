"""MinerU2.5 helpers for Hugging Face OCR backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from churro_ocr.providers._hf_helpers import (
    _MINERU25_SAMPLING_FIELD_NAMES,
    _MINERU25_SCOPED_PREFIXES,
    _MINERU25_STEP_ALIASES,
    _resolve_model_max_length,
)
from churro_ocr.providers._mineru25 import (
    MinerU25PipelineHelper,
    MinerU25SamplingParams,
    replace_sampling_param,
)
from churro_ocr.providers._shared import normalize_media_inputs
from churro_ocr.templates import (
    MINERU2_5_2509_1_2B_FORMULA_PROMPT,
    MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
    MINERU2_5_2509_1_2B_LAYOUT_PROMPT,
    MINERU2_5_2509_1_2B_OCR_PROMPT,
    MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    MINERU2_5_2509_1_2B_TABLE_PROMPT,
    OCRPromptTemplateLike,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(slots=True, frozen=True)
class _MinerU25StepContext:
    runtime: object
    processor: object
    model: object
    batch_size: int


@dataclass(slots=True, frozen=True)
class _MinerU25Templates:
    default_template: OCRPromptTemplateLike
    layout_template: OCRPromptTemplateLike
    table_template: OCRPromptTemplateLike
    formula_template: OCRPromptTemplateLike
    image_analysis_template: OCRPromptTemplateLike


def _default_mineru25_helper() -> MinerU25PipelineHelper:
    return MinerU25PipelineHelper(
        prompts={
            "[default]": MINERU2_5_2509_1_2B_OCR_PROMPT,
            "[layout]": MINERU2_5_2509_1_2B_LAYOUT_PROMPT,
            "table": MINERU2_5_2509_1_2B_TABLE_PROMPT,
            "equation": MINERU2_5_2509_1_2B_FORMULA_PROMPT,
            "image": MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
            "chart": MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
        },
        system_prompt=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    )


def _template_for_step(
    step_key: str,
    templates: _MinerU25Templates,
) -> OCRPromptTemplateLike:
    if step_key == "[layout]":
        return templates.layout_template
    if step_key == "table":
        return templates.table_template
    if step_key == "equation":
        return templates.formula_template
    if step_key in {"image", "chart"}:
        return templates.image_analysis_template
    return templates.default_template


def _resolve_rendered_prompt(
    rendered: object,
    *,
    provider_error: Callable[[str], Exception],
) -> str:
    if isinstance(rendered, tuple):
        if not rendered:
            message = "MinerU2.5 returned an empty chat template render."
            raise provider_error(message)
        rendered = rendered[0]
    if not isinstance(rendered, str):
        message = "MinerU2.5 chat template did not render text."
        raise provider_error(message)
    return rendered


def _resolve_sampling_override(
    generation_kwargs: Mapping[str, object],
    *,
    effective_step: str,
    field_name: str,
) -> float | int | None:
    override_value: float | int | None = None
    global_value = generation_kwargs.get(field_name)
    if isinstance(global_value, (int, float)):
        override_value = global_value
    step_value = generation_kwargs.get(f"{effective_step}_{field_name}")
    if isinstance(step_value, (int, float)):
        override_value = step_value
    return override_value


def _resolve_step_sampling(
    *,
    helper: MinerU25PipelineHelper,
    generation_kwargs: Mapping[str, object],
    step_key: str,
) -> MinerU25SamplingParams:
    effective_step = _MINERU25_STEP_ALIASES.get(step_key, "default")
    sampling = helper.sampling_for(step_key)
    changes = {
        field_name: override_value
        for field_name in _MINERU25_SAMPLING_FIELD_NAMES
        if (
            override_value := _resolve_sampling_override(
                generation_kwargs,
                effective_step=effective_step,
                field_name=field_name,
            )
        )
        is not None
    }
    return replace_sampling_param(sampling, **changes) if changes else sampling


def _scoped_generation_keys() -> frozenset[str]:
    return frozenset(
        (f"{prefix}{field_name}" if prefix else field_name)
        for prefix in ("", *_MINERU25_SCOPED_PREFIXES)
        for field_name in _MINERU25_SAMPLING_FIELD_NAMES
    )


_MINERU25_GENERATION_OVERRIDE_KEYS = _scoped_generation_keys()


def _should_sample(sampling: MinerU25SamplingParams) -> bool:
    return ((sampling.temperature or 0.0) > 0.0) and ((sampling.top_k or 1) > 1)


def _set_generation_kwarg(
    generation_kwargs: dict[str, object],
    key: str,
    value: object,
    *,
    enabled: bool = True,
) -> None:
    if enabled and value is not None:
        generation_kwargs[key] = value


def _resolve_generation_length(
    *,
    sampling: MinerU25SamplingParams,
    generation_kwargs: Mapping[str, object],
    model: object,
) -> tuple[str, int] | None:
    if sampling.max_new_tokens is not None:
        return ("max_new_tokens", sampling.max_new_tokens)
    max_length = generation_kwargs.get("max_length", _resolve_model_max_length(model))
    if isinstance(max_length, str):
        max_length = int(max_length)
    if isinstance(max_length, int):
        return ("max_length", max_length)
    return None


def _extra_generation_kwargs(generation_kwargs: Mapping[str, object]) -> dict[str, object]:
    extra_kwargs = dict(generation_kwargs)
    extra_kwargs.pop("max_length", None)
    for override_key in _MINERU25_GENERATION_OVERRIDE_KEYS:
        extra_kwargs.pop(override_key, None)
    return extra_kwargs


def _resolve_generation_kwargs(
    *,
    helper: MinerU25PipelineHelper,
    generation_kwargs: Mapping[str, object],
    step_key: str,
    model: object,
) -> dict[str, object]:
    sampling = _resolve_step_sampling(
        helper=helper,
        generation_kwargs=generation_kwargs,
        step_key=step_key,
    )
    do_sample = _should_sample(sampling)
    resolved_generation_kwargs: dict[str, object] = {"do_sample": do_sample}
    _set_generation_kwarg(
        resolved_generation_kwargs,
        "temperature",
        sampling.temperature,
        enabled=do_sample,
    )
    _set_generation_kwarg(
        resolved_generation_kwargs,
        "top_p",
        sampling.top_p,
        enabled=do_sample,
    )
    _set_generation_kwarg(
        resolved_generation_kwargs,
        "top_k",
        sampling.top_k,
        enabled=do_sample,
    )
    _set_generation_kwarg(
        resolved_generation_kwargs,
        "repetition_penalty",
        sampling.repetition_penalty,
    )
    _set_generation_kwarg(
        resolved_generation_kwargs,
        "no_repeat_ngram_size",
        sampling.no_repeat_ngram_size,
    )
    generation_length = _resolve_generation_length(
        sampling=sampling,
        generation_kwargs=generation_kwargs,
        model=model,
    )
    if generation_length is not None:
        key, value = generation_length
        resolved_generation_kwargs[key] = value
    resolved_generation_kwargs.update(_extra_generation_kwargs(generation_kwargs))
    return resolved_generation_kwargs


def _build_step_batch_kwargs(
    *,
    rendered_prompt: str,
    image_inputs: object,
    video_inputs: object,
) -> dict[str, object]:
    batch_kwargs: dict[str, object] = {
        "text": [rendered_prompt],
        "images": normalize_media_inputs(image_inputs),
        "return_tensors": "pt",
        "padding": True,
    }
    normalized_video_inputs = normalize_media_inputs(video_inputs)
    if normalized_video_inputs is not None:
        batch_kwargs["videos"] = normalized_video_inputs
    return batch_kwargs

"""Generic decoding, prompt, and generation helpers for Hugging Face OCR backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from churro_ocr.providers._hf_runtime import (
    _configuration_error,
    _HFProcessorDecoder,
    _load_torch_module,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from churro_ocr.templates import OCRConversation
    from churro_ocr.types import OCRConversationContentItem


def _default_chandra_ocr_2_model_kwargs() -> dict[str, object]:
    model_kwargs: dict[str, object] = {
        "device_map": "auto",
        "dtype": "auto",
    }
    try:
        torch = _load_torch_module()
    except ImportError:  # pragma: no cover - torch is installed separately for local HF use
        return model_kwargs

    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16
    return model_kwargs


def _default_mineru25_model_kwargs() -> dict[str, object]:
    model_kwargs: dict[str, object] = {"device_map": "auto"}
    dtype_key = "dtype"
    transformers_version: str
    try:
        from transformers import __version__ as imported_transformers_version

        transformers_version = str(imported_transformers_version)
    except ImportError:  # pragma: no cover - transformers is installed via the hf runtime
        transformers_version = ""

    version_parts = transformers_version.split(".")
    if len(version_parts) >= 2:
        try:
            major = int(version_parts[0])
            minor = int(version_parts[1])
        except ValueError:
            major = 0
            minor = 0
        if major < 4 or (major == 4 and minor < 56):
            dtype_key = "torch_dtype"
    model_kwargs[dtype_key] = "auto"
    return model_kwargs


def _conversation_content_items(message: object) -> list[OCRConversationContentItem]:
    return cast("list[OCRConversationContentItem]", cast("Mapping[str, object]", message)["content"])


def _message_text_lines(content_items: list[OCRConversationContentItem]) -> list[str]:
    return [
        cast("str", item["text"]).strip()
        for item in content_items
        if item.get("type") == "text" and isinstance(item.get("text"), str)
    ]


def _message_text(content_items: list[OCRConversationContentItem]) -> str:
    return "\n".join(_message_text_lines(content_items)).strip()


def _has_image_content(content_items: list[OCRConversationContentItem]) -> bool:
    return any(item.get("type") == "image" for item in content_items)


def _deepseek_ocr_2_prompt_from_conversation(conversation: OCRConversation) -> str:
    prompt_lines: list[str] = []
    has_image = False
    for message in conversation:
        role = message.get("role")
        content_items = _conversation_content_items(message)
        if role == "system":
            system_text = _message_text(content_items)
            if system_text:
                message = "DeepSeek-OCR-2 does not support system prompts in the HF backend."
                raise _configuration_error(message)
            continue
        if role != "user":
            continue
        has_image = has_image or _has_image_content(content_items)
        prompt_lines.extend(text for text in _message_text_lines(content_items) if text)
    prompt_text = "\n".join(prompt_lines).strip()
    if not prompt_text:
        message = "DeepSeek-OCR-2 requires a non-empty OCR prompt."
        raise _configuration_error(message)
    if has_image:
        return f"<image>\n{prompt_text}"
    return prompt_text


def _move_batch_to_model(batch: dict[str, object], model: object) -> dict[str, object]:
    model_device = getattr(model, "device", None)
    if hasattr(batch, "to") and model_device is not None:
        batch = cast("dict[str, object]", cast("Any", batch).to(model_device))
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        for key, value in batch.items():
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating_point", False):
                batch[key] = cast("Any", value).to(dtype=model_dtype)
    return batch


def _decode_completion_texts(
    processor: object,
    batch: Mapping[str, object],
    generated_ids: object,
) -> list[str]:
    return _decode_completion_texts_with_options(
        processor,
        batch,
        generated_ids,
        skip_special_tokens=True,
    )


def _completion_ids_from_generated_ids(batch: Mapping[str, object], generated_ids: object) -> object:
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None and hasattr(attention_mask, "sum"):
        prompt_lengths = cast("Any", attention_mask).sum(dim=1).tolist()
        return [
            output_ids[int(prompt_length) :]
            for prompt_length, output_ids in zip(prompt_lengths, cast("Any", generated_ids), strict=True)
        ]
    prompt_length = cast("Any", batch["input_ids"]).shape[1]
    return cast("Any", generated_ids)[:, prompt_length:]


def _decode_completion_texts_with_options(
    processor: object,
    batch: Mapping[str, object],
    generated_ids: object,
    *,
    skip_special_tokens: bool,
) -> list[str]:
    completion_ids = _completion_ids_from_generated_ids(batch, generated_ids)
    return cast("_HFProcessorDecoder", processor).batch_decode(
        completion_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def _resolve_model_max_length(model: object) -> int | None:
    config = getattr(model, "config", None)
    max_length = getattr(config, "max_position_embeddings", None)
    if isinstance(max_length, int):
        return max_length
    text_config = getattr(config, "text_config", None)
    text_max_length = getattr(text_config, "max_position_embeddings", None)
    if isinstance(text_max_length, int):
        return text_max_length
    return None


_MINERU25_STEP_ALIASES = {
    "[layout]": "layout",
    "table": "table",
    "equation": "equation",
    "image": "image",
    "chart": "chart",
}
_MINERU25_SAMPLING_FIELD_NAMES = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "max_new_tokens",
)
_MINERU25_SCOPED_PREFIXES = (
    "layout_",
    "table_",
    "equation_",
    "image_",
    "chart_",
    "default_",
)


def _paddleocr_vl_processor_kwargs(*, processor: object, padding: bool) -> dict[str, object]:
    processor_kwargs: dict[str, object] = {
        "text_kwargs": {
            "padding": padding,
            "return_mm_token_type_ids": True,
        }
    }
    image_processor = getattr(processor, "image_processor", None)
    images_kwargs: dict[str, int] = {}
    for key in ("min_pixels", "max_pixels"):
        value = getattr(image_processor, key, None)
        if isinstance(value, int):
            images_kwargs[key] = value
    if images_kwargs:
        processor_kwargs["images_kwargs"] = images_kwargs
    return processor_kwargs

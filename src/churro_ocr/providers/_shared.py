"""Internal shared helpers for OCR provider adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRResult
from churro_ocr.page_detection import DocumentPage
from churro_ocr.providers.specs import ImagePreprocessor, TextPostprocessor
from churro_ocr.templates import (
    OCRConversation,
    OCRPromptTemplateLike,
    build_ocr_conversation,
)


def preprocess_backend_page(
    page: DocumentPage,
    *,
    image_preprocessor: ImagePreprocessor,
) -> DocumentPage:
    """Return a page copy with the backend's explicit image preprocessing applied."""
    return replace(page, image=image_preprocessor(page.image))


def render_ocr_prompt(
    processor: object,
    template: OCRPromptTemplateLike,
    page: DocumentPage,
    *,
    add_generation_prompt: bool,
) -> tuple[str, OCRConversation]:
    """Render a provider prompt and preserve the structured OCR conversation."""
    conversation = build_ocr_conversation(template, page)
    processor_apply = getattr(processor, "apply_chat_template", None)
    if callable(processor_apply):
        rendered = processor_apply(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        return rendered, conversation

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(tokenizer_apply):
        rendered = tokenizer_apply(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        return rendered, conversation

    raise ConfigurationError(
        "OCR prompt rendering requires either `processor.apply_chat_template(...)`, "
        "or `processor.tokenizer.apply_chat_template(...)`."
    )


def normalize_media_inputs(media_inputs: object | None) -> object | None:
    """Normalize provider media inputs to list form when needed."""
    if media_inputs is None:
        return None
    if isinstance(media_inputs, (list, tuple)):
        return media_inputs
    return [media_inputs]


def build_ocr_result(
    text: str,
    *,
    provider_name: str,
    model_name: str,
    text_postprocessor: TextPostprocessor,
    metadata: dict[str, Any] | None = None,
) -> OCRResult:
    """Build a normalized OCR result after postprocessing."""
    return OCRResult(
        text=text_postprocessor(text),
        provider_name=provider_name,
        model_name=model_name,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "build_ocr_result",
    "normalize_media_inputs",
    "preprocess_backend_page",
    "render_ocr_prompt",
]

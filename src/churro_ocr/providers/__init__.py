"""Public OCR builders and page detection backends."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from churro_ocr.ocr import BatchOCRBackend

if TYPE_CHECKING:
    from churro_ocr.providers.builder import build_ocr_backend
    from churro_ocr.providers.page_detection import (
        AzurePageDetector,
        LLMPageDetector,
        locate_text_block_bbox_with_llm,
        locate_text_block_bbox_with_llm_sync,
    )
    from churro_ocr.providers.specs import (
        DEFAULT_OCR_MAX_TOKENS,
        AzureDocumentIntelligenceOptions,
        HuggingFaceOptions,
        LiteLLMTransportConfig,
        MistralOptions,
        OCRBackendSpec,
        OCRModelProfile,
        OpenAICompatibleOptions,
        VLLMOptions,
        resolve_ocr_profile,
    )


_LAZY_EXPORTS = {
    "AzureDocumentIntelligenceOptions": (
        "churro_ocr.providers.specs",
        "AzureDocumentIntelligenceOptions",
    ),
    "AzurePageDetector": ("churro_ocr.providers.page_detection", "AzurePageDetector"),
    "build_ocr_backend": ("churro_ocr.providers.builder", "build_ocr_backend"),
    "DEFAULT_OCR_MAX_TOKENS": ("churro_ocr.providers.specs", "DEFAULT_OCR_MAX_TOKENS"),
    "HuggingFaceOptions": ("churro_ocr.providers.specs", "HuggingFaceOptions"),
    "LiteLLMTransportConfig": ("churro_ocr.providers.specs", "LiteLLMTransportConfig"),
    "LLMPageDetector": ("churro_ocr.providers.page_detection", "LLMPageDetector"),
    "locate_text_block_bbox_with_llm": (
        "churro_ocr.providers.page_detection",
        "locate_text_block_bbox_with_llm",
    ),
    "locate_text_block_bbox_with_llm_sync": (
        "churro_ocr.providers.page_detection",
        "locate_text_block_bbox_with_llm_sync",
    ),
    "MistralOptions": ("churro_ocr.providers.specs", "MistralOptions"),
    "OCRBackendSpec": ("churro_ocr.providers.specs", "OCRBackendSpec"),
    "OCRModelProfile": ("churro_ocr.providers.specs", "OCRModelProfile"),
    "OpenAICompatibleOptions": ("churro_ocr.providers.specs", "OpenAICompatibleOptions"),
    "resolve_ocr_profile": ("churro_ocr.providers.specs", "resolve_ocr_profile"),
    "VLLMOptions": ("churro_ocr.providers.specs", "VLLMOptions"),
}

__all__ = [
    "AzureDocumentIntelligenceOptions",
    "AzurePageDetector",
    "BatchOCRBackend",
    "build_ocr_backend",
    "DEFAULT_OCR_MAX_TOKENS",
    "HuggingFaceOptions",
    "LiteLLMTransportConfig",
    "LLMPageDetector",
    "locate_text_block_bbox_with_llm",
    "locate_text_block_bbox_with_llm_sync",
    "MistralOptions",
    "OCRBackendSpec",
    "OCRModelProfile",
    "OpenAICompatibleOptions",
    "resolve_ocr_profile",
    "VLLMOptions",
]


def __getattr__(name: str) -> Any:
    """Lazy-load provider exports to avoid circular imports during package init."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the lazy exports to interactive tooling."""
    return sorted(set(globals()) | set(__all__))

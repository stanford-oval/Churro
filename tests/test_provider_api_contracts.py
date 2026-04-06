from __future__ import annotations

import re
from typing import Any, cast

import pytest

import churro_ocr.providers as providers
from churro_ocr.errors import ConfigurationError
from churro_ocr.providers import (
    AzureDocumentIntelligenceOptions,
    HuggingFaceOptions,
    LiteLLMTransportConfig,
    MistralOptions,
    OCRBackendSpec,
    OpenAICompatibleOptions,
    build_ocr_backend,
    resolve_ocr_profile,
)
from churro_ocr.providers.builder import _merge_mapping
from churro_ocr.providers.ocr import (
    AzureDocumentIntelligenceOCRBackend,
    MistralOCRBackend,
    OpenAICompatibleOCRBackend,
)


def test_merge_mapping_merges_nested_dictionaries() -> None:
    merged = _merge_mapping(
        {"outer": {"left": 1, "right": 2}, "flat": 1},
        {"outer": {"right": 3, "bottom": 4}, "flat": 2},
    )

    assert merged == {"outer": {"left": 1, "right": 3, "bottom": 4}, "flat": 2}


def test_resolve_ocr_profile_rejects_unknown_profile_name() -> None:
    with pytest.raises(ValueError, match="Unknown OCR profile 'missing-profile'"):
        resolve_ocr_profile(model_id=None, profile="missing-profile")


def test_provider_lazy_exports_reject_unknown_attributes() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'missing_export'"):
        providers.__getattr__("missing_export")


def test_provider_dir_lists_lazy_exports() -> None:
    exported = providers.__dir__()

    assert "build_ocr_backend" in exported
    assert "OCRBackendSpec" in exported
    assert "LiteLLMTransportConfig" in exported


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (OCRBackendSpec(provider="litellm"), "OCR provider 'litellm' requires `model`."),
        (
            OCRBackendSpec(provider="openai-compatible", model="local-model"),
            "OCR provider 'openai-compatible' requires `transport.api_base`.",
        ),
        (
            OCRBackendSpec(provider="azure"),
            "OCR provider 'azure' requires AzureDocumentIntelligenceOptions(endpoint=..., api_key=...).",
        ),
        (
            OCRBackendSpec(provider="mistral"),
            "OCR provider 'mistral' requires MistralOptions(api_key=...).",
        ),
        (
            OCRBackendSpec(
                provider="mistral",
                options=MistralOptions(api_key="secret"),
            ),
            "OCR provider 'mistral' requires `model` to be one of: mistral-ocr-2505, mistral-ocr-2512.",
        ),
        (
            OCRBackendSpec(
                provider="mistral",
                model="mistral-ocr-latest",
                options=MistralOptions(api_key="secret"),
            ),
            (
                "OCR provider 'mistral' only supports `model` values mistral-ocr-2505, "
                "mistral-ocr-2512; got 'mistral-ocr-latest'."
            ),
        ),
        (
            OCRBackendSpec(
                provider="hf",
                model="example/model",
                options=cast("Any", OpenAICompatibleOptions()),
            ),
            "OCR provider 'hf' requires options of type HuggingFaceOptions, got OpenAICompatibleOptions.",
        ),
        (
            OCRBackendSpec(
                provider="openai-compatible",
                model="example/model",
                options=cast("Any", HuggingFaceOptions()),
            ),
            (
                "OCR provider 'openai-compatible' requires options of type "
                "OpenAICompatibleOptions, got HuggingFaceOptions."
            ),
        ),
    ],
)
def test_build_ocr_backend_validation_errors(spec: OCRBackendSpec, expected: str) -> None:
    with pytest.raises(ConfigurationError, match=re.escape(expected)):
        build_ocr_backend(spec)


def test_build_ocr_backend_supports_custom_openai_model_prefix() -> None:
    backend = build_ocr_backend(
        OCRBackendSpec(
            provider="openai-compatible",
            model="local-model",
            transport=LiteLLMTransportConfig(
                api_base="http://127.0.0.1:8000/v1",
            ),
            options=OpenAICompatibleOptions(model_prefix="custom"),
        )
    )

    assert isinstance(backend, OpenAICompatibleOCRBackend)
    assert backend.model == "custom/local-model"
    assert backend.model_name == "local-model"


def test_build_ocr_backend_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported OCR provider 'bogus'"):
        build_ocr_backend(
            OCRBackendSpec(
                provider=cast("Any", "bogus"),
                model="example/model",
            )
        )


def test_build_ocr_backend_accepts_provider_specific_options() -> None:
    azure_backend = build_ocr_backend(
        OCRBackendSpec(
            provider="azure",
            model="layout-model",
            options=AzureDocumentIntelligenceOptions(
                endpoint="https://example.invalid",
                api_key="secret",
            ),
        )
    )
    mistral_backend = build_ocr_backend(
        OCRBackendSpec(
            provider="mistral",
            model="mistral-ocr-2512",
            options=MistralOptions(api_key="secret"),
        )
    )

    assert isinstance(azure_backend, AzureDocumentIntelligenceOCRBackend)
    assert isinstance(mistral_backend, MistralOCRBackend)
    assert azure_backend.model_id == "layout-model"
    assert azure_backend.model_name == "layout-model"
    assert mistral_backend.model == "mistral-ocr-2512"

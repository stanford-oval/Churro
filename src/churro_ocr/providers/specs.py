"""Public OCR provider specs, options, and model profile resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image

from churro_ocr._internal.image import prepare_ocr_image
from churro_ocr.prompts import DEFAULT_OCR_OUTPUT_TAG, strip_ocr_output_tag
from churro_ocr.templates import (
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    OCRConversation,
    OCRPromptTemplateLike,
)

if TYPE_CHECKING:
    pass


OCRProvider = Literal["litellm", "openai-compatible", "azure", "mistral", "hf", "vllm"]
ImagePreprocessor = Callable[[Image.Image], Image.Image]
TextPostprocessor = Callable[[str], str]
VisionInputBuilder = Callable[[OCRConversation], object]
DEFAULT_OCR_MAX_TOKENS = 20_000


def identity_text_postprocessor(text: str) -> str:
    """Return OCR text unchanged.

    :param text: OCR text to return.
    :returns: The original ``text`` value.
    """
    return text


def default_ocr_image_preprocessor(image: Image.Image) -> Image.Image:
    """Apply the default OCR image preprocessing.

    :param image: Source page image.
    :returns: Preprocessed image ready for OCR.
    """
    return prepare_ocr_image(image)


def default_ocr_text_postprocessor(text: str) -> str:
    """Strip the default OCR output tag wrapper.

    :param text: Raw OCR response text.
    :returns: OCR text with the default wrapper removed when present.
    """
    return strip_ocr_output_tag(text, output_tag=DEFAULT_OCR_OUTPUT_TAG)


@dataclass(slots=True, frozen=True)
class LiteLLMTransportConfig:
    """Shared transport config for LiteLLM-based multimodal requests.

    :param api_base: Optional API base URL override.
    :param api_key: Optional API key forwarded to LiteLLM.
    :param api_version: Optional API version string for providers that need one.
    :param image_detail: Optional image-detail hint supported by some providers.
    :param completion_kwargs: Extra completion kwargs merged into LiteLLM calls.
    :param cache_dir: Optional disk-cache directory for LiteLLM request caching.
    """

    api_base: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    image_detail: str | None = None
    completion_kwargs: dict[str, object] = field(default_factory=dict)
    cache_dir: str | Path | None = None


@dataclass(slots=True, frozen=True)
class OpenAICompatibleOptions:
    """Provider options for OpenAI-compatible OCR servers.

    :param model_prefix: Provider prefix prepended to the configured model name.
    """

    model_prefix: str | None = None


@dataclass(slots=True, frozen=True)
class HuggingFaceOptions:
    """Provider options for local Hugging Face OCR backends.

    :param trust_remote_code: Whether to allow remote model code execution.
    :param processor_kwargs: Extra kwargs passed to ``AutoProcessor.from_pretrained``.
    :param model_kwargs: Extra kwargs passed to model loading.
    :param generation_kwargs: Extra generation kwargs passed at inference time.
    :param vision_input_builder: Optional override for building multimodal inputs.
    :param backend_variant: Optional implementation preset such as ``"dots-ocr-1.5"``.
    """

    trust_remote_code: bool | None = None
    processor_kwargs: dict[str, object] = field(default_factory=dict)
    model_kwargs: dict[str, object] = field(default_factory=dict)
    generation_kwargs: dict[str, object] = field(default_factory=dict)
    vision_input_builder: VisionInputBuilder | None = None
    backend_variant: str | None = None


@dataclass(slots=True, frozen=True)
class VLLMOptions:
    """Provider options for local vLLM OCR backends.

    :param trust_remote_code: Whether to allow remote model code execution.
    :param processor_kwargs: Extra kwargs passed to ``AutoProcessor.from_pretrained``.
    :param llm_kwargs: Extra kwargs passed to the vLLM ``LLM`` constructor.
    :param sampling_kwargs: Extra kwargs passed to vLLM sampling params.
    :param limit_mm_per_prompt: Per-request multimodal limits passed to vLLM.
    """

    trust_remote_code: bool | None = None
    processor_kwargs: dict[str, object] = field(default_factory=dict)
    llm_kwargs: dict[str, object] = field(default_factory=dict)
    sampling_kwargs: dict[str, object] = field(default_factory=dict)
    limit_mm_per_prompt: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AzureDocumentIntelligenceOptions:
    """Provider options for Azure Document Intelligence OCR.

    :param endpoint: Azure Document Intelligence endpoint URL.
    :param api_key: Azure API key for the configured resource.
    """

    endpoint: str | None = None
    api_key: str | None = None


@dataclass(slots=True, frozen=True)
class MistralOptions:
    """Provider options for Mistral OCR.

    :param api_key: Mistral API key used for OCR requests.
    """

    api_key: str | None = None


OCRProviderOptions = (
    OpenAICompatibleOptions
    | HuggingFaceOptions
    | VLLMOptions
    | AzureDocumentIntelligenceOptions
    | MistralOptions
)


@dataclass(slots=True, frozen=True)
class OCRModelProfile:
    """Model-level OCR behavior shared across provider adapters.

    :param profile_name: Stable profile identifier.
    :param template: Prompt template used to render OCR input.
    :param image_preprocessor: Image preprocessor applied before OCR.
    :param text_postprocessor: Text postprocessor applied after OCR.
    :param display_name: Optional human-readable model name.
    :param transport: Default LiteLLM transport settings for this profile.
    :param huggingface: Default Hugging Face backend options for this profile.
    :param vllm: Default vLLM backend options for this profile.
    """

    profile_name: str
    template: OCRPromptTemplateLike = DEFAULT_OCR_TEMPLATE
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = default_ocr_text_postprocessor
    display_name: str | None = None
    transport: LiteLLMTransportConfig = field(default_factory=LiteLLMTransportConfig)
    huggingface: HuggingFaceOptions = field(default_factory=HuggingFaceOptions)
    vllm: VLLMOptions = field(default_factory=VLLMOptions)


@dataclass(slots=True, frozen=True)
class OCRBackendSpec:
    """Declarative builder input for OCR backends.

    :param provider: OCR provider identifier.
    :param model: Provider-specific model identifier.
    :param profile: Optional built-in or custom model profile.
    :param transport: Optional transport settings for LiteLLM-based providers.
    :param options: Optional provider-specific options dataclass.
    """

    provider: OCRProvider
    model: str | None = None
    profile: str | OCRModelProfile | None = None
    transport: LiteLLMTransportConfig | None = None
    options: OCRProviderOptions | None = None


def default_ocr_profile() -> OCRModelProfile:
    """Return the generic OCR model profile.

    :returns: Baseline profile used when no more specific profile matches.
    """
    return OCRModelProfile(profile_name="default")


def churro_3b_profile() -> OCRModelProfile:
    """Return the built-in ``stanford-oval/churro-3B`` OCR profile.

    :returns: Profile configured for the built-in CHURRO 3B template.
    """
    return OCRModelProfile(
        profile_name=CHURRO_3B_MODEL_ID,
        template=CHURRO_3B_XML_TEMPLATE,
        text_postprocessor=identity_text_postprocessor,
        display_name="churro-3B",
    )


def dots_ocr_1_5_profile() -> OCRModelProfile:
    """Return the built-in ``kristaller486/dots.ocr-1.5`` OCR profile.

    :returns: Profile configured for the built-in Dots OCR 1.5 template.
    """
    return OCRModelProfile(
        profile_name=DOTS_OCR_1_5_MODEL_ID,
        template=DOTS_OCR_1_5_OCR_TEMPLATE,
        text_postprocessor=identity_text_postprocessor,
        display_name="dots.ocr-1.5",
        huggingface=HuggingFaceOptions(
            trust_remote_code=True,
            backend_variant="dots-ocr-1.5",
        ),
        vllm=VLLMOptions(
            trust_remote_code=True,
        ),
    )


def _profile_registry() -> dict[str, OCRModelProfile]:
    default_profile = default_ocr_profile()
    churro_profile = churro_3b_profile()
    dots_profile = dots_ocr_1_5_profile()
    return {
        default_profile.profile_name: default_profile,
        churro_profile.profile_name: churro_profile,
        dots_profile.profile_name: dots_profile,
    }


def resolve_ocr_profile(
    *,
    model_id: str | None,
    profile: str | OCRModelProfile | None = None,
) -> OCRModelProfile:
    """Resolve the OCR model profile for a model or explicit profile.

    :param model_id: Model identifier that may map to a built-in profile.
    :param profile: Explicit profile name or profile object to use.
    :returns: Resolved OCR model profile.
    :raises ValueError: If ``profile`` is a string that does not match a known profile.
    """
    if isinstance(profile, OCRModelProfile):
        return profile

    registry = _profile_registry()
    if isinstance(profile, str):
        try:
            return registry[profile]
        except KeyError as exc:
            raise ValueError(f"Unknown OCR profile '{profile}'.") from exc

    if model_id is not None and model_id in registry:
        return registry[model_id]
    return registry["default"]


__all__ = [
    "AzureDocumentIntelligenceOptions",
    "DEFAULT_OCR_MAX_TOKENS",
    "default_ocr_image_preprocessor",
    "default_ocr_profile",
    "default_ocr_text_postprocessor",
    "HuggingFaceOptions",
    "identity_text_postprocessor",
    "ImagePreprocessor",
    "LiteLLMTransportConfig",
    "MistralOptions",
    "OCRBackendSpec",
    "OCRModelProfile",
    "OCRProvider",
    "OpenAICompatibleOptions",
    "resolve_ocr_profile",
    "TextPostprocessor",
    "VisionInputBuilder",
    "VLLMOptions",
]

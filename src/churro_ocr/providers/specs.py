"""Public OCR provider specs, options, and model profile resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image

from churro_ocr._internal.image import ensure_rgb
from churro_ocr.errors import ConfigurationError
from churro_ocr.providers._ocr_processing import (
    chandra_image_preprocessor,
    chandra_text_postprocessor,
    deepseek_ocr_2_text_postprocessor,
    default_ocr_image_preprocessor,
    default_ocr_text_postprocessor,
    firered_ocr_text_postprocessor,
    glm_ocr_image_preprocessor,
    glm_ocr_text_postprocessor,
    identity_text_postprocessor,
    infinity_parser_7b_text_postprocessor,
    lfm2_5_vl_text_postprocessor,
    olmocr_image_preprocessor,
    olmocr_text_postprocessor,
    paddleocr_vl_text_postprocessor,
)
from churro_ocr.templates import (
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DEEPSEEK_OCR_2_MODEL_ID,
    DEEPSEEK_OCR_2_OCR_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
    DOTS_MOCR_MODEL_ID,
    DOTS_MOCR_OCR_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    FIRERED_OCR_MODEL_ID,
    FIRERED_OCR_OCR_TEMPLATE,
    GLM_OCR_MODEL_ID,
    GLM_OCR_OCR_TEMPLATE,
    INFINITY_PARSER_7B_MODEL_ID,
    INFINITY_PARSER_7B_OCR_TEMPLATE,
    LFM2_5_VL_1_6B_MODEL_ID,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_MODEL_ID,
    MINERU2_5_2509_1_2B_OCR_TEMPLATE,
    OLMOCR_2_7B_1025_FP8_MODEL_ID,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
    PADDLEOCR_VL_1_5_MODEL_ID,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
    OCRConversation,
    OCRPromptTemplateLike,
)
from churro_ocr.types import MetadataDict

if TYPE_CHECKING:
    from pathlib import Path


OCRProvider = Literal["litellm", "openai-compatible", "azure", "mistral", "hf"]
MistralOCRModel = Literal["mistral-ocr-2505", "mistral-ocr-2512"]
ImagePreprocessor = Callable[[Image.Image], Image.Image]
TextPostprocessorResult = str | tuple[str, MetadataDict]
TextPostprocessor = Callable[[str], TextPostprocessorResult]
VisionInputBuilder = Callable[[OCRConversation], object]
DEFAULT_OCR_MAX_TOKENS = 25_000
CHANDRA_OCR_MAX_TOKENS = 12_384
DEEPSEEK_OCR_2_MAX_TOKENS = 8_192
FIRERED_OCR_MAX_TOKENS = 4_096
GLM_OCR_MAX_TOKENS = 8_192
INFINITY_PARSER_7B_MAX_TOKENS = 8_192
OLMOCR_MAX_TOKENS = 8_000
PADDLEOCR_VL_MAX_TOKENS = 4_096
INFINITY_PARSER_7B_MIN_PIXELS = 256 * 28 * 28
INFINITY_PARSER_7B_MAX_PIXELS = 2304 * 28 * 28
MISTRAL_OCR_MODEL_IDS: tuple[MistralOCRModel, ...] = (
    "mistral-ocr-2505",
    "mistral-ocr-2512",
)


def _configuration_error(message: str) -> ConfigurationError:
    return ConfigurationError(message)


def validate_mistral_ocr_model(
    model: str | None,
    *,
    context: str = "OCR provider 'mistral'",
) -> MistralOCRModel:
    """Return a supported pinned Mistral OCR model id or raise a configuration error."""
    supported_models = ", ".join(MISTRAL_OCR_MODEL_IDS)
    if model is None:
        message = f"{context} requires `model` to be one of: {supported_models}."
        raise _configuration_error(message)
    if model not in MISTRAL_OCR_MODEL_IDS:
        message = f"{context} only supports `model` values {supported_models}; got {model!r}."
        raise _configuration_error(message)
    return cast("MistralOCRModel", model)


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
    OpenAICompatibleOptions | HuggingFaceOptions | AzureDocumentIntelligenceOptions | MistralOptions
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
    """

    profile_name: str
    template: OCRPromptTemplateLike = DEFAULT_OCR_TEMPLATE
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = default_ocr_text_postprocessor
    display_name: str | None = None
    transport: LiteLLMTransportConfig = field(default_factory=LiteLLMTransportConfig)
    huggingface: HuggingFaceOptions = field(default_factory=HuggingFaceOptions)


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


def chandra_ocr_2_profile() -> OCRModelProfile:
    """Return the built-in ``datalab-to/chandra-ocr-2`` OCR profile."""
    return OCRModelProfile(
        profile_name=CHANDRA_OCR_2_MODEL_ID,
        template=CHANDRA_OCR_2_OCR_TEMPLATE,
        image_preprocessor=chandra_image_preprocessor,
        text_postprocessor=chandra_text_postprocessor,
        display_name="chandra-ocr-2",
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": CHANDRA_OCR_MAX_TOKENS,
                "temperature": 0.0,
                "top_p": 0.1,
            }
        ),
        huggingface=HuggingFaceOptions(
            generation_kwargs={
                "max_new_tokens": CHANDRA_OCR_MAX_TOKENS,
            },
            backend_variant="chandra-ocr-2",
        ),
    )


def deepseek_ocr_2_profile() -> OCRModelProfile:
    """Return the built-in ``deepseek-ai/DeepSeek-OCR-2`` OCR profile."""
    return OCRModelProfile(
        profile_name=DEEPSEEK_OCR_2_MODEL_ID,
        template=DEEPSEEK_OCR_2_OCR_TEMPLATE,
        text_postprocessor=deepseek_ocr_2_text_postprocessor,
        display_name="DeepSeek-OCR-2",
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": DEEPSEEK_OCR_2_MAX_TOKENS,
                "temperature": 0.0,
            }
        ),
        huggingface=HuggingFaceOptions(
            model_kwargs={
                "use_safetensors": True,
            },
            generation_kwargs={
                "max_new_tokens": DEEPSEEK_OCR_2_MAX_TOKENS,
            },
            trust_remote_code=True,
            backend_variant="deepseek-ocr-2",
        ),
    )


def firered_ocr_profile() -> OCRModelProfile:
    """Return the built-in ``FireRedTeam/FireRed-OCR`` OCR profile."""
    return OCRModelProfile(
        profile_name=FIRERED_OCR_MODEL_ID,
        template=FIRERED_OCR_OCR_TEMPLATE,
        image_preprocessor=default_ocr_image_preprocessor,
        text_postprocessor=firered_ocr_text_postprocessor,
        display_name="FireRed-OCR",
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": FIRERED_OCR_MAX_TOKENS,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        ),
        huggingface=HuggingFaceOptions(
            generation_kwargs={
                "max_new_tokens": FIRERED_OCR_MAX_TOKENS,
                "do_sample": False,
            },
        ),
    )


def glm_ocr_profile() -> OCRModelProfile:
    """Return the built-in ``zai-org/GLM-OCR`` OCR profile."""
    return OCRModelProfile(
        profile_name=GLM_OCR_MODEL_ID,
        template=GLM_OCR_OCR_TEMPLATE,
        image_preprocessor=glm_ocr_image_preprocessor,
        text_postprocessor=glm_ocr_text_postprocessor,
        display_name="GLM-OCR",
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": GLM_OCR_MAX_TOKENS,
                "temperature": 0.0,
            }
        ),
        huggingface=HuggingFaceOptions(
            generation_kwargs={
                "max_new_tokens": GLM_OCR_MAX_TOKENS,
                "do_sample": False,
            },
            backend_variant="glm-ocr",
        ),
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
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": 2_048,
                "temperature": 0.0,
            }
        ),
        huggingface=HuggingFaceOptions(
            trust_remote_code=True,
            backend_variant="dots-ocr-1.5",
        ),
    )


def dots_mocr_profile() -> OCRModelProfile:
    """Return the built-in ``rednote-hilab/dots.mocr`` OCR profile."""
    return OCRModelProfile(
        profile_name=DOTS_MOCR_MODEL_ID,
        template=DOTS_MOCR_OCR_TEMPLATE,
        text_postprocessor=identity_text_postprocessor,
        display_name="dots.mocr",
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": DEFAULT_OCR_MAX_TOKENS,
                "temperature": 0.0,
            }
        ),
        huggingface=HuggingFaceOptions(
            trust_remote_code=True,
            backend_variant="dots-mocr",
        ),
    )


def paddleocr_vl_1_5_profile() -> OCRModelProfile:
    """Return the built-in ``PaddlePaddle/PaddleOCR-VL-1.5`` OCR profile."""
    return OCRModelProfile(
        profile_name=PADDLEOCR_VL_1_5_MODEL_ID,
        template=PADDLEOCR_VL_1_5_OCR_TEMPLATE,
        text_postprocessor=paddleocr_vl_text_postprocessor,
        display_name="PaddleOCR-VL-1.5",
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": PADDLEOCR_VL_MAX_TOKENS,
                "temperature": 0.0,
            }
        ),
        huggingface=HuggingFaceOptions(
            generation_kwargs={
                "max_new_tokens": PADDLEOCR_VL_MAX_TOKENS,
                "do_sample": False,
            },
            backend_variant="paddleocr-vl-1.5",
        ),
    )


def infinity_parser_7b_profile() -> OCRModelProfile:
    """Return the built-in ``infly/Infinity-Parser-7B`` OCR profile."""
    return OCRModelProfile(
        profile_name=INFINITY_PARSER_7B_MODEL_ID,
        template=INFINITY_PARSER_7B_OCR_TEMPLATE,
        image_preprocessor=ensure_rgb,
        text_postprocessor=infinity_parser_7b_text_postprocessor,
        display_name="Infinity-Parser-7B",
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": INFINITY_PARSER_7B_MAX_TOKENS,
                "temperature": 0.0,
                "top_p": 0.95,
            }
        ),
        huggingface=HuggingFaceOptions(
            processor_kwargs={
                "min_pixels": INFINITY_PARSER_7B_MIN_PIXELS,
                "max_pixels": INFINITY_PARSER_7B_MAX_PIXELS,
            },
            generation_kwargs={
                "max_new_tokens": 4_096,
            },
        ),
    )


def mineru2_5_2509_1_2b_profile() -> OCRModelProfile:
    """Return the built-in ``opendatalab/MinerU2.5-2509-1.2B`` OCR profile."""
    return OCRModelProfile(
        profile_name=MINERU2_5_2509_1_2B_MODEL_ID,
        template=MINERU2_5_2509_1_2B_OCR_TEMPLATE,
        image_preprocessor=ensure_rgb,
        text_postprocessor=identity_text_postprocessor,
        display_name="MinerU2.5-2509-1.2B",
        huggingface=HuggingFaceOptions(
            processor_kwargs={
                "use_fast": True,
            },
            backend_variant="mineru2.5",
        ),
    )


def _olmocr_profile(*, profile_name: str, display_name: str) -> OCRModelProfile:
    return OCRModelProfile(
        profile_name=profile_name,
        template=OLMOCR_2_7B_1025_OCR_TEMPLATE,
        image_preprocessor=olmocr_image_preprocessor,
        text_postprocessor=olmocr_text_postprocessor,
        display_name=display_name,
        transport=LiteLLMTransportConfig(
            completion_kwargs={
                "max_tokens": OLMOCR_MAX_TOKENS,
                "temperature": 0.1,
            }
        ),
        huggingface=HuggingFaceOptions(
            generation_kwargs={
                "max_new_tokens": OLMOCR_MAX_TOKENS,
                "temperature": 0.1,
                "do_sample": True,
            },
        ),
    )


def lfm2_5_vl_1_6b_profile() -> OCRModelProfile:
    """Return the built-in ``LiquidAI/LFM2.5-VL-1.6B`` OCR profile."""
    return OCRModelProfile(
        profile_name=LFM2_5_VL_1_6B_MODEL_ID,
        template=LFM2_5_VL_1_6B_OCR_TEMPLATE,
        text_postprocessor=lfm2_5_vl_text_postprocessor,
        display_name="LFM2.5-VL-1.6B",
        huggingface=HuggingFaceOptions(
            generation_kwargs={
                "max_new_tokens": 512,
                "do_sample": False,
                "repetition_penalty": 1.05,
            },
            backend_variant="lfm2.5-vl",
        ),
    )


def olmocr_2_7b_1025_profile() -> OCRModelProfile:
    """Return the built-in ``allenai/olmOCR-2-7B-1025`` OCR profile."""
    return _olmocr_profile(
        profile_name=OLMOCR_2_7B_1025_MODEL_ID,
        display_name="olmOCR-2-7B-1025",
    )


def olmocr_2_7b_1025_fp8_profile() -> OCRModelProfile:
    """Return the built-in ``allenai/olmOCR-2-7B-1025-FP8`` OCR profile."""
    return _olmocr_profile(
        profile_name=OLMOCR_2_7B_1025_FP8_MODEL_ID,
        display_name="olmOCR-2-7B-1025-FP8",
    )


def _profile_registry() -> dict[str, OCRModelProfile]:
    default_profile = default_ocr_profile()
    churro_profile = churro_3b_profile()
    chandra_profile = chandra_ocr_2_profile()
    deepseek_profile = deepseek_ocr_2_profile()
    firered_profile = firered_ocr_profile()
    glm_profile = glm_ocr_profile()
    dots_mocr = dots_mocr_profile()
    dots_profile = dots_ocr_1_5_profile()
    infinity_parser_profile = infinity_parser_7b_profile()
    lfm2_5_vl_profile = lfm2_5_vl_1_6b_profile()
    mineru2_5_profile = mineru2_5_2509_1_2b_profile()
    olmocr_profile = olmocr_2_7b_1025_profile()
    olmocr_fp8_profile = olmocr_2_7b_1025_fp8_profile()
    paddleocr_vl_profile = paddleocr_vl_1_5_profile()
    return {
        default_profile.profile_name: default_profile,
        churro_profile.profile_name: churro_profile,
        chandra_profile.profile_name: chandra_profile,
        deepseek_profile.profile_name: deepseek_profile,
        firered_profile.profile_name: firered_profile,
        glm_profile.profile_name: glm_profile,
        dots_mocr.profile_name: dots_mocr,
        dots_profile.profile_name: dots_profile,
        infinity_parser_profile.profile_name: infinity_parser_profile,
        lfm2_5_vl_profile.profile_name: lfm2_5_vl_profile,
        mineru2_5_profile.profile_name: mineru2_5_profile,
        olmocr_profile.profile_name: olmocr_profile,
        olmocr_fp8_profile.profile_name: olmocr_fp8_profile,
        paddleocr_vl_profile.profile_name: paddleocr_vl_profile,
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
            message = f"Unknown OCR profile '{profile}'."
            raise ValueError(message) from exc

    if model_id is not None and model_id in registry:
        return registry[model_id]
    return registry["default"]


__all__ = [
    "DEFAULT_OCR_MAX_TOKENS",
    "MISTRAL_OCR_MODEL_IDS",
    "AzureDocumentIntelligenceOptions",
    "HuggingFaceOptions",
    "ImagePreprocessor",
    "LiteLLMTransportConfig",
    "MistralOCRModel",
    "MistralOptions",
    "OCRBackendSpec",
    "OCRModelProfile",
    "OCRProvider",
    "OpenAICompatibleOptions",
    "TextPostprocessor",
    "VisionInputBuilder",
    "chandra_image_preprocessor",
    "chandra_ocr_2_profile",
    "chandra_text_postprocessor",
    "deepseek_ocr_2_profile",
    "deepseek_ocr_2_text_postprocessor",
    "default_ocr_image_preprocessor",
    "default_ocr_profile",
    "default_ocr_text_postprocessor",
    "firered_ocr_profile",
    "firered_ocr_text_postprocessor",
    "glm_ocr_image_preprocessor",
    "glm_ocr_profile",
    "glm_ocr_text_postprocessor",
    "identity_text_postprocessor",
    "infinity_parser_7b_profile",
    "infinity_parser_7b_text_postprocessor",
    "lfm2_5_vl_1_6b_profile",
    "lfm2_5_vl_text_postprocessor",
    "mineru2_5_2509_1_2b_profile",
    "olmocr_image_preprocessor",
    "olmocr_text_postprocessor",
    "paddleocr_vl_1_5_profile",
    "paddleocr_vl_text_postprocessor",
    "resolve_ocr_profile",
    "validate_mistral_ocr_model",
]

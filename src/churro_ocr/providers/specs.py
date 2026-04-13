"""Public OCR provider specs, options, and model profile resolution."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from PIL import Image

from churro_ocr._internal.image import ensure_rgb, prepare_ocr_image, resize_image_to_fit
from churro_ocr.errors import ConfigurationError
from churro_ocr.prompts import (
    DEFAULT_OCR_OUTPUT_TAG,
    parse_chandra_response,
    parse_olmocr_response,
    strip_ocr_output_tag,
    strip_rich_ocr_markup_to_plain_text,
)
from churro_ocr.templates import (
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DEEPSEEK_OCR_2_MODEL_ID,
    DEEPSEEK_OCR_2_OCR_PROMPT,
    DEEPSEEK_OCR_2_OCR_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
    DOTS_MOCR_MODEL_ID,
    DOTS_MOCR_OCR_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    INFINITY_PARSER_7B_MODEL_ID,
    INFINITY_PARSER_7B_OCR_PROMPT,
    INFINITY_PARSER_7B_OCR_TEMPLATE,
    INFINITY_PARSER_7B_SYSTEM_PROMPT,
    LFM2_5_VL_1_6B_MODEL_ID,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_MODEL_ID,
    MINERU2_5_2509_1_2B_OCR_TEMPLATE,
    OLMOCR_2_7B_1025_FP8_MODEL_ID,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
    PADDLEOCR_VL_1_5_MODEL_ID,
    PADDLEOCR_VL_1_5_OCR_PROMPT,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
    OCRConversation,
    OCRPromptTemplateLike,
)

if TYPE_CHECKING:
    pass


OCRProvider = Literal["litellm", "openai-compatible", "azure", "mistral", "hf"]
MistralOCRModel = Literal["mistral-ocr-2505", "mistral-ocr-2512"]
ImagePreprocessor = Callable[[Image.Image], Image.Image]
TextPostprocessorResult = str | tuple[str, dict[str, Any]]
TextPostprocessor = Callable[[str], TextPostprocessorResult]
VisionInputBuilder = Callable[[OCRConversation], object]
DEFAULT_OCR_MAX_TOKENS = 25_000
CHANDRA_OCR_MAX_TOKENS = 12_384
DEEPSEEK_OCR_2_MAX_TOKENS = 8_192
INFINITY_PARSER_7B_MAX_TOKENS = 8_192
OLMOCR_MAX_TOKENS = 8_000
PADDLEOCR_VL_MAX_TOKENS = 4_096
INFINITY_PARSER_7B_MIN_PIXELS = 256 * 28 * 28
INFINITY_PARSER_7B_MAX_PIXELS = 2304 * 28 * 28
CHANDRA_MAX_IMAGE_SIZE = (3_072, 2_048)
CHANDRA_MIN_IMAGE_SIZE = (1_792, 28)
CHANDRA_IMAGE_GRID_SIZE = 28
OLMOCR_TARGET_LONGEST_IMAGE_DIM = 1_288
MISTRAL_OCR_MODEL_IDS: tuple[MistralOCRModel, ...] = (
    "mistral-ocr-2505",
    "mistral-ocr-2512",
)


def validate_mistral_ocr_model(
    model: str | None,
    *,
    context: str = "OCR provider 'mistral'",
) -> MistralOCRModel:
    """Return a supported pinned Mistral OCR model id or raise a configuration error."""
    supported_models = ", ".join(MISTRAL_OCR_MODEL_IDS)
    if model is None:
        raise ConfigurationError(f"{context} requires `model` to be one of: {supported_models}.")
    if model not in MISTRAL_OCR_MODEL_IDS:
        raise ConfigurationError(f"{context} only supports `model` values {supported_models}; got {model!r}.")
    return cast(MistralOCRModel, model)


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


_CHAT_ROLE_PREFIXES = {
    "assistant",
    "assistant:",
    "user",
    "user:",
    "system",
    "system:",
    "<assistant>",
    "<user>",
    "<system>",
    "<|assistant|>",
    "<|assistant|>:",
    "<|user|>",
    "<|user|>:",
    "<|system|>",
    "<|system|>:",
    "<｜assistant｜>",
    "<｜assistant｜>:",
    "<｜user｜>",
    "<｜user｜>:",
    "<｜system｜>",
    "<｜system｜>:",
}
_OUTER_FENCED_CODE_BLOCK_RE = re.compile(
    r"^(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n(?P<body>.*)\n(?P=fence)$",
    flags=re.DOTALL,
)


def _strip_leading_chat_scaffold(text: str, *, prompts: Sequence[str]) -> str:
    """Remove echoed prompts and leading chat role markers from model output."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    normalized_prompts = tuple(prompt.strip() for prompt in prompts if prompt and prompt.strip())
    for _ in range(8):
        previous = cleaned
        lowered = cleaned.casefold()
        stripped_prompt = False
        for prompt in normalized_prompts:
            if lowered.startswith(prompt.casefold()):
                cleaned = cleaned[len(prompt) :].lstrip()
                stripped_prompt = True
                break
        if stripped_prompt:
            continue

        lines = cleaned.splitlines()
        if not lines:
            return ""
        first_line = lines[0].strip()
        if first_line.casefold() in _CHAT_ROLE_PREFIXES:
            cleaned = "\n".join(lines[1:]).lstrip()
            continue
        if re.fullmatch(r"<\|?(?:assistant|user|system)\|?>", first_line, flags=re.IGNORECASE):
            cleaned = "\n".join(lines[1:]).lstrip()
            continue
        if cleaned == previous:
            break
    return cleaned.strip()


def _strip_outer_fenced_code_block(text: str) -> str:
    """Unwrap a single outer fenced code block while preserving its inner content."""
    cleaned = text.strip()
    match = _OUTER_FENCED_CODE_BLOCK_RE.fullmatch(cleaned)
    if match is None:
        return cleaned
    return match.group("body").strip()


def olmocr_image_preprocessor(image: Image.Image) -> Image.Image:
    """Resize an image to olmOCR's expected 1288px longest side and normalize to RGB."""
    return ensure_rgb(
        resize_image_to_fit(
            image,
            OLMOCR_TARGET_LONGEST_IMAGE_DIM,
            OLMOCR_TARGET_LONGEST_IMAGE_DIM,
        )
    )


def olmocr_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Extract plain text and metadata from olmOCR YAML/markdown output."""
    return parse_olmocr_response(text)


def lfm2_5_vl_text_postprocessor(text: str) -> str:
    """Strip Liquid LFM2.5-VL chat scaffold and OCR wrapper tags."""
    prompt = getattr(LFM2_5_VL_1_6B_OCR_TEMPLATE, "user_prompt", None)
    cleaned = _strip_leading_chat_scaffold(text, prompts=[prompt] if isinstance(prompt, str) else [])
    return strip_ocr_output_tag(cleaned, output_tag=DEFAULT_OCR_OUTPUT_TAG)


def infinity_parser_7b_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Normalize Infinity-Parser markdown output to plain text and preserve raw markdown."""
    cleaned = _strip_leading_chat_scaffold(
        text,
        prompts=[
            INFINITY_PARSER_7B_OCR_PROMPT,
            INFINITY_PARSER_7B_SYSTEM_PROMPT,
        ],
    )
    raw_markdown = _strip_outer_fenced_code_block(cleaned)
    return strip_rich_ocr_markup_to_plain_text(raw_markdown), {
        "raw_markdown": raw_markdown,
    }


def deepseek_ocr_2_text_postprocessor(text: str) -> str:
    """Strip DeepSeek OCR 2 prompt echoes, chat scaffold, and trailing stop tokens."""
    cleaned = text.strip()
    stop_token = "<｜end▁of▁sentence｜>"
    while cleaned.endswith(stop_token):
        cleaned = cleaned[: -len(stop_token)].rstrip()
    cleaned = _strip_leading_chat_scaffold(
        cleaned,
        prompts=[
            f"<image>\n{DEEPSEEK_OCR_2_OCR_PROMPT}",
            DEEPSEEK_OCR_2_OCR_PROMPT,
        ],
    )
    return cleaned.strip()


def paddleocr_vl_text_postprocessor(text: str) -> str:
    """Strip PaddleOCR-VL prompt echoes and leading chat scaffold from OCR output."""
    return _strip_leading_chat_scaffold(text, prompts=[PADDLEOCR_VL_1_5_OCR_PROMPT])


def chandra_image_preprocessor(image: Image.Image) -> Image.Image:
    """Resize an image using Chandra OCR 2's pixel-budget and 28px-grid scaling."""
    width, height = image.size
    if width <= 0 or height <= 0:
        return ensure_rgb(image)

    max_pixels = CHANDRA_MAX_IMAGE_SIZE[0] * CHANDRA_MAX_IMAGE_SIZE[1]
    min_pixels = CHANDRA_MIN_IMAGE_SIZE[0] * CHANDRA_MIN_IMAGE_SIZE[1]
    current_pixels = width * height
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    original_aspect_ratio = width / height
    width_blocks = max(1, round((width * scale) / CHANDRA_IMAGE_GRID_SIZE))
    height_blocks = max(1, round((height * scale) / CHANDRA_IMAGE_GRID_SIZE))

    while (width_blocks * height_blocks * CHANDRA_IMAGE_GRID_SIZE**2) > max_pixels:
        if width_blocks == 1 and height_blocks == 1:
            break
        if width_blocks == 1:
            height_blocks -= 1
            continue
        if height_blocks == 1:
            width_blocks -= 1
            continue

        width_loss = abs(((width_blocks - 1) / height_blocks) - original_aspect_ratio)
        height_loss = abs((width_blocks / (height_blocks - 1)) - original_aspect_ratio)
        if width_loss < height_loss:
            width_blocks -= 1
        else:
            height_blocks -= 1

    new_size = (
        width_blocks * CHANDRA_IMAGE_GRID_SIZE,
        height_blocks * CHANDRA_IMAGE_GRID_SIZE,
    )
    if new_size == (width, height):
        return ensure_rgb(image)
    return ensure_rgb(image.resize(new_size, resample=Image.Resampling.LANCZOS))


def chandra_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Extract plain text and metadata from Chandra OCR 2 HTML-layout output."""
    return parse_chandra_response(text)


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
            raise ValueError(f"Unknown OCR profile '{profile}'.") from exc

    if model_id is not None and model_id in registry:
        return registry[model_id]
    return registry["default"]


__all__ = [
    "AzureDocumentIntelligenceOptions",
    "DEFAULT_OCR_MAX_TOKENS",
    "chandra_image_preprocessor",
    "chandra_ocr_2_profile",
    "chandra_text_postprocessor",
    "deepseek_ocr_2_profile",
    "deepseek_ocr_2_text_postprocessor",
    "default_ocr_image_preprocessor",
    "default_ocr_profile",
    "default_ocr_text_postprocessor",
    "HuggingFaceOptions",
    "infinity_parser_7b_profile",
    "infinity_parser_7b_text_postprocessor",
    "identity_text_postprocessor",
    "lfm2_5_vl_text_postprocessor",
    "lfm2_5_vl_1_6b_profile",
    "ImagePreprocessor",
    "LiteLLMTransportConfig",
    "MistralOCRModel",
    "MISTRAL_OCR_MODEL_IDS",
    "MistralOptions",
    "mineru2_5_2509_1_2b_profile",
    "olmocr_image_preprocessor",
    "olmocr_text_postprocessor",
    "paddleocr_vl_1_5_profile",
    "paddleocr_vl_text_postprocessor",
    "OCRBackendSpec",
    "OCRModelProfile",
    "OCRProvider",
    "OpenAICompatibleOptions",
    "resolve_ocr_profile",
    "TextPostprocessor",
    "validate_mistral_ocr_model",
    "VisionInputBuilder",
]

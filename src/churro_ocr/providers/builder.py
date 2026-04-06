"""Public OCR backend builder."""

from __future__ import annotations

from churro_ocr._internal.litellm import LiteLLMTransport
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRBackend
from churro_ocr.providers.hf import (
    ChandraOCR2OCRBackend,
    DotsOCR15OCRBackend,
    HuggingFaceVisionOCRBackend,
    LFM25VLOCRBackend,
    _default_dots_ocr_1_5_model_kwargs,
)
from churro_ocr.providers.ocr import (
    AzureDocumentIntelligenceOCRBackend,
    LiteLLMVisionOCRBackend,
    MistralOCRBackend,
    OpenAICompatibleOCRBackend,
)
from churro_ocr.providers.specs import (
    AzureDocumentIntelligenceOptions,
    HuggingFaceOptions,
    LiteLLMTransportConfig,
    MistralOptions,
    OCRBackendSpec,
    OCRModelProfile,
    OpenAICompatibleOptions,
    resolve_ocr_profile,
    validate_mistral_ocr_model,
)


def _merge_mapping(
    base: dict[str, object],
    override: dict[str, object],
) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **value}
            continue
        merged[key] = value
    return merged


def _merge_transport_config(
    base: LiteLLMTransportConfig,
    override: LiteLLMTransportConfig | None,
) -> LiteLLMTransportConfig:
    if override is None:
        return LiteLLMTransportConfig(
            api_base=base.api_base,
            api_key=base.api_key,
            api_version=base.api_version,
            image_detail=base.image_detail,
            completion_kwargs=dict(base.completion_kwargs),
            cache_dir=base.cache_dir,
        )
    return LiteLLMTransportConfig(
        api_base=override.api_base if override.api_base is not None else base.api_base,
        api_key=override.api_key if override.api_key is not None else base.api_key,
        api_version=override.api_version if override.api_version is not None else base.api_version,
        image_detail=override.image_detail if override.image_detail is not None else base.image_detail,
        completion_kwargs=_merge_mapping(base.completion_kwargs, override.completion_kwargs),
        cache_dir=override.cache_dir if override.cache_dir is not None else base.cache_dir,
    )


def _merge_huggingface_options(
    base: HuggingFaceOptions,
    override: HuggingFaceOptions | None,
) -> HuggingFaceOptions:
    if override is None:
        return HuggingFaceOptions(
            trust_remote_code=base.trust_remote_code,
            processor_kwargs=dict(base.processor_kwargs),
            model_kwargs=dict(base.model_kwargs),
            generation_kwargs=dict(base.generation_kwargs),
            vision_input_builder=base.vision_input_builder,
            backend_variant=base.backend_variant,
        )
    return HuggingFaceOptions(
        trust_remote_code=(
            override.trust_remote_code if override.trust_remote_code is not None else base.trust_remote_code
        ),
        processor_kwargs=_merge_mapping(base.processor_kwargs, override.processor_kwargs),
        model_kwargs=_merge_mapping(base.model_kwargs, override.model_kwargs),
        generation_kwargs=_merge_mapping(base.generation_kwargs, override.generation_kwargs),
        vision_input_builder=override.vision_input_builder or base.vision_input_builder,
        backend_variant=override.backend_variant or base.backend_variant,
    )


def _merge_openai_options(
    override: OpenAICompatibleOptions | None,
) -> OpenAICompatibleOptions:
    if override is None:
        return OpenAICompatibleOptions()
    return OpenAICompatibleOptions(model_prefix=override.model_prefix)


def _ensure_options_type[T](options: object | None, expected: type[T], *, provider: str) -> T | None:
    if options is None:
        return None
    if not isinstance(options, expected):
        raise ConfigurationError(
            f"OCR provider '{provider}' requires options of type {expected.__name__}, "
            f"got {type(options).__name__}."
        )
    return options


def _resolve_model_name(profile: OCRModelProfile, model: str | None, *, fallback: str) -> str:
    if profile.display_name is not None:
        return profile.display_name
    if model is not None:
        return model
    return fallback


def _build_litellm_backend(spec: OCRBackendSpec, profile: OCRModelProfile) -> OCRBackend:
    if spec.model is None:
        raise ConfigurationError("OCR provider 'litellm' requires `model`.")
    transport_config = _merge_transport_config(profile.transport, spec.transport)
    return LiteLLMVisionOCRBackend(
        model=spec.model,
        template=profile.template,
        model_name=_resolve_model_name(profile, spec.model, fallback=spec.model),
        transport=LiteLLMTransport(transport_config),
        image_preprocessor=profile.image_preprocessor,
        text_postprocessor=profile.text_postprocessor,
    )


def _build_openai_compatible_backend(spec: OCRBackendSpec, profile: OCRModelProfile) -> OCRBackend:
    if spec.model is None:
        raise ConfigurationError("OCR provider 'openai-compatible' requires `model`.")
    options = _merge_openai_options(
        _ensure_options_type(spec.options, OpenAICompatibleOptions, provider=spec.provider)
    )
    transport_config = _merge_transport_config(profile.transport, spec.transport)
    if not transport_config.api_base:
        raise ConfigurationError("OCR provider 'openai-compatible' requires `transport.api_base`.")
    return OpenAICompatibleOCRBackend(
        model=spec.model,
        model_prefix=options.model_prefix or "openai",
        model_name=_resolve_model_name(profile, spec.model, fallback=spec.model),
        template=profile.template,
        transport=LiteLLMTransport(transport_config),
        image_preprocessor=profile.image_preprocessor,
        text_postprocessor=profile.text_postprocessor,
    )


def _build_huggingface_backend(spec: OCRBackendSpec, profile: OCRModelProfile) -> OCRBackend:
    if spec.model is None:
        raise ConfigurationError("OCR provider 'hf' requires `model`.")
    options = _merge_huggingface_options(
        profile.huggingface,
        _ensure_options_type(spec.options, HuggingFaceOptions, provider=spec.provider),
    )
    backend_cls: type[HuggingFaceVisionOCRBackend] = HuggingFaceVisionOCRBackend
    model_kwargs = dict(options.model_kwargs)
    if options.backend_variant == "dots-ocr-1.5":
        backend_cls = DotsOCR15OCRBackend
        model_kwargs = _merge_mapping(_default_dots_ocr_1_5_model_kwargs(), model_kwargs)
    elif options.backend_variant == "chandra-ocr-2":
        backend_cls = ChandraOCR2OCRBackend
    elif options.backend_variant == "lfm2.5-vl":
        backend_cls = LFM25VLOCRBackend
    return backend_cls(
        model_id=spec.model,
        template=profile.template,
        model_name=_resolve_model_name(profile, spec.model, fallback=spec.model),
        trust_remote_code=bool(options.trust_remote_code),
        processor_kwargs=dict(options.processor_kwargs),
        model_kwargs=model_kwargs,
        generation_kwargs=dict(options.generation_kwargs),
        vision_input_builder=options.vision_input_builder,
        image_preprocessor=profile.image_preprocessor,
        text_postprocessor=profile.text_postprocessor,
    )


def _build_azure_backend(spec: OCRBackendSpec, profile: OCRModelProfile) -> OCRBackend:
    options = _ensure_options_type(spec.options, AzureDocumentIntelligenceOptions, provider=spec.provider)
    if options is None or not options.endpoint or not options.api_key:
        raise ConfigurationError(
            "OCR provider 'azure' requires AzureDocumentIntelligenceOptions(endpoint=..., api_key=...)."
        )
    model_id = spec.model or "prebuilt-layout"
    return AzureDocumentIntelligenceOCRBackend(
        endpoint=options.endpoint,
        api_key=options.api_key,
        model_id=model_id,
        model_name=_resolve_model_name(profile, spec.model, fallback=model_id),
        image_preprocessor=profile.image_preprocessor,
        text_postprocessor=profile.text_postprocessor,
    )


def _build_mistral_backend(spec: OCRBackendSpec, profile: OCRModelProfile) -> OCRBackend:
    options = _ensure_options_type(spec.options, MistralOptions, provider=spec.provider)
    if options is None or not options.api_key:
        raise ConfigurationError("OCR provider 'mistral' requires MistralOptions(api_key=...).")
    model = validate_mistral_ocr_model(spec.model)
    return MistralOCRBackend(
        api_key=options.api_key,
        model=model,
        model_name=_resolve_model_name(profile, spec.model, fallback=model),
        image_preprocessor=profile.image_preprocessor,
        text_postprocessor=profile.text_postprocessor,
    )


def build_ocr_backend(spec: OCRBackendSpec) -> OCRBackend:
    """Build an OCR backend from a declarative spec.

    :param spec: Declarative backend specification.
    :returns: Configured OCR backend ready for use with ``OCRClient`` or
        ``DocumentOCRPipeline``.
    :raises ConfigurationError: If the provider is unsupported or required
        provider-specific configuration is missing.
    """
    profile = resolve_ocr_profile(model_id=spec.model, profile=spec.profile)
    if spec.provider == "litellm":
        return _build_litellm_backend(spec, profile)
    if spec.provider == "openai-compatible":
        return _build_openai_compatible_backend(spec, profile)
    if spec.provider == "hf":
        return _build_huggingface_backend(spec, profile)
    if spec.provider == "azure":
        return _build_azure_backend(spec, profile)
    if spec.provider == "mistral":
        return _build_mistral_backend(spec, profile)
    raise ConfigurationError(f"Unsupported OCR provider '{spec.provider}'.")


__all__ = ["build_ocr_backend"]

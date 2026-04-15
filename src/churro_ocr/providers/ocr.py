"""Built-in OCR backends."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from io import BytesIO
from threading import Lock
from typing import TYPE_CHECKING, Protocol, cast

from churro_ocr._internal.image import ensure_rgb, image_to_base64
from churro_ocr._internal.install import install_command_hint
from churro_ocr._internal.litellm import LiteLLMTransport
from churro_ocr._internal.prompt_logging import log_prompt_payload_once
from churro_ocr._internal.retry import retry_api_call
from churro_ocr.errors import ConfigurationError, ProviderError
from churro_ocr.ocr import OCRBackend, OCRResult
from churro_ocr.page_detection import DocumentPage
from churro_ocr.providers._mineru25 import (
    MinerU25PipelineHelper,
    MinerU25SamplingParams,
)
from churro_ocr.providers._shared import build_ocr_result, preprocess_backend_page
from churro_ocr.providers.specs import (
    DEFAULT_OCR_MAX_TOKENS,
    ImagePreprocessor,
    LiteLLMTransportConfig,
    TextPostprocessor,
    default_ocr_image_preprocessor,
    identity_text_postprocessor,
    validate_mistral_ocr_model,
)
from churro_ocr.templates import (
    DEFAULT_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_FORMULA_PROMPT,
    MINERU2_5_2509_1_2B_FORMULA_TEMPLATE,
    MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
    MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE,
    MINERU2_5_2509_1_2B_LAYOUT_PROMPT,
    MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE,
    MINERU2_5_2509_1_2B_MODEL_ID,
    MINERU2_5_2509_1_2B_OCR_PROMPT,
    MINERU2_5_2509_1_2B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    MINERU2_5_2509_1_2B_TABLE_PROMPT,
    MINERU2_5_2509_1_2B_TABLE_TEMPLATE,
    OCRPromptTemplateLike,
    build_ocr_conversation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image


class _AzureAnalyzeResultLike(Protocol):
    content: object


class _AzurePollerLike(Protocol):
    async def result(self) -> _AzureAnalyzeResultLike: ...


class _AzureDocumentIntelligenceClientLike(Protocol):
    async def begin_analyze_document(
        self,
        *,
        model_id: str,
        body: BytesIO,
        content_type: str,
    ) -> _AzurePollerLike: ...


class _MistralOCRPageLike(Protocol):
    markdown: str


class _MistralOCRResponseLike(Protocol):
    pages: Sequence[_MistralOCRPageLike] | None


class _MistralOCRNamespaceLike(Protocol):
    async def process_async(
        self,
        *,
        model: str,
        document: dict[str, str],
    ) -> _MistralOCRResponseLike: ...


class _MistralClientLike(Protocol):
    ocr: _MistralOCRNamespaceLike


_MISTRAL_REQUEST_TIMEOUT_SECONDS = 60.0


def _configuration_error(message: str) -> ConfigurationError:
    return ConfigurationError(message)


def _provider_error(message: str) -> ProviderError:
    return ProviderError(message)


def _with_default_ocr_completion_kwargs(config: LiteLLMTransportConfig) -> LiteLLMTransportConfig:
    completion_kwargs: dict[str, object] = {"max_tokens": DEFAULT_OCR_MAX_TOKENS}
    completion_kwargs.update(config.completion_kwargs)
    if completion_kwargs == config.completion_kwargs:
        return config
    return LiteLLMTransportConfig(
        api_base=config.api_base,
        api_key=config.api_key,
        api_version=config.api_version,
        image_detail=config.image_detail,
        completion_kwargs=completion_kwargs,
        cache_dir=config.cache_dir,
    )


@dataclass(slots=True)
class LiteLLMVisionOCRBackend(OCRBackend):
    """OCR backend for any LiteLLM-supported multimodal provider.

    :param model: LiteLLM model identifier to call.
    :param template: Prompt template used to render OCR input.
    :param model_name: Optional human-readable model name for result metadata.
    :param transport: LiteLLM transport used for request execution.
    :param image_preprocessor: Image preprocessor applied before OCR.
    :param text_postprocessor: Text postprocessor applied after OCR.
    :param provider_name: Provider identifier written into OCR results.
    """

    model: str
    template: OCRPromptTemplateLike = DEFAULT_OCR_TEMPLATE
    model_name: str | None = None
    transport: LiteLLMTransport = field(default_factory=LiteLLMTransport)
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = identity_text_postprocessor
    provider_name: str = "litellm"
    _has_logged_prompt: bool = field(default=False, init=False, repr=False)
    _prompt_log_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Apply default OCR completion settings to the shared transport."""
        config = _with_default_ocr_completion_kwargs(self.transport.config)
        if config != self.transport.config:
            object.__setattr__(self.transport, "_config", config)

    async def ocr(self, page: DocumentPage) -> OCRResult:
        """Run OCR for one page through LiteLLM.

        :param page: Page to transcribe.
        :returns: Provider-agnostic OCR result.
        """
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        conversation = build_ocr_conversation(self.template, prepared_page)
        messages = await asyncio.to_thread(
            self.transport.prepare_messages_from_conversation,
            conversation,
        )
        log_prompt_payload_once(
            payload={"messages": messages},
            provider_name=self.provider_name,
            has_logged=lambda: self._has_logged_prompt,
            lock=self._prompt_log_lock,
            set_logged=lambda: setattr(self, "_has_logged_prompt", True),
        )
        text = await self.transport.complete_text(
            model=self.model,
            messages=messages,
            allow_empty=True,
        )
        return build_ocr_result(
            text,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model,
            text_postprocessor=self.text_postprocessor,
        )


@dataclass(slots=True, init=False)
class OpenAICompatibleOCRBackend(LiteLLMVisionOCRBackend):
    """OCR backend for OpenAI-compatible servers."""

    def __init__(
        self,
        *,
        model: str,
        transport: LiteLLMTransport | None = None,
        model_prefix: str = "openai",
        template: OCRPromptTemplateLike = DEFAULT_OCR_TEMPLATE,
        image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor,
        text_postprocessor: TextPostprocessor = identity_text_postprocessor,
        model_name: str | None = None,
    ) -> None:
        """Create an OCR backend for an OpenAI-compatible server.

        :param model: Model identifier exposed by the target server.
        :param transport: LiteLLM transport used for request execution.
        :param model_prefix: LiteLLM provider prefix prepended to ``model``.
        :param template: Prompt template used to render OCR input.
        :param image_preprocessor: Image preprocessor applied before OCR.
        :param text_postprocessor: Text postprocessor applied after OCR.
        :param model_name: Optional human-readable model name for result metadata.
        """
        LiteLLMVisionOCRBackend.__init__(
            self,
            model=f"{model_prefix}/{model}",
            template=template,
            model_name=model_name or model,
            transport=transport or LiteLLMTransport(),
            image_preprocessor=image_preprocessor,
            text_postprocessor=text_postprocessor,
            provider_name="openai-compatible",
        )


def _clone_transport_config(
    config: LiteLLMTransportConfig,
    *,
    completion_kwargs: dict[str, object],
) -> LiteLLMTransportConfig:
    return LiteLLMTransportConfig(
        api_base=config.api_base,
        api_key=config.api_key,
        api_version=config.api_version,
        image_detail=config.image_detail,
        completion_kwargs=completion_kwargs,
        cache_dir=config.cache_dir,
    )


def _mineru25_completion_kwargs(sampling: MinerU25SamplingParams) -> dict[str, object]:
    kwargs: dict[str, object] = {"skip_special_tokens": False}
    if sampling.temperature is not None:
        kwargs["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        kwargs["top_p"] = sampling.top_p
    if sampling.top_k is not None:
        kwargs["top_k"] = sampling.top_k
    if sampling.presence_penalty is not None:
        kwargs["presence_penalty"] = sampling.presence_penalty
    if sampling.frequency_penalty is not None:
        kwargs["frequency_penalty"] = sampling.frequency_penalty
    if sampling.repetition_penalty is not None:
        kwargs["repetition_penalty"] = sampling.repetition_penalty
    if sampling.no_repeat_ngram_size is not None:
        kwargs["vllm_xargs"] = {
            "no_repeat_ngram_size": sampling.no_repeat_ngram_size,
            "debug": False,
        }
    if sampling.max_new_tokens is not None:
        kwargs["max_completion_tokens"] = sampling.max_new_tokens
    return kwargs


def _merge_completion_kwargs(
    step_defaults: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    merged = dict(step_defaults)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **value}
            continue
        merged[key] = value
    return merged


class MinerU25OpenAICompatibleOCRBackend(OpenAICompatibleOCRBackend):
    """Two-step MinerU2.5 OCR backend for OpenAI-compatible servers such as vLLM."""

    __slots__ = (
        "_helper",
        "formula_template",
        "image_analysis_template",
        "layout_template",
        "table_template",
    )

    layout_template: OCRPromptTemplateLike
    table_template: OCRPromptTemplateLike
    formula_template: OCRPromptTemplateLike
    image_analysis_template: OCRPromptTemplateLike
    _helper: MinerU25PipelineHelper

    def __init__(
        self,
        *,
        model: str,
        transport: LiteLLMTransport | None = None,
        model_prefix: str = "openai",
        template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_OCR_TEMPLATE,
        layout_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE,
        table_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_TABLE_TEMPLATE,
        formula_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_FORMULA_TEMPLATE,
        image_analysis_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE,
        image_preprocessor: ImagePreprocessor = ensure_rgb,
        text_postprocessor: TextPostprocessor = identity_text_postprocessor,
        model_name: str | None = "MinerU2.5-2509-1.2B",
    ) -> None:
        """Create a two-step MinerU2.5 OCR backend for an OpenAI-compatible server."""
        super().__init__(
            model=model,
            transport=transport,
            model_prefix=model_prefix,
            template=template,
            image_preprocessor=image_preprocessor,
            text_postprocessor=text_postprocessor,
            model_name=model_name or model,
        )
        self.layout_template = layout_template
        self.table_template = table_template
        self.formula_template = formula_template
        self.image_analysis_template = image_analysis_template
        self._helper = MinerU25PipelineHelper(
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

    def __post_init__(self) -> None:
        """Skip the generic max-token injection for the MinerU2.5 two-step pipeline."""

    def _template_for_step(self, step_key: str) -> OCRPromptTemplateLike:
        if step_key == "[layout]":
            return self.layout_template
        if step_key == "table":
            return self.table_template
        if step_key == "equation":
            return self.formula_template
        if step_key in {"image", "chart"}:
            return self.image_analysis_template
        return self.template

    def _transport_for_step(self, sampling: MinerU25SamplingParams) -> LiteLLMTransport:
        config = self.transport.config
        return LiteLLMTransport(
            _clone_transport_config(
                config,
                completion_kwargs=_merge_completion_kwargs(
                    _mineru25_completion_kwargs(sampling),
                    config.completion_kwargs,
                ),
            )
        )

    async def _infer_step(
        self,
        image: Image.Image,
        step_key: str,
        sampling: MinerU25SamplingParams,
    ) -> str:
        conversation = build_ocr_conversation(
            self._template_for_step(step_key),
            DocumentPage.from_image(image),
        )
        step_transport = self._transport_for_step(sampling)
        messages = await asyncio.to_thread(
            step_transport.prepare_messages_from_conversation,
            conversation,
        )
        log_prompt_payload_once(
            payload={
                "step_key": step_key,
                "conversation": conversation,
                "messages": messages,
            },
            provider_name=self.provider_name,
            has_logged=lambda: self._has_logged_prompt,
            lock=self._prompt_log_lock,
            set_logged=lambda: setattr(self, "_has_logged_prompt", True),
        )
        text = await step_transport.complete_text(
            model=self.model,
            messages=messages,
            allow_empty=True,
        )
        return self._helper.clean_response(text, step_key=step_key)

    async def ocr(self, page: DocumentPage) -> OCRResult:
        """Run the MinerU2.5 two-step OCR pipeline for one page."""
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        markdown, blocks, metrics = await self._helper.arun_two_step(
            prepared_page.image,
            infer_step=self._infer_step,
        )
        return build_ocr_result(
            markdown,
            provider_name=self.provider_name,
            model_name=self.model_name or MINERU2_5_2509_1_2B_MODEL_ID,
            text_postprocessor=self.text_postprocessor,
            metadata={
                "output_format": "markdown",
                "blocks": [dict(block) for block in blocks],
                "pipeline_metrics": metrics,
            },
        )

    async def ocr_batch(self, pages: list[DocumentPage]) -> list[OCRResult]:
        """Run the MinerU2.5 two-step OCR pipeline for multiple pages."""
        return [await self.ocr(page) for page in pages]


@dataclass(slots=True)
class AzureDocumentIntelligenceOCRBackend(OCRBackend):
    """Azure Document Intelligence OCR backend.

    :param endpoint: Azure Document Intelligence endpoint URL.
    :param api_key: Azure API key for the configured resource.
    :param model_id: Azure model ID used for OCR.
    :param model_name: Optional human-readable model name for result metadata.
    :param image_preprocessor: Image preprocessor applied before OCR.
    :param text_postprocessor: Text postprocessor applied after OCR.
    """

    endpoint: str
    api_key: str
    model_id: str = "prebuilt-layout"
    model_name: str | None = None
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = identity_text_postprocessor
    _client: _AzureDocumentIntelligenceClientLike | None = field(default=None, init=False, repr=False)
    _client_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _has_logged_prompt: bool = field(default=False, init=False, repr=False)
    _prompt_log_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    async def _get_client(self) -> _AzureDocumentIntelligenceClientLike:
        client = self._client
        if client is not None:
            return client

        async with self._client_lock:
            client = self._client
            if client is not None:
                return client
            try:
                from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError as exc:  # pragma: no cover - optional extra path
                message = f"Azure OCR requires the `azure` runtime. {install_command_hint('azure')}"
                raise _configuration_error(message) from exc

            client = cast(
                "_AzureDocumentIntelligenceClientLike",
                DocumentIntelligenceClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key),
                ),
            )
            self._client = client
            return client

    async def ocr(self, page: DocumentPage) -> OCRResult:
        """Run OCR for one page through Azure Document Intelligence.

        :param page: Page to transcribe.
        :returns: Provider-agnostic OCR result.
        :raises ConfigurationError: If the optional Azure dependency is not installed.
        :raises ProviderError: If Azure returns no text content.
        """
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        buffer = (await asyncio.to_thread(image_to_base64, prepared_page.image, "JPEG"))[1]
        image_bytes = base64.b64decode(buffer)
        log_prompt_payload_once(
            payload={
                "model_id": self.model_id,
                "content_type": "application/octet-stream",
                "document": {"type": "image", "image": prepared_page.image},
            },
            provider_name="azure-document-intelligence",
            has_logged=lambda: self._has_logged_prompt,
            lock=self._prompt_log_lock,
            set_logged=lambda: setattr(self, "_has_logged_prompt", True),
        )
        client = await self._get_client()

        async def _analyze_document() -> _AzureAnalyzeResultLike:
            poller = await client.begin_analyze_document(
                model_id=self.model_id,
                body=BytesIO(image_bytes),
                content_type="application/octet-stream",
            )
            return await poller.result()

        result = await retry_api_call(
            _analyze_document,
            operation_name="Azure OCR request",
            context=f"for model {self.model_id}",
        )
        if not isinstance(result.content, str):
            message = "Azure Document Intelligence returned no OCR text."
            raise _provider_error(message)
        return build_ocr_result(
            result.content,
            provider_name="azure-document-intelligence",
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
        )


@dataclass(slots=True)
class MistralOCRBackend(OCRBackend):
    """Mistral OCR backend.

    :param api_key: Mistral API key used for OCR requests.
    :param model: Pinned Mistral OCR model identifier.
    :param model_name: Optional human-readable model name for result metadata.
    :param image_preprocessor: Image preprocessor applied before OCR.
    :param text_postprocessor: Text postprocessor applied after OCR.
    """

    api_key: str
    model: str
    model_name: str | None = None
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = identity_text_postprocessor
    _client: _MistralClientLike | None = field(default=None, init=False, repr=False)
    _client_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _has_logged_prompt: bool = field(default=False, init=False, repr=False)
    _prompt_log_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject unsupported Mistral OCR aliases and unpinned model ids."""
        validate_mistral_ocr_model(self.model, context="Mistral OCR backend")

    async def _get_client(self) -> _MistralClientLike:
        client = self._client
        if client is not None:
            return client

        async with self._client_lock:
            client = self._client
            if client is not None:
                return client
            try:
                from mistralai import Mistral
            except ImportError as exc:  # pragma: no cover - optional extra path
                message = f"Mistral OCR requires the `mistral` runtime. {install_command_hint('mistral')}"
                raise _configuration_error(message) from exc

            client = cast("_MistralClientLike", Mistral(api_key=self.api_key))
            self._client = client
            return client

    async def ocr(self, page: DocumentPage) -> OCRResult:
        """Run OCR for one page through Mistral OCR.

        :param page: Page to transcribe.
        :returns: Provider-agnostic OCR result.
        :raises ConfigurationError: If the optional Mistral dependency is not installed.
        :raises ProviderError: If the response does not contain any OCR pages.
        """
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        _, encoded = await asyncio.to_thread(image_to_base64, prepared_page.image, "JPEG")
        image_url = f"data:image/jpeg;base64,{encoded}"
        log_prompt_payload_once(
            payload={
                "model": self.model,
                "document": {"type": "image_url", "image_url": image_url},
            },
            provider_name="mistral",
            has_logged=lambda: self._has_logged_prompt,
            lock=self._prompt_log_lock,
            set_logged=lambda: setattr(self, "_has_logged_prompt", True),
        )
        client = await self._get_client()
        document = {"type": "image_url", "image_url": image_url}

        async def _process_ocr() -> _MistralOCRResponseLike:
            return await asyncio.wait_for(
                client.ocr.process_async(
                    model=self.model,
                    document=document,
                ),
                timeout=_MISTRAL_REQUEST_TIMEOUT_SECONDS,
            )

        response = await retry_api_call(
            _process_ocr,
            operation_name="Mistral OCR request",
            context=f"for model {self.model}",
        )
        if not response.pages:
            message = "Mistral OCR returned no pages."
            raise _provider_error(message)
        return build_ocr_result(
            response.pages[0].markdown,
            provider_name="mistral",
            model_name=self.model_name or self.model,
            text_postprocessor=self.text_postprocessor,
        )

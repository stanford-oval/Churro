"""Built-in OCR backends."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from io import BytesIO
from threading import Lock
from typing import Any

from churro_ocr._internal.image import image_to_base64
from churro_ocr._internal.litellm import LiteLLMTransport
from churro_ocr._internal.prompt_logging import log_prompt_payload_once
from churro_ocr.errors import ConfigurationError, ProviderError
from churro_ocr.ocr import OCRBackend, OCRResult
from churro_ocr.page_detection import DocumentPage
from churro_ocr.providers._shared import build_ocr_result, preprocess_backend_page
from churro_ocr.providers.specs import (
    DEFAULT_OCR_MAX_TOKENS,
    ImagePreprocessor,
    LiteLLMTransportConfig,
    TextPostprocessor,
    default_ocr_image_preprocessor,
    identity_text_postprocessor,
)
from churro_ocr.templates import (
    DEFAULT_OCR_TEMPLATE,
    OCRPromptTemplateLike,
    build_ocr_conversation,
)


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
    _client: Any | None = field(default=None, init=False, repr=False)
    _client_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _has_logged_prompt: bool = field(default=False, init=False, repr=False)
    _prompt_log_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    async def _get_client(self) -> Any:
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
                raise ConfigurationError(
                    "Azure OCR requires the 'azure' extra. Install with `pip install \"churro-ocr[azure]\"`."
                ) from exc

            client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
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
        poller = await client.begin_analyze_document(
            model_id=self.model_id,
            body=BytesIO(image_bytes),
            content_type="application/octet-stream",
        )
        result = await poller.result()
        if not isinstance(result.content, str):
            raise ProviderError("Azure Document Intelligence returned no OCR text.")
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
    :param model: Mistral OCR model identifier.
    :param model_name: Optional human-readable model name for result metadata.
    :param image_preprocessor: Image preprocessor applied before OCR.
    :param text_postprocessor: Text postprocessor applied after OCR.
    """

    api_key: str
    model: str = "mistral-ocr-latest"
    model_name: str | None = None
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = identity_text_postprocessor
    _client: Any | None = field(default=None, init=False, repr=False)
    _client_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _has_logged_prompt: bool = field(default=False, init=False, repr=False)
    _prompt_log_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    async def _get_client(self) -> Any:
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
                raise ConfigurationError(
                    "Mistral OCR requires the 'mistral' extra. "
                    'Install with `pip install "churro-ocr[mistral]"`.'
                ) from exc

            client = Mistral(api_key=self.api_key)
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
        response = await client.ocr.process_async(
            model=self.model,
            document={"type": "image_url", "image_url": image_url},
        )
        if not response.pages:
            raise ProviderError("Mistral OCR returned no pages.")
        return build_ocr_result(
            response.pages[0].markdown,
            provider_name="mistral",
            model_name=self.model_name or self.model,
            text_postprocessor=self.text_postprocessor,
        )

"""vLLM OCR backends."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

from churro_ocr._internal.prompt_logging import log_prompt_payload_once
from churro_ocr.errors import ConfigurationError, ProviderError
from churro_ocr.ocr import OCRBackend, OCRResult
from churro_ocr.page_detection import DocumentPage
from churro_ocr.providers._shared import build_ocr_result, preprocess_backend_page, render_ocr_prompt
from churro_ocr.providers.specs import (
    DEFAULT_OCR_MAX_TOKENS,
    ImagePreprocessor,
    TextPostprocessor,
    default_ocr_image_preprocessor,
    identity_text_postprocessor,
)
from churro_ocr.templates import (
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    OCRPromptTemplateLike,
)


def _load_vllm_processor_cls() -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(
            'vLLM OCR requires transformers. Install with `pip install "churro-ocr[vllm]"`.'
        ) from exc

    return AutoProcessor


def _load_vllm_runtime() -> tuple[Any, Any]:
    try:
        vllm = import_module("vllm")
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(
            'vLLM OCR requires the "vllm" extra. Install with `pip install "churro-ocr[vllm]"`.'
        ) from exc

    vllm_any = cast(Any, vllm)
    return vllm_any.LLM, vllm_any.SamplingParams


@dataclass(slots=True)
class VLLMVisionOCRBackend(OCRBackend):
    """OCR backend for local multimodal models served by vLLM.

    :param model_id: Model identifier served by vLLM.
    :param template: Prompt template used to render OCR input.
    :param model_name: Optional human-readable model name for result metadata.
    :param trust_remote_code: Whether to allow remote model code execution.
    :param processor_kwargs: Extra kwargs passed to processor loading.
    :param llm_kwargs: Extra kwargs passed to the vLLM ``LLM`` constructor.
    :param sampling_kwargs: Extra sampling kwargs passed at inference time.
    :param limit_mm_per_prompt: Multimodal limits passed to vLLM.
    :param image_preprocessor: Image preprocessor applied before OCR.
    :param text_postprocessor: Text postprocessor applied after OCR.
    :param provider_name: Provider identifier written into OCR results.
    """

    model_id: str
    template: OCRPromptTemplateLike
    model_name: str | None = None
    trust_remote_code: bool = False
    processor_kwargs: dict[str, object] = field(default_factory=dict)
    llm_kwargs: dict[str, object] = field(default_factory=dict)
    sampling_kwargs: dict[str, object] = field(default_factory=dict)
    limit_mm_per_prompt: dict[str, int] = field(default_factory=lambda: {"image": 1})
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = identity_text_postprocessor
    provider_name: str = "vllm"
    _processor: object | None = field(default=None, init=False, repr=False)
    _llm: object | None = field(default=None, init=False, repr=False)
    _init_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _has_logged_prompt: bool = field(default=False, init=False, repr=False)
    _prompt_log_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Apply default sampling settings after dataclass initialization."""
        self.sampling_kwargs = {
            "max_tokens": DEFAULT_OCR_MAX_TOKENS,
            **self.sampling_kwargs,
        }

    async def ocr(self, page: DocumentPage) -> OCRResult:
        """Run OCR for one page.

        :param page: Page to transcribe.
        :returns: Provider-agnostic OCR result.
        """
        return (await self.ocr_batch([page]))[0]

    async def ocr_batch(self, pages: list[DocumentPage]) -> list[OCRResult]:
        """Run OCR for multiple pages in one batch.

        :param pages: Pages to transcribe in batch order.
        :returns: OCR results in the same order as ``pages``.
        """
        return await asyncio.to_thread(self._ocr_batch_sync, pages)

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        processor = self._get_processor()
        llm = self._get_llm()
        _, sampling_params_cls = _load_vllm_runtime()
        prompts: list[dict[str, object]] = []

        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            rendered, _ = render_ocr_prompt(
                processor,
                self.template,
                prepared_page,
                add_generation_prompt=True,
            )
            prompt_payload: dict[str, object] = {
                "prompt": rendered,
                "multi_modal_data": {"image": prepared_page.image},
            }
            prompts.append(prompt_payload)
            if not self._has_logged_prompt:
                log_prompt_payload_once(
                    payload={
                        "batch_size": len(pages),
                        "prompt": prompt_payload,
                    },
                    provider_name=self.provider_name,
                    has_logged=lambda: self._has_logged_prompt,
                    lock=self._prompt_log_lock,
                    set_logged=lambda: setattr(self, "_has_logged_prompt", True),
                )

        request_outputs = llm.generate(
            prompts,
            sampling_params_cls(**self.sampling_kwargs),
            use_tqdm=False,
        )
        results: list[OCRResult] = []
        for request_output in request_outputs:
            outputs = getattr(request_output, "outputs", None)
            if not outputs:
                raise ProviderError("vLLM OCR returned no outputs.")
            text = getattr(outputs[0], "text", None)
            if not isinstance(text, str):
                raise ProviderError("vLLM OCR returned a non-text response.")
            results.append(
                build_ocr_result(
                    text,
                    provider_name=self.provider_name,
                    model_name=self.model_name or self.model_id,
                    text_postprocessor=self.text_postprocessor,
                )
            )
        return results

    def _get_processor(self) -> Any:
        if self._processor is None:
            with self._init_lock:
                if self._processor is None:
                    processor_cls = _load_vllm_processor_cls()
                    self._processor = processor_cls.from_pretrained(
                        self.model_id,
                        trust_remote_code=self.trust_remote_code,
                        **self.processor_kwargs,
                    )
        return self._processor

    def _get_llm(self) -> Any:
        if self._llm is None:
            with self._init_lock:
                if self._llm is None:
                    llm_cls, _ = _load_vllm_runtime()
                    self._llm = llm_cls(
                        model=self.model_id,
                        trust_remote_code=self.trust_remote_code,
                        limit_mm_per_prompt=self.limit_mm_per_prompt,
                        **self.llm_kwargs,
                    )
        return self._llm


@dataclass(slots=True)
class DotsOCR15VLLMOCRBackend(VLLMVisionOCRBackend):
    """Preset vLLM OCR backend for ``kristaller486/dots.ocr-1.5``."""

    model_id: str = DOTS_OCR_1_5_MODEL_ID
    template: OCRPromptTemplateLike = DOTS_OCR_1_5_OCR_TEMPLATE
    model_name: str | None = "dots.ocr-1.5"
    trust_remote_code: bool = True


__all__ = [
    "DotsOCR15VLLMOCRBackend",
    "VLLMVisionOCRBackend",
]

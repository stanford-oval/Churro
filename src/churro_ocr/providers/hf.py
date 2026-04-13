"""Hugging Face OCR backends for template-aware multimodal models."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

from churro_ocr._internal.image import ensure_rgb
from churro_ocr._internal.prompt_logging import log_prompt_payload_once
from churro_ocr.ocr import OCRBackend, OCRResult
from churro_ocr.page_detection import DocumentPage
from churro_ocr.providers import _hf_dots as _dots
from churro_ocr.providers import _hf_helpers as _helpers
from churro_ocr.providers import _hf_mineru as _mineru
from churro_ocr.providers import _hf_runtime as _runtime
from churro_ocr.providers._shared import (
    build_ocr_result,
    normalize_media_inputs,
    preprocess_backend_page,
    render_ocr_prompt,
)
from churro_ocr.providers.specs import (
    DEFAULT_OCR_MAX_TOKENS,
    ImagePreprocessor,
    TextPostprocessor,
    VisionInputBuilder,
    default_ocr_image_preprocessor,
    identity_text_postprocessor,
)
from churro_ocr.templates import (
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DEEPSEEK_OCR_2_MODEL_ID,
    DEEPSEEK_OCR_2_OCR_TEMPLATE,
    DOTS_MOCR_MODEL_ID,
    DOTS_MOCR_OCR_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    GLM_OCR_MODEL_ID,
    GLM_OCR_OCR_TEMPLATE,
    LFM2_5_VL_1_6B_MODEL_ID,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_FORMULA_TEMPLATE,
    MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE,
    MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE,
    MINERU2_5_2509_1_2B_MODEL_ID,
    MINERU2_5_2509_1_2B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_TABLE_TEMPLATE,
    PADDLEOCR_VL_1_5_MODEL_ID,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
    OCRConversation,
    OCRPromptTemplateLike,
    build_ocr_conversation,
)

if TYPE_CHECKING:
    from PIL import Image

    from churro_ocr.providers._mineru25 import MinerU25PipelineHelper

_HF_EXTRA_INSTALL_HINT = _runtime._HF_EXTRA_INSTALL_HINT
_HFRuntime = _runtime._HFRuntime
_apply_chat_template = _runtime._apply_chat_template
_call_processor = _runtime._call_processor
_configuration_error = _runtime._configuration_error
_default_chandra_ocr_2_model_kwargs = _helpers._default_chandra_ocr_2_model_kwargs
_default_mineru25_model_kwargs = _helpers._default_mineru25_model_kwargs
_decode_completion_texts = _helpers._decode_completion_texts
_decode_completion_texts_with_options = _helpers._decode_completion_texts_with_options
_deepseek_ocr_2_prompt_from_conversation = _helpers._deepseek_ocr_2_prompt_from_conversation
_generate_with_model = _runtime._generate_with_model
_move_batch_to_model = _helpers._move_batch_to_model
_paddleocr_vl_processor_kwargs = _helpers._paddleocr_vl_processor_kwargs
_provider_error = _runtime._provider_error
_resolve_model_max_length = _helpers._resolve_model_max_length


def _load_torch_module() -> _runtime._TorchModuleLike:
    return _runtime._load_torch_module_with_import(import_module)


def _load_hf_runtime() -> _HFRuntime:
    return _runtime._load_hf_runtime_with_import(import_module)


def _load_hf_causal_runtime() -> _HFRuntime:
    return _runtime._load_hf_causal_runtime_with_import(import_module)


def _load_hf_auto_model_runtime() -> _HFRuntime:
    return _runtime._load_hf_auto_model_runtime_with_import(import_module)


def _load_hf_auto_processor_model_runtime() -> _HFRuntime:
    return _runtime._load_hf_auto_processor_model_runtime_with_import(import_module)


def _ensure_deepseek_ocr_2_cuda_runtime() -> _runtime._TorchModuleLike:
    return _runtime._ensure_deepseek_ocr_2_cuda_runtime_with_import(import_module)


_DOTS_OCR_1_5_LOCAL_DIRNAME = _dots._DOTS_OCR_1_5_LOCAL_DIRNAME
_DOTS_FLASH_ATTN_IMPORT = _dots._DOTS_FLASH_ATTN_IMPORT
_DOTS_FLASH_ATTN_FALLBACK = _dots._DOTS_FLASH_ATTN_FALLBACK
_DOTS_FORCE_BFLOAT16_LINE = _dots._DOTS_FORCE_BFLOAT16_LINE
_DOTS_WEIGHT_DTYPE_LINE = _dots._DOTS_WEIGHT_DTYPE_LINE


def _patch_dots_ocr_vision_module(model_dir: Path) -> None:
    _dots._patch_dots_ocr_vision_module(model_dir)


def _prepare_dots_ocr_model_dir(model_id: str) -> str:
    return _dots._prepare_dots_ocr_model_dir(
        model_id,
        home_dir=Path.home(),
        patch_vision_module=_patch_dots_ocr_vision_module,
        configuration_error=_configuration_error,
        extra_install_hint=_HF_EXTRA_INSTALL_HINT,
    )


def _patch_dots_ocr_prepare_inputs_for_generation(model: object) -> None:
    _dots._patch_dots_ocr_prepare_inputs_for_generation(model)


def _default_dots_ocr_1_5_model_kwargs() -> dict[str, object]:
    return _dots._default_dots_ocr_1_5_model_kwargs(load_torch_module=_load_torch_module)


@dataclass(slots=True)
class HuggingFaceVisionOCRBackend(OCRBackend):
    """OCR backend for local Hugging Face multimodal models with custom templates.

    :param model_id: Hugging Face model identifier.
    :param template: Prompt template used to render OCR input.
    :param model_name: Optional human-readable model name for result metadata.
    :param trust_remote_code: Whether to allow remote model code execution.
    :param processor_kwargs: Extra kwargs passed to processor loading.
    :param model_kwargs: Extra kwargs passed to model loading.
    :param generation_kwargs: Extra generation kwargs passed at inference time.
    :param vision_input_builder: Optional override for building multimodal inputs.
    :param image_preprocessor: Image preprocessor applied before OCR.
    :param text_postprocessor: Text postprocessor applied after OCR.
    :param provider_name: Provider identifier written into OCR results.
    """

    model_id: str
    template: OCRPromptTemplateLike
    model_name: str | None = None
    trust_remote_code: bool = False
    processor_kwargs: dict[str, object] = field(default_factory=dict)
    model_kwargs: dict[str, object] = field(default_factory=dict)
    generation_kwargs: dict[str, object] = field(default_factory=dict)
    vision_input_builder: VisionInputBuilder | None = None
    image_preprocessor: ImagePreprocessor = default_ocr_image_preprocessor
    text_postprocessor: TextPostprocessor = identity_text_postprocessor
    provider_name: str = "huggingface-transformers"
    _processor: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _model_source: str | None = field(default=None, init=False, repr=False)
    _init_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _has_logged_prompt: bool = field(default=False, init=False, repr=False)
    _prompt_log_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Apply default generation settings after dataclass initialization."""
        self.generation_kwargs = {
            "max_new_tokens": DEFAULT_OCR_MAX_TOKENS,
            **self.generation_kwargs,
        }

    async def ocr(self, page: DocumentPage) -> OCRResult:
        """Run OCR for one page.

        :param page: Page to transcribe.
        :returns: Provider-agnostic OCR result.
        """
        return await asyncio.to_thread(self._ocr_sync, page)

    async def ocr_batch(self, pages: list[DocumentPage]) -> list[OCRResult]:
        """Run OCR for multiple pages in one batch.

        :param pages: Pages to transcribe in batch order.
        :returns: OCR results in the same order as ``pages``.
        """
        return await asyncio.to_thread(self._ocr_batch_sync, pages)

    def _ocr_sync(self, page: DocumentPage) -> OCRResult:
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)

        rendered, conversation = render_ocr_prompt(
            processor,
            self.template,
            prepared_page,
            add_generation_prompt=True,
        )
        self._log_prompt_payload(
            rendered_prompt=rendered,
            conversation=conversation,
            batch_size=1,
        )
        image_inputs, video_inputs = self._build_vision_inputs(runtime, conversation)
        batch_kwargs: dict[str, object] = {
            "text": [rendered],
            "images": normalize_media_inputs(image_inputs),
            "return_tensors": "pt",
            "padding": True,
        }
        normalized_video_inputs = normalize_media_inputs(video_inputs)
        if normalized_video_inputs is not None:
            batch_kwargs["videos"] = normalized_video_inputs
        batch = _call_processor(processor, **batch_kwargs)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(
            model,
            **self._generation_inputs(batch),
            **self.generation_kwargs,
        )
        text = _decode_completion_texts(processor, batch, generated_ids)[0]
        return build_ocr_result(
            text,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
        )

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)
        rendered_prompts: list[str] = []
        image_batch: list[object] = []
        video_batch: list[object] = []
        has_videos = False

        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            rendered, conversation = render_ocr_prompt(
                processor,
                self.template,
                prepared_page,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = self._build_vision_inputs(runtime, conversation)
            rendered_prompts.append(rendered)
            image_batch.append(normalize_media_inputs(image_inputs))
            normalized_video_inputs = normalize_media_inputs(video_inputs)
            if normalized_video_inputs is not None:
                has_videos = True
            video_batch.append(normalized_video_inputs)
            if not self._has_logged_prompt:
                self._log_prompt_payload(
                    rendered_prompt=rendered,
                    conversation=conversation,
                    batch_size=len(pages),
                )

        batch_kwargs: dict[str, object] = {
            "text": rendered_prompts,
            "images": image_batch,
            "return_tensors": "pt",
            "padding": True,
        }
        if has_videos:
            batch_kwargs["videos"] = video_batch
        batch = _call_processor(processor, **batch_kwargs)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(
            model,
            **self._generation_inputs(batch),
            **self.generation_kwargs,
        )
        texts = _decode_completion_texts(processor, batch, generated_ids)
        return [
            build_ocr_result(
                text,
                provider_name=self.provider_name,
                model_name=self.model_name or self.model_id,
                text_postprocessor=self.text_postprocessor,
            )
            for text in texts
        ]

    def _load_runtime(self) -> _HFRuntime:
        return _load_hf_runtime()

    def _generation_inputs(self, batch: object) -> dict[str, object]:
        return dict(cast("dict[str, object]", batch))

    def _resolve_model_source(self) -> str:
        return self.model_id

    def _get_model_source(self) -> str:
        if self._model_source is None:
            with self._init_lock:
                if self._model_source is None:
                    self._model_source = self._resolve_model_source()
        return self._model_source

    def _build_vision_inputs(
        self,
        runtime: _HFRuntime,
        conversation: OCRConversation,
    ) -> tuple[object | None, object | None]:
        if self.vision_input_builder is not None:
            built_inputs = self.vision_input_builder(conversation)
            if isinstance(built_inputs, tuple) and len(built_inputs) == 2:
                return built_inputs[0], built_inputs[1]
            return built_inputs, None
        image_inputs, video_inputs, _ = runtime.process_vision_info(
            conversation,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        return image_inputs, video_inputs

    def _get_processor(self, runtime: _HFRuntime) -> object:
        if self._processor is None:
            with self._init_lock:
                if self._processor is None:
                    self._processor = runtime.processor_cls.from_pretrained(
                        self._get_model_source(),
                        trust_remote_code=self.trust_remote_code,
                        **self.processor_kwargs,
                    )
        return self._processor

    def _get_model(self, runtime: _HFRuntime) -> object:
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    self._model = runtime.model_cls.from_pretrained(
                        self._get_model_source(),
                        trust_remote_code=self.trust_remote_code,
                        **self.model_kwargs,
                    )
        return self._model

    def _log_prompt_payload(
        self,
        *,
        rendered_prompt: str,
        conversation: OCRConversation,
        batch_size: int,
    ) -> None:
        log_prompt_payload_once(
            payload={
                "batch_size": batch_size,
                "conversation": conversation,
                "rendered_prompt": rendered_prompt,
            },
            provider_name=self.provider_name,
            has_logged=lambda: self._has_logged_prompt,
            lock=self._prompt_log_lock,
            set_logged=lambda: setattr(self, "_has_logged_prompt", True),
        )


@dataclass(slots=True)
class ChandraOCR2OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``datalab-to/chandra-ocr-2``."""

    model_id: str = CHANDRA_OCR_2_MODEL_ID
    template: OCRPromptTemplateLike = CHANDRA_OCR_2_OCR_TEMPLATE
    model_name: str | None = "chandra-ocr-2"

    def _get_processor(self, runtime: _HFRuntime) -> object:
        processor = super()._get_processor(runtime)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        return processor

    def _get_model(self, runtime: _HFRuntime) -> object:
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    model_kwargs = {
                        **_default_chandra_ocr_2_model_kwargs(),
                        **self.model_kwargs,
                    }
                    self._model = runtime.model_cls.from_pretrained(
                        self._get_model_source(),
                        trust_remote_code=self.trust_remote_code,
                        **model_kwargs,
                    )
                    eval_method = getattr(self._model, "eval", None)
                    if callable(eval_method):
                        eval_method()
        return self._model

    def _build_chandra_batch(
        self,
        processor: object,
        conversations: list[OCRConversation],
    ) -> dict[str, object]:
        processor_apply = getattr(processor, "apply_chat_template", None)
        if not callable(processor_apply):
            message = "Chandra OCR 2 requires `processor.apply_chat_template(...)` support."
            raise _configuration_error(message)
        return _apply_chat_template(
            processor,
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

    def _resolve_chandra_generation_kwargs(self, processor: object, model: object) -> dict[str, object]:
        generation_kwargs = dict(self.generation_kwargs)
        eos_token_ids: list[int] = []
        eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_token_ids.append(eos_token_id)
        elif isinstance(eos_token_id, list):
            eos_token_ids.extend(token_id for token_id in eos_token_id if isinstance(token_id, int))

        tokenizer = getattr(processor, "tokenizer", None)
        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            im_end_id = convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_ids:
                eos_token_ids.append(im_end_id)

        if eos_token_ids:
            generation_kwargs["eos_token_id"] = eos_token_ids
        return generation_kwargs

    def _ocr_sync(self, page: DocumentPage) -> OCRResult:
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)

        rendered, conversation = render_ocr_prompt(
            processor,
            self.template,
            prepared_page,
            add_generation_prompt=True,
        )
        self._log_prompt_payload(
            rendered_prompt=rendered,
            conversation=conversation,
            batch_size=1,
        )
        batch = self._build_chandra_batch(processor, [conversation])
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(
            model,
            **batch,
            **self._resolve_chandra_generation_kwargs(processor, model),
        )
        text = _decode_completion_texts(processor, batch, generated_ids)[0]
        return build_ocr_result(
            text,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
        )

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)
        conversations: list[OCRConversation] = []

        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            rendered, conversation = render_ocr_prompt(
                processor,
                self.template,
                prepared_page,
                add_generation_prompt=True,
            )
            conversations.append(conversation)
            if not self._has_logged_prompt:
                self._log_prompt_payload(
                    rendered_prompt=rendered,
                    conversation=conversation,
                    batch_size=len(pages),
                )

        batch = self._build_chandra_batch(processor, conversations)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(
            model,
            **batch,
            **self._resolve_chandra_generation_kwargs(processor, model),
        )
        texts = _decode_completion_texts(processor, batch, generated_ids)
        return [
            build_ocr_result(
                text,
                provider_name=self.provider_name,
                model_name=self.model_name or self.model_id,
                text_postprocessor=self.text_postprocessor,
            )
            for text in texts
        ]


@dataclass(slots=True)
class Churro3BOCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``stanford-oval/churro-3B``."""

    model_id: str = CHURRO_3B_MODEL_ID
    template: OCRPromptTemplateLike = CHURRO_3B_XML_TEMPLATE
    model_name: str | None = "churro-3B"


@dataclass(slots=True)
class GlmOCROCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``zai-org/GLM-OCR``."""

    model_id: str = GLM_OCR_MODEL_ID
    template: OCRPromptTemplateLike = GLM_OCR_OCR_TEMPLATE
    model_name: str | None = "GLM-OCR"
    generation_kwargs: dict[str, object] = field(
        default_factory=lambda: {"max_new_tokens": 8_192, "do_sample": False}
    )

    def _get_processor(self, runtime: _HFRuntime) -> object:
        processor = HuggingFaceVisionOCRBackend._get_processor(self, runtime)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        return processor

    def _get_model(self, runtime: _HFRuntime) -> object:
        model = HuggingFaceVisionOCRBackend._get_model(self, runtime)
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        return model

    def _build_glm_batch(
        self,
        processor: object,
        conversations: OCRConversation | list[OCRConversation],
        *,
        padding: bool,
    ) -> dict[str, object]:
        processor_apply = getattr(processor, "apply_chat_template", None)
        if not callable(processor_apply):
            message = "GLM-OCR requires `processor.apply_chat_template(...)` support."
            raise _configuration_error(message)
        return _apply_chat_template(
            processor,
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=padding,
        )

    def _generation_inputs(self, batch: object) -> dict[str, object]:
        generation_inputs = HuggingFaceVisionOCRBackend._generation_inputs(self, batch)
        generation_inputs.pop("token_type_ids", None)
        return generation_inputs

    def _ocr_sync(self, page: DocumentPage) -> OCRResult:
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)

        rendered, conversation = render_ocr_prompt(
            processor,
            self.template,
            prepared_page,
            add_generation_prompt=True,
        )
        self._log_prompt_payload(
            rendered_prompt=rendered,
            conversation=conversation,
            batch_size=1,
        )
        batch = self._build_glm_batch(processor, conversation, padding=False)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(
            model,
            **self._generation_inputs(batch),
            **self.generation_kwargs,
        )
        text = _decode_completion_texts(processor, batch, generated_ids)[0]
        return build_ocr_result(
            text,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
        )

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)
        conversations: list[OCRConversation] = []

        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            rendered, conversation = render_ocr_prompt(
                processor,
                self.template,
                prepared_page,
                add_generation_prompt=True,
            )
            conversations.append(conversation)
            if not self._has_logged_prompt:
                self._log_prompt_payload(
                    rendered_prompt=rendered,
                    conversation=conversation,
                    batch_size=len(pages),
                )

        batch = self._build_glm_batch(processor, conversations, padding=True)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(
            model,
            **self._generation_inputs(batch),
            **self.generation_kwargs,
        )
        texts = _decode_completion_texts(processor, batch, generated_ids)
        return [
            build_ocr_result(
                text,
                provider_name=self.provider_name,
                model_name=self.model_name or self.model_id,
                text_postprocessor=self.text_postprocessor,
            )
            for text in texts
        ]


@dataclass(slots=True)
class DeepSeekOCR2OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``deepseek-ai/DeepSeek-OCR-2``."""

    model_id: str = DEEPSEEK_OCR_2_MODEL_ID
    template: OCRPromptTemplateLike = DEEPSEEK_OCR_2_OCR_TEMPLATE
    model_name: str | None = "DeepSeek-OCR-2"
    trust_remote_code: bool = True
    model_kwargs: dict[str, object] = field(default_factory=lambda: {"use_safetensors": True})
    generation_kwargs: dict[str, object] = field(default_factory=lambda: {"max_new_tokens": 8_192})
    base_size: int = 1_024
    image_size: int = 768
    crop_mode: bool = True

    def _load_runtime(self) -> _HFRuntime:
        return _load_hf_auto_model_runtime()

    def _get_model(self, runtime: _HFRuntime) -> object:
        torch = _ensure_deepseek_ocr_2_cuda_runtime()
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    model_kwargs = dict(self.model_kwargs)
                    model = runtime.model_cls.from_pretrained(
                        self._get_model_source(),
                        trust_remote_code=self.trust_remote_code,
                        **model_kwargs,
                    )
                    eval_method = getattr(model, "eval", None)
                    if callable(eval_method):
                        model = eval_method()
                    cuda_method = getattr(model, "cuda", None)
                    if callable(cuda_method) and "device_map" not in model_kwargs:
                        model = cuda_method()
                    to_method = getattr(model, "to", None)
                    if (
                        callable(to_method)
                        and "torch_dtype" not in model_kwargs
                        and "dtype" not in model_kwargs
                    ):
                        model = to_method(cast("Any", torch).bfloat16)
                    self._model = model
        return self._model

    def _infer_deepseek_page(
        self,
        *,
        tokenizer: object,
        model: object,
        page: DocumentPage,
        batch_size: int,
    ) -> OCRResult:
        conversation = build_ocr_conversation(self.template, page)
        rendered_prompt = _deepseek_ocr_2_prompt_from_conversation(conversation)
        self._log_prompt_payload(
            rendered_prompt=rendered_prompt,
            conversation=conversation,
            batch_size=batch_size,
        )

        infer_method = getattr(model, "infer", None)
        if not callable(infer_method):
            message = "DeepSeek-OCR-2 requires a model object with `infer(...)` support."
            raise _configuration_error(message)

        with TemporaryDirectory(prefix="churro-deepseek-ocr-2-") as output_dir:
            image_path = Path(output_dir) / "page.png"
            page.image.save(image_path)
            text = infer_method(
                tokenizer,
                prompt=rendered_prompt,
                image_file=str(image_path),
                output_path=output_dir,
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                save_results=False,
                eval_mode=True,
            )
        if not isinstance(text, str):
            message = "DeepSeek-OCR-2 returned no OCR text."
            raise _provider_error(message)
        return build_ocr_result(
            text,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
        )

    def _ocr_sync(self, page: DocumentPage) -> OCRResult:
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        runtime = self._load_runtime()
        tokenizer = self._get_processor(runtime)
        model = self._get_model(runtime)
        return self._infer_deepseek_page(
            tokenizer=tokenizer,
            model=model,
            page=prepared_page,
            batch_size=1,
        )

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        runtime = self._load_runtime()
        tokenizer = self._get_processor(runtime)
        model = self._get_model(runtime)
        results: list[OCRResult] = []
        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            results.append(
                self._infer_deepseek_page(
                    tokenizer=tokenizer,
                    model=model,
                    page=prepared_page,
                    batch_size=len(pages),
                )
            )
        return results


@dataclass(slots=True)
class DotsOCR15OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``kristaller486/dots.ocr-1.5``.

    :param model_kwargs: Extra kwargs passed to model loading on top of the
        built-in Dots OCR defaults.
    """

    model_id: str = DOTS_OCR_1_5_MODEL_ID
    template: OCRPromptTemplateLike = DOTS_OCR_1_5_OCR_TEMPLATE
    model_name: str | None = "dots.ocr-1.5"
    trust_remote_code: bool = True
    model_kwargs: dict[str, object] = field(default_factory=_default_dots_ocr_1_5_model_kwargs)

    def _load_runtime(self) -> _HFRuntime:
        return _load_hf_causal_runtime()

    def _generation_inputs(self, batch: object) -> dict[str, object]:
        generation_inputs = super()._generation_inputs(batch)
        generation_inputs.pop("mm_token_type_ids", None)
        return generation_inputs

    def _resolve_model_source(self) -> str:
        return _prepare_dots_ocr_model_dir(self.model_id)

    def _get_model(self, runtime: _HFRuntime) -> object:
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    from transformers import AutoConfig

                    model_source = self._get_model_source()
                    config = AutoConfig.from_pretrained(
                        model_source,
                        trust_remote_code=self.trust_remote_code,
                    )
                    vision_config = getattr(config, "vision_config", None)
                    if isinstance(vision_config, dict):
                        vision_config["attn_implementation"] = "sdpa"
                    elif vision_config is not None:
                        vision_config.attn_implementation = "sdpa"
                    self._model = runtime.model_cls.from_pretrained(
                        model_source,
                        config=config,
                        trust_remote_code=self.trust_remote_code,
                        **self.model_kwargs,
                    )
                    _patch_dots_ocr_prepare_inputs_for_generation(self._model)
        return self._model


@dataclass(slots=True)
class DotsMOCROCRBackend(DotsOCR15OCRBackend):
    """Preset OCR backend for ``rednote-hilab/dots.mocr``."""

    model_id: str = DOTS_MOCR_MODEL_ID
    template: OCRPromptTemplateLike = DOTS_MOCR_OCR_TEMPLATE
    model_name: str | None = "dots.mocr"


@dataclass(slots=True)
class MinerU25OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``opendatalab/MinerU2.5-2509-1.2B``."""

    model_id: str = MINERU2_5_2509_1_2B_MODEL_ID
    template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_OCR_TEMPLATE
    layout_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE
    table_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_TABLE_TEMPLATE
    formula_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_FORMULA_TEMPLATE
    image_analysis_template: OCRPromptTemplateLike = MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE
    model_name: str | None = "MinerU2.5-2509-1.2B"
    image_preprocessor: ImagePreprocessor = ensure_rgb
    _helper: MinerU25PipelineHelper = field(
        default_factory=_mineru._default_mineru25_helper,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Preserve user-supplied MinerU2.5 generation overrides without generic defaults."""
        self.generation_kwargs = dict(self.generation_kwargs)

    def _load_runtime(self) -> _HFRuntime:
        return _load_hf_runtime()

    def _get_model(self, runtime: _HFRuntime) -> object:
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    self._model = runtime.model_cls.from_pretrained(
                        self._get_model_source(),
                        trust_remote_code=self.trust_remote_code,
                        **{
                            **_default_mineru25_model_kwargs(),
                            **self.model_kwargs,
                        },
                    )
                    eval_method = getattr(self._model, "eval", None)
                    if callable(eval_method):
                        eval_method()
        return self._model

    def _infer_step(
        self,
        *,
        context: _mineru._MinerU25StepContext,
        image: Image.Image,
        step_key: str,
    ) -> str:
        rendered, conversation = render_ocr_prompt(
            context.processor,
            _mineru._template_for_step(
                step_key,
                _mineru._MinerU25Templates(
                    default_template=self.template,
                    layout_template=self.layout_template,
                    table_template=self.table_template,
                    formula_template=self.formula_template,
                    image_analysis_template=self.image_analysis_template,
                ),
            ),
            DocumentPage.from_image(image),
            add_generation_prompt=True,
        )
        rendered_prompt = _mineru._resolve_rendered_prompt(
            rendered,
            provider_error=_provider_error,
        )
        self._log_prompt_payload(
            rendered_prompt=rendered_prompt,
            conversation=conversation,
            batch_size=context.batch_size,
        )
        image_inputs, video_inputs = self._build_vision_inputs(
            cast("_HFRuntime", context.runtime),
            conversation,
        )
        batch = _call_processor(
            context.processor,
            **_mineru._build_step_batch_kwargs(
                rendered_prompt=rendered_prompt,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
            ),
        )
        batch = _move_batch_to_model(batch, context.model)
        generated_ids = _generate_with_model(
            context.model,
            **self._generation_inputs(batch),
            **_mineru._resolve_generation_kwargs(
                helper=self._helper,
                generation_kwargs=self.generation_kwargs,
                step_key=step_key,
                model=context.model,
            ),
        )
        text = _decode_completion_texts_with_options(
            context.processor,
            batch,
            generated_ids,
            skip_special_tokens=False,
        )[0]
        return self._helper.clean_response(text, step_key=step_key)

    def _build_result(
        self,
        *,
        markdown: str,
        blocks: list[dict[str, object]],
        metrics: dict[str, float | int],
    ) -> OCRResult:
        return build_ocr_result(
            markdown,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
            metadata={
                "output_format": "markdown",
                "blocks": blocks,
                "pipeline_metrics": metrics,
            },
        )

    def _ocr_sync(self, page: DocumentPage) -> OCRResult:
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)
        step_context = _mineru._MinerU25StepContext(
            runtime=runtime,
            processor=processor,
            model=model,
            batch_size=1,
        )

        markdown, blocks, metrics = self._helper.run_two_step(
            prepared_page.image,
            infer_step=lambda image, step_key, _sampling: self._infer_step(
                context=step_context,
                image=image,
                step_key=step_key,
            ),
        )
        return self._build_result(
            markdown=markdown,
            blocks=[dict(block) for block in blocks],
            metrics=metrics,
        )

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)
        batch_size = len(pages)
        step_context = _mineru._MinerU25StepContext(
            runtime=runtime,
            processor=processor,
            model=model,
            batch_size=batch_size,
        )
        results: list[OCRResult] = []
        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            markdown, blocks, metrics = self._helper.run_two_step(
                prepared_page.image,
                infer_step=lambda image, step_key, _sampling: self._infer_step(
                    context=step_context,
                    image=image,
                    step_key=step_key,
                ),
            )
            results.append(
                self._build_result(
                    markdown=markdown,
                    blocks=[dict(block) for block in blocks],
                    metrics=metrics,
                )
            )
        return results


@dataclass(slots=True)
class PaddleOCRVL15OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``PaddlePaddle/PaddleOCR-VL-1.5``."""

    model_id: str = PADDLEOCR_VL_1_5_MODEL_ID
    template: OCRPromptTemplateLike = PADDLEOCR_VL_1_5_OCR_TEMPLATE
    model_name: str | None = "PaddleOCR-VL-1.5"
    generation_kwargs: dict[str, object] = field(
        default_factory=lambda: {"max_new_tokens": 4_096, "do_sample": False}
    )

    def _get_processor(self, runtime: _HFRuntime) -> object:
        processor = super()._get_processor(runtime)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        return processor

    def _get_model(self, runtime: _HFRuntime) -> object:
        model = super()._get_model(runtime)
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        return model

    def _build_paddleocr_vl_batch(
        self,
        processor: object,
        conversations: OCRConversation | list[OCRConversation],
        *,
        padding: bool,
    ) -> dict[str, object]:
        processor_apply = getattr(processor, "apply_chat_template", None)
        if not callable(processor_apply):
            message = "PaddleOCR-VL-1.5 requires `processor.apply_chat_template(...)` support."
            raise _configuration_error(message)
        return _apply_chat_template(
            processor,
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=_paddleocr_vl_processor_kwargs(
                processor=processor,
                padding=padding,
            ),
        )

    def _ocr_sync(self, page: DocumentPage) -> OCRResult:
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)

        rendered, conversation = render_ocr_prompt(
            processor,
            self.template,
            prepared_page,
            add_generation_prompt=True,
        )
        self._log_prompt_payload(
            rendered_prompt=rendered,
            conversation=conversation,
            batch_size=1,
        )
        batch = self._build_paddleocr_vl_batch(processor, conversation, padding=False)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(model, **batch, **self.generation_kwargs)
        text = _decode_completion_texts(processor, batch, generated_ids)[0]
        return build_ocr_result(
            text,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
        )

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)
        conversations: list[OCRConversation] = []

        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            rendered, conversation = render_ocr_prompt(
                processor,
                self.template,
                prepared_page,
                add_generation_prompt=True,
            )
            conversations.append(conversation)
            if not self._has_logged_prompt:
                self._log_prompt_payload(
                    rendered_prompt=rendered,
                    conversation=conversation,
                    batch_size=len(pages),
                )

        batch = self._build_paddleocr_vl_batch(processor, conversations, padding=True)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(model, **batch, **self.generation_kwargs)
        texts = _decode_completion_texts(processor, batch, generated_ids)
        return [
            build_ocr_result(
                text,
                provider_name=self.provider_name,
                model_name=self.model_name or self.model_id,
                text_postprocessor=self.text_postprocessor,
            )
            for text in texts
        ]


@dataclass(slots=True)
class LFM25VLOCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``LiquidAI/LFM2.5-VL-1.6B``."""

    model_id: str = LFM2_5_VL_1_6B_MODEL_ID
    template: OCRPromptTemplateLike = LFM2_5_VL_1_6B_OCR_TEMPLATE
    model_name: str | None = "LFM2.5-VL-1.6B"
    _has_tied_lm_head: bool = field(default=False, init=False, repr=False)

    def _get_processor(self, runtime: _HFRuntime) -> object:
        processor = super()._get_processor(runtime)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        return processor

    def _get_model(self, runtime: _HFRuntime) -> object:
        model = super()._get_model(runtime)
        if self._has_tied_lm_head:
            return model

        with self._init_lock:
            if self._has_tied_lm_head:
                return model
            lm_head = getattr(model, "lm_head", None)
            get_input_embeddings = getattr(model, "get_input_embeddings", None)
            if lm_head is None or not callable(get_input_embeddings):
                self._has_tied_lm_head = True
                return model
            input_embeddings = get_input_embeddings()
            weight = getattr(input_embeddings, "weight", None)
            if weight is not None:
                lm_head.weight = weight
            self._has_tied_lm_head = True
        return model

    def _build_lfm_batch(
        self,
        processor: object,
        conversations: OCRConversation | list[OCRConversation],
        *,
        padding: bool,
    ) -> dict[str, object]:
        processor_apply = getattr(processor, "apply_chat_template", None)
        if not callable(processor_apply):
            message = "Liquid LFM2.5-VL requires `processor.apply_chat_template(...)` support."
            raise _configuration_error(message)
        return _apply_chat_template(
            processor,
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=padding,
        )

    def _ocr_sync(self, page: DocumentPage) -> OCRResult:
        prepared_page = preprocess_backend_page(
            page,
            image_preprocessor=self.image_preprocessor,
        )
        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)

        rendered, conversation = render_ocr_prompt(
            processor,
            self.template,
            prepared_page,
            add_generation_prompt=True,
        )
        self._log_prompt_payload(
            rendered_prompt=rendered,
            conversation=conversation,
            batch_size=1,
        )
        batch = self._build_lfm_batch(processor, conversation, padding=False)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(model, **batch, **self.generation_kwargs)
        text = _decode_completion_texts(processor, batch, generated_ids)[0]
        return build_ocr_result(
            text,
            provider_name=self.provider_name,
            model_name=self.model_name or self.model_id,
            text_postprocessor=self.text_postprocessor,
        )

    def _ocr_batch_sync(self, pages: list[DocumentPage]) -> list[OCRResult]:
        if not pages:
            return []

        runtime = self._load_runtime()
        processor = self._get_processor(runtime)
        model = self._get_model(runtime)
        conversations: list[OCRConversation] = []

        for page in pages:
            prepared_page = preprocess_backend_page(
                page,
                image_preprocessor=self.image_preprocessor,
            )
            rendered, conversation = render_ocr_prompt(
                processor,
                self.template,
                prepared_page,
                add_generation_prompt=True,
            )
            conversations.append(conversation)
            if not self._has_logged_prompt:
                self._log_prompt_payload(
                    rendered_prompt=rendered,
                    conversation=conversation,
                    batch_size=len(pages),
                )

        batch = self._build_lfm_batch(processor, conversations, padding=True)
        batch = _move_batch_to_model(batch, model)
        generated_ids = _generate_with_model(model, **batch, **self.generation_kwargs)
        texts = _decode_completion_texts(processor, batch, generated_ids)
        return [
            build_ocr_result(
                text,
                provider_name=self.provider_name,
                model_name=self.model_name or self.model_id,
                text_postprocessor=self.text_postprocessor,
            )
            for text in texts
        ]


__all__ = [
    "ChandraOCR2OCRBackend",
    "Churro3BOCRBackend",
    "DeepSeekOCR2OCRBackend",
    "DotsMOCROCRBackend",
    "DotsOCR15OCRBackend",
    "GlmOCROCRBackend",
    "HuggingFaceVisionOCRBackend",
    "LFM25VLOCRBackend",
    "MinerU25OCRBackend",
    "PaddleOCRVL15OCRBackend",
]

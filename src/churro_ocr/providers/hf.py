"""Hugging Face OCR backends for template-aware multimodal models."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from churro_ocr._internal.prompt_logging import log_prompt_payload_once
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRBackend, OCRResult
from churro_ocr.page_detection import DocumentPage
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
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    OCRConversation,
    OCRPromptTemplateLike,
)


@dataclass(slots=True)
class _HFRuntime:
    processor_cls: Any
    model_cls: Any
    process_vision_info: Any


def _load_hf_runtime() -> _HFRuntime:
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(
            "Hugging Face OCR requires the 'hf' extra. Install with `pip install \"churro-ocr[hf]\"`."
        ) from exc

    return _HFRuntime(
        processor_cls=AutoProcessor,
        model_cls=AutoModelForImageTextToText,
        process_vision_info=process_vision_info,
    )


def _load_hf_causal_runtime() -> _HFRuntime:
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(
            "Hugging Face OCR requires the 'hf' extra. Install with `pip install \"churro-ocr[hf]\"`."
        ) from exc

    return _HFRuntime(
        processor_cls=AutoProcessor,
        model_cls=AutoModelForCausalLM,
        process_vision_info=process_vision_info,
    )


_DOTS_OCR_1_5_LOCAL_DIRNAME = "DotsOCR_1_5"
_DOTS_FLASH_ATTN_IMPORT = "from flash_attn import flash_attn_varlen_func"
_DOTS_FLASH_ATTN_FALLBACK = """try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
"""
_DOTS_FORCE_BFLOAT16_LINE = "            hidden_states = hidden_states.bfloat16()"
_DOTS_WEIGHT_DTYPE_LINE = (
    "            hidden_states = hidden_states.to(self.patch_embed.patchifier.proj.weight.dtype)"
)


def _patch_dots_ocr_vision_module(model_dir: Path) -> None:
    vision_module_path = model_dir / "modeling_dots_vision.py"
    vision_module = vision_module_path.read_text()
    if _DOTS_FLASH_ATTN_IMPORT not in vision_module and _DOTS_FLASH_ATTN_FALLBACK in vision_module:
        return
    vision_lines = vision_module.splitlines()
    import_index = next(
        (index for index, line in enumerate(vision_lines) if _DOTS_FLASH_ATTN_IMPORT in line),
        None,
    )
    if import_index is None:
        return

    block_tokens = {"", "try:", "except ImportError:", "flash_attn_varlen_func = None"}
    block_start = import_index
    while block_start > 0 and vision_lines[block_start - 1].strip() in block_tokens:
        block_start -= 1

    block_end = import_index + 1
    while block_end < len(vision_lines) and vision_lines[block_end].strip() in block_tokens:
        block_end += 1

    patched_lines = (
        vision_lines[:block_start]
        + _DOTS_FLASH_ATTN_FALLBACK.rstrip("\n").splitlines()
        + vision_lines[block_end:]
    )
    patched_vision_module = "\n".join(patched_lines) + "\n"
    if _DOTS_FORCE_BFLOAT16_LINE in patched_vision_module:
        patched_vision_module = patched_vision_module.replace(
            _DOTS_FORCE_BFLOAT16_LINE,
            _DOTS_WEIGHT_DTYPE_LINE,
        )
    vision_module_path.write_text(patched_vision_module)


def _prepare_dots_ocr_model_dir(model_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - transitively provided by transformers
        raise ConfigurationError(
            'Hugging Face OCR requires huggingface_hub. Install with `pip install "churro-ocr[hf]"`.'
        ) from exc

    model_dir = (
        Path.home()
        / ".cache"
        / "churro-ocr"
        / "hf"
        / _DOTS_OCR_1_5_LOCAL_DIRNAME
        / model_id.replace("/", "__").replace(".", "_")
    )
    snapshot_download(repo_id=model_id, local_dir=model_dir)
    _patch_dots_ocr_vision_module(model_dir)
    return str(model_dir)


def _default_dots_ocr_1_5_model_kwargs() -> dict[str, object]:
    model_kwargs: dict[str, object] = {"dtype": "auto"}
    try:
        import torch
    except ImportError:  # pragma: no cover - torch comes from the hf extra
        return model_kwargs

    if not torch.cuda.is_available():
        return model_kwargs

    free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = max(1, int(free_bytes / (1024**3)) - 1)
    if free_gib < 8:
        return {"dtype": "float32"}

    model_kwargs["device_map"] = "auto"
    model_kwargs["max_memory"] = {0: f"{free_gib}GiB", "cpu": "128GiB"}
    return model_kwargs


@dataclass(slots=True)
class HuggingFaceVisionOCRBackend(OCRBackend):
    """OCR backend for local Hugging Face multimodal models with custom templates."""

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
        self.generation_kwargs = {
            "max_new_tokens": DEFAULT_OCR_MAX_TOKENS,
            **self.generation_kwargs,
        }

    async def ocr(self, page: DocumentPage) -> OCRResult:
        return await asyncio.to_thread(self._ocr_sync, page)

    async def ocr_batch(self, pages: list[DocumentPage]) -> list[OCRResult]:
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
        batch = processor(**batch_kwargs)
        model_device = getattr(model, "device", None)
        if hasattr(batch, "to") and model_device is not None:
            batch = batch.to(model_device)
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None:
            for key, value in batch.items():
                if hasattr(value, "dtype") and getattr(value.dtype, "is_floating_point", False):
                    batch[key] = value.to(dtype=model_dtype)

        generated_ids = model.generate(**batch, **self.generation_kwargs)
        prompt_length = batch["input_ids"].shape[1]
        completion_ids = generated_ids[:, prompt_length:]
        text = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
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
        batch = processor(**batch_kwargs)
        model_device = getattr(model, "device", None)
        if hasattr(batch, "to") and model_device is not None:
            batch = batch.to(model_device)
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None:
            for key, value in batch.items():
                if hasattr(value, "dtype") and getattr(value.dtype, "is_floating_point", False):
                    batch[key] = value.to(dtype=model_dtype)

        generated_ids = model.generate(**batch, **self.generation_kwargs)
        prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
        completion_ids = [
            output_ids[int(prompt_length) :]
            for prompt_length, output_ids in zip(prompt_lengths, generated_ids, strict=True)
        ]
        texts = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
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

    def _get_processor(self, runtime: _HFRuntime) -> Any:
        if self._processor is None:
            with self._init_lock:
                if self._processor is None:
                    self._processor = runtime.processor_cls.from_pretrained(
                        self._get_model_source(),
                        trust_remote_code=self.trust_remote_code,
                        **self.processor_kwargs,
                    )
        return self._processor

    def _get_model(self, runtime: _HFRuntime) -> Any:
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
class Churro3BOCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for `stanford-oval/churro-3B`."""

    model_id: str = CHURRO_3B_MODEL_ID
    template: OCRPromptTemplateLike = CHURRO_3B_XML_TEMPLATE
    model_name: str | None = "churro-3B"


@dataclass(slots=True)
class DotsOCR15OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for `kristaller486/dots.ocr-1.5`."""

    model_id: str = DOTS_OCR_1_5_MODEL_ID
    template: OCRPromptTemplateLike = DOTS_OCR_1_5_OCR_TEMPLATE
    model_name: str | None = "dots.ocr-1.5"
    trust_remote_code: bool = True
    model_kwargs: dict[str, object] = field(default_factory=_default_dots_ocr_1_5_model_kwargs)

    def _load_runtime(self) -> _HFRuntime:
        return _load_hf_causal_runtime()

    def _resolve_model_source(self) -> str:
        return _prepare_dots_ocr_model_dir(self.model_id)

    def _get_model(self, runtime: _HFRuntime) -> Any:
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
        return self._model


__all__ = [
    "Churro3BOCRBackend",
    "DotsOCR15OCRBackend",
    "HuggingFaceVisionOCRBackend",
]

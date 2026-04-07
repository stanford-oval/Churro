"""Hugging Face OCR backends for template-aware multimodal models."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MethodType
from typing import Any, cast

from churro_ocr._internal.install import install_command_hint
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
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    LFM2_5_VL_1_6B_MODEL_ID,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    PADDLEOCR_VL_1_5_MODEL_ID,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
    OCRConversation,
    OCRPromptTemplateLike,
)

_HF_EXTRA_INSTALL_HINT = install_command_hint("hf")
_HF_TORCH_INSTALL_HINT = (
    f"Hugging Face OCR requires a separately installed PyTorch runtime. {_HF_EXTRA_INSTALL_HINT}"
)


@dataclass(slots=True)
class _HFRuntime:
    processor_cls: Any
    model_cls: Any
    process_vision_info: Any


def _ensure_hf_torch_runtime() -> None:
    try:
        import_module("torch")
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(_HF_TORCH_INSTALL_HINT) from exc


def _load_hf_runtime() -> _HFRuntime:
    _ensure_hf_torch_runtime()
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(
            f"Hugging Face OCR requires the `hf` runtime. {_HF_EXTRA_INSTALL_HINT}"
        ) from exc

    return _HFRuntime(
        processor_cls=AutoProcessor,
        model_cls=AutoModelForImageTextToText,
        process_vision_info=process_vision_info,
    )


def _load_hf_causal_runtime() -> _HFRuntime:
    _ensure_hf_torch_runtime()
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(
            f"Hugging Face OCR requires the `hf` runtime. {_HF_EXTRA_INSTALL_HINT}"
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
            f"Hugging Face OCR requires the `hf` runtime. {_HF_EXTRA_INSTALL_HINT}"
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


def _patch_dots_ocr_prepare_inputs_for_generation(model: Any) -> None:
    prepare_inputs_for_generation = getattr(model, "prepare_inputs_for_generation", None)
    if not callable(prepare_inputs_for_generation):
        return
    if getattr(model, "_churro_dots_prepare_inputs_patched", False):
        return

    original_prepare_inputs = getattr(prepare_inputs_for_generation, "__func__", None)
    base_prepare_inputs_for_generation = prepare_inputs_for_generation
    if original_prepare_inputs is not None:
        for candidate in type(model).__mro__[1:]:
            candidate_prepare_inputs = candidate.__dict__.get("prepare_inputs_for_generation")
            if candidate_prepare_inputs is None or candidate_prepare_inputs is original_prepare_inputs:
                continue
            base_prepare_inputs_for_generation = cast("Any", candidate_prepare_inputs).__get__(
                model,
                type(model),
            )
            break
    if not callable(base_prepare_inputs_for_generation):
        return

    def _patched_prepare_inputs_for_generation(
        self: Any,
        input_ids: object,
        past_key_values: object = None,
        inputs_embeds: object = None,
        pixel_values: object = None,
        attention_mask: object = None,
        cache_position: object = None,
        num_logits_to_keep: object = None,
        **kwargs: object,
    ) -> Any:
        model_inputs = base_prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        first_cache_position: int | None = None
        if cache_position is not None:
            try:
                first_cache_position = int(cast("Any", cache_position)[0])
            except Exception:
                first_cache_position = None
        if first_cache_position in (None, 0):
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    model.prepare_inputs_for_generation = MethodType(_patched_prepare_inputs_for_generation, model)
    model._churro_dots_prepare_inputs_patched = True


def _default_dots_ocr_1_5_model_kwargs() -> dict[str, object]:
    model_kwargs: dict[str, object] = {"dtype": "auto"}
    try:
        torch = import_module("torch")
    except ImportError:  # pragma: no cover - torch is installed separately for local HF use
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


def _default_chandra_ocr_2_model_kwargs() -> dict[str, object]:
    model_kwargs: dict[str, object] = {
        "device_map": "auto",
        "dtype": "auto",
    }
    try:
        torch = import_module("torch")
    except ImportError:  # pragma: no cover - torch is installed separately for local HF use
        return model_kwargs

    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16
    return model_kwargs


def _move_batch_to_model(batch: Any, model: Any) -> Any:
    model_device = getattr(model, "device", None)
    if hasattr(batch, "to") and model_device is not None:
        batch = batch.to(model_device)
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        batch_mapping = cast(dict[str, object], batch)
        for key, value in batch_mapping.items():
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating_point", False):
                batch_mapping[key] = cast(Any, value).to(dtype=model_dtype)
    return batch


def _decode_completion_texts(processor: Any, batch: Any, generated_ids: Any) -> list[str]:
    batch_mapping = cast(dict[str, object], batch)
    attention_mask = batch_mapping.get("attention_mask")
    if attention_mask is not None and hasattr(attention_mask, "sum"):
        prompt_lengths = cast(Any, attention_mask).sum(dim=1).tolist()
        completion_ids = [
            output_ids[int(prompt_length) :]
            for prompt_length, output_ids in zip(prompt_lengths, generated_ids, strict=True)
        ]
    else:
        prompt_length = cast(Any, batch_mapping["input_ids"]).shape[1]
        completion_ids = generated_ids[:, prompt_length:]
    return processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _paddleocr_vl_processor_kwargs(*, processor: Any, padding: bool) -> dict[str, object]:
    processor_kwargs: dict[str, object] = {
        "text_kwargs": {
            "padding": padding,
            "return_mm_token_type_ids": True,
        }
    }
    image_processor = getattr(processor, "image_processor", None)
    images_kwargs: dict[str, int] = {}
    for key in ("min_pixels", "max_pixels"):
        value = getattr(image_processor, key, None)
        if isinstance(value, int):
            images_kwargs[key] = value
    if images_kwargs:
        processor_kwargs["images_kwargs"] = images_kwargs
    return processor_kwargs


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
        batch = processor(**batch_kwargs)
        batch = _move_batch_to_model(batch, model)
        generated_ids = model.generate(
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
        batch = processor(**batch_kwargs)
        batch = _move_batch_to_model(batch, model)
        generated_ids = model.generate(
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

    def _generation_inputs(self, batch: Any) -> dict[str, object]:
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
class ChandraOCR2OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``datalab-to/chandra-ocr-2``."""

    model_id: str = CHANDRA_OCR_2_MODEL_ID
    template: OCRPromptTemplateLike = CHANDRA_OCR_2_OCR_TEMPLATE
    model_name: str | None = "chandra-ocr-2"

    def _get_processor(self, runtime: _HFRuntime) -> Any:
        processor = super()._get_processor(runtime)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        return processor

    def _get_model(self, runtime: _HFRuntime) -> Any:
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

    def _build_chandra_batch(self, processor: Any, conversations: list[OCRConversation]) -> Any:
        processor_apply = getattr(processor, "apply_chat_template", None)
        if not callable(processor_apply):
            raise ConfigurationError("Chandra OCR 2 requires `processor.apply_chat_template(...)` support.")
        return processor_apply(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

    def _resolve_chandra_generation_kwargs(self, processor: Any, model: Any) -> dict[str, object]:
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
        generated_ids = model.generate(
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
        generated_ids = model.generate(
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

    def _generation_inputs(self, batch: Any) -> dict[str, object]:
        generation_inputs = super()._generation_inputs(batch)
        generation_inputs.pop("mm_token_type_ids", None)
        return generation_inputs

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
                    _patch_dots_ocr_prepare_inputs_for_generation(self._model)
        return self._model


@dataclass(slots=True)
class PaddleOCRVL15OCRBackend(HuggingFaceVisionOCRBackend):
    """Preset OCR backend for ``PaddlePaddle/PaddleOCR-VL-1.5``."""

    model_id: str = PADDLEOCR_VL_1_5_MODEL_ID
    template: OCRPromptTemplateLike = PADDLEOCR_VL_1_5_OCR_TEMPLATE
    model_name: str | None = "PaddleOCR-VL-1.5"
    generation_kwargs: dict[str, object] = field(
        default_factory=lambda: {"max_new_tokens": 4_096, "do_sample": False}
    )

    def _get_processor(self, runtime: _HFRuntime) -> Any:
        processor = super()._get_processor(runtime)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        return processor

    def _get_model(self, runtime: _HFRuntime) -> Any:
        model = super()._get_model(runtime)
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        return model

    def _build_paddleocr_vl_batch(
        self,
        processor: Any,
        conversations: OCRConversation | list[OCRConversation],
        *,
        padding: bool,
    ) -> Any:
        processor_apply = getattr(processor, "apply_chat_template", None)
        if not callable(processor_apply):
            raise ConfigurationError(
                "PaddleOCR-VL-1.5 requires `processor.apply_chat_template(...)` support."
            )
        return processor_apply(
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
        generated_ids = model.generate(**batch, **self.generation_kwargs)
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
        generated_ids = model.generate(**batch, **self.generation_kwargs)
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

    def _get_processor(self, runtime: _HFRuntime) -> Any:
        processor = super()._get_processor(runtime)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        return processor

    def _get_model(self, runtime: _HFRuntime) -> Any:
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
        processor: Any,
        conversations: OCRConversation | list[OCRConversation],
        *,
        padding: bool,
    ) -> Any:
        processor_apply = getattr(processor, "apply_chat_template", None)
        if not callable(processor_apply):
            raise ConfigurationError(
                "Liquid LFM2.5-VL requires `processor.apply_chat_template(...)` support."
            )
        return processor_apply(
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
        generated_ids = model.generate(**batch, **self.generation_kwargs)
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
        generated_ids = model.generate(**batch, **self.generation_kwargs)
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
    "DotsOCR15OCRBackend",
    "HuggingFaceVisionOCRBackend",
    "LFM25VLOCRBackend",
    "PaddleOCRVL15OCRBackend",
]

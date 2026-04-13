"""Runtime loading and generic protocol helpers for Hugging Face OCR backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, cast

from churro_ocr._internal.install import install_command_hint
from churro_ocr.errors import ConfigurationError, ProviderError

if TYPE_CHECKING:
    from collections.abc import Callable

_HF_EXTRA_INSTALL_HINT = install_command_hint("hf")
_HF_TORCH_INSTALL_HINT = (
    f"Hugging Face OCR requires a separately installed PyTorch runtime. {_HF_EXTRA_INSTALL_HINT}"
)


def _configuration_error(message: str) -> ConfigurationError:
    return ConfigurationError(message)


def _provider_error(message: str) -> ProviderError:
    return ProviderError(message)


@dataclass(slots=True)
class _HFRuntime:
    processor_cls: Any
    model_cls: Any
    process_vision_info: Any


class _HFProcessorCallable(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


class _HFProcessorDecoder(Protocol):
    def batch_decode(
        self,
        token_ids: object,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]: ...


class _HFChatTemplateProcessor(Protocol):
    def apply_chat_template(
        self,
        conversations: object,
        **kwargs: object,
    ) -> dict[str, object]: ...


class _HFGenerativeModel(Protocol):
    def generate(self, **kwargs: object) -> object: ...


class _TorchCudaNamespace(Protocol):
    def is_available(self) -> bool: ...

    def mem_get_info(self) -> tuple[int, int]: ...


class _TorchModuleLike(Protocol):
    cuda: _TorchCudaNamespace
    bfloat16: object


def _ensure_hf_torch_runtime() -> None:
    _ensure_hf_torch_runtime_with_import(import_module)


def _ensure_hf_torch_runtime_with_import(import_module_fn: Callable[[str], object]) -> None:
    try:
        import_module_fn("torch")
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(_HF_TORCH_INSTALL_HINT) from exc


def _load_torch_module() -> _TorchModuleLike:
    return _load_torch_module_with_import(import_module)


def _load_torch_module_with_import(import_module_fn: Callable[[str], object]) -> _TorchModuleLike:
    return cast("_TorchModuleLike", import_module_fn("torch"))


def _call_processor(processor: object, **kwargs: object) -> dict[str, object]:
    return cast("dict[str, object]", cast("_HFProcessorCallable", processor)(**kwargs))


def _generate_with_model(model: object, **kwargs: object) -> object:
    return cast("_HFGenerativeModel", model).generate(**kwargs)


def _apply_chat_template(
    processor: object,
    conversations: object,
    **kwargs: object,
) -> dict[str, object]:
    return cast("_HFChatTemplateProcessor", processor).apply_chat_template(conversations, **kwargs)


def _load_hf_runtime() -> _HFRuntime:
    return _load_hf_runtime_with_import(import_module)


def _load_hf_runtime_with_import(import_module_fn: Callable[[str], object]) -> _HFRuntime:
    _ensure_hf_torch_runtime_with_import(import_module_fn)
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        message = f"Hugging Face OCR requires the `hf` runtime. {_HF_EXTRA_INSTALL_HINT}"
        raise _configuration_error(message) from exc

    return _HFRuntime(
        processor_cls=AutoProcessor,
        model_cls=AutoModelForImageTextToText,
        process_vision_info=process_vision_info,
    )


def _load_hf_causal_runtime() -> _HFRuntime:
    return _load_hf_causal_runtime_with_import(import_module)


def _load_hf_causal_runtime_with_import(import_module_fn: Callable[[str], object]) -> _HFRuntime:
    _ensure_hf_torch_runtime_with_import(import_module_fn)
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        message = f"Hugging Face OCR requires the `hf` runtime. {_HF_EXTRA_INSTALL_HINT}"
        raise _configuration_error(message) from exc

    return _HFRuntime(
        processor_cls=AutoProcessor,
        model_cls=AutoModelForCausalLM,
        process_vision_info=process_vision_info,
    )


def _load_hf_auto_model_runtime() -> _HFRuntime:
    return _load_hf_auto_model_runtime_with_import(import_module)


def _load_hf_auto_model_runtime_with_import(import_module_fn: Callable[[str], object]) -> _HFRuntime:
    _ensure_hf_torch_runtime_with_import(import_module_fn)
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional extra path
        message = f"Hugging Face OCR requires the `hf` runtime. {_HF_EXTRA_INSTALL_HINT}"
        raise _configuration_error(message) from exc

    return _HFRuntime(
        processor_cls=AutoTokenizer,
        model_cls=AutoModel,
        process_vision_info=None,
    )


def _load_hf_auto_processor_model_runtime() -> _HFRuntime:
    return _load_hf_auto_processor_model_runtime_with_import(import_module)


def _load_hf_auto_processor_model_runtime_with_import(
    import_module_fn: Callable[[str], object],
) -> _HFRuntime:
    _ensure_hf_torch_runtime_with_import(import_module_fn)
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional extra path
        message = f"Hugging Face OCR requires the `hf` runtime. {_HF_EXTRA_INSTALL_HINT}"
        raise _configuration_error(message) from exc

    return _HFRuntime(
        processor_cls=AutoProcessor,
        model_cls=AutoModel,
        process_vision_info=None,
    )


def _ensure_deepseek_ocr_2_cuda_runtime() -> _TorchModuleLike:
    return _ensure_deepseek_ocr_2_cuda_runtime_with_import(import_module)


def _ensure_deepseek_ocr_2_cuda_runtime_with_import(
    import_module_fn: Callable[[str], object],
) -> _TorchModuleLike:
    _ensure_hf_torch_runtime_with_import(import_module_fn)
    torch = _load_torch_module_with_import(import_module_fn)
    if not torch.cuda.is_available():
        message = (
            "DeepSeek-OCR-2 HF backend requires a CUDA-capable PyTorch runtime because "
            "the upstream `infer(...)` implementation moves inputs to CUDA."
        )
        raise _configuration_error(message)
    return torch

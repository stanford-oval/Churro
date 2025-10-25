"""OCR systems package."""

from importlib import import_module
from typing import TYPE_CHECKING


__all__ = [
    "BaseOCR",
    "ZeroShotLLMOCR",
    "AzureOCR",
    "MistralOCR",
    "FineTunedOCR",
    "OCRFactory",
    "LLMImprover",
]

_LAZY_MAP = {
    "BaseOCR": "churro.systems.base_ocr:BaseOCR",
    "ZeroShotLLMOCR": "churro.systems.llm_ocr:ZeroShotLLMOCR",
    "AzureOCR": "churro.systems.azure_ocr:AzureOCR",
    "MistralOCR": "churro.systems.mistral_ocr:MistralOCR",
    "FineTunedOCR": "churro.systems.finetuned_ocr:FineTunedOCR",
    "OCRFactory": "churro.systems.ocr_factory:OCRFactory",
    "LLMImprover": "churro.systems.llm_improver:LLMImprover",
}

_CACHE: dict[str, object] = {}

if TYPE_CHECKING:
    from .azure_ocr import AzureOCR
    from .base_ocr import BaseOCR
    from .finetuned_ocr import FineTunedOCR
    from .llm_improver import LLMImprover
    from .llm_ocr import ZeroShotLLMOCR
    from .mistral_ocr import MistralOCR
    from .ocr_factory import OCRFactory


def __getattr__(name: str) -> object:
    if name not in _LAZY_MAP:
        raise AttributeError(f"module 'churro.systems' has no attribute '{name}'")
    if name in _CACHE:
        return _CACHE[name]
    module_path, attr = _LAZY_MAP[name].split(":", 1)
    module = import_module(module_path)
    resolved = getattr(module, attr)
    _CACHE[name] = resolved
    return resolved


def __dir__() -> list[str]:
    return sorted(__all__)

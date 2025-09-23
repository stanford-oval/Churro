"""OCR systems module with class-based implementations."""

from .azure_ocr import AzureOCR
from .base_ocr import BaseOCR
from .finetuned_ocr import FineTunedOCR
from .hybrid_pipeline_ocr import HybridPipelineOCR
from .llm_improver import LLMImprover
from .llm_ocr import ZeroShotLLMOCR
from .mistral_ocr import MistralOCR
from .ocr_factory import OCRFactory


__all__ = [
    # Base class
    "BaseOCR",
    # OCR implementations
    "ZeroShotLLMOCR",
    "AzureOCR",
    "MistralOCR",
    "HybridPipelineOCR",
    "FineTunedOCR",
    # Factory
    "OCRFactory",
    # LLM-based OCR improvement
    "LLMImprover",
]

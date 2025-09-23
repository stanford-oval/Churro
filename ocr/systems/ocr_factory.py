"""OCR Factory for creating OCR system instances."""

from typing import Any, Dict

from .azure_ocr import AzureOCR
from .base_ocr import BaseOCR
from .finetuned_ocr import FineTunedOCR
from .hybrid_pipeline_ocr import HybridPipelineOCR
from .llm_ocr import ZeroShotLLMOCR
from .mistral_ocr import MistralOCR


class OCRFactory:
    """Factory class for creating OCR system instances."""

    _ocr_classes = {
        "azure": AzureOCR,
        "mistral_ocr": MistralOCR,
        "llm": ZeroShotLLMOCR,
        "hybrid": HybridPipelineOCR,
        "finetuned": FineTunedOCR,
    }

    @classmethod
    def get_available_systems(cls) -> list[str]:
        """Get list of available OCR system names."""
        return list(cls._ocr_classes.keys())

    @classmethod
    def create_ocr_system(cls, args) -> BaseOCR:
        """Create an OCR system instance by name."""
        system_name = args.system
        if system_name not in cls._ocr_classes:
            available_systems = ", ".join(cls._ocr_classes.keys())
            raise ValueError(
                f"Invalid system: {system_name}. Available systems: {available_systems}"
            )

        ocr_class = cls._ocr_classes[system_name]
        return ocr_class(**_extract_system_config(args))


def _extract_system_config(args) -> Dict[str, Any]:
    """Extract configuration parameters for the OCR system from args."""
    config = {}

    # Common parameters
    if hasattr(args, "engine"):
        config["engine"] = args.engine
    if hasattr(args, "max_tokens"):
        config["max_tokens"] = args.max_tokens
    if hasattr(args, "max_concurrency"):
        config["max_concurrency"] = args.max_concurrency

    # LLM-specific parameters
    if hasattr(args, "resize"):
        config["resize"] = args.resize
    if hasattr(args, "reasoning_effort"):
        config["reasoning_effort"] = args.reasoning_effort

    # Pipeline-specific parameters
    if hasattr(args, "layout_detection_models"):
        config["layout_detection_models"] = args.layout_detection_models
    if hasattr(args, "layout_detection_num_splits"):
        config["layout_detection_num_splits"] = args.layout_detection_num_splits
    if hasattr(args, "layout_detection_num_iterations"):
        config["layout_detection_num_iterations"] = args.layout_detection_num_iterations

    return config

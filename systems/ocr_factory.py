"""OCR Factory for creating OCR system instances."""

from argparse import Namespace
from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .base_ocr import BaseOCR


class OCRFactory:
    """Factory class for creating OCR system instances."""

    _REGISTRY: dict[str, str] = {
        "azure": "churro.systems.azure_ocr:AzureOCR",
        "mistral_ocr": "churro.systems.mistral_ocr:MistralOCR",
        "llm": "churro.systems.llm_ocr:ZeroShotLLMOCR",
        "finetuned": "churro.systems.finetuned_ocr:FineTunedOCR",
    }

    _CACHE: dict[str, type["BaseOCR"]] = {}

    @classmethod
    def get_available_systems(cls) -> list[str]:
        """Get list of available OCR system names."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def create_ocr_system(cls, args: Namespace) -> "BaseOCR":
        """Create an OCR system instance by name."""
        system_name = args.system
        if system_name not in cls._REGISTRY:
            available_systems = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Invalid system: {system_name}. Available systems: {available_systems}"
            )

        ocr_class = cls._load_class(system_name)
        return ocr_class(**_extract_system_config(args))

    @classmethod
    def _load_class(cls, system_name: str) -> type["BaseOCR"]:
        if system_name in cls._CACHE:
            return cls._CACHE[system_name]
        module_path, class_name = cls._REGISTRY[system_name].split(":", 1)
        module = import_module(module_path)
        ocr_class = getattr(module, class_name)
        cls._CACHE[system_name] = ocr_class
        return ocr_class


def _extract_system_config(args: Namespace) -> dict[str, Any]:
    """Extract configuration parameters for the OCR system from args."""
    config = {}

    # Common parameters
    if hasattr(args, "engine"):
        config["engine"] = args.engine
    if hasattr(args, "backup_engine"):
        config["backup_engine"] = args.backup_engine
    if hasattr(args, "max_tokens"):
        config["max_tokens"] = args.max_tokens
    if getattr(args, "system", None) == "finetuned" and hasattr(args, "strip_xml"):
        config["strip_xml"] = args.strip_xml

    # LLM-specific parameters
    if hasattr(args, "resize"):
        config["resize"] = args.resize
    if hasattr(args, "reasoning_effort"):
        config["reasoning_effort"] = args.reasoning_effort
    if hasattr(args, "output_markdown"):
        config["output_markdown"] = args.output_markdown

    # Pipeline-specific parameters
    if hasattr(args, "layout_detection_models"):
        config["layout_detection_models"] = args.layout_detection_models
    if hasattr(args, "layout_detection_num_splits"):
        config["layout_detection_num_splits"] = args.layout_detection_num_splits
    if hasattr(args, "layout_detection_num_iterations"):
        config["layout_detection_num_iterations"] = args.layout_detection_num_iterations

    return config

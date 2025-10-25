"""Public interfaces for Gemini page boundary detection."""

from .detector import GeminiPageBoundaryDetector, run_page_detection


__all__ = [
    "GeminiPageBoundaryDetector",
    "run_page_detection",
]

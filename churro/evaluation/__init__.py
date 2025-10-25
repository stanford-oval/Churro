"""Evaluation utilities for OCR outputs."""

from .evaluate_page import (
    batch_evaluate,
    calculate_metrics,
    calculate_metrics_from_text,
    evaluate_page,
    initialize_metrics,
)
from .metrics import compute_metrics


__all__ = [
    "batch_evaluate",
    "calculate_metrics",
    "calculate_metrics_from_text",
    "compute_metrics",
    "evaluate_page",
    "initialize_metrics",
]

"""Shared dataset, prediction, and result types for CHURRO tooling evaluation flows."""

from __future__ import annotations

from typing import Any, TypedDict

from PIL import Image

EVALUATION_EXAMPLE_FIELDS = (
    "cleaned_transcription",
    "dataset_id",
    "document_type",
    "example_id",
    "main_language",
    "main_script",
)


class MetricInputExample(TypedDict):
    """Dataset fields required to evaluate one OCR prediction."""

    cleaned_transcription: str
    example_id: str
    main_language: str
    main_script: str


class EvaluationExample(MetricInputExample):
    """Dataset fields retained after OCR for aggregate evaluation."""

    dataset_id: str
    document_type: str


class BenchmarkDatasetExample(EvaluationExample):
    """Full CHURRO dataset example used by the benchmark runner."""

    image: Image.Image


class PageEvaluationMetrics(TypedDict):
    """Computed metrics for one evaluated example."""

    normalized_levenshtein_similarity: float
    repetition: float
    is_empty: float
    bleu: float
    normalized_predicted_text: str
    normalized_gold_text: str
    main_language: str
    main_script: str


class PageEvaluationResult(PageEvaluationMetrics):
    """Metric row enriched with example metadata and raw texts."""

    example_id: str
    predicted_text: str
    gold_text: str
    dataset_id: str
    document_type: str


class BenchmarkPrediction(TypedDict):
    """OCR output retained during benchmarking before metrics are computed."""

    text: str
    metadata: dict[str, Any]


class BenchmarkOutputRow(PageEvaluationResult):
    """Serialized benchmark output row written to ``outputs.json``."""

    metadata: dict[str, Any]


def to_evaluation_example(example: BenchmarkDatasetExample) -> EvaluationExample:
    """Keep only the dataset fields needed after OCR completes."""
    return {field_name: example[field_name] for field_name in EVALUATION_EXAMPLE_FIELDS}

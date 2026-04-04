"""Aggregate metrics and output writers for repo-only benchmark tooling."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from churro_ocr._internal.logging import logger
from tooling.evaluation.evaluate_page import batch_evaluate
from tooling.evaluation.types import EvaluationExample, PageEvaluationResult


def _get_llm_total_cost() -> float:
    try:  # pragma: no cover - optional integration path
        from litellm import completion_cost
    except ImportError:
        return 0.0
    del completion_cost
    return 0.0


def _get_azure_total_cost() -> float:
    return 0.0


def round_metric(value: float | int) -> float:
    """Round a numeric value to one decimal place."""
    return float(f"{value:.1f}")


def to_rounded_percentage(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert numeric values to percentages rounded to one decimal place."""
    return {
        key: round_metric(value * 100) if isinstance(value, int | float) else value
        for key, value in metrics.items()
    }


def calculate_language_and_type_metrics(
    outputs: list[PageEvaluationResult],
    main_metric: str = "normalized_levenshtein_similarity",
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute averages grouped by language, document type, and both."""
    language_to_metrics: dict[str, list[float]] = defaultdict(list)
    language_type_to_metrics: dict[str, list[float]] = defaultdict(list)
    type_to_metrics: dict[str, list[float]] = {"print": [], "handwriting": []}

    for output in outputs:
        main_language = str(output["main_language"])
        metric = float(output[main_metric])
        document_type = str(output["document_type"])
        language_to_metrics[main_language].append(metric)
        type_to_metrics.setdefault(document_type, []).append(metric)
        language_type_to_metrics[f"{main_language}_{document_type}"].append(metric)

    averaged_language = {
        language: sum(values) / len(values) if values else 0.0
        for language, values in language_to_metrics.items()
    }
    averaged_type = {
        document_type: sum(values) / len(values) if values else 0.0
        for document_type, values in type_to_metrics.items()
    }
    averaged_language_type = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in language_type_to_metrics.items()
    }
    return averaged_language, averaged_type, averaged_language_type


def _build_output_rows(
    dataset: list[EvaluationExample],
    per_example_outputs: list[PageEvaluationResult],
) -> list[PageEvaluationResult]:
    """Attach stable example ids to per-example outputs before writing them."""
    return [
        {"example_id": str(example["example_id"]), **evaluation_output}
        for evaluation_output, example in zip(per_example_outputs, dataset, strict=False)
    ]


def compute_metrics(
    dataset: list[EvaluationExample],
    predicted_texts: list[str],
    output_prefix: str | Path,
    elapsed_time: float,
    main_metric: str = "normalized_levenshtein_similarity",
) -> dict[str, Any]:
    """Compute aggregate metrics and write benchmark output files."""
    sanitized_predictions = [prediction or "" for prediction in predicted_texts]
    aggregate_metrics, per_example_outputs = batch_evaluate(dataset, sanitized_predictions)
    outputs = _build_output_rows(dataset, per_example_outputs)

    output_dir = Path(output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False))

    language_metrics, type_metrics, language_type_metrics = calculate_language_and_type_metrics(
        outputs,
        main_metric,
    )

    aggregate_metrics = to_rounded_percentage(aggregate_metrics)
    language_metrics = to_rounded_percentage(language_metrics)
    type_metrics = to_rounded_percentage(type_metrics)
    language_type_metrics = to_rounded_percentage(language_type_metrics)
    aggregate_metrics["llm_cost ($)"] = round_metric(_get_llm_total_cost())
    aggregate_metrics["azure_cost ($)"] = round_metric(_get_azure_total_cost())
    aggregate_metrics["elapsed_time (s)"] = round_metric(elapsed_time)

    combined_metrics = {
        "main_language_metrics": language_metrics,
        "type_metrics": type_metrics,
        "aggregate_metrics": aggregate_metrics,
        "main_language_and_type_metrics": language_type_metrics,
    }
    (output_dir / "all_metrics.json").write_text(json.dumps(combined_metrics, indent=2))

    logger.info("Average metrics per document type: %s", json.dumps(type_metrics, indent=2))
    logger.info("Average metrics per main language: %s", json.dumps(language_metrics, indent=2))
    logger.info(
        "Average metrics per main language and type: %s",
        json.dumps(language_type_metrics, indent=2),
    )
    logger.info("Aggregated metrics: %s", json.dumps(aggregate_metrics, indent=2))

    return combined_metrics

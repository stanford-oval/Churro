"""Per-page evaluation helpers for repo-only benchmark tooling."""

from __future__ import annotations

import multiprocessing
from typing import Any

try:  # pragma: no cover - optional dependency
    import nltk
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    nltk = None  # type: ignore[assignment]
from rapidfuzz import distance as rf_distance
from tqdm import tqdm

from churro_ocr._internal.logging import logger
from churro_ocr.prompts import strip_ocr_output_tag
from tooling.evaluation.normalization import normalize_text_for_evaluation
from tooling.evaluation.repetition import has_long_repetition
from tooling.evaluation.types import (
    EvaluationExample,
    MetricInputExample,
    PageEvaluationMetrics,
    PageEvaluationResult,
)
from tooling.evaluation.xml_utils import extract_actual_text_from_xml

bleu_metric: Any | None = None


def initialize_metrics() -> None:
    """Lazily load BLEU resources."""
    global bleu_metric
    if bleu_metric is not None:
        return
    if nltk is None:
        raise ModuleNotFoundError("BLEU evaluation requires the optional dependency 'nltk'.")
    try:  # pragma: no cover - optional dependency
        import evaluate
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "BLEU evaluation requires the optional dependency 'evaluate'."
        ) from exc
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    bleu_metric = evaluate.load("bleu")


def levenshtein_distance(a: str, b: str, max_cost: int | None = None) -> int:
    """Compute Levenshtein distance with optional cutoff."""
    if max_cost is not None:
        return rf_distance.Levenshtein.distance(a, b, score_cutoff=max_cost)
    return rf_distance.Levenshtein.distance(a, b)


def _compute_text_metrics_core(
    predicted_text: str,
    gold_text: str,
    language: str,
    script: str,
) -> PageEvaluationMetrics:
    bleu_result = 0.0
    normalized_levenshtein_similarity = 0.0
    has_repetition_flag = False
    is_empty = 0.0

    try:
        predicted_text = strip_ocr_output_tag(predicted_text)
        predicted_text = extract_actual_text_from_xml(predicted_text)
        predicted_text = normalize_text_for_evaluation(
            predicted_text,
            normalize_arabic=language in {"Arabic", "Persian"},
        )
        is_empty = 1.0 if not predicted_text.strip() else 0.0

        gold_text = strip_ocr_output_tag(gold_text)
        gold_text = extract_actual_text_from_xml(gold_text)
        gold_text = normalize_text_for_evaluation(
            gold_text,
            normalize_arabic=language in {"Arabic", "Persian"},
        )

        denominator = max(len(predicted_text), len(gold_text))
        if denominator == 0:
            normalized_levenshtein_similarity = 1.0
        else:
            normalized_levenshtein_similarity = (
                1 - levenshtein_distance(predicted_text, gold_text) / denominator
            )

        has_repetition_flag = has_long_repetition(predicted_text)

        if is_empty != 1.0:
            assert bleu_metric is not None
            bleu_result = bleu_metric.compute(
                predictions=[predicted_text],
                references=[[gold_text]],
            )["bleu"]
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Error in metric computation: %s", exc)

    return {
        "normalized_levenshtein_similarity": normalized_levenshtein_similarity,
        "repetition": float(has_repetition_flag),
        "is_empty": is_empty,
        "bleu": bleu_result,
        "normalized_predicted_text": predicted_text,
        "normalized_gold_text": gold_text,
        "main_language": language,
        "main_script": script,
    }


def _extract_metric_inputs(example: MetricInputExample) -> tuple[str, str, str, str]:
    """Extract the fields needed to compute metrics for one example."""
    return (
        str(example["cleaned_transcription"]),
        str(example["main_language"]),
        str(example["main_script"]),
        str(example["example_id"]),
    )


def _build_failed_metrics(
    *,
    predicted_text: str,
    gold_text: str,
    language: str,
    script: str,
) -> PageEvaluationMetrics:
    """Return the fallback metric row used when evaluation fails."""
    return {
        "normalized_levenshtein_similarity": 0.0,
        "repetition": 0.0,
        "is_empty": 1.0,
        "bleu": 0.0,
        "normalized_predicted_text": predicted_text,
        "normalized_gold_text": gold_text,
        "main_language": language,
        "main_script": script,
    }


def calculate_metrics_from_text(
    predicted_text: str,
    gold_text: str,
    language: str,
    script: str,
) -> PageEvaluationMetrics:
    """Evaluate metrics from raw predicted and gold text."""
    initialize_metrics()
    return _compute_text_metrics_core(predicted_text, gold_text, language, script)


def calculate_metrics(inputs: tuple[MetricInputExample, str]) -> PageEvaluationMetrics:
    """Evaluate metrics for one dataset example."""
    example, predicted_text = inputs
    gold_text, main_language, main_script, example_id = _extract_metric_inputs(example)
    try:
        return _compute_text_metrics_core(
            predicted_text=predicted_text,
            gold_text=gold_text,
            language=main_language,
            script=main_script,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Error in evaluation of %s: %s", example_id, exc)
        return _build_failed_metrics(
            predicted_text=predicted_text,
            gold_text=gold_text,
            language=main_language,
            script=main_script,
        )


def evaluate_page(inputs: tuple[EvaluationExample, str]) -> PageEvaluationResult:
    """Evaluate a single predicted transcription against one dataset example."""
    example, predicted_text = inputs
    metrics_dict = calculate_metrics((example, predicted_text))
    metrics_dict["example_id"] = example["example_id"]
    metrics_dict["predicted_text"] = predicted_text
    metrics_dict["gold_text"] = example["cleaned_transcription"]
    metrics_dict["main_language"] = example["main_language"]
    metrics_dict["main_script"] = example["main_script"]
    metrics_dict["document_type"] = example["document_type"]
    metrics_dict["dataset_id"] = example["dataset_id"]
    return metrics_dict


def aggregate_results(
    results: list[PageEvaluationResult],
) -> tuple[dict[str, float], list[PageEvaluationResult]]:
    """Average numeric metrics across all page-level results."""
    if not results:
        return {}, []

    aggregated_metrics = {
        key: 0.0 for key, value in results[0].items() if isinstance(value, int | float | bool)
    }
    for metric_row in results:
        for key, value in metric_row.items():
            if key in aggregated_metrics and isinstance(value, int | float | bool):
                aggregated_metrics[key] += float(value)
    averaged = {key: value / len(results) for key, value in aggregated_metrics.items()}
    return averaged, results


def batch_evaluate(
    dataset: list[EvaluationExample],
    predicted_texts: list[str],
) -> tuple[dict[str, float], list[PageEvaluationResult]]:
    """Evaluate pages in parallel and return aggregate plus per-example metrics."""
    initialize_metrics()
    if len(dataset) <= 1:
        results = [evaluate_page(pair) for pair in zip(dataset, predicted_texts, strict=False)]
        return aggregate_results(results)

    processes = min(8, max(1, multiprocessing.cpu_count()))
    with multiprocessing.Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap(evaluate_page, zip(dataset, predicted_texts, strict=False)),
                total=len(dataset),
                mininterval=0.5,
            )
        )
    return aggregate_results(results)

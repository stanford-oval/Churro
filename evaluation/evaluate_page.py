import multiprocessing
from typing import Any

import evaluate
from rapidfuzz import distance as rf_distance
from tqdm import tqdm

from evaluation.normalization import normalize_text_for_evaluation
from evaluation.repetition import has_long_repetition
from evaluation.xml_utils import extract_actual_text_from_xml
from utils.log_utils import logger


bleu_metric = None


def initialize_metrics():
    """Lazily load BLEU metric and required NLTK resources."""
    global bleu_metric
    if not bleu_metric:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        bleu_metric = evaluate.load("bleu")


def levenshtein_distance(a: str, b: str, max_cost: int | None = None) -> int:
    """Compute Levenshtein distance with optional cutoff.

    Stops early if `max_cost` is exceeded (using RapidFuzz score cutoff).
    """
    if max_cost is not None:
        return rf_distance.Levenshtein.distance(a, b, score_cutoff=max_cost)
    return rf_distance.Levenshtein.distance(a, b)


def evaluate_page(
    _input: tuple,
) -> dict[str, Any]:
    """Evaluate predicted text vs. gold page returning a metrics dict."""
    example, predicted_text = _input

    metrics_dict = calculate_metrics((example, predicted_text))

    metrics_dict["file_name"] = example["file_name"]
    metrics_dict["predicted_text"] = predicted_text
    metrics_dict["gold_text"] = example["transcription"]
    metrics_dict["main_language"] = example["main_language"]
    metrics_dict["main_script"] = example["main_script"]
    metrics_dict["document_type"] = example["document_type"]
    metrics_dict["dataset_id"] = example["dataset_id"]

    return metrics_dict


def _compute_text_metrics_core(
    predicted_text: str,
    gold_text: str,
    language: str,
    script: str,
) -> dict[str, Any]:
    """Core text metric computation independent of ChurroExample objects.

    Args:
        predicted_text: The model predicted transcription (raw or XML).
        gold_text: The gold transcription (raw or XML).
        language: Primary language label (affects normalization rules).
        script: Primary script label.

    Returns:
        Dictionary containing evaluation metrics and normalized texts.
    """
    global bleu_metric

    bleu_result: float = 0.0
    normalized_levenshtein_similarity: float = 0.0
    has_repetition_flag: bool = False
    is_empty: float = 0.0

    try:
        predicted_text = extract_actual_text_from_xml(predicted_text)
        predicted_text = normalize_text_for_evaluation(
            predicted_text, normalize_arabic=(language in ["Arabic", "Persian"])
        )
        is_empty = 1.0 if not predicted_text.strip() else 0.0

        gold_text = extract_actual_text_from_xml(gold_text)
        gold_text_lines: list[str] = gold_text.splitlines()
        gold_text_lines = [normalize_text_for_evaluation(line) for line in gold_text_lines]

        gold_text = normalize_text_for_evaluation(
            gold_text, normalize_arabic=(language in ["Arabic", "Persian"])
        )

        denom: int = max(len(predicted_text), len(gold_text))
        if denom == 0:
            normalized_levenshtein_similarity = 1.0
        else:
            normalized_levenshtein_similarity = (
                1 - levenshtein_distance(predicted_text, gold_text) / denom
            )

        has_repetition_flag = has_long_repetition(predicted_text)

        if is_empty == 1.0:
            bleu_result = 0.0
        else:
            bleu_result = bleu_metric.compute(  # type: ignore
                predictions=[predicted_text],
                references=[[gold_text]],
            )["bleu"]
    except Exception as e:
        logger.error(f"Error in metric computation: {e}")

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


def calculate_metrics_from_text(
    predicted_text: str,
    gold_text: str,
    language: str,
    script: str,
) -> dict[str, Any]:
    """Evaluate metrics from raw predicted and gold text inputs."""
    return _compute_text_metrics_core(predicted_text, gold_text, language, script)


def calculate_metrics(_input: tuple) -> dict[str, Any]:
    """Wrapper for multiprocessing: (ChurroExample, predicted_text) -> metrics."""
    example, predicted_text = _input
    try:
        metrics = _compute_text_metrics_core(
            predicted_text=predicted_text,
            gold_text=example["transcription"],
            language=example["main_language"],
            script=example["main_script"],
        )
    except Exception as e:
        logger.error(f"Error in evaluation of {example['file_name']}: {e}")
        metrics = {
            "normalized_levenshtein_similarity": 0.0,
            "repetition": 0.0,
            "is_empty": 1.0,
            "bleu": 0.0,
            "normalized_predicted_text": predicted_text,
            "normalized_gold_text": example.transcription,
            "main_language": example.main_language,
            "main_script": example.main_script,
        }
    return metrics


def aggregate_results(results):
    """Average numeric metrics across page-level results (bools treated as 0/1)."""
    if not results:
        return {}, []

    # Initialize all numeric keys to 0.0
    aggregated_metrics: dict[str, float] = {}
    for k, v in results[0].items():
        if isinstance(v, (int, float, bool)):
            aggregated_metrics[k] = 0.0

    # Sum numeric values (treat bool as int -> 0/1)
    for m in results:
        for k, v in m.items():
            if isinstance(v, (int, float, bool)) and k in aggregated_metrics:
                aggregated_metrics[k] += float(v)

    # Average
    aggregated_metrics = {k: v / len(results) for k, v in aggregated_metrics.items()}
    return aggregated_metrics, results


def batch_evaluate(
    dataset: list[dict],
    predicted_texts: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate pages in parallel and return (aggregated_metrics, per_page)."""
    initialize_metrics()
    results = []
    with multiprocessing.Pool(processes=8) as pool:
        results = list(
            tqdm(
                pool.imap(evaluate_page, zip(dataset, predicted_texts)),
                total=len(dataset),
                mininterval=0.5,  # Update at most twice a second
            )
        )

    return aggregate_results(results)

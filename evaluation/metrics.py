from collections import defaultdict
import json
from typing import Any

from evaluation.evaluate_page import batch_evaluate
from ocr.systems.detect_layout import get_total_azure_cost
from utils.llm.cost import get_llm_total_cost
from utils.log_utils import logger


def to_rounded_percentage(metrics: dict[str, Any]) -> dict[str, Any]:
    """Round numeric metric values to one decimal percentage points."""
    return {
        key: round(value * 100) if isinstance(value, (int, float)) else value
        for key, value in metrics.items()
    }


def round(value: float | int) -> float:
    """Round a float to one decimal place."""
    return float(f"{value:.1f}")


def calculate_language_and_type_metrics(
    outputs: list[dict[str, Any]],
    main_metric: str = "normalized_levenshtein_similarity",
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute averages grouped by language, type, and their combination."""
    main_language_to_metrics = defaultdict(list)
    main_language_and_type_to_metrics = defaultdict(list)
    type_to_metrics = {"print": [], "handwriting": []}

    for output in outputs:
        main_language = output["main_language"]
        metric = output[main_metric]
        document_type = output["document_type"]

        main_language_to_metrics[main_language].append(metric)
        type_to_metrics[document_type].append(metric)
        main_language_and_type_to_metrics[f"{main_language}_{document_type}"].append(metric)

    # Calculate averages
    averaged_main_language: dict[str, float] = {}
    for language, vals in main_language_to_metrics.items():
        averaged_main_language[language] = sum(vals) / len(vals) if vals else 0.0

    averaged_type: dict[str, float] = {}
    for doc_type, vals in type_to_metrics.items():
        averaged_type[doc_type] = sum(vals) / len(vals) if vals else 0.0

    averaged_lang_type: dict[str, float] = {}
    for lt, vals in main_language_and_type_to_metrics.items():
        averaged_lang_type[lt] = sum(vals) / len(vals) if vals else 0.0
    return averaged_main_language, averaged_type, averaged_lang_type


def compute_metrics(
    dataset: list[dict],
    predicted_texts: list[str],
    output_prefix: str,
    elapsed_time: float,
    main_metric: str = "normalized_levenshtein_similarity",
) -> dict[str, Any]:
    """Compute metrics, save per-example outputs, plots, and summary JSON."""
    # Ensure all predicted texts are strings
    for i in range(len(predicted_texts)):
        if not predicted_texts[i]:
            predicted_texts[i] = ""

    # Run evaluation
    aggregate_metrics, per_example_evaluation_outputs = batch_evaluate(dataset, predicted_texts)

    # Create outputs with evaluation results
    outputs = []
    for evaluation_output, example in zip(per_example_evaluation_outputs, dataset):
        outputs.append(
            {
                "file_name": example["file_name"],
                **evaluation_output,
            }
        )

    # Save detailed outputs
    with open(f"{output_prefix}/outputs.json", "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    # Calculate metrics by language and type
    language_metrics, type_metrics, lang_type_metrics = calculate_language_and_type_metrics(
        outputs, main_metric
    )

    # Round all metrics
    language_metrics = to_rounded_percentage(language_metrics)
    type_metrics = to_rounded_percentage(type_metrics)
    aggregate_metrics = to_rounded_percentage(aggregate_metrics)
    lang_type_metrics = to_rounded_percentage(lang_type_metrics)
    aggregate_metrics["llm_cost ($)"] = round(get_llm_total_cost())  # don't convert to percentage
    aggregate_metrics["azure_cost ($)"] = round(
        get_total_azure_cost()
    )  # don't convert to percentage
    aggregate_metrics["elapsed_time (s)"] = round(elapsed_time)  # don't convert to percentage

    # Create combined metrics dictionary
    combined_metrics = {
        "main_language_metrics": language_metrics,
        "type_metrics": type_metrics,
        "aggregate_metrics": aggregate_metrics,
        "main_language_and_type_metrics": lang_type_metrics,
    }

    # Save all metrics
    with open(f"{output_prefix}/all_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    # Log results
    logger.info("Average metrics per document type:")
    logger.info(json.dumps(type_metrics, indent=2))

    logger.info("Average metrics per main language:")
    logger.info(json.dumps(language_metrics, indent=2))

    logger.info("Average metrics per main language and type:")
    logger.info(json.dumps(lang_type_metrics, indent=2))

    logger.info("Aggregated metrics:")
    logger.info(json.dumps(aggregate_metrics, indent=2))

    return combined_metrics

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from tooling.evaluation import metrics

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tooling.evaluation.types import BenchmarkPrediction, EvaluationExample, PageEvaluationResult


def test_calculate_language_and_type_metrics_handles_missing_categories() -> None:
    outputs = [
        cast(
            "PageEvaluationResult",
            {
                "main_language": "english",
                "document_type": "print",
                "normalized_levenshtein_similarity": 0.8,
            },
        ),
        cast(
            "PageEvaluationResult",
            {
                "main_language": "spanish",
                "document_type": "handwriting",
                "normalized_levenshtein_similarity": 0.6,
            },
        ),
    ]

    language_metrics, type_metrics, combined_metrics = metrics.calculate_language_and_type_metrics(outputs)

    assert language_metrics == {"english": 0.8, "spanish": 0.6}
    assert type_metrics == {"print": 0.8, "handwriting": 0.6}
    assert combined_metrics == {"english_print": 0.8, "spanish_handwriting": 0.6}


def test_to_rounded_percentage_preserves_non_numeric_values() -> None:
    source = {"metric": 0.756, "label": "keep"}
    result = metrics.to_rounded_percentage(source)
    assert result == {"metric": 75.6, "label": "keep"}


def test_compute_metrics_writes_expected_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset: list[EvaluationExample] = [
        {
            "example_id": "file1",
            "cleaned_transcription": "",
            "dataset_id": "ahisto",
            "main_language": "english",
            "main_script": "Latin",
            "document_type": "print",
        },
    ]
    predictions: list[BenchmarkPrediction] = [{"text": "", "metadata": {"raw_html": "<p></p>"}}]

    def fake_batch_evaluate(
        ds: list[EvaluationExample],
        preds: list[str],
    ) -> tuple[dict[str, float], list[PageEvaluationResult]]:
        assert ds == dataset
        assert preds == [""]
        return (
            {"normalized_levenshtein_similarity": 0.9},
            [
                cast(
                    "PageEvaluationResult",
                    {
                        "main_language": "english",
                        "document_type": "print",
                        "normalized_levenshtein_similarity": 0.9,
                    },
                )
            ],
        )

    monkeypatch.setattr(metrics, "batch_evaluate", fake_batch_evaluate)
    monkeypatch.setattr(metrics, "_get_llm_total_cost", lambda: 1.234)
    monkeypatch.setattr(metrics, "_get_azure_total_cost", lambda: 2.345)

    output_prefix = tmp_path / "results"
    combined = metrics.compute_metrics(
        dataset=dataset,
        predictions=predictions,
        output_prefix=output_prefix,
        elapsed_time=4.567,
    )

    outputs = json.loads((output_prefix / "outputs.json").read_text())
    assert outputs == [
        {
            "example_id": "file1",
            "metadata": {"raw_html": "<p></p>"},
            "main_language": "english",
            "document_type": "print",
            "normalized_levenshtein_similarity": 0.9,
        }
    ]

    all_metrics = json.loads((output_prefix / "all_metrics.json").read_text())
    assert all_metrics["aggregate_metrics"]["llm_cost ($)"] == 1.2
    assert all_metrics["aggregate_metrics"]["azure_cost ($)"] == 2.3
    assert all_metrics["aggregate_metrics"]["elapsed_time (s)"] == 4.6
    assert combined == all_metrics

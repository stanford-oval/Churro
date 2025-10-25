from __future__ import annotations

import json
from pathlib import Path

import pytest

from churro.evaluation import metrics


def test_calculate_language_and_type_metrics_handles_missing_categories() -> None:
    outputs = [
        {
            "main_language": "english",
            "document_type": "print",
            "normalized_levenshtein_similarity": 0.8,
        },
        {
            "main_language": "spanish",
            "document_type": "handwriting",
            "normalized_levenshtein_similarity": 0.6,
        },
    ]

    language_metrics, type_metrics, combined_metrics = metrics.calculate_language_and_type_metrics(
        outputs
    )

    assert language_metrics == {"english": 0.8, "spanish": 0.6}
    assert type_metrics == {"print": 0.8, "handwriting": 0.6}
    assert combined_metrics == {
        "english_print": 0.8,
        "spanish_handwriting": 0.6,
    }


def test_to_rounded_percentage_preserves_non_numeric_values() -> None:
    source = {"metric": 0.756, "label": "keep"}
    result = metrics.to_rounded_percentage(source)
    assert result == {"metric": 75.6, "label": "keep"}


def test_compute_metrics_writes_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = [
        {"file_name": "file1", "main_language": "english", "document_type": "print"},
    ]
    predicted_texts = [""]  # exercise empty-text correction

    def fake_batch_evaluate(
        ds: list[dict[str, str]],
        preds: list[str],
    ) -> tuple[
        dict[str, float],
        list[dict[str, float | str]],
    ]:
        assert preds == [""]
        return (
            {"normalized_levenshtein_similarity": 0.9},
            [
                {
                    "main_language": "english",
                    "document_type": "print",
                    "normalized_levenshtein_similarity": 0.9,
                }
            ],
        )

    monkeypatch.setattr(metrics, "batch_evaluate", fake_batch_evaluate)
    monkeypatch.setattr(metrics, "get_llm_total_cost", lambda: 1.234)
    monkeypatch.setattr(metrics, "get_total_azure_cost", lambda: 2.345)

    output_prefix = tmp_path / "results"
    output_prefix.mkdir()

    combined = metrics.compute_metrics(
        dataset=dataset,
        predicted_texts=predicted_texts,
        output_prefix=str(output_prefix),
        elapsed_time=4.567,
    )

    outputs = json.loads((output_prefix / "outputs.json").read_text())
    assert outputs == [
        {
            "file_name": "file1",
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

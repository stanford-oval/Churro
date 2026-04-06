from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from tooling.evaluation.types import EvaluationExample, MetricInputExample

evaluate_page_module = importlib.import_module("tooling.evaluation.evaluate_page")


def test_evaluate_page_supports_current_example_fields(monkeypatch) -> None:
    example: EvaluationExample = {
        "example_id": "ahisto/1069_69",
        "cleaned_transcription": "<xml>clean</xml>",
        "main_language": "Czech",
        "main_script": "Latin",
        "document_type": "print",
        "dataset_id": "ahisto",
    }

    monkeypatch.setattr(
        evaluate_page_module,
        "calculate_metrics",
        lambda _: {"normalized_levenshtein_similarity": 1.0},
    )

    result = evaluate_page_module.evaluate_page((example, "predicted"))

    assert result["example_id"] == "ahisto/1069_69"
    assert result["gold_text"] == "<xml>clean</xml>"
    assert result["predicted_text"] == "predicted"
    assert result["main_language"] == "Czech"
    assert result["main_script"] == "Latin"
    assert result["document_type"] == "print"
    assert result["dataset_id"] == "ahisto"


def test_calculate_metrics_uses_cleaned_transcription(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_core(predicted_text: str, gold_text: str, language: str, script: str) -> dict[str, object]:
        captured["predicted_text"] = predicted_text
        captured["gold_text"] = gold_text
        captured["language"] = language
        captured["script"] = script
        return {"normalized_levenshtein_similarity": 1.0}

    monkeypatch.setattr(evaluate_page_module, "_compute_text_metrics_core", fake_core)

    example: MetricInputExample = {
        "example_id": "ahisto/1069_69",
        "cleaned_transcription": "new",
        "main_language": "Czech",
        "main_script": "Latin",
    }

    result = evaluate_page_module.calculate_metrics((example, "pred"))

    assert captured == {
        "predicted_text": "pred",
        "gold_text": "new",
        "language": "Czech",
        "script": "Latin",
    }
    assert result["normalized_levenshtein_similarity"] == 1.0


@pytest.mark.parametrize(
    ("predicted_text", "expected"),
    [
        ("<output> Pred", "pred"),
        ("Pred </output>", "pred"),
        ("<output>\nPred\n</output>", "pred"),
    ],
)
def test_calculate_metrics_strips_output_tags_before_normalization(
    monkeypatch,
    predicted_text: str,
    expected: str,
) -> None:
    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", lambda: None)
    monkeypatch.setattr(
        evaluate_page_module,
        "bleu_metric",
        SimpleNamespace(compute=lambda *_args, **_kwargs: {"bleu": 0.0}),
    )

    example: MetricInputExample = {
        "example_id": "ahisto/1069_69",
        "cleaned_transcription": "gold",
        "main_language": "Czech",
        "main_script": "Latin",
    }

    result = evaluate_page_module.calculate_metrics((example, predicted_text))

    assert result["normalized_predicted_text"] == expected


def test_calculate_metrics_from_text_lazily_initializes_bleu_metric(monkeypatch) -> None:
    init_calls = 0
    fake_metric = SimpleNamespace(compute=lambda *_args, **_kwargs: {"bleu": 0.25})

    def fake_initialize_metrics() -> None:
        nonlocal init_calls
        init_calls += 1
        monkeypatch.setattr(evaluate_page_module, "bleu_metric", fake_metric)

    monkeypatch.setattr(evaluate_page_module, "bleu_metric", None)
    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", fake_initialize_metrics)

    result = evaluate_page_module.calculate_metrics_from_text("pred", "gold", "English", "Latin")

    assert init_calls == 1
    assert result["bleu"] == 0.25

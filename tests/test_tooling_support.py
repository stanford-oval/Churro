from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import cast

import datasets
import pytest
from PIL import Image

import tooling.benchmarking.dataset as dataset_module
import tooling.evaluation.normalization as normalization_module
import tooling.evaluation.xml_utils as xml_utils_module
from tooling.evaluation.repetition import has_long_repetition
from tooling.evaluation.types import BenchmarkDatasetExample, MetricInputExample, PageEvaluationResult

evaluate_page_module = importlib.import_module("tooling.evaluation.evaluate_page")


def test_extract_actual_text_from_xml_handles_plain_text_namespaces_and_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []

    class _FakeLogger:
        def warning(self, message: str, *args: object) -> None:
            warnings.append(message % args if args else message)

    monkeypatch.setattr("tooling.evaluation.xml_utils.logger", _FakeLogger())

    xml_content = """
    <HistoricalDocument xmlns="urn:test">
      <Page>
        <Header>Header line</Header>
        <Body>Body line</Body>
        <Footer>Footer line</Footer>
        <Description>Ignore me</Description>
      </Page>
    </HistoricalDocument>
    """

    assert xml_utils_module.extract_actual_text_from_xml("plain text") == "plain text"
    assert xml_utils_module.extract_actual_text_from_xml(xml_content) == "Header line\nBody line\nFooter line"
    assert xml_utils_module.extract_actual_text_from_xml("<HistoricalDocument>") == ""
    assert warnings and "Failed to parse XML content during evaluation" in warnings[0]


def test_normalize_text_for_evaluation_handles_markdown_linebreaks_and_substitutions() -> None:
    text = "A~word\n![img](x)\n[figure 3]\nfoo-\nbar – baz ſ \ueada"

    normalized = normalization_module.normalize_text_for_evaluation(text)

    assert normalized == "aword foobar - baz s st"


def test_normalize_text_for_evaluation_converts_markdown_with_embedded_html_to_plain_text() -> None:
    text = (
        "# Heading\n\n"
        "<table><tr><td>Year</td><td>Value</td></tr><tr><td>1900</td><td>42</td></tr></table>\n\n"
        "- Bullet item"
    )

    normalized = normalization_module.normalize_text_for_evaluation(text)

    assert normalized == "heading year | value 1900 | 42 bullet item"


def test_normalize_text_for_evaluation_supports_arabic_normalization_and_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(normalization_module, "strip_tashkeel", lambda value: value + "1")
    monkeypatch.setattr(normalization_module, "strip_harakat", lambda value: value + "2")
    monkeypatch.setattr(normalization_module, "strip_lastharaka", lambda value: value + "3")
    monkeypatch.setattr(normalization_module, "strip_tatweel", lambda value: value + "4")
    monkeypatch.setattr(normalization_module, "normalize_hamza", lambda value: value + "5")

    assert normalization_module.normalize_text_for_evaluation("AR", normalize_arabic=True) == "ar12345"

    monkeypatch.setattr(normalization_module, "strip_tashkeel", None)
    with pytest.raises(ModuleNotFoundError, match="pyarabic"):
        normalization_module.normalize_text_for_evaluation("AR", normalize_arabic=True)


def test_has_long_repetition_distinguishes_repeated_suffixes() -> None:
    assert has_long_repetition("a") is False
    assert has_long_repetition("abcdef") is False
    assert has_long_repetition("xyzxyzxyz") is True


def test_load_dataset_split_uses_parquet_shards_and_falls_back_to_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_calls: list[tuple[str, dict[str, object]]] = []

    class _BuilderWithFiles:
        config = SimpleNamespace(data_files={"dev": ["dev.parquet"]})
        info = SimpleNamespace(features={"keep": "feature-keep", "other": "feature-other"})

    class _BuilderWithoutFiles:
        config = SimpleNamespace(data_files={})
        info = SimpleNamespace(features=None)

    monkeypatch.setattr(datasets, "Features", lambda mapping: {"wrapped": mapping})
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda name, **kwargs: parquet_calls.append((name, kwargs)) or {"name": name, "kwargs": kwargs},
    )
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda dataset_id: _BuilderWithFiles())

    parquet_result = dataset_module.load_dataset_split("dataset/id", "dev", columns=["keep"])

    monkeypatch.setattr(datasets, "load_dataset_builder", lambda dataset_id: _BuilderWithoutFiles())
    fallback_result = dataset_module.load_dataset_split("dataset/id", "test")

    assert parquet_result == {
        "name": "parquet",
        "kwargs": {
            "data_files": {"dev": ["dev.parquet"]},
            "split": "dev",
            "columns": ["keep"],
            "features": {"wrapped": {"keep": "feature-keep"}},
        },
    }
    assert fallback_result == {
        "name": "dataset/id",
        "kwargs": {"split": "test"},
    }


def test_dataset_subset_and_selection_cover_iterable_and_materialized_paths() -> None:
    subset = dataset_module.DatasetSubset.from_raw(language=" English ", document_type="Handwritten Page")
    selection = dataset_module.DatasetSelection(subset=subset, offset=1, limit=1)

    examples: list[BenchmarkDatasetExample] = [
        {
            "image": Image.new("RGB", (4, 4), color="white"),
            "cleaned_transcription": "",
            "dataset_id": "dataset-1",
            "document_type": "handwritten page",
            "example_id": "one",
            "main_language": "english",
            "main_script": "Latin",
        },
        {
            "image": Image.new("RGB", (4, 4), color="white"),
            "cleaned_transcription": "",
            "dataset_id": "dataset-2",
            "document_type": "handwritten page",
            "example_id": "two",
            "main_language": "english",
            "main_script": "Latin",
        },
        {
            "image": Image.new("RGB", (4, 4), color="white"),
            "cleaned_transcription": "",
            "dataset_id": "dataset-3",
            "document_type": "print",
            "example_id": "three",
            "main_language": "english",
            "main_script": "Latin",
        },
    ]

    assert subset.is_active() is True
    assert subset.output_suffixes() == ["language_english", "document_type_handwritten_page"]
    assert [example["example_id"] for example in selection.select(examples)] == ["two"]

    materialized = datasets.Dataset.from_list(
        [
            {
                "main_language": "english",
                "document_type": "handwritten page",
                "example_id": "one",
            },
            {
                "main_language": "english",
                "document_type": "handwritten page",
                "example_id": "two",
            },
            {
                "main_language": "english",
                "document_type": "print",
                "example_id": "three",
            },
        ]
    )
    selected = selection._select_materialized_dataset(materialized)
    assert selected.num_rows == 1
    assert cast("str", selected[0]["example_id"]) == "two"

    no_count_dataset = SimpleNamespace(filter=lambda *_args, **_kwargs: "filtered", num_rows="unknown")
    assert selection._select_materialized_dataset(no_count_dataset) == "filtered"


def test_evaluate_page_helpers_cover_failure_and_aggregation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example: MetricInputExample = {
        "example_id": "example-1",
        "cleaned_transcription": "gold",
        "main_language": "English",
        "main_script": "Latin",
    }

    monkeypatch.setattr(
        evaluate_page_module,
        "_compute_text_metrics_core",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed = evaluate_page_module.calculate_metrics((example, "predicted"))
    assert failed["is_empty"] == 1.0
    assert failed["normalized_gold_text"] == "gold"

    assert evaluate_page_module.aggregate_results([]) == ({}, [])
    aggregate, rows = evaluate_page_module.aggregate_results(
        [
            cast(
                "PageEvaluationResult",
                {
                    "example_id": "one",
                    "normalized_levenshtein_similarity": 0.5,
                    "is_empty": 0.0,
                },
            ),
            cast(
                "PageEvaluationResult",
                {
                    "example_id": "two",
                    "normalized_levenshtein_similarity": 1.0,
                    "is_empty": 1.0,
                },
            ),
        ]
    )
    assert aggregate == {"normalized_levenshtein_similarity": 0.75, "is_empty": 0.5}
    assert len(rows) == 2


def test_evaluate_page_metric_helpers_cover_initialization_and_single_batch_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert evaluate_page_module.levenshtein_distance("abc", "adc", max_cost=1) == 1

    monkeypatch.setattr(evaluate_page_module, "nltk", None)
    monkeypatch.setattr(evaluate_page_module, "bleu_metric", None)
    with pytest.raises(ModuleNotFoundError, match="nltk"):
        evaluate_page_module.initialize_metrics()

    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", lambda: None)
    monkeypatch.setattr(
        evaluate_page_module,
        "evaluate_page",
        lambda inputs: cast(
            "PageEvaluationResult",
            {
                "example_id": inputs[0]["example_id"],
                "normalized_levenshtein_similarity": 1.0,
                "is_empty": 0.0,
            },
        ),
    )
    aggregate, rows = evaluate_page_module.batch_evaluate(
        dataset=[
            cast(
                "BenchmarkDatasetExample",
                {
                    "image": "image",
                    "cleaned_transcription": "",
                    "dataset_id": "dataset-1",
                    "document_type": "print",
                    "example_id": "row-1",
                    "main_language": "English",
                    "main_script": "Latin",
                },
            )
        ],
        predicted_texts=["predicted"],
    )

    assert aggregate == {"normalized_levenshtein_similarity": 1.0, "is_empty": 0.0}
    assert rows[0]["example_id"] == "row-1"


def test_batch_evaluate_initializes_worker_metrics_for_multi_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls = 0
    captured_initializer = None

    def fake_initialize_metrics() -> None:
        nonlocal init_calls
        init_calls += 1

    class _FakePool:
        def __init__(self, *, processes: int, initializer) -> None:  # noqa: ANN001
            nonlocal captured_initializer
            assert processes == 2
            captured_initializer = initializer

        def __enter__(self) -> _FakePool:
            assert captured_initializer is not None
            captured_initializer()
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        def imap(self, func, iterable):  # noqa: ANN001
            return map(func, iterable)

    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", fake_initialize_metrics)
    monkeypatch.setattr(evaluate_page_module, "_should_use_multiprocessing_pool", lambda: True)
    monkeypatch.setattr(evaluate_page_module.multiprocessing, "cpu_count", lambda: 2)
    monkeypatch.setattr(evaluate_page_module.multiprocessing, "Pool", _FakePool)
    monkeypatch.setattr(
        evaluate_page_module,
        "evaluate_page",
        lambda inputs: cast(
            "PageEvaluationResult",
            {
                "example_id": inputs[0]["example_id"],
                "normalized_levenshtein_similarity": 1.0,
                "is_empty": 0.0,
            },
        ),
    )

    aggregate, rows = evaluate_page_module.batch_evaluate(
        dataset=[
            cast(
                "BenchmarkDatasetExample",
                {
                    "image": "image",
                    "cleaned_transcription": "",
                    "dataset_id": "dataset-1",
                    "document_type": "print",
                    "example_id": "row-1",
                    "main_language": "English",
                    "main_script": "Latin",
                },
            ),
            cast(
                "BenchmarkDatasetExample",
                {
                    "image": "image",
                    "cleaned_transcription": "",
                    "dataset_id": "dataset-2",
                    "document_type": "print",
                    "example_id": "row-2",
                    "main_language": "English",
                    "main_script": "Latin",
                },
            ),
        ],
        predicted_texts=["predicted-1", "predicted-2"],
    )

    assert captured_initializer is fake_initialize_metrics
    assert init_calls == 2
    assert aggregate == {"normalized_levenshtein_similarity": 1.0, "is_empty": 0.0}
    assert [row["example_id"] for row in rows] == ["row-1", "row-2"]


def test_batch_evaluate_uses_in_process_path_when_multiprocessing_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls = 0

    def fake_initialize_metrics() -> None:
        nonlocal init_calls
        init_calls += 1

    def _unexpected_pool(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("multiprocessing pool should not be used")

    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", fake_initialize_metrics)
    monkeypatch.setattr(evaluate_page_module, "_should_use_multiprocessing_pool", lambda: False)
    monkeypatch.setattr(evaluate_page_module.multiprocessing, "Pool", _unexpected_pool)
    monkeypatch.setattr(evaluate_page_module, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        evaluate_page_module,
        "evaluate_page",
        lambda inputs: cast(
            "PageEvaluationResult",
            {
                "example_id": inputs[0]["example_id"],
                "normalized_levenshtein_similarity": 1.0,
                "is_empty": 0.0,
            },
        ),
    )

    aggregate, rows = evaluate_page_module.batch_evaluate(
        dataset=[
            cast(
                "BenchmarkDatasetExample",
                {
                    "image": "image",
                    "cleaned_transcription": "",
                    "dataset_id": "dataset-1",
                    "document_type": "print",
                    "example_id": "row-1",
                    "main_language": "English",
                    "main_script": "Latin",
                },
            ),
            cast(
                "BenchmarkDatasetExample",
                {
                    "image": "image",
                    "cleaned_transcription": "",
                    "dataset_id": "dataset-2",
                    "document_type": "print",
                    "example_id": "row-2",
                    "main_language": "English",
                    "main_script": "Latin",
                },
            ),
        ],
        predicted_texts=["predicted-1", "predicted-2"],
    )

    assert init_calls == 1
    assert aggregate == {"normalized_levenshtein_similarity": 1.0, "is_empty": 0.0}
    assert [row["example_id"] for row in rows] == ["row-1", "row-2"]


def test_calculate_metrics_from_text_and_internal_error_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", lambda: None)
    monkeypatch.setattr(
        evaluate_page_module,
        "bleu_metric",
        SimpleNamespace(compute=lambda *_args, **_kwargs: {"bleu": 0.5}),
    )

    result = evaluate_page_module.calculate_metrics_from_text("pred", "gold", "English", "Latin")
    assert result["bleu"] == 0.5

    errors: list[str] = []

    class _FakeLogger:
        def error(self, message: str, *args: object) -> None:
            errors.append(message % args if args else message)

    monkeypatch.setattr(evaluate_page_module, "logger", _FakeLogger())
    monkeypatch.setattr(
        evaluate_page_module,
        "strip_ocr_output_tag",
        lambda text: (_ for _ in ()).throw(ValueError("bad")),
    )

    failed = evaluate_page_module._compute_text_metrics_core("pred", "gold", "English", "Latin")

    assert failed["normalized_levenshtein_similarity"] == 0.0
    assert failed["repetition"] == 0.0
    assert failed["is_empty"] == 0.0
    assert errors and "Error in metric computation: bad" in errors[0]

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Never, cast

import datasets
import pytest
from PIL import Image

import tooling.benchmarking.dataset as dataset_module
import tooling.evaluation.normalization as normalization_module
import tooling.evaluation.xml_utils as xml_utils_module
from tooling.evaluation.repetition import has_long_repetition

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from tooling.evaluation.types import BenchmarkDatasetExample, MetricInputExample, PageEvaluationResult

evaluate_page_module = importlib.import_module("tooling.evaluation.evaluate_page")


def _boom_error() -> RuntimeError:
    return RuntimeError("boom")


def _bad_value_error() -> ValueError:
    return ValueError("bad")


def _raise_core_metrics_error(*_args: object, **_kwargs: object) -> Never:
    raise _boom_error()


def _raise_bad_value_error(_text: str) -> Never:
    raise _bad_value_error()


def _metric_input_example(
    example_id: str,
    *,
    cleaned_transcription: str = "",
    main_language: str = "English",
    main_script: str = "Latin",
) -> MetricInputExample:
    return {
        "example_id": example_id,
        "cleaned_transcription": cleaned_transcription,
        "main_language": main_language,
        "main_script": main_script,
    }


def _benchmark_dataset_example(
    example_id: str,
    *,
    image: object = "image",
    cleaned_transcription: str = "",
    dataset_id: str | None = None,
    document_type: str = "print",
    main_language: str = "English",
    main_script: str = "Latin",
) -> BenchmarkDatasetExample:
    return cast(
        "BenchmarkDatasetExample",
        {
            "image": image,
            "cleaned_transcription": cleaned_transcription,
            "dataset_id": dataset_id or f"dataset-{example_id}",
            "document_type": document_type,
            "example_id": example_id,
            "main_language": main_language,
            "main_script": main_script,
        },
    )


def _page_evaluation_result(
    example_id: str,
    *,
    normalized_levenshtein_similarity: float = 1.0,
    is_empty: float = 0.0,
) -> PageEvaluationResult:
    return cast(
        "PageEvaluationResult",
        {
            "example_id": example_id,
            "normalized_levenshtein_similarity": normalized_levenshtein_similarity,
            "is_empty": is_empty,
        },
    )


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
    assert warnings
    assert "Failed to parse XML content during evaluation" in warnings[0]


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

    def _load_dataset_builder_with_files(dataset_id: str) -> _BuilderWithFiles:
        del dataset_id
        return _BuilderWithFiles()

    monkeypatch.setattr(datasets, "load_dataset_builder", _load_dataset_builder_with_files)

    parquet_result = dataset_module.load_dataset_split("dataset/id", "dev", columns=["keep"])

    def _load_dataset_builder_without_files(dataset_id: str) -> _BuilderWithoutFiles:
        del dataset_id
        return _BuilderWithoutFiles()

    monkeypatch.setattr(datasets, "load_dataset_builder", _load_dataset_builder_without_files)
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
        _benchmark_dataset_example(
            "one",
            image=Image.new("RGB", (4, 4), color="white"),
            document_type="handwritten page",
            main_language="english",
        ),
        _benchmark_dataset_example(
            "two",
            image=Image.new("RGB", (4, 4), color="white"),
            document_type="handwritten page",
            main_language="english",
        ),
        _benchmark_dataset_example(
            "three",
            image=Image.new("RGB", (4, 4), color="white"),
            main_language="english",
        ),
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
    selected = cast("datasets.Dataset", selection._select_materialized_dataset(materialized))
    assert selected.num_rows == 1
    assert cast("str", selected[0]["example_id"]) == "two"

    no_count_dataset = SimpleNamespace(filter=lambda *_args, **_kwargs: "filtered", num_rows="unknown")
    assert selection._select_materialized_dataset(no_count_dataset) == "filtered"


def test_evaluate_page_helpers_cover_failure_and_aggregation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _metric_input_example("example-1", cleaned_transcription="gold")

    monkeypatch.setattr(
        evaluate_page_module,
        "_compute_text_metrics_core",
        _raise_core_metrics_error,
    )
    failed = evaluate_page_module.calculate_metrics((example, "predicted"))
    assert failed["is_empty"] == 1.0
    assert failed["normalized_gold_text"] == "gold"

    assert evaluate_page_module.aggregate_results([]) == ({}, [])
    aggregate, rows = evaluate_page_module.aggregate_results(
        [
            _page_evaluation_result("one", normalized_levenshtein_similarity=0.5),
            _page_evaluation_result("two", is_empty=1.0),
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
        lambda inputs: _page_evaluation_result(str(inputs[0]["example_id"])),
    )
    aggregate, rows = evaluate_page_module.batch_evaluate(
        dataset=[_benchmark_dataset_example("row-1", dataset_id="dataset-1")],
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

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> bool:
            del exc_type, exc, tb
            return False

        def imap(self, func: Callable[[object], object], iterable: Iterable[object]) -> Iterator[object]:
            return map(func, iterable)

    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", fake_initialize_metrics)
    monkeypatch.setattr(evaluate_page_module, "_should_use_multiprocessing_pool", lambda: True)
    monkeypatch.setattr(evaluate_page_module.multiprocessing, "cpu_count", lambda: 2)
    monkeypatch.setattr(evaluate_page_module.multiprocessing, "Pool", _FakePool)
    monkeypatch.setattr(
        evaluate_page_module,
        "evaluate_page",
        lambda inputs: _page_evaluation_result(str(inputs[0]["example_id"])),
    )

    aggregate, rows = evaluate_page_module.batch_evaluate(
        dataset=[
            _benchmark_dataset_example("row-1", dataset_id="dataset-1"),
            _benchmark_dataset_example("row-2", dataset_id="dataset-2"),
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

    def _unexpected_pool(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        message = "multiprocessing pool should not be used"
        raise AssertionError(message)

    monkeypatch.setattr(evaluate_page_module, "initialize_metrics", fake_initialize_metrics)
    monkeypatch.setattr(evaluate_page_module, "_should_use_multiprocessing_pool", lambda: False)
    monkeypatch.setattr(evaluate_page_module.multiprocessing, "Pool", _unexpected_pool)
    monkeypatch.setattr(evaluate_page_module, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        evaluate_page_module,
        "evaluate_page",
        lambda inputs: _page_evaluation_result(str(inputs[0]["example_id"])),
    )

    aggregate, rows = evaluate_page_module.batch_evaluate(
        dataset=[
            _benchmark_dataset_example("row-1", dataset_id="dataset-1"),
            _benchmark_dataset_example("row-2", dataset_id="dataset-2"),
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
        _raise_bad_value_error,
    )

    failed = evaluate_page_module._compute_text_metrics_core("pred", "gold", "English", "Latin")

    assert failed["normalized_levenshtein_similarity"] == 0.0
    assert failed["repetition"] == 0.0
    assert failed["is_empty"] == 0.0
    assert errors
    assert "Error in metric computation: bad" in errors[0]

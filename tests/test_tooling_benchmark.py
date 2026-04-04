from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from datasets import Dataset
from PIL import Image

from churro_ocr.providers.hf import HuggingFaceVisionOCRBackend
from churro_ocr.providers.ocr import LiteLLMVisionOCRBackend
from churro_ocr.providers.specs import DEFAULT_OCR_MAX_TOKENS
from churro_ocr.providers.vllm import VLLMVisionOCRBackend
from churro_ocr.templates import CHURRO_3B_XML_TEMPLATE
from tooling.benchmarking import benchmark
from tooling.evaluation.types import BenchmarkDatasetExample


def _benchmark_example(
    example_id: str,
    *,
    size: tuple[int, int] = (8, 8),
    transcription: str = "",
    dataset_id: str | None = None,
    document_type: str = "print",
    main_language: str = "English",
    main_script: str = "Latin",
) -> BenchmarkDatasetExample:
    return {
        "image": Image.new("RGB", size, color="white"),
        "cleaned_transcription": transcription,
        "dataset_id": dataset_id or f"dataset-{example_id}",
        "document_type": document_type,
        "example_id": example_id,
        "main_language": main_language,
        "main_script": main_script,
    }


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--use-page-detection"],
        ["--page-detector", "azure"],
        ["--trim-margin", "30"],
        ["--dpi", "300"],
    ],
)
def test_parse_args_rejects_removed_page_detection_options(removed_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--backend",
                "azure",
                "--dataset-split",
                "dev",
                "--endpoint",
                "https://example.invalid",
                "--api-key",
                "secret",
                *removed_args,
            ]
        )


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--system-prompt", "system"],
        ["--prompt", "custom prompt"],
    ],
)
def test_parse_args_rejects_removed_prompt_override_options(removed_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--backend",
                "hf",
                "--dataset-split",
                "dev",
                "--model",
                "example/model",
                *removed_args,
            ]
        )


def test_validate_options_requires_model_for_litellm() -> None:
    options = benchmark.BenchmarkOptions(
        backend="litellm",
        model=None,
        dataset_split="dev",
    )
    assert benchmark._validate_options(options) == 1


def test_validate_options_requires_model_for_hf() -> None:
    options = benchmark.BenchmarkOptions(
        backend="hf",
        model=None,
        dataset_split="dev",
    )
    assert benchmark._validate_options(options) == 1


def test_validate_options_requires_model_for_vllm() -> None:
    options = benchmark.BenchmarkOptions(
        backend="vllm",
        model=None,
        dataset_split="dev",
    )
    assert benchmark._validate_options(options) == 1


def test_validate_options_rejects_invalid_vllm_gpu_memory_utilization() -> None:
    options = benchmark.BenchmarkOptions(
        backend="vllm",
        model="Qwen/Qwen3.5-0.8B",
        dataset_split="dev",
        vllm_gpu_memory_utilization=1.5,
    )
    assert benchmark._validate_options(options) == 1


def test_validate_options_rejects_invalid_split() -> None:
    options = benchmark.BenchmarkOptions(
        backend="azure",
        dataset_split="train",
        endpoint="https://example.invalid",
        api_key="secret",
    )
    assert benchmark._validate_options(options) == 1


def test_parse_args_accepts_subset_filters() -> None:
    options = benchmark.parse_args(
        [
            "--backend",
            "azure",
            "--dataset-split",
            "dev",
            "--endpoint",
            "https://example.invalid",
            "--api-key",
            "secret",
            "--language",
            "Chinese",
            "--document-type",
            "print",
        ]
    )

    assert options.language == "Chinese"
    assert options.document_type == "print"


def test_parse_args_accepts_vllm_resource_overrides() -> None:
    options = benchmark.parse_args(
        [
            "--backend",
            "vllm",
            "--dataset-split",
            "dev",
            "--model",
            "Qwen/Qwen3.5-0.8B",
            "--vllm-gpu-memory-utilization",
            "0.25",
            "--vllm-cpu-offload-gb",
            "8",
        ]
    )

    assert options.vllm_gpu_memory_utilization == pytest.approx(0.25)
    assert options.vllm_cpu_offload_gb == pytest.approx(8.0)


def test_build_ocr_backend_enables_disk_cache_for_litellm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "litellm-cache"

    monkeypatch.setattr(benchmark, "_default_litellm_cache_dir", lambda: cache_dir)

    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="litellm",
                dataset_split="dev",
                model="gpt-4.1-mini",
            )
        ),
    )

    assert backend.transport.config.cache_dir == cache_dir
    assert backend.transport.config.completion_kwargs == {"max_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_uses_dots_preset_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="kristaller486/dots.ocr-1.5",
            )
        ),
    )

    assert backend.model_name == "dots.ocr-1.5"
    assert backend.processor_kwargs == {}
    assert backend.trust_remote_code is True
    assert backend.model_kwargs["dtype"] in {"auto", "float32"}
    if backend.model_kwargs["dtype"] == "auto":
        assert backend.model_kwargs["device_map"] == "auto"
        assert "max_memory" in backend.model_kwargs
    assert backend.generation_kwargs == {"max_new_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_uses_dots_preset_for_vllm() -> None:
    backend = cast(
        "VLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="vllm",
                dataset_split="dev",
                model="kristaller486/dots.ocr-1.5",
            )
        ),
    )

    assert backend.model_name == "dots.ocr-1.5"
    assert backend.processor_kwargs == {}
    assert backend.trust_remote_code is True
    assert backend.sampling_kwargs == {"max_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_uses_churro_preset_template_for_vllm() -> None:
    backend = cast(
        "VLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="vllm",
                dataset_split="dev",
                model="stanford-oval/churro-3B",
            )
        ),
    )

    assert backend.template == CHURRO_3B_XML_TEMPLATE
    assert backend.model_name == "churro-3B"


def test_build_ocr_backend_uses_generic_qwen_model_name_for_vllm() -> None:
    backend = cast(
        "VLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="vllm",
                dataset_split="dev",
                model="Qwen/Qwen3.5-0.8B",
                vllm_gpu_memory_utilization=0.25,
                vllm_cpu_offload_gb=8.0,
            )
        ),
    )

    assert backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert backend.llm_kwargs == {
        "gpu_memory_utilization": 0.25,
        "cpu_offload_gb": 8.0,
    }
    assert backend.sampling_kwargs == {"max_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_aligns_hf_and_vllm_templates_for_generic_models() -> None:
    hf_backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="example/model",
            )
        ),
    )
    vllm_backend = cast(
        "VLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="vllm",
                dataset_split="dev",
                model="example/model",
            )
        ),
    )

    assert hf_backend.template == vllm_backend.template


@pytest.mark.asyncio
async def test_run_executes_pipeline(monkeypatch, tmp_path: Path) -> None:
    dataset: list[BenchmarkDatasetExample] = [
        _benchmark_example("0", transcription="first"),
        _benchmark_example(
            "1",
            transcription="second",
            dataset_id="dataset-1",
            document_type="handwriting",
            main_language="Persian",
            main_script="Arabic",
        ),
        _benchmark_example(
            "2",
            transcription="third",
            dataset_id="dataset-2",
            main_language="French",
        ),
    ]

    def fake_load_dataset(dataset_id: str, *, split: str):  # noqa: ANN001
        assert dataset_id == benchmark.CHURRO_DATASET_ID
        assert split == "dev"
        return dataset

    monkeypatch.setattr(benchmark, "_load_dataset", fake_load_dataset)

    async def fake_predict(ds, options, *, total_pages):  # noqa: ANN001
        selected = list(ds)
        assert len(selected) == 1
        assert selected[0]["example_id"] == "1"
        assert options.max_concurrency == 2
        assert total_pages is None
        return [benchmark._build_evaluation_example(selected[0])], ["prediction"]

    monkeypatch.setattr(benchmark, "_predict_texts", fake_predict)

    captured: dict[str, object] = {}

    def fake_compute_metrics(ds, predictions, output_prefix, elapsed_time):  # noqa: ANN001
        captured["dataset"] = ds
        captured["predictions"] = predictions
        captured["output_prefix"] = output_prefix
        captured["elapsed_time"] = elapsed_time
        return {"status": "ok"}

    monkeypatch.setattr(benchmark, "compute_metrics", fake_compute_metrics)
    time_values = iter([10.0, 13.5])
    monkeypatch.setattr(benchmark, "time", lambda: next(time_values))

    options = benchmark.BenchmarkOptions(
        backend="azure",
        dataset_split="dev",
        endpoint="https://example.invalid",
        api_key="secret",
        max_concurrency=2,
        input_size=1,
        offset=1,
        output_dir=tmp_path / "outputs",
    )

    result = await benchmark.run(options)

    assert result == 0
    assert captured["dataset"] == [benchmark._build_evaluation_example(dataset[1])]
    assert captured["predictions"] == ["prediction"]
    assert captured["output_prefix"] == str(tmp_path / "outputs")
    assert captured["elapsed_time"] == pytest.approx(3.5)


def test_create_output_prefix_includes_subset_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "__file__", str(tmp_path / "tooling" / "benchmarking" / "benchmark.py"))

    output_prefix = benchmark.create_output_prefix(
        benchmark.BenchmarkOptions(
            backend="azure",
            dataset_split="dev",
            endpoint="https://example.invalid",
            api_key="secret",
            language="Chinese Simplified",
            document_type="print",
        )
    )

    assert output_prefix.endswith("dev/azure_language_chinese_simplified_document_type_print")


def test_selected_dataset_examples_preserves_slice_without_materializing() -> None:
    dataset_stream = (cast("BenchmarkDatasetExample", {"example_id": str(index)}) for index in range(5))

    selected = benchmark._selected_dataset_examples(
        dataset_stream,
        benchmark.BenchmarkOptions(
            backend="azure",
            dataset_split="dev",
            endpoint="https://example.invalid",
            api_key="secret",
            input_size=2,
            offset=1,
        ),
    )

    assert list(selected) == [{"example_id": "1"}, {"example_id": "2"}]


def test_selected_dataset_examples_filters_before_offset_and_input_size() -> None:
    dataset_stream = iter(
        [
            cast(
                "BenchmarkDatasetExample",
                {"example_id": "0", "main_language": "English", "document_type": "print"},
            ),
            cast(
                "BenchmarkDatasetExample",
                {"example_id": "1", "main_language": "Chinese", "document_type": "handwriting"},
            ),
            cast(
                "BenchmarkDatasetExample",
                {"example_id": "2", "main_language": "Chinese", "document_type": "print"},
            ),
            cast(
                "BenchmarkDatasetExample",
                {"example_id": "3", "main_language": "Chinese", "document_type": "print"},
            ),
            cast(
                "BenchmarkDatasetExample",
                {"example_id": "4", "main_language": "French", "document_type": "print"},
            ),
        ]
    )

    selected = benchmark._selected_dataset_examples(
        dataset_stream,
        benchmark.BenchmarkOptions(
            backend="azure",
            dataset_split="dev",
            endpoint="https://example.invalid",
            api_key="secret",
            language=" chinese ",
            document_type="PRINT",
            input_size=1,
            offset=1,
        ),
    )

    assert list(selected) == [{"example_id": "3", "main_language": "Chinese", "document_type": "print"}]


def test_selected_dataset_examples_filters_materialized_dataset() -> None:
    dataset = Dataset.from_list(
        [
            {"example_id": "0", "main_language": "English", "document_type": "print"},
            {"example_id": "1", "main_language": "Chinese", "document_type": "handwriting"},
            {"example_id": "2", "main_language": "Chinese", "document_type": "print"},
            {"example_id": "3", "main_language": "Chinese", "document_type": "print"},
        ]
    )

    selected = benchmark._selected_dataset_examples(
        dataset,
        benchmark.BenchmarkOptions(
            backend="azure",
            dataset_split="dev",
            endpoint="https://example.invalid",
            api_key="secret",
            language=" chinese ",
            document_type="PRINT",
            input_size=1,
            offset=1,
        ),
    )
    selected_dataset = cast("Dataset", selected)

    assert selected_dataset.num_rows == 1
    assert selected_dataset[0] == {
        "example_id": "3",
        "main_language": "Chinese",
        "document_type": "print",
    }


@pytest.mark.asyncio
async def test_predict_texts_updates_progress_and_preserves_order(monkeypatch) -> None:
    dataset: list[BenchmarkDatasetExample] = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example(
            "1",
            size=(1, 1),
            transcription="beta",
            dataset_id="dataset-1",
            document_type="handwriting",
            main_language="Persian",
            main_script="Arabic",
        ),
        _benchmark_example(
            "2",
            size=(2, 2),
            transcription="gamma",
            dataset_id="dataset-2",
            main_language="French",
        ),
    ]

    class FakeProgressBar:
        def __init__(self, *, total: int | None, desc: str, unit: str) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit
            self.updates: list[int] = []
            self.postfixes: list[dict[str, int]] = []
            self.refresh_count = 0

        def __enter__(self) -> FakeProgressBar:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def update(self, amount: int) -> None:
            self.updates.append(amount)

        def set_postfix(self, *, submitted: int, in_flight: int, refresh: bool) -> None:
            self.postfixes.append(
                {
                    "submitted": submitted,
                    "in_flight": in_flight,
                    "refresh": int(refresh),
                }
            )

        def refresh(self) -> None:
            self.refresh_count += 1

    progress_bars: list[FakeProgressBar] = []

    def fake_tqdm(*, total: int | None, desc: str, unit: str) -> FakeProgressBar:
        progress_bar = FakeProgressBar(total=total, desc=desc, unit=unit)
        progress_bars.append(progress_bar)
        return progress_bar

    class FakeOCRResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeOCRBackend:
        async def ocr(self, page):  # noqa: ANN001
            await asyncio.sleep(page.width / 1000)
            return FakeOCRResult(text=f"page-{page.width}")

    monkeypatch.setattr(benchmark, "tqdm", fake_tqdm)
    monkeypatch.setattr(benchmark, "_build_ocr_backend", lambda _: FakeOCRBackend())

    options = benchmark.BenchmarkOptions(
        backend="azure",
        dataset_split="dev",
        endpoint="https://example.invalid",
        api_key="secret",
        max_concurrency=2,
    )

    evaluation_examples, predictions = await benchmark._predict_texts(
        dataset,
        options,
        total_pages=3,
    )

    assert predictions == ["page-3", "page-1", "page-2"]
    assert evaluation_examples == [benchmark._build_evaluation_example(example) for example in dataset]
    assert len(progress_bars) == 1
    assert progress_bars[0].total == 3
    assert progress_bars[0].desc == "OCR"
    assert progress_bars[0].unit == "page"
    assert progress_bars[0].updates == [1, 1, 1]
    assert progress_bars[0].postfixes[-1] == {
        "submitted": 3,
        "in_flight": 0,
        "refresh": 0,
    }
    assert progress_bars[0].refresh_count >= 1


@pytest.mark.asyncio
async def test_predict_texts_uses_batch_backend_with_max_concurrency_as_batch_size(monkeypatch) -> None:
    dataset: list[BenchmarkDatasetExample] = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example(
            "1",
            size=(1, 1),
            transcription="beta",
            dataset_id="dataset-1",
            document_type="handwriting",
            main_language="Persian",
            main_script="Arabic",
        ),
        _benchmark_example(
            "2",
            size=(2, 2),
            transcription="gamma",
            dataset_id="dataset-2",
            main_language="French",
        ),
    ]
    captured_batch_sizes: list[int] = []

    class FakeOCRResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeBatchBackend:
        async def ocr_batch(self, pages):  # noqa: ANN001
            captured_batch_sizes.append(len(pages))
            return [FakeOCRResult(text=f"page-{page.width}") for page in pages]

    monkeypatch.setattr(benchmark, "_build_ocr_backend", lambda _: FakeBatchBackend())

    options = benchmark.BenchmarkOptions(
        backend="hf",
        dataset_split="dev",
        model="kristaller486/dots.ocr-1.5",
        max_concurrency=2,
    )

    evaluation_examples, predictions = await benchmark._predict_texts(
        dataset,
        options,
        total_pages=3,
    )

    assert captured_batch_sizes == [2, 1]
    assert predictions == ["page-3", "page-1", "page-2"]
    assert evaluation_examples == [benchmark._build_evaluation_example(example) for example in dataset]


@pytest.mark.asyncio
async def test_predict_texts_logs_first_batch_output_once(monkeypatch) -> None:
    dataset = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example("1", size=(1, 1), transcription="beta"),
    ]
    logged_messages: list[str] = []

    class FakeLogger:
        def info(self, message: str, *args: object) -> None:
            logged_messages.append(message % args if args else message)

    class FakeOCRResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeBatchBackend:
        async def ocr_batch(self, pages):  # noqa: ANN001
            return [FakeOCRResult(text=f"page-{page.width}") for page in pages]

    monkeypatch.setattr(benchmark, "logger", FakeLogger())
    monkeypatch.setattr(benchmark, "_build_ocr_backend", lambda _: FakeBatchBackend())

    options = benchmark.BenchmarkOptions(
        backend="hf",
        dataset_split="dev",
        model="kristaller486/dots.ocr-1.5",
        max_concurrency=2,
    )

    _evaluation_examples, predictions = await benchmark._predict_texts(
        dataset,
        options,
        total_pages=2,
    )

    assert predictions == ["page-3", "page-1"]
    assert logged_messages == [
        "First benchmark OCR output for backend=hf model=kristaller486/dots.ocr-1.5:\npage-3"
    ]


@pytest.mark.asyncio
async def test_predict_texts_uses_max_concurrency_for_vllm_batch_backend(monkeypatch) -> None:
    dataset = [
        _benchmark_example(str(index), size=(index + 1, index + 1), transcription=f"text-{index}")
        for index in range(10)
    ]
    captured_batch_sizes: list[int] = []

    class FakeOCRResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeBatchBackend:
        async def ocr_batch(self, pages):  # noqa: ANN001
            captured_batch_sizes.append(len(pages))
            return [FakeOCRResult(text=f"page-{page.width}") for page in pages]

    monkeypatch.setattr(benchmark, "_build_ocr_backend", lambda _: FakeBatchBackend())

    options = benchmark.BenchmarkOptions(
        backend="vllm",
        dataset_split="dev",
        model="Qwen/Qwen3.5-0.8B",
        max_concurrency=2,
    )

    evaluation_examples, predictions = await benchmark._predict_texts(
        dataset,
        options,
        total_pages=10,
    )

    assert captured_batch_sizes == [2, 2, 2, 2, 2]
    assert predictions == [f"page-{index + 1}" for index in range(10)]
    assert evaluation_examples == [benchmark._build_evaluation_example(example) for example in dataset]


@pytest.mark.asyncio
async def test_predict_texts_logs_first_submitted_output_once_for_non_batch_backend(monkeypatch) -> None:
    dataset: list[BenchmarkDatasetExample] = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example("1", size=(1, 1), transcription="beta"),
        _benchmark_example("2", size=(2, 2), transcription="gamma"),
    ]
    logged_messages: list[str] = []

    class FakeLogger:
        def info(self, message: str, *args: object) -> None:
            logged_messages.append(message % args if args else message)

    class FakeOCRResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeOCRBackend:
        async def ocr(self, page):  # noqa: ANN001
            await asyncio.sleep(page.width / 1000)
            return FakeOCRResult(text=f"page-{page.width}")

    monkeypatch.setattr(benchmark, "logger", FakeLogger())
    monkeypatch.setattr(benchmark, "_build_ocr_backend", lambda _: FakeOCRBackend())

    options = benchmark.BenchmarkOptions(
        backend="azure",
        dataset_split="dev",
        endpoint="https://example.invalid",
        api_key="secret",
        max_concurrency=2,
    )

    _evaluation_examples, predictions = await benchmark._predict_texts(
        dataset,
        options,
        total_pages=3,
    )

    assert predictions == ["page-3", "page-1", "page-2"]
    assert logged_messages == ["First benchmark OCR output for backend=azure model=<default>:\npage-3"]

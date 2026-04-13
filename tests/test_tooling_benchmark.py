from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest
from datasets import Dataset
from PIL import Image

from churro_ocr.ocr import OCRResult
from churro_ocr.providers.hf import GlmOCROCRBackend, HuggingFaceVisionOCRBackend
from churro_ocr.providers.specs import DEFAULT_OCR_MAX_TOKENS
from churro_ocr.templates import (
    CHURRO_3B_XML_TEMPLATE,
    DEEPSEEK_OCR_2_OCR_TEMPLATE,
    DOTS_MOCR_OCR_TEMPLATE,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    GLM_OCR_OCR_TEMPLATE,
    INFINITY_PARSER_7B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_OCR_TEMPLATE,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
)
from tooling.benchmarking import benchmark

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from churro_ocr.providers.ocr import LiteLLMVisionOCRBackend
    from tooling.evaluation.types import BenchmarkDatasetExample


def _benchmark_runtime_error(message: str) -> RuntimeError:
    return RuntimeError(message)


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


def test_validate_options_requires_pinned_model_for_mistral() -> None:
    missing_model = benchmark.BenchmarkOptions(
        backend="mistral",
        model=None,
        dataset_split="dev",
        api_key="secret",
    )
    alias_model = benchmark.BenchmarkOptions(
        backend="mistral",
        model="mistral-ocr-latest",
        dataset_split="dev",
        api_key="secret",
    )
    pinned_model = benchmark.BenchmarkOptions(
        backend="mistral",
        model="mistral-ocr-2512",
        dataset_split="dev",
        api_key="secret",
    )

    assert benchmark._validate_options(missing_model) == 1
    assert benchmark._validate_options(alias_model) == 1
    assert benchmark._validate_options(pinned_model) == 0


def test_validate_options_allows_openai_compatible_without_api_key() -> None:
    options = benchmark.BenchmarkOptions(
        backend="openai-compatible",
        dataset_split="dev",
        model="local-model",
        base_url="http://127.0.0.1:8000/v1",
    )
    assert benchmark._validate_options(options) == 0


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


def test_parse_args_accepts_reasoning_effort() -> None:
    options = benchmark.parse_args(
        [
            "--backend",
            "litellm",
            "--dataset-split",
            "dev",
            "--model",
            "gpt-5.4",
            "--reasoning-effort",
            "low",
        ]
    )

    assert options.reasoning_effort == "low"


def test_parse_args_rejects_unsupported_backend() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--backend",
                "unsupported",
                "--dataset-split",
                "dev",
                "--model",
                "Qwen/Qwen3.5-0.8B",
            ]
        )


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


def test_validate_options_rejects_reasoning_effort_for_hf() -> None:
    options = benchmark.BenchmarkOptions(
        backend="hf",
        dataset_split="dev",
        model="example/model",
        reasoning_effort="high",
    )

    assert benchmark._validate_options(options) == 1


def test_build_ocr_backend_passes_reasoning_effort_for_litellm(
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
                model="gpt-5.4",
                reasoning_effort="low",
            )
        ),
    )

    assert backend.transport.config.cache_dir == cache_dir
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": DEFAULT_OCR_MAX_TOKENS,
        "reasoning_effort": "low",
    }


def test_build_ocr_backend_allows_openai_compatible_without_api_key() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="local-model",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.transport.config.api_base == "http://127.0.0.1:8000/v1"
    assert backend.transport.config.api_key is None


def test_build_ocr_backend_passes_reasoning_effort_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="gpt-5.4",
                base_url="http://127.0.0.1:8000/v1",
                reasoning_effort="medium",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": DEFAULT_OCR_MAX_TOKENS,
        "reasoning_effort": "medium",
    }


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
    if backend.model_kwargs["dtype"] == "auto" and "device_map" in backend.model_kwargs:
        assert backend.model_kwargs["device_map"] == "auto"
    if "max_memory" in backend.model_kwargs:
        assert backend.model_kwargs["max_memory"]
    assert backend.generation_kwargs == {"max_new_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_uses_dots_preset_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="kristaller486/dots.ocr-1.5",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.model_name == "dots.ocr-1.5"
    assert backend.template == DOTS_OCR_1_5_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 2_048,
        "temperature": 0.0,
    }


def test_build_ocr_backend_uses_dots_mocr_preset_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="rednote-hilab/dots.mocr",
            )
        ),
    )

    assert backend.model_name == "dots.mocr"
    assert backend.processor_kwargs == {}
    assert backend.trust_remote_code is True
    assert backend.model_kwargs["dtype"] in {"auto", "float32"}
    if backend.model_kwargs["dtype"] == "auto" and "device_map" in backend.model_kwargs:
        assert backend.model_kwargs["device_map"] == "auto"
    if "max_memory" in backend.model_kwargs:
        assert backend.model_kwargs["max_memory"]
    assert backend.generation_kwargs == {"max_new_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_uses_dots_mocr_preset_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="rednote-hilab/dots.mocr",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.model_name == "dots.mocr"
    assert backend.template == DOTS_MOCR_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": DEFAULT_OCR_MAX_TOKENS,
        "temperature": 0.0,
    }


def test_build_ocr_backend_uses_deepseek_ocr_2_preset_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="deepseek-ai/DeepSeek-OCR-2",
            )
        ),
    )

    assert backend.model_name == "DeepSeek-OCR-2"
    assert backend.processor_kwargs == {}
    assert backend.trust_remote_code is True
    assert backend.model_kwargs == {
        "device_map": "auto",
        "torch_dtype": "auto",
        "use_safetensors": True,
    }
    assert backend.generation_kwargs == {"max_new_tokens": 8_192}


def test_build_ocr_backend_uses_deepseek_ocr_2_preset_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="deepseek-ai/DeepSeek-OCR-2",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.model_name == "DeepSeek-OCR-2"
    assert backend.template == DEEPSEEK_OCR_2_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 8_192,
        "temperature": 0.0,
    }


def test_build_ocr_backend_uses_glm_ocr_preset_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="zai-org/GLM-OCR",
            )
        ),
    )

    assert isinstance(backend, GlmOCROCRBackend)
    assert backend.model_name == "GLM-OCR"
    assert backend.trust_remote_code is False
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}
    assert backend.generation_kwargs == {
        "max_new_tokens": 8_192,
        "do_sample": False,
    }


def test_build_ocr_backend_uses_glm_ocr_preset_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="zai-org/GLM-OCR",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.model_name == "GLM-OCR"
    assert backend.template == GLM_OCR_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 8_192,
        "temperature": 0.0,
    }
    assert backend.image_preprocessor(Image.new("RGB", (3_508, 2_720), color="white")).size == (
        2_464,
        1_904,
    )


def test_build_ocr_backend_uses_paddleocr_vl_preset_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="PaddlePaddle/PaddleOCR-VL-1.5",
            )
        ),
    )

    assert backend.model_name == "PaddleOCR-VL-1.5"
    assert backend.processor_kwargs == {}
    assert backend.trust_remote_code is False
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}
    assert backend.generation_kwargs == {
        "max_new_tokens": 4_096,
        "do_sample": False,
    }


def test_build_ocr_backend_uses_paddleocr_vl_preset_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="PaddlePaddle/PaddleOCR-VL-1.5",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.model_name == "PaddleOCR-VL-1.5"
    assert backend.template == PADDLEOCR_VL_1_5_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 4_096,
        "temperature": 0.0,
    }


def test_build_ocr_backend_uses_infinity_parser_preset_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="infly/Infinity-Parser-7B",
            )
        ),
    )

    assert type(backend) is HuggingFaceVisionOCRBackend
    assert backend.template == INFINITY_PARSER_7B_OCR_TEMPLATE
    assert backend.model_name == "Infinity-Parser-7B"
    assert backend.processor_kwargs == {
        "min_pixels": 200_704,
        "max_pixels": 1_806_336,
    }
    assert backend.trust_remote_code is False
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}
    assert backend.generation_kwargs == {
        "max_new_tokens": 4_096,
    }


def test_build_ocr_backend_uses_infinity_parser_preset_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="infly/Infinity-Parser-7B",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.model_name == "Infinity-Parser-7B"
    assert backend.template == INFINITY_PARSER_7B_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 8_192,
        "temperature": 0.0,
        "top_p": 0.95,
    }


def test_build_ocr_backend_uses_mineru2_5_preset_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="hf",
                dataset_split="dev",
                model="opendatalab/MinerU2.5-2509-1.2B",
            )
        ),
    )

    assert backend.model_name == "MinerU2.5-2509-1.2B"
    assert backend.processor_kwargs == {"use_fast": True}
    assert backend.trust_remote_code is False
    assert backend.model_kwargs == {
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    assert backend.generation_kwargs == {}


def test_build_ocr_backend_uses_mineru2_5_preset_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="opendatalab/MinerU2.5-2509-1.2B",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.provider_name == "openai-compatible"
    assert backend.model_name == "MinerU2.5-2509-1.2B"
    assert backend.template == MINERU2_5_2509_1_2B_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {}


def test_build_ocr_backend_uses_churro_preset_template_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="stanford-oval/churro-3B",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.template == CHURRO_3B_XML_TEMPLATE
    assert backend.model_name == "churro-3B"


def test_build_ocr_backend_uses_generic_qwen_model_name_for_openai_compatible() -> None:
    backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="Qwen/Qwen3.5-0.8B",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert backend.transport.config.completion_kwargs == {"max_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_aligns_hf_and_openai_compatible_templates_for_generic_models() -> None:
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
    openai_backend = cast(
        "LiteLLMVisionOCRBackend",
        benchmark._build_ocr_backend(
            benchmark.BenchmarkOptions(
                backend="openai-compatible",
                dataset_split="dev",
                model="example/model",
                base_url="http://127.0.0.1:8000/v1",
            )
        ),
    )

    assert hf_backend.template == openai_backend.template


@pytest.mark.asyncio
async def test_run_executes_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    def fake_load_dataset(dataset_id: str, *, split: str) -> list[BenchmarkDatasetExample]:
        assert dataset_id == benchmark.CHURRO_DATASET_ID
        assert split == "dev"
        return dataset

    monkeypatch.setattr(benchmark, "_load_dataset", fake_load_dataset)

    async def fake_predict(
        ds: Iterable[BenchmarkDatasetExample],
        options: benchmark.BenchmarkOptions,
        *,
        total_pages: int | None,
    ) -> tuple[list[object], list[dict[str, object]]]:
        selected = list(ds)
        assert len(selected) == 1
        assert selected[0]["example_id"] == "1"
        assert options.max_concurrency == 2
        assert total_pages is None
        return [benchmark._build_evaluation_example(selected[0])], [
            {"text": "prediction", "metadata": {"raw_html": "<p>prediction</p>"}}
        ]

    monkeypatch.setattr(benchmark, "_predict_texts", fake_predict)
    cleanup_calls: list[str] = []
    call_order: list[str] = []

    async def fake_cleanup() -> None:
        cleanup_calls.append("closed")
        call_order.append("cleanup")

    monkeypatch.setattr(
        benchmark,
        "close_litellm_async_clients",
        fake_cleanup,
    )

    captured: dict[str, object] = {}

    def fake_compute_metrics(
        ds: list[object],
        predictions: list[dict[str, object]],
        output_prefix: str,
        elapsed_time: float,
    ) -> dict[str, str]:
        call_order.append("compute_metrics")
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
    assert captured["predictions"] == [{"text": "prediction", "metadata": {"raw_html": "<p>prediction</p>"}}]
    assert captured["output_prefix"] == str(tmp_path / "outputs")
    assert captured["elapsed_time"] == pytest.approx(3.5)
    assert call_order == ["cleanup", "compute_metrics"]
    assert cleanup_calls == ["closed"]


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
async def test_predict_texts_updates_progress_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        def __init__(
            self,
            *,
            total: int | None,
            desc: str,
            unit: str,
            mininterval: float,
            smoothing: float,
        ) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit
            self.mininterval = mininterval
            self.smoothing = smoothing
            self.updates: list[int] = []
            self.postfixes: list[dict[str, int]] = []
            self.refresh_count = 0

        def __enter__(self) -> FakeProgressBar:
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> None:  # noqa: ANN001
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

    def fake_tqdm(
        *,
        total: int | None,
        desc: str,
        unit: str,
        mininterval: float,
        smoothing: float,
    ) -> FakeProgressBar:
        progress_bar = FakeProgressBar(
            total=total,
            desc=desc,
            unit=unit,
            mininterval=mininterval,
            smoothing=smoothing,
        )
        progress_bars.append(progress_bar)
        return progress_bar

    class FakeOCRBackend:
        async def ocr(self, page: benchmark.DocumentPage) -> OCRResult:
            await asyncio.sleep(page.width / 1000)
            return OCRResult(
                text=f"page-{page.width}",
                provider_name="fake",
                model_name="fake-model",
                metadata={"page_width": page.width},
            )

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

    assert predictions == [
        {"text": "page-3", "metadata": {"page_width": 3}},
        {"text": "page-1", "metadata": {"page_width": 1}},
        {"text": "page-2", "metadata": {"page_width": 2}},
    ]
    assert evaluation_examples == [benchmark._build_evaluation_example(example) for example in dataset]
    assert len(progress_bars) == 1
    assert progress_bars[0].total == 3
    assert progress_bars[0].desc == "OCR"
    assert progress_bars[0].unit == "page"
    assert progress_bars[0].mininterval == benchmark.PROGRESS_BAR_MININTERVAL_SECONDS
    assert progress_bars[0].smoothing == benchmark.PROGRESS_BAR_SMOOTHING
    assert progress_bars[0].updates == [1, 1, 1]
    assert progress_bars[0].postfixes[-1] == {
        "submitted": 3,
        "in_flight": 0,
        "refresh": 0,
    }
    assert progress_bars[0].refresh_count >= 1


@pytest.mark.asyncio
async def test_predict_texts_uses_batch_backend_with_max_concurrency_as_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    class FakeBatchBackend:
        async def ocr_batch(self, pages: list[benchmark.DocumentPage]) -> list[OCRResult]:
            captured_batch_sizes.append(len(pages))
            return [
                OCRResult(
                    text=f"page-{page.width}",
                    provider_name="fake",
                    model_name="fake-model",
                    metadata={"page_width": page.width},
                )
                for page in pages
            ]

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
    assert predictions == [
        {"text": "page-3", "metadata": {"page_width": 3}},
        {"text": "page-1", "metadata": {"page_width": 1}},
        {"text": "page-2", "metadata": {"page_width": 2}},
    ]
    assert evaluation_examples == [benchmark._build_evaluation_example(example) for example in dataset]


@pytest.mark.asyncio
async def test_predict_texts_logs_first_batch_output_once(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example("1", size=(1, 1), transcription="beta"),
    ]
    logged_messages: list[str] = []

    class FakeLogger:
        def info(self, message: str, *args: object) -> None:
            logged_messages.append(message % args if args else message)

    class FakeBatchBackend:
        async def ocr_batch(self, pages: list[benchmark.DocumentPage]) -> list[OCRResult]:
            return [
                OCRResult(
                    text=f"page-{page.width}",
                    provider_name="fake",
                    model_name="fake-model",
                    metadata={"page_width": page.width},
                )
                for page in pages
            ]

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

    assert predictions == [
        {"text": "page-3", "metadata": {"page_width": 3}},
        {"text": "page-1", "metadata": {"page_width": 1}},
    ]
    assert logged_messages == [
        "First benchmark OCR output for backend=hf model=kristaller486/dots.ocr-1.5:\npage-3"
    ]


@pytest.mark.asyncio
async def test_predict_texts_logs_first_submitted_output_once_for_non_batch_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset: list[BenchmarkDatasetExample] = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example("1", size=(1, 1), transcription="beta"),
        _benchmark_example("2", size=(2, 2), transcription="gamma"),
    ]
    logged_messages: list[str] = []

    class FakeLogger:
        def info(self, message: str, *args: object) -> None:
            logged_messages.append(message % args if args else message)

    class FakeOCRBackend:
        async def ocr(self, page: benchmark.DocumentPage) -> OCRResult:
            await asyncio.sleep(page.width / 1000)
            return OCRResult(
                text=f"page-{page.width}",
                provider_name="fake",
                model_name="fake-model",
                metadata={"page_width": page.width},
            )

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

    assert predictions == [
        {"text": "page-3", "metadata": {"page_width": 3}},
        {"text": "page-1", "metadata": {"page_width": 1}},
        {"text": "page-2", "metadata": {"page_width": 2}},
    ]
    assert logged_messages == ["First benchmark OCR output for backend=azure model=<default>:\npage-3"]


@pytest.mark.asyncio
async def test_predict_texts_continues_after_non_batch_page_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset: list[BenchmarkDatasetExample] = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example("1", size=(1, 1), transcription="beta"),
        _benchmark_example("2", size=(2, 2), transcription="gamma"),
    ]
    logged_messages: list[str] = []

    class FakeLogger:
        def info(self, _message: str, *_args: object) -> None:
            return None

        def exception(self, message: str, *args: object) -> None:
            logged_messages.append(message % args if args else message)

    class FakeOCRBackend:
        async def ocr(self, page: benchmark.DocumentPage) -> OCRResult:
            if page.width == 1:
                message = "timed out"
                raise _benchmark_runtime_error(message)
            return OCRResult(
                text=f"page-{page.width}",
                provider_name="fake",
                model_name="fake-model",
                metadata={"page_width": page.width},
            )

    monkeypatch.setattr(benchmark, "logger", FakeLogger())
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

    assert evaluation_examples == [benchmark._build_evaluation_example(example) for example in dataset]
    assert predictions == [
        {"text": "page-3", "metadata": {"page_width": 3}},
        {
            "text": "",
            "metadata": {"benchmark_error": {"type": "RuntimeError", "message": "timed out"}},
        },
        {"text": "page-2", "metadata": {"page_width": 2}},
    ]
    assert logged_messages == [
        "Benchmark OCR failed for example_id=1 dataset_id=dataset-1 backend=azure model=<default>; "
        "treating prediction as empty."
    ]


@pytest.mark.asyncio
async def test_predict_texts_continues_after_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset: list[BenchmarkDatasetExample] = [
        _benchmark_example("0", size=(3, 3), transcription="alpha"),
        _benchmark_example("1", size=(1, 1), transcription="beta"),
        _benchmark_example("2", size=(2, 2), transcription="gamma"),
    ]
    logged_messages: list[str] = []
    call_count = {"ocr_batch": 0}

    class FakeLogger:
        def info(self, _message: str, *_args: object) -> None:
            return None

        def exception(self, message: str, *args: object) -> None:
            logged_messages.append(message % args if args else message)

    class FakeBatchBackend:
        async def ocr_batch(self, pages: list[benchmark.DocumentPage]) -> list[OCRResult]:
            call_count["ocr_batch"] += 1
            if call_count["ocr_batch"] == 1:
                message = "batch timed out"
                raise _benchmark_runtime_error(message)
            return [
                OCRResult(
                    text=f"page-{page.width}",
                    provider_name="fake",
                    model_name="fake-model",
                    metadata={"page_width": page.width},
                )
                for page in pages
            ]

    monkeypatch.setattr(benchmark, "logger", FakeLogger())
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

    assert evaluation_examples == [benchmark._build_evaluation_example(example) for example in dataset]
    assert predictions == [
        {
            "text": "",
            "metadata": {"benchmark_error": {"type": "RuntimeError", "message": "batch timed out"}},
        },
        {
            "text": "",
            "metadata": {"benchmark_error": {"type": "RuntimeError", "message": "batch timed out"}},
        },
        {"text": "page-2", "metadata": {"page_width": 2}},
    ]
    assert logged_messages == [
        "Benchmark OCR failed for example_id=0 dataset_id=dataset-0 backend=hf "
        "model=kristaller486/dots.ocr-1.5; treating prediction as empty.",
        "Benchmark OCR failed for example_id=1 dataset_id=dataset-1 backend=hf "
        "model=kristaller486/dots.ocr-1.5; treating prediction as empty.",
    ]

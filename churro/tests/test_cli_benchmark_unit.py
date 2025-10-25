from __future__ import annotations

from collections.abc import Iterator
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from churro.cli import benchmark


@pytest.fixture(autouse=True)
def stub_model_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "MODEL_MAP", {"valid-engine": object()})


def test_validate_options_requires_engine() -> None:
    options = benchmark.BenchmarkOptions(
        system="llm",
        engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        resize=None,
        max_concurrency=1,
        input_size=1,
        dataset_split="dev",
        offset=0,
    )
    assert benchmark._validate_options(options) == 1


def test_validate_options_rejects_invalid_split() -> None:
    options = benchmark.BenchmarkOptions(
        system="azure",
        engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        resize=None,
        max_concurrency=1,
        input_size=1,
        dataset_split="train",
        offset=0,
    )
    assert benchmark._validate_options(options) == 1


@pytest.mark.asyncio
async def test_run_executes_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = [
        {"image": "img0", "file_name": "file0"},
        {"image": "img1", "file_name": "file1"},
        {"image": "img2", "file_name": "file2"},
    ]

    def fake_load_dataset(dataset_id: str, split: str, streaming: bool) -> list[dict[str, str]]:
        assert dataset_id == benchmark.CHURRO_DATASET_ID
        assert split == "dev"
        assert streaming is True
        return dataset

    monkeypatch.setattr(benchmark, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(benchmark, "create_output_prefix", lambda _: str(tmp_path))

    time_values = iter([10.0, 13.5])
    monkeypatch.setattr(benchmark, "time", lambda: next(time_values))

    class FakeOCR:
        async def process_images(self, images: list[str], max_concurrency: int) -> list[str]:
            assert images == ["img1"]
            assert max_concurrency == 2
            return ["prediction"]

    monkeypatch.setattr(benchmark.OCRFactory, "create_ocr_system", lambda _: FakeOCR())

    container_calls: list[dict[str, object]] = []

    @contextlib.contextmanager
    def fake_managed_container(**kwargs: object) -> Iterator[SimpleNamespace]:
        container_calls.append(kwargs)
        yield SimpleNamespace()

    monkeypatch.setattr(benchmark, "managed_vllm_container", fake_managed_container)

    captured = {}

    def fake_compute_metrics(
        ds: list[dict[str, str]],
        predictions: list[str],
        output_prefix: str,
        elapsed_time: float,
    ) -> dict[str, str]:
        captured["dataset"] = ds
        captured["predictions"] = predictions
        captured["output_prefix"] = output_prefix
        captured["elapsed_time"] = elapsed_time
        return {"status": "ok"}

    monkeypatch.setattr(benchmark, "compute_metrics", fake_compute_metrics)

    options = benchmark.BenchmarkOptions(
        system="azure",
        engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        resize=None,
        max_concurrency=2,
        input_size=1,
        dataset_split="dev",
        offset=1,
    )

    result = await benchmark.run(options)

    assert result == 0
    assert container_calls == [
        {
            "engine": None,
            "backup_engine": None,
            "system": "azure",
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
        }
    ]
    assert captured["dataset"] == [{"image": "img1", "file_name": "file1"}]
    assert captured["predictions"] == ["prediction"]
    assert captured["output_prefix"] == str(tmp_path)
    assert captured["elapsed_time"] == pytest.approx(3.5)

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from churro.cli.docs_to_images import DocsToImagesOptions
from churro.cli.main import app


def test_docs_to_images_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, DocsToImagesOptions] = {}

    async def fake_run(options: DocsToImagesOptions) -> int:
        captured["options"] = options
        return 0

    monkeypatch.setattr("churro.cli.docs_to_images.run", fake_run)

    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "sample.pdf").write_bytes(b"%PDF-1.4")

    exit_code = app(
        [
            "docs-to-images",
            "--input-dir",
            str(tmp_path / "inputs"),
            "--output-dir",
            str(tmp_path / "out"),
            "--suffix",
            "pdf",
            "--suffix",
            "PNG",
            "--dry-run",
        ],
        standalone_mode=False,
    )

    assert exit_code == 0
    options = cast(DocsToImagesOptions, captured["options"])
    assert options.dry_run is True
    assert options.pattern == "*"
    assert options.extensions == [".pdf", ".png"]
    assert options.dpi is None


def test_infer_invokes_ocr_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    from PIL import Image

    Image.new("RGB", (10, 10), color="white").save(image_path)

    class DummyOCR:
        async def process_images_from_files(
            self, paths: Sequence[str], max_concurrency: int
        ) -> list[str]:
            assert paths == [str(image_path)]
            return ["dummy"]

    captured: dict[str, argparse.Namespace] = {}

    def capture_and_build(args: argparse.Namespace) -> DummyOCR:
        captured["options"] = args
        return DummyOCR()

    monkeypatch.setattr(
        "churro.systems.ocr_factory.OCRFactory.create_ocr_system",
        capture_and_build,
    )

    exit_code = app(
        [
            "infer",
            "--system",
            "azure",
            "--image",
            str(image_path),
        ],
        standalone_mode=False,
    )

    assert exit_code == 0
    options = captured["options"]
    assert options.suffixes == [".png"]


def test_benchmark_invokes_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color="white").save(image_path)
    sample: list[dict[str, Image.Image]] = [{"image": Image.open(image_path)}]

    def load_dataset_stub(
        dataset_id: str, split: str, streaming: bool
    ) -> list[dict[str, Image.Image]]:
        return sample

    monkeypatch.setattr("churro.cli.benchmark.load_dataset", load_dataset_stub)

    class DummyOCR:
        async def process_images(
            self, images: Sequence[Image.Image], max_concurrency: int
        ) -> list[str]:
            return ["text"] * len(images)

    def create_ocr_system_stub(args: argparse.Namespace) -> DummyOCR:
        return DummyOCR()

    def compute_metrics_stub(
        dataset: Sequence[dict[str, bytes]],
        texts: Sequence[str],
        prefix: str,
        elapsed_time: float,
    ) -> None:
        return None

    monkeypatch.setattr(
        "churro.systems.ocr_factory.OCRFactory.create_ocr_system",
        create_ocr_system_stub,
    )
    monkeypatch.setattr("churro.cli.benchmark.compute_metrics", compute_metrics_stub)

    exit_code = app(
        [
            "benchmark",
            "--system",
            "azure",
            "--dataset-split",
            "dev",
            "--input-size",
            "1",
            "--engine",
            "azure",
        ],
        standalone_mode=False,
    )

    assert exit_code == 0

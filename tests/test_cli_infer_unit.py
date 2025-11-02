from __future__ import annotations

from collections.abc import Iterator
import contextlib
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from churro.cli import infer


@pytest.fixture(autouse=True)
def stub_model_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MODEL_MAP has predictable contents for validation tests."""
    monkeypatch.setattr(infer, "MODEL_MAP", {"valid-engine": object()})


def test_validate_options_requires_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    options = infer.InferOptions(
        system="llm",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=None,
        pattern="*.png",
        suffixes=[".png"],
        recursive=False,
        output_dir=None,
        skip_existing=False,
        max_concurrency=1,
    )
    assert infer._validate_options(options) == 1


def test_validate_options_rejects_invalid_backup_engine() -> None:
    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine="not-real",
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=None,
        pattern="*.png",
        suffixes=[".png"],
        recursive=False,
        output_dir=None,
        skip_existing=False,
        max_concurrency=1,
    )
    assert infer._validate_options(options) == 1


def test_validate_options_filters_invalid_suffixes() -> None:
    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=None,
        pattern="*.png",
        suffixes=[".png", ".unsupported"],
        recursive=False,
        output_dir=None,
        skip_existing=False,
        max_concurrency=1,
    )
    result = infer._validate_options(options)
    assert result == 0
    assert options.suffixes == [".png"]


def test_validate_options_errors_when_suffixes_removed() -> None:
    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=None,
        pattern="*.png",
        suffixes=[".unsupported"],
        recursive=False,
        output_dir=None,
        skip_existing=False,
        max_concurrency=1,
    )
    assert infer._validate_options(options) == 1


def test_collect_images_deduplicates_and_filters(tmp_path: Path) -> None:
    (tmp_path / "keep.png").write_text("one")
    (tmp_path / "skip.txt").write_text("two")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "keep2.png").write_text("three")
    duplicate = nested / "keep.png"
    duplicate.write_text("four")

    images = infer._collect_images(
        image=None,
        image_dir=tmp_path,
        suffixes=[".png"],
        recursive=True,
    )
    assert len(images) == 3
    assert {path.resolve() for path in images} == {
        (tmp_path / "keep.png").resolve(),
        (nested / "keep2.png").resolve(),
        duplicate.resolve(),
    }


def test_write_output_skips_existing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "image.txt"
    target.write_text("original")
    image_path = tmp_path / "image.png"
    image_path.write_text("img")

    infer._write_or_print_output(
        img_path=image_path,
        text="new",
        output_dir=out_dir,
        skip_existing=True,
        multi_mode=False,
    )
    assert target.read_text() == "original"


@pytest.mark.asyncio
async def test_run_requires_image_selection(tmp_path: Path) -> None:
    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=None,
        pattern="*.png",
        suffixes=[".png"],
        recursive=False,
        output_dir=None,
        skip_existing=False,
        max_concurrency=1,
    )
    result = await infer.run(options)
    assert result == 1


@pytest.mark.asyncio
async def test_run_rejects_image_and_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_text("img")
    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=image_path,
        image_dir=tmp_path,
        pattern="*.png",
        suffixes=[".png"],
        recursive=False,
        output_dir=None,
        skip_existing=False,
        max_concurrency=1,
    )
    result = await infer.run(options)
    assert result == 1


@pytest.mark.asyncio
async def test_run_returns_error_when_no_images_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=tmp_path,
        pattern="*.png",
        suffixes=[".png"],
        recursive=False,
        output_dir=None,
        skip_existing=False,
        max_concurrency=1,
    )

    monkeypatch.setattr(infer, "_collect_images", lambda *_, **__: [])

    result = await infer.run(options)
    assert result == 1


@pytest.mark.asyncio
async def test_run_processes_images_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first = image_dir / "img1.png"
    second = image_dir / "img2.png"
    first.write_text("a")
    second.write_text("b")

    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=image_dir,
        pattern="*.png",
        suffixes=[".png"],
        recursive=False,
        output_dir=tmp_path / "outputs",
        skip_existing=False,
        max_concurrency=0,
    )

    class FakeOCR:
        async def process_images_from_files(
            self, image_paths: list[str], max_concurrency: int
        ) -> list[str]:
            assert image_paths == [str(first), str(second)]
            assert max_concurrency == 1  # coerced from 0
            return ["first text", "second text"]

    container_calls: list[dict[str, object]] = []

    @contextlib.contextmanager
    def fake_managed_container(**kwargs: object) -> Iterator[SimpleNamespace]:
        container_calls.append(kwargs)
        yield SimpleNamespace()

    monkeypatch.setattr(infer, "managed_vllm_container", fake_managed_container)
    monkeypatch.setattr(infer.OCRFactory, "create_ocr_system", lambda _: FakeOCR())

    result = await infer.run(options)

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

    out_dir = options.output_dir
    assert out_dir is not None
    assert (out_dir / "img1.txt").read_text() == "first text"
    assert (out_dir / "img2.txt").read_text() == "second text"


@pytest.mark.asyncio
async def test_run_binarizes_inputs_before_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first = image_dir / "img1.png"
    second = image_dir / "img2.png"
    Image.new("RGB", (10, 10), color="white").save(first)
    Image.new("RGB", (10, 10), color="white").save(second)

    options = infer.InferOptions(
        system="azure",
        engine=None,
        backup_engine=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
        image=None,
        image_dir=image_dir,
        pattern="*.png",
        suffixes=[".png"],
        recursive=False,
        output_dir=tmp_path / "outputs",
        skip_existing=False,
        max_concurrency=2,
        binarize=True,
    )

    class FakeBinarizer:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def binarize_pil_batch(
            self, images: list[Image.Image], scale: float = 1.0, n_batch_inference: int = 16
        ) -> list[Image.Image]:
            del scale, n_batch_inference  # unused in fake implementation
            self.calls.extend(image.size for image in images)
            return [Image.new("L", image.size, color=0) for image in images]

    fake_binarizer = FakeBinarizer()

    class FakeOCR:
        async def process_images(
            self, images: list[Image.Image], *, max_concurrency: int
        ) -> list[str]:
            assert len(images) == 2
            assert all(isinstance(image, Image.Image) for image in images)
            assert all(image.mode == "L" for image in images)
            assert max_concurrency == 2
            return ["binarized 1", "binarized 2"]

        async def process_images_from_files(
            self, image_paths: list[str], max_concurrency: int
        ) -> list[str]:  # pragma: no cover - defensive guard
            raise AssertionError("process_images_from_files should not be called when binarizing")

    container_calls: list[dict[str, object]] = []

    @contextlib.contextmanager
    def fake_managed_container(**kwargs: object) -> Iterator[SimpleNamespace]:
        container_calls.append(kwargs)
        yield SimpleNamespace()

    monkeypatch.setattr(infer, "ImageBinarizer", lambda: fake_binarizer)
    monkeypatch.setattr(infer, "managed_vllm_container", fake_managed_container)
    monkeypatch.setattr(infer.OCRFactory, "create_ocr_system", lambda _: FakeOCR())

    result = await infer.run(options)

    assert result == 0
    assert fake_binarizer.calls == [(10, 10), (10, 10)]

    out_dir = options.output_dir
    assert out_dir is not None
    assert (out_dir / "img1.txt").read_text() == "binarized 1"
    assert (out_dir / "img2.txt").read_text() == "binarized 2"

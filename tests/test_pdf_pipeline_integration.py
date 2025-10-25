from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor, Future
from pathlib import Path
from types import TracebackType

from PIL import Image
import pytest


ASSETS_DIR = Path(__file__).resolve().parent.parent


class _InlineExecutor(Executor):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

    def submit(
        self, fn: Callable[..., object], /, *args: object, **kwargs: object
    ) -> Future[object]:
        future: Future[object] = Future()
        try:
            result: object = fn(*args, **kwargs)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        return None

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False


@pytest.fixture(autouse=True)
def inline_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "churro.utils.pdf.runner.ProcessPoolExecutor",
        _InlineExecutor,
    )


class PassthroughSplitter:
    async def split(self, image: Image.Image) -> list[Image.Image]:
        return [image]


class DuplicateFirstSplitter:
    def __init__(self) -> None:
        self._invocations = 0

    async def split(self, image: Image.Image) -> list[Image.Image]:
        self._invocations += 1
        if self._invocations == 1:
            # Return two copies to emulate a two-page spread.
            return [image.copy(), image.copy()]
        return [image]


class NoOpTrimmer:
    async def trim(self, image: Image.Image) -> Image.Image:
        return image


def _asset(name: str) -> Path:
    return ASSETS_DIR / name


@pytest.mark.asyncio
async def test_pdf_pipeline_produces_png_from_minimal_pdf(tmp_path: Path) -> None:
    from churro.utils.pdf.runner import run_pdf_pipeline

    output_dir = tmp_path / "out"
    pdf_path = _asset("minimal-document.pdf")

    await run_pdf_pipeline(
        pdf_paths=[str(pdf_path)],
        output_dir=str(output_dir),
        engine="gpt-5-low",
        raster_workers=1,
        page_workers=1,
        llm_concurrency_limit=1,
        splitter_factory=lambda _engine: PassthroughSplitter(),
        trimmer_factory=lambda _trim: NoOpTrimmer(),
    )

    pngs = sorted(output_dir.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].name == "minimal-document_page_0000.png"


@pytest.mark.asyncio
async def test_pdf_pipeline_handles_mixed_inputs(tmp_path: Path) -> None:
    from churro.utils.pdf.runner import run_pdf_pipeline

    output_dir = tmp_path / "out"
    pdf_path = _asset("minimal-document.pdf")
    image_paths = [
        _asset("churro_dataset_sample_1.jpeg"),
        _asset("churro_dataset_sample_2.jpeg"),
    ]

    splitter = DuplicateFirstSplitter()

    await run_pdf_pipeline(
        pdf_paths=[str(pdf_path)],
        output_dir=str(output_dir),
        engine="gpt-5-low",
        raster_workers=1,
        page_workers=1,
        llm_concurrency_limit=1,
        image_paths=[str(path) for path in image_paths],
        splitter_factory=lambda _engine: splitter,
        trimmer_factory=lambda _trim: NoOpTrimmer(),
    )

    pngs = sorted(output_dir.glob("*.png"))
    assert pngs, "Expected PNG outputs to be written"
    names = [png.name for png in pngs]
    counts = Counter(names)
    assert counts["minimal-document_page_0000.png"] == 1
    assert counts["churro_dataset_sample_1_page_0000.png"] == 1
    assert counts["churro_dataset_sample_2_page_0000.png"] == 1
    assert sum(name.endswith("_page_0001.png") for name in names) == 1
    assert len(names) == 4


def test_docs_to_images_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from churro.cli.main import app

    output_dir = tmp_path / "out"
    pdf_path = _asset("minimal-document.pdf")

    monkeypatch.setattr(
        "churro.cli.docs_to_images.default_splitter_factory",
        lambda _engine: PassthroughSplitter(),
    )
    monkeypatch.setattr(
        "churro.cli.docs_to_images.default_trimmer_factory",
        lambda _trim: NoOpTrimmer(),
    )

    exit_code = app(
        [
            "docs-to-images",
            "--input-file",
            str(pdf_path),
            "--output-dir",
            str(output_dir),
            "--engine",
            "gpt-5-low",
            "--raster-workers",
            "1",
            "--page-workers",
            "1",
            "--llm-concurrency-limit",
            "1",
            "--no-trim",
        ],
        standalone_mode=False,
    )

    assert exit_code == 0

    pngs = sorted(output_dir.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].name == "minimal-document_page_0000.png"

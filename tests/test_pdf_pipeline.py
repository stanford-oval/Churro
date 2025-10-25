from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from churro.utils.llm import ImageDetail
from churro.utils.pdf.runner import (
    IdentityTrimmer,
    LLMPageSplitter,
    PageProcessingStage,
    PageSplitGroup,
    PageSplitter,
    PageTrimmer,
    RasterTask,
    run_pdf_pipeline,
)


class _StubSplitter(PageSplitter):
    async def split(self, image: Image.Image) -> list[Image.Image]:
        return [image, image.copy()]


class _RecordingTrimmer(PageTrimmer):
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def trim(self, image: Image.Image) -> Image.Image:
        self.calls.append(id(image))
        return image


@pytest.mark.asyncio
async def test_llm_page_splitter_skips_narrow_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail_run_llm_async(
        model: str,
        system_prompt_text: str | None,
        user_message_text: str | None,
        user_message_image: Image.Image | list[Image.Image] | None = None,
        image_detail: ImageDetail | None = None,
        output_json: bool = False,
        pydantic_class: type | None = None,
        timeout: int = 0,
    ) -> str:
        raise AssertionError("LLM should not be invoked for narrow pages.")

    monkeypatch.setattr("churro.utils.pdf.runner.run_llm_async", _fail_run_llm_async)

    splitter = LLMPageSplitter(engine="dummy")
    image = Image.new("RGB", (600, 1200), color="white")

    result = await splitter.split(image)

    assert len(result) == 1


@pytest.mark.asyncio
async def test_llm_page_splitter_invokes_llm_for_wide_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, bool] = {"value": False}

    async def _stub_run_llm_async(
        model: str,
        system_prompt_text: str | None,
        user_message_text: str | None,
        user_message_image: Image.Image | list[Image.Image] | None = None,
        image_detail: ImageDetail | None = None,
        output_json: bool = False,
        pydantic_class: type | None = None,
        timeout: int = 0,
    ) -> str:
        called["value"] = True
        return "<number_of_pages>\n1\n</number_of_pages>"

    monkeypatch.setattr("churro.utils.pdf.runner.run_llm_async", _stub_run_llm_async)

    splitter = LLMPageSplitter(engine="dummy")
    image = Image.new("RGB", (1400, 900), color="white")

    result = await splitter.split(image)

    assert called["value"] is True
    assert len(result) == 1


@pytest.mark.asyncio
async def test_page_processing_stage_splits_and_trims() -> None:
    trimmer = _RecordingTrimmer()
    stage = PageProcessingStage(
        splitter=_StubSplitter(),
        trimmer=trimmer,
        concurrency_limit=1,
    )

    raster_queue: asyncio.Queue[RasterTask | None] = asyncio.Queue()
    processed_queue: asyncio.Queue[PageSplitGroup | None] = asyncio.Queue()

    workers = stage.spawn_workers(
        count=1, raster_queue=raster_queue, processed_queue=processed_queue
    )

    image = Image.new("RGB", (10, 10), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    await raster_queue.put(
        RasterTask(pdf_id=0, pdf_path="synthetic.pdf", page_index=0, png_bytes=buffer.getvalue())
    )
    await raster_queue.put(None)

    await raster_queue.join()
    await asyncio.gather(*workers)

    group = processed_queue.get_nowait()
    processed_queue.task_done()
    await processed_queue.join()

    assert group is not None
    assert len(group.images) == 2
    assert len(trimmer.calls) == 2


class _PipelineStubSplitter:
    def __init__(self, *, engine: str) -> None:
        self.engine = engine

    async def split(self, image: Image.Image) -> list[Image.Image]:
        return [image]


class _PipelineStubTrimmer(IdentityTrimmer):
    async def trim(self, image: Image.Image) -> Image.Image:
        return image


@pytest.mark.asyncio
async def test_run_pdf_pipeline_with_images(tmp_path: Path) -> None:
    def splitter_factory(engine: str) -> PageSplitter:
        return _PipelineStubSplitter(engine=engine)

    def trimmer_factory(enable_trim: bool) -> PageTrimmer:
        return _PipelineStubTrimmer() if enable_trim else IdentityTrimmer()

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (12, 12), color="white").save(image_path)

    output_dir = tmp_path / "out"
    await run_pdf_pipeline(
        pdf_paths=[],
        output_dir=str(output_dir),
        engine="dummy",
        image_paths=[str(image_path)],
        splitter_factory=splitter_factory,
        trimmer_factory=trimmer_factory,
    )

    expected = output_dir / "sample_page_0000.png"
    assert expected.exists()

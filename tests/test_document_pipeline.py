from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from churro_ocr.document import DocumentOCRPipeline
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRBackend, OCRResult
from churro_ocr.page_detection import DocumentPage, PageCandidate, PageDetectionRequest

if TYPE_CHECKING:
    from pathlib import Path


class _EchoOCRBackend(OCRBackend):
    async def ocr(self, page: DocumentPage) -> OCRResult:
        image = page.image
        return OCRResult(
            text=f"{page.metadata['page_index']}:{image.width}x{image.height}",
            provider_name="echo",
            model_name="echo-model",
            metadata=dict(page.metadata),
        )


async def _tight_boundary(_: Image.Image) -> list[PageCandidate]:
    return [PageCandidate(bbox=(5.0, 5.0, 25.0, 25.0), metadata={"kind": "tight"})]


def test_document_ocr_pipeline_process_image_sync() -> None:
    pipeline = DocumentOCRPipeline(
        _EchoOCRBackend(),
        detection_backend=_tight_boundary,
    )

    result = pipeline.process_image_sync(
        PageDetectionRequest(image=Image.new("RGB", (40, 30), color="white"), trim_margin=0),
        ocr_metadata={"source": "test"},
    )

    assert result.source_type == "image"
    assert result.texts() == ["0:20x20"]
    assert result.pages[0].ocr_metadata["source"] == "test"
    assert result.pages[0].metadata["kind"] == "tight"


def test_document_ocr_pipeline_process_pdf_sync(minimal_pdf_path: Path) -> None:
    result = DocumentOCRPipeline(_EchoOCRBackend()).process_pdf_sync(
        minimal_pdf_path,
        dpi=150,
        trim_margin=0,
    )

    assert result.source_type == "pdf"
    assert len(result.pages) >= 1
    assert (result.pages[0].text or "").startswith("0:")
    assert result.metadata["path"].endswith("minimal-document.pdf")


def test_document_ocr_pipeline_rejects_non_positive_max_concurrency() -> None:
    with pytest.raises(ConfigurationError, match="max_concurrency"):
        DocumentOCRPipeline(_EchoOCRBackend(), max_concurrency=0)


@pytest.mark.asyncio
async def test_document_ocr_pipeline_respects_max_concurrency() -> None:
    in_flight = 0
    max_seen = 0

    class _TrackingOCRBackend(OCRBackend):
        async def ocr(self, page: DocumentPage) -> OCRResult:
            nonlocal in_flight, max_seen
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return OCRResult(
                text=f"page-{page.page_index}",
                provider_name="tracking",
                model_name="tracking-model",
            )

    async def _four_pages(image: Image.Image) -> list[PageCandidate]:
        return [PageCandidate(image=image.copy(), metadata={"slot": index}) for index in range(4)]

    pipeline = DocumentOCRPipeline(
        _TrackingOCRBackend(),
        detection_backend=_four_pages,
        max_concurrency=2,
    )

    result = await pipeline.process_image(
        PageDetectionRequest(image=Image.new("RGB", (40, 30), color="white"))
    )

    assert len(result.pages) == 4
    assert max_seen == 2

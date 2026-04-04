from __future__ import annotations

import pytest
from PIL import Image

from churro_ocr.errors import ConfigurationError
from churro_ocr.page_detection import (
    DocumentPageDetector,
    PageCandidate,
    PageDetectionRequest,
    PageDetector,
)


async def _two_pages(_: Image.Image) -> list[PageCandidate]:
    return [
        PageCandidate(bbox=(5.0, 5.0, 25.0, 25.0), metadata={"kind": "left"}),
        PageCandidate(bbox=(30.0, 5.0, 55.0, 25.0), metadata={"kind": "right"}),
    ]


@pytest.mark.asyncio
async def test_document_page_detector_detects_multiple_pages() -> None:
    page_detector = DocumentPageDetector(backend=_two_pages)

    result = await page_detector.detect_image(
        PageDetectionRequest(image=Image.new("RGB", (60, 30), color="white"), trim_margin=0)
    )

    assert len(result.pages) == 2
    assert result.pages[0].image.size == (20, 20)
    assert result.pages[1].metadata["kind"] == "right"


def test_page_detector_returns_page_list() -> None:
    pages = PageDetector(_two_pages).detect(
        PageDetectionRequest(image=Image.new("RGB", (60, 30), color="white"), trim_margin=0)
    )

    assert len(pages) == 2
    assert pages[0].page_index == 0
    assert pages[1].page_index == 1


def test_document_page_detector_detect_pdf_sync_uses_real_pdf(minimal_pdf_path) -> None:
    result = DocumentPageDetector().detect_pdf_sync(minimal_pdf_path, dpi=150, trim_margin=0)

    assert result.source_type == "pdf"
    assert len(result.pages) >= 1
    assert result.pages[0].image.width > 0


def test_page_detection_request_requires_exactly_one_image_input(write_image_file) -> None:
    image_path = write_image_file(size=(12, 12))
    with pytest.raises(ConfigurationError, match="exactly one"):
        PageDetectionRequest().require_image()

    with pytest.raises(ConfigurationError, match="exactly one"):
        PageDetectionRequest(
            image=Image.new("RGB", (12, 12), color="white"),
            image_path=image_path,
        ).require_image()

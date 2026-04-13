from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

from churro_ocr.document import DocumentOCRResult
from churro_ocr.ocr import OCRBackend, OCRClient, OCRResult
from churro_ocr.page_detection import (
    DocumentPage,
    DocumentPageDetector,
    PageCandidate,
    PageDetectionRequest,
    PageDetector,
)

if TYPE_CHECKING:
    from tests._types import WriteImageFile


class _MetadataEchoOCRBackend(OCRBackend):
    async def ocr(self, page: DocumentPage) -> OCRResult:
        return OCRResult(
            text=f"{page.page_index}:{page.source_index}",
            provider_name="echo",
            model_name="echo-model",
            metadata=dict(page.metadata),
        )


def _fake_rasterize_pdf(_path: str, *, dpi: int) -> list[Image.Image]:
    assert dpi == 144
    return [
        Image.new("RGB", (10, 10), color="white"),
        Image.new("RGB", (10, 10), color="white"),
    ]


def test_document_page_properties_and_with_ocr() -> None:
    page = DocumentPage.from_image(
        Image.new("RGB", (14, 9), color="white"),
        page_index=3,
        source_index=2,
        metadata={"kind": "original"},
    )

    result = page.with_ocr(
        text="transcribed",
        provider_name="fake",
        model_name="fake-model",
        ocr_metadata={"score": 0.9},
    )

    assert page.width == 14
    assert page.height == 9
    assert page.text is None
    assert result.text == "transcribed"
    assert result.provider_name == "fake"
    assert result.model_name == "fake-model"
    assert result.ocr_metadata == {"score": 0.9}
    assert result.metadata == {"kind": "original"}


def test_page_detector_defaults_to_full_image_when_no_backend() -> None:
    image = Image.new("RGB", (20, 12), color="white")

    pages = PageDetector().detect(PageDetectionRequest(image=image, trim_margin=0))

    assert len(pages) == 1
    assert pages[0].image.size == (20, 12)
    assert pages[0].bbox == (0.0, 0.0, 20.0, 12.0)


def test_page_detector_falls_back_to_full_image_when_backend_returns_no_candidates() -> None:
    async def _empty_backend(_: Image.Image) -> list[PageCandidate]:
        return []

    pages = PageDetector(_empty_backend).detect(
        PageDetectionRequest(image=Image.new("RGB", (18, 11), color="white"), trim_margin=0)
    )

    assert len(pages) == 1
    assert pages[0].image.size == (18, 11)
    assert pages[0].bbox == (0.0, 0.0, 18.0, 11.0)


def test_page_detector_uses_candidate_image_directly() -> None:
    async def _candidate_image(_: Image.Image) -> list[PageCandidate]:
        return [PageCandidate(image=Image.new("RGB", (7, 6), color="black"))]

    pages = PageDetector(_candidate_image).detect(
        PageDetectionRequest(image=Image.new("RGB", (20, 20), color="white"))
    )

    assert len(pages) == 1
    assert pages[0].image.size == (7, 6)


def test_page_detector_uses_polygon_crop_and_masks_background() -> None:
    async def _triangle(_: Image.Image) -> list[PageCandidate]:
        return [PageCandidate(polygon=((5.0, 5.0), (15.0, 5.0), (10.0, 15.0)))]

    pages = PageDetector(_triangle).detect(
        PageDetectionRequest(image=Image.new("RGB", (20, 20), color="black"), trim_margin=0)
    )

    assert len(pages) == 1
    assert pages[0].image.size == (10, 10)
    assert pages[0].image.getpixel((5, 2)) == (0, 0, 0)
    assert pages[0].image.getpixel((0, 9)) == (255, 255, 255)


def test_page_detector_uses_source_copy_when_candidate_has_no_bbox_or_polygon() -> None:
    source = Image.new("RGB", (17, 13), color="white")

    async def _candidate_without_bounds(_: Image.Image) -> list[PageCandidate]:
        return [PageCandidate(metadata={"kind": "full-copy"})]

    pages = PageDetector(_candidate_without_bounds).detect(PageDetectionRequest(image=source, trim_margin=0))

    assert len(pages) == 1
    assert pages[0].image.size == source.size
    assert pages[0].metadata == {"kind": "full-copy"}
    assert pages[0].image is not source


def test_document_page_detector_detect_image_sync_returns_page_detection_result() -> None:
    image = Image.new("RGB", (21, 10), color="white")

    result = DocumentPageDetector().detect_image_sync(PageDetectionRequest(image=image, trim_margin=0))

    assert result.source_type == "image"
    assert len(result.pages) == 1
    assert result.pages[0].image.size == (21, 10)


@pytest.mark.asyncio
async def test_document_page_detector_detect_pdf_async_preserves_source_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "churro_ocr.page_detection.rasterize_pdf",
        _fake_rasterize_pdf,
    )

    result = await DocumentPageDetector().detect_pdf("sample.pdf", dpi=144, trim_margin=0)

    assert result.source_type == "pdf"
    assert result.metadata == {"dpi": 144, "path": "sample.pdf"}
    assert [page.source_index for page in result.pages] == [0, 1]
    assert [page.page_index for page in result.pages] == [0, 1]


def test_ocr_client_ocr_image_propagates_metadata_and_indexes() -> None:
    client = OCRClient(_MetadataEchoOCRBackend())

    page = client.ocr_image(
        image=Image.new("RGB", (9, 7), color="white"),
        page_index=4,
        source_index=3,
        metadata={"source": "direct"},
    )

    assert page.page_index == 4
    assert page.source_index == 3
    assert page.text == "4:3"
    assert page.provider_name == "echo"
    assert page.model_name == "echo-model"
    assert page.ocr_metadata == {"source": "direct"}


@pytest.mark.asyncio
async def test_ocr_client_aocr_image_from_path_propagates_metadata_and_indexes(
    write_image_file: WriteImageFile,
) -> None:
    image_path = write_image_file(size=(9, 7))
    page = await OCRClient(_MetadataEchoOCRBackend()).aocr_image(
        image_path=image_path,
        page_index=6,
        source_index=5,
        metadata={"source": "path"},
    )

    assert page.page_index == 6
    assert page.source_index == 5
    assert page.text == "6:5"
    assert page.ocr_metadata == {"source": "path"}


def test_document_ocr_result_as_ocr_results_preserves_metadata_and_order() -> None:
    first = DocumentPage.from_image(Image.new("RGB", (4, 4), color="white"), metadata={"slot": 1}).with_ocr(
        text="one",
        provider_name="provider-a",
        model_name="model-a",
        ocr_metadata={"cost": 0.1},
    )
    second = DocumentPage.from_image(Image.new("RGB", (4, 4), color="white"), metadata={"slot": 2}).with_ocr(
        text="two",
        provider_name="provider-b",
        model_name="model-b",
        ocr_metadata={"cost": 0.2},
    )
    result = DocumentOCRResult(pages=[first, second], source_type="image", metadata={"path": "scan.png"})

    ocr_results = result.as_ocr_results()

    assert result.texts() == ["one", "two"]
    assert [ocr_result.text for ocr_result in ocr_results] == ["one", "two"]
    assert [ocr_result.provider_name for ocr_result in ocr_results] == ["provider-a", "provider-b"]
    assert [ocr_result.metadata for ocr_result in ocr_results] == [{"cost": 0.1}, {"cost": 0.2}]

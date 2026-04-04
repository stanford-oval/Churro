from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRBackend, OCRClient, OCRResult, prepare_ocr_page
from churro_ocr.page_detection import DocumentPage
from churro_ocr.prompts import strip_ocr_output_tag


class _FakeOCRBackend(OCRBackend):
    async def ocr(self, page: DocumentPage) -> OCRResult:
        image = page.image
        return OCRResult(
            text=f"{image.width}x{image.height}",
            provider_name="fake",
            model_name="fake-model",
        )


def test_document_page_loads_image_from_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (12, 34), color="white").save(image_path)

    page = DocumentPage.from_image_path(image_path)
    image = page.image

    assert image.size == (12, 34)


def test_ocr_client_sync(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), color="white").save(image_path)

    result = OCRClient(_FakeOCRBackend()).ocr(DocumentPage.from_image_path(image_path))

    assert result.text == "20x10"
    assert result.provider_name == "fake"


@pytest.mark.asyncio
async def test_ocr_client_async_with_callable() -> None:
    async def _callable_backend(page: DocumentPage) -> OCRResult:
        image = page.image
        return OCRResult(
            text=str(image.size),
            provider_name="callable",
            model_name="callable-model",
        )

    result = await OCRClient(_callable_backend).aocr(
        DocumentPage.from_image(Image.new("RGB", (9, 7), color="white"))
    )

    assert result.text == "(9, 7)"
    assert result.provider_name == "callable"


def test_prepare_ocr_page_resizes_and_normalizes_image() -> None:
    page = DocumentPage.from_image(Image.new("RGBA", (5_000, 3_000), color=(255, 255, 255, 255)))

    prepared_page = prepare_ocr_page(page)

    assert page.image.size == (5_000, 3_000)
    assert page.image.mode == "RGBA"
    assert prepared_page.image.size == (2_500, 1_500)
    assert prepared_page.image.mode == "RGB"


def test_ocr_client_image_helpers_require_exactly_one_input(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), color="white").save(image_path)
    client = OCRClient(_FakeOCRBackend())

    with pytest.raises(ConfigurationError, match="exactly one"):
        client.ocr_image()

    with pytest.raises(ConfigurationError, match="exactly one"):
        client.ocr_image(
            image=Image.new("RGB", (20, 10), color="white"),
            image_path=image_path,
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<output>\nhello\n</output>", "hello"),
        ("<output> hello", "hello"),
        ("hello </output>", "hello"),
        ("plain text", "plain text"),
    ],
)
def test_strip_ocr_output_tag_removes_outer_and_stray_tags(text: str, expected: str) -> None:
    assert strip_ocr_output_tag(text) == expected

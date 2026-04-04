"""Public OCR interfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from churro_ocr._internal.image import prepare_ocr_image
from churro_ocr._internal.runtime import run_sync
from churro_ocr.errors import ConfigurationError
from churro_ocr.page_detection import DocumentPage


@dataclass(slots=True)
class OCRResult:
    """Provider-agnostic OCR result."""

    text: str
    provider_name: str
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OCRBackend(Protocol):
    """Async OCR backend interface."""

    async def ocr(self, page: DocumentPage) -> OCRResult: ...


@runtime_checkable
class BatchOCRBackend(Protocol):
    """Async batch OCR backend interface."""

    async def ocr_batch(self, pages: list[DocumentPage]) -> list[OCRResult]: ...


OCRCallable = Callable[[DocumentPage], Awaitable[OCRResult]]
OCRBackendLike = OCRBackend | OCRCallable


def prepare_ocr_page(page: DocumentPage) -> DocumentPage:
    """Return a page copy with the shared OCR image preprocessing applied."""
    return replace(page, image=prepare_ocr_image(page.image))


class OCRClient:
    """User-facing OCR client with page-first sync and async entrypoints."""

    def __init__(self, backend: OCRBackendLike) -> None:
        self._backend = backend

    async def aocr(self, page: DocumentPage) -> DocumentPage:
        """Run OCR asynchronously for one page."""
        if callable(self._backend) and not isinstance(self._backend, OCRBackend):
            result = await self._backend(page)
        else:
            assert isinstance(self._backend, OCRBackend)
            result = await self._backend.ocr(page)
        return page.with_ocr(
            text=result.text,
            provider_name=result.provider_name,
            model_name=result.model_name,
            ocr_metadata=result.metadata,
        )

    def ocr(self, page: DocumentPage) -> DocumentPage:
        """Run OCR synchronously for one page."""
        return run_sync(self.aocr(page))

    async def aocr_image(
        self,
        *,
        image: Image.Image | None = None,
        image_path: str | Path | None = None,
        page_index: int = 0,
        source_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentPage:
        """Create a single page from an image input and OCR it."""
        page = _page_from_image_input(
            image=image,
            image_path=image_path,
            page_index=page_index,
            source_index=source_index,
            metadata=metadata,
        )
        return await self.aocr(page)

    def ocr_image(
        self,
        *,
        image: Image.Image | None = None,
        image_path: str | Path | None = None,
        page_index: int = 0,
        source_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentPage:
        """Create a single page from an image input and OCR it synchronously."""
        return run_sync(
            self.aocr_image(
                image=image,
                image_path=image_path,
                page_index=page_index,
                source_index=source_index,
                metadata=metadata,
            )
        )


def _page_from_image_input(
    *,
    image: Image.Image | None,
    image_path: str | Path | None,
    page_index: int,
    source_index: int,
    metadata: dict[str, Any] | None,
) -> DocumentPage:
    if (image is None) == (image_path is None):
        raise ConfigurationError("OCR image helpers require exactly one of `image` or `image_path`.")
    if image is not None:
        return DocumentPage.from_image(
            image,
            page_index=page_index,
            source_index=source_index,
            metadata=metadata,
        )
    if image_path is not None:
        return DocumentPage.from_image_path(
            image_path,
            page_index=page_index,
            source_index=source_index,
            metadata=metadata,
        )
    raise AssertionError("Unreachable exact-one image input guard.")

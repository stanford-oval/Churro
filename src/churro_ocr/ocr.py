"""Public OCR interfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from churro_ocr._internal.image import prepare_ocr_image
from churro_ocr._internal.runtime import run_sync
from churro_ocr.errors import ConfigurationError
from churro_ocr.page_detection import DocumentPage

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from churro_ocr.types import MetadataDict


def _assertion_error(message: str) -> AssertionError:
    return AssertionError(message)


def _configuration_error(message: str) -> ConfigurationError:
    return ConfigurationError(message)


@dataclass(slots=True)
class OCRResult:
    """Provider-agnostic OCR result.

    :param text: OCR text after any backend-specific postprocessing.
    :param provider_name: Stable provider identifier attached to the result.
    :param model_name: Human-readable model name attached to the result.
    :param metadata: Provider-returned metadata for this OCR call.
    """

    text: str
    provider_name: str
    model_name: str
    metadata: MetadataDict = field(default_factory=dict)


@runtime_checkable
class OCRBackend(Protocol):
    """Async OCR backend interface."""

    async def ocr(self, page: DocumentPage) -> OCRResult:
        """Run OCR for one page.

        :param page: Page image and page metadata to transcribe.
        :returns: Provider-agnostic OCR result for the page.
        """
        ...


@runtime_checkable
class BatchOCRBackend(Protocol):
    """Async batch OCR backend interface."""

    async def ocr_batch(self, pages: list[DocumentPage]) -> list[OCRResult]:
        """Run OCR for multiple pages in one batch.

        :param pages: Pages to transcribe in batch order.
        :returns: OCR results in the same order as ``pages``.
        """
        ...


OCRCallable = Callable[[DocumentPage], Awaitable[OCRResult]]
OCRBackendLike = OCRBackend | OCRCallable


def prepare_ocr_page(page: DocumentPage) -> DocumentPage:
    """Return a page copy with the shared OCR image preprocessing applied.

    :param page: Page to preprocess for OCR.
    :returns: Copy of ``page`` with its image replaced by the preprocessed image.
    """
    return replace(page, image=prepare_ocr_image(page.image))


class OCRClient:
    """User-facing OCR client with page-first sync and async entrypoints."""

    def __init__(self, backend: OCRBackendLike) -> None:
        """Create an OCR client.

        :param backend: OCR backend or async callable used for page OCR.
        """
        self._backend = backend

    async def aocr(self, page: DocumentPage) -> DocumentPage:
        """Run OCR asynchronously for one page.

        :param page: Page to transcribe.
        :returns: Copy of ``page`` with OCR output attached.
        """
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
        """Run OCR synchronously for one page.

        :param page: Page to transcribe.
        :returns: Copy of ``page`` with OCR output attached.
        """
        return run_sync(self.aocr(page))

    async def aocr_image(
        self,
        *,
        image: Image.Image | None = None,
        image_path: str | Path | None = None,
        page_index: int = 0,
        source_index: int = 0,
        metadata: MetadataDict | None = None,
    ) -> DocumentPage:
        """Create a single page from an image input and OCR it.

        :param image: In-memory page image. Mutually exclusive with ``image_path``.
        :param image_path: Path to a page image on disk. Mutually exclusive with ``image``.
        :param page_index: Page position to attach to the generated page.
        :param source_index: Original source index to attach to the generated page.
        :param metadata: Optional caller-side metadata attached before OCR runs.
        :returns: OCR-enriched page object.
        :raises ConfigurationError: If both or neither of ``image`` and
            ``image_path`` are provided.
        """
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
        metadata: MetadataDict | None = None,
    ) -> DocumentPage:
        """Create a single page from an image input and OCR it synchronously.

        :param image: In-memory page image. Mutually exclusive with ``image_path``.
        :param image_path: Path to a page image on disk. Mutually exclusive with ``image``.
        :param page_index: Page position to attach to the generated page.
        :param source_index: Original source index to attach to the generated page.
        :param metadata: Optional caller-side metadata attached before OCR runs.
        :returns: OCR-enriched page object.
        :raises ConfigurationError: If both or neither of ``image`` and
            ``image_path`` are provided.
        """
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
    metadata: MetadataDict | None,
) -> DocumentPage:
    if (image is None) == (image_path is None):
        message = "OCR image helpers require exactly one of `image` or `image_path`."
        raise _configuration_error(message)
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
    message = "Unreachable exact-one image input guard."
    raise _assertion_error(message)

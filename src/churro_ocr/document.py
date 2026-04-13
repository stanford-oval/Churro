"""Document-level OCR pipeline built on the page detection and OCR APIs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from churro_ocr._internal.runtime import run_sync
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRBackendLike, OCRClient, OCRResult
from churro_ocr.page_detection import (
    DocumentPage,
    DocumentPageDetector,
    PageDetectionBackendLike,
    PageDetectionRequest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from churro_ocr.types import MetadataDict


@dataclass(slots=True)
class DocumentOCRResult:
    """Document OCR output across all detected pages.

    :param pages: OCR-enriched pages in output order.
    :param source_type: Input source type, typically ``"image"`` or ``"pdf"``.
    :param metadata: Document-level metadata carried forward from page detection.
    """

    pages: list[DocumentPage]
    source_type: str
    metadata: MetadataDict = field(default_factory=dict)

    def texts(self) -> list[str]:
        """Return OCR text for each page in order.

        :returns: Plain OCR text for each page. Missing page text is normalized to ``""``.
        """
        return [page.text or "" for page in self.pages]

    def as_ocr_results(self) -> list[OCRResult]:
        """Return plain OCR results in page order.

        :returns: ``OCRResult`` objects derived from the current pages.
        """
        return [
            OCRResult(
                text=page.text or "",
                provider_name=page.provider_name or "",
                model_name=page.model_name or "",
                metadata=dict(page.ocr_metadata),
            )
            for page in self.pages
        ]


class DocumentOCRPipeline:
    """Run page detection and OCR as one document-level pipeline.

    The pipeline is the highest-level API in the package. It detects pages from
    an image or PDF, runs OCR on each detected page, and preserves the page
    objects in the final result.
    """

    def __init__(
        self,
        ocr_backend: OCRBackendLike,
        *,
        page_detector: DocumentPageDetector | None = None,
        detection_backend: PageDetectionBackendLike | None = None,
        max_concurrency: int = 8,
    ) -> None:
        """Create a document OCR pipeline.

        :param ocr_backend: OCR backend or async OCR callable used for each page.
        :param page_detector: Optional fully constructed page detector to reuse.
        :param detection_backend: Optional low-level detection backend used when
            ``page_detector`` is not provided.
        :param max_concurrency: Maximum number of page OCR jobs run at once.
        :raises ConfigurationError: If ``max_concurrency`` is less than 1.
        """
        if max_concurrency < 1:
            raise ConfigurationError("DocumentOCRPipeline max_concurrency must be at least 1.")
        self._ocr_client = OCRClient(ocr_backend)
        self._page_detector = page_detector or DocumentPageDetector(backend=detection_backend)
        self.max_concurrency = max_concurrency

    async def process_image(
        self,
        request: PageDetectionRequest,
        *,
        ocr_metadata: MetadataDict | None = None,
    ) -> DocumentOCRResult:
        """Detect pages and OCR a single input image.

        :param request: Image detection request describing the source image.
        :param ocr_metadata: Optional caller-side metadata merged into each page
            before OCR runs.
        :returns: Document OCR result preserving page order and page images.
        """
        detection_result = await self._page_detector.detect_image(request)
        return await self._ocr_detection_result(
            detection_result.pages,
            detection_result.source_type,
            detection_result.metadata,
            ocr_metadata,
        )

    def process_image_sync(
        self,
        request: PageDetectionRequest,
        *,
        ocr_metadata: MetadataDict | None = None,
    ) -> DocumentOCRResult:
        """Synchronously detect pages and OCR a single input image.

        :param request: Image detection request describing the source image.
        :param ocr_metadata: Optional caller-side metadata merged into each page
            before OCR runs.
        :returns: Document OCR result preserving page order and page images.
        """
        return run_sync(self.process_image(request, ocr_metadata=ocr_metadata))

    async def process_pdf(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        trim_margin: int = 30,
        ocr_metadata: MetadataDict | None = None,
    ) -> DocumentOCRResult:
        """Rasterize, detect pages, and OCR a PDF.

        :param path: PDF path to rasterize and process.
        :param dpi: Rasterization DPI used before page detection.
        :param trim_margin: Pixel margin added around detected crops.
        :param ocr_metadata: Optional caller-side metadata merged into each page
            before OCR runs.
        :returns: Document OCR result across the rasterized PDF pages.
        """
        detection_result = await self._page_detector.detect_pdf(
            path,
            dpi=dpi,
            trim_margin=trim_margin,
        )
        return await self._ocr_detection_result(
            detection_result.pages,
            detection_result.source_type,
            detection_result.metadata,
            ocr_metadata,
        )

    def process_pdf_sync(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        trim_margin: int = 30,
        ocr_metadata: MetadataDict | None = None,
    ) -> DocumentOCRResult:
        """Synchronously rasterize, detect pages, and OCR a PDF.

        :param path: PDF path to rasterize and process.
        :param dpi: Rasterization DPI used before page detection.
        :param trim_margin: Pixel margin added around detected crops.
        :param ocr_metadata: Optional caller-side metadata merged into each page
            before OCR runs.
        :returns: Document OCR result across the rasterized PDF pages.
        """
        return run_sync(
            self.process_pdf(
                path,
                dpi=dpi,
                trim_margin=trim_margin,
                ocr_metadata=ocr_metadata,
            )
        )

    async def _ocr_detection_result(
        self,
        detected_pages: list[DocumentPage],
        source_type: str,
        metadata: MetadataDict,
        ocr_metadata: MetadataDict | None,
    ) -> DocumentOCRResult:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _ocr_page_with_limit(page: DocumentPage) -> DocumentPage:
            async with semaphore:
                return await self._ocr_page(page, ocr_metadata=ocr_metadata)

        results = await asyncio.gather(*(_ocr_page_with_limit(page) for page in detected_pages))
        return DocumentOCRResult(
            pages=results,
            source_type=source_type,
            metadata=dict(metadata),
        )

    async def _ocr_page(
        self,
        page: DocumentPage,
        *,
        ocr_metadata: MetadataDict | None,
    ) -> DocumentPage:
        page_metadata = dict(page.metadata)
        page_metadata.update(ocr_metadata or {})
        page_metadata.setdefault("page_index", page.page_index)
        page_metadata.setdefault("source_index", page.source_index)
        ocr_page = page.__class__(
            page_index=page.page_index,
            source_index=page.source_index,
            image=page.image,
            bbox=page.bbox,
            polygon=page.polygon,
            metadata=page_metadata,
            text=page.text,
            provider_name=page.provider_name,
            model_name=page.model_name,
            ocr_metadata=dict(page.ocr_metadata),
        )
        return await self._ocr_client.aocr(ocr_page)

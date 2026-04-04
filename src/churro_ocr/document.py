"""Document-level OCR pipeline built on the page detection and OCR APIs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from churro_ocr._internal.runtime import run_sync
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRBackendLike, OCRClient, OCRResult
from churro_ocr.page_detection import (
    DocumentPage,
    DocumentPageDetector,
    PageDetectionBackendLike,
    PageDetectionRequest,
)


@dataclass(slots=True)
class DocumentOCRResult:
    """Document OCR output across all detected pages."""

    pages: list[DocumentPage]
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def texts(self) -> list[str]:
        """Return OCR text for each page in order."""
        return [page.text or "" for page in self.pages]

    def as_ocr_results(self) -> list[OCRResult]:
        """Return plain OCR results in page order."""
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
    """Run page detection and OCR as one document-level pipeline."""

    def __init__(
        self,
        ocr_backend: OCRBackendLike,
        *,
        page_detector: DocumentPageDetector | None = None,
        detection_backend: PageDetectionBackendLike | None = None,
        max_concurrency: int = 8,
    ) -> None:
        if max_concurrency < 1:
            raise ConfigurationError("DocumentOCRPipeline max_concurrency must be at least 1.")
        self._ocr_client = OCRClient(ocr_backend)
        self._page_detector = page_detector or DocumentPageDetector(backend=detection_backend)
        self.max_concurrency = max_concurrency

    async def process_image(
        self,
        request: PageDetectionRequest,
        *,
        ocr_metadata: dict[str, Any] | None = None,
    ) -> DocumentOCRResult:
        """Detect pages and OCR a single input image."""
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
        ocr_metadata: dict[str, Any] | None = None,
    ) -> DocumentOCRResult:
        """Synchronous wrapper for image OCR."""
        return run_sync(self.process_image(request, ocr_metadata=ocr_metadata))

    async def process_pdf(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        trim_margin: int = 30,
        ocr_metadata: dict[str, Any] | None = None,
    ) -> DocumentOCRResult:
        """Rasterize, detect pages, and OCR a PDF."""
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
        ocr_metadata: dict[str, Any] | None = None,
    ) -> DocumentOCRResult:
        """Synchronous wrapper for PDF OCR."""
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
        metadata: dict[str, Any],
        ocr_metadata: dict[str, Any] | None,
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
        ocr_metadata: dict[str, Any] | None,
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

"""Public page detection interfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image, ImageDraw

from churro_ocr._internal.image import load_image
from churro_ocr._internal.pdf import rasterize_pdf
from churro_ocr._internal.runtime import run_sync
from churro_ocr.errors import ConfigurationError


@dataclass(slots=True)
class PageCandidate:
    """Intermediate page candidate returned by a page detector."""

    bbox: tuple[float, float, float, float] | None = None
    image: Image.Image | None = None
    polygon: tuple[tuple[float, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentPage:
    """A document page image with optional OCR output attached."""

    page_index: int
    image: Image.Image
    source_index: int
    bbox: tuple[float, float, float, float] | None = None
    polygon: tuple[tuple[float, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None
    provider_name: str | None = None
    model_name: str | None = None
    ocr_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @classmethod
    def from_image(
        cls,
        image: Image.Image,
        *,
        page_index: int = 0,
        source_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentPage:
        """Create a document page from an in-memory image."""
        return cls(
            page_index=page_index,
            source_index=source_index,
            image=image.copy(),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_image_path(
        cls,
        path: str | Path,
        *,
        page_index: int = 0,
        source_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentPage:
        """Create a document page from an image path."""
        return cls.from_image(
            load_image(path),
            page_index=page_index,
            source_index=source_index,
            metadata=metadata,
        )

    def with_ocr(
        self,
        *,
        text: str,
        provider_name: str,
        model_name: str,
        ocr_metadata: dict[str, Any] | None = None,
    ) -> DocumentPage:
        """Return a copy of the page with OCR output attached."""
        return replace(
            self,
            text=text,
            provider_name=provider_name,
            model_name=model_name,
            ocr_metadata=dict(ocr_metadata or {}),
        )


@dataclass(slots=True)
class PageDetectionRequest:
    """Request payload for image page detection."""

    image: Image.Image | None = None
    image_path: str | Path | None = None
    trim_margin: int = 30

    def require_image(self) -> Image.Image:
        """Return the input image, loading it from disk when needed."""
        if (self.image is None) == (self.image_path is None):
            raise ConfigurationError("PageDetectionRequest requires exactly one of `image` or `image_path`.")
        if self.image is not None:
            return self.image.copy()
        if self.image_path is not None:
            return load_image(self.image_path)
        raise AssertionError("Unreachable exact-one image input guard.")


@dataclass(slots=True)
class PageDetectionResult:
    """Page detection output for an image or PDF."""

    pages: list[DocumentPage]
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PageDetectionBackend(Protocol):
    """Async interface for page detection."""

    async def detect(self, image: Image.Image) -> list[PageCandidate]: ...


PageDetectionCallable = Callable[[Image.Image], Awaitable[list[PageCandidate]]]
PageDetectionBackendLike = PageDetectionBackend | PageDetectionCallable


class PageDetector:
    """Detect one or more page crops from an input image."""

    def __init__(self, backend: PageDetectionBackendLike | None = None) -> None:
        self._backend = backend

    async def adetect(self, request: PageDetectionRequest) -> list[DocumentPage]:
        """Asynchronously detect pages for a single image."""
        image = request.require_image()
        candidates = await self._detect_candidates(image)
        detected_pages: list[DocumentPage] = []
        for candidate in candidates:
            detected_pages.append(
                DocumentPage(
                    page_index=len(detected_pages),
                    image=self._materialize_candidate(
                        source_image=image,
                        candidate=candidate,
                        trim_margin=request.trim_margin,
                    ),
                    source_index=0,
                    bbox=candidate.bbox,
                    polygon=candidate.polygon,
                    metadata=dict(candidate.metadata),
                )
            )
        return detected_pages

    def detect(self, request: PageDetectionRequest) -> list[DocumentPage]:
        """Synchronous wrapper for page detection."""
        return run_sync(self.adetect(request))

    async def _detect_candidates(self, image: Image.Image) -> list[PageCandidate]:
        if self._backend is None:
            return [PageCandidate(bbox=(0.0, 0.0, float(image.width), float(image.height)))]
        if callable(self._backend) and not isinstance(self._backend, PageDetectionBackend):
            candidates = await self._backend(image)
        else:
            assert isinstance(self._backend, PageDetectionBackend)
            candidates = await self._backend.detect(image)
        return candidates or [PageCandidate(bbox=(0.0, 0.0, float(image.width), float(image.height)))]

    def _materialize_candidate(
        self,
        *,
        source_image: Image.Image,
        candidate: PageCandidate,
        trim_margin: int,
    ) -> Image.Image:
        if candidate.image is not None:
            return candidate.image.copy()
        if candidate.polygon:
            return _crop_polygon(source_image, candidate.polygon, trim_margin=trim_margin)
        if candidate.bbox is None:
            return source_image.copy()
        return _crop_bbox(source_image, candidate.bbox, trim_margin=trim_margin)


class DocumentPageDetector:
    """Detect pages from raw images or PDFs."""

    def __init__(
        self,
        *,
        backend: PageDetectionBackendLike | None = None,
    ) -> None:
        self._page_detector = PageDetector(backend)

    async def detect_image(self, request: PageDetectionRequest) -> PageDetectionResult:
        """Detect pages in a single image."""
        pages = await self._page_detector.adetect(request)
        return PageDetectionResult(pages=pages, source_type="image")

    def detect_image_sync(self, request: PageDetectionRequest) -> PageDetectionResult:
        """Synchronous wrapper for image page detection."""
        return run_sync(self.detect_image(request))

    async def detect_pdf(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        trim_margin: int = 30,
    ) -> PageDetectionResult:
        """Rasterize a PDF and detect pages on each image."""
        images = rasterize_pdf(path, dpi=dpi)
        pages: list[DocumentPage] = []
        for pdf_index, image in enumerate(images):
            detected_pages = await self._page_detector.adetect(
                PageDetectionRequest(image=image, trim_margin=trim_margin)
            )
            for page in detected_pages:
                pages.append(
                    DocumentPage(
                        page_index=len(pages),
                        image=page.image,
                        source_index=pdf_index,
                        bbox=page.bbox,
                        polygon=page.polygon,
                        metadata=dict(page.metadata),
                    )
                )
        return PageDetectionResult(
            pages=pages,
            source_type="pdf",
            metadata={"dpi": dpi, "path": str(path)},
        )

    def detect_pdf_sync(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        trim_margin: int = 30,
    ) -> PageDetectionResult:
        """Synchronous wrapper for PDF page detection."""
        return run_sync(self.detect_pdf(path, dpi=dpi, trim_margin=trim_margin))


def _crop_bbox(
    source_image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    trim_margin: int,
) -> Image.Image:
    left, top, right, bottom = bbox
    expanded_left = max(int(left - trim_margin), 0)
    expanded_top = max(int(top - trim_margin), 0)
    expanded_right = min(int(right + trim_margin), source_image.width)
    expanded_bottom = min(int(bottom + trim_margin), source_image.height)
    return source_image.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))


def _crop_polygon(
    source_image: Image.Image,
    polygon: tuple[tuple[float, float], ...],
    *,
    trim_margin: int,
) -> Image.Image:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    cropped = _crop_bbox(source_image, bbox, trim_margin=trim_margin)
    left = max(int(bbox[0] - trim_margin), 0)
    top = max(int(bbox[1] - trim_margin), 0)

    mask = Image.new("L", cropped.size, 0)
    relative_points = [(x - left, y - top) for x, y in polygon]
    ImageDraw.Draw(mask).polygon(relative_points, fill=255)

    background = Image.new(cropped.mode, cropped.size, color="white")
    background.paste(cropped, mask=mask)
    return background

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
    """Intermediate page candidate returned by a page detector.

    :param bbox: Bounding box in source-image coordinates.
    :param image: Optional already-cropped page image. When provided, detection
        callers use this image directly instead of cropping from ``bbox`` or ``polygon``.
    :param polygon: Optional polygon in source-image coordinates.
    :param metadata: Detector-side metadata attached to the candidate.
    """

    bbox: tuple[float, float, float, float] | None = None
    image: Image.Image | None = None
    polygon: tuple[tuple[float, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentPage:
    """A document page image with optional OCR output attached.

    :param page_index: Page position in the current output sequence.
    :param image: Page image.
    :param source_index: Index of the original source item that produced the page.
    :param bbox: Bounding box in source-image coordinates when available.
    :param polygon: Polygon in source-image coordinates when available.
    :param metadata: Caller-side or detector-side metadata for the page.
    :param text: OCR text attached to the page when OCR has been run.
    :param provider_name: Provider identifier attached by OCR.
    :param model_name: Model name attached by OCR.
    :param ocr_metadata: Provider-returned OCR metadata for this page.
    """

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
        """Return the current page image width in pixels."""
        return self.image.width

    @property
    def height(self) -> int:
        """Return the current page image height in pixels."""
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
        """Create a document page from an in-memory image.

        :param image: Source page image.
        :param page_index: Page position to attach to the page.
        :param source_index: Source index to attach to the page.
        :param metadata: Optional caller-side metadata for the page.
        :returns: New page object with a copied image.
        """
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
        """Create a document page from an image path.

        :param path: Path to the page image on disk.
        :param page_index: Page position to attach to the page.
        :param source_index: Source index to attach to the page.
        :param metadata: Optional caller-side metadata for the page.
        :returns: New page object loaded from ``path``.
        """
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
        """Return a copy of the page with OCR output attached.

        :param text: OCR text for the page.
        :param provider_name: Provider identifier to attach.
        :param model_name: Model name to attach.
        :param ocr_metadata: Provider-returned OCR metadata.
        :returns: Copy of the current page with OCR fields filled in.
        """
        return replace(
            self,
            text=text,
            provider_name=provider_name,
            model_name=model_name,
            ocr_metadata=dict(ocr_metadata or {}),
        )


@dataclass(slots=True)
class PageDetectionRequest:
    """Request payload for image page detection.

    :param image: In-memory image to detect pages from. Mutually exclusive with
        ``image_path``.
    :param image_path: Path to an image on disk. Mutually exclusive with ``image``.
    :param trim_margin: Margin in pixels to add around detected crops.
    """

    image: Image.Image | None = None
    image_path: str | Path | None = None
    trim_margin: int = 30

    def require_image(self) -> Image.Image:
        """Return the input image, loading it from disk when needed.

        :returns: Copy of the requested image.
        :raises ConfigurationError: If both or neither of ``image`` and
            ``image_path`` are provided.
        """
        if (self.image is None) == (self.image_path is None):
            raise ConfigurationError("PageDetectionRequest requires exactly one of `image` or `image_path`.")
        if self.image is not None:
            return self.image.copy()
        if self.image_path is not None:
            return load_image(self.image_path)
        raise AssertionError("Unreachable exact-one image input guard.")


@dataclass(slots=True)
class PageDetectionResult:
    """Page detection output for an image or PDF.

    :param pages: Detected pages in output order.
    :param source_type: Input source type, typically ``"image"`` or ``"pdf"``.
    :param metadata: Detection-level metadata, such as PDF rasterization settings.
    """

    pages: list[DocumentPage]
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PageDetectionBackend(Protocol):
    """Async interface for page detection."""

    async def detect(self, image: Image.Image) -> list[PageCandidate]:
        """Detect page candidates from one image.

        :param image: Source image to analyze.
        :returns: Page candidates in reading order.
        """
        ...


PageDetectionCallable = Callable[[Image.Image], Awaitable[list[PageCandidate]]]
PageDetectionBackendLike = PageDetectionBackend | PageDetectionCallable


class PageDetector:
    """Detect one or more page crops from an input image."""

    def __init__(self, backend: PageDetectionBackendLike | None = None) -> None:
        """Create a page detector.

        :param backend: Optional low-level backend or async callable. When not
            provided, the full input image is treated as a single page.
        """
        self._backend = backend

    async def adetect(self, request: PageDetectionRequest) -> list[DocumentPage]:
        """Asynchronously detect pages for a single image.

        :param request: Detection request describing the source image.
        :returns: Detected page crops in reading order.
        """
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
        """Synchronously detect pages for a single image.

        :param request: Detection request describing the source image.
        :returns: Detected page crops in reading order.
        """
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
        """Create a document page detector.

        :param backend: Optional low-level detection backend or async callable.
        """
        self._page_detector = PageDetector(backend)

    async def detect_image(self, request: PageDetectionRequest) -> PageDetectionResult:
        """Detect pages in a single image.

        :param request: Detection request describing the source image.
        :returns: Detection result for one image input.
        """
        pages = await self._page_detector.adetect(request)
        return PageDetectionResult(pages=pages, source_type="image")

    def detect_image_sync(self, request: PageDetectionRequest) -> PageDetectionResult:
        """Synchronously detect pages in a single image.

        :param request: Detection request describing the source image.
        :returns: Detection result for one image input.
        """
        return run_sync(self.detect_image(request))

    async def detect_pdf(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        trim_margin: int = 30,
    ) -> PageDetectionResult:
        """Rasterize a PDF and detect pages on each image.

        :param path: PDF path to rasterize.
        :param dpi: Rasterization DPI used before detection.
        :param trim_margin: Pixel margin added around detected crops.
        :returns: Detection result containing all detected pages from the PDF.
        """
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
        """Synchronously rasterize a PDF and detect pages on each image.

        :param path: PDF path to rasterize.
        :param dpi: Rasterization DPI used before detection.
        :param trim_margin: Pixel margin added around detected crops.
        :returns: Detection result containing all detected pages from the PDF.
        """
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

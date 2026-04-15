"""Library-first public API for churro-ocr."""

from churro_ocr.document import DocumentOCRPipeline, DocumentOCRResult
from churro_ocr.errors import ChurroError, ConfigurationError, ProviderError
from churro_ocr.ocr import BatchOCRBackend, OCRBackend, OCRClient, OCRResult
from churro_ocr.page_detection import (
    DocumentPage,
    DocumentPageDetector,
    PageCandidate,
    PageDetectionBackend,
    PageDetectionRequest,
    PageDetectionResult,
    PageDetector,
)
from churro_ocr.templates import (
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
)

__all__ = [
    "CHURRO_3B_MODEL_ID",
    "CHURRO_3B_XML_TEMPLATE",
    "DEFAULT_OCR_TEMPLATE",
    "BatchOCRBackend",
    "ChurroError",
    "ConfigurationError",
    "DocumentOCRPipeline",
    "DocumentOCRResult",
    "DocumentPage",
    "DocumentPageDetector",
    "OCRBackend",
    "OCRClient",
    "OCRResult",
    "PageCandidate",
    "PageDetectionBackend",
    "PageDetectionRequest",
    "PageDetectionResult",
    "PageDetector",
    "ProviderError",
]

"""Public prompt defaults used by churro-ocr backends."""

from churro_ocr.prompts.layout import (
    DEFAULT_BOUNDARY_DETECTION_PROMPT,
)
from churro_ocr.prompts.ocr import (
    CHANDRA_OCR_LAYOUT_PROMPT,
    DEFAULT_MARKDOWN_OCR_USER_PROMPT,
    DEFAULT_OCR_OUTPUT_TAG,
    DEFAULT_OCR_SYSTEM_PROMPT,
    DEFAULT_OCR_USER_PROMPT,
    OLMOCR_V4_YAML_PROMPT,
    parse_chandra_response,
    parse_olmocr_response,
    strip_ocr_output_tag,
)

__all__ = [
    "CHANDRA_OCR_LAYOUT_PROMPT",
    "DEFAULT_BOUNDARY_DETECTION_PROMPT",
    "DEFAULT_MARKDOWN_OCR_USER_PROMPT",
    "DEFAULT_OCR_OUTPUT_TAG",
    "DEFAULT_OCR_SYSTEM_PROMPT",
    "DEFAULT_OCR_USER_PROMPT",
    "OLMOCR_V4_YAML_PROMPT",
    "parse_chandra_response",
    "parse_olmocr_response",
    "strip_ocr_output_tag",
]

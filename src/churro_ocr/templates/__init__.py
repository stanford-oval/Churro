"""Template helpers for model-specific OCR input rendering."""

from churro_ocr.templates.base import (
    OCRConversation,
    OCRPromptTemplate,
    OCRPromptTemplateCallable,
    OCRPromptTemplateLike,
    build_ocr_conversation,
)
from churro_ocr.templates.hf import HFChatTemplate
from churro_ocr.templates.presets import (
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_PROMPT,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    OLMOCR_2_7B_1025_FP8_MODEL_ID,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
)

__all__ = [
    "build_ocr_conversation",
    "CHURRO_3B_MODEL_ID",
    "CHURRO_3B_XML_TEMPLATE",
    "DEFAULT_OCR_TEMPLATE",
    "DOTS_OCR_1_5_MODEL_ID",
    "DOTS_OCR_1_5_OCR_PROMPT",
    "DOTS_OCR_1_5_OCR_TEMPLATE",
    "HFChatTemplate",
    "OLMOCR_2_7B_1025_FP8_MODEL_ID",
    "OLMOCR_2_7B_1025_MODEL_ID",
    "OLMOCR_2_7B_1025_OCR_TEMPLATE",
    "OCRConversation",
    "OCRPromptTemplate",
    "OCRPromptTemplateCallable",
    "OCRPromptTemplateLike",
]

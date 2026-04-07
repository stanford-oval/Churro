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
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    CHURRO_3B_MODEL_ID,
    CHURRO_3B_XML_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_PROMPT,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    LFM2_5_VL_1_6B_MODEL_ID,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    OLMOCR_2_7B_1025_FP8_MODEL_ID,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
    PADDLEOCR_VL_1_5_MODEL_ID,
    PADDLEOCR_VL_1_5_OCR_PROMPT,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
)

__all__ = [
    "build_ocr_conversation",
    "CHURRO_3B_MODEL_ID",
    "CHURRO_3B_XML_TEMPLATE",
    "CHANDRA_OCR_2_MODEL_ID",
    "CHANDRA_OCR_2_OCR_TEMPLATE",
    "DEFAULT_OCR_TEMPLATE",
    "DOTS_OCR_1_5_MODEL_ID",
    "DOTS_OCR_1_5_OCR_PROMPT",
    "DOTS_OCR_1_5_OCR_TEMPLATE",
    "LFM2_5_VL_1_6B_MODEL_ID",
    "LFM2_5_VL_1_6B_OCR_TEMPLATE",
    "PADDLEOCR_VL_1_5_MODEL_ID",
    "PADDLEOCR_VL_1_5_OCR_PROMPT",
    "PADDLEOCR_VL_1_5_OCR_TEMPLATE",
    "HFChatTemplate",
    "OLMOCR_2_7B_1025_FP8_MODEL_ID",
    "OLMOCR_2_7B_1025_MODEL_ID",
    "OLMOCR_2_7B_1025_OCR_TEMPLATE",
    "OCRConversation",
    "OCRPromptTemplate",
    "OCRPromptTemplateCallable",
    "OCRPromptTemplateLike",
]

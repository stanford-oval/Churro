"""Built-in OCR template presets."""

from __future__ import annotations

from churro_ocr.prompts import (
    DEFAULT_OCR_SYSTEM_PROMPT,
    DEFAULT_OCR_USER_PROMPT,
    OLMOCR_V4_YAML_PROMPT,
)
from churro_ocr.templates.hf import HFChatTemplate

CHURRO_3B_MODEL_ID = "stanford-oval/churro-3B"
DOTS_OCR_1_5_MODEL_ID = "kristaller486/dots.ocr-1.5"
OLMOCR_2_7B_1025_MODEL_ID = "allenai/olmOCR-2-7B-1025"
OLMOCR_2_7B_1025_FP8_MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
DEFAULT_OCR_TEMPLATE = HFChatTemplate(
    system_message=DEFAULT_OCR_SYSTEM_PROMPT,
    user_prompt=DEFAULT_OCR_USER_PROMPT,
)

CHURRO_3B_XML_TEMPLATE = HFChatTemplate(
    system_message="Transcribe the entirety of this historical document to XML format.",
    user_prompt=None,
)
DOTS_OCR_1_5_OCR_PROMPT = "Extract the text content from this image."
DOTS_OCR_1_5_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=DOTS_OCR_1_5_OCR_PROMPT,
)
OLMOCR_2_7B_1025_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=OLMOCR_V4_YAML_PROMPT,
    user_prompt_first=True,
)


__all__ = [
    "CHURRO_3B_MODEL_ID",
    "CHURRO_3B_XML_TEMPLATE",
    "DEFAULT_OCR_TEMPLATE",
    "DOTS_OCR_1_5_MODEL_ID",
    "DOTS_OCR_1_5_OCR_PROMPT",
    "DOTS_OCR_1_5_OCR_TEMPLATE",
    "OLMOCR_2_7B_1025_FP8_MODEL_ID",
    "OLMOCR_2_7B_1025_MODEL_ID",
    "OLMOCR_2_7B_1025_OCR_TEMPLATE",
]

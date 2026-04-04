"""Built-in OCR template presets."""

from __future__ import annotations

from churro_ocr.prompts import DEFAULT_OCR_SYSTEM_PROMPT, DEFAULT_OCR_USER_PROMPT
from churro_ocr.templates.hf import HFChatTemplate

CHURRO_3B_MODEL_ID = "stanford-oval/churro-3B"
DOTS_OCR_1_5_MODEL_ID = "kristaller486/dots.ocr-1.5"
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


__all__ = [
    "CHURRO_3B_MODEL_ID",
    "CHURRO_3B_XML_TEMPLATE",
    "DEFAULT_OCR_TEMPLATE",
    "DOTS_OCR_1_5_MODEL_ID",
    "DOTS_OCR_1_5_OCR_PROMPT",
    "DOTS_OCR_1_5_OCR_TEMPLATE",
]

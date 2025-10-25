"""Finetuned OCR implementation using a vLLM OpenAI-compatible server."""

from __future__ import annotations

from typing import override

from PIL import Image

from churro.evaluation.xml_utils import extract_actual_text_from_xml
from churro.utils.llm.core import run_llm_async
from churro.utils.llm.models import COMPLETION_TOKENS_FOR_STANDARD_MODELS
from churro.utils.log_utils import logger

from .base_ocr import BaseOCR


SYSTEM_MESSAGE = """Transcribe the entiretly of this historical documents to XML format."""


class FineTunedOCR(BaseOCR):
    """Finetuned model OCR using a locally hosted vLLM HTTP server."""

    def __init__(
        self,
        engine: str,
        max_new_tokens: int = COMPLETION_TOKENS_FOR_STANDARD_MODELS,
        strip_xml: bool = False,
        **_: object,
    ) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self._system_message: str = SYSTEM_MESSAGE
        self._strip_xml = strip_xml

    @override
    async def process_image(self, image: Image.Image) -> str:
        """Process a single image using the finetuned OCR model."""
        llm_output = await run_llm_async(
            model=self.engine,
            system_prompt_text=self._system_message,
            user_message_text=None,
            user_message_image=image,
        )
        if not isinstance(llm_output, str):
            llm_output = ""
        if self._strip_xml and llm_output:
            try:
                llm_output = extract_actual_text_from_xml(llm_output)
            except Exception:
                logger.exception(
                    "Failed to extract text from finetuned XML output; returning raw XML."
                )
        return llm_output

    def get_system_name(self) -> str:
        """Return human-readable system name."""
        return "Fine-tuned OCR Model"

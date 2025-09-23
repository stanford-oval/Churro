"""Finetuned OCR implementation using a vLLM OpenAI-compatible server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from utils.llm.core import run_llm_async
from utils.llm.models import COMPLETION_TOKENS_FOR_STANDARD_MODELS
from utils.utils import run_async_in_parallel

from .base_ocr import BaseOCR


if TYPE_CHECKING:  # pragma: no cover - type checking only
    pass

SYSTEM_MESSAGE = """Transcribe the entiretly of this historical documents to XML format."""


class FineTunedOCR(BaseOCR):
    """Finetuned model OCR using a locally hosted vLLM HTTP server."""

    def __init__(
        self,
        engine: str,
        max_concurrency: int = 4,
        max_new_tokens: int = COMPLETION_TOKENS_FOR_STANDARD_MODELS,
        **_: Any,
    ) -> None:
        self.engine = engine
        self.max_concurrency = max_concurrency
        self.max_new_tokens = max_new_tokens
        self._system_message: str = SYSTEM_MESSAGE

    @override
    async def process(self, dataset: list[dict]) -> list[str]:
        return await run_async_in_parallel(
            self._process_single_example,
            dataset,
            max_concurrency=self.max_concurrency,
            desc="Fine-tuned OCR",
        )

    async def _process_single_example(self, example: dict) -> str:
        """Process a single example using the finetuned OCR model."""
        llm_output = await run_llm_async(
            model=self.engine,
            system_prompt_text=self._system_message,
            user_message_text=None,
            user_message_image=example["image"],
        )
        if not isinstance(llm_output, str):
            llm_output = ""
        return llm_output

    def get_system_name(self) -> str:
        """Return human-readable system name."""
        return "Fine-tuned OCR Model"

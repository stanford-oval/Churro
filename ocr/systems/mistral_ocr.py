"""Mistral OCR implementations."""

import os

import mistralai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from utils.llm import encode_image
from utils.utils import run_async_in_parallel

from .base_ocr import BaseOCR


class MistralOCR(BaseOCR):
    """Mistral OCR implementation."""

    def __init__(self, max_concurrency: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_concurrency = max_concurrency
        self._client = None

    def _get_client(self) -> mistralai.Mistral:
        """Get or create Mistral client."""
        if self._client is None:
            self._client = mistralai.Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        return self._client

    async def process(self, dataset: list[dict]) -> list[str]:
        """Process examples using Mistral OCR."""
        predicted_texts = await run_async_in_parallel(
            self._process_single_example,
            dataset,
            desc="Mistral OCR",
            max_concurrency=self.max_concurrency,
        )
        return predicted_texts

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(30),
        retry=retry_if_exception_type(mistralai.models.sdkerror.SDKError),
    )
    async def _process_single_example(self, example: dict) -> str:
        """Process a single image using Mistral OCR."""
        client = self._get_client()

        image = example["image"]
        base64_image = encode_image(image)
        if base64_image is None:
            return ""

        image_url = f"data:image/jpeg;base64,{base64_image}"
        response = await client.ocr.process_async(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": image_url,
            },
        )

        return response.pages[0].markdown

    def get_system_name(self) -> str:
        """Return human-readable system name."""
        return "Mistral OCR"

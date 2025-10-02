"""Mistral OCR implementations."""

import os
from typing import override

import mistralai
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from utils.llm import encode_image

from .base_ocr import BaseOCR


class MistralOCR(BaseOCR):
    """Mistral OCR implementation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = None

    def _get_client(self) -> mistralai.Mistral:
        """Get or create Mistral client."""
        if self._client is None:
            self._client = mistralai.Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(30),
        retry=retry_if_exception_type(mistralai.models.sdkerror.SDKError),
    )
    @override
    async def process_image(self, image: Image.Image) -> str:
        """Process a single image using Mistral OCR."""
        client = self._get_client()

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

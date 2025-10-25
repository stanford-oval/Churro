"""Azure OCR implementations."""

from typing import override

from PIL import Image

from .base_ocr import BaseOCR
from .detect_layout import (
    run_azure_document_analysis_on_image,
)


class AzureOCR(BaseOCR):
    """Azure Document Intelligence OCR."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._initialized = False

    @override
    async def process_image(self, image: Image.Image) -> str:
        """Process a single image using Azure OCR."""
        azure_output = await run_azure_document_analysis_on_image(
            image, skip_paragraphs=False, output_ocr_text=True
        )
        assert isinstance(azure_output, str), (
            "Azure OCR output must be a string when output_ocr_text=True"
        )
        return azure_output

    def get_system_name(self) -> str:
        """Return human-readable system name."""
        return "Azure Document Intelligence OCR"

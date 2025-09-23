"""Azure OCR implementations."""

from utils.utils import run_async_in_parallel

from .base_ocr import BaseOCR
from .detect_layout import (
    run_azure_document_analysis_on_image,
)


class AzureOCR(BaseOCR):
    """Azure Document Intelligence OCR."""

    def __init__(self, max_concurrency: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_concurrency = max_concurrency
        self._initialized = False

    async def process(self, dataset: list[dict]) -> list[str]:
        """Process examples using Azure OCR."""
        predicted_texts = await run_async_in_parallel(
            self._process_single_example,
            dataset,
            desc="Azure OCR",
            max_concurrency=self.max_concurrency,
        )
        return predicted_texts

    async def _process_single_example(self, example: dict) -> str:
        """Process a single image using Azure OCR."""
        azure_output = await run_azure_document_analysis_on_image(
            example["image"], skip_paragraphs=False, output_ocr_text=True
        )
        assert isinstance(azure_output, str), (
            "Azure OCR output must be a string when output_ocr_text=True"
        )
        return azure_output

    def get_system_name(self) -> str:
        """Return human-readable system name."""
        return "Azure Document Intelligence OCR"

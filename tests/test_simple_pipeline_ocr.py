from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from ocr.systems.hybrid_pipeline_ocr import HybridPipelineOCR
from utils.log_utils import logger


@pytest.mark.asyncio
async def test_simple_pipeline_ocr() -> None:
    """Test that the hybrid OCR pipeline returns a sufficiently long transcription for a sample image."""
    image_path: str = str(Path(__file__).with_name("ahisto_103_84.jpeg"))
    image = Image.open(image_path)
    _example: dict = {
        "image": image,
        "file_name": image_path,
        "transcription": "",
        "main_language": "en",
        "main_script": "Latin",
        "languages": ["en"],
        "scripts": ["Latin"],
        "document_type": "generic",
        "dataset_id": "test-dataset",
    }
    output: str = await HybridPipelineOCR(engine="gpt-4.1").process_single_example(_example)

    logger.info(output)
    assert len(output) > 100

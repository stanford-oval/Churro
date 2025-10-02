from __future__ import annotations

from pathlib import Path

import pytest

from ocr.systems.hybrid_pipeline_ocr import HybridPipelineOCR
from utils.log_utils import logger


@pytest.mark.asyncio
async def test_hybrid_ocr() -> None:
    """Test that the hybrid OCR pipeline returns a sufficiently long transcription for a sample image."""
    image_path: str = str(Path(__file__).with_name("ahisto_103_84.jpeg"))
    output: str = await HybridPipelineOCR(engine="gpt-4.1").process_image_from_file(image_path)

    logger.info(output)
    assert len(output) > 100

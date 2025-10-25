from __future__ import annotations

from pathlib import Path

import pytest

from churro.systems.llm_ocr import ZeroShotLLMOCR


@pytest.mark.asyncio
async def test_llm_ocr() -> None:
    """Test that the LLM OCR returns a sufficiently long transcription for a sample image."""
    image_path: str = str(Path(__file__).with_name("churro_dataset_sample_1.jpeg"))
    output: str = await ZeroShotLLMOCR(engine="gpt-4.1-mini").process_image_from_file(image_path)

    assert len(output) > 500, "OCR output is unexpectedly short"

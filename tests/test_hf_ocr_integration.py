from __future__ import annotations

import os

import pytest

from churro_ocr.document import DocumentOCRPipeline
from churro_ocr.providers import HuggingFaceOptions, OCRBackendSpec, build_ocr_backend

_LIVE_FLAG = "CHURRO_RUN_LIVE_HF_TESTS"
_ALLOW_CPU_FLAG = "CHURRO_ALLOW_CPU_HF_TESTS"
_MODEL_ENV = "CHURRO_HF_MODEL_ID"
_DEVICE_MAP_ENV = "CHURRO_HF_DEVICE_MAP"
_DEFAULT_MODEL_ID = "stanford-oval/churro-3B"


@pytest.mark.integration
def test_churro_3b_live_hf_ocr_on_minimal_pdf(minimal_pdf_path, test_artifact_dir_path) -> None:
    if os.getenv(_LIVE_FLAG) != "1":
        pytest.skip(f"Set {_LIVE_FLAG}=1 to run live Hugging Face OCR integration tests.")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() and os.getenv(_ALLOW_CPU_FLAG) != "1":
        pytest.skip(
            "CUDA is unavailable. Set CHURRO_ALLOW_CPU_HF_TESTS=1 "
            "to allow the HF OCR integration test on CPU."
        )

    model_id = os.getenv(_MODEL_ENV, _DEFAULT_MODEL_ID)
    device_map = os.getenv(_DEVICE_MAP_ENV, "auto")

    backend = build_ocr_backend(
        OCRBackendSpec(
            provider="hf",
            model=model_id,
            profile=_DEFAULT_MODEL_ID,
            options=HuggingFaceOptions(
                model_kwargs={"device_map": device_map},
            ),
        )
    )
    pipeline = DocumentOCRPipeline(backend)

    result = pipeline.process_pdf_sync(minimal_pdf_path, dpi=150, trim_margin=0)

    test_artifact_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = test_artifact_dir_path / "hf-churro3b-minimal-document.xml"
    output_path.write_text("\n\n".join(result.texts()))

    assert result.source_type == "pdf"
    assert len(result.pages) >= 1
    assert result.pages[0].provider_name == "huggingface-transformers"
    assert result.pages[0].model_name == "churro-3B"
    assert result.pages[0].text is not None
    assert result.pages[0].text.strip()

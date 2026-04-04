from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

import churro_ocr.cli as cli_module
from churro_ocr.cli import app
from churro_ocr.ocr import OCRResult
from churro_ocr.page_detection import DocumentPage, PageDetectionResult
from churro_ocr.prompts import DEFAULT_OCR_OUTPUT_TAG
from churro_ocr.templates import DEFAULT_OCR_TEMPLATE, DOTS_OCR_1_5_OCR_TEMPLATE

runner = CliRunner()


def test_transcribe_cli_writes_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    output_path = tmp_path / "out.txt"
    Image.new("RGB", (10, 10), color="white").save(image_path)

    class _FakeBackend:
        async def ocr(self, page):  # noqa: ANN001
            image = page.image
            return OCRResult(
                text=f"ocr:{image.width}x{image.height}",
                provider_name="fake",
                model_name="fake-model",
            )

    monkeypatch.setattr("churro_ocr.cli._build_ocr_backend", lambda **_: _FakeBackend())

    result = runner.invoke(
        app,
        [
            "transcribe",
            "--image",
            str(image_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.read_text() == "ocr:10x10"


def test_extract_pages_cli_writes_page_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    output_dir = tmp_path / "pages"
    Image.new("RGB", (10, 10), color="white").save(image_path)

    class _FakePageDetector:
        def __init__(self, **_: object) -> None:
            pass

        def detect_image_sync(self, request):  # noqa: ANN001
            _ = request.require_image()
            return PageDetectionResult(
                pages=[
                    DocumentPage(
                        page_index=0,
                        image=Image.new("RGB", (8, 8), color="white"),
                        source_index=0,
                    )
                ],
                source_type="image",
            )

    monkeypatch.setattr("churro_ocr.cli.DocumentPageDetector", _FakePageDetector)

    result = runner.invoke(
        app,
        [
            "extract-pages",
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "page_0000.png").exists()


def test_build_ocr_backend_aligns_templates_for_generic_models() -> None:
    litellm_backend = cli_module._build_ocr_backend(
        backend="litellm",
        model="example/model",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )
    hf_backend = cli_module._build_ocr_backend(
        backend="hf",
        model="example/model",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )
    vllm_backend = cli_module._build_ocr_backend(
        backend="vllm",
        model="example/model",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )

    assert litellm_backend.template == DEFAULT_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == vllm_backend.template
    assert litellm_backend.model_name == "example/model"
    assert hf_backend.model_name == "example/model"
    assert vllm_backend.model_name == "example/model"
    assert f"<{DEFAULT_OCR_OUTPUT_TAG}>" in litellm_backend.template.system_message
    assert f"</{DEFAULT_OCR_OUTPUT_TAG}>" in litellm_backend.template.system_message
    assert f"<{DEFAULT_OCR_OUTPUT_TAG}>" in litellm_backend.template.user_prompt
    assert f"</{DEFAULT_OCR_OUTPUT_TAG}>" in litellm_backend.template.user_prompt


def test_build_ocr_backend_aligns_templates_for_dots() -> None:
    litellm_backend = cli_module._build_ocr_backend(
        backend="litellm",
        model="kristaller486/dots.ocr-1.5",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )
    hf_backend = cli_module._build_ocr_backend(
        backend="hf",
        model="kristaller486/dots.ocr-1.5",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )
    vllm_backend = cli_module._build_ocr_backend(
        backend="vllm",
        model="kristaller486/dots.ocr-1.5",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )

    assert litellm_backend.template == DOTS_OCR_1_5_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == vllm_backend.template
    assert litellm_backend.model_name == "dots.ocr-1.5"
    assert hf_backend.model_name == "dots.ocr-1.5"
    assert vllm_backend.model_name == "dots.ocr-1.5"


def test_build_ocr_backend_uses_generic_defaults_for_qwen_3_5_0_8b() -> None:
    litellm_backend = cli_module._build_ocr_backend(
        backend="litellm",
        model="Qwen/Qwen3.5-0.8B",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )
    hf_backend = cli_module._build_ocr_backend(
        backend="hf",
        model="Qwen/Qwen3.5-0.8B",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )
    vllm_backend = cli_module._build_ocr_backend(
        backend="vllm",
        model="Qwen/Qwen3.5-0.8B",
        endpoint=None,
        api_key=None,
        base_url=None,
        api_version=None,
    )

    assert litellm_backend.template == DEFAULT_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == vllm_backend.template
    assert litellm_backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert hf_backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert vllm_backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert vllm_backend.llm_kwargs == {}


def test_module_entrypoint_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "churro_ocr", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "transcribe" in result.stdout
    assert "extract-pages" in result.stdout


def test_console_script_help() -> None:
    executable = shutil.which("churro-ocr")
    assert executable is not None

    result = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "transcribe" in result.stdout
    assert "extract-pages" in result.stdout

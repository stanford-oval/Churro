from __future__ import annotations

import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import typer
from PIL import Image

import churro_ocr.cli as cli_module
from churro_ocr.cli import app
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRResult
from churro_ocr.page_detection import DocumentPage, PageDetectionResult
from churro_ocr.prompts import DEFAULT_OCR_OUTPUT_TAG
from churro_ocr.providers.specs import DEFAULT_OCR_MAX_TOKENS
from churro_ocr.templates import (
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    DEEPSEEK_OCR_2_MODEL_ID,
    DEEPSEEK_OCR_2_OCR_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
    DOTS_MOCR_OCR_TEMPLATE,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    INFINITY_PARSER_7B_MODEL_ID,
    INFINITY_PARSER_7B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_MODEL_ID,
    MINERU2_5_2509_1_2B_OCR_TEMPLATE,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
    PADDLEOCR_VL_1_5_MODEL_ID,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
)

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import CliRunner

    from churro_ocr.page_detection import PageDetectionRequest
    from churro_ocr.providers.hf import HuggingFaceVisionOCRBackend
    from churro_ocr.providers.ocr import LiteLLMVisionOCRBackend, MistralOCRBackend
    from churro_ocr.templates.hf import HFChatTemplate
    from tests._types import WriteImageFile


def _build_litellm_backend(model: str) -> LiteLLMVisionOCRBackend:
    return cast(
        "LiteLLMVisionOCRBackend",
        cli_module._build_ocr_backend(
            backend="litellm",
            model=model,
            endpoint=None,
            api_key=None,
            base_url=None,
            api_version=None,
        ),
    )


def _build_hf_backend(model: str) -> HuggingFaceVisionOCRBackend:
    return cast(
        "HuggingFaceVisionOCRBackend",
        cli_module._build_ocr_backend(
            backend="hf",
            model=model,
            endpoint=None,
            api_key=None,
            base_url=None,
            api_version=None,
        ),
    )


def _build_openai_compatible_backend(model: str) -> LiteLLMVisionOCRBackend:
    return cast(
        "LiteLLMVisionOCRBackend",
        cli_module._build_ocr_backend(
            backend="openai-compatible",
            model=model,
            endpoint=None,
            api_key=None,
            base_url="http://127.0.0.1:8000/v1",
            api_version=None,
        ),
    )


def _build_mistral_backend(model: str) -> MistralOCRBackend:
    return cast(
        "MistralOCRBackend",
        cli_module._build_ocr_backend(
            backend="mistral",
            model=model,
            endpoint=None,
            api_key="secret",
            base_url=None,
            api_version=None,
        ),
    )


def test_transcribe_cli_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cli_runner: CliRunner,
    write_image_file: WriteImageFile,
) -> None:
    image_path = write_image_file(size=(10, 10))
    output_path = tmp_path / "out.txt"

    class _FakeBackend:
        async def ocr(self, page: DocumentPage) -> OCRResult:
            image = page.image
            return OCRResult(
                text=f"ocr:{image.width}x{image.height}",
                provider_name="fake",
                model_name="fake-model",
            )

    monkeypatch.setattr("churro_ocr.cli._build_ocr_backend", lambda **_: _FakeBackend())

    result = cli_runner.invoke(
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


def test_extract_pages_cli_writes_page_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cli_runner: CliRunner,
    write_image_file: WriteImageFile,
) -> None:
    image_path = write_image_file(size=(10, 10))
    output_dir = tmp_path / "pages"

    class _FakePageDetector:
        def __init__(self, **_: object) -> None:
            pass

        def detect_image_sync(self, request: PageDetectionRequest) -> PageDetectionResult:
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

    result = cli_runner.invoke(
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
    litellm_backend = _build_litellm_backend("example/model")
    hf_backend = _build_hf_backend("example/model")
    openai_backend = _build_openai_compatible_backend("example/model")

    assert litellm_backend.template == DEFAULT_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "example/model"
    assert hf_backend.model_name == "example/model"
    assert openai_backend.model_name == "example/model"
    template = cast("HFChatTemplate", litellm_backend.template)
    assert template.system_message is not None
    assert template.user_prompt is not None
    assert f"<{DEFAULT_OCR_OUTPUT_TAG}>" in template.system_message
    assert f"</{DEFAULT_OCR_OUTPUT_TAG}>" in template.system_message
    assert f"<{DEFAULT_OCR_OUTPUT_TAG}>" in template.user_prompt
    assert f"</{DEFAULT_OCR_OUTPUT_TAG}>" in template.user_prompt


def test_build_ocr_backend_requires_pinned_mistral_model() -> None:
    with pytest.raises(typer.BadParameter, match="mistral-ocr-2505, mistral-ocr-2512"):
        cli_module._build_ocr_backend(
            backend="mistral",
            model="mistral-ocr-latest",
            endpoint=None,
            api_key="secret",
            base_url=None,
            api_version=None,
        )


def test_build_ocr_backend_accepts_pinned_mistral_model() -> None:
    mistral_backend = _build_mistral_backend("mistral-ocr-2512")

    assert mistral_backend.model == "mistral-ocr-2512"


def test_build_ocr_backend_aligns_templates_for_dots() -> None:
    litellm_backend = _build_litellm_backend("kristaller486/dots.ocr-1.5")
    hf_backend = _build_hf_backend("kristaller486/dots.ocr-1.5")
    openai_backend = _build_openai_compatible_backend("kristaller486/dots.ocr-1.5")

    assert litellm_backend.template == DOTS_OCR_1_5_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "dots.ocr-1.5"
    assert hf_backend.model_name == "dots.ocr-1.5"
    assert openai_backend.model_name == "dots.ocr-1.5"
    assert litellm_backend.transport.config.completion_kwargs == {
        "max_tokens": 2_048,
        "temperature": 0.0,
    }
    assert openai_backend.transport.config.completion_kwargs == {
        "max_tokens": 2_048,
        "temperature": 0.0,
    }


def test_build_ocr_backend_aligns_templates_for_dots_mocr() -> None:
    litellm_backend = _build_litellm_backend("rednote-hilab/dots.mocr")
    hf_backend = _build_hf_backend("rednote-hilab/dots.mocr")
    openai_backend = _build_openai_compatible_backend("rednote-hilab/dots.mocr")

    assert litellm_backend.template == DOTS_MOCR_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "dots.mocr"
    assert hf_backend.model_name == "dots.mocr"
    assert openai_backend.model_name == "dots.mocr"
    assert litellm_backend.transport.config.completion_kwargs == {
        "max_tokens": DEFAULT_OCR_MAX_TOKENS,
        "temperature": 0.0,
    }
    assert openai_backend.transport.config.completion_kwargs == {
        "max_tokens": DEFAULT_OCR_MAX_TOKENS,
        "temperature": 0.0,
    }


def test_build_ocr_backend_aligns_templates_for_deepseek_ocr_2() -> None:
    litellm_backend = _build_litellm_backend(DEEPSEEK_OCR_2_MODEL_ID)
    hf_backend = _build_hf_backend(DEEPSEEK_OCR_2_MODEL_ID)
    openai_backend = _build_openai_compatible_backend(DEEPSEEK_OCR_2_MODEL_ID)

    assert litellm_backend.template == DEEPSEEK_OCR_2_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "DeepSeek-OCR-2"
    assert hf_backend.model_name == "DeepSeek-OCR-2"
    assert openai_backend.model_name == "DeepSeek-OCR-2"
    assert litellm_backend.transport.config.completion_kwargs == {
        "max_tokens": 8_192,
        "temperature": 0.0,
    }
    assert openai_backend.transport.config.completion_kwargs == {
        "max_tokens": 8_192,
        "temperature": 0.0,
    }


def test_build_ocr_backend_aligns_templates_for_paddleocr_vl() -> None:
    litellm_backend = _build_litellm_backend(PADDLEOCR_VL_1_5_MODEL_ID)
    hf_backend = _build_hf_backend(PADDLEOCR_VL_1_5_MODEL_ID)
    openai_backend = _build_openai_compatible_backend(PADDLEOCR_VL_1_5_MODEL_ID)

    assert litellm_backend.template == PADDLEOCR_VL_1_5_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "PaddleOCR-VL-1.5"
    assert hf_backend.model_name == "PaddleOCR-VL-1.5"
    assert openai_backend.model_name == "PaddleOCR-VL-1.5"
    assert litellm_backend.transport.config.completion_kwargs == {
        "max_tokens": 4_096,
        "temperature": 0.0,
    }
    assert openai_backend.transport.config.completion_kwargs == {
        "max_tokens": 4_096,
        "temperature": 0.0,
    }


def test_build_ocr_backend_aligns_templates_for_mineru2_5() -> None:
    hf_backend = _build_hf_backend(MINERU2_5_2509_1_2B_MODEL_ID)
    openai_backend = _build_openai_compatible_backend(MINERU2_5_2509_1_2B_MODEL_ID)

    assert hf_backend.template == MINERU2_5_2509_1_2B_OCR_TEMPLATE
    assert hf_backend.template == openai_backend.template
    assert hf_backend.model_name == "MinerU2.5-2509-1.2B"
    assert openai_backend.model_name == "MinerU2.5-2509-1.2B"
    assert openai_backend.transport.config.completion_kwargs == {}


def test_build_ocr_backend_aligns_templates_for_infinity_parser() -> None:
    litellm_backend = _build_litellm_backend(INFINITY_PARSER_7B_MODEL_ID)
    hf_backend = _build_hf_backend(INFINITY_PARSER_7B_MODEL_ID)
    openai_backend = _build_openai_compatible_backend(INFINITY_PARSER_7B_MODEL_ID)

    assert litellm_backend.template == INFINITY_PARSER_7B_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "Infinity-Parser-7B"
    assert hf_backend.model_name == "Infinity-Parser-7B"
    assert openai_backend.model_name == "Infinity-Parser-7B"
    assert litellm_backend.transport.config.completion_kwargs == {
        "max_tokens": 8_192,
        "temperature": 0.0,
        "top_p": 0.95,
    }
    assert openai_backend.transport.config.completion_kwargs == {
        "max_tokens": 8_192,
        "temperature": 0.0,
        "top_p": 0.95,
    }


def test_build_ocr_backend_rejects_mineru2_5_for_litellm() -> None:
    with pytest.raises(ConfigurationError, match=r"MinerU2\.5 requires the built-in two-step pipeline"):
        cli_module._build_ocr_backend(
            backend="litellm",
            model=MINERU2_5_2509_1_2B_MODEL_ID,
            endpoint=None,
            api_key=None,
            base_url=None,
            api_version=None,
        )


def test_build_ocr_backend_uses_generic_defaults_for_qwen_3_5_0_8b() -> None:
    litellm_backend = _build_litellm_backend("Qwen/Qwen3.5-0.8B")
    hf_backend = _build_hf_backend("Qwen/Qwen3.5-0.8B")
    openai_backend = _build_openai_compatible_backend("Qwen/Qwen3.5-0.8B")

    assert litellm_backend.template == DEFAULT_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert hf_backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert openai_backend.model_name == "Qwen/Qwen3.5-0.8B"


def test_build_ocr_backend_aligns_templates_for_olmocr() -> None:
    litellm_backend = _build_litellm_backend(OLMOCR_2_7B_1025_MODEL_ID)
    hf_backend = _build_hf_backend(OLMOCR_2_7B_1025_MODEL_ID)
    openai_backend = _build_openai_compatible_backend(OLMOCR_2_7B_1025_MODEL_ID)

    assert litellm_backend.template == OLMOCR_2_7B_1025_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "olmOCR-2-7B-1025"
    assert hf_backend.model_name == "olmOCR-2-7B-1025"
    assert openai_backend.model_name == "olmOCR-2-7B-1025"
    assert openai_backend.transport.config.completion_kwargs == {
        "max_tokens": 8_000,
        "temperature": 0.1,
    }


def test_build_ocr_backend_aligns_templates_for_chandra() -> None:
    litellm_backend = _build_litellm_backend(CHANDRA_OCR_2_MODEL_ID)
    hf_backend = _build_hf_backend(CHANDRA_OCR_2_MODEL_ID)
    openai_backend = _build_openai_compatible_backend(CHANDRA_OCR_2_MODEL_ID)

    assert litellm_backend.template == CHANDRA_OCR_2_OCR_TEMPLATE
    assert litellm_backend.template == hf_backend.template == openai_backend.template
    assert litellm_backend.model_name == "chandra-ocr-2"
    assert hf_backend.model_name == "chandra-ocr-2"
    assert openai_backend.model_name == "chandra-ocr-2"
    assert openai_backend.transport.config.completion_kwargs == {
        "max_tokens": 12_384,
        "temperature": 0.0,
        "top_p": 0.1,
    }


def test_install_command_invokes_runtime_installer(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner: CliRunner,
) -> None:
    captured: dict[str, object] = {}

    def _fake_install_runtime_dependencies(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            target="local",
            notes=("runtime ready",),
        )

    monkeypatch.setattr(
        "churro_ocr.cli.install_runtime_dependencies",
        _fake_install_runtime_dependencies,
    )

    result = cli_runner.invoke(
        app,
        [
            "install",
            "local",
            "--torch-backend",
            "cu126",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "target": "local",
        "torch_backend": "cu126",
    }
    assert "Installed runtime target: local" in result.output
    assert "runtime ready" in result.output


def test_install_command_surfaces_configuration_errors(
    monkeypatch: pytest.MonkeyPatch,
    cli_runner: CliRunner,
) -> None:
    def _raise_configuration_error(**_: object) -> SimpleNamespace:
        message = "missing uv"
        raise ConfigurationError(message)

    monkeypatch.setattr(
        "churro_ocr.cli.install_runtime_dependencies",
        _raise_configuration_error,
    )

    result = cli_runner.invoke(app, ["install", "hf"])

    assert result.exit_code == 1
    assert "missing uv" in result.output


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
    assert "install" in result.stdout


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
    assert "install" in result.stdout

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

import churro_ocr.cli as cli_module
from churro_ocr.cli import app
from churro_ocr.ocr import OCRResult
from churro_ocr.page_detection import DocumentPage, PageDetectionResult

runner = CliRunner()


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (12, 12), color="white").save(image_path)
    return image_path


def _sample_pdf_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "minimal-document.pdf"


@pytest.mark.parametrize(
    ("args", "expected_parts"),
    [
        (["--backend", "litellm"], ("--model is required for backend=litellm",)),
        (
            ["--backend", "openai-compatible", "--model", "local-model"],
            ("required for", "backend=openai-compatible"),
        ),
        (["--backend", "azure"], ("--endpoint and --api-key are required for backend=azure",)),
        (["--backend", "mistral"], ("--api-key is required for backend=mistral",)),
        (["--backend", "hf"], ("--model is required for backend=hf",)),
        (["--backend", "vllm"], ("--model is required for backend=vllm",)),
    ],
)
def test_transcribe_cli_validates_backend_requirements(
    sample_image_path: Path,
    args: list[str],
    expected_parts: tuple[str, ...],
) -> None:
    result = runner.invoke(
        app,
        ["transcribe", "--image", str(sample_image_path), *args],
    )

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    for expected_part in expected_parts:
        assert expected_part in output


def test_transcribe_cli_rejects_unsupported_backend(sample_image_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--image",
            str(sample_image_path),
            "--backend",
            "unsupported",
            "--model",
            "example/model",
        ],
    )

    assert result.exit_code != 0
    assert "Unsupported backend: unsupported" in result.output


def test_transcribe_cli_echoes_text_without_output(
    monkeypatch: pytest.MonkeyPatch,
    sample_image_path: Path,
) -> None:
    class _FakeBackend:
        async def ocr(self, page: DocumentPage) -> OCRResult:
            return OCRResult(
                text=f"plain:{page.image.width}x{page.image.height}",
                provider_name="fake",
                model_name="fake-model",
            )

    monkeypatch.setattr("churro_ocr.cli._build_ocr_backend", lambda **_: _FakeBackend())

    result = runner.invoke(
        app,
        [
            "transcribe",
            "--image",
            str(sample_image_path),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "plain:12x12"


@pytest.mark.parametrize(
    "command_args",
    [
        ["extract-pages", "--output-dir", "unused"],
        [
            "extract-pages",
            "--image",
            "scan.png",
            "--pdf",
            "document.pdf",
            "--output-dir",
            "unused",
        ],
    ],
)
def test_extract_pages_cli_requires_exactly_one_image_or_pdf(
    sample_image_path: Path,
    tmp_path: Path,
    command_args: list[str],
) -> None:
    output_dir = tmp_path / "pages"
    pdf_path = _sample_pdf_path()
    args = [
        str(sample_image_path) if token == "scan.png" else str(pdf_path) if token == "document.pdf" else token
        for token in command_args
    ]
    args = [str(output_dir) if token == "unused" else token for token in args]

    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "Provide exactly one of --image or --pdf." in result.output


@pytest.mark.parametrize(
    ("args", "expected_parts"),
    [
        (
            ["--page-detector", "llm"],
            ("--model is required when --page-detector=llm",),
        ),
        (
            ["--page-detector", "azure"],
            ("required when", "--page-detector=azure"),
        ),
    ],
)
def test_extract_pages_cli_validates_page_detector_requirements(
    sample_image_path: Path,
    tmp_path: Path,
    args: list[str],
    expected_parts: tuple[str, ...],
) -> None:
    output_dir = tmp_path / "pages"
    result = runner.invoke(
        app,
        [
            "extract-pages",
            "--image",
            str(sample_image_path),
            "--output-dir",
            str(output_dir),
            *args,
        ],
    )

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    for expected_part in expected_parts:
        assert expected_part in output


def test_extract_pages_cli_writes_pdf_page_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class _FakePageDetector:
        def __init__(self, *, backend: object | None = None) -> None:
            calls["backend"] = backend

        def detect_image_sync(self, request: object) -> PageDetectionResult:
            raise AssertionError(f"Unexpected image request: {request!r}")

        def detect_pdf_sync(self, path: Path, *, dpi: int, trim_margin: int) -> PageDetectionResult:
            calls["path"] = Path(path)
            calls["dpi"] = dpi
            calls["trim_margin"] = trim_margin
            return PageDetectionResult(
                pages=[
                    DocumentPage(
                        page_index=0,
                        image=Image.new("RGB", (8, 8), color="white"),
                        source_index=0,
                    )
                ],
                source_type="pdf",
            )

    monkeypatch.setattr("churro_ocr.cli.DocumentPageDetector", _FakePageDetector)

    output_dir = tmp_path / "pages"
    pdf_path = _sample_pdf_path()
    result = runner.invoke(
        app,
        [
            "extract-pages",
            "--pdf",
            str(pdf_path),
            "--output-dir",
            str(output_dir),
            "--dpi",
            "150",
            "--trim-margin",
            "0",
        ],
    )

    output_path = output_dir / "page_0000.png"
    assert result.exit_code == 0
    assert output_path.exists()
    assert str(output_path) in result.output
    assert calls == {
        "backend": None,
        "path": pdf_path,
        "dpi": 150,
        "trim_margin": 0,
    }


def test_module_main_invokes_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_main() -> None:
        calls["count"] += 1

    monkeypatch.setattr(cli_module, "main", _fake_main)

    runpy.run_module("churro_ocr.__main__", run_name="__main__")

    assert calls["count"] == 1

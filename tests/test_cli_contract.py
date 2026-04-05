from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from PIL import Image

import churro_ocr.cli as cli_module
from churro_ocr.cli import app
from churro_ocr.ocr import OCRResult
from churro_ocr.page_detection import DocumentPage, PageDetectionResult


@pytest.fixture
def sample_image_path(write_image_file) -> Path:
    return write_image_file(size=(12, 12))


@pytest.mark.parametrize(
    ("args", "expected_parts"),
    [
        (["--backend", "litellm"], ("--model is required for backend=litellm",)),
        (
            ["--backend", "openai-compatible", "--model", "local-model"],
            ("--model and --base-url are required for", "backend=openai-compatible"),
        ),
        (["--backend", "azure"], ("--endpoint and --api-key are required for backend=azure",)),
        (["--backend", "mistral"], ("--api-key is required for backend=mistral",)),
        (["--backend", "hf"], ("--model is required for backend=hf",)),
    ],
)
def test_transcribe_cli_validates_backend_requirements(
    sample_image_path: Path,
    args: list[str],
    expected_parts: tuple[str, ...],
    cli_runner,
) -> None:
    result = cli_runner.invoke(
        app,
        ["transcribe", "--image", str(sample_image_path), *args],
    )

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    for expected_part in expected_parts:
        assert expected_part in output


def test_transcribe_cli_allows_openai_compatible_backend_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    sample_image_path: Path,
    cli_runner,
) -> None:
    captured: dict[str, object] = {}

    class _FakeBackend:
        async def ocr(self, page: DocumentPage) -> OCRResult:
            captured["size"] = (page.image.width, page.image.height)
            return OCRResult(text="ok", provider_name="fake", model_name="fake-model")

    def _fake_build_ocr_backend(spec: cli_module.OCRBackendSpec) -> _FakeBackend:
        captured["spec"] = spec
        return _FakeBackend()

    monkeypatch.setattr(
        "churro_ocr.cli.build_ocr_backend",
        _fake_build_ocr_backend,
    )

    result = cli_runner.invoke(
        app,
        [
            "transcribe",
            "--image",
            str(sample_image_path),
            "--backend",
            "openai-compatible",
            "--model",
            "local-model",
            "--base-url",
            "http://127.0.0.1:8000/v1",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "ok"
    spec = captured["spec"]
    assert isinstance(spec, cli_module.OCRBackendSpec)
    assert spec.transport is not None
    assert spec.transport.api_base == "http://127.0.0.1:8000/v1"
    assert spec.transport.api_key is None


@pytest.mark.parametrize("backend", ["unsupported", "vllm"])
def test_transcribe_cli_rejects_unsupported_backend(
    sample_image_path: Path,
    backend: str,
    cli_runner,
) -> None:
    result = cli_runner.invoke(
        app,
        [
            "transcribe",
            "--image",
            str(sample_image_path),
            "--backend",
            backend,
            "--model",
            "example/model",
        ],
    )

    assert result.exit_code != 0
    assert f"Unsupported backend: {backend}" in result.output


def test_transcribe_cli_echoes_text_without_output(
    monkeypatch: pytest.MonkeyPatch,
    sample_image_path: Path,
    cli_runner,
) -> None:
    class _FakeBackend:
        async def ocr(self, page: DocumentPage) -> OCRResult:
            return OCRResult(
                text=f"plain:{page.image.width}x{page.image.height}",
                provider_name="fake",
                model_name="fake-model",
            )

    monkeypatch.setattr("churro_ocr.cli._build_ocr_backend", lambda **_: _FakeBackend())

    result = cli_runner.invoke(
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
    minimal_pdf_path: Path,
    cli_runner,
) -> None:
    output_dir = tmp_path / "pages"
    args = [
        str(sample_image_path)
        if token == "scan.png"
        else str(minimal_pdf_path)
        if token == "document.pdf"
        else token
        for token in command_args
    ]
    args = [str(output_dir) if token == "unused" else token for token in args]

    result = cli_runner.invoke(app, args)

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
    cli_runner,
) -> None:
    output_dir = tmp_path / "pages"
    result = cli_runner.invoke(
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
    minimal_pdf_path: Path,
    cli_runner,
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
    result = cli_runner.invoke(
        app,
        [
            "extract-pages",
            "--pdf",
            str(minimal_pdf_path),
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
        "path": minimal_pdf_path,
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

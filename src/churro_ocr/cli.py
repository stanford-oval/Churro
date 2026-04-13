"""Minimal public CLI for OCR and page detection."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer evaluates these annotations at runtime.

import typer

from churro_ocr._internal.install import (
    INSTALL_TARGETS,
    install_runtime_dependencies,
)
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRBackend, OCRClient
from churro_ocr.page_detection import (
    DocumentPage,
    DocumentPageDetector,
    PageDetectionBackendLike,
    PageDetectionRequest,
)
from churro_ocr.providers import (
    AzureDocumentIntelligenceOptions,
    AzurePageDetector,
    HuggingFaceOptions,
    LiteLLMTransportConfig,
    LLMPageDetector,
    MistralOptions,
    OCRBackendSpec,
    OpenAICompatibleOptions,
    build_ocr_backend,
)
from churro_ocr.providers.specs import MISTRAL_OCR_MODEL_IDS, validate_mistral_ocr_model

app = typer.Typer(help="churro-ocr library-first CLI")

_INSTALL_TARGET_METAVAR = "{" + "|".join(INSTALL_TARGETS) + "}"
_MISTRAL_MODEL_OPTION_ERROR = "--model is required for backend=mistral and must be one of: " + ", ".join(
    MISTRAL_OCR_MODEL_IDS
)


def _bad_parameter(message: str) -> typer.BadParameter:
    return typer.BadParameter(message)


def _build_ocr_backend(
    *,
    backend: str,
    model: str | None,
    endpoint: str | None,
    api_key: str | None,
    base_url: str | None,
    api_version: str | None,
) -> OCRBackend:
    if backend == "litellm":
        if not model:
            message = "--model is required for backend=litellm"
            raise _bad_parameter(message)
        return build_ocr_backend(
            OCRBackendSpec(
                provider="litellm",
                model=model,
                transport=LiteLLMTransportConfig(
                    api_base=base_url,
                    api_key=api_key,
                    api_version=api_version,
                ),
            )
        )
    if backend == "openai-compatible":
        if not model or not base_url:
            message = "--model and --base-url are required for backend=openai-compatible"
            raise _bad_parameter(message)
        return build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model=model,
                transport=LiteLLMTransportConfig(
                    api_base=base_url,
                    api_key=api_key,
                    api_version=api_version,
                ),
                options=OpenAICompatibleOptions(),
            )
        )
    if backend == "azure":
        if not endpoint or not api_key:
            message = "--endpoint and --api-key are required for backend=azure"
            raise _bad_parameter(message)
        return build_ocr_backend(
            OCRBackendSpec(
                provider="azure",
                model=model,
                options=AzureDocumentIntelligenceOptions(
                    endpoint=endpoint,
                    api_key=api_key,
                ),
            )
        )
    if backend == "mistral":
        if not api_key:
            message = "--api-key is required for backend=mistral"
            raise _bad_parameter(message)
        try:
            mistral_model = validate_mistral_ocr_model(model)
        except ConfigurationError as exc:
            raise typer.BadParameter(_MISTRAL_MODEL_OPTION_ERROR) from exc
        return build_ocr_backend(
            OCRBackendSpec(
                provider="mistral",
                model=mistral_model,
                options=MistralOptions(api_key=api_key),
            )
        )
    if backend == "hf":
        if not model:
            message = "--model is required for backend=hf"
            raise _bad_parameter(message)
        return build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=model,
                options=HuggingFaceOptions(model_kwargs={"device_map": "auto", "torch_dtype": "auto"}),
            )
        )
    message = f"Unsupported backend: {backend}"
    raise _bad_parameter(message)


def _build_page_detector(
    *,
    page_detector: str,
    model: str | None,
    endpoint: str | None,
    api_key: str | None,
    base_url: str | None,
    api_version: str | None,
) -> PageDetectionBackendLike | None:
    transport = None
    if base_url or api_key or api_version:
        transport = LiteLLMTransportConfig(
            api_base=base_url,
            api_key=api_key,
            api_version=api_version,
        )
    detector_backend = None
    if page_detector == "llm":
        if not model:
            message = "--model is required when --page-detector=llm"
            raise _bad_parameter(message)
        detector_backend = LLMPageDetector(
            model=model,
            transport=transport,
        )
    elif page_detector == "azure":
        if not endpoint or not api_key:
            message = "--endpoint and --api-key are required when --page-detector=azure"
            raise _bad_parameter(message)
        detector_backend = AzurePageDetector(endpoint=endpoint, api_key=api_key)
    return detector_backend


@app.command("transcribe")
def transcribe_command(
    image: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    backend: str = typer.Option("litellm"),
    model: str | None = typer.Option(None),
    endpoint: str | None = typer.Option(None),
    api_key: str | None = typer.Option(None),
    base_url: str | None = typer.Option(None),
    api_version: str | None = typer.Option(None),
    output: Path | None = typer.Option(None),
) -> None:
    """Transcribe text from a single image."""
    ocr_backend = _build_ocr_backend(
        backend=backend,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
    )
    result = OCRClient(ocr_backend).ocr(DocumentPage.from_image_path(image))
    if output:
        output.write_text(result.text or "")
        typer.echo(str(output))
        return
    typer.echo(result.text or "")


@app.command("extract-pages")
def extract_pages_command(
    image: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input image to split into page crops.",
    ),
    pdf: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input PDF to rasterize and split into page crops.",
    ),
    output_dir: Path = typer.Option(
        ...,
        file_okay=False,
        writable=True,
        help=(
            "Directory where extracted pages are written as sequential PNG files such as "
            "`page_0000.png`, `page_0001.png`, and so on."
        ),
    ),
    page_detector: str = typer.Option("none"),
    model: str | None = typer.Option(None),
    endpoint: str | None = typer.Option(None),
    api_key: str | None = typer.Option(None),
    base_url: str | None = typer.Option(None),
    api_version: str | None = typer.Option(None),
    dpi: int = typer.Option(300),
    trim_margin: int = typer.Option(30),
) -> None:
    """Extract page crops as PNG files and print each written path."""
    if (image is None) == (pdf is None):
        message = "Provide exactly one of --image or --pdf."
        raise _bad_parameter(message)
    detector_backend = _build_page_detector(
        page_detector=page_detector,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
    )
    page_detector_client = DocumentPageDetector(backend=detector_backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    if image is not None:
        result = page_detector_client.detect_image_sync(
            PageDetectionRequest(image_path=image, trim_margin=trim_margin)
        )
    else:
        assert pdf is not None
        result = page_detector_client.detect_pdf_sync(pdf, dpi=dpi, trim_margin=trim_margin)

    for page in result.pages:
        output_path = output_dir / f"page_{page.page_index:04d}.png"
        page.image.save(output_path)
        typer.echo(str(output_path))


@app.command("install")
def install_command(
    target: str = typer.Argument(
        ...,
        metavar=_INSTALL_TARGET_METAVAR,
        help="Runtime target to install into the active environment with uv.",
    ),
    torch_backend: str = typer.Option(
        "auto",
        help="PyTorch backend passed through to uv when a local runtime needs torch.",
    ),
) -> None:
    """Install optional runtime dependencies with uv."""
    try:
        result = install_runtime_dependencies(
            target=target,
            torch_backend=torch_backend,
        )
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Installed runtime target: {result.target}")
    for note in result.notes:
        typer.echo(note)


def main() -> None:
    """Console-script entrypoint."""
    app()

"""PDF rasterization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from churro_ocr._internal.install import install_command_hint
from churro_ocr.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image


def _configuration_error(message: str) -> ConfigurationError:
    return ConfigurationError(message)


def rasterize_pdf(path: str | Path, *, dpi: int = 300) -> list[Image.Image]:
    """Rasterize a PDF into PIL images."""
    from pathlib import Path

    try:
        import pypdfium2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        message = f"PDF support requires the `pdf` runtime. {install_command_hint('pdf')}"
        raise _configuration_error(message) from exc

    resolved = Path(path)
    if not resolved.exists():
        message = f"PDF path does not exist: {resolved}"
        raise _configuration_error(message)

    images: list[Image.Image] = []
    scale = max(dpi, 1) / 72.0
    document = pypdfium2.PdfDocument(str(resolved))
    try:
        for page in document:
            try:
                bitmap = page.render(scale=scale)
                try:
                    rendered = bitmap.to_pil()
                    try:
                        images.append(rendered.convert("RGB"))
                    finally:
                        rendered.close()
                finally:
                    bitmap.close()
            finally:
                page.close()
    finally:
        document.close()
    return images

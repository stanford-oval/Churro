"""Image loading, encoding, and normalization helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from churro_ocr.errors import ConfigurationError

MAX_INLINE_IMAGE_DIM = 2_500


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk and normalize EXIF orientation."""
    resolved = Path(path)
    if not resolved.exists():
        raise ConfigurationError(f"Image path does not exist: {resolved}")
    with Image.open(resolved) as image:
        normalized = ImageOps.exif_transpose(image)
        assert normalized is not None
        return normalized.copy()


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB image copy when needed."""
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Resize an image to fit within the provided bounds."""
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image
    scale = min(max_width / width, max_height / height)
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )


def prepare_ocr_image(image: Image.Image) -> Image.Image:
    """Normalize and resize an image for OCR provider transport."""
    return ensure_rgb(resize_image_to_fit(image, MAX_INLINE_IMAGE_DIM, MAX_INLINE_IMAGE_DIM))


def image_to_base64(image: Image.Image, format_name: str | None = None) -> tuple[str, str]:
    """Encode an image for provider transport."""
    resolved_format = (format_name or image.format or "PNG").upper()
    if resolved_format not in {"PNG", "JPEG", "WEBP"}:
        resolved_format = "PNG"
    mime_type = f"image/{resolved_format.lower()}"
    buffer = BytesIO()
    save_kwargs = {"quality": 95, "optimize": True} if resolved_format == "JPEG" else {}
    image.save(buffer, format=resolved_format, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return mime_type, encoded

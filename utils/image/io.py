"""Async image loading helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import aiofiles
from PIL import Image


async def load_image_async(image_path: str | Path) -> Image.Image:
    """Load an image from disk without blocking the event loop."""
    path = Path(image_path)
    async with aiofiles.open(path, "rb") as file_obj:
        image_bytes = await file_obj.read()
    image = Image.open(BytesIO(image_bytes))
    image.load()
    return image


__all__ = ["load_image_async"]

"""Base OCR class and interfaces for all OCR systems."""

from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import aiofiles
from PIL import Image

from utils.utils import run_async_in_parallel


class BaseOCR(ABC):
    """Base class for all OCR systems."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the OCR system with configuration parameters."""
        self.config = kwargs

    @abstractmethod
    async def process_image(self, image: Image.Image) -> str:
        """Process a single image and return the OCR result."""
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """Return the name of the OCR system."""
        pass

    async def process_examples(self, dataset: list[dict], max_concurrency: int) -> list[str]:
        return await run_async_in_parallel(
            self.process_image,
            [e["image"] for e in dataset],
            max_concurrency=max_concurrency,
            desc=self.get_system_name(),
        )

    async def process_image_from_file(self, image_path: str) -> str:
        """Process a single image from a file path and return the OCR result."""
        image = await self.open_image(image_path)
        return await self.process_image(image)

    async def open_image(self, image_path: str) -> Image.Image:
        """Open an image from a file path and return the image object."""
        async with aiofiles.open(image_path, "rb") as f:
            image_bytes = await f.read()
            image = Image.open(BytesIO(image_bytes))
            image.load()
        return image

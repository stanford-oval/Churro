"""Base OCR class and interfaces for all OCR systems."""

from abc import ABC, abstractmethod
from typing import cast

from PIL import Image

from churro.utils.concurrency import run_async_in_parallel
from churro.utils.image.io import load_image_async
from churro.utils.log_utils import logger


class BaseOCR(ABC):
    """Base class for all OCR systems."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the OCR system with configuration parameters."""
        self.config: dict[str, object] = kwargs

    @abstractmethod
    async def process_image(self, image: Image.Image) -> str:
        """Process a single image and return the OCR result."""
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """Return the name of the OCR system."""
        pass

    async def process_images(self, images: list[Image.Image], *, max_concurrency: int) -> list[str]:
        """Process a list of images and return the OCR results."""
        raw_results = await run_async_in_parallel(
            self.process_image,
            images,
            max_concurrency=max_concurrency,
            desc=self.get_system_name(),
        )
        processed: list[str] = []
        for index, result in enumerate(raw_results):
            if result is None:
                message = (
                    f"{self.get_system_name()} failed to process in-memory image at index {index}."
                )
                logger.error(message)
                result = ""
            processed.append(cast(str, result))
        return processed

    async def process_image_from_file(self, image_path: str) -> str:
        """Process a single image from a file path and return the OCR result."""
        image = await load_image_async(image_path)
        return await self.process_image(image)

    async def process_images_from_files(
        self, image_paths: list[str], max_concurrency: int
    ) -> list[str]:
        """Process a list of images from file paths and return the OCR results."""
        raw_results = await run_async_in_parallel(
            self.process_image_from_file,
            image_paths,
            max_concurrency=max_concurrency,
            desc=self.get_system_name(),
        )
        processed: list[str] = []
        for path, result in zip(image_paths, raw_results, strict=False):
            if result is None:
                message = f"{self.get_system_name()} failed to process file {path}."
                logger.error(message)
                result = ""
            processed.append(cast(str, result))
        return processed

"""Base OCR class and interfaces for all OCR systems."""

from abc import ABC, abstractmethod
from typing import Any


class BaseOCR(ABC):
    """Base class for all OCR systems."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the OCR system with configuration parameters."""
        self.config = kwargs

    @abstractmethod
    async def process(self, dataset: list[dict]) -> list[str]:
        """Process examples and return OCR results."""
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """Return the name of the OCR system."""
        pass

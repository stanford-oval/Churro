"""Public exception types for the churro-ocr package."""

from __future__ import annotations


class ChurroError(RuntimeError):
    """Base exception for package-level failures."""


class ConfigurationError(ChurroError):
    """Raised when a backend is missing required runtime configuration."""


class ProviderError(ChurroError):
    """Raised when an OCR or page detection provider returns an unusable response."""

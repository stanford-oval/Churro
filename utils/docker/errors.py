"""Custom exception types for Docker helpers."""

from __future__ import annotations


class DockerError(RuntimeError):
    """Raised when Docker-related operations fail.

    This includes SDK import issues, daemon connectivity, timeouts, or
    container crashes detected during readiness wait.
    """

    pass


__all__ = ["DockerError"]

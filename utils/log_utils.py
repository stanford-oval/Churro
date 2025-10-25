"""Logging utilities shared across the churro package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from rich.logging import RichHandler


_CONFIGURED: bool = False

DEFAULT_CONSOLE_LEVEL = "INFO"
DEFAULT_FILE_LEVEL = "DEBUG"
DEFAULT_FILE_PATH = "debug_logs.log"
DEFAULT_FILE_ROTATION = "5 MB"
DEFAULT_FILE_RETENTION = 2

_RICH_HANDLER_KWARGS: dict[str, Any] = {
    "markup": True,
    "show_time": False,
}


def _configure_logging(*, force: bool = False) -> None:
    """Configure the shared logger once per process."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    logger.remove()

    logger.add(
        RichHandler(**_RICH_HANDLER_KWARGS),  # type: ignore[arg-type]
        level=DEFAULT_CONSOLE_LEVEL,
        format="{message}",
    )

    if DEFAULT_FILE_PATH:
        resolved_file_path = Path(DEFAULT_FILE_PATH).expanduser().resolve()
        resolved_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(resolved_file_path),
            level=DEFAULT_FILE_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
            rotation=DEFAULT_FILE_ROTATION,
            retention=DEFAULT_FILE_RETENTION,
            enqueue=True,
        )

    _CONFIGURED = True


__all__ = ["logger"]

# Configure logging on import so callers only need to import `logger`.
_configure_logging()

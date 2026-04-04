"""Internal logging utilities for the standalone churro-ocr package."""

from __future__ import annotations

from typing import Any

from loguru import logger as _loguru_logger
from rich.logging import RichHandler


def _configure_default_logger() -> Any:
    _loguru_logger.remove()
    _loguru_logger.add(
        RichHandler(markup=True, show_time=False),
        level="WARNING",
        format="{message}",
    )
    return _loguru_logger


_default_logger = _configure_default_logger().bind(app="churro-ocr")


class _LoggerAdapter:
    """Compatibility wrapper for stdlib-style formatting on top of the default logger."""

    __slots__ = ("_logger",)

    def __init__(self, wrapped_logger: Any) -> None:
        self._logger = wrapped_logger

    def _format(self, message: str, *args: object) -> str:
        return message % args if args else message

    def debug(self, message: str, *args: object) -> None:
        self._logger.debug(self._format(message, *args))

    def info(self, message: str, *args: object) -> None:
        self._logger.info(self._format(message, *args))

    def success(self, message: str, *args: object) -> None:
        success = getattr(self._logger, "success", None)
        if success is not None:
            success(self._format(message, *args))
            return
        self._logger.info(self._format(message, *args))

    def warning(self, message: str, *args: object) -> None:
        self._logger.warning(self._format(message, *args))

    def error(self, message: str, *args: object) -> None:
        self._logger.error(self._format(message, *args))

    def critical(self, message: str, *args: object) -> None:
        self._logger.critical(self._format(message, *args))

    def exception(self, message: str, *args: object) -> None:
        self._logger.exception(self._format(message, *args))

    def log(self, level: str, message: str, *args: object) -> None:
        self._logger.log(level, self._format(message, *args))


logger = _LoggerAdapter(_default_logger)

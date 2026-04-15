"""Internal logging utilities for the standalone churro-ocr package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from loguru import logger as _loguru_logger
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Callable


class _RichLoggerLike(Protocol):
    def bind(self, **kwargs: object) -> _RichLoggerLike: ...

    def debug(self, message: str) -> None: ...

    def info(self, message: str) -> None: ...

    def success(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...

    def critical(self, message: str) -> None: ...

    def exception(self, message: str) -> None: ...

    def log(self, level: str, message: str) -> None: ...


def _configure_default_logger() -> _RichLoggerLike:
    _loguru_logger.remove()
    _loguru_logger.add(
        RichHandler(markup=True, show_time=False),
        level="WARNING",
        format="{message}",
    )
    return cast("_RichLoggerLike", _loguru_logger)


_default_logger = _configure_default_logger().bind(app="churro-ocr")


class _LoggerAdapter:
    """Compatibility wrapper for stdlib-style formatting on top of the default logger."""

    __slots__ = ("_logger",)

    def __init__(self, wrapped_logger: object) -> None:
        self._logger = wrapped_logger

    def _format(self, message: str, *args: object) -> str:
        return message % args if args else message

    def _message_logger(self, method_name: str) -> Callable[[str], object]:
        logger_method = getattr(self._logger, method_name, None)
        if callable(logger_method):
            return cast("Callable[[str], object]", logger_method)
        message = f"Wrapped logger does not define `{method_name}(...)`."
        raise AttributeError(message)

    def _level_logger(self) -> Callable[[str, str], object]:
        logger_method = getattr(self._logger, "log", None)
        if callable(logger_method):
            return cast("Callable[[str, str], object]", logger_method)
        message = "Wrapped logger does not define `log(level, message)`."
        raise AttributeError(message)

    def debug(self, message: str, *args: object) -> None:
        self._message_logger("debug")(self._format(message, *args))

    def info(self, message: str, *args: object) -> None:
        self._message_logger("info")(self._format(message, *args))

    def success(self, message: str, *args: object) -> None:
        success = getattr(self._logger, "success", None)
        if callable(success):
            cast("Callable[[str], object]", success)(self._format(message, *args))
            return
        self._message_logger("info")(self._format(message, *args))

    def warning(self, message: str, *args: object) -> None:
        self._message_logger("warning")(self._format(message, *args))

    def error(self, message: str, *args: object) -> None:
        self._message_logger("error")(self._format(message, *args))

    def critical(self, message: str, *args: object) -> None:
        self._message_logger("critical")(self._format(message, *args))

    def exception(self, message: str, *args: object) -> None:
        self._message_logger("exception")(self._format(message, *args))

    def log(self, level: str, message: str, *args: object) -> None:
        self._level_logger()(level, self._format(message, *args))


logger = _LoggerAdapter(_default_logger)

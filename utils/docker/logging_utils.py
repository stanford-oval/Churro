"""Internal logging helpers for Docker utilities.

Separated to keep concerns modular. Not part of the public API.
"""

from __future__ import annotations

import re
from re import Pattern

from churro.utils.log_utils import logger


try:  # Rich may already be installed (used by log_utils)
    from rich.markup import escape as _rich_escape  # type: ignore
except Exception:  # pragma: no cover - fallback if Rich missing in some envs

    def _rich_escape(s: str) -> str:  # type: ignore
        return s.replace("[", "[[")  # minimal safe fallback


ANSI_ESCAPE_RE: Pattern[str] = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def format_prefix(prefix: str | None) -> str:
    """Return a logging prefix safe for Rich markup.

    Args:
        prefix: Optional raw prefix.

    Returns:
        Escaped prefix ending with a space if non-empty.
    """
    if not prefix:
        return ""
    p = _rich_escape(prefix.rstrip())
    return p + (" " if not p.endswith(" ") else "")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return ANSI_ESCAPE_RE.sub("", text)


def log_multiline(text: str, log_prefix: str | None, level: str = "info") -> None:
    """Log multiline output line-by-line with optional prefix."""
    if not text:
        return
    pf = format_prefix(log_prefix)
    log_fn = getattr(logger, level, logger.info)
    for raw_line in text.splitlines():
        sanitized = strip_ansi(raw_line).replace("\r", "")
        log_fn(f"{pf}{sanitized}")


__all__ = ["format_prefix", "strip_ansi", "log_multiline", "ANSI_ESCAPE_RE"]

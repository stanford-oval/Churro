"""Normalization helpers used by repo-only evaluation tooling."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from churro_ocr.prompts import strip_rich_ocr_markup_to_plain_text


def _missing_dependency_error(message: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(message)


normalize_hamza: Callable[..., str] | None
strip_harakat: Callable[..., str] | None
strip_lastharaka: Callable[..., str] | None
strip_tashkeel: Callable[..., str] | None
strip_tatweel: Callable[..., str] | None

try:  # pragma: no cover - optional dependency
    from pyarabic.araby import (
        normalize_hamza as _normalize_hamza,
    )
    from pyarabic.araby import (
        strip_harakat as _strip_harakat,
    )
    from pyarabic.araby import (
        strip_lastharaka as _strip_lastharaka,
    )
    from pyarabic.araby import (
        strip_tashkeel as _strip_tashkeel,
    )
    from pyarabic.araby import (
        strip_tatweel as _strip_tatweel,
    )

    normalize_hamza = cast("Callable[..., str]", _normalize_hamza)
    strip_harakat = cast("Callable[..., str]", _strip_harakat)
    strip_lastharaka = cast("Callable[..., str]", _strip_lastharaka)
    strip_tashkeel = cast("Callable[..., str]", _strip_tashkeel)
    strip_tatweel = cast("Callable[..., str]", _strip_tatweel)
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    normalize_hamza = None
    strip_harakat = None
    strip_lastharaka = None
    strip_tashkeel = None
    strip_tatweel = None

SUBSTITUTIONS = {
    "\ueada": "st",
    "\ueec5": "ct",
    "\ueba6": "ss",
    "\ueba2": "si",
    "\ueba7": "ssi",
    "\ueba3": "sl",
    "’": "'",
    "¬": "-",
}
SUBSTITUTION_PATTERN = re.compile("|".join(map(re.escape, SUBSTITUTIONS.keys())))


def normalize_characters(text: str, *, keep_long_s: bool = True) -> str:
    """Replace document-specific glyphs with normalized equivalents."""
    text = re.sub(r"(?<=\d)(?=[↉½⅓¼⅕⅙⅐⅛⅑⅒⅔⅖¾⅗⅜⅘⅚⅞])", " ", text)

    placeholder = "\ue000"
    if keep_long_s:
        text = text.replace("ſ", placeholder)

    text = unicodedata.normalize("NFKC", text)

    if keep_long_s:
        text = text.replace(placeholder, "ſ")

    return re.sub(
        r"(^|\s)~(?=\w)",
        r"\1",
        SUBSTITUTION_PATTERN.sub(lambda match: SUBSTITUTIONS[match.group(0)], text),
    )


def normalize_text_for_evaluation(text: str, *, normalize_arabic: bool = False) -> str:
    """Normalize raw OCR text before metric computation."""
    text = strip_rich_ocr_markup_to_plain_text(text)

    if normalize_arabic:
        if (
            strip_tashkeel is None
            or strip_harakat is None
            or strip_lastharaka is None
            or strip_tatweel is None
            or normalize_hamza is None
        ):
            message = "Arabic normalization requires the optional dependency 'pyarabic'."
            raise _missing_dependency_error(message)
        strip_tashkeel_fn = cast("Callable[[str], str]", strip_tashkeel)
        strip_harakat_fn = cast("Callable[[str], str]", strip_harakat)
        strip_lastharaka_fn = cast("Callable[[str], str]", strip_lastharaka)
        strip_tatweel_fn = cast("Callable[[str], str]", strip_tatweel)
        normalize_hamza_fn = cast("Callable[[str], str]", normalize_hamza)

        text = strip_tashkeel_fn(text)
        text = strip_harakat_fn(text)
        text = strip_lastharaka_fn(text)
        text = strip_tatweel_fn(text)
        text = normalize_hamza_fn(text)

    text = text.lower()
    text = re.sub(r"[*_`~#]", "", text)
    text = re.sub(r"[–—−‑‒―‐]", "-", text)
    text = re.sub(r"\[figure\s+\d+\]", "", text)
    text = re.sub(r"-{3,}", "", text)
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = text.strip("-")
    return re.sub(r"\s+", " ", normalize_characters(text, keep_long_s=False)).strip()

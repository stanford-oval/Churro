"""Helpers for one-time OCR prompt payload logging."""

from __future__ import annotations

import json
from base64 import b64encode
from typing import TYPE_CHECKING

from PIL import Image

from churro_ocr._internal.image import image_to_base64
from churro_ocr._internal.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from threading import Lock

_IMAGE_PREVIEW_CHARS = 96


def _truncate_text(value: str, *, limit: int = _IMAGE_PREVIEW_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _encode_image_preview(image: Image.Image, *, format_name: str | None = None) -> dict[str, object]:
    mime_type, encoded = image_to_base64(image, format_name)
    return {
        "image_size": image.size,
        "image_mode": image.mode,
        "image_preview": _truncate_text(f"data:{mime_type};base64,{encoded}"),
    }


def _encode_bytes_preview(payload: bytes, *, mime_type: str) -> str:
    encoded = b64encode(payload).decode("utf-8")
    return _truncate_text(f"data:{mime_type};base64,{encoded}")


def _sanitize_prompt_payload(payload: object) -> object:
    if isinstance(payload, Image.Image):
        return {
            "type": "image",
            **_encode_image_preview(payload),
        }
    if isinstance(payload, str) and payload.startswith("data:") and ";base64," in payload:
        return _truncate_text(payload)
    if isinstance(payload, bytes):
        return {
            "type": "bytes",
            "byte_length": len(payload),
            "data_preview": _encode_bytes_preview(payload, mime_type="application/octet-stream"),
        }
    if isinstance(payload, dict):
        return {str(key): _sanitize_prompt_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_prompt_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [_sanitize_prompt_payload(item) for item in payload]
    return payload


def log_prompt_payload_once(
    *,
    payload: object,
    provider_name: str,
    has_logged: Callable[[], bool],
    lock: Lock,
    set_logged: Callable[[], None],
) -> None:
    """Log one OCR prompt payload preview for a backend instance."""
    if has_logged():
        return
    with lock:
        if has_logged():
            return
        sanitized_payload = _sanitize_prompt_payload(payload)
        logger.debug(
            "First OCR prompt payload for %s:\n%s",
            provider_name,
            json.dumps(sanitized_payload, ensure_ascii=False, indent=2, default=str),
        )
        set_logged()


__all__ = ["log_prompt_payload_once"]

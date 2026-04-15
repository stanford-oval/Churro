"""Shared structural type aliases for CHURRO public interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from PIL import Image

# Provider metadata is forwarded through public results with backend-specific
# nested payloads, so keep this alias intentionally loose and centralize it.
type MetadataDict = dict[str, Any]
type BoundingBox = tuple[float, float, float, float]
type Polygon = tuple[tuple[float, float], ...]
type OCRRole = Literal["assistant", "system", "user"]
type OCRConversationContentItem = dict[str, Any]
type OCRConversationMessage = dict[str, Any]


class OCRImageContentItem(TypedDict):
    """Built-in image content item used by chat-template OCR prompts."""

    type: Literal["image"]
    image: Image.Image


class OCRTextContentItem(TypedDict):
    """Built-in text content item used by chat-template OCR prompts."""

    type: Literal["text"]
    text: str


type OCRBuiltInConversationContentItem = OCRImageContentItem | OCRTextContentItem
type OCRConversation = list[OCRConversationMessage]

__all__ = [
    "BoundingBox",
    "MetadataDict",
    "OCRBuiltInConversationContentItem",
    "OCRConversation",
    "OCRConversationContentItem",
    "OCRConversationMessage",
    "OCRImageContentItem",
    "OCRRole",
    "OCRTextContentItem",
    "Polygon",
]

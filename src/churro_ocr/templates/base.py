"""Provider-neutral template protocols for OCR backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from churro_ocr.page_detection import DocumentPage

OCRConversation = list[dict[str, Any]]


@runtime_checkable
class OCRPromptTemplate(Protocol):
    """Protocol for OCR templates that build model conversations."""

    def build_conversation(self, page: DocumentPage) -> OCRConversation:
        """Build a model conversation for one page.

        :param page: Page to convert into a model-specific prompt payload.
        :returns: Structured conversation ready for backend-specific rendering.
        """
        ...


OCRPromptTemplateCallable = Callable[[DocumentPage], OCRConversation]
OCRPromptTemplateLike = OCRPromptTemplate | OCRPromptTemplateCallable


def build_ocr_conversation(template: OCRPromptTemplateLike, page: DocumentPage) -> OCRConversation:
    """Build an OCR conversation from a template or template callable.

    :param template: Prompt template object or callable.
    :param page: Page to convert into a conversation.
    :returns: Structured OCR conversation for ``page``.
    """
    if callable(template) and not isinstance(template, OCRPromptTemplate):
        return template(page)
    return template.build_conversation(page)


# Internal aliases kept to make the refactor incremental while the
# provider implementations migrate off the old HF-prefixed names.
HFConversation = OCRConversation
HFOCRTemplate = OCRPromptTemplate
HFOCRTemplateCallable = OCRPromptTemplateCallable
HFOCRTemplateLike = OCRPromptTemplateLike


__all__ = [
    "build_ocr_conversation",
    "OCRConversation",
    "OCRPromptTemplate",
    "OCRPromptTemplateCallable",
    "OCRPromptTemplateLike",
]

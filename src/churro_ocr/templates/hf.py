"""Built-in OCR templates."""

from __future__ import annotations

from dataclasses import dataclass

from churro_ocr.page_detection import DocumentPage
from churro_ocr.templates.base import OCRConversation


@dataclass(slots=True, frozen=True)
class HFChatTemplate:
    """Template for processor/tokenizer chat-template OCR models."""

    system_message: str | None = None
    user_prompt: str | None = None
    include_image: bool = True

    def build_conversation(self, page: DocumentPage) -> OCRConversation:
        """Build a structured multimodal conversation for one OCR page."""
        conversation: OCRConversation = []
        if self.system_message:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_message}],
                }
            )

        user_content: list[dict[str, object]] = []
        if self.include_image:
            user_content.append({"type": "image", "image": page.image.copy()})
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})

        conversation.append({"role": "user", "content": user_content})
        return conversation


__all__ = ["HFChatTemplate"]

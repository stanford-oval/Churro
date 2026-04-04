"""Default OCR prompts."""

from __future__ import annotations

import re

DEFAULT_OCR_OUTPUT_TAG = "output"

DEFAULT_OCR_SYSTEM_PROMPT = (
    "You are an expert in diplomatic transcription of historical documents from various "
    "languages. Your task is to extract the full text from a given page. Only output the "
    f"transcribed text between <{DEFAULT_OCR_OUTPUT_TAG}> and </{DEFAULT_OCR_OUTPUT_TAG}> tags."
)

DEFAULT_OCR_USER_PROMPT = (
    "Follow these instructions:\n\n"
    "1. You will be provided with a scanned document page.\n\n"
    "2. Perform transcription on the entirety of the page, converting all visible text into "
    "the following format. Include handwritten and print text, if any. Include tables, "
    "captions, headers, main text and all other visible text.\n\n"
    "3. If you encounter any non-text elements, simply skip them without attempting to "
    "describe them.\n\n"
    "4. Do not modernize or standardize the text. For example, if the transcription is using "
    '"ſ" instead of "s" or "а" instead of "a", keep it that way.\n\n'
    "5. When you come across text in languages other than English, transcribe it as "
    "accurately as possible without translation.\n\n"
    "6. Output the OCR result in the following format:\n\n"
    f"<{DEFAULT_OCR_OUTPUT_TAG}>\n"
    "extracted text here\n"
    f"</{DEFAULT_OCR_OUTPUT_TAG}>\n\n"
    "Remember, your goal is to accurately transcribe the text from the scanned page as much "
    "as possible. Process the entire page, even if it contains a large amount of text, and "
    "provide clear, well-formatted output. Pay attention to the appropriate reading order "
    "and layout of the text."
)

DEFAULT_MARKDOWN_OCR_USER_PROMPT = (
    "Transcribe the full page in reading order as Markdown. Preserve headings, lists, "
    "tables, and line breaks when they are visible."
)


def strip_ocr_output_tag(text: str, *, output_tag: str = DEFAULT_OCR_OUTPUT_TAG) -> str:
    """Remove outer OCR output tags and any stray tag tokens when present.

    :param text: Raw OCR response text.
    :param output_tag: Expected wrapper tag name.
    :returns: OCR text with the outer wrapper removed when present.
    """
    outer_wrapper_pattern = re.compile(
        rf"^\s*<{re.escape(output_tag)}>\s*(.*?)\s*</{re.escape(output_tag)}>\s*$",
        flags=re.DOTALL,
    )
    match = outer_wrapper_pattern.match(text)
    if match is not None:
        return match.group(1).strip()

    stray_tag_pattern = re.compile(rf"</?{re.escape(output_tag)}\b[^>]*>", flags=re.IGNORECASE)
    return stray_tag_pattern.sub("", text).strip()

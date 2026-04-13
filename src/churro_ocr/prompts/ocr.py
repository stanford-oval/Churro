"""Default OCR prompts."""

from __future__ import annotations

import html
import re
from typing import Any

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

CHANDRA_OCR_LAYOUT_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks. Each layout block should be a div "
    "with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 "
    "format. Bboxes are normalized 0-1000. The data-label attribute is the label for the block.\n\n"
    "Use the following labels:\n"
    "- Caption\n"
    "- Footnote\n"
    "- Equation-Block\n"
    "- List-Group\n"
    "- Page-Header\n"
    "- Page-Footer\n"
    "- Image\n"
    "- Section-Header\n"
    "- Table\n"
    "- Text\n"
    "- Complex-Block\n"
    "- Code-Block\n"
    "- Form\n"
    "- Table-Of-Contents\n"
    "- Figure\n"
    "- Chemical-Block\n"
    "- Diagram\n"
    "- Bibliography\n"
    "- Blank-Page\n\n"
    "Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', "
    "'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', "
    "'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code', "
    "'chem'], and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', "
    "'type', 'border', 'value', 'style', 'href', 'alt', 'align', 'data-bbox', 'data-label'].\n\n"
    "Guidelines:\n"
    "* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered "
    "in KaTeX-compatible LaTeX. Use display for block math.\n"
    "* Tables: Use colspan and rowspan attributes to match table structure.\n"
    "* Formatting: Maintain consistent formatting with the image, including spacing, indentation, "
    "subscripts/superscripts, and special characters.\n"
    "* Images: Include a description of any images in the alt attribute of an <img> tag. Do not "
    "fill out the src property. Describe in detail inside the div tag. Also convert charts to high "
    "fidelity data, and convert diagrams to mermaid.\n"
    "* Forms: Mark checkboxes and radio buttons properly.\n"
    "* Text: join lines together properly into paragraphs using <p>...</p> tags. Use <br> tags for "
    "line breaks within paragraphs, but only when absolutely necessary to maintain meaning.\n"
    "* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.\n"
    "* Lists: Preserve indents and proper list markers.\n"
    "* Use the simplest possible HTML structure that accurately represents the content of the block.\n"
    "* Make sure the text is accurate and easy for a human to read and interpret. Reading order "
    "should be correct and natural."
)

OLMOCR_V4_YAML_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
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


def _extract_yaml_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Return YAML front matter fields and the remaining markdown body."""
    stripped = text.strip()
    if not stripped.startswith("---\n"):
        return {}, stripped

    end_index = stripped.find("\n---", 4)
    if end_index == -1:
        return {}, stripped

    front_matter_block = stripped[4:end_index]
    body = stripped[end_index + 4 :].strip()
    front_matter: dict[str, object] = {}
    for line in front_matter_block.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        lower = value.lower()
        if lower == "null":
            parsed: object = None
        elif lower == "true":
            parsed = True
        elif lower == "false":
            parsed = False
        elif re.fullmatch(r"-?\d+", value):
            parsed = int(value)
        else:
            parsed = value
        front_matter[key] = parsed
    return front_matter, body


def strip_rich_ocr_markup_to_plain_text(text: str) -> str:
    """Best-effort plain-text conversion for OCR markdown/HTML output."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"!\[[^\]]*]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", cleaned)

    html_replacements = (
        (r"(?is)<\s*input\b[^>]*\bchecked(?:=(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?[^>]*>", "[x]"),
        (r"(?is)<\s*input\b[^>]*>", "[ ]"),
        (r"(?i)<\s*br\s*/?\s*>", "\n"),
        (r"(?i)<\s*hr\s*/?\s*>", "\n"),
        (r"(?i)</\s*(?:p|div|h[1-6]|ul|ol|table|tr|li|pre|caption)\s*>", "\n"),
        (r"(?i)</\s*(?:td|th)\s*>", " | "),
        (r"(?i)<\s*li\b[^>]*>", ""),
        (
            r"(?i)</?\s*(?:table|thead|tbody|tfoot|tr|td|th|p|div|span|h[1-6]|ul|ol|"
            r"strong|em|b|i|u|sup|sub|code|pre|a|math|chem|caption|small|big)\b[^>]*>",
            "",
        ),
        (r"(?i)<\s*img\b[^>]*>", ""),
    )
    for pattern, replacement in html_replacements:
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"(?is)</?[a-z][^>]*>", "", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-+*]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*>\s?", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    normalized_lines: list[str] = []
    saw_content = False
    for raw_line in cleaned.splitlines():
        line = re.sub(r"\s*\|\s*$", "", raw_line.strip())
        line = re.sub(r"^\|\s*", "", line)
        line = re.sub(r"\s*\|\s*", " | ", line)
        if line:
            normalized_lines.append(line)
            saw_content = True
            continue
        if saw_content and normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")
    return "\n".join(normalized_lines).strip()


def parse_olmocr_response(text: str) -> tuple[str, dict[str, Any]]:
    """Extract plain text and metadata from an olmOCR YAML-front-matter response."""
    front_matter, markdown_body = _extract_yaml_front_matter(text)
    return strip_rich_ocr_markup_to_plain_text(markdown_body), {
        "front_matter": front_matter,
        "raw_markdown": markdown_body,
    }


def parse_chandra_response(text: str) -> tuple[str, dict[str, Any]]:
    """Extract plain text and metadata from a Chandra HTML-layout response."""
    raw_html = text.strip()
    return strip_rich_ocr_markup_to_plain_text(raw_html), {
        "raw_html": raw_html,
    }

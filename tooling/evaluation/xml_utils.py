"""XML extraction helpers for repo-only evaluation tooling."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from churro_ocr._internal.logging import logger


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _remove_tag(xml_content: str, tag_name: str) -> str:
    if f"<{tag_name}" not in xml_content:
        return xml_content
    return re.sub(
        rf"<{tag_name}\b[^>]*/>",
        "",
        re.sub(rf"<{tag_name}\b[^>]*>.*?</{tag_name}>", "", xml_content, flags=re.DOTALL),
    )


def extract_actual_text_from_xml(xml_content: str) -> str:
    """Extract text from HistoricalDocument XML, or return the raw input when not XML."""
    if "HistoricalDocument" not in xml_content:
        return xml_content

    for tag_name in ("Description", "Deletion", "Illegible", "Gap"):
        xml_content = _remove_tag(xml_content, tag_name)

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        logger.warning("Failed to parse XML content during evaluation: %s", exc)
        return ""

    page_texts: list[str] = []
    for page in root.iter():
        if _local_name(page.tag) != "Page":
            continue
        section_texts: list[str] = []
        for child in page.iter():
            if _local_name(child.tag) not in {"Header", "Body", "Footer"}:
                continue
            lines = [line.strip() for line in child.itertext() if line.strip()]
            if lines:
                section_texts.append("\n".join(lines))
        if section_texts:
            page_texts.append("\n".join(section_texts))

    return "\n\n".join(page_texts).strip()

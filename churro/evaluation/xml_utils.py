from pathlib import Path
import re
import xml.etree.ElementTree as ET

from lxml import etree  # type: ignore
import xmlschema

from churro.utils.log_utils import logger


historical_doc_schema = None
allowed_xml_pattern = None
SCHEMA_PATH = Path(__file__).resolve().parent / "historical_doc.xsd"


def _escape_chunk(text: str) -> str:
    # only escape &, <, >
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_xml(s: str) -> str:
    """Escape ``<``, ``>``, ``&`` except inside allowed XML tags/structures."""
    global allowed_xml_pattern
    # Compile the pattern once and stash it on the function
    if allowed_xml_pattern is None:
        tags = _get_list_of_valid_xml_tags()
        # build a regex that matches:
        #   <?xml ...?>, <tag ...>, </tag>, <tag/>, etc.
        tag_alts = "|".join(re.escape(t) for t in tags)
        allowed_xml_pattern = re.compile(
            rf"(<\?xml.*?\?>|</?(?:{tag_alts})(?:\b[^>]*?)?/?>)", re.DOTALL
        )

    pat = allowed_xml_pattern
    parts = []
    last = 0

    # Iterate through all matches of allowed XML fragments
    for m in pat.finditer(s):
        if m.start() > last:
            parts.append(_escape_chunk(s[last : m.start()]))
        parts.append(m.group(0))
        last = m.end()

    # escape any remaining tail
    if last < len(s):
        parts.append(_escape_chunk(s[last:]))

    return "".join(parts)


def _get_list_of_valid_xml_tags() -> list[str]:
    """Return valid XML tags for the historical document schema."""
    global historical_doc_schema
    if historical_doc_schema is None:
        historical_doc_schema = xmlschema.XMLSchema(str(SCHEMA_PATH))
    return list(historical_doc_schema.elements.keys()) + [
        "PhysicalDescription",
        "Language",
        "Script",
        "PhysicalDescription",
        "Description",
        "WritingDirection",
        "TranscriptionNote",
        "Footer",
        "Header",
    ]


def extract_actual_text_from_xml(xml_content: str) -> str:
    """Extract concatenated text from ``<Header>``, ``<Body>``, ``<Footer>`` of each page."""
    if "HistoricalDocument" not in xml_content:
        return xml_content

    xml_content = _escape_xml(xml_content)
    try:
        xml_content = _remove_tag(
            xml_content, "Description"
        )  # ignore description tags, e.g. in Figure, Stamp, Seal tags etc.
        xml_content = _remove_tag(xml_content, "Deletion")
        xml_content = _remove_tag(xml_content, "Illegible")
        xml_content = _remove_tag(xml_content, "Gap")

        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xml_content.encode("utf-8"), parser)

        # Build namespace map & prefix for XPath
        default_ns: str = root.nsmap.get(None) or ""
        xpath_nsmap: dict[str, str] = {}
        prefix: str = ""
        if default_ns:
            prefix = "docns"
            xpath_nsmap[prefix] = default_ns

        # Find all <Page> elements in document order
        page_query = f".//{prefix + ':' if prefix else ''}Page"
        pages: list[etree._Element] = root.xpath(page_query, namespaces=xpath_nsmap)

        text_parts_per_page: list[str] = []
        tags_to_extract: list[str] = ["Header", "Body", "Footer"]

        for page in pages:
            page_text_parts: list[str] = []
            for tag_name in tags_to_extract:
                qname = f"{prefix + ':' if prefix else ''}{tag_name}"
                elements: list[etree._Element] = page.xpath(f".//{qname}", namespaces=xpath_nsmap)
                for elem in elements:
                    # collect all text inside the element
                    lines: list[str] = []
                    for t_node in elem.itertext():
                        cleaned = re.sub(
                            r"<(/)?(lb|br)\s*/?>", "", t_node, flags=re.IGNORECASE
                        ).strip()
                        if cleaned:
                            lines.append(cleaned)
                    if lines:
                        page_text_parts.append("\n".join(lines))

            # join this page's parts if any
            if page_text_parts:
                text_parts_per_page.append("\n".join(page_text_parts))

        if not text_parts_per_page:
            logger.warning("No text found in any <Page>/<Header>,<Body>,<Footer> tags.")
            return ""

        full_text = "\n\n".join(text_parts_per_page)
        # cleanup stray <lb/> or <br> in text and collapse multiple blank lines
        full_text = re.sub(r"<(/)?(lb|br)\s*/?>", "", full_text, flags=re.IGNORECASE)
        full_text = re.sub(r"\n\s*\n+", "\n\n", full_text)
        return full_text.strip()

    except (etree.XMLSyntaxError, ET.ParseError) as e:
        logger.error(f"Failed to parse XML content: {e}")
        return ""
    except Exception as e:
        logger.exception(f"Unexpected error during text extraction: {e}")
        return ""


def _remove_tag(xml_content: str, tag_name: str) -> str:
    import re

    # Only process if tag present
    if f"<{tag_name}" not in xml_content:
        return xml_content
    # Remove opening to closing tags with any content in between
    xml_content = re.sub(rf"<{tag_name}\b[^>]*>.*?</{tag_name}>", "", xml_content, flags=re.DOTALL)
    # Remove self-closing tags
    xml_content = re.sub(rf"<{tag_name}\b[^>]*/>", "", xml_content)
    return xml_content

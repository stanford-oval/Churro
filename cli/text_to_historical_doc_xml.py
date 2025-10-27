from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import re

from lxml import etree  # type: ignore[import]
import xmlschema  # type: ignore[import]

from churro.utils.concurrency import run_async_in_parallel
from churro.utils.image.io import load_image_async
from churro.utils.llm import log_total_llm_cost, run_llm_async
from churro.utils.log_utils import logger


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "historical_doc.xsd"
_SCHEMA: xmlschema.XMLSchema | None = None
_SCHEMA_TEXT: str | None = None

DEFAULT_ENGINE = "gemini-2.5-pro-medium"
DEFAULT_MAX_CONCURRENCY = 64


@dataclass(frozen=True)
class DocumentExample:
    """Container for paired image/text inputs and the target XML output path."""

    stem: str
    image_path: Path
    text_path: Path
    xml_path: Path


def _load_schema() -> xmlschema.XMLSchema:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = xmlschema.XMLSchema(str(SCHEMA_PATH))
    return _SCHEMA


def get_historical_doc_xml_schema() -> str:
    global _SCHEMA_TEXT
    if _SCHEMA_TEXT is None:
        _SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8")
    return _SCHEMA_TEXT


def get_historical_doc_xml_errors(xml: str) -> str | None:
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:  # XML is malformed before schema validation
        return f"XMLSyntaxError: {exc}"

    schema = _load_schema()
    if schema.is_valid(root):
        return None

    try:
        schema.validate(root)
    except xmlschema.validators.exceptions.XMLSchemaValidationError as exc:
        return str(exc)
    except Exception as exc:  # Defensive guard for unexpected validation issues
        return f"ValidationError: {exc}"

    return "Unknown validation failure"


def normalize_xml_string(xml: str) -> str:
    xml = xml.strip()
    if not xml:
        return xml
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml.encode("utf-8"), parser=parser)
        return etree.tostring(root, encoding="unicode")
    except etree.XMLSyntaxError:
        return xml


def prettify_xml(xml: str) -> str:
    """Return formatted XML with consistent indentation."""
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml.encode("utf-8"), parser=parser)
        return etree.tostring(root, encoding="unicode", pretty_print=True)
    except etree.XMLSyntaxError:
        return xml


def strip_xml_tag(xml: str) -> str:
    if "```xml" in xml:
        last_xml = xml.rfind("```xml")
        end_xml = xml.find("```", last_xml + 1)
        if end_xml == -1:
            end_xml = len(xml)
        xml = xml[last_xml + 6 : end_xml].strip()

    xml = re.sub(r"<!--.*?-->", "", xml, flags=re.DOTALL).strip()
    xml = xml.replace(
        "<HistoricalDocument>",
        '<HistoricalDocument xmlns="http://example.com/historicaldocument">',
    )
    return normalize_xml_string(xml)


async def llm_fix_xml_syntax(xml: str, example: DocumentExample, engine: str) -> str:
    xml_errors = get_historical_doc_xml_errors(xml)
    if xml_errors is None:
        return xml

    logger.info(
        f"XML errors found for '{example.xml_path}': {xml_errors}. Fixing with another LLM call."
    )
    user_message = (
        f"You will be given an invalid XML with the following error: {xml_errors}.\n"
        "Minimally modify the XML so that it matches the provided schema while preserving the original content.\n\n"
        f"Schema:\n{get_historical_doc_xml_schema()}\n"
        f"Invalid XML:\n{xml}"
    )

    corrected_xml = await run_llm_async(
        model=engine,
        system_prompt_text="Output the entire fixed XML, do not output any other text.",
        user_message_text=user_message,
        user_message_image=None,
    )
    return strip_xml_tag(corrected_xml)


async def llm_transcribe(
    example: DocumentExample,
    engine: str,
    corpus_description: str,
) -> str:
    ocr_text = example.text_path.read_text(encoding="utf-8")
    if not ocr_text.strip():
        logger.warning(f"OCR text is empty in {example.text_path.name}")

    schema_text = get_historical_doc_xml_schema()
    corpus_line = (
        f"The document belongs to the corpus {corpus_description}.\n\n"
        if corpus_description
        else ""
    )

    with await load_image_async(example.image_path) as image:
        llm_output = await run_llm_async(
            model=engine,
            system_prompt_text=(
                "You are an expert XML generator. Output a single, valid XML document that adheres to "
                "the HistoricalDocument schema. Do not output explanations or markdown fences.\n\n"
                f"XSD Schema:\n{schema_text}"
            ),
            user_message_text=(
                "You are provided with a scanned historical document image and an OCR transcription.\n"
                "Produce a diplomatic transcription that preserves original spellings, punctuation, "
                "capitalization, and abbreviations. Correct OCR errors using the image as reference, "
                "add any missing but legible text, and follow the reading order of the document.\n\n"
                f"{corpus_line}"
                f"OCR Transcription:\n{ocr_text}\n\n"
                "Return the complete XML string that validates against the schema."
            ),
            user_message_image=image,
        )

    xml_content = strip_xml_tag(llm_output)
    if not xml_content:
        logger.warning(
            f"Output for {example.image_path} is empty after stripping formatting. Original LLM output: {llm_output}"
        )
        return ""

    xml_content = await llm_fix_xml_syntax(xml_content, example, engine)

    if not xml_content:
        logger.error(f"Failed to obtain valid XML for {example.stem} after LLM retries.")
        return ""

    if get_historical_doc_xml_errors(xml_content):
        logger.error(f"Final XML for {example.stem} still fails schema validation after retries.")
        return ""

    example.xml_path.write_text(prettify_xml(xml_content), encoding="utf-8")
    return xml_content


def collect_document_examples(input_dir: Path) -> list[DocumentExample]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    buckets: dict[str, dict[str, Path]] = {}
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".png", ".txt"}:
            continue
        bucket = buckets.setdefault(path.stem, {})
        if suffix == ".png":
            bucket["image"] = path
        else:
            bucket["text"] = path

    examples: list[DocumentExample] = []
    missing_components: list[str] = []

    for stem, files in sorted(buckets.items()):
        image_path = files.get("image")
        text_path = files.get("text")
        if image_path and text_path:
            xml_path = image_path.with_suffix(".xml")
            examples.append(DocumentExample(stem, image_path, text_path, xml_path))
        else:
            missing_components.append(stem)

    if missing_components:
        logger.warning(
            f"Skipping {len(missing_components)} unmatched file pair(s): {', '.join(missing_components)}"
        )

    logger.info(f"Found {len(examples)} complete PNG/TXT pair(s) in {input_dir}")
    return examples


async def process_examples(
    examples: list[DocumentExample],
    engine: str,
    max_concurrency: int,
    corpus_description: str,
) -> None:
    if not examples:
        logger.warning("No document pairs found to process.")
        return

    results = await run_async_in_parallel(
        partial(llm_transcribe, engine=engine, corpus_description=corpus_description),
        examples,
        max_concurrency=max_concurrency,
        desc="Generating XML",
    )

    success_count = sum(1 for result in results if isinstance(result, str) and result)
    logger.info(f"Successfully generated XML for {success_count}/{len(examples)} document(s)")


async def run_text_to_historical_doc_xml(
    input_dir: Path,
    engine: str,
    max_concurrency: int,
    corpus_description: str,
    overwrite: bool,
) -> None:
    examples = collect_document_examples(input_dir)

    if not overwrite:
        before = len(examples)
        examples = [example for example in examples if not example.xml_path.exists()]
        skipped = before - len(examples)
        if skipped:
            logger.info("Skipping %d existing XML file(s). Use --overwrite to regenerate.", skipped)

    await process_examples(examples, engine, max_concurrency, corpus_description)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HistoricalDocument XML for PNG/TXT pairs."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing paired X.png and X.txt files.",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help="Logical model key to use for LLM transcription.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum number of concurrent LLM calls.",
    )
    parser.add_argument(
        "--corpus-description",
        default="",
        help="Optional free-form description of the corpus."
        " Included in the LLM prompt for additional context.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate XML even when an output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    asyncio.run(
        run_text_to_historical_doc_xml(
            input_dir,
            args.engine,
            args.max_concurrency,
            args.corpus_description,
            args.overwrite,
        )
    )
    log_total_llm_cost()


if __name__ == "__main__":
    main()

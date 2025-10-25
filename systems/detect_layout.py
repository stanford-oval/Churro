"""Azure Document Intelligence-powered layout detection utilities.

This module provides helpers to detect page layout using Azure Document
Intelligence (prebuilt-layout) and apply basic geometric transformations
to align and then restore coordinates on the original image. It exposes a
single high-level coroutine, `detect_layout`, that returns:

- the processed `Page` (post-crop/rotation),
- the transformed `Image` (post-crop/rotation), and
- a second `Page` whose coordinates refer back to the original image
    (all transformations reversed).

"""

from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
import math
from pathlib import Path
import struct
from typing import Any, Literal, cast

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from diskcache import Cache
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import documentai
from PIL.Image import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from churro.config.settings import get_settings
from churro.page.page import Page
from churro.page.page_object import PageObject
from churro.page.visualization import crop_image_to_objects
from churro.utils.image.transform import adjust_image, rotate_image_and_page
from churro.utils.log_utils import logger


# -----------------------------
# Constants and type aliases
# -----------------------------
_SETTINGS = get_settings()

# Azure Document Intelligence model and request settings
AZURE_DI_ENDPOINT: str | None = _SETTINGS.azure_document_intelligence.endpoint
AZURE_DI_API_KEY: str | None = _SETTINGS.azure_document_intelligence.api_key
AZURE_LAYOUT_MODEL_ID: str = "prebuilt-layout"
AZURE_CONTENT_TYPE: str = "application/octet-stream"

# Pricing ($10 layout + $6 high-res) per 1000 pages
AZURE_DI_COST_PER_PAGE_USD: float = 16 / 1000

# Transformation tracking: either ("rotate", angle) or ("shift", dx, dy)
Transformation = tuple[Literal["rotate"], float] | tuple[Literal["shift"], float, float]

# Google Document AI OCR settings
GOOGLE_PROJECT_ID: str | None = _SETTINGS.vertex_ai.project_id
GOOGLE_LOCATION: str = _SETTINGS.vertex_ai.document_ai_location
GOOGLE_OCR_PROCESSOR_ID: str | None = _SETTINGS.vertex_ai.ocr_processor_id
GOOGLE_OCR_PROCESSOR_VERSION: str | None = _SETTINGS.vertex_ai.ocr_processor_version

GOOGLE_DOCUMENT_AI_COST_PER_PAGE_USD: float = (
    1.5 / 1000
)  # https://cloud.google.com/document-ai/pricing


# -----------------------------
# Module state
# -----------------------------

azure_document_intelligence_client: DocumentIntelligenceClient | None = None
total_azure_cost: float = 0.0

google_document_ai_client: documentai.DocumentProcessorServiceClient | None = None
google_document_ai_endpoint: str | None = None
total_google_document_ai_cost: float = 0.0

AZURE_DI_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / ".diskcache" / "azure_document_intelligence"
)
azure_di_cache = Cache(str(AZURE_DI_CACHE_PATH))


def log_total_azure_cost() -> None:
    """Log the running total Azure Document Intelligence cost for this process."""
    global total_azure_cost
    logger.info(f"Total Azure Document Intelligence cost: ${total_azure_cost:.2f}")


def get_total_azure_cost() -> float:
    """Return the accumulated Azure API cost."""
    global total_azure_cost
    return total_azure_cost


def _build_azure_cache_key(
    image_bytes: bytes,
    skip_paragraphs: bool,
    output_ocr_text: bool,
) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"azure_di:{digest}:{int(skip_paragraphs)}:{int(output_ocr_text)}"


def log_total_google_document_ai_cost() -> None:
    """Log the running total Google Document AI cost for this process."""
    global total_google_document_ai_cost
    logger.info(f"Total Google Document AI cost: ${total_google_document_ai_cost:.2f}")


def get_total_google_document_ai_cost() -> float:
    """Return the accumulated Google Document AI cost."""
    global total_google_document_ai_cost
    return total_google_document_ai_cost


def _determine_google_endpoint(location: str) -> str:
    if location and location.lower() != "global":
        return f"{location}-documentai.googleapis.com"
    return "documentai.googleapis.com"


def _ensure_google_document_ai_client(
    location: str,
) -> documentai.DocumentProcessorServiceClient:
    """Initialize or return a cached Google Document AI client for the given location."""
    global google_document_ai_client, google_document_ai_endpoint
    endpoint = _determine_google_endpoint(location)
    if google_document_ai_client is not None and google_document_ai_endpoint == endpoint:
        return google_document_ai_client

    client_options = ClientOptions(api_endpoint=endpoint)
    google_document_ai_client = documentai.DocumentProcessorServiceClient(
        client_options=client_options
    )
    google_document_ai_endpoint = endpoint
    return google_document_ai_client


def _resolve_google_processor_name(
    client: documentai.DocumentProcessorServiceClient,
) -> str:
    if GOOGLE_PROJECT_ID is None:
        raise RuntimeError(
            "Google Document AI project is not configured. Set VERTEX_AI_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
        )
    if GOOGLE_OCR_PROCESSOR_ID is None:
        raise RuntimeError(
            "Google Document AI processor is not configured. Set VERTEX_AI_OCR_PROCESSOR_ID."
        )

    processor_name = client.processor_path(
        GOOGLE_PROJECT_ID, GOOGLE_LOCATION, GOOGLE_OCR_PROCESSOR_ID
    )
    if GOOGLE_OCR_PROCESSOR_VERSION:
        processor_name = client.processor_version_path(
            GOOGLE_PROJECT_ID,
            GOOGLE_LOCATION,
            GOOGLE_OCR_PROCESSOR_ID,
            GOOGLE_OCR_PROCESSOR_VERSION,
        )
    return processor_name


def _layout_to_coordinates(
    layout: documentai.Document.Page.Layout | None, width: float, height: float
) -> list[float] | None:
    if layout is None or layout.bounding_poly is None:
        return None

    coords: list[float] = []
    vertices = layout.bounding_poly.vertices or []
    if vertices:
        for vertex in vertices:
            if vertex.x is None or vertex.y is None:
                continue
            coords.extend([float(vertex.x), float(vertex.y)])
    else:
        normalized_vertices = layout.bounding_poly.normalized_vertices or []
        for vertex in normalized_vertices:
            if vertex.x is None or vertex.y is None:
                continue
            coords.extend([float(vertex.x) * width, float(vertex.y) * height])

    if len(coords) < 6 or len(coords) % 2 != 0:
        logger.debug(f"Skipping polygon creation due to invalid coordinates: {coords}")
        return None
    return coords


def _collect_text_spans(
    layout: documentai.Document.Page.Layout | None,
) -> list[tuple[int, int]]:
    if layout is None or layout.text_anchor is None:
        return []
    spans: list[tuple[int, int]] = []
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index) if segment.start_index is not None else 0
        end_index = int(segment.end_index) if segment.end_index is not None else start_index
        spans.append((start_index, end_index))
    return spans


def _is_span_within(span: tuple[int, int], containers: list[tuple[int, int]]) -> bool:
    start, end = span
    for container_start, container_end in containers:
        if start >= container_start and end <= container_end:
            return True
    return False


def _build_page_from_google_document(
    document: documentai.Document,
    skip_paragraphs: bool,
    image_width: int,
    image_height: int,
) -> Page:
    if not document.pages:
        raise RuntimeError("Google Document AI response did not contain any pages.")

    page_proto = document.pages[0]
    width = float(page_proto.dimension.width or image_width)
    height = float(page_proto.dimension.height or image_height)

    page_objects: list[PageObject] = []
    object_id = 1
    paragraph_spans: list[tuple[int, int]] = []

    if not skip_paragraphs:
        for paragraph in getattr(page_proto, "paragraphs", []):
            coords = _layout_to_coordinates(paragraph.layout, width, height)
            if coords is None:
                continue
            paragraph_spans.extend(_collect_text_spans(paragraph.layout))
            page_objects.append(
                PageObject(
                    object_id=str(object_id),
                    coordinates=coords,
                )
            )
            object_id += 1

    lines_added = 0
    for line in getattr(page_proto, "lines", []):
        spans = _collect_text_spans(line.layout)
        if spans and all(_is_span_within(span, paragraph_spans) for span in spans):
            continue
        coords = _layout_to_coordinates(line.layout, width, height)
        if coords is None:
            continue
        lines_added += 1
        page_objects.append(
            PageObject(
                object_id=str(object_id),
                coordinates=coords,
            )
        )
        object_id += 1

    if lines_added > 0:
        logger.info(f"Added {lines_added} Google OCR lines that were not part of any paragraph")

    page = Page(
        page_objects=page_objects,
    )

    return page


def _google_skew_from_transforms(page_proto: documentai.Document.Page) -> float | None:
    transforms = getattr(page_proto, "transforms", [])
    if not transforms:
        return None
    matrix = transforms[0]
    if matrix.rows != 2 or matrix.cols != 3:
        logger.debug(f"Unexpected transform matrix dimensions: {matrix.rows}x{matrix.cols}")
        return None
    try:
        a, b, _tx, c, d, _ty = struct.unpack("<6d", matrix.data)
    except struct.error as exc:
        logger.debug(f"Failed to unpack Google transform matrix: {exc}")
        return None
    try:
        rotation_radians = math.atan2(b, a)
    except (TypeError, ValueError) as exc:
        logger.debug(f"Invalid rotation components in transform matrix: {exc}")
        return None
    return -math.degrees(rotation_radians)


def _google_orientation_to_angle(page_proto: documentai.Document.Page) -> float:
    skew_angle = _google_skew_from_transforms(page_proto)
    if skew_angle is not None:
        return skew_angle

    if page_proto.layout and page_proto.layout.orientation is not None:
        orientation_map = {
            documentai.Document.Page.Layout.Orientation.PAGE_UP: 0.0,
            documentai.Document.Page.Layout.Orientation.PAGE_RIGHT: -90.0,
            documentai.Document.Page.Layout.Orientation.PAGE_DOWN: 180.0,
            documentai.Document.Page.Layout.Orientation.PAGE_LEFT: 90.0,
        }
        try:
            return orientation_map.get(page_proto.layout.orientation, 0.0)
        except AttributeError:
            orientation_name = str(page_proto.layout.orientation)
            fallback_map = {
                "PAGE_UP": 0.0,
                "PAGE_RIGHT": -90.0,
                "PAGE_DOWN": 180.0,
                "PAGE_LEFT": 90.0,
            }
            return fallback_map.get(orientation_name.upper(), 0.0)

    return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(3),
    retry=retry_if_exception_type(HttpResponseError),
)
async def run_azure_document_analysis_on_image(
    image: Image, skip_paragraphs: bool, output_ocr_text: bool = False
) -> tuple[Page, float] | str:
    """Analyze an image with Azure Document Intelligence.

    Args:
        image: PIL image to analyze. The image is serialized to JPEG in memory.
        skip_paragraphs: Passed to Page conversion logic to optionally skip paragraphs.
        output_ocr_text: If True, returns the raw OCR text content instead of a Page.

    Returns:
        Either a tuple of (Page, angle_degrees) where angle is the page rotation angle
        reported by Azure, or a string containing the OCR text when output_ocr_text is True.
    """
    logger.info("Running Azure Document Analysis on image")
    _ensure_azure_di_initialized()

    # Preprocess to improve OCR/layout analysis
    image = adjust_image(image, thresholding=True)

    # Save the processed image to an in-memory stream
    image_stream = BytesIO()
    image.save(image_stream, format="jpeg")
    image_bytes = image_stream.getvalue()

    cache_key = _build_azure_cache_key(image_bytes, skip_paragraphs, output_ocr_text)
    cached_payload_raw = await asyncio.to_thread(azure_di_cache.get, cache_key)
    if cached_payload_raw is not None:
        cached_payload = cast(dict[str, Any], cached_payload_raw)
        logger.info("Azure Document Intelligence cache hit; returning cached result")
        if cached_payload["kind"] == "text":
            return cached_payload["content"]
        cached_page = Page.model_validate(cached_payload["page"])
        cached_angle = float(cached_payload.get("angle", 0.0))
        return cached_page, cached_angle

    image_stream = BytesIO(image_bytes)

    assert isinstance(azure_document_intelligence_client, DocumentIntelligenceClient), (
        "Azure Document Intelligence client is not initialized"
    )
    poller = await azure_document_intelligence_client.begin_analyze_document(
        model_id=AZURE_LAYOUT_MODEL_ID,
        body=image_stream,
        content_type=AZURE_CONTENT_TYPE,
        features=[
            DocumentAnalysisFeature.OCR_HIGH_RESOLUTION,
        ],
    )
    analysis_result: AnalyzeResult = await poller.result()
    global total_azure_cost
    total_azure_cost += AZURE_DI_COST_PER_PAGE_USD

    if output_ocr_text:
        cache_value: dict[str, Any] = {"kind": "text", "content": analysis_result.content}
        await asyncio.to_thread(azure_di_cache.set, cache_key, cache_value)
        return analysis_result.content

    page = Page.from_azure_analysis_result(analysis_result, skip_paragraphs=skip_paragraphs)
    # Robustly handle missing/None angles
    try:
        angle_value = analysis_result.pages[0].angle  # type: ignore[assignment]
    except (AttributeError, IndexError):
        angle_value = 0.0
    angle: float = float(angle_value or 0.0)

    cache_value: dict[str, Any] = {
        "kind": "page",
        "page": page.model_dump(mode="python"),
        "angle": angle,
    }
    await asyncio.to_thread(azure_di_cache.set, cache_key, cache_value)

    return page, angle


async def shutdown_layout_clients() -> None:
    """Close any cached async clients used for layout detection."""
    global azure_document_intelligence_client, google_document_ai_client

    if azure_document_intelligence_client is not None:
        try:
            await azure_document_intelligence_client.close()
        finally:
            azure_document_intelligence_client = None

    if google_document_ai_client is not None:
        close_fn = getattr(google_document_ai_client, "close", None)
        transport = getattr(google_document_ai_client, "transport", None)
        try:
            if callable(close_fn):
                close_fn()
            elif transport is not None:
                transport_close = getattr(transport, "close", None)
                if callable(transport_close):
                    transport_close()
        finally:
            google_document_ai_client = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(3),
    retry=retry_if_exception_type(GoogleAPICallError),
)
async def run_google_document_analysis_on_image(
    image: Image, skip_paragraphs: bool, output_ocr_text: bool = False
) -> tuple[Page, float] | str:
    """Analyze an image with Google Document AI OCR processor."""
    logger.info("Running Google Document AI OCR on image")

    client = _ensure_google_document_ai_client(GOOGLE_LOCATION)
    processor_name = _resolve_google_processor_name(client)

    image_stream = BytesIO()
    image.save(image_stream, format="jpeg")
    document_bytes = image_stream.getvalue()

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(content=document_bytes, mime_type="image/jpeg"),
        process_options=documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_image_quality_scores=True,
            )
        ),
    )

    response = await asyncio.to_thread(client.process_document, request=request)

    global total_google_document_ai_cost
    total_google_document_ai_cost += GOOGLE_DOCUMENT_AI_COST_PER_PAGE_USD

    document = response.document
    if output_ocr_text:
        return document.text or ""

    page = _build_page_from_google_document(
        document,
        skip_paragraphs=skip_paragraphs,
        image_width=image.width,
        image_height=image.height,
    )
    page_proto = document.pages[0]
    angle = _google_orientation_to_angle(page_proto)

    return page, angle


async def tidy_image_via_layout_detection(
    image: Image,
    margin: int = 30,
) -> Image:
    """Detect layout and return processed and original-coordinate pages.

    This performs light preprocessing, calls Azure DI layout analysis, applies
    rotation and content-cropping, and then returns a deep-copied `Page` with
    all transformations reversed so the coordinates map to the original image.

    Args:
        image: Input page image.
        margin: Margin in pixels to add around the detected content when cropping.

    Returns:
        processed_image.
    """
    result = await run_azure_document_analysis_on_image(image, skip_paragraphs=False)
    # We always call with output_ocr_text=False, so result must be (Page, float)
    if isinstance(result, str):  # defensive runtime check
        raise RuntimeError("Unexpected OCR text output when a Page was expected.")
    page, page_angle = result

    page.remove_subsumed_page_objects()
    page.remove_small_page_objects_in_margins()

    # Apply rotation if needed
    image, page = rotate_image_and_page(image, page_angle, page)

    # Crop to content and shift page objects accordingly
    image = crop_image_to_objects(image, page.page_objects, margin=margin)

    return image


def _ensure_azure_di_initialized() -> DocumentIntelligenceClient:
    """Initialize or return a cached Azure DI client.

    Reads the API key from the AZURE_DOC_KEY environment variable and the endpoint
    from AZURE_DI_ENDPOINT (optional). A singleton client is cached in the module
    state for reuse.

    Returns:
        A configured `DocumentIntelligenceClient` instance.

    Raises:
        ValueError: If the AZURE_DOC_KEY environment variable is not set.
    """
    global azure_document_intelligence_client
    if azure_document_intelligence_client is not None:
        return azure_document_intelligence_client

    api_key = AZURE_DI_API_KEY
    if not api_key:
        raise ValueError("No Azure Document Key found (AZURE_DOC_KEY not set)")

    if AZURE_DI_ENDPOINT is None:
        raise RuntimeError(
            "AZURE_DI_ENDPOINT environment variable must be set before using Azure layout detection."
        )
    azure_document_intelligence_client = DocumentIntelligenceClient(
        endpoint=AZURE_DI_ENDPOINT,
        credential=AzureKeyCredential(api_key),
    )
    return azure_document_intelligence_client

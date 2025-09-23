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

from io import BytesIO
import os
from typing import Literal, Optional

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import dotenv
from PIL.Image import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from page.image_utils import adjust_image, rotate_image_and_page
from page.page import Page
from page.visualization import crop_image_and_page_to_content
from utils.log_utils import logger


# -----------------------------
# Constants and type aliases
# -----------------------------
dotenv.load_dotenv()

# Azure Document Intelligence model and request settings
AZURE_DI_ENDPOINT: str | None = os.getenv("AZURE_DI_ENDPOINT")
AZURE_LAYOUT_MODEL_ID: str = "prebuilt-layout"
AZURE_CONTENT_TYPE: str = "application/octet-stream"

# Pricing ($10 layout + $6 high-res) per 1000 pages
COST_PER_PAGE_USD: float = 16 / 1000

# Transformation tracking: either ("rotate", angle) or ("shift", dx, dy)
Transformation = tuple[Literal["rotate"], float] | tuple[Literal["shift"], float, float]


# -----------------------------
# Module state
# -----------------------------

azure_document_intelligence_client: Optional[DocumentIntelligenceClient] = None
total_azure_cost: float = 0.0


def log_total_azure_cost() -> None:
    """Log the running total Azure Document Intelligence cost for this process."""
    global total_azure_cost
    logger.info(f"Total Azure Document Intelligence cost: ${total_azure_cost:.2f}")


def get_total_azure_cost() -> float:
    """Return the accumulated Azure API cost."""
    global total_azure_cost
    return total_azure_cost


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

    # Save the processed image to an in-memory stream
    image_stream = BytesIO()
    image.save(image_stream, format="jpeg")
    image_stream.seek(0)  # Reset the stream position to the beginning

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
    total_azure_cost += COST_PER_PAGE_USD

    if output_ocr_text:
        return analysis_result.content

    page = Page.from_azure_analysis_result(analysis_result, skip_paragraphs=skip_paragraphs)
    # Robustly handle missing/None angles
    try:
        angle_value = analysis_result.pages[0].angle  # type: ignore[assignment]
    except (AttributeError, IndexError):
        angle_value = 0.0
    angle: float = float(angle_value or 0.0)

    return page, angle


async def detect_layout(
    image: Image,
) -> tuple[Page, Image, Page]:
    """Detect layout and return processed and original-coordinate pages.

    This performs light preprocessing, calls Azure DI layout analysis, applies
    rotation and content-cropping, and then returns a deep-copied `Page` with
    all transformations reversed so the coordinates map to the original image.

    Args:
        image: Input page image.

    Returns:
        A tuple of (processed_page, processed_image, original_coordinate_page).
    """
    # Record transformations in application order to allow later reversal.
    transformation_history: list[Transformation] = []

    # Preprocess to improve OCR/layout analysis
    image = adjust_image(image, thresholding=True)

    # Analyze with Azure Document Intelligence (additional models could be added here)
    all_pages: list[Page] = []
    all_page_angles: list[float] = []
    result = await run_azure_document_analysis_on_image(image, skip_paragraphs=False)
    # We always call with output_ocr_text=False, so result must be (Page, float)
    if isinstance(result, str):  # defensive runtime check
        raise RuntimeError("Unexpected OCR text output when a Page was expected.")
    ldf_page, ldf_page_angle = result
    all_pages.append(ldf_page)
    all_page_angles.append(ldf_page_angle)

    # Choose the most confident rotation (largest absolute angle)
    page_angle: float = max(all_page_angles, key=abs) if all_page_angles else 0.0

    # Merge page objects from all models without duplication of the first page's objects
    page = Page(
        page_objects=[],
        image_path=None,
    )
    for p in all_pages:
        page.page_objects.extend(p.page_objects)
    page.remove_subsumed_page_objects(coverage_ratio=0.7)

    # Apply rotation if needed
    if page_angle != 0:
        if abs(page_angle) > 3:
            logger.info(f"Rotating page by {page_angle:.1f} degrees")
        transformation_history.append(("rotate", page_angle))
        image, page = rotate_image_and_page(image, page_angle, page)

    # Crop to content and shift page objects accordingly
    image, left_margin_removed, top_margin_removed = crop_image_and_page_to_content(image, page)
    for o in page.page_objects:
        o.bounding_region = o.bounding_region.shift_by_amount(
            -left_margin_removed, -top_margin_removed
        )
    transformation_history.append(("shift", -left_margin_removed, -top_margin_removed))

    # Reverse the transformations so coordinates refer to the original image
    page_with_original_coordinates = page.model_copy(deep=True)
    for operation in reversed(transformation_history):
        if operation[0] == "shift":
            dx, dy = operation[1], operation[2]
            logger.debug(f"Reversing shift by ({dx}, {dy})")
            for o in page_with_original_coordinates.page_objects:
                o.bounding_region = o.bounding_region.shift_by_amount(-dx, -dy)
        elif operation[0] == "rotate":
            angle = operation[1]
            logger.debug(f"Reversing rotation by {-angle:.2f} degrees")
            _, page_with_original_coordinates = rotate_image_and_page(
                image, -angle, page_with_original_coordinates
            )

    return page, image, page_with_original_coordinates


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

    api_key = os.getenv("AZURE_DOC_KEY")
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

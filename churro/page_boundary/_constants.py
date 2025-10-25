"""Shared constants for Gemini page boundary detection."""

from __future__ import annotations


PAGE_RESPONSE_INSTRUCTIONS = """- Return a JSON object with a key "pages".
- "pages" must be a list containing zero or more objects.
- Each object must have:
        * "page_index": 1-based index in reading order.
        * "left": integer normalized 0-1000 describing the minimum horizontal coordinate.
        * "top": integer normalized 0-1000 describing the minimum vertical coordinate.
        * "right": integer normalized 0-1000 describing the maximum horizontal coordinate.
        * "bottom": integer normalized 0-1000 describing the maximum vertical coordinate.
- Provide all coordinates as integers (no decimals) and keep them normalized to the
    0-1000 range.
- The box should cover every part of the physical page contents, including all printed
    text, handwriting, stamps, or drawings that belong on the page surface.
- Bounding boxes must cover the entire page content, excluding empty margins.
- Do not include printer artifacts, scanner noise, shadows, binding rings, or any
    background beyond the true page edges.
- If no page is visible, return {"pages": []}.
"""

PAGE_DETECTION_PROMPT = (
    "You are a careful vision assistant that locates the page content inside\n"
    "scanned or photographed document images. Return tight bounding boxes around each\n"
    "page (there will be either one page or a two-page spread). Ensure coordinates are\n"
    "precise.\n\n"
    "Identify every document page in this image. Follow these rules:\n"
    f"{PAGE_RESPONSE_INSTRUCTIONS}"
)

PAGE_REVIEW_PROMPT_TEMPLATE = (
    "You are an expert reviewer of document page boundary annotations. Evaluate the provided\n"
    "image with red rectangles drawn from the most recent prediction and output corrected boxes\n"
    "in the same JSON structure when adjustments are needed.\n\n"
    "Only the annotated preview image is supplied, so rely on the drawn rectangles and the\n"
    "visible page content to decide whether corrections are necessary.\n\n"
    "The currently drawn bounding boxes are summarized with lines such as\n"
    '"left : 412" for each page. Use them as context, but respond with a fresh JSON object that\n'
    "follows the required schema.\n\n"
    "Inspect the red rectangles and correct them if they are inaccurate or incomplete. Think about how much they should be moved to better fit the page content.\n"
    "Respond with JSON following these rules:\n"
    f"{PAGE_RESPONSE_INSTRUCTIONS}"
    "If the rectangles are already correct, you may return the same coordinates."
)

PAGE_DETECTION_BOX_WIDTH = 10
MAX_PAGE_REVIEW_ROUNDS = 4
BORDER_FRACTION = 0.10
PROCESSED_MAX_DIM = 1500
GUIDELINE_COLOR = "#ff3b30"
DEFAULT_MODEL_KEY = "gemini-2.5-pro-high"
DAMPENING_CONSTANT = 1

_SCALE_WITH_BORDER = 1 + 2 * BORDER_FRACTION
NORMALIZED_MIN_COORD = (BORDER_FRACTION / _SCALE_WITH_BORDER) * 1000
NORMALIZED_MAX_COORD = ((1 + BORDER_FRACTION) / _SCALE_WITH_BORDER) * 1000

__all__ = [
    "PAGE_RESPONSE_INSTRUCTIONS",
    "PAGE_DETECTION_PROMPT",
    "PAGE_REVIEW_PROMPT_TEMPLATE",
    "PAGE_DETECTION_BOX_WIDTH",
    "MAX_PAGE_REVIEW_ROUNDS",
    "BORDER_FRACTION",
    "DAMPENING_CONSTANT",
    "PROCESSED_MAX_DIM",
    "GUIDELINE_COLOR",
    "DEFAULT_MODEL_KEY",
    "NORMALIZED_MIN_COORD",
    "NORMALIZED_MAX_COORD",
]

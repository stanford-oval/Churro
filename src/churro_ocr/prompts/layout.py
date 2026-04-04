"""Default prompts used by page-detection backends."""

from __future__ import annotations

import json

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
- CRITICAL GOAL: Find the **tightest possible bounding box** that contains **ALL**
    meaningful content on the page.
- INCLUDE:
    * All printed text (headers, footers, page numbers, body text).
    * All handwriting (signatures, marginalia, corrections).
    * All stamps, seals, logos, and drawings.
    * Any content that conveys information.
- EXCLUDE:
    * Empty page margins (white space).
    * Content belonging to a different page (even if it appears in the same image).
    * Partial/overflowing text or graphics from a neighboring page.
    * Dark edges or background from the scanner/camera.
    * Binding rings, spiral binding, or book spines.
    * Shadows, scanner noise, or artifacts outside the content area.
- The box should be as small as possible while still containing every pixel of ink/content.
- Each returned box must correspond to exactly one page's content region.
- If no page is visible, return {"pages": []}.
"""

DEFAULT_BOUNDARY_DETECTION_PROMPT = (
    "You are an expert document analysis AI. Your task is to detect the precise boundaries\n"
    "of document pages to prepare them for OCR.\n"
    "The goal is to crop the image to the **tightest possible rectangle** that contains all\n"
    "the content, removing as much empty margin as possible.\n\n"
    "IMPORTANT TERMINOLOGY: In this task, 'page' means the tightest bounding box around the\n"
    "visible content on a page, NOT the full sheet of paper. Exclude blank/white paper margins\n"
    "whenever they do not contain meaningful content.\n\n"
    "If a neighboring page intrudes into the image (or overlaps visually), do NOT include that\n"
    "spillover content in this page's box. Each box should isolate one page only.\n\n"
    "Identify every document page in this image (usually 1 or 2). For each page, define a\n"
    "bounding box following these strict rules:\n"
    f"{PAGE_RESPONSE_INSTRUCTIONS}"
)


def build_boundary_review_prompt(
    *,
    edge_name: str,
    page_index: int,
    strip_axis: str,
) -> str:
    """Build the per-edge review prompt used for iterative page-boundary refinement."""
    return (
        "You are an expert reviewer of a document page boundary annotation.\n"
        "You are reviewing ONE EDGE of the red rectangle using an edge strip crop.\n"
        "The crop is a narrow strip centered on the target edge of the current red boundary.\n\n"
        "IMPORTANT TERMINOLOGY: 'Page boundary' means the tightest box around page CONTENT\n"
        "(ink/text/stamps/handwriting), not the full paper edge. Exclude white/blank paper margins\n"
        "unless they contain meaningful content.\n\n"
        "CRITICAL ISOLATION RULE: Exclude content from neighboring pages. If this strip shows\n"
        "partial/overflowing text or graphics from another page crossing near the target edge,\n"
        "do not expand the target page box to include that other page's content.\n\n"
        f"Target page_index: {page_index}\n"
        f"Target edge: {edge_name}\n\n"
        "Task:\n"
        f"- Judge ONLY the {edge_name} edge of the red rectangle.\n"
        "- Decide whether that edge should expand, shrink, or stay unchanged.\n"
        "- Base the decision on visible content near that edge in this strip crop.\n"
        "- Keep only the target page's content; exclude neighboring-page spillover.\n"
        "- Do not make decisions for the other three edges.\n\n"
        "Output JSON only with this schema:\n"
        "{\n"
        '  "page_index": <integer>,\n'
        f'  "edge": "{edge_name}",\n'
        '  "action": "expand" | "shrink" | "no_change",\n'
        '  "amount": <integer>\n'
        "}\n\n"
        "Amount semantics:\n"
        f"- 'amount' is a normalized 0-1000 delta along the strip's {strip_axis}.\n"
        "- For 'no_change', return amount = 0.\n"
        "- Prefer 'no_change' unless there is a clear improvement.\n"
        "- Do NOT return absolute box coordinates.\n"
    )


def build_text_block_localization_prompt(*, block_tag: str, block_text: str) -> str:
    """Build the initial prompt for locating a single rendered text block."""
    target_tag = json.dumps(block_tag, ensure_ascii=False)
    target_text = json.dumps(block_text, ensure_ascii=False)
    return (
        "You are an expert reviewer of historical document layout.\n"
        "Your task is to find the SINGLE rendered content block in the image that matches the\n"
        "target HDML block below.\n\n"
        f"Target block tag (JSON string): {target_tag}\n"
        f"Target block text (JSON string): {target_text}\n\n"
        "MATCHING RULES:\n"
        "- The target may be a heading, paragraph, date line, marginal note, block quotation,\n"
        "  list, table, figure caption, or page number depending on the tag.\n"
        "- Match the same block even if whitespace, punctuation, line breaks, ligatures, or\n"
        "  historical glyphs differ slightly from the normalized transcription.\n"
        "- Return one box for the WHOLE target block, not for an individual line, word, cell,\n"
        "  or inline span inside it.\n"
        "- If multiple visually identical matches exist and you cannot confidently choose one,\n"
        "  return not found.\n\n"
        "BOUNDING BOX RULES:\n"
        "- Return the tightest possible bounding box around the entire target block only.\n"
        "- INCLUDE all ink that belongs to that block.\n"
        "- For tables, include all cells and ruling lines belonging to the target table.\n"
        "- For lists, include all items belonging to the target list.\n"
        "- Exclude neighboring paragraphs, headings, notes, page numbers, tables, and unrelated\n"
        "  marks or decorations.\n\n"
        "Output JSON only with this schema:\n"
        "{\n"
        '  "block_found": <true|false>,\n'
        '  "block": {\n'
        '    "left": <integer normalized 0-1000>,\n'
        '    "top": <integer normalized 0-1000>,\n'
        '    "right": <integer normalized 0-1000>,\n'
        '    "bottom": <integer normalized 0-1000>\n'
        "  } | null\n"
        "}\n\n"
        "- If the target block is absent or cannot be uniquely identified, return "
        '{"block_found": false, "block": null}.\n'
        "- Provide all coordinates as integers in the 0-1000 range.\n"
        "- Do not return any explanation.\n"
    )


def build_text_block_boundary_review_prompt(
    *,
    edge_name: str,
    block_tag: str,
    block_text: str,
    strip_axis: str,
) -> str:
    """Build the per-edge review prompt used for iterative text-block refinement."""
    target_tag = json.dumps(block_tag, ensure_ascii=False)
    target_text = json.dumps(block_text, ensure_ascii=False)
    return (
        "You are an expert reviewer of a content-block bounding box annotation.\n"
        "You are reviewing ONE EDGE of the red rectangle using an edge strip crop.\n"
        "The crop is a narrow strip centered on the target edge of the current red boundary.\n\n"
        f"Target block tag (JSON string): {target_tag}\n"
        f"Target block text (JSON string): {target_text}\n"
        f"Target edge: {edge_name}\n\n"
        "CRITICAL GOAL:\n"
        "- The rectangle must tightly contain only the single target block.\n"
        "- Include all ink from that block and exclude neighboring blocks.\n"
        "- For tables, include all table text and ruling lines belonging to that table.\n"
        "- For lists, include all items belonging to that list.\n"
        "- Exclude nearby paragraphs, headings, marginal notes, page numbers, stains, borders,\n"
        "  and unrelated annotations.\n\n"
        "Task:\n"
        f"- Judge ONLY the {edge_name} edge of the red rectangle.\n"
        "- Decide whether that edge should expand, shrink, or stay unchanged.\n"
        "- Base the decision on the target block only.\n"
        "- Do not make decisions for the other three edges.\n\n"
        "Output JSON only with this schema:\n"
        "{\n"
        f'  "edge": "{edge_name}",\n'
        '  "action": "expand" | "shrink" | "no_change",\n'
        '  "amount": <integer>\n'
        "}\n\n"
        "Amount semantics:\n"
        f"- 'amount' is a normalized 0-1000 delta along the strip's {strip_axis}.\n"
        "- For 'no_change', return amount = 0.\n"
        "- Prefer 'no_change' unless there is a clear improvement.\n"
        "- Do NOT return absolute box coordinates.\n"
    )

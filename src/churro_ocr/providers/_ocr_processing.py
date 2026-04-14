"""Shared OCR preprocessing and postprocessing helpers for built-in profiles."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from PIL import Image

from churro_ocr._internal.image import ensure_rgb, prepare_ocr_image, resize_image_to_fit
from churro_ocr.prompts import (
    DEFAULT_OCR_OUTPUT_TAG,
    parse_chandra_response,
    parse_olmocr_response,
    strip_ocr_output_tag,
    strip_rich_ocr_markup_to_plain_text,
)
from churro_ocr.templates import (
    DEEPSEEK_OCR_2_OCR_PROMPT,
    FIRERED_OCR_OCR_PROMPT,
    GLM_OCR_OCR_PROMPT,
    INFINITY_PARSER_7B_OCR_PROMPT,
    INFINITY_PARSER_7B_SYSTEM_PROMPT,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    NANONETS_OCR2_3B_OCR_PROMPT,
    NANONETS_OCR2_3B_SYSTEM_PROMPT,
    PADDLEOCR_VL_1_5_OCR_PROMPT,
)
from churro_ocr.types import MetadataDict

if TYPE_CHECKING:
    from collections.abc import Sequence

TextPostprocessorResult = str | tuple[str, MetadataDict]
CHANDRA_MAX_IMAGE_SIZE = (3_072, 2_048)
CHANDRA_MIN_IMAGE_SIZE = (1_792, 28)
CHANDRA_IMAGE_GRID_SIZE = 28
GLM_OCR_IMAGE_GRID_SIZE = 28
GLM_OCR_TEMPORAL_PATCH_SIZE = 2
GLM_OCR_VLLM_MAX_IMAGE_ITEM_LENGTH = 6_084
GLM_OCR_VLLM_MAX_PIXELS = (
    GLM_OCR_IMAGE_GRID_SIZE**2 * GLM_OCR_TEMPORAL_PATCH_SIZE * GLM_OCR_VLLM_MAX_IMAGE_ITEM_LENGTH
)
OLMOCR_TARGET_LONGEST_IMAGE_DIM = 1_288
_CHAT_ROLE_PREFIXES = {
    "assistant",
    "assistant:",
    "user",
    "user:",
    "system",
    "system:",
    "<assistant>",
    "<user>",
    "<system>",
    "<|assistant|>",
    "<|assistant|>:",
    "<|user|>",
    "<|user|>:",
    "<|system|>",
    "<|system|>:",
    "<｜assistant｜>",
    "<｜assistant｜>:",
    "<｜user｜>",
    "<｜user｜>:",
    "<｜system｜>",
    "<｜system｜>:",
}
_OUTER_FENCED_CODE_BLOCK_RE = re.compile(
    r"^(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n(?P<body>.*)\n(?P=fence)$",
    flags=re.DOTALL,
)


def strip_leading_chat_scaffold(text: str, *, prompts: Sequence[str]) -> str:
    """Remove echoed prompts and leading chat role markers from model output."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    normalized_prompts = tuple(prompt.strip() for prompt in prompts if prompt and prompt.strip())
    for _ in range(8):
        previous = cleaned
        lowered = cleaned.casefold()
        stripped_prompt = False
        for prompt in normalized_prompts:
            if lowered.startswith(prompt.casefold()):
                cleaned = cleaned[len(prompt) :].lstrip()
                stripped_prompt = True
                break
        if stripped_prompt:
            continue

        lines = cleaned.splitlines()
        if not lines:
            return ""
        first_line = lines[0].strip()
        if first_line.casefold() in _CHAT_ROLE_PREFIXES:
            cleaned = "\n".join(lines[1:]).lstrip()
            continue
        if re.fullmatch(r"<\|?(?:assistant|user|system)\|?>", first_line, flags=re.IGNORECASE):
            cleaned = "\n".join(lines[1:]).lstrip()
            continue
        if cleaned == previous:
            break
    return cleaned.strip()


def strip_outer_fenced_code_block(text: str) -> str:
    """Unwrap a single outer fenced code block while preserving its inner content."""
    cleaned = text.strip()
    match = _OUTER_FENCED_CODE_BLOCK_RE.fullmatch(cleaned)
    if match is None:
        return cleaned
    return match.group("body").strip()


def identity_text_postprocessor(text: str) -> str:
    """Return OCR text unchanged.

    :param text: OCR text to return.
    :returns: The original ``text`` value.
    """
    return text


def default_ocr_image_preprocessor(image: Image.Image) -> Image.Image:
    """Apply the default OCR image preprocessing.

    :param image: Source page image.
    :returns: Preprocessed image ready for OCR.
    """
    return prepare_ocr_image(image)


def default_ocr_text_postprocessor(text: str) -> str:
    """Strip the default OCR output tag wrapper.

    :param text: Raw OCR response text.
    :returns: OCR text with the default wrapper removed when present.
    """
    return strip_ocr_output_tag(text, output_tag=DEFAULT_OCR_OUTPUT_TAG)


def olmocr_image_preprocessor(image: Image.Image) -> Image.Image:
    """Resize an image to olmOCR's expected 1288px longest side and normalize to RGB."""
    return ensure_rgb(
        resize_image_to_fit(
            image,
            OLMOCR_TARGET_LONGEST_IMAGE_DIM,
            OLMOCR_TARGET_LONGEST_IMAGE_DIM,
        )
    )


def olmocr_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Extract plain text and metadata from olmOCR YAML/markdown output."""
    return parse_olmocr_response(text)


def lfm2_5_vl_text_postprocessor(text: str) -> str:
    """Strip Liquid LFM2.5-VL chat scaffold and OCR wrapper tags."""
    prompt = getattr(LFM2_5_VL_1_6B_OCR_TEMPLATE, "user_prompt", None)
    cleaned = strip_leading_chat_scaffold(text, prompts=[prompt] if isinstance(prompt, str) else [])
    return strip_ocr_output_tag(cleaned, output_tag=DEFAULT_OCR_OUTPUT_TAG)


def infinity_parser_7b_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Normalize Infinity-Parser markdown output to plain text and preserve raw markdown."""
    cleaned = strip_leading_chat_scaffold(
        text,
        prompts=[
            INFINITY_PARSER_7B_OCR_PROMPT,
            INFINITY_PARSER_7B_SYSTEM_PROMPT,
        ],
    )
    raw_markdown = strip_outer_fenced_code_block(cleaned)
    return strip_rich_ocr_markup_to_plain_text(raw_markdown), {
        "raw_markdown": raw_markdown,
    }


def firered_ocr_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Normalize FireRed-OCR markdown output to plain text and preserve raw markdown."""
    cleaned = strip_leading_chat_scaffold(text, prompts=[FIRERED_OCR_OCR_PROMPT])
    for _ in range(8):
        previous = cleaned
        for token in ("<|im_end|>", "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"):
            if cleaned.endswith(token):
                cleaned = cleaned[: -len(token)].rstrip()
                break
        if cleaned == previous:
            break
    raw_markdown = strip_outer_fenced_code_block(cleaned)
    return strip_rich_ocr_markup_to_plain_text(raw_markdown), {
        "raw_markdown": raw_markdown,
    }


def nanonets_ocr2_3b_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Normalize Nanonets-OCR2 markdown output to plain text and preserve raw markdown."""
    cleaned = strip_leading_chat_scaffold(
        text,
        prompts=[
            NANONETS_OCR2_3B_SYSTEM_PROMPT,
            NANONETS_OCR2_3B_OCR_PROMPT,
        ],
    )
    for _ in range(8):
        previous = cleaned
        for token in ("<|im_end|>", "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"):
            if cleaned.endswith(token):
                cleaned = cleaned[: -len(token)].rstrip()
                break
        if cleaned == previous:
            break
    raw_markdown = strip_outer_fenced_code_block(cleaned)
    return strip_rich_ocr_markup_to_plain_text(raw_markdown), {
        "raw_markdown": raw_markdown,
    }


def deepseek_ocr_2_text_postprocessor(text: str) -> str:
    """Strip DeepSeek OCR 2 prompt echoes, chat scaffold, and trailing stop tokens."""
    cleaned = text.strip()
    stop_token = "<｜end▁of▁sentence｜>"
    while cleaned.endswith(stop_token):
        cleaned = cleaned[: -len(stop_token)].rstrip()
    cleaned = strip_leading_chat_scaffold(
        cleaned,
        prompts=[
            f"<image>\n{DEEPSEEK_OCR_2_OCR_PROMPT}",
            DEEPSEEK_OCR_2_OCR_PROMPT,
        ],
    )
    return cleaned.strip()


def glm_ocr_text_postprocessor(text: str) -> str:
    """Strip GLM-OCR prompt echoes, chat scaffold, and trailing special tokens."""
    cleaned = strip_leading_chat_scaffold(text, prompts=[GLM_OCR_OCR_PROMPT])
    for _ in range(8):
        previous = cleaned
        for token in ("<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"):
            if cleaned.endswith(token):
                cleaned = cleaned[: -len(token)].rstrip()
                break
        if cleaned == previous:
            break
    return cleaned.strip()


def glm_ocr_image_preprocessor(image: Image.Image) -> Image.Image:
    """Resize GLM-OCR inputs to stay within vLLM's encoder image-item budget."""
    prepared = prepare_ocr_image(image)
    width, height = prepared.size
    if width < GLM_OCR_IMAGE_GRID_SIZE or height < GLM_OCR_IMAGE_GRID_SIZE:
        return prepared

    rounded_width = round(width / GLM_OCR_IMAGE_GRID_SIZE) * GLM_OCR_IMAGE_GRID_SIZE
    rounded_height = round(height / GLM_OCR_IMAGE_GRID_SIZE) * GLM_OCR_IMAGE_GRID_SIZE
    if GLM_OCR_TEMPORAL_PATCH_SIZE * rounded_width * rounded_height <= GLM_OCR_VLLM_MAX_PIXELS:
        return prepared

    scale = math.sqrt((GLM_OCR_TEMPORAL_PATCH_SIZE * width * height) / GLM_OCR_VLLM_MAX_PIXELS)
    target_width = max(
        GLM_OCR_IMAGE_GRID_SIZE,
        math.floor(width / scale / GLM_OCR_IMAGE_GRID_SIZE) * GLM_OCR_IMAGE_GRID_SIZE,
    )
    target_height = max(
        GLM_OCR_IMAGE_GRID_SIZE,
        math.floor(height / scale / GLM_OCR_IMAGE_GRID_SIZE) * GLM_OCR_IMAGE_GRID_SIZE,
    )
    return prepared.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)


def paddleocr_vl_text_postprocessor(text: str) -> str:
    """Strip PaddleOCR-VL prompt echoes and leading chat scaffold from OCR output."""
    return strip_leading_chat_scaffold(text, prompts=[PADDLEOCR_VL_1_5_OCR_PROMPT])


def chandra_image_preprocessor(image: Image.Image) -> Image.Image:
    """Resize an image using Chandra OCR 2's pixel-budget and 28px-grid scaling."""
    width, height = image.size
    if width <= 0 or height <= 0:
        return ensure_rgb(image)

    max_pixels = CHANDRA_MAX_IMAGE_SIZE[0] * CHANDRA_MAX_IMAGE_SIZE[1]
    min_pixels = CHANDRA_MIN_IMAGE_SIZE[0] * CHANDRA_MIN_IMAGE_SIZE[1]
    current_pixels = width * height
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    original_aspect_ratio = width / height
    width_blocks = max(1, round((width * scale) / CHANDRA_IMAGE_GRID_SIZE))
    height_blocks = max(1, round((height * scale) / CHANDRA_IMAGE_GRID_SIZE))

    while (width_blocks * height_blocks * CHANDRA_IMAGE_GRID_SIZE**2) > max_pixels:
        if width_blocks == 1 and height_blocks == 1:
            break
        if width_blocks == 1:
            height_blocks -= 1
            continue
        if height_blocks == 1:
            width_blocks -= 1
            continue

        width_loss = abs(((width_blocks - 1) / height_blocks) - original_aspect_ratio)
        height_loss = abs((width_blocks / (height_blocks - 1)) - original_aspect_ratio)
        if width_loss < height_loss:
            width_blocks -= 1
        else:
            height_blocks -= 1

    new_size = (
        width_blocks * CHANDRA_IMAGE_GRID_SIZE,
        height_blocks * CHANDRA_IMAGE_GRID_SIZE,
    )
    if new_size == (width, height):
        return ensure_rgb(image)
    return ensure_rgb(image.resize(new_size, resample=Image.Resampling.LANCZOS))


def chandra_text_postprocessor(text: str) -> TextPostprocessorResult:
    """Extract plain text and metadata from Chandra OCR 2 HTML-layout output."""
    return parse_chandra_response(text)

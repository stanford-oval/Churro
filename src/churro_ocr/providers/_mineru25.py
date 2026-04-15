"""Helpers for MinerU2.5 two-step OCR pipelines."""

from __future__ import annotations

import asyncio
import base64
import html
import itertools
import math
import random
import re
from dataclasses import dataclass, replace
from io import BytesIO
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image, ImageDraw, ImageFont

from churro_ocr.providers._ocr_processing import strip_leading_chat_scaffold

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

MINERU2_5_LAYOUT_IMAGE_SIZE = (1_036, 1_036)
MINERU2_5_MIN_IMAGE_EDGE = 28
MINERU2_5_MAX_IMAGE_EDGE_RATIO = 50
MINERU2_5_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
MINERU2_5_PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}
_ANGLE_MAPPING: dict[str, Literal[0, 90, 180, 270]] = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}
_LAYOUT_RE = re.compile(
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?"
    r"(.*?)(?=<\|box_start\|>|$)",
    flags=re.DOTALL,
)
_TABLE_IMAGE_TOKEN_TEMPLATE = "[{idx}]"
_TABLE_IMAGE_TOKEN_LETTERS = "ACDGHKTWXYZ"
_TABLE_IMAGE_TOKEN_NUMBERS = "2345678"
_TABLE_IMAGE_TOKEN_LENGTH = 4
_TABLE_IMAGE_TOKEN_CHARS = _TABLE_IMAGE_TOKEN_LETTERS + _TABLE_IMAGE_TOKEN_NUMBERS
_TABLE_IMAGE_TOKEN_MAP_KEY = "_table_image_token_map"
_TABLE_IMAGE_ABSORBED_KEY = "_absorbed_by_table"
_FONT_PATH_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
_OTSL_NL = "<nl>"
_OTSL_FCEL = "<fcel>"
_OTSL_ECEL = "<ecel>"
_OTSL_LCEL = "<lcel>"
_OTSL_UCEL = "<ucel>"
_OTSL_XCEL = "<xcel>"
_OTSL_TOKENS = {_OTSL_NL, _OTSL_FCEL, _OTSL_ECEL, _OTSL_LCEL, _OTSL_UCEL, _OTSL_XCEL}
_OTSL_PATTERN = re.compile(
    "("
    + "|".join(
        re.escape(token) for token in (_OTSL_NL, _OTSL_FCEL, _OTSL_ECEL, _OTSL_LCEL, _OTSL_UCEL, _OTSL_XCEL)
    )
    + ")"
)
_IMAGE_ANALYSIS_TYPES = {"image", "chart"}
_IMAGE_CAPTION_CONTAINER_TYPES = {"image", "chart", "image_block"}
_INTERNAL_BLOCK_THRESHOLD = 0.9
_SUPPORTED_BLOCK_TYPES = {
    "text",
    "title",
    "table",
    "equation",
    "code",
    "algorithm",
    "aside_text",
    "ref_text",
    "phonetic",
    "list_item",
    "table_caption",
    "image_caption",
    "code_caption",
    "table_footnote",
    "image_footnote",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "image",
    "chart",
    "list",
    "image_block",
    "equation_block",
    "unknown",
}


def _attribute_error(message: str) -> AttributeError:
    return AttributeError(message)


def _runtime_error(message: str) -> RuntimeError:
    return RuntimeError(message)


def _type_error(message: str) -> TypeError:
    return TypeError(message)


def _value_error(message: str) -> ValueError:
    return ValueError(message)


@dataclass(slots=True, frozen=True)
class MinerU25SamplingParams:
    """Sampling parameters used by the MinerU2.5 two-step pipeline."""

    temperature: float | None = 0.0
    top_p: float | None = 0.01
    top_k: int | None = 1
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    repetition_penalty: float | None = 1.0
    no_repeat_ngram_size: int | None = 100
    max_new_tokens: int | None = None


DEFAULT_MINERU2_5_SAMPLING_PARAMS: dict[str, MinerU25SamplingParams] = {
    "table": MinerU25SamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": MinerU25SamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "image": MinerU25SamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "chart": MinerU25SamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": MinerU25SamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": MinerU25SamplingParams(),
}


class MinerU25ContentBlock(dict[str, object]):
    """Dictionary-backed content block compatible with MinerU-style postprocessing."""

    def __init__(
        self,
        type: str,
        bbox: list[float],
        angle: Literal[None, 0, 90, 180, 270] = None,
        content: str | None = None,
        merge_prev: bool = False,
    ) -> None:
        super().__init__()
        if type not in _SUPPORTED_BLOCK_TYPES:
            message = f"Unknown MinerU2.5 block type {type!r}."
            raise _value_error(message)
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            message = f"Invalid MinerU2.5 bbox {bbox!r}."
            raise _value_error(message)
        self["type"] = type
        self["bbox"] = bbox
        self["angle"] = angle
        self["content"] = content
        if type == "text":
            self["merge_prev"] = merge_prev

    @property
    def type(self) -> str:
        return str(self["type"])

    @type.setter
    def type(self, value: str) -> None:
        if value not in _SUPPORTED_BLOCK_TYPES:
            message = f"Unknown MinerU2.5 block type {value!r}."
            raise _value_error(message)
        merge_prev = self.get("merge_prev", False)
        self["type"] = value
        if value == "text":
            self["merge_prev"] = bool(merge_prev)
        else:
            self.pop("merge_prev", None)

    @property
    def bbox(self) -> list[float]:
        bbox = self["bbox"]
        if not isinstance(bbox, list):
            message = f"MinerU2.5 bbox payload must be a list, got {type(bbox).__name__}."
            raise _type_error(message)
        return [float(coord) for coord in cast("list[int | float]", bbox)]

    @bbox.setter
    def bbox(self, value: list[float]) -> None:
        self["bbox"] = value

    @property
    def angle(self) -> Literal[None, 0, 90, 180, 270]:
        return cast("Literal[None, 0, 90, 180, 270]", self["angle"])

    @angle.setter
    def angle(self, value: Literal[None, 0, 90, 180, 270]) -> None:
        self["angle"] = value

    @property
    def content(self) -> str | None:
        content = self.get("content")
        return None if content is None else str(content)

    @content.setter
    def content(self, value: str | None) -> None:
        self["content"] = value

    @property
    def merge_prev(self) -> bool:
        return bool(self.get("merge_prev", False))

    @merge_prev.setter
    def merge_prev(self, value: bool) -> None:
        if self.type != "text":
            message = "merge_prev is only valid for MinerU2.5 text blocks."
            raise _attribute_error(message)
        self["merge_prev"] = bool(value)


@dataclass(slots=True, frozen=True)
class _TableCell:
    text: str
    row_span: int
    col_span: int
    start_row: int
    start_col: int


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_PATH_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _get_optimal_pil_font(
    text: str,
    box_w: int,
    box_h: int,
    *,
    fill_ratio: float = 0.7,
    min_size: int = 4,
    max_size: int = 256,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
    left, right = min_size, max_size
    best_font = _load_font(left)
    best_w = 0
    best_h = 0
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for _ in range(30):
        if left > right:
            break
        mid = (left + right) // 2
        font = _load_font(mid)
        bbox = measure_draw.textbbox((0, 0), text, font=font)
        width = int(bbox[2] - bbox[0])
        height = int(bbox[3] - bbox[1])
        if width <= box_w * fill_ratio and height <= box_h * fill_ratio:
            best_font = font
            best_w = width
            best_h = height
            left = mid + 1
        else:
            right = mid - 1
    return best_font, best_w, best_h


def _pil_image_to_jpg_data_uri(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _normalize_rotation_angle(angle: int | None) -> int:
    return angle if angle in {90, 180, 270} else 0


def _rotate_image_by_angle(image: Image.Image, angle: int | None) -> Image.Image:
    normalized_angle = _normalize_rotation_angle(angle)
    if normalized_angle == 0:
        return image
    return image.rotate(normalized_angle, expand=True)


def _rotate_box_in_image(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    angle: int | None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = image_size
    normalized_angle = _normalize_rotation_angle(angle)
    if normalized_angle == 0:
        return box
    if normalized_angle == 90:
        return (y1, width - x2, y2, width - x1)
    if normalized_angle == 180:
        return (width - x2, height - y2, width - x1, height - y1)
    return (height - y2, x1, height - y1, x2)


def _get_average_color(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    left, upper, right, lower = box
    width, height = image.size
    pad = 2
    mid_x = (left + right) // 2
    mid_y = (upper + lower) // 2
    points = [
        (left - pad, upper - pad),
        (mid_x, upper - pad),
        (right + pad, upper - pad),
        (right + pad, mid_y),
        (right + pad, lower + pad),
        (mid_x, lower + pad),
        (left - pad, lower + pad),
        (left - pad, mid_y),
    ]
    pixels: list[tuple[int, int, int]] = []
    for px, py in points:
        px = max(0, min(int(px), width - 1))
        py = max(0, min(int(py), height - 1))
        pixel = image.getpixel((px, py))
        if isinstance(pixel, int):
            pixels.append((pixel, pixel, pixel))
            continue
        pixel_channels = cast("tuple[int, ...]", pixel)
        if len(pixel_channels) >= 3:
            pixels.append(
                (
                    int(pixel_channels[0]),
                    int(pixel_channels[1]),
                    int(pixel_channels[2]),
                )
            )
            continue
        if pixel_channels:
            channel = int(pixel_channels[0])
            pixels.append((channel, channel, channel))
    if not pixels:
        return (255, 255, 255)
    return (
        sum(pixel[0] for pixel in pixels) // len(pixels),
        sum(pixel[1] for pixel in pixels) // len(pixels),
        sum(pixel[2] for pixel in pixels) // len(pixels),
    )


def _get_contrast_text_color(bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = bg_color
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _bbox_area(a: list[float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _bbox_cover_ratio(inner: list[float], outer: list[float]) -> float:
    inner_area = _bbox_area(inner)
    if inner_area == 0.0:
        return 0.0
    return _bbox_intersection_area(inner, outer) / inner_area


def _generate_uid(length: int = _TABLE_IMAGE_TOKEN_LENGTH) -> str:
    return "".join(random.choices(_TABLE_IMAGE_TOKEN_CHARS, k=length))


def _build_table_image_map(
    blocks: list[MinerU25ContentBlock],
    *,
    threshold: float = 0.9,
    table_indices: list[int] | None = None,
) -> dict[int, list[int]]:
    if table_indices is None:
        table_indices = [index for index, block in enumerate(blocks) if block.type == "table"]
    table_to_images = {table_index: [] for table_index in table_indices}
    if not table_indices:
        return table_to_images

    for image_index, block in enumerate(blocks):
        if block.type != "image":
            continue
        best_table_index: int | None = None
        best_ratio = threshold
        best_area: float | None = None
        for table_index in table_indices:
            table_block = blocks[table_index]
            ratio = _bbox_cover_ratio(block.bbox, table_block.bbox)
            if ratio < threshold:
                continue
            area = _bbox_area(table_block.bbox)
            if (
                best_table_index is None
                or ratio > best_ratio
                or (ratio == best_ratio and best_area is not None and area < best_area)
            ):
                best_table_index = table_index
                best_ratio = ratio
                best_area = area
        if best_table_index is not None:
            table_to_images[best_table_index].append(image_index)

    for image_indices in table_to_images.values():
        image_indices.sort(key=lambda image_index: (blocks[image_index].bbox[1], blocks[image_index].bbox[0]))
    return table_to_images


def _mark_absorbed_table_images(blocks: list[MinerU25ContentBlock], image_indices: list[int]) -> None:
    for image_index in image_indices:
        blocks[image_index][_TABLE_IMAGE_ABSORBED_KEY] = True


def _is_absorbed_table_image(block: MinerU25ContentBlock) -> bool:
    return bool(block.get(_TABLE_IMAGE_ABSORBED_KEY))


def _replace_table_image_tokens(content: str | None, token_map: dict[str, str] | None) -> str | None:
    if not content or not token_map:
        return content
    for token, data_uri in token_map.items():
        token_inner = token[1:-1]
        pattern = r"\[\s*" + re.escape(token_inner) + r"\s*\]"
        content = re.sub(pattern, f'<img src="{data_uri}"/>', content)
    return content


def _replace_table_formula_delimiters(content: str | None, *, enabled: bool) -> str | None:
    if not enabled or not content:
        return content

    inline_pattern = re.compile(r"\\\((.+?)\\\)", flags=re.DOTALL)
    block_pattern = re.compile(r"\\\[(.+?)\\\]", flags=re.DOTALL)
    eq_tag_pattern = re.compile(r"(<eq>.*?</eq>)", flags=re.DOTALL)

    def _wrap_formula(pattern: re.Pattern[str], text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            inner_content = match.group(1).strip()
            return f"<eq>{inner_content}</eq>"

        return pattern.sub(_replace, text)

    parts = eq_tag_pattern.split(content)
    for index, part in enumerate(parts):
        if not part or eq_tag_pattern.fullmatch(part):
            continue
        part = _wrap_formula(inline_pattern, part)
        part = _wrap_formula(block_pattern, part)
        parts[index] = part
    return "".join(parts)


def _cleanup_table_image_metadata(blocks: list[MinerU25ContentBlock]) -> list[MinerU25ContentBlock]:
    for block in blocks:
        block.pop(_TABLE_IMAGE_TOKEN_MAP_KEY, None)
        block.pop(_TABLE_IMAGE_ABSORBED_KEY, None)
    return blocks


def _mask_and_encode_table_image(
    page_image: Image.Image,
    table_block: MinerU25ContentBlock,
    image_entries: list[tuple[int, MinerU25ContentBlock]],
    table_image: Image.Image,
) -> tuple[Image.Image, dict[str, str]]:
    width, height = page_image.size
    x1_t, y1_t, _, _ = table_block.bbox
    abs_x1_t = int(x1_t * width)
    abs_y1_t = int(y1_t * height)
    original_table_size = table_image.size
    masked_table_image = _rotate_image_by_angle(table_image.copy(), table_block.angle)
    draw = ImageDraw.Draw(masked_table_image)
    token_map: dict[str, str] = {}
    used_token_codes: set[str] = set()
    max_token_count = len(_TABLE_IMAGE_TOKEN_CHARS) ** _TABLE_IMAGE_TOKEN_LENGTH
    font_cache: dict[tuple[int, int], tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]] = {}

    def _font_for_box(
        box_w: int,
        box_h: int,
        token_text: str,
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
        bucket_height = int(box_h // 16)
        key = (bucket_height, len(token_text))
        cached = font_cache.get(key)
        if cached is not None and cached[1] <= box_w and cached[2] <= box_h:
            return cached
        resolved = _get_optimal_pil_font(
            token_text,
            box_w,
            box_h,
            fill_ratio=0.7,
            min_size=4,
            max_size=max(100, int(box_h * 0.7)),
        )
        font_cache[key] = resolved
        return resolved

    for _, image_block in image_entries:
        ix1, iy1, ix2, iy2 = image_block.bbox
        abs_ix1 = ix1 * width
        abs_iy1 = iy1 * height
        abs_ix2 = ix2 * width
        abs_iy2 = iy2 * height

        rel_x1 = int(max(0, abs_ix1 - abs_x1_t))
        rel_y1 = int(max(0, abs_iy1 - abs_y1_t))
        rel_x2 = int(min(original_table_size[0], abs_ix2 - abs_x1_t))
        rel_y2 = int(min(original_table_size[1], abs_iy2 - abs_y1_t))
        if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
            continue

        crop_box = (int(abs_ix1), int(abs_iy1), int(abs_ix2), int(abs_iy2))
        crop_image = page_image.crop(crop_box)
        if crop_image.width < 1 or crop_image.height < 1:
            continue

        if len(used_token_codes) >= max_token_count:
            message = "Exhausted MinerU2.5 table image token space."
            raise _runtime_error(message)

        while True:
            token_code = _generate_uid()
            if token_code not in used_token_codes:
                used_token_codes.add(token_code)
                break

        token_text = _TABLE_IMAGE_TOKEN_TEMPLATE.format(idx=token_code)
        rotated_crop_image = _rotate_image_by_angle(crop_image, table_block.angle)
        token_map[token_text] = _pil_image_to_jpg_data_uri(rotated_crop_image)

        image_mask_bbox = _rotate_box_in_image(
            (rel_x1, rel_y1, rel_x2, rel_y2),
            original_table_size,
            table_block.angle,
        )
        average_color = _get_average_color(masked_table_image, image_mask_bbox)
        draw.rectangle(image_mask_bbox, fill=average_color, outline=None)

        box_w = image_mask_bbox[2] - image_mask_bbox[0]
        box_h = image_mask_bbox[3] - image_mask_bbox[1]
        font, text_w, text_h = _font_for_box(box_w, box_h, token_text)
        if text_w <= box_w and text_h <= box_h:
            center_x = image_mask_bbox[0] + box_w / 2
            center_y = image_mask_bbox[1] + box_h / 2
            text_position = (center_x - text_w / 2, center_y - text_h / 2)
            text_color = _get_contrast_text_color(average_color)
            draw.text(text_position, token_text, fill=text_color, font=font)

    return masked_table_image, token_map


def _convert_bbox(raw_bbox: tuple[str, str, str, str]) -> list[float] | None:
    x1, y1, x2, y2 = map(int, raw_bbox)
    if any(coord < 0 or coord > 1_000 for coord in (x1, y1, x2, y2)):
        return None
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [value / 1_000.0 for value in (x1, y1, x2, y2)]


def _parse_angle(token: str | None) -> Literal[None, 0, 90, 180, 270]:
    if token is None:
        return None
    return _ANGLE_MAPPING.get(token)


def _parse_merge_prev(tail: str) -> bool:
    return "txt_contd_tgt" in tail


def _get_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def _resize_image_by_need(
    image: Image.Image,
    *,
    min_image_edge: int,
    max_image_edge_ratio: float,
) -> Image.Image:
    edge_ratio = max(image.size) / min(image.size)
    if edge_ratio > max_image_edge_ratio:
        width, height = image.size
        if width > height:
            new_width, new_height = width, math.ceil(width / max_image_edge_ratio)
        else:
            new_width, new_height = math.ceil(height / max_image_edge_ratio), height
        padded = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
        padded.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
        image = padded
    if min(image.size) < min_image_edge:
        scale = min_image_edge / min(image.size)
        image = image.resize(
            (math.ceil(image.width * scale), math.ceil(image.height * scale)),
            Image.Resampling.BICUBIC,
        )
    return image


def _trim_stop_strings(text: str) -> str:
    cleaned = text
    for stop in MINERU2_5_STOP_TOKENS:
        cleaned = cleaned.split(stop, 1)[0]
    return cleaned.strip()


def _extract_otsl_tokens_and_text(raw_text: str) -> tuple[list[str], list[str]]:
    tokens = _OTSL_PATTERN.findall(raw_text)
    text_parts = [part for part in _OTSL_PATTERN.split(raw_text) if part and part.strip()]
    return tokens, text_parts


def _count_span_right(rows: list[list[str]], row_idx: int, col_idx: int, span_tokens: set[str]) -> int:
    span = 0
    cursor = col_idx
    while cursor < len(rows[row_idx]) and rows[row_idx][cursor] in span_tokens:
        span += 1
        cursor += 1
    return span


def _count_span_down(rows: list[list[str]], row_idx: int, col_idx: int, span_tokens: set[str]) -> int:
    span = 0
    cursor = row_idx
    while cursor < len(rows) and col_idx < len(rows[cursor]) and rows[cursor][col_idx] in span_tokens:
        span += 1
        cursor += 1
    return span


def _group_otsl_rows(tokens: list[str]) -> list[list[str]]:
    return [
        list(group)
        for is_newline, group in itertools.groupby(tokens, lambda item: item == _OTSL_NL)
        if not is_newline
    ]


def _pad_otsl_rows(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    max_cols = max(len(row) for row in rows)
    padded_rows = [row + ([_OTSL_ECEL] * (max_cols - len(row))) for row in rows]
    return padded_rows, max_cols


def _normalize_otsl_parts(rows: list[list[str]], mixed_texts: list[str]) -> list[str]:
    normalized_parts: list[str] = []
    text_idx = 0
    for row in rows:
        for token in row:
            normalized_parts.append(token)
            if text_idx < len(mixed_texts) and mixed_texts[text_idx] == token:
                text_idx += 1
                if text_idx < len(mixed_texts) and mixed_texts[text_idx] not in _OTSL_TOKENS:
                    normalized_parts.append(mixed_texts[text_idx])
                    text_idx += 1
        normalized_parts.append(_OTSL_NL)
        if text_idx < len(mixed_texts) and mixed_texts[text_idx] == _OTSL_NL:
            text_idx += 1
    return normalized_parts


def _cell_text_and_offset(parts: list[str], index: int) -> tuple[str, int]:
    next_index = index + 1
    if next_index < len(parts) and parts[next_index] not in _OTSL_TOKENS:
        return parts[next_index].strip(), 2
    return "", 1


def _next_otsl_right_token(parts: list[str], *, index: int, next_offset: int) -> str:
    next_index = index + next_offset
    return parts[next_index] if next_index < len(parts) else ""


def _next_otsl_down_token(rows: list[list[str]], *, row_idx: int, col_idx: int) -> str:
    if row_idx + 1 >= len(rows) or col_idx >= len(rows[row_idx + 1]):
        return ""
    return rows[row_idx + 1][col_idx]


def _otsl_cell_spans(
    rows: list[list[str]],
    parts: list[str],
    *,
    row_idx: int,
    col_idx: int,
    index: int,
    next_offset: int,
) -> tuple[int, int]:
    row_span = 1
    col_span = 1
    next_right = _next_otsl_right_token(parts, index=index, next_offset=next_offset)
    if next_right in {_OTSL_LCEL, _OTSL_XCEL}:
        col_span += _count_span_right(rows, row_idx, col_idx + 1, {_OTSL_LCEL, _OTSL_XCEL})
    next_down = _next_otsl_down_token(rows, row_idx=row_idx, col_idx=col_idx)
    if next_down in {_OTSL_UCEL, _OTSL_XCEL}:
        row_span += _count_span_down(rows, row_idx + 1, col_idx, {_OTSL_UCEL, _OTSL_XCEL})
    return row_span, col_span


def _collect_otsl_cells(rows: list[list[str]], parts: list[str]) -> list[_TableCell]:
    cells: list[_TableCell] = []
    row_idx = 0
    col_idx = 0
    for index, part in enumerate(parts):
        if part in {_OTSL_FCEL, _OTSL_ECEL}:
            cell_text, next_offset = _cell_text_and_offset(parts, index)
            row_span, col_span = _otsl_cell_spans(
                rows,
                parts,
                row_idx=row_idx,
                col_idx=col_idx,
                index=index,
                next_offset=next_offset,
            )
            cells.append(
                _TableCell(
                    text=cell_text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row=row_idx,
                    start_col=col_idx,
                )
            )
        if part in _OTSL_TOKENS - {_OTSL_NL}:
            col_idx += 1
        if part == _OTSL_NL:
            row_idx += 1
            col_idx = 0
    return cells


def _render_otsl_html(rows: list[list[str]], *, max_cols: int, cells: list[_TableCell]) -> str:
    cells_by_position = {(cell.start_row, cell.start_col): cell for cell in cells}
    html_parts = ["<table>"]
    for row in range(len(rows)):
        html_parts.append("<tr>")
        for col in range(max_cols):
            cell = cells_by_position.get((row, col))
            if cell is None:
                continue
            attrs: list[str] = []
            if cell.row_span > 1:
                attrs.append(f' rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                attrs.append(f' colspan="{cell.col_span}"')
            html_parts.append(f"<td{''.join(attrs)}>{html.escape(cell.text)}</td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")
    return "".join(html_parts)


def convert_mineru2_5_otsl_to_html(otsl_content: str) -> str:
    """Convert a MinerU2.5 OTSL table prediction to HTML."""
    if otsl_content.startswith("<table") and otsl_content.endswith("</table>"):
        return otsl_content
    tokens, mixed_texts = _extract_otsl_tokens_and_text(otsl_content)
    rows = _group_otsl_rows(tokens)
    if not rows:
        return otsl_content.strip()
    rows, max_cols = _pad_otsl_rows(rows)
    normalized_parts = _normalize_otsl_parts(rows, mixed_texts)
    cells = _collect_otsl_cells(rows, normalized_parts)
    return _render_otsl_html(rows, max_cols=max_cols, cells=cells)


def wrap_mineru2_5_equation(content: str) -> str:
    """Wrap a MinerU2.5 formula prediction as display math."""
    cleaned = content.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("\\["):
        cleaned = cleaned[2:].strip()
    if cleaned.endswith("\\]"):
        cleaned = cleaned[:-2].strip()
    return f"\\[\n{cleaned}\n\\]"


def _try_fix_equation_delimiters(latex: str) -> str:
    cleaned = latex.strip()
    if cleaned.startswith("\\["):
        cleaned = cleaned[2:]
    if cleaned.endswith("\\]"):
        cleaned = cleaned[:-2]
    return cleaned.strip()


def _try_convert_display_to_inline(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if re.fullmatch(r"[–\d\-,\s]+", inner):
            return r"\[" + inner + r"\]"
        return r"\(" + inner + r"\)"

    return re.sub(r"\\\[(.*?)\\\]", _replace, text, flags=re.DOTALL)


def _try_fix_macro_spacing_in_markdown(text: str) -> str:
    known_macros = {r"\top", r"\int", r"\inf"}
    target_macros = [r"\cong", r"\to", r"\times", r"\subset", r"\in"]

    def _fix_macro_spacing(value: str, macro: str) -> str:
        pattern = re.escape(macro) + r"([a-zA-Z])(?![a-zA-Z])"

        def _replace(match: re.Match[str]) -> str:
            letter = match.group(1)
            if (macro + letter) in known_macros:
                return match.group(0)
            return macro + " " + letter

        return re.sub(pattern, _replace, value)

    result: list[str] = []
    parts = re.split(r"(\\\(.*?\\\))", text, flags=re.DOTALL)
    for part in parts:
        if part.startswith(r"\(") and part.endswith(r"\)"):
            inner = part[2:-2]
            for macro in target_macros:
                inner = _fix_macro_spacing(inner, macro)
            result.append(r"\(" + inner + r"\)")
            continue
        result.append(part)
    return "".join(result)


def _try_move_underscores_outside(text: str) -> str:
    def _process_match(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = re.split(r"(_{3,})", inner)
        if len(parts) == 1:
            return match.group(0)
        result: list[str] = []
        for part in parts:
            if re.fullmatch(r"_{3,}", part):
                result.append(part)
            elif part.strip():
                result.append(r"\(" + part + r"\)")
        return " ".join(result)

    return re.sub(r"\\\((.+?)\\\)", _process_match, text, flags=re.DOTALL)


def _do_handle_equation_block(blocks: list[MinerU25ContentBlock]) -> list[MinerU25ContentBlock]:
    equation_block_indices = [index for index, block in enumerate(blocks) if block.type == "equation_block"]
    equation_indices = [index for index, block in enumerate(blocks) if block.type == "equation"]
    combined_indices: dict[int, list[int]] = {}
    for block_index in equation_block_indices:
        covered = [
            equation_index
            for equation_index in equation_indices
            if _bbox_cover_ratio(blocks[block_index].bbox, blocks[equation_index].bbox) > 0.9
        ]
        if len(covered) > 1:
            combined_indices[block_index] = covered

    combined_equation_indices = {index for indices in combined_indices.values() for index in indices}
    rendered_blocks: list[MinerU25ContentBlock] = []
    for index, block in enumerate(blocks):
        if index in combined_equation_indices:
            continue
        if index in combined_indices:
            contents = [blocks[covered_index].content or "" for covered_index in combined_indices[index]]
            tag_count = sum(len(re.findall(r"\\tag\s*\{[^}]*\}", content)) for content in contents)
            if tag_count > 1:
                contents = [re.sub(r"\\tag\s*\{([^}]*)\}", r"(\1)", content) for content in contents]
            combined_content = (
                "\\begin{array}{l} "
                + " \\\\ ".join(content.strip() for content in contents)
                + " \\end{array}"
            )
            rendered_blocks.append(
                MinerU25ContentBlock(
                    type="equation",
                    bbox=block.bbox,
                    angle=block.angle,
                    content=combined_content,
                )
            )
            continue
        if block.type == "equation_block":
            continue
        rendered_blocks.append(block)
    return rendered_blocks


def json2md(blocks: list[MinerU25ContentBlock]) -> str:
    """Render MinerU2.5 blocks to markdown-like text."""
    content_list: list[str] = []
    last_text_contd_index = -1
    for block in blocks:
        content = block.content
        if not content:
            continue
        if block.merge_prev and last_text_contd_index >= 0:
            if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", content) is not None:
                content_list[last_text_contd_index] += content
            else:
                content_list[last_text_contd_index] += " " + content
            continue
        content_list.append(content)
        if block.type == "text":
            last_text_contd_index = len(content_list) - 1
    return "\n\n".join(content_list).strip()


@dataclass(slots=True)
class MinerU25PipelineHelper:
    """Shared MinerU2.5 layout, extraction, and markdown postprocessing helper."""

    prompts: dict[str, str]
    system_prompt: str
    sampling_params: dict[str, MinerU25SamplingParams] | None = None
    layout_image_size: tuple[int, int] = MINERU2_5_LAYOUT_IMAGE_SIZE
    min_image_edge: int = MINERU2_5_MIN_IMAGE_EDGE
    max_image_edge_ratio: float = MINERU2_5_MAX_IMAGE_EDGE_RATIO
    simple_post_process: bool = False
    handle_equation_block: bool = True
    abandon_list: bool = False
    abandon_paratext: bool = False
    image_analysis: bool = False
    enable_table_formula_eq_wrap: bool = False

    def __post_init__(self) -> None:
        self.prompts = dict(self.prompts)
        merged_sampling_params = dict(DEFAULT_MINERU2_5_SAMPLING_PARAMS)
        if self.sampling_params is not None:
            merged_sampling_params.update(self.sampling_params)
        self.sampling_params = merged_sampling_params

    def prompt_for(self, step_key: str) -> str:
        return self.prompts.get(step_key) or self.prompts["[default]"]

    def sampling_for(self, step_key: str) -> MinerU25SamplingParams:
        sampling_params = self.sampling_params or DEFAULT_MINERU2_5_SAMPLING_PARAMS
        return sampling_params.get(step_key) or sampling_params["[default]"]

    def clean_response(self, text: str, *, step_key: str) -> str:
        cleaned = _trim_stop_strings(text)
        return strip_leading_chat_scaffold(
            cleaned,
            prompts=[self.system_prompt, self.prompt_for(step_key), self.prompt_for(step_key).strip()],
        )

    def prepare_for_layout(self, image: Image.Image) -> Image.Image:
        image = _get_rgb_image(image)
        return image.resize(self.layout_image_size, Image.Resampling.BICUBIC)

    def parse_layout_output(self, output: str) -> list[MinerU25ContentBlock]:
        blocks: list[MinerU25ContentBlock] = []
        for match in re.finditer(_LAYOUT_RE, output):
            x1, y1, x2, y2, ref_type, rotate_token, tail = match.groups()
            bbox = _convert_bbox((x1, y1, x2, y2))
            if bbox is None:
                continue
            ref_type = ref_type.lower()
            if ref_type == "inline_formula" or ref_type not in _SUPPORTED_BLOCK_TYPES:
                continue
            angle = _parse_angle(rotate_token)
            if ref_type == "text":
                blocks.append(
                    MinerU25ContentBlock(
                        ref_type,
                        bbox,
                        angle=angle,
                        merge_prev=_parse_merge_prev(tail),
                    )
                )
                continue
            blocks.append(MinerU25ContentBlock(ref_type, bbox, angle=angle))
        return blocks

    def _find_covered_block_indices(
        self,
        blocks: list[MinerU25ContentBlock],
        *,
        candidate_types: set[str],
        container_types: set[str],
        threshold: float = _INTERNAL_BLOCK_THRESHOLD,
    ) -> set[int]:
        container_indices = [idx for idx, block in enumerate(blocks) if block.type in container_types]
        if not container_indices:
            return set()
        covered_indices: set[int] = set()
        for idx, block in enumerate(blocks):
            if block.type not in candidate_types:
                continue
            for container_idx in container_indices:
                if idx == container_idx:
                    continue
                if _bbox_cover_ratio(block.bbox, blocks[container_idx].bbox) >= threshold:
                    covered_indices.add(idx)
                    break
        return covered_indices

    def _prepare_block_image(
        self,
        page_image: Image.Image,
        block: MinerU25ContentBlock,
    ) -> Image.Image:
        image = _get_rgb_image(page_image)
        width, height = image.size
        left = max(0, min(width - 1, math.floor(block.bbox[0] * width)))
        top = max(0, min(height - 1, math.floor(block.bbox[1] * height)))
        right = max(left + 1, min(width, math.ceil(block.bbox[2] * width)))
        bottom = max(top + 1, min(height, math.ceil(block.bbox[3] * height)))
        cropped = image.crop((left, top, right, bottom))
        if block.angle in {90, 180, 270}:
            cropped = cropped.rotate(block.angle, expand=True)
        return _resize_image_by_need(
            cropped,
            min_image_edge=self.min_image_edge,
            max_image_edge_ratio=self.max_image_edge_ratio,
        )

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[MinerU25ContentBlock],
        *,
        not_extract_list: list[str] | None = None,
    ) -> list[tuple[int, Image.Image]]:
        internal_caption_indices = self._find_covered_block_indices(
            blocks,
            candidate_types={"image_caption"},
            container_types=_IMAGE_CAPTION_CONTAINER_TYPES,
        )
        if internal_caption_indices:
            blocks[:] = [block for idx, block in enumerate(blocks) if idx not in internal_caption_indices]

        skip_types = {"list", "equation_block", "image_block"}
        if not self.image_analysis:
            skip_types.update(_IMAGE_ANALYSIS_TYPES)
        if not_extract_list is not None:
            skip_types.update(not_extract_list)

        table_indices = [
            idx for idx, block in enumerate(blocks) if block.type == "table" and block.type not in skip_types
        ]
        table_to_images = _build_table_image_map(blocks, table_indices=table_indices)
        absorbed_image_indices = sorted(
            {image_idx for image_indices in table_to_images.values() for image_idx in image_indices}
        )
        _mark_absorbed_table_images(blocks, absorbed_image_indices)

        prepared: list[tuple[int, Image.Image]] = []
        rgb_image = _get_rgb_image(image)
        width, height = rgb_image.size
        for index, block in enumerate(blocks):
            if block.type in skip_types:
                continue
            if block.type == "image" and _is_absorbed_table_image(block):
                continue
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = rgb_image.crop(scaled_bbox)
            if block_image.width < 1 or block_image.height < 1:
                continue
            if block.type == "table":
                image_entries = [
                    (image_idx, blocks[image_idx]) for image_idx in table_to_images.get(index, [])
                ]
                block_image, token_map = _mask_and_encode_table_image(
                    rgb_image,
                    block,
                    image_entries,
                    block_image,
                )
                if token_map:
                    block[_TABLE_IMAGE_TOKEN_MAP_KEY] = token_map
            elif block.angle in {90, 180, 270}:
                block_image = block_image.rotate(block.angle, expand=True)
            block_image = _resize_image_by_need(
                block_image,
                min_image_edge=self.min_image_edge,
                max_image_edge_ratio=self.max_image_edge_ratio,
            )
            prepared.append((index, block_image))
        return prepared

    def post_process(self, blocks: list[MinerU25ContentBlock]) -> list[MinerU25ContentBlock]:
        for block in blocks:
            content = (block.content or "").strip()
            if not content:
                block.content = None
                continue
            if block.type == "table":
                token_map_value = block.get(_TABLE_IMAGE_TOKEN_MAP_KEY)
                token_map = token_map_value if isinstance(token_map_value, dict) else None
                table_html = convert_mineru2_5_otsl_to_html(content)
                table_html = _replace_table_image_tokens(
                    table_html,
                    cast("dict[str, str] | None", token_map),
                )
                block.content = _replace_table_formula_delimiters(
                    table_html,
                    enabled=self.enable_table_formula_eq_wrap,
                )
                continue
            if block.type == "equation":
                fixed = _try_fix_equation_delimiters(content)
                block.content = wrap_mineru2_5_equation(fixed)
                continue
            if block.type == "text":
                fixed = _try_convert_display_to_inline(content)
                fixed = _try_fix_macro_spacing_in_markdown(fixed)
                fixed = _try_move_underscores_outside(fixed)
                block.content = fixed

        processed_blocks = blocks
        if not self.simple_post_process and self.handle_equation_block:
            processed_blocks = _do_handle_equation_block(processed_blocks)

        rendered_blocks: list[MinerU25ContentBlock] = []
        for block in processed_blocks:
            if block.type == "equation_block":
                continue
            if block.type == "image" and _is_absorbed_table_image(block):
                continue
            if self.abandon_list and block.type == "list":
                continue
            if self.abandon_paratext and block.type in MINERU2_5_PARATEXT_TYPES:
                continue
            rendered_blocks.append(block)
        return _cleanup_table_image_metadata(rendered_blocks)

    def render_markdown(self, blocks: list[MinerU25ContentBlock]) -> str:
        return json2md(blocks)

    def run_two_step(
        self,
        image: Image.Image,
        *,
        infer_step: Callable[[Image.Image, str, MinerU25SamplingParams], str],
        not_extract_list: list[str] | None = None,
    ) -> tuple[str, list[MinerU25ContentBlock], dict[str, float | int]]:
        from time import perf_counter

        started_at = perf_counter()
        layout_started_at = perf_counter()
        layout_output = infer_step(
            self.prepare_for_layout(image),
            "[layout]",
            self.sampling_for("[layout]"),
        )
        layout_finished_at = perf_counter()
        layout_blocks = self.parse_layout_output(layout_output)
        extract_started_at = perf_counter()
        prepared_blocks = self.prepare_for_extract(
            image,
            layout_blocks,
            not_extract_list=not_extract_list,
        )
        for index, block_image in prepared_blocks:
            layout_blocks[index].content = infer_step(
                block_image,
                layout_blocks[index].type,
                self.sampling_for(layout_blocks[index].type),
            )
        processed_blocks = self.post_process(layout_blocks)
        markdown = self.render_markdown(processed_blocks)
        finished_at = perf_counter()
        return (
            markdown,
            processed_blocks,
            {
                "layout_elapsed": layout_finished_at - layout_started_at,
                "extract_elapsed": finished_at - extract_started_at,
                "num_blocks": len(processed_blocks),
                "total_elapsed": finished_at - started_at,
            },
        )

    async def arun_two_step(
        self,
        image: Image.Image,
        *,
        infer_step: Callable[[Image.Image, str, MinerU25SamplingParams], Awaitable[str]],
        not_extract_list: list[str] | None = None,
    ) -> tuple[str, list[MinerU25ContentBlock], dict[str, float | int]]:
        from time import perf_counter

        started_at = perf_counter()
        layout_started_at = perf_counter()
        layout_output = await infer_step(
            self.prepare_for_layout(image),
            "[layout]",
            self.sampling_for("[layout]"),
        )
        layout_finished_at = perf_counter()
        layout_blocks = self.parse_layout_output(layout_output)
        extract_started_at = perf_counter()
        prepared_blocks = self.prepare_for_extract(
            image,
            layout_blocks,
            not_extract_list=not_extract_list,
        )
        for index, block_image in prepared_blocks:
            layout_blocks[index].content = await infer_step(
                block_image,
                layout_blocks[index].type,
                self.sampling_for(layout_blocks[index].type),
            )
        processed_blocks = await asyncio.to_thread(self.post_process, layout_blocks)
        markdown = self.render_markdown(processed_blocks)
        finished_at = perf_counter()
        return (
            markdown,
            processed_blocks,
            {
                "layout_elapsed": layout_finished_at - layout_started_at,
                "extract_elapsed": finished_at - extract_started_at,
                "num_blocks": len(processed_blocks),
                "total_elapsed": finished_at - started_at,
            },
        )


def replace_sampling_param(
    sampling: MinerU25SamplingParams,
    **changes: float | int | None,
) -> MinerU25SamplingParams:
    """Return a MinerU2.5 sampling config with selected fields replaced."""
    return replace(sampling, **changes)


__all__ = [
    "DEFAULT_MINERU2_5_SAMPLING_PARAMS",
    "MINERU2_5_LAYOUT_IMAGE_SIZE",
    "MINERU2_5_MAX_IMAGE_EDGE_RATIO",
    "MINERU2_5_MIN_IMAGE_EDGE",
    "MINERU2_5_PARATEXT_TYPES",
    "MINERU2_5_STOP_TOKENS",
    "MinerU25ContentBlock",
    "MinerU25PipelineHelper",
    "MinerU25SamplingParams",
    "convert_mineru2_5_otsl_to_html",
    "json2md",
    "replace_sampling_param",
    "wrap_mineru2_5_equation",
]

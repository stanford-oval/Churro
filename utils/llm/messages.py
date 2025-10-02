"""Message preparation and image encoding helpers for LLM calls."""

import base64
from io import BytesIO
import textwrap
from typing import Any, Dict, Optional, Tuple
import weakref

from PIL import Image

from utils.log_utils import logger
from utils.utils import resize_image_to_fit

from .types import ImageDetail, MessageContent, Messages


# Maximum dimensions for any image sent to the LLM (fits inside 2500 x 2500)
_MAX_IMAGE_DIM: int = 2500

# Encoding cache keyed by id(image) because PIL Image objects are unhashable.
# Value is a tuple of (weakref to image, per-format map). We verify the weakref
# still points to the same object before trusting the cached encodings. If the
# object is gone or the id has been reused for a new image, we rebuild.
_ENCODE_CACHE: dict[int, Tuple["weakref.ReferenceType[Image.Image]", Dict[str, str]]] = {}


def encode_image(image: Image.Image, format: str = "JPEG") -> str:
    """Encode PIL Image to base64 string with downscale guard & weakref cache.

    Steps:
      1. Downscale image in-place clone if it exceeds 2500x2500 (aspect preserved).
      2. Normalize for JPEG (RGB/L only).
      3. Consult weakref cache for previously encoded (image, format) pair.
      4. Encode with stable parameters and store in cache.
    """
    fmt = (format or "JPEG").upper()

    key = id(image)
    cache_entry = _ENCODE_CACHE.get(key)
    cache_bucket: Dict[str, str]
    if cache_entry is not None:
        img_ref, cache_bucket = cache_entry
        existing = img_ref()
        if existing is image:
            cached = cache_bucket.get(fmt)
            if cached is not None:
                logger.debug(f"Image served from encode cache ({fmt})")
                return cached
        else:
            # Stale entry (id reused or image collected); discard
            _ENCODE_CACHE.pop(key, None)
            cache_bucket = {}
    else:
        cache_bucket = {}

    img = image
    # Downscale guard (only if larger than box)
    if img.width > _MAX_IMAGE_DIM or img.height > _MAX_IMAGE_DIM:
        orig_w, orig_h = img.width, img.height
        img = resize_image_to_fit(img, _MAX_IMAGE_DIM, _MAX_IMAGE_DIM)
        logger.debug(
            "Downscaled image %sx%s -> %sx%s for format %s",
            orig_w,
            orig_h,
            img.width,
            img.height,
            fmt,
        )

    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buffer = BytesIO()
    save_kwargs: dict[str, Any] = {}
    if fmt == "JPEG":
        save_kwargs = {"quality": 95, "optimize": True}
    img.save(buffer, format=fmt, **save_kwargs)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    cache_bucket[fmt] = b64_str
    # Store / update cache entry
    _ENCODE_CACHE[key] = (weakref.ref(image), cache_bucket)
    return b64_str


def prepare_messages(
    system_prompt_text: Optional[str],
    user_message_text: Optional[str],
    user_message_image: Optional[Image.Image] | list[Image.Image],
    image_detail: Optional[ImageDetail] = None,
) -> Messages:
    """Prepare messages for LLM inference with optional images."""
    user_message_content: list[MessageContent] = []

    if user_message_image:
        if not isinstance(user_message_image, list):
            user_message_image = [user_message_image]

        for umi in user_message_image:
            if umi.height <= 0 or umi.width <= 0:
                logger.warning(f"Invalid image dimensions: {umi.width}x{umi.height}")
                continue

            # Detect image format, default to PNG if unknown
            image_format = umi.format or "PNG"
            if image_format not in ["PNG", "JPEG", "WEBP"]:
                logger.warning(f"Unsupported image format: {image_format}, defaulting to PNG")
                image_format = "PNG"

            mime_type = f"image/{image_format.lower()}"
            try:
                encoded_image = encode_image(umi, image_format)
                image_url_payload: dict[str, Any] = {
                    "url": f"data:{mime_type};base64,{encoded_image}",
                }
                if image_detail is not None:
                    image_url_payload["detail"] = image_detail
                user_message_content.append(
                    {
                        "type": "image_url",
                        "image_url": image_url_payload,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to encode image: {e}")
                continue

    if user_message_text:
        user_message_content.append({"type": "text", "text": textwrap.dedent(user_message_text)})

    messages: Messages = []

    if system_prompt_text:
        system_message: dict[str, Any] = {
            "role": "system",
            "content": [{"type": "text", "text": textwrap.dedent(system_prompt_text)}],
        }
        messages.append(system_message)

    messages.append(
        {
            "role": "user",
            "content": user_message_content,
        }
    )

    return messages

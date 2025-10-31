#!/usr/bin/env python3
"""Run the fine-tuned Churro VLM on a single image using only Transformers and Pytorch.

This script is a lightweight fallback for environments that cannot install the
full Churro package. It loads the `stanford-oval/churro-3B` model from the
Hugging Face Hub and transcribes a document page to XML.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


DEFAULT_MODEL_ID = "stanford-oval/churro-3B"
DEFAULT_SYSTEM_MESSAGE = "Transcribe the entiretly of this historical documents to XML format."
MAX_IMAGE_DIM = 2500
MIN_PIXELS = 512 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Churro OCR inference")
    parser.add_argument("image", type=Path, help="Path to the page image (PNG, JPG, or WebP)")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID to load (defaults to stanford-oval/churro-3B)",
    )
    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="System prompt to prepend before presenting the image",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20_000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' picks CUDA when available",
    )
    return parser.parse_args()


def _resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Match Churro's LLM preprocessing guard (<=2500px on the longest side)."""
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if hasattr(Image, "Resampling"):
        resample_filter = Image.Resampling.LANCZOS
    else:  # pragma: no cover - Pillow < 10 fallback
        resample_filter = Image.LANCZOS  # type: ignore[attr-defined]
    return image.resize(new_size, resample=resample_filter)


def _load_processor(model_id: str) -> AutoProcessor:
    """Instantiate the processor with the same pixel bounds used during fine-tuning."""
    processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
    return AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        **processor_kwargs,
    )


def _select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_inputs(
    processor: AutoProcessor,
    image_path: Path,
    system_message: str,
    device: torch.device,
) -> dict[str, Any]:
    image = load_image(str(image_path))
    if not isinstance(image, Image.Image):  # pragma: no cover - defensive
        raise TypeError(f"Unexpected image type: {type(image)!r}")
    image = image.convert("RGB")
    image = _resize_image_to_fit(image, MAX_IMAGE_DIM, MAX_IMAGE_DIM)
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }
    encoded["prompt_text"] = prompt
    encoded["conversation"] = conversation
    return encoded


def _run_generation(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    inputs: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
) -> str:
    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if processor.tokenizer.pad_token_id is not None:
        generation_kwargs.setdefault("pad_token_id", processor.tokenizer.pad_token_id)
    if processor.tokenizer.eos_token_id is not None:
        generation_kwargs.setdefault("eos_token_id", processor.tokenizer.eos_token_id)

    with torch.inference_mode():
        generated = model.generate(
            **{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}, **generation_kwargs
        )

    new_tokens = generated[0, input_length:]
    transcription = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return transcription.strip()


def main() -> None:
    args = _parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = _select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = _load_processor(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = _prepare_inputs(processor, args.image, args.system_message, device)
    transcription = _run_generation(
        model,
        processor,
        inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(transcription)


if __name__ == "__main__":
    main()

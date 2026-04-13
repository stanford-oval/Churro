"""Dots OCR helpers for Hugging Face OCR backends."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_DOTS_OCR_1_5_LOCAL_DIRNAME = "DotsOCR_1_5"
_DOTS_FLASH_ATTN_IMPORT = "from flash_attn import flash_attn_varlen_func"
_DOTS_FLASH_ATTN_FALLBACK = """try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
"""
_DOTS_FORCE_BFLOAT16_LINE = "            hidden_states = hidden_states.bfloat16()"
_DOTS_WEIGHT_DTYPE_LINE = (
    "            hidden_states = hidden_states.to(self.patch_embed.patchifier.proj.weight.dtype)"
)


def _patch_dots_ocr_vision_module(model_dir: Path) -> None:
    vision_module_path = model_dir / "modeling_dots_vision.py"
    vision_module = vision_module_path.read_text()
    if _DOTS_FLASH_ATTN_IMPORT not in vision_module and _DOTS_FLASH_ATTN_FALLBACK in vision_module:
        return
    vision_lines = vision_module.splitlines()
    import_index = next(
        (index for index, line in enumerate(vision_lines) if _DOTS_FLASH_ATTN_IMPORT in line),
        None,
    )
    if import_index is None:
        return

    block_tokens = {"", "try:", "except ImportError:", "flash_attn_varlen_func = None"}
    block_start = import_index
    while block_start > 0 and vision_lines[block_start - 1].strip() in block_tokens:
        block_start -= 1

    block_end = import_index + 1
    while block_end < len(vision_lines) and vision_lines[block_end].strip() in block_tokens:
        block_end += 1

    patched_lines = (
        vision_lines[:block_start]
        + _DOTS_FLASH_ATTN_FALLBACK.rstrip("\n").splitlines()
        + vision_lines[block_end:]
    )
    patched_vision_module = "\n".join(patched_lines) + "\n"
    if _DOTS_FORCE_BFLOAT16_LINE in patched_vision_module:
        patched_vision_module = patched_vision_module.replace(
            _DOTS_FORCE_BFLOAT16_LINE,
            _DOTS_WEIGHT_DTYPE_LINE,
        )
    vision_module_path.write_text(patched_vision_module)


def _prepare_dots_ocr_model_dir(
    model_id: str,
    *,
    home_dir: Path,
    patch_vision_module: Callable[[Path], None],
    configuration_error: Callable[[str], Exception],
    extra_install_hint: str,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - transitively provided by transformers
        message = f"Hugging Face OCR requires the `hf` runtime. {extra_install_hint}"
        raise configuration_error(message) from exc

    model_dir = (
        home_dir
        / ".cache"
        / "churro-ocr"
        / "hf"
        / _DOTS_OCR_1_5_LOCAL_DIRNAME
        / model_id.replace("/", "__").replace(".", "_")
    )
    snapshot_download(repo_id=model_id, local_dir=model_dir)
    patch_vision_module(model_dir)
    return str(model_dir)


def _resolve_base_prepare_inputs_for_generation(
    model: object,
    prepare_inputs_for_generation: Callable[..., object],
) -> Callable[..., object] | None:
    original_prepare_inputs = getattr(prepare_inputs_for_generation, "__func__", None)
    base_prepare_inputs_for_generation = prepare_inputs_for_generation
    if original_prepare_inputs is not None:
        for candidate in type(model).__mro__[1:]:
            candidate_prepare_inputs = candidate.__dict__.get("prepare_inputs_for_generation")
            if candidate_prepare_inputs is None or candidate_prepare_inputs is original_prepare_inputs:
                continue
            base_prepare_inputs_for_generation = cast("Any", candidate_prepare_inputs).__get__(
                model,
                type(model),
            )
            break
    return base_prepare_inputs_for_generation if callable(base_prepare_inputs_for_generation) else None


def _first_cache_position(cache_position: object) -> int | None:
    if cache_position is None:
        return None
    try:
        return int(cast("Any", cache_position)[0])
    except (IndexError, TypeError, ValueError):
        return None


def _patch_dots_ocr_prepare_inputs_for_generation(model: object) -> None:
    prepare_inputs_for_generation = getattr(model, "prepare_inputs_for_generation", None)
    if not callable(prepare_inputs_for_generation):
        return
    if getattr(model, "_churro_dots_prepare_inputs_patched", False):
        return

    base_prepare_inputs_for_generation = _resolve_base_prepare_inputs_for_generation(
        model,
        prepare_inputs_for_generation,
    )
    if base_prepare_inputs_for_generation is None:
        return

    def _patched_prepare_inputs_for_generation(
        _self: object,
        input_ids: object,
        *,
        pixel_values: object = None,
        cache_position: object = None,
        **kwargs: object,
    ) -> dict[str, object]:
        model_inputs = cast(
            "dict[str, object]",
            base_prepare_inputs_for_generation(
                input_ids,
                cache_position=cache_position,
                **kwargs,
            ),
        )
        if _first_cache_position(cache_position) in {None, 0}:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs

    model_any = cast("Any", model)
    model_any.prepare_inputs_for_generation = MethodType(_patched_prepare_inputs_for_generation, model)
    model_any._churro_dots_prepare_inputs_patched = True


def _default_dots_ocr_1_5_model_kwargs(
    *,
    load_torch_module: Callable[[], object],
) -> dict[str, object]:
    model_kwargs: dict[str, object] = {"dtype": "auto"}
    try:
        torch = load_torch_module()
    except ImportError:  # pragma: no cover - torch is installed separately for local HF use
        return model_kwargs

    if not cast("Any", torch).cuda.is_available():
        return model_kwargs

    try:
        free_bytes, _ = cast("Any", torch).cuda.mem_get_info()
    except RuntimeError:
        return model_kwargs
    free_gib = max(1, int(free_bytes / (1024**3)) - 1)
    if free_gib < 8:
        return {"dtype": "float32"}

    model_kwargs["device_map"] = "auto"
    model_kwargs["max_memory"] = {0: f"{free_gib}GiB", "cpu": "128GiB"}
    return model_kwargs

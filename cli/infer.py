from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from PIL import Image

from churro.systems.llm_improver import LLMImprover
from churro.systems.ocr_factory import OCRFactory
from churro.utils.image.binarizer import ImageBinarizer
from churro.utils.llm.models import MODEL_MAP
from churro.utils.log_utils import logger

from .helpers import managed_vllm_container


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SYSTEM_CHOICES = ["azure", "mistral_ocr", "llm", "finetuned"]
DEFAULT_PATTERN = "*.png"
DEFAULT_SUFFIXES: list[str] = [".png"]
DEFAULT_MAX_CONCURRENCY = 64
DEFAULT_IMPROVER_ENGINE = "gemini-2.5-pro-low"
DEFAULT_IMPROVER_BACKUP_ENGINE = "gpt-5-low"


@dataclass(slots=True)
class InferOptions:
    system: str
    engine: str | None
    backup_engine: str | None
    tensor_parallel_size: int
    data_parallel_size: int
    image: Path | None
    image_dir: Path | None
    pattern: str
    suffixes: list[str]
    recursive: bool
    output_dir: Path | None
    skip_existing: bool
    max_concurrency: int
    strip_xml: bool = False
    use_improver: bool = False
    improver_engine: str | None = None
    improver_backup_engine: str | None = None
    improver_resize: int | None = None
    output_markdown: bool = False
    binarize: bool = False


def _validate_options(options: InferOptions) -> int:
    systems_requiring_engine = {"llm", "finetuned"}
    valid_engines_string: str = json.dumps(sorted(MODEL_MAP.keys()), indent=2)
    if options.system in systems_requiring_engine:
        if not options.engine:
            logger.error(f"--engine is required when using system '{options.system}'.")
            return 1
        if options.engine not in MODEL_MAP:
            logger.error(
                f"Invalid --engine '{options.engine}'. Available options: {valid_engines_string}."
            )
            return 1
    elif options.engine and options.engine not in MODEL_MAP:
        logger.warning(
            f"--engine '{options.engine}' not found in MODEL_MAP; continuing without vLLM validation."
        )
    if options.backup_engine and options.backup_engine not in MODEL_MAP:
        logger.error(
            f"Invalid --backup-engine '{options.backup_engine}'. Available options: {valid_engines_string}."
        )
        return 1
    if options.strip_xml and options.system != "finetuned":
        logger.warning("--strip-xml only affects the 'finetuned' system.")
    if options.output_markdown and options.system != "llm":
        logger.error("--output-markdown is only supported when --system llm.")
        return 1
    improver_fields_provided = any(
        [
            options.improver_engine,
            options.improver_backup_engine,
            options.improver_resize is not None,
        ]
    )
    if options.use_improver:
        if not options.improver_engine:
            options.improver_engine = DEFAULT_IMPROVER_ENGINE
            logger.info(
                f"No --improver-engine provided; defaulting to '{DEFAULT_IMPROVER_ENGINE}'."
            )
        if options.improver_engine not in MODEL_MAP:
            logger.error(
                f"Invalid --improver-engine '{options.improver_engine}'. "
                f"Available options: {valid_engines_string}."
            )
            return 1
        if not options.improver_backup_engine:
            options.improver_backup_engine = DEFAULT_IMPROVER_BACKUP_ENGINE
            logger.info(
                f"No --improver-backup-engine provided; defaulting to '{DEFAULT_IMPROVER_BACKUP_ENGINE}'."
            )
        if options.improver_backup_engine and options.improver_backup_engine not in MODEL_MAP:
            logger.error(
                f"Invalid --improver-backup-engine '{options.improver_backup_engine}'. "
                f"Available options: {valid_engines_string}."
            )
            return 1
    elif improver_fields_provided:
        logger.warning(
            "Improver options provided without --use-improver; OCR outputs will not be post-processed."
        )
    invalid_suffixes = [s for s in options.suffixes if s not in ALLOWED_IMAGE_EXTS]
    if invalid_suffixes:
        preview = ", ".join(invalid_suffixes)
        logger.warning(f"Ignoring unsupported suffixes for infer command: {preview}")
        options.suffixes = [s for s in options.suffixes if s in ALLOWED_IMAGE_EXTS]
        if not options.suffixes:
            logger.error("No valid suffixes remain after filtering unsupported values.")
            return 1
    return 0


def _collect_images(
    image: Path | None,
    image_dir: Path | None,
    suffixes: list[str],
    recursive: bool,
) -> list[Path]:
    images: list[Path] = []
    if image and image.is_file():
        if image.suffix.lower() in ALLOWED_IMAGE_EXTS:
            images.append(image)
        else:  # pragma: no cover - defensive
            logger.warning(f"--image {image} does not have a supported extension; skipping.")
    if image_dir and image_dir.is_dir():
        suffix_filter = {s.lower() for s in suffixes}
        iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
        for path in iterator:
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in ALLOWED_IMAGE_EXTS and ext in suffix_filter:
                images.append(path)
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in images:
        key = str(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    return ordered


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _write_or_print_output(
    *,
    img_path: Path,
    text: str,
    output_dir: Path | None,
    skip_existing: bool,
    multi_mode: bool,
) -> None:
    out_path: Path | None = None
    if output_dir:
        out_path = output_dir / (img_path.stem + ".txt")
        if skip_existing and out_path.exists():
            logger.info(f"Skipping existing {out_path.name}")
            return
    if out_path is not None:
        out_path.write_text(text)
        logger.info(f"Wrote {out_path}")
    else:
        if multi_mode:
            header = f"===== {img_path} ====="
            print(header)
        print(text)


async def run(options: InferOptions) -> int:
    """Execute the ad-hoc inference workflow."""
    validation_status = _validate_options(options)
    if validation_status != 0:
        return validation_status

    if not options.image and not options.image_dir:
        logger.error("Either --image or --image-dir must be provided.")
        return 1
    if options.image and options.image_dir:
        logger.error("Specify only one of --image or --image-dir.")
        return 1

    images = _collect_images(
        options.image,
        options.image_dir,
        options.suffixes,
        options.recursive,
    )
    images = sorted(images, key=_natural_key)
    if not images:
        logger.error("No images found to process.")
        return 1

    original_image_paths = [str(path) for path in images]
    binarized_images: list[Image.Image] | None = None
    opened_originals: list[Image.Image] = []
    try:
        if options.binarize:
            logger.info(f"Binarizing {len(images)} input image(s) prior to OCR.")
            try:
                binarizer = ImageBinarizer()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error(f"Failed to initialize image binarizer: {exc}")
                return 1
            for img_path in images:
                try:
                    opened_image = Image.open(img_path)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.error(f"Failed to open {img_path} for binarization: {exc}")
                    for image in opened_originals:
                        image.close()
                    return 1
                opened_originals.append(opened_image)

            try:
                binarized_images = binarizer.binarize_pil_batch(opened_originals)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error(f"Binarizer batch inference failed: {exc}")
                return 1

            if len(binarized_images) != len(images):  # pragma: no cover - defensive guard
                logger.error(
                    f"Binarizer returned {len(binarized_images)} outputs for {len(images)} inputs."
                )
                for image in binarized_images:
                    image.close()
                return 1

        multi_mode = len(images) > 1
        if options.output_dir:
            options.output_dir.mkdir(parents=True, exist_ok=True)

        max_concurrency = options.max_concurrency if options.max_concurrency > 0 else 1
        if max_concurrency != options.max_concurrency:
            logger.warning("--max-concurrency < 1 ignored; defaulting to 1")

        with managed_vllm_container(
            engine=options.engine,
            backup_engine=options.backup_engine,
            system=options.system,
            tensor_parallel_size=options.tensor_parallel_size,
            data_parallel_size=options.data_parallel_size,
        ):
            ocr_system = OCRFactory.create_ocr_system(options)  # type: ignore[arg-type]
            try:
                if binarized_images is not None:
                    processed_outputs = await ocr_system.process_images(
                        binarized_images,
                        max_concurrency=max_concurrency,
                    )
                else:
                    processed_outputs = await ocr_system.process_images_from_files(
                        original_image_paths,
                        max_concurrency,
                    )
            except RuntimeError as exc:
                logger.error(f"OCR processing failed: {exc}")
                return 1
            outputs: list[str] = processed_outputs

        if options.use_improver:
            improver_engine = options.improver_engine
            if not improver_engine:
                logger.error("Internal error: --use-improver enabled without --improver-engine.")
                return 1
            backup_suffix = (
                f" and backup engine '{options.improver_backup_engine}'"
                if options.improver_backup_engine
                else ""
            )
            logger.info(f"Running LLMImprover with engine '{improver_engine}'{backup_suffix}.")
            improver = LLMImprover(
                engine=improver_engine,
                backup_engine=options.improver_backup_engine,
                resize=options.improver_resize,
                image_fidelity="high",
            )
            outputs = await improver.process_batch_inputs(
                image_paths=original_image_paths,
                texts=outputs,
                max_concurrency=max_concurrency,
            )

        for img_path, text in zip(images, outputs, strict=False):
            _write_or_print_output(
                img_path=img_path,
                text=text,
                output_dir=options.output_dir,
                skip_existing=options.skip_existing,
                multi_mode=multi_mode,
            )

        if multi_mode:
            logger.info(f"Processed {len(images)} image(s).")
        return 0
    finally:
        if binarized_images is not None:
            for image in binarized_images:
                try:
                    image.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
        for image in opened_originals:
            try:
                image.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

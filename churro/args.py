import argparse
from pathlib import Path
import shutil
import sys

from churro.utils.llm.models import MODEL_MAP
from churro.utils.log_utils import logger


# Central place to define canonical HF identifiers
CHURRO_DATASET_ID: str = "stanford-oval/churro-dataset"


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    """Create the benchmark CLI argument parser without parsing."""
    parser = argparse.ArgumentParser(add_help=add_help)

    parser.add_argument(
        "--system",
        required=True,
        choices=[
            "azure",
            "mistral_ocr",
            "llm",
            "finetuned",
        ],
        help="Specify the system to run.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="For LLM baseline, specify the LLM to use.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size. Only used for local models.",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Data parallel size. Only used for local models.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="If set, will resize large images to fit inside a square of this size (in pixels). ",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=50,
        help="Maximum number of LLM requests to allow at once.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=0,
        help="Number of images to process. 0 means all images.",
    )
    parser.add_argument(
        "--dataset-split",
        required=True,
        type=str,
        choices=["dev", "test"],
        help="Data split to use.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Offset for the input images.")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments and validate."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations."""
    if args.system == "llm":
        assert args.engine is not None, "LLM engine must be specified for LLM baseline."
        valid_engines = ", ".join(sorted(MODEL_MAP.keys()))
        assert args.engine in MODEL_MAP, (
            f"Invalid engine: {args.engine}. Possible values are: {valid_engines}"
        )


def create_output_prefix(args: argparse.Namespace) -> str:
    """Create output directory path based on arguments."""
    base_workdir = Path(__file__).resolve().parent / "workdir"
    output_dir = base_workdir / "results" / args.dataset_split / args.system

    if args.system not in ["azure", "mistral_ocr"]:
        engine_name = args.engine
        if "/" in engine_name:
            engine_name = args.engine.split("/")
            engine_name = [p for p in engine_name if "workdir" not in p and p]
            engine_name = "_".join(engine_name)
        output_dir = output_dir.parent / f"{output_dir.name}_{engine_name}"

    # If directory exists already, warn user and exit to avoid overwriting prior results.
    if output_dir.exists():
        try:
            # Abort (after confirmation) only if directory exists AND is not empty.
            if any(output_dir.iterdir()):
                if not sys.stdin.isatty():
                    logger.warning(
                        f"Output directory '{output_dir}' already exists and is not empty, "
                        "but cannot prompt in non-interactive mode. Aborting to avoid overwrite."
                    )
                    raise SystemExit(1)
                response = (
                    input(
                        f"Output directory '{output_dir}' already exists and is not empty. "
                        "Overwrite (this will delete existing contents)? [y/N]: "
                    )
                    .strip()
                    .lower()
                )
                if response in {"y", "yes"}:
                    try:
                        shutil.rmtree(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Overwrote existing directory '{output_dir}'.")
                    except OSError as exc:
                        logger.error(f"Failed to overwrite directory '{output_dir}': {exc}")
                        raise SystemExit(1) from exc
                else:
                    logger.info("User declined overwrite. Aborting.")
                    raise SystemExit(1)
            # Directory exists but is empty: reuse it.
        except OSError as exc:
            logger.warning(f"Could not inspect contents of '{output_dir}'. Aborting for safety.")
            raise SystemExit(1) from exc
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir)

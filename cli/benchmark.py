from __future__ import annotations

from dataclasses import dataclass
from time import time

from datasets import load_dataset

from churro.args import CHURRO_DATASET_ID, create_output_prefix
from churro.evaluation.metrics import compute_metrics
from churro.systems.ocr_factory import OCRFactory
from churro.utils.llm.models import MODEL_MAP
from churro.utils.log_utils import logger

from .helpers import managed_vllm_container


VALID_DATASET_SPLITS = {"dev", "test"}


@dataclass(slots=True)
class BenchmarkOptions:
    system: str
    engine: str | None
    tensor_parallel_size: int
    data_parallel_size: int
    resize: int | None
    max_concurrency: int
    input_size: int
    dataset_split: str
    offset: int


def _validate_options(options: BenchmarkOptions) -> int:
    if options.system == "llm":
        if not options.engine:
            logger.error("LLM engine must be specified for the LLM baseline.")
            return 1
        if options.engine not in MODEL_MAP:
            valid_engines = ", ".join(sorted(MODEL_MAP.keys()))
            logger.error(f"Invalid engine: {options.engine}. Possible values are: {valid_engines}")
            return 1
    if options.dataset_split not in VALID_DATASET_SPLITS:
        valid = ", ".join(sorted(VALID_DATASET_SPLITS))
        logger.error(f"Invalid dataset split '{options.dataset_split}'. Choose from: {valid}.")
        return 1
    return 0


async def run(options: BenchmarkOptions) -> int:
    """Execute the benchmark workflow using the provided options."""
    validation_status = _validate_options(options)
    if validation_status != 0:
        return validation_status

    output_prefix = create_output_prefix(options)  # type: ignore[arg-type]

    start_index = options.offset
    end_index = options.offset + options.input_size if options.input_size > 0 else None

    logger.info(
        f"Loading dataset slice: split={options.dataset_split}, offset={options.offset}, "
        f"limit={options.input_size if end_index is None else options.input_size}"
    )
    dataset = list(load_dataset(CHURRO_DATASET_ID, split=options.dataset_split, streaming=True))
    dataset = dataset[start_index:end_index]

    elapsed_time = 0.0
    with managed_vllm_container(
        engine=options.engine,
        backup_engine=None,
        system=options.system,
        tensor_parallel_size=options.tensor_parallel_size,
        data_parallel_size=options.data_parallel_size,
    ):
        ocr_system = OCRFactory.create_ocr_system(options)  # type: ignore[arg-type]
        start_time = time()
        images = [example["image"] for example in dataset]
        predicted_texts = await ocr_system.process_images(
            images,
            max_concurrency=options.max_concurrency,
        )
        elapsed_time = time() - start_time

    assert len(dataset) == len(predicted_texts), (
        f"Mismatch in number of examples ({len(dataset)}) and predicted texts ({len(predicted_texts)})."
    )

    compute_metrics(dataset, predicted_texts, output_prefix, elapsed_time)
    return 0

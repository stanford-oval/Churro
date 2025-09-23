import asyncio
import contextlib
import os
from time import time

from datasets import load_dataset

from evaluation.metrics import compute_metrics
from ocr.args import CHURRO_DATASET_ID, create_output_prefix, parse_args
from ocr.systems.ocr_factory import OCRFactory
from utils.docker.vllm import (
    maybe_start_vllm_server_for_engine,
)
from utils.log_utils import logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def run_ocr_system(args, dataset: list[dict]) -> list[str]:
    """Run the specified OCR system on the examples and return predicted texts."""
    # Create OCR system
    ocr_system = OCRFactory.create_ocr_system(args)

    # Process examples using the unified interface
    return await ocr_system.process(dataset)


async def main() -> None:
    """CLI entry: orchestrate loading, inference (or reuse), and evaluation."""
    # Parse arguments and setup
    args = parse_args()
    output_prefix = create_output_prefix(args)

    # Optionally spin up local containers for hosted engine variants (vLLM)
    llm_containers = []

    # Calculate start and end index based on args.input_size and args.offset
    start_index = args.offset
    end_index = args.offset + args.input_size if args.input_size > 0 else None

    # Load examples
    dataset: list[dict] = list(
        load_dataset(CHURRO_DATASET_ID, split=args.dataset_split, streaming=True)
    )
    dataset = dataset[start_index:end_index]

    elapsed_time = 0.0

    try:
        # Conditionally start local vLLM container if required
        # Cast args for type checker; runtime object has required attributes set by parse_args().
        llm_containers.append(
            maybe_start_vllm_server_for_engine(
                engine=args.engine,
                system=args.system,
                tensor_parallel_size=args.tensor_parallel_size,
                data_parallel_size=args.data_parallel_size,
            )
        )

        start_time = time()
        predicted_texts = await run_ocr_system(args, dataset)
        end_time = time()
        elapsed_time = end_time - start_time
    finally:
        for c in llm_containers:
            logger.info("Stopping container...")
            with contextlib.suppress(Exception):
                c.stop()

    # Validate results
    assert len(dataset) == len(predicted_texts), (
        f"Mismatch in number of examples ({len(dataset)}) and predicted texts ({len(predicted_texts)})."
    )

    # Calculate metrics and save them to file
    compute_metrics(dataset, predicted_texts, output_prefix, elapsed_time)


if __name__ == "__main__":
    asyncio.run(main())

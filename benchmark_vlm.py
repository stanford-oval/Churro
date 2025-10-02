"""Benchmark OCR systems on the Churro historical document dataset.

This script orchestrates end-to-end OCR evaluation:
1. Loads a slice of the Churro dataset based on CLI arguments
2. Optionally starts local vLLM container for hosted inference
3. Runs the specified OCR system (via OCRFactory) on all examples
4. Computes evaluation metrics and saves results to disk

Usage:
    pixi run python benchmark_vlm.py --system <system> --engine <engine> [options]
"""

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


async def run_ocr_system(args, dataset: list[dict], max_concurrency: int) -> list[str]:
    """Run the specified OCR system on the examples and return predicted texts.

    Creates an OCR system instance via OCRFactory (which uses args.system to select
    the concrete implementation) and processes all examples through its async interface.

    Args:
        args: Parsed command-line arguments containing system, engine, and other config.
        dataset: List of example dicts from the Churro dataset (image paths + ground truth).
        max_concurrency: Maximum simultaneous inference calls permitted when processing examples.

    Returns:
        List of predicted text strings, one per input example.
    """
    # Create OCR system via factory (ensures correct concrete class for --system flag)
    ocr_system = OCRFactory.create_ocr_system(args)

    # Process examples using the unified BaseOCR.process() interface
    return await ocr_system.process_examples(dataset, max_concurrency=max_concurrency)


async def main() -> None:
    """CLI entry point: orchestrate dataset loading, OCR inference, and metric evaluation.

    Pipeline:
    1. Parse CLI args (system, engine, dataset split, input size, offset, etc.)
    2. Load specified slice of Churro dataset from HuggingFace
    3. Optionally start local vLLM container if engine requires hosted inference
    4. Run OCR system on all examples
    5. Compute metrics (CER, WER, costs) and write outputs.json + all_metrics.json
    6. Clean up any Docker containers
    """
    # Parse arguments and setup output directory prefix (results/<split>/<system>_<engine>)
    args = parse_args()
    output_prefix = create_output_prefix(args)

    # Track any local vLLM containers we start for cleanup
    llm_container = None

    # Calculate dataset slice indices based on offset and input_size
    start_index = args.offset
    end_index = args.offset + args.input_size if args.input_size > 0 else None

    # Load examples from HuggingFace Churro dataset (streaming mode for memory efficiency)
    dataset: list[dict] = list(
        load_dataset(CHURRO_DATASET_ID, split=args.dataset_split, streaming=True)
    )
    dataset = dataset[start_index:end_index]

    elapsed_time = 0.0

    try:
        # Conditionally start local vLLM container if engine requires it (e.g., vllm_local)
        # This handles model loading, GPU allocation, and server startup
        llm_container = maybe_start_vllm_server_for_engine(
            engine=args.engine,
            system=args.system,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
        )

        # Run OCR system on all examples and measure elapsed time
        start_time = time()
        predicted_texts = await run_ocr_system(args, dataset, max_concurrency=args.max_concurrency)
        end_time = time()
        elapsed_time = end_time - start_time
    finally:
        # Ensure Docker containers are stopped even if OCR fails
        if llm_container is not None:
            logger.info("Stopping container...")
            with contextlib.suppress(Exception):
                llm_container.stop()

    # Sanity check: ensure we got predictions for all examples
    assert len(dataset) == len(predicted_texts), (
        f"Mismatch in number of examples ({len(dataset)}) and predicted texts ({len(predicted_texts)})."
    )

    # Calculate metrics (CER, WER, costs, etc.) and save to results directory
    # Writes: outputs.json, all_metrics.json, and scatter plot
    compute_metrics(dataset, predicted_texts, output_prefix, elapsed_time)


if __name__ == "__main__":
    asyncio.run(main())

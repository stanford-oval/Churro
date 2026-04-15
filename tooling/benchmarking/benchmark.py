"""Benchmark helpers for running repo-local OCR evaluations on CHURRO-DS."""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, cast

from PIL import Image
from tqdm import tqdm

_REPO_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
_REPO_SRC_PATH_STR = str(_REPO_SRC_PATH)
if _REPO_SRC_PATH_STR in sys.path:
    sys.path.remove(_REPO_SRC_PATH_STR)
sys.path.insert(0, _REPO_SRC_PATH_STR)

from churro_ocr._internal.litellm import close_litellm_async_clients  # noqa: E402
from churro_ocr._internal.logging import logger  # noqa: E402
from churro_ocr.errors import ConfigurationError  # noqa: E402
from churro_ocr.ocr import BatchOCRBackend, OCRBackend, OCRBackendLike  # noqa: E402
from churro_ocr.page_detection import DocumentPage  # noqa: E402
from churro_ocr.providers import (  # noqa: E402
    AzureDocumentIntelligenceOptions,
    HuggingFaceOptions,
    LiteLLMTransportConfig,
    MistralOptions,
    OCRBackendSpec,
    OpenAICompatibleOptions,
    build_ocr_backend,
)
from churro_ocr.providers.specs import MISTRAL_OCR_MODEL_IDS, validate_mistral_ocr_model  # noqa: E402
from tooling.benchmarking.dataset import (  # noqa: E402
    DatasetSelection,
    DatasetSubset,
    load_dataset_split,
)
from tooling.evaluation.metrics import compute_metrics  # noqa: E402
from tooling.evaluation.types import to_evaluation_example  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tooling.evaluation.types import (
        BenchmarkDatasetExample,
        BenchmarkPrediction,
        EvaluationExample,
    )

CHURRO_DATASET_ID = "stanford-oval/churro-dataset"
VALID_DATASET_SPLITS = {"dev", "test"}
VALID_OCR_BACKENDS = {"litellm", "openai-compatible", "azure", "mistral", "hf"}
PROGRESS_BAR_SMOOTHING = 0.05
PROGRESS_BAR_MININTERVAL_SECONDS = 1.0
BENCHMARK_DATASET_COLUMNS = (
    "image",
    "cleaned_transcription",
    "dataset_id",
    "document_type",
    "example_id",
    "main_language",
    "main_script",
)
_PREDICTION_FAILURES = (
    AssertionError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(slots=True)
class BenchmarkOptions:
    """Normalized options for dataset benchmarking."""

    backend: str
    dataset_split: str
    language: str | None = None
    document_type: str | None = None
    model: str | None = None
    input_size: int = 0
    offset: int = 0
    output_dir: Path | None = None
    max_concurrency: int = 10
    endpoint: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    reasoning_effort: str | None = None

    def dataset_subset(self) -> DatasetSubset:
        """Return the normalized subset filters for this benchmark run."""
        return DatasetSubset.from_raw(
            language=self.language,
            document_type=self.document_type,
        )

    def dataset_selection(self) -> DatasetSelection:
        """Return the dataset selection window for this benchmark run."""
        return DatasetSelection(
            subset=self.dataset_subset(),
            offset=self.offset,
            limit=self.input_size,
        )


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    """Create the benchmark CLI parser."""
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--backend", required=True, choices=sorted(VALID_OCR_BACKENDS))
    parser.add_argument("--dataset-split", required=True, choices=sorted(VALID_DATASET_SPLITS))
    parser.add_argument("--language", default=None)
    parser.add_argument("--document-type", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--input-size", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> BenchmarkOptions:
    """Parse benchmark CLI args into a normalized options object."""
    namespace = build_parser().parse_args(argv)
    return BenchmarkOptions(
        backend=namespace.backend,
        dataset_split=namespace.dataset_split,
        language=namespace.language,
        document_type=namespace.document_type,
        model=namespace.model,
        input_size=namespace.input_size,
        offset=namespace.offset,
        output_dir=namespace.output_dir,
        max_concurrency=namespace.max_concurrency,
        endpoint=namespace.endpoint,
        api_key=namespace.api_key,
        base_url=namespace.base_url,
        api_version=namespace.api_version,
        reasoning_effort=namespace.reasoning_effort,
    )


def _validate_options(options: BenchmarkOptions) -> int:
    if options.dataset_split not in VALID_DATASET_SPLITS:
        logger.error("Invalid dataset split '%s'.", options.dataset_split)
        return 1
    if options.input_size < 0 or options.offset < 0:
        logger.error("input-size and offset must be non-negative.")
        return 1
    if options.output_dir is not None and options.output_dir.exists() and not options.output_dir.is_dir():
        logger.error("Output path '%s' exists and is not a directory.", options.output_dir)
        return 1
    if options.reasoning_effort is not None and options.backend not in {"litellm", "openai-compatible"}:
        logger.error(
            "--reasoning-effort is only supported for backend=litellm and backend=openai-compatible."
        )
        return 1
    if options.backend == "litellm" and not options.model:
        logger.error("--model is required for backend=litellm.")
        return 1
    if options.backend == "openai-compatible" and (not options.model or not options.base_url):
        logger.error("--model and --base-url are required for backend=openai-compatible.")
        return 1
    if options.backend == "hf" and not options.model:
        logger.error("--model is required for backend=hf.")
        return 1
    if options.backend == "azure" and (not options.endpoint or not options.api_key):
        logger.error("--endpoint and --api-key are required for backend=azure.")
        return 1
    if options.backend == "mistral":
        if not options.api_key:
            logger.error("--api-key is required for backend=mistral.")
            return 1
        try:
            validate_mistral_ocr_model(options.model)
        except ConfigurationError:
            logger.error(
                "--model is required for backend=mistral and must be one of: %s.",
                ", ".join(MISTRAL_OCR_MODEL_IDS),
            )
            return 1
    return 0


def create_output_prefix(options: BenchmarkOptions) -> str:
    """Create the output directory for benchmark artifacts."""
    if options.output_dir is not None:
        output_dir = options.output_dir
    else:
        suffix = options.backend
        if options.model:
            suffix = f"{suffix}_{options.model.replace('/', '_')}"
        filter_suffixes = options.dataset_subset().output_suffixes()
        if filter_suffixes:
            suffix = "_".join([suffix, *filter_suffixes])
        output_dir = (
            Path(__file__).resolve().parents[2] / "workdir" / "results" / options.dataset_split / suffix
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def _load_dataset(dataset_id: str, *, split: str) -> Iterable[BenchmarkDatasetExample]:
    return cast(
        "Iterable[BenchmarkDatasetExample]",
        load_dataset_split(dataset_id, split, columns=BENCHMARK_DATASET_COLUMNS),
    )


def _default_litellm_cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "workdir" / "cache" / "litellm"


def _transport_completion_kwargs(options: BenchmarkOptions) -> dict[str, object]:
    completion_kwargs: dict[str, object] = {}
    if options.reasoning_effort is not None:
        completion_kwargs["reasoning_effort"] = options.reasoning_effort
    return completion_kwargs


def _create_progress_bar(*, total: int | None, desc: str, unit: str) -> tqdm[object]:
    """Return a tqdm progress bar tuned for steadier ETA updates."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        mininterval=PROGRESS_BAR_MININTERVAL_SECONDS,
        smoothing=PROGRESS_BAR_SMOOTHING,
    )


def _build_evaluation_example(example: BenchmarkDatasetExample) -> EvaluationExample:
    """Keep only the fields needed for evaluation after OCR completes."""
    return to_evaluation_example(example)


def _selected_dataset_examples(
    dataset_stream: Iterable[BenchmarkDatasetExample],
    options: BenchmarkOptions,
) -> Iterable[BenchmarkDatasetExample]:
    """Yield the requested dataset slice without materializing it upfront."""
    return options.dataset_selection().select(dataset_stream)


def _build_ocr_backend(options: BenchmarkOptions) -> OCRBackendLike:
    if options.backend == "litellm":
        assert options.model is not None
        return build_ocr_backend(
            OCRBackendSpec(
                provider="litellm",
                model=options.model,
                transport=LiteLLMTransportConfig(
                    api_base=options.base_url,
                    api_key=options.api_key,
                    api_version=options.api_version,
                    completion_kwargs=_transport_completion_kwargs(options),
                    cache_dir=_default_litellm_cache_dir(),
                ),
            )
        )
    if options.backend == "openai-compatible":
        assert options.model is not None
        assert options.base_url is not None
        return build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model=options.model,
                transport=LiteLLMTransportConfig(
                    api_base=options.base_url,
                    api_key=options.api_key,
                    api_version=options.api_version,
                    completion_kwargs=_transport_completion_kwargs(options),
                ),
                options=OpenAICompatibleOptions(),
            )
        )
    if options.backend == "azure":
        assert options.endpoint is not None
        assert options.api_key is not None
        return build_ocr_backend(
            OCRBackendSpec(
                provider="azure",
                model=options.model,
                options=AzureDocumentIntelligenceOptions(
                    endpoint=options.endpoint,
                    api_key=options.api_key,
                ),
            )
        )
    if options.backend == "hf":
        assert options.model is not None
        return build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=options.model,
                options=HuggingFaceOptions(model_kwargs={"device_map": "auto", "torch_dtype": "auto"}),
            )
        )
    assert options.api_key is not None
    assert options.model is not None
    mistral_model = validate_mistral_ocr_model(options.model)
    return build_ocr_backend(
        OCRBackendSpec(
            provider="mistral",
            model=mistral_model,
            options=MistralOptions(api_key=options.api_key),
        )
    )


def _log_first_benchmark_output(*, options: BenchmarkOptions, text: str) -> None:
    model_name = options.model or "<default>"
    logger.info(
        "First benchmark OCR output for backend=%s model=%s:\n%s",
        options.backend,
        model_name,
        text,
    )


def _failure_metadata(exc: BaseException) -> dict[str, object]:
    message = str(exc).strip()
    metadata: dict[str, object] = {
        "benchmark_error": {
            "type": type(exc).__name__,
        }
    }
    if message:
        metadata["benchmark_error"]["message"] = message
    return metadata


def _empty_prediction_for_failure(exc: BaseException) -> BenchmarkPrediction:
    return {
        "text": "",
        "metadata": _failure_metadata(exc),
    }


def _log_prediction_failure(
    *,
    options: BenchmarkOptions,
    example: BenchmarkDatasetExample,
    exc: BaseException,
) -> None:
    del exc
    logger.exception(
        "Benchmark OCR failed for example_id=%s dataset_id=%s backend=%s model=%s; "
        "treating prediction as empty.",
        example["example_id"],
        example["dataset_id"],
        options.backend,
        options.model or "<default>",
    )


async def _predict_texts(
    dataset: Iterable[BenchmarkDatasetExample],
    options: BenchmarkOptions,
    *,
    total_pages: int | None = None,
) -> tuple[list[EvaluationExample], list[BenchmarkPrediction]]:
    ocr_backend = _build_ocr_backend(options)
    max_in_flight = max(1, options.max_concurrency)
    has_logged_first_output = False
    if isinstance(ocr_backend, BatchOCRBackend):
        dataset_iterator = iter(dataset)
        evaluation_examples: list[EvaluationExample] = []
        predictions: list[BenchmarkPrediction] = []
        submitted_pages = 0

        with _create_progress_bar(total=total_pages, desc="OCR", unit="page") as progress:
            while True:
                batch_examples: list[BenchmarkDatasetExample] = []
                pages: list[DocumentPage] = []
                for batch_index in range(max_in_flight):
                    try:
                        example = next(dataset_iterator)
                    except StopIteration:
                        break
                    image = example["image"]
                    assert isinstance(image, Image.Image)
                    batch_examples.append(example)
                    evaluation_examples.append(_build_evaluation_example(example))
                    pages.append(DocumentPage(page_index=batch_index, source_index=0, image=image))
                    submitted_pages += 1

                if not pages:
                    break

                progress.set_postfix(submitted=submitted_pages, in_flight=len(pages), refresh=False)
                try:
                    batch_results = await ocr_backend.ocr_batch(pages)
                    assert len(batch_results) == len(pages), (
                        f"HF OCR batch returned {len(batch_results)} results for {len(pages)} pages."
                    )
                    if not has_logged_first_output and batch_results:
                        _log_first_benchmark_output(
                            options=options,
                            text=batch_results[0].text or "",
                        )
                        has_logged_first_output = True
                    predictions.extend(
                        {
                            "text": result.text or "",
                            "metadata": dict(result.metadata),
                        }
                        for result in batch_results
                    )
                except _PREDICTION_FAILURES as exc:
                    for example in batch_examples:
                        _log_prediction_failure(options=options, example=example, exc=exc)
                    predictions.extend(_empty_prediction_for_failure(exc) for _ in pages)
                progress.update(len(pages))
                progress.set_postfix(submitted=submitted_pages, in_flight=0, refresh=False)

        return evaluation_examples, predictions

    async def _predict(index: int, example: BenchmarkDatasetExample) -> tuple[int, BenchmarkPrediction]:
        image = example["image"]
        page = DocumentPage(page_index=index, source_index=0, image=image)
        try:
            if callable(ocr_backend) and not isinstance(ocr_backend, OCRBackend):
                result = await ocr_backend(page)
            else:
                assert isinstance(ocr_backend, OCRBackend)
                result = await ocr_backend.ocr(page)
        except _PREDICTION_FAILURES as exc:
            _log_prediction_failure(options=options, example=example, exc=exc)
            return index, _empty_prediction_for_failure(exc)
        return index, {
            "text": result.text or "",
            "metadata": dict(result.metadata),
        }

    dataset_iterator = iter(dataset)
    evaluation_examples: list[EvaluationExample] = []
    pending_tasks: set[asyncio.Task[tuple[int, BenchmarkPrediction]]] = set()
    predictions: list[BenchmarkPrediction] = []
    next_index = 0
    wait_poll_seconds = 1.0

    def _update_progress_status(progress: tqdm[object], *, force_refresh: bool = False) -> None:
        progress.set_postfix(
            submitted=next_index,
            in_flight=len(pending_tasks),
            refresh=False,
        )
        if force_refresh:
            progress.refresh()

    def _progress_heartbeat(progress: tqdm[object], stop_event: threading.Event) -> None:
        while not stop_event.wait(wait_poll_seconds):
            _update_progress_status(progress, force_refresh=True)

    with _create_progress_bar(total=total_pages, desc="OCR", unit="page") as progress:
        heartbeat_stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_progress_heartbeat,
            args=(progress, heartbeat_stop_event),
            daemon=True,
        )
        heartbeat_thread.start()
        _update_progress_status(progress, force_refresh=True)
        try:
            while True:
                while len(pending_tasks) < max_in_flight:
                    try:
                        example = next(dataset_iterator)
                    except StopIteration:
                        break
                    image = example["image"]
                    assert isinstance(image, Image.Image)
                    evaluation_examples.append(_build_evaluation_example(example))
                    predictions.append({"text": "", "metadata": {}})
                    pending_tasks.add(asyncio.create_task(_predict(next_index, example)))
                    next_index += 1
                    _update_progress_status(progress)

                if not pending_tasks:
                    break

                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done_tasks:
                    index, prediction = await task
                    predictions[index] = prediction
                    if not has_logged_first_output and index == 0:
                        _log_first_benchmark_output(options=options, text=prediction["text"])
                        has_logged_first_output = True
                    progress.update(1)
                _update_progress_status(progress)
        except Exception:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            raise
        finally:
            heartbeat_stop_event.set()
            heartbeat_thread.join(timeout=wait_poll_seconds * 2)
            _update_progress_status(progress, force_refresh=True)

    return evaluation_examples, predictions


async def run(options: BenchmarkOptions) -> int:
    """Execute a benchmark run against CHURRO-DS."""
    validation_status = _validate_options(options)
    if validation_status != 0:
        return validation_status

    clients_closed = False
    try:
        dataset_stream = _load_dataset(CHURRO_DATASET_ID, split=options.dataset_split)
        dataset = _selected_dataset_examples(dataset_stream, options)
        total_pages = getattr(dataset, "num_rows", None)
        if not isinstance(total_pages, int):
            total_pages = None

        output_prefix = create_output_prefix(options)
        start_time = time()
        evaluation_examples, predictions = await _predict_texts(
            dataset,
            options,
            total_pages=total_pages,
        )
        elapsed_time = time() - start_time
        await close_litellm_async_clients()
        clients_closed = True

        assert len(evaluation_examples) == len(predictions), (
            f"Mismatch in dataset size ({len(evaluation_examples)}) and predictions ({len(predictions)})."
        )
        compute_metrics(evaluation_examples, predictions, output_prefix, elapsed_time)
        return 0
    finally:
        if not clients_closed:
            await close_litellm_async_clients()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for repo-local benchmarking."""
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

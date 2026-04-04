"""CHURRO dataset selection helpers for benchmark evaluation runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

from tooling.evaluation.types import BenchmarkDatasetExample, EvaluationExample


def _normalize_filter_value(value: str | None) -> str | None:
    """Normalize subset filter values for case-insensitive matching."""
    if value is None:
        return None
    normalized_value = value.strip()
    return normalized_value.casefold() if normalized_value else None


def load_dataset_split(
    dataset_id: str,
    split: str,
    *,
    columns: Sequence[str] | None = None,
) -> Any:
    """Load one dataset split directly from its parquet shards."""
    from datasets import Features, load_dataset, load_dataset_builder

    builder = load_dataset_builder(dataset_id)
    split_files = getattr(builder.config, "data_files", {}).get(split)
    if not split_files:
        return load_dataset(dataset_id, split=split)

    load_kwargs: dict[str, object] = {
        "data_files": {split: split_files},
        "split": split,
    }
    if columns is not None:
        load_kwargs["columns"] = list(columns)
        features = getattr(builder.info, "features", None)
        if features is not None:
            load_kwargs["features"] = Features({column: features[column] for column in columns})
    return load_dataset("parquet", **load_kwargs)


@dataclass(frozen=True, slots=True)
class DatasetSubset:
    """Normalized subset filters applied to the CHURRO dataset stream."""

    language: str | None = None
    document_type: str | None = None

    @classmethod
    def from_raw(cls, *, language: str | None, document_type: str | None) -> DatasetSubset:
        """Create a normalized subset from raw CLI option values."""
        return cls(
            language=_normalize_filter_value(language),
            document_type=_normalize_filter_value(document_type),
        )

    def is_active(self) -> bool:
        """Return whether any subset filter is active."""
        return self.language is not None or self.document_type is not None

    def matches(self, example: EvaluationExample | BenchmarkDatasetExample) -> bool:
        """Return whether an example belongs to this subset."""
        if self.language is not None and _normalize_filter_value(example["main_language"]) != self.language:
            return False
        if (
            self.document_type is not None
            and _normalize_filter_value(example["document_type"]) != self.document_type
        ):
            return False
        return True

    def output_suffixes(self) -> list[str]:
        """Build stable directory suffixes for filtered benchmark runs."""
        suffixes: list[str] = []
        if self.language is not None:
            suffixes.append(f"language_{self.language.replace(' ', '_')}")
        if self.document_type is not None:
            suffixes.append(f"document_type_{self.document_type.replace(' ', '_')}")
        return suffixes


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """Windowing and subset filters for one benchmark evaluation run."""

    subset: DatasetSubset
    offset: int = 0
    limit: int = 0

    def select(self, dataset_stream: Iterable[BenchmarkDatasetExample]) -> Iterable[BenchmarkDatasetExample]:
        """Yield the requested dataset subset without materializing it upfront."""
        if hasattr(dataset_stream, "filter") and hasattr(dataset_stream, "select"):
            return self._select_materialized_dataset(dataset_stream)

        filtered_stream = (example for example in dataset_stream if self.subset.matches(example))
        end_index = self.offset + self.limit if self.limit > 0 else None
        return islice(filtered_stream, self.offset, end_index)

    def _select_materialized_dataset(self, dataset: Any) -> Any:
        """Apply subset filters and slicing to a materialized HF dataset."""
        selected = dataset
        if self.subset.is_active():
            selected = selected.filter(
                self._matches_materialized_row,
                input_columns=["main_language", "document_type"],
            )

        if self.offset <= 0 and self.limit <= 0:
            return selected

        total_rows = getattr(selected, "num_rows", None)
        if not isinstance(total_rows, int):
            return selected

        start_index = min(self.offset, total_rows)
        end_index = total_rows if self.limit <= 0 else min(start_index + self.limit, total_rows)
        return selected.select(range(start_index, end_index))

    def _matches_materialized_row(self, main_language: str, document_type: str) -> bool:
        """Return whether one materialized row matches the active subset filters."""
        if self.subset.language is not None and _normalize_filter_value(main_language) != self.subset.language:
            return False
        if (
            self.subset.document_type is not None
            and _normalize_filter_value(document_type) != self.subset.document_type
        ):
            return False
        return True

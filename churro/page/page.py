"""Core page abstractions and helpers for OCR layout post-processing."""

from collections import defaultdict
import math

from azure.ai.documentintelligence.models import AnalyzeResult
from pydantic import BaseModel, ConfigDict, Field
import rtree

from churro.utils.log_utils import logger

from .page_object import PageObject


class Page(BaseModel):
    """Represents a single OCR page with its objects, reading order, and metadata.

    This model stores detected layout elements (`page_objects`), optional reading order
    (as a directed graph), raw/full text, and auxiliary metadata such as languages.
    Utility methods provide convenience accessors for path-derived attributes and
    transformations (filtering, merging, saving, etc.).
    """

    page_objects: list[PageObject] = Field(..., description="List of objects on the page")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def from_azure_analysis_result(result: AnalyzeResult, skip_paragraphs: bool = False) -> "Page":
        """Build a `Page` instance from an Azure Document Intelligence result."""
        page_objects = []

        object_id = 1

        paragraph_spans = []  # each span is (offset, length)
        if result.paragraphs and not skip_paragraphs:
            for paragraph in result.paragraphs:
                if not paragraph.bounding_regions or len(paragraph.bounding_regions) != 1:
                    continue
                for span in paragraph.spans or []:
                    paragraph_spans.append((span.offset, span.length))
                polygon_coords = paragraph.bounding_regions[0].polygon or []
                if not polygon_coords:
                    continue
                page_objects.append(
                    PageObject(
                        object_id=str(object_id),
                        coordinates=polygon_coords,
                    )
                )
                object_id += 1
        if result.figures:
            for figure in result.figures:
                if not figure.bounding_regions or len(figure.bounding_regions) != 1:
                    continue
                fig_coords = figure.bounding_regions[0].polygon or []
                if not fig_coords:
                    continue
                page_objects.append(
                    PageObject(
                        object_id=str(object_id),
                        coordinates=fig_coords,
                    )
                )
                object_id += 1

        assert len(result.pages) == 1, "Expected only one page"
        p = result.pages[0]

        # use spans to determine which lines are already included in paragraphs
        lines_added = 0
        for line in p.lines or []:
            for span in line.spans:
                # check if span fully falls within a paragraph
                found = False
                for offset, length in paragraph_spans:
                    if span.offset >= offset and span.offset + span.length <= offset + length:
                        found = True
                        break
                if not found:
                    lines_added += 1
                    if line.polygon:
                        page_objects.append(
                            PageObject(
                                object_id=str(object_id),
                                coordinates=line.polygon,
                            )
                        )
                        object_id += 1

        # page objects with weird angles are often mistakes. We remove them here so that they can be added as their individual lines below
        old_size = len(page_objects)
        page_objects = [po for po in page_objects if po.get_top_edge_angle() <= 10]
        if old_size - len(page_objects) > 0:
            logger.info(f"Removed {old_size - len(page_objects)} page objects with weird angles")

        if lines_added > 0:
            logger.info(f"Added {lines_added} lines that were not part of any paragraphs to page")

        page = Page(
            page_objects=page_objects,
        )
        return page

    def remove_subsumed_page_objects(self, coverage_ratio: float = 0.7) -> None:
        """Remove page_objects that are subsumed by other page_objects using an R-tree for efficient spatial queries.

        When coverage_ratio is 100.0 (default), an object is removed if it is fully subsumed (using the default
        PageObject.remove_subsumed_objects logic). If a lower ratio is provided, an object is removed if at least
        that ratio of its area is covered by any other page object.

        Args:
            coverage_ratio (float): The coverage ratio for determining if a page object is subsumed by another. A value of 0.7 means
                that a page object is considered subsumed if 70% of its area is covered by another page object.
        """
        if not self.page_objects:
            return
        assert 0 <= coverage_ratio <= 1, "coverage_ratio must be between 0 and 1"

        # Record (index, page object) pairs.
        enumerated_polygons = [(i, obj) for i, obj in enumerate(self.page_objects)]
        page_objects = [p for _, p in enumerated_polygons]

        # Remove subsumed polygons; returns a list of polygon objects in the original order.
        polygons_to_keep = PageObject.remove_subsumed_objects(
            page_objects, tolerance=1 - coverage_ratio
        )

        # Map each polygon object back to its original indices.
        polygon_to_indices = defaultdict(list)
        for i, poly in enumerated_polygons:
            polygon_to_indices[poly].append(i)

        kept_indices = []
        for poly in polygons_to_keep:
            # Pop the first stored index for this polygon in case of duplicates.
            kept_indices.append(polygon_to_indices[poly].pop(0))
        kept_indices.sort()
        self.page_objects = [self.page_objects[i] for i in kept_indices]

    def remove_small_page_objects_in_margins(self) -> None:
        """Drop tiny page objects hugging the margins to reduce noise."""
        sample_count = len(self.page_objects)
        if sample_count < 2:
            return

        index = rtree.index.Index()
        # Spatial index accelerates intersection checks against margin bands.
        bounds: list[tuple[float, float, float, float]] = []
        widths: list[float] = []
        heights: list[float] = []
        for idx, obj in enumerate(self.page_objects):
            left, top, right, bottom = obj.bounds
            bounds.append((left, top, right, bottom))
            widths.append(obj.width)
            heights.append(obj.height)
            index.insert(idx, (left, top, right, bottom))

        smallest_count = max(1, math.ceil(sample_count * 0.1))
        smallest_width_indices = set(
            sorted(range(sample_count), key=lambda i: widths[i])[:smallest_count]
        )
        smallest_height_indices = set(
            sorted(range(sample_count), key=lambda i: heights[i])[:smallest_count]
        )

        global_left = min(b[0] for b in bounds)
        global_top = min(b[1] for b in bounds)
        global_right = max(b[2] for b in bounds)
        global_bottom = max(b[3] for b in bounds)

        page_width = max(global_right - global_left, 1.0)
        page_height = max(global_bottom - global_top, 1.0)
        x_tolerance = max(5.0, page_width * 0.02)
        y_tolerance = max(5.0, page_height * 0.02)

        def margin_rectangles() -> list[tuple[float, float, float, float]]:
            return [
                (
                    max(global_left, global_right - x_tolerance),
                    global_top,
                    global_right,
                    global_bottom,
                ),
                (
                    global_left,
                    global_top,
                    min(global_right, global_left + x_tolerance),
                    global_bottom,
                ),
                (
                    global_left,
                    global_top,
                    global_right,
                    min(global_bottom, global_top + y_tolerance),
                ),
                (
                    global_left,
                    max(global_top, global_bottom - y_tolerance),
                    global_right,
                    global_bottom,
                ),
            ]

        def is_on_margin(obj_bounds: tuple[float, float, float, float]) -> bool:
            left, top, right, bottom = obj_bounds
            return (
                (left - global_left) <= x_tolerance
                or (global_right - right) <= x_tolerance
                or (top - global_top) <= y_tolerance
                or (global_bottom - bottom) <= y_tolerance
            )

        candidate_indices: set[int] = set()
        for rect in margin_rectangles():
            if rect[0] > rect[2] or rect[1] > rect[3]:
                continue
            candidate_indices.update(index.intersection(rect))

        indices_to_remove = {
            idx
            for idx in candidate_indices
            if idx in smallest_width_indices
            and idx in smallest_height_indices
            and is_on_margin(bounds[idx])
        }
        # Keep only elements that are both tiny and sit within a margin band.

        if not indices_to_remove:
            return

        if len(indices_to_remove) == sample_count:
            candidate_to_keep = max(
                indices_to_remove,
                key=lambda idx_: widths[idx_] * heights[idx_],
            )
            indices_to_remove.remove(candidate_to_keep)
            if not indices_to_remove:
                return

        removed_count = len(indices_to_remove)
        self.page_objects = [
            obj for idx, obj in enumerate(self.page_objects) if idx not in indices_to_remove
        ]

        logger.info(f"Removed {removed_count} small margin page objects out of {sample_count}")

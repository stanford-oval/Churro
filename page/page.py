from collections import defaultdict
from typing import Optional

from azure.ai.documentintelligence.models import AnalyzeResult
from pydantic import BaseModel, ConfigDict, Field

from page.bounding_box_element import (
    PageObject,
)
from page.polygon import Polygon
from utils.log_utils import logger


class Page(BaseModel):
    """Represents a single OCR page with its objects, reading order, and metadata.

    This model stores detected layout elements (`page_objects`), optional reading order
    (as a directed graph), raw/full text, and auxiliary metadata such as languages.
    Utility methods provide convenience accessors for path-derived attributes and
    transformations (filtering, merging, saving, etc.).
    """

    page_objects: list[PageObject] = Field(..., description="List of objects on the page")

    image_path: Optional[str] = Field(
        None,
        description="Path to the image file of the page.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata associated with the page. E.g. languages",
    )

    def get_directory(self) -> str:
        """Get the directory of the image path.

        Returns an empty string if the image path is unset.
        """
        if not self.image_path:
            return ""
        return "/".join(self.image_path.split("/")[:-1])

    def get_json_path(self) -> str:
        """Return the JSON side-car path for this page's image."""
        if not self.image_path:
            raise ValueError("Image path is not set, cannot determine JSON path.")
        return self.image_path.replace(".jpeg", ".json")

    @property
    def languages(self) -> list[str]:
        """Return list of language tags associated with this page (may be empty)."""
        return self.metadata.get("languages", [])

    @languages.setter
    def languages(self, value: list[str]) -> None:
        self.metadata["languages"] = value

    @property
    def main_language(self) -> Optional[str]:
        """Return the first language or None if no languages are set."""
        if not self.languages:
            return None
        return self.languages[0]

    @property
    def document_type(self) -> Optional[str]:
        """Return document type stored in metadata if present."""
        return self.metadata.get("document_type", None)

    @document_type.setter
    def document_type(self, value: str) -> None:
        """Set the document type in the metadata."""
        self.metadata["document_type"] = value

    @property
    def scripts(self) -> Optional[list[str]]:
        """Return list of script tags for the page if present."""
        return self.metadata.get("scripts", None)

    @scripts.setter
    def scripts(self, value: list[str]) -> None:
        self.metadata["scripts"] = value

    @property
    def main_script(self) -> Optional[str]:
        """Return the first script or None if no scripts are set."""
        if not self.scripts:
            return None
        return self.scripts[0]

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
                        bounding_region=Polygon(coordinates=polygon_coords),
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
                        bounding_region=Polygon(coordinates=fig_coords),
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
                                bounding_region=Polygon(coordinates=line.polygon),
                            )
                        )
                        object_id += 1

        # page objects with weird angles are often mistakes. We remove them here so that they can be added as their individual lines below
        old_size = len(page_objects)
        page_objects = [po for po in page_objects if po.bounding_region.get_top_edge_angle() <= 10]
        if old_size - len(page_objects) > 0:
            logger.info(f"Removed {old_size - len(page_objects)} page objects with weird angles")

        if lines_added > 0:
            logger.info(f"Added {lines_added} lines that were not part of any paragraphs to page")

        page = Page(
            page_objects=page_objects,
            image_path=None,
        )
        return page

    def remove_subsumed_page_objects(self, coverage_ratio: float = 0.8) -> None:
        """Remove page_objects that are subsumed by other page_objects using an R-tree for efficient spatial queries.

        When coverage_ratio is 100.0 (default), an object is removed if it is fully subsumed (using the default
        Polygon.remove_subsumed_polygons logic). If a lower ratio is provided, an object is removed if at least
        that ratio of its area is covered by any other page object.

        Args:
            coverage_ratio (float): The coverage ratio for determining if a page object is subsumed by another. A value of 0.8 means
                that a page object is considered subsumed if 80% of its area is covered by another page object.
        """
        if not self.page_objects:
            return
        assert 0 <= coverage_ratio <= 1, "coverage_ratio must be between 0 and 1"

        # Record (index, bounding_region) pairs.
        enumerated_polygons = [(i, obj.bounding_region) for i, obj in enumerate(self.page_objects)]
        polygons = [p for _, p in enumerated_polygons]

        # Remove subsumed polygons; returns a list of polygon objects in the original order.
        polygons_to_keep = Polygon.remove_subsumed_polygons(polygons, tolerance=1 - coverage_ratio)

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

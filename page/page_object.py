from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
import rtree
from shapely.affinity import translate
from shapely.geometry import Polygon as ShapelyPolygon

from churro.utils.log_utils import logger


def _rotate_point(
    px: float, py: float, angle: float, center: tuple[float, float]
) -> tuple[float, float]:
    """Rotate (px, py) around center by angle degrees."""
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    ox, oy = center
    dx = px - ox
    dy = py - oy
    qx = ox + cos_a * dx - sin_a * dy
    qy = oy + sin_a * dx + cos_a * dy
    return qx, qy


class PageObject(BaseModel):
    object_id: str
    coordinates: list[float]
    text: str | None = None

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )

    _shapely_polygon: ShapelyPolygon = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def convert_llm_ocr_text(cls, data: dict[str, object]) -> dict[str, object]:
        if isinstance(data, dict) and data.get("text") is None and "llm_ocr_text" in data:
            data["text"] = data.pop("llm_ocr_text")
        return data

    @model_validator(mode="after")
    def initialize_polygon(self) -> PageObject:
        self._set_polygon(self.coordinates)
        return self

    def _set_polygon(self, coordinates: list[float]) -> None:
        if not coordinates or len(coordinates) % 2 != 0:
            raise ValueError("Coordinates must contain an even number of values.")
        points = [
            (float(coordinates[i]), float(coordinates[i + 1]))
            for i in range(0, len(coordinates), 2)
        ]
        if points[0] != points[-1]:
            points.append(points[0])

        poly = ShapelyPolygon(shell=points)
        # Round coordinates for consistency
        int_coords = [(int(round(x)), int(round(y))) for x, y in poly.exterior.coords]
        poly = ShapelyPolygon(shell=int_coords)

        self._shapely_polygon = poly
        flattened = [coord for point in poly.exterior.coords for coord in point]
        object.__setattr__(self, "coordinates", flattened)

    def update_coordinates(self, coordinates: list[float]) -> None:
        """Replace the polygon coordinates and refresh the cached Shapely polygon."""
        self._set_polygon(coordinates)

    @property
    def width(self) -> float:
        minx, _, maxx, _ = self._shapely_polygon.bounds
        return maxx - minx

    @property
    def height(self) -> float:
        _, miny, _, maxy = self._shapely_polygon.bounds
        return maxy - miny

    @property
    def left(self) -> float:
        return self._shapely_polygon.bounds[0]

    @property
    def right(self) -> float:
        return self._shapely_polygon.bounds[2]

    @property
    def top(self) -> float:
        return self._shapely_polygon.bounds[1]

    @property
    def bottom(self) -> float:
        return self._shapely_polygon.bounds[3]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    @property
    def area(self) -> float:
        return self._shapely_polygon.area

    def get_top_edge_angle(self) -> float:
        coords = list(self._shapely_polygon.exterior.coords)
        edges = zip(coords, coords[1:], strict=False)
        top_edge = max(edges, key=lambda edge: (edge[0][1] + edge[1][1]) / 2)
        (x1, y1), (x2, y2) = top_edge
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        angle = angle % 90
        return min(angle, 90 - angle)

    def contains(self, other: PageObject, tolerance: float = 0.05) -> bool:
        intersection = self._shapely_polygon.intersection(other._shapely_polygon)
        outside_area = other._shapely_polygon.area - intersection.area
        if other._shapely_polygon.area == 0:
            return False
        return (outside_area / other._shapely_polygon.area) <= tolerance

    def rotate(
        self,
        angle: float,
        center: tuple[float, float],
        offset_x: float,
        offset_y: float,
    ) -> None:
        rotated = (
            _rotate_point(x, y, angle, center)
            for x, y in zip(self.coordinates[::2], self.coordinates[1::2], strict=False)
        )
        coords = [
            coord + offset
            for point in rotated
            for coord, offset in zip(point, (offset_x, offset_y), strict=False)
        ]
        self._set_polygon(coords)

    def relative_coordinates(self) -> list[tuple[float, float]]:
        """Return polygon coordinates translated to the bounding box origin."""
        shift_x = self.left
        shift_y = self.top
        shifted = translate(self._shapely_polygon, xoff=-shift_x, yoff=-shift_y)
        return [(int(round(x)), int(round(y))) for x, y in shifted.exterior.coords]

    @staticmethod
    def all_encompassing_rectangle(
        page_objects: list[PageObject],
        object_id: str | None = None,
    ) -> PageObject:
        """Return a rectangle covering all provided page objects.

        Useful for creating smaller, more focused visualizations.
        """
        assert page_objects, "Cannot create an all-encompassing rectangle with no page objects"

        minx, miny, maxx, maxy = (
            page_objects[0].left,
            page_objects[0].top,
            page_objects[0].right,
            page_objects[0].bottom,
        )

        for obj in page_objects[1:]:
            b_left, b_top, b_right, b_bottom = obj.left, obj.top, obj.right, obj.bottom
            minx, miny = min(minx, b_left), min(miny, b_top)
            maxx, maxy = max(maxx, b_right), max(maxy, b_bottom)

        coords = [minx, miny, maxx, miny, maxx, maxy, minx, maxy]
        rect_id = object_id or f"{page_objects[0].object_id}-encompassing"
        return PageObject(object_id=rect_id, coordinates=coords)

    @staticmethod
    def from_bounds(
        left: float,
        top: float,
        right: float,
        bottom: float,
        object_id: str = "bounds",
        text: str | None = None,
    ) -> PageObject:
        coords = [left, top, right, top, right, bottom, left, bottom]
        return PageObject(object_id=object_id, coordinates=coords, text=text)

    @staticmethod
    def remove_subsumed_objects(
        page_objects: list[PageObject], tolerance: float = 0.2
    ) -> list[PageObject]:
        if not page_objects:
            return page_objects

        indexed_page_objects = sorted(
            enumerate(page_objects), key=lambda pair: pair[1].area, reverse=True
        )

        idx = rtree.index.Index()
        page_objects_to_keep: list[tuple[int, PageObject]] = []

        for original_idx, page_object in indexed_page_objects:
            bounds = page_object.bounds
            potential_subsumers = list(idx.intersection(bounds))

            for sub_idx in potential_subsumers:
                _, other_page_object = page_objects_to_keep[sub_idx]
                if other_page_object.contains(page_object, tolerance=tolerance):
                    break
            else:
                page_objects_to_keep.append((original_idx, page_object))
                idx.insert(len(page_objects_to_keep) - 1, bounds)

        removed_count = len(page_objects) - len(page_objects_to_keep)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} subsumed page objects out of {len(page_objects)}")

        page_objects_to_keep.sort(key=lambda pair: pair[0])

        return [page_object for _, page_object in page_objects_to_keep]

    def __hash__(self) -> int:
        """Return a stable hash for comparing page objects in sets and dicts."""
        return hash((self.object_id, tuple(self.coordinates), self.text))

    def __eq__(self, other: object) -> bool:
        """Return True if the other page object matches by id, text, and coordinates."""
        if not isinstance(other, PageObject):
            return False
        return (
            self.object_id == other.object_id
            and self.text == other.text
            and tuple(self.coordinates) == tuple(other.coordinates)
        )

    def __repr__(self) -> str:
        """Return a concise string representation for debugging the page object."""
        return (
            f"PageObject(id={self.object_id}, left={self.left}, top={self.top}, "
            f"right={self.right}, bottom={self.bottom})"
        )

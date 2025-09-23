from dataclasses import dataclass

import numpy as np
import rtree
from shapely.affinity import translate
from shapely.geometry import Polygon as ShapelyPolygon

from utils.log_utils import logger


def _rotate_point(
    px: float, py: float, angle: float, center: tuple[float, float]
) -> tuple[float, float]:
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    ox, oy = center
    dx = px - ox
    dy = py - oy
    qx = ox + cos_a * dx - sin_a * dy
    qy = oy + sin_a * dx + cos_a * dy
    return qx, qy


@dataclass
class Polygon:
    """A wrapper around Shapely Polygon to provide additional functionality and serialization.

    The 'coordinates' field is a flat list of floats [x1, y1, x2, y2, ..., xn, yn].
    The polygon will be closed automatically if not already.
    """

    def __init__(self, coordinates: list[float]):
        if not coordinates or len(coordinates) % 2 != 0:
            raise ValueError("Coordinates must contain an even number of values.")
        # Convert coordinates into a list of points
        points = [
            (float(coordinates[i]), float(coordinates[i + 1]))
            for i in range(0, len(coordinates), 2)
        ]
        if points[0] != points[-1]:
            points.append(points[0])

        # Create initial Shapely polygon
        poly = ShapelyPolygon(shell=points)

        # Round coordinates for consistency
        int_coords = [(int(round(x)), int(round(y))) for x, y in poly.exterior.coords]
        poly = ShapelyPolygon(shell=int_coords)

        self._shapely_polygon: ShapelyPolygon = poly

    def __repr__(self) -> str:
        """Return a concise string representation with coordinates."""
        return f"Polygon(coordinates={self.coordinates})"

    @property
    def coordinates(self) -> list[float]:
        """Return a copy of the polygon's coordinates as a flat list of floats."""
        return [coord for point in self._shapely_polygon.exterior.coords for coord in point]

    @property
    def width(self) -> float:
        """Width of the polygon's bounding box."""
        minx, _, maxx, _ = self._shapely_polygon.bounds
        return maxx - minx

    @property
    def height(self) -> float:
        """Height of the polygon's bounding box."""
        _, miny, _, maxy = self._shapely_polygon.bounds
        return maxy - miny

    @property
    def left(self) -> float:
        """Left (minimum x) bound."""
        return self._shapely_polygon.bounds[0]

    @property
    def right(self) -> float:
        """Right (maximum x) bound."""
        return self._shapely_polygon.bounds[2]

    @property
    def top(self) -> float:
        """Top (minimum y) bound."""
        return self._shapely_polygon.bounds[1]

    @property
    def bottom(self) -> float:
        """Bottom (maximum y) bound."""
        return self._shapely_polygon.bounds[3]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)

    @property
    def area(self) -> float:
        """Area of the polygon."""
        return self._shapely_polygon.area

    @staticmethod
    def from_bounds(left: float, top: float, right: float, bottom: float) -> "Polygon":
        """Create a Polygon from bounding box coordinates."""
        return Polygon(coordinates=[left, top, right, top, right, bottom, left, bottom])

    @staticmethod
    def remove_subsumed_polygons(
        polygons: list["Polygon"], tolerance: float = 0.2
    ) -> list["Polygon"]:
        if not polygons:
            return polygons

        # Enumerate and sort by descending area
        indexed_polygons = sorted(enumerate(polygons), key=lambda pair: pair[1].area, reverse=True)

        # Initialize R-tree index
        idx = rtree.index.Index()
        # Keep (original_index, polygon)
        polygons_to_keep: list[tuple[int, Polygon]] = []

        for original_idx, polygon in indexed_polygons:
            bounds = polygon.bounds
            # Potential polygons that might subsume this one
            potential_subsumers = list(idx.intersection(bounds))

            # Check if this polygon is subsumed
            for sub_idx in potential_subsumers:
                # sub_idx corresponds to position in polygons_to_keep
                _, other_polygon = polygons_to_keep[sub_idx]
                if other_polygon.contains(polygon, tolerance=tolerance):
                    # Polygon is subsumed; skip it
                    break
            else:
                # If not subsumed, add it to keep list and R-tree index
                polygons_to_keep.append((original_idx, polygon))
                idx.insert(len(polygons_to_keep) - 1, bounds)

        # Report how many were removed
        removed_count = len(polygons) - len(polygons_to_keep)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} subsumed polygons out of {len(polygons)}")

        # Sort back by original index to preserve input order
        polygons_to_keep.sort(key=lambda pair: pair[0])

        # Return just the Polygon objects, in the original order
        return [polygon for _, polygon in polygons_to_keep]

    def get_top_edge_angle(self) -> float:
        coords = list(self._shapely_polygon.exterior.coords)
        edges = zip(coords, coords[1:])

        # Identify the edge with the highest average y-coordinate
        top_edge = max(edges, key=lambda edge: (edge[0][1] + edge[1][1]) / 2)
        (x1, y1), (x2, y2) = top_edge

        # convert to angle between 0 and 180
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

        # get the difference from the normal orientation of a rectangle
        angle = angle % 90
        return min(angle, 90 - angle)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polygon):
            return False
        return self._shapely_polygon.equals(other._shapely_polygon)

    def __hash__(self) -> int:
        return hash(self._shapely_polygon)

    def contains(self, other: "Polygon", tolerance: float = 0.05) -> bool:
        """Return True if this polygon contains another within a tolerance.

        Tolerance is the ratio of the other polygon's non-overlapping area.
        """
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
    ) -> "Polygon":
        rotated = (
            _rotate_point(x, y, angle, center)
            for x, y in zip(self.coordinates[::2], self.coordinates[1::2])
        )
        coordinates = [
            coord + offset
            for point in rotated
            for coord, offset in zip(point, (offset_x, offset_y))
        ]
        return Polygon(coordinates=coordinates)

    def shift_to_coordinates(self, x: float, y: float) -> "Polygon":
        """Shift polygon so its top-left corner becomes (x, y)."""
        shift_x = x - self.left
        shift_y = y - self.top
        shifted = translate(self._shapely_polygon, xoff=shift_x, yoff=shift_y)
        coords = [coord for point in shifted.exterior.coords for coord in point]
        return Polygon(coordinates=coords)

    def shift_by_amount(self, x_shift: float, y_shift: float) -> "Polygon":
        """Shift polygon by ``x_shift`` horizontally and ``y_shift`` vertically."""
        shifted = translate(self._shapely_polygon, xoff=x_shift, yoff=y_shift)
        coords = [coord for point in shifted.exterior.coords for coord in point]
        return Polygon(coordinates=coords)

    @property
    def is_empty(self) -> bool:
        return self._shapely_polygon.is_empty

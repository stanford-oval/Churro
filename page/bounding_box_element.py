from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from page.polygon import Polygon


class PageObject(BaseModel):
    object_id: str
    bounding_region: Polygon
    text: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    def convert_llm_ocr_text(cls, data):
        if isinstance(data, dict) and data.get("text") is None and "llm_ocr_text" in data:
            data["text"] = data.pop("llm_ocr_text")
        return data

    @property
    def top(self) -> float:
        return self.bounding_region.top

    @property
    def left(self) -> float:
        return self.bounding_region.left

    @property
    def right(self) -> float:
        return self.bounding_region.right

    @property
    def bottom(self) -> float:
        return self.bounding_region.bottom

    @staticmethod
    def all_encompassing_rectangle(
        page_objects: list["PageObject"] | list[Polygon],
    ) -> Polygon:
        """Return a rectangle covering all provided page objects.

        Useful for creating smaller, more focused visualizations.
        """
        assert page_objects, "Cannot create an all-encompassing rectangle with no page objects"

        # Get initial min/max from the first bounding region
        minx, miny, maxx, maxy = (
            page_objects[0].left,
            page_objects[0].top,
            page_objects[0].right,
            page_objects[0].bottom,
        )

        # Expand the bounding box by checking each page_object
        for obj in page_objects[1:]:
            bx, by, bX, bY = obj.left, obj.top, obj.right, obj.bottom
            minx, miny = min(minx, bx), min(miny, by)
            maxx, maxy = max(maxx, bX), max(maxy, bY)

        # Construct a rectangular polygon from the final bounding box
        return Polygon(coordinates=[minx, miny, maxx, miny, maxx, maxy, minx, maxy])

    def __hash__(self) -> int:
        return hash(self.bounding_region)

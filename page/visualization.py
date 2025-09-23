from PIL import Image, ImageDraw

from page.bounding_box_element import (
    PageObject,
    Polygon,
)
from page.page import Page


def extract_polygon_region(
    image: Image.Image,
    polygon: Polygon,
) -> Image.Image:
    """Crop a PIL image using a Polygon.

    Parameters:
        image (PIL.Image.Image): The input image to crop.
        polygon (Polygon): The polygon defining the crop area.
        background_color (tuple[int, int, int]): The background color to use for areas outside the polygon.

    Returns:
        PIL.Image.Image: The cropped image with the polygon applied as a mask.
    """
    # Extract bounding box dimensions
    width = int(polygon.width)
    height = int(polygon.height)
    min_x = int(polygon.left)
    min_y = int(polygon.top)

    # Create a mask image with the same size as the bounding box
    mask = Image.new("L", (width, height), 0)

    # Adjust polygon coordinates relative to the bounding box
    relative_polygon = polygon.shift_to_coordinates(x=0, y=0)

    # Draw the polygon on the mask
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(relative_polygon.coordinates, fill=255)

    # Crop the original image to the bounding box
    cropped_image = image.crop((min_x, min_y, min_x + width, min_y + height))

    # Create a new image
    result_image = Image.new("RGB", (width, height), color=(255, 255, 255))  # type: ignore

    # Paste the cropped image onto the result image using the mask
    result_image.paste(cropped_image, (0, 0), mask)

    return result_image


def crop_page_objects_from_image(
    page_objects: list[PageObject],
    original_image: Image.Image,
) -> list[Image.Image]:
    cropped_images = []

    for o in page_objects:
        polygon = o.bounding_region
        cropped_img = extract_polygon_region(
            original_image,
            polygon,
        )

        cropped_images.append(cropped_img)

    return cropped_images


def crop_image_and_page_to_content(
    image: Image.Image, page: Page, margin: int = 0
) -> tuple[Image.Image, int, int]:
    """Remove empty margins around page content.

    The function detects the minimal bounding rectangle over all page objects and
    crops the image to it, optionally expanding by ``margin``.
    """
    if margin < 0:
        raise ValueError("Margin must be a non-negative integer.")

    polygon = PageObject.all_encompassing_rectangle(page.page_objects)
    if margin > 0:
        left, top, right, bottom = polygon.bounds
        left = max(left - margin, 0)
        top = max(top - margin, 0)
        right = min(right + margin, image.width)
        bottom = min(bottom + margin, image.height)
        polygon = Polygon.from_bounds(left, top, right, bottom)
    left, top = int(polygon.left), int(polygon.top)
    return (
        extract_polygon_region(image, polygon),
        left,
        top,
    )

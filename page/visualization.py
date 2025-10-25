from PIL import Image, ImageDraw

from .page_object import PageObject


def extract_polygon_region(
    image: Image.Image,
    page_object: PageObject,
) -> Image.Image:
    """Crop a PIL image using a PageObject's polygon.

    Parameters:
        image (PIL.Image.Image): The input image to crop.
        page_object (PageObject): The polygon defining the crop area.
        background_color (tuple[int, int, int]): The background color to use for areas outside the polygon.

    Returns:
        PIL.Image.Image: The cropped image with the polygon applied as a mask.
    """
    # Extract bounding box dimensions
    width = int(page_object.width)
    height = int(page_object.height)
    min_x = int(page_object.left)
    min_y = int(page_object.top)

    # Create a mask image with the same size as the bounding box
    mask = Image.new("L", (width, height), 0)

    # Adjust polygon coordinates relative to the bounding box
    relative_points = page_object.relative_coordinates()

    # Draw the polygon on the mask
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(relative_points, fill=255)

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

    for page_object in page_objects:
        cropped_img = extract_polygon_region(
            original_image,
            page_object,
        )

        cropped_images.append(cropped_img)

    return cropped_images


def crop_image_to_objects(
    image: Image.Image, page_objects: list[PageObject], margin: int = 0
) -> Image.Image:
    """Remove empty margins around page content.

    The function detects the minimal bounding rectangle over all page objects and
    crops the image to it, optionally expanding by ``margin``.
    """
    if margin < 0:
        raise ValueError("Margin must be a non-negative integer.")

    polygon = PageObject.all_encompassing_rectangle(page_objects)
    if margin > 0:
        left, top, right, bottom = polygon.bounds
        left = max(left - margin, 0)
        top = max(top - margin, 0)
        right = min(right + margin, image.width)
        bottom = min(bottom + margin, image.height)
        polygon = PageObject.from_bounds(
            left,
            top,
            right,
            bottom,
            object_id=f"{polygon.object_id}-margin",
        )

    return extract_polygon_region(image, polygon)

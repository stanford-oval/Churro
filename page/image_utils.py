import numpy as np
from PIL.Image import Image, Resampling, fromarray

from .page import Page


def rotate_image_and_page(image: Image, angle: float, page: Page) -> tuple[Image, Page]:
    original_width, original_height = image.size

    # Rotate the image
    rotated_image = image.rotate(
        angle, resample=Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255)
    )
    rotated_width, rotated_height = rotated_image.size

    # Calculate the offset caused by the expansion
    offset_x = (rotated_width - original_width) / 2
    offset_y = (rotated_height - original_height) / 2

    # Get the center of the original image
    center = (original_width / 2, original_height / 2)

    if page:
        for page_object in page.page_objects:
            polygon = page_object.bounding_region
            polygon = polygon.rotate(-angle, center, offset_x, offset_y)
            page_object.bounding_region = polygon

    return (
        rotated_image,
        page,
    )


def adjust_image(
    image: Image,
    thresholding: bool = False,
) -> Image:
    """Preprocess a scanned newspaper image for OCR without OpenCV.

    Steps:
        1. Convert to grayscale (Pillow).
        2. (Optional) Apply Sauvola threshold (skimage).
        3. Convert back to original mode (e.g. RGB) for downstream consistency.
    """
    from skimage.filters import threshold_sauvola
    from skimage.util import img_as_ubyte

    original_mode = image.mode

    # Grayscale via Pillow
    gray_image = image.convert("L")

    if thresholding:
        gray_arr = np.array(gray_image)
        # Sauvola thresholding
        thresh = threshold_sauvola(gray_arr, window_size=15, k=0.2)
        binarized = img_as_ubyte(gray_arr > thresh)
        gray_image = fromarray(binarized)

    # Restore original mode if needed
    if gray_image.mode != original_mode:
        gray_image = gray_image.convert(original_mode)

    return gray_image

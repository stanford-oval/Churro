import numpy as np
from PIL import Image

from churro.page.page import Page
from churro.utils.log_utils import logger


def resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Resize to fit inside (``max_width``, ``max_height``) maintaining aspect ratio if larger."""
    original_width, original_height = image.size

    # Only scale if the image is larger than the max dimensions
    if original_width <= max_width and original_height <= max_height:
        return image

    scale = min(max_width / original_width, max_height / original_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    return image.resize(new_size, resample=Image.LANCZOS)  # type: ignore


def adjust_image(
    image: Image.Image,
    thresholding: bool = False,
) -> Image.Image:
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
        gray_image = Image.fromarray(binarized)

    # Restore original mode if needed
    if gray_image.mode != original_mode:
        gray_image = gray_image.convert(original_mode)

    return gray_image


def rotate_image_and_page(image: Image.Image, angle: float, page: Page) -> tuple[Image.Image, Page]:
    if angle == 0.0:
        return image, page

    if abs(angle) > 3.0:
        logger.info(f"Rotating page by {angle:.1f} degrees")

    original_width, original_height = image.size

    # Rotate the image
    rotated_image = image.rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255)
    )
    rotated_width, rotated_height = rotated_image.size

    # Calculate the offset caused by the expansion
    offset_x = (rotated_width - original_width) / 2
    offset_y = (rotated_height - original_height) / 2

    # Get the center of the original image
    center = (original_width / 2, original_height / 2)

    page_copy = page.model_copy(deep=True)
    for page_object in page_copy.page_objects:
        page_object.rotate(-angle, center, offset_x, offset_y)

    return (
        rotated_image,
        page_copy,
    )

"""Simple Pipeline OCR with layout+LLM post-processing via detect_layout."""

import asyncio
from functools import partial
from typing import override

from PIL import Image

from page.bounding_box_element import PageObject
from page.image_utils import adjust_image
from page.page import Page
from page.polygon import Polygon
from page.visualization import (
    crop_page_objects_from_image,
)
from utils.llm import run_llm_async
from utils.llm.utils import extract_tag_from_llm_output
from utils.utils import run_async_in_parallel

from .base_ocr import BaseOCR
from .detect_layout import detect_layout


class HybridPipelineOCR(BaseOCR):
    """OCR that calls Azure Document Intelligence via `detect_layout`.

    Leverages the Azure-backed analyzer in `detect_layout` to obtain OCR text
    and then refines each detected region with an LLM pass.
    """

    def __init__(
        self,
        engine: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.engine = engine

    async def llm_ocr(
        self,
        page_object: PageObject,
        object_image: Image.Image,
        include_original_text: bool,
    ) -> str:
        """Run LLM OCR on the given image and return the output in Markdown format."""
        instructions = f"""You are an AI assistant tasked with performing Optical Character Recognition (OCR) on historical newspapers. You are provided with an image showing a newspaper excerpt, and rectangle marked on it with red background. Follow these instructions:

        1. Perform OCR on the entirety of the red rectangle, converting all visible text inside the rectangle into Markdown format. If there is no text in the rectangle, output an empty string.

        2. Handle partial words, punctuation marks, quotation marks, and dashes correctly.

        3. If you come across text in languages other than English, transcribe it as accurately as possible without translation.
        
        4. If you encounter any issues or uncertainties during the OCR process, note them at the end of your output.

        {"5. The output of a weaker OCR engine is provided below for reference. You may use it as the basis of your output, and correct its errors." if include_original_text else ""}

        Your output should look like this:
        
        <ocr_result>
        Extracted Markdown
        </ocr_result>

        <ocr_notes>
        Any notes about potential issues, uncertainties, or areas that may need human review
        </ocr_notes>

      
        """ + (
            f"This is the text extracted from a weaker OCR engine:\n{page_object.text}"
            if include_original_text
            else ""
        )

        llm_output = await run_llm_async(
            model=self.engine,
            system_prompt_text=None,
            user_message_text=instructions,
            user_message_image=object_image,
        )

        ocr_result = extract_tag_from_llm_output(llm_output, "ocr_result")
        assert isinstance(ocr_result, str)

        return ocr_result

    async def llm_ocr_batch(
        self,
        original_image: Image.Image,
        page_objects: list[PageObject],
        include_original_text: bool,
    ):
        """Run LLM OCR concurrently for all page objects and update text in-place."""
        page_object_images = crop_page_objects_from_image(
            page_objects,
            original_image,
        )
        ocr_outputs = await run_async_in_parallel(
            partial(
                self.llm_ocr,
                include_original_text=include_original_text,
            ),
            page_objects,
            page_object_images,
            max_concurrency=50,
        )

        for page_object, ocr_output in zip(page_objects, ocr_outputs):
            page_object.text = ocr_output

    @override
    async def process_image(self, image: Image.Image) -> str:
        """Detect layout, run LLM OCR per box, and merge text.

        Steps:
        - Open and lightly preprocess the image
        - Detect layout via Azure DI (detect_layout)
        - Run LLM OCR in batch on all boxes, in-place updating PageObject.text
        - Merge texts in reading order
        """
        merged_text = ""

        image = adjust_image(image)  # converts to grayscale

        try:
            try:
                page, image, _ = await asyncio.wait_for(detect_layout(image), timeout=30)
            except asyncio.TimeoutError as exc:
                raise TimeoutError("detect_layout exceeded 30 seconds") from exc
        except Exception:
            # create a single PageObject as a fallback
            page = Page(
                page_objects=[
                    PageObject(
                        object_id="1",
                        bounding_region=Polygon.from_bounds(0, 0, image.width, image.height),
                        text=None,
                    )
                ],
                image_path=None,
            )

        # Run LLM OCR on each bounding box and replace text in-place
        await self.llm_ocr_batch(
            image,
            page.page_objects,
            include_original_text=False,
        )

        merged_text = "\n".join([o.text or "" for o in page.page_objects])
        return merged_text

    def get_system_name(self) -> str:
        """Return human-readable system name."""
        return "Hybrid Pipeline"

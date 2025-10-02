"""LLM-based OCR improvement implementation."""

from functools import partial
import textwrap
from typing import Literal, cast

from PIL import Image

from utils.llm import extract_tag_from_llm_output, run_llm_async
from utils.log_utils import logger
from utils.utils import resize_image_to_fit, run_async_in_parallel


class LLMImprover:
    """LLM-based OCR improvement using vision models."""

    def __init__(
        self,
        engine: str,
        resize: int | None = None,
        image_fidelity: Literal["high", "low"] = "high",
        backup_engine: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.engine = engine
        self.resize = resize
        self.image_fidelity = image_fidelity

        if not backup_engine:
            self.backup_engine = engine
        else:
            self.backup_engine = backup_engine

        self.system_prompt = (
            "You are an AI assistant specialized in improving and correcting OCR "
            "text from historical documents. Your task "
            "is to identify and fix OCR errors and formatting of the text."
            "Output your corrected text between <improved_text> and </improved_text> tags."
        )

        self.instruction_template = """You are provided with an image and corresponding OCR text that needs improvement.

        Follow these instructions:

        1. Carefully examine the provided image and compare it with the OCR text below.
        2. Identify and correct any OCR errors.
        3. Handle partial words, punctuation marks, and dashes correctly. You can merge words that are split across lines.
        4. If you encounter text in languages other than English, preserve it accurately without translation.
        5. Add appropriate Markdown formatting if clearly inferrable from the image. For example, use `**bold**` for bold text, `*italic*` for italic text, and `# Heading`, `## Subheading` for headings.

        Original OCR text to correct and improve:
        {ocr_text}

        Output your corrected text in the following format:

        <improved_text>
        [Your corrected and improved text here]
        </improved_text>"""

    async def process_batch_inputs(
        self, image_paths: list[str], texts: list[str], max_concurrency: int
    ) -> list[str]:
        """Process a batch of image-text pairs using LLM OCR improvement."""
        return await run_async_in_parallel(
            partial(self._process_single_input),
            image_paths,
            texts,
            max_concurrency=max_concurrency,
            desc="LLM OCR Improvement",
        )

    async def _process_single_input(self, image_path: str, ocr_text: str) -> str:
        """Process a single image-text pair using LLM OCR improvement."""
        # TODO use the already opened image instead of reopening from a path
        image = Image.open(image_path)

        if self.resize:
            image = resize_image_to_fit(image, self.resize, self.resize)

        instruction = textwrap.dedent(self.instruction_template.format(ocr_text=ocr_text))

        kwargs_for_llm = {
            "model": self.engine,
            "system_prompt_text": self.system_prompt,
            "user_message_text": instruction,
            "user_message_image": image,
            "image_detail": self.image_fidelity,
            "timeout": 60 * 5,
        }

        llm_output = await run_llm_async(**kwargs_for_llm)

        if "<improved_text>" not in llm_output:
            logger.warning(
                f"LLM output does not contain <improved_text> tag. Output: {llm_output}."
            )
            improved_output = ""
        else:
            improved_output = cast(
                str, extract_tag_from_llm_output(llm_output, tags="improved_text")
            )

        if not improved_output:
            logger.warning(f"LLM output is empty. Retrying with {self.backup_engine}.")
            kwargs_for_llm["model"] = self.backup_engine
            llm_output = await run_llm_async(**kwargs_for_llm)
            improved_output = cast(
                str, extract_tag_from_llm_output(llm_output, tags="improved_text")
            )

        return improved_output.strip() if improved_output else ocr_text

"""LLM-based OCR improvement implementation."""

from functools import partial
import textwrap
from typing import Literal, cast

from churro.utils.concurrency import run_async_in_parallel
from churro.utils.image.io import load_image_async
from churro.utils.image.transform import resize_image_to_fit
from churro.utils.llm import LLMInferenceError, extract_tag_from_llm_output, run_llm_async
from churro.utils.log_utils import logger


class LLMImprover:
    """LLM-based OCR improvement using vision models."""

    def __init__(
        self,
        engine: str,
        resize: int | None = None,
        image_fidelity: Literal["high", "low"] = "high",
        backup_engine: str | None = None,
        **kwargs: object,
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
            "You are a meticulous historical document editor. Improve OCR text by fixing "
            "recognition errors and formatting while preserving every original word and "
            "punctuation mark. Return the final document inside <improved_text> and "
            "</improved_text> tags."
        )

        self.instruction_template = """You receive a document image and its OCR text.

        Follow these steps:

        1. Compare the image and OCR text, correcting transcription and formatting mistakes.
        2. Merge split words, handle punctuation precisely, and preserve any non-English text.
        3. Apply Markdown only when it is clearly supported by the image (headings, bold, italics).
        4. Clean Markdown layout without changing words or punctuation; adjust only structure, whitespace, and line breaks. Rejoin hyphenated line breaks such as 'co-\\noperate' -> 'cooperate'.
        5. Return the complete improved document exactly once between the tags, with no commentary, metadata, or code fences.

        Original OCR text:
        {ocr_text}

        Output format:

        <improved_text>
        [Improved document here]
        </improved_text>"""

    async def process_batch_inputs(
        self, image_paths: list[str], texts: list[str], max_concurrency: int
    ) -> list[str]:
        """Process a batch of image-text pairs using LLM OCR improvement."""
        raw_results = await run_async_in_parallel(
            partial(self._process_single_input),
            image_paths,
            texts,
            max_concurrency=max_concurrency,
            desc="LLM OCR Improvement",
        )
        improved_texts: list[str] = []
        for image_path, original_text, result in zip(image_paths, texts, raw_results, strict=False):
            if result is None:
                logger.warning(f"LLM improvement failed for {image_path}; returning original text.")
                improved_texts.append(original_text)
            else:
                improved_texts.append(result)
        return improved_texts

    async def _process_single_input(self, image_path: str, ocr_text: str) -> str:
        """Process a single image-text pair using LLM OCR improvement."""
        image = await load_image_async(image_path)

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

        try:
            llm_output = await run_llm_async(**kwargs_for_llm)
        except LLMInferenceError as exc:
            logger.warning(
                f"LLM improvement primary engine '{self.engine}' failed for {image_path}: {exc}"
            )
            if self.backup_engine and self.backup_engine != self.engine:
                kwargs_for_llm["model"] = self.backup_engine
                try:
                    llm_output = await run_llm_async(**kwargs_for_llm)
                except LLMInferenceError as backup_exc:
                    logger.error(
                        f"LLM improvement backup engine '{self.backup_engine}' failed for {image_path}: {backup_exc}"
                    )
                    raise
            else:
                raise

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
            if self.backup_engine and self.backup_engine != self.engine:
                logger.warning(f"LLM output is empty. Retrying with {self.backup_engine}.")
                kwargs_for_llm["model"] = self.backup_engine
                try:
                    llm_output = await run_llm_async(**kwargs_for_llm)
                except LLMInferenceError as backup_exc:
                    logger.error(
                        f"LLM improvement backup engine '{self.backup_engine}' failed for {image_path}: {backup_exc}"
                    )
                    raise
                improved_output = cast(
                    str, extract_tag_from_llm_output(llm_output, tags="improved_text")
                )

        return improved_output.strip() if improved_output else ocr_text

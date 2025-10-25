"""LLM-based OCR implementation."""

from typing import Literal, override

from PIL import Image

from churro.utils.image.transform import resize_image_to_fit
from churro.utils.llm import LLMInferenceError, extract_tag_from_llm_output, run_llm_async
from churro.utils.log_utils import logger

from .base_ocr import BaseOCR


class ZeroShotLLMOCR(BaseOCR):
    """LLM-based OCR using vision models."""

    OUTPUT_TAG = "answer"

    def __init__(
        self,
        engine: str,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        resize: int | None = None,
        output_markdown: bool = False,
        backup_engine: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.engine = engine
        self.resize = resize
        self.reasoning_effort = reasoning_effort
        self.backup_engine = backup_engine

        if output_markdown:
            self.system_prompt = (
                "You are an expert in transcription of historical documents "
                "from various languages. Your task is to extract the full text from a "
                f"given page in Markdown format. Only output the markdown text between <{self.OUTPUT_TAG}> and </{self.OUTPUT_TAG}> tags."
            )
        else:
            self.system_prompt = (
                "You are an expert in diplomatic transcription of historical documents "
                "from various languages. Your task is to extract the full text from a "
                f"given page. Only output the transcribed text between <{self.OUTPUT_TAG}> and </{self.OUTPUT_TAG}> tags."
            )

        self.instruction = f"""Follow these instructions:

        1. You will be provided with a scanned document page.

        2. Perform transcription on the entirety of the page, converting all visible text into the following format. Include handwritten and print text, if any. Include tables, captions, headers, main text and all other visible text.

        3. If you encounter any non-text elements, simply skip them without attempting to describe them.

        4. Do not modernize or standardize the text. For example, if the transcription is using "ſ" instead of "s" or "а" instead of "a", keep it that way.

        5. When you come across text in languages other than English, transcribe it as accurately as possible without translation.

        6. Output the OCR result in the following format:

        <{self.OUTPUT_TAG}> 
        extracted text here
        </{self.OUTPUT_TAG}>

        Remember, your goal is to accurately transcribe the text from the scanned page as much as possible. Process the entire page, even if it contains a large amount of text, and provide clear, well-formatted output. Pay attention to the appropriate reading order and layout of the text."""

    @override
    async def process_image(self, image: Image.Image) -> str:
        """Process a single image using LLM OCR."""
        if self.resize:
            image = resize_image_to_fit(image, self.resize, self.resize)

        kwargs_for_llm = {
            "model": self.engine,
            "system_prompt_text": self.system_prompt,
            "user_message_text": self.instruction,
            "user_message_image": image,
            "image_detail": "high",
            "timeout": 60 * 10,  # 10 minutes
        }
        if self.reasoning_effort:
            kwargs_for_llm["reasoning_effort"] = self.reasoning_effort

        engines_to_try: list[str] = [self.engine]
        if self.backup_engine and self.backup_engine != self.engine:
            engines_to_try.append(self.backup_engine)

        last_error: Exception | None = None
        extracted_output: str | None = None
        for current_engine in engines_to_try:
            kwargs_for_llm["model"] = current_engine
            try:
                llm_output = await run_llm_async(**kwargs_for_llm)
            except LLMInferenceError as exc:
                last_error = exc
                logger.warning(
                    f"LLM engine '{current_engine}' failed for ZeroShotLLMOCR: {exc}. "
                    "Trying next candidate if available."
                )
                continue

            if f"<{self.OUTPUT_TAG}>" not in llm_output:
                logger.warning(
                    f"LLM output does not contain <{self.OUTPUT_TAG}> tag. Output: {llm_output}"
                )
                if current_engine in {"nanonets-ocr-s", "rolmocr"}:
                    extracted_output = llm_output
                else:
                    extracted_output = ""
            else:
                extracted_output = extract_tag_from_llm_output(llm_output, tags=self.OUTPUT_TAG)

            if extracted_output:
                break

            if current_engine == engines_to_try[-1]:
                break

            logger.warning(
                f"LLM output from '{current_engine}' is empty. Retrying with backup engine."
            )
        else:
            if last_error:
                raise LLMInferenceError(
                    f"All OCR engines failed for {self.get_system_name()}"
                ) from last_error
            raise LLMInferenceError(f"No OCR output received for {self.get_system_name()}.")

        if not extracted_output:
            if last_error:
                raise LLMInferenceError(
                    f"Failed to obtain OCR output after exhausting engines: {last_error}"
                ) from last_error
            raise LLMInferenceError(
                f"Failed to obtain OCR output for {self.get_system_name()} (empty response)."
            )
        assert isinstance(extracted_output, str)

        while f"<{self.OUTPUT_TAG}>" in extracted_output:
            extracted_output = extracted_output[
                extracted_output.index(f"<{self.OUTPUT_TAG}>") + len(f"<{self.OUTPUT_TAG}>") :
            ]
        if f"</{self.OUTPUT_TAG}>" in extracted_output:
            extracted_output = extracted_output[: extracted_output.index(f"</{self.OUTPUT_TAG}>")]

        return extracted_output

    @override
    def get_system_name(self) -> str:
        return "Zero-Shot LLM"

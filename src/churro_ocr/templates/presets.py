"""Built-in OCR template presets."""

from __future__ import annotations

from churro_ocr.prompts import (
    CHANDRA_OCR_LAYOUT_PROMPT,
    DEFAULT_OCR_SYSTEM_PROMPT,
    DEFAULT_OCR_USER_PROMPT,
    OLMOCR_V4_YAML_PROMPT,
)
from churro_ocr.templates.hf import HFChatTemplate

CHURRO_3B_MODEL_ID = "stanford-oval/churro-3B"
CHANDRA_OCR_2_MODEL_ID = "datalab-to/chandra-ocr-2"
DEEPSEEK_OCR_2_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
GLM_OCR_MODEL_ID = "zai-org/GLM-OCR"
FIRERED_OCR_MODEL_ID = "FireRedTeam/FireRed-OCR"
NANONETS_OCR2_3B_MODEL_ID = "nanonets/Nanonets-OCR2-3B"
DOTS_OCR_1_5_MODEL_ID = "kristaller486/dots.ocr-1.5"
DOTS_MOCR_MODEL_ID = "rednote-hilab/dots.mocr"
INFINITY_PARSER_7B_MODEL_ID = "infly/Infinity-Parser-7B"
MINERU2_5_2509_1_2B_MODEL_ID = "opendatalab/MinerU2.5-2509-1.2B"
PADDLEOCR_VL_1_5_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
OLMOCR_2_7B_1025_MODEL_ID = "allenai/olmOCR-2-7B-1025"
OLMOCR_2_7B_1025_FP8_MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
LFM2_5_VL_1_6B_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"
DEFAULT_OCR_TEMPLATE = HFChatTemplate(
    system_message=DEFAULT_OCR_SYSTEM_PROMPT,
    user_prompt=DEFAULT_OCR_USER_PROMPT,
)

CHURRO_3B_XML_TEMPLATE = HFChatTemplate(
    system_message="Transcribe the entirety of this historical document to XML format.",
    user_prompt=None,
)
CHANDRA_OCR_2_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=CHANDRA_OCR_LAYOUT_PROMPT,
)
DEEPSEEK_OCR_2_OCR_PROMPT = "Free OCR."
DEEPSEEK_OCR_2_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=DEEPSEEK_OCR_2_OCR_PROMPT,
)
GLM_OCR_OCR_PROMPT = "Text Recognition:"
GLM_OCR_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=GLM_OCR_OCR_PROMPT,
)
FIRERED_OCR_OCR_PROMPT = (
    "You are an AI assistant specialized in converting PDF images to Markdown format. "
    "Please follow these instructions for the conversion:\n\n"
    "1. Text Processing:\n"
    "- Accurately recognize all text content in the PDF image without guessing or inferring.\n"
    "- Convert the recognized text into Markdown format.\n"
    "- Maintain the original document structure, including headings, paragraphs, lists, etc.\n\n"
    "2. Mathematical Formula Processing:\n"
    "- Convert all mathematical formulas to LaTeX format.\n"
    "- Enclose inline formulas with,(,). For example: This is an inline formula,( E = mc^2,)\n"
    "- Enclose block formulas with,\\[,\\]. For example:,[,frac{-b,pm,sqrt{b^2 - 4ac}}{2a},]\n\n"
    "3. Table Processing:\n"
    "- Convert tables to HTML format.\n"
    "- Wrap the entire table with <table> and </table>.\n\n"
    "4. Figure Handling:\n"
    "- Ignore figures content in the PDF image. Do not attempt to describe or convert images.\n\n"
    "5. Output Format:\n"
    "- Ensure the output Markdown document has a clear structure with appropriate line breaks "
    "between elements.\n"
    "- For complex layouts, try to maintain the original document's structure and format as "
    "closely as possible.\n\n"
    "Please strictly follow these guidelines to ensure accuracy and consistency in the "
    "conversion. Your task is to accurately convert the content of the PDF image into Markdown "
    "format without adding any extra explanations or comments."
)
FIRERED_OCR_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=FIRERED_OCR_OCR_PROMPT,
)
NANONETS_OCR2_3B_SYSTEM_PROMPT = "You are a helpful assistant."
NANONETS_OCR2_3B_OCR_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. Ex: "
    "<watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets.\n"
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using "
    "☐ and ☑ for check boxes."
)
NANONETS_OCR2_3B_OCR_TEMPLATE = HFChatTemplate(
    system_message=NANONETS_OCR2_3B_SYSTEM_PROMPT,
    user_prompt=NANONETS_OCR2_3B_OCR_PROMPT,
)
DOTS_OCR_1_5_OCR_PROMPT = "Extract the text content from this image."
DOTS_OCR_1_5_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=DOTS_OCR_1_5_OCR_PROMPT,
)
DOTS_MOCR_OCR_PROMPT = DOTS_OCR_1_5_OCR_PROMPT
DOTS_MOCR_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=DOTS_MOCR_OCR_PROMPT,
)
INFINITY_PARSER_7B_SYSTEM_PROMPT = "You are a helpful assistant."
INFINITY_PARSER_7B_OCR_PROMPT = (
    "Convert this document page to Markdown.\n"
    "- Transcribe all visible text accurately without guessing.\n"
    "- Preserve the reading order and the document structure, including headings, paragraphs, and lists.\n"
    "- Convert mathematical expressions to LaTeX, using \\(...\\) for inline math and "
    "\\[...\\] for display math.\n"
    "- Convert tables to HTML wrapped in <table>...</table>.\n"
    "- Ignore figures and other purely graphical content instead of describing them.\n"
    "- Return only the converted Markdown with no extra commentary."
)
INFINITY_PARSER_7B_OCR_TEMPLATE = HFChatTemplate(
    system_message=INFINITY_PARSER_7B_SYSTEM_PROMPT,
    user_prompt=INFINITY_PARSER_7B_OCR_PROMPT,
)
MINERU2_5_2509_1_2B_SYSTEM_PROMPT = "You are a helpful assistant."
MINERU2_5_2509_1_2B_LAYOUT_PROMPT = "\nLayout Detection:"
MINERU2_5_2509_1_2B_TABLE_PROMPT = "\nTable Recognition:"
MINERU2_5_2509_1_2B_FORMULA_PROMPT = "\nFormula Recognition:"
MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT = "\nImage Analysis:"
MINERU2_5_2509_1_2B_OCR_PROMPT = "\nText Recognition:"
MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE = HFChatTemplate(
    system_message=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    user_prompt=MINERU2_5_2509_1_2B_LAYOUT_PROMPT,
)
MINERU2_5_2509_1_2B_TABLE_TEMPLATE = HFChatTemplate(
    system_message=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    user_prompt=MINERU2_5_2509_1_2B_TABLE_PROMPT,
)
MINERU2_5_2509_1_2B_FORMULA_TEMPLATE = HFChatTemplate(
    system_message=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    user_prompt=MINERU2_5_2509_1_2B_FORMULA_PROMPT,
)
MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE = HFChatTemplate(
    system_message=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    user_prompt=MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
)
MINERU2_5_2509_1_2B_OCR_TEMPLATE = HFChatTemplate(
    system_message=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    user_prompt=MINERU2_5_2509_1_2B_OCR_PROMPT,
)
PADDLEOCR_VL_1_5_OCR_PROMPT = "OCR:"
PADDLEOCR_VL_1_5_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=PADDLEOCR_VL_1_5_OCR_PROMPT,
)
OLMOCR_2_7B_1025_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt=OLMOCR_V4_YAML_PROMPT,
    user_prompt_first=True,
)
LFM2_5_VL_1_6B_OCR_TEMPLATE = HFChatTemplate(
    system_message=None,
    user_prompt="Transcribe all visible text from this historical document page in reading order.",
)


__all__ = [
    "CHANDRA_OCR_2_MODEL_ID",
    "CHANDRA_OCR_2_OCR_TEMPLATE",
    "CHURRO_3B_MODEL_ID",
    "CHURRO_3B_XML_TEMPLATE",
    "DEEPSEEK_OCR_2_MODEL_ID",
    "DEEPSEEK_OCR_2_OCR_PROMPT",
    "DEEPSEEK_OCR_2_OCR_TEMPLATE",
    "DEFAULT_OCR_TEMPLATE",
    "DOTS_MOCR_MODEL_ID",
    "DOTS_MOCR_OCR_PROMPT",
    "DOTS_MOCR_OCR_TEMPLATE",
    "DOTS_OCR_1_5_MODEL_ID",
    "DOTS_OCR_1_5_OCR_PROMPT",
    "DOTS_OCR_1_5_OCR_TEMPLATE",
    "FIRERED_OCR_MODEL_ID",
    "FIRERED_OCR_OCR_PROMPT",
    "FIRERED_OCR_OCR_TEMPLATE",
    "GLM_OCR_MODEL_ID",
    "GLM_OCR_OCR_PROMPT",
    "GLM_OCR_OCR_TEMPLATE",
    "INFINITY_PARSER_7B_MODEL_ID",
    "INFINITY_PARSER_7B_OCR_PROMPT",
    "INFINITY_PARSER_7B_OCR_TEMPLATE",
    "INFINITY_PARSER_7B_SYSTEM_PROMPT",
    "LFM2_5_VL_1_6B_MODEL_ID",
    "LFM2_5_VL_1_6B_OCR_TEMPLATE",
    "MINERU2_5_2509_1_2B_FORMULA_PROMPT",
    "MINERU2_5_2509_1_2B_FORMULA_TEMPLATE",
    "MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT",
    "MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE",
    "MINERU2_5_2509_1_2B_LAYOUT_PROMPT",
    "MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE",
    "MINERU2_5_2509_1_2B_MODEL_ID",
    "MINERU2_5_2509_1_2B_OCR_PROMPT",
    "MINERU2_5_2509_1_2B_OCR_TEMPLATE",
    "MINERU2_5_2509_1_2B_SYSTEM_PROMPT",
    "MINERU2_5_2509_1_2B_TABLE_PROMPT",
    "MINERU2_5_2509_1_2B_TABLE_TEMPLATE",
    "NANONETS_OCR2_3B_MODEL_ID",
    "NANONETS_OCR2_3B_OCR_PROMPT",
    "NANONETS_OCR2_3B_OCR_TEMPLATE",
    "NANONETS_OCR2_3B_SYSTEM_PROMPT",
    "OLMOCR_2_7B_1025_FP8_MODEL_ID",
    "OLMOCR_2_7B_1025_MODEL_ID",
    "OLMOCR_2_7B_1025_OCR_TEMPLATE",
    "PADDLEOCR_VL_1_5_MODEL_ID",
    "PADDLEOCR_VL_1_5_OCR_PROMPT",
    "PADDLEOCR_VL_1_5_OCR_TEMPLATE",
]

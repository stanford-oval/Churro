from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

import churro_ocr.providers.hf as hf_module
from churro_ocr.errors import ConfigurationError
from churro_ocr.ocr import OCRClient
from churro_ocr.page_detection import DocumentPage
from churro_ocr.prompts import (
    CHANDRA_OCR_LAYOUT_PROMPT,
    DEFAULT_OCR_OUTPUT_TAG,
    OLMOCR_V4_YAML_PROMPT,
    parse_chandra_response,
    parse_olmocr_response,
)
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend
from churro_ocr.providers._mineru25 import (
    MinerU25PipelineHelper,
    convert_mineru2_5_otsl_to_html,
    wrap_mineru2_5_equation,
)
from churro_ocr.providers.hf import (
    ChandraOCR2OCRBackend,
    Churro3BOCRBackend,
    DeepSeekOCR2OCRBackend,
    DotsMOCROCRBackend,
    DotsOCR15OCRBackend,
    GlmOCROCRBackend,
    HuggingFaceVisionOCRBackend,
    LFM25VLOCRBackend,
    MinerU25OCRBackend,
    PaddleOCRVL15OCRBackend,
)
from churro_ocr.providers.specs import (
    DEFAULT_OCR_MAX_TOKENS,
    deepseek_ocr_2_text_postprocessor,
    glm_ocr_text_postprocessor,
    infinity_parser_7b_text_postprocessor,
    lfm2_5_vl_text_postprocessor,
)
from churro_ocr.templates import (
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    CHURRO_3B_XML_TEMPLATE,
    DEEPSEEK_OCR_2_MODEL_ID,
    DEEPSEEK_OCR_2_OCR_PROMPT,
    DEEPSEEK_OCR_2_OCR_TEMPLATE,
    DOTS_MOCR_MODEL_ID,
    DOTS_MOCR_OCR_PROMPT,
    DOTS_MOCR_OCR_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_PROMPT,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    GLM_OCR_MODEL_ID,
    GLM_OCR_OCR_PROMPT,
    GLM_OCR_OCR_TEMPLATE,
    INFINITY_PARSER_7B_MODEL_ID,
    INFINITY_PARSER_7B_OCR_PROMPT,
    INFINITY_PARSER_7B_OCR_TEMPLATE,
    INFINITY_PARSER_7B_SYSTEM_PROMPT,
    LFM2_5_VL_1_6B_MODEL_ID,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_FORMULA_PROMPT,
    MINERU2_5_2509_1_2B_FORMULA_TEMPLATE,
    MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
    MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE,
    MINERU2_5_2509_1_2B_LAYOUT_PROMPT,
    MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE,
    MINERU2_5_2509_1_2B_MODEL_ID,
    MINERU2_5_2509_1_2B_OCR_PROMPT,
    MINERU2_5_2509_1_2B_OCR_TEMPLATE,
    MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    MINERU2_5_2509_1_2B_TABLE_PROMPT,
    MINERU2_5_2509_1_2B_TABLE_TEMPLATE,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
    PADDLEOCR_VL_1_5_MODEL_ID,
    PADDLEOCR_VL_1_5_OCR_PROMPT,
    PADDLEOCR_VL_1_5_OCR_TEMPLATE,
    HFChatTemplate,
    OCRConversation,
)


def test_hf_chat_template_builds_expected_conversation() -> None:
    template = HFChatTemplate(
        system_message="system text",
        user_prompt="user text",
    )
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    conversation = template.build_conversation(page)

    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"][0]["text"] == "system text"
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"][0]["type"] == "image"
    assert conversation[1]["content"][1]["text"] == "user text"


def test_olmocr_template_builds_prompt_before_image() -> None:
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    conversation = OLMOCR_2_7B_1025_OCR_TEMPLATE.build_conversation(page)

    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"][0]["text"] == OLMOCR_V4_YAML_PROMPT
    assert conversation[0]["content"][1]["type"] == "image"


def test_chandra_template_builds_image_before_prompt() -> None:
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    conversation = CHANDRA_OCR_2_OCR_TEMPLATE.build_conversation(page)

    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"][0]["type"] == "image"
    assert conversation[0]["content"][1]["text"] == CHANDRA_OCR_LAYOUT_PROMPT


def test_deepseek_ocr_2_template_builds_image_before_prompt() -> None:
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    conversation = DEEPSEEK_OCR_2_OCR_TEMPLATE.build_conversation(page)

    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"][0]["type"] == "image"
    assert conversation[0]["content"][1]["text"] == DEEPSEEK_OCR_2_OCR_PROMPT


def test_infinity_parser_template_matches_documented_prompt_shape() -> None:
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    conversation = INFINITY_PARSER_7B_OCR_TEMPLATE.build_conversation(page)

    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"][0]["text"] == INFINITY_PARSER_7B_SYSTEM_PROMPT
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"][0]["type"] == "image"
    assert conversation[1]["content"][1]["text"] == INFINITY_PARSER_7B_OCR_PROMPT


def test_mineru2_5_template_matches_upstream_prompt_shape() -> None:
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    conversation = MINERU2_5_2509_1_2B_OCR_TEMPLATE.build_conversation(page)

    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"][0]["text"] == MINERU2_5_2509_1_2B_SYSTEM_PROMPT
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"][0]["type"] == "image"
    assert conversation[1]["content"][1]["text"] == MINERU2_5_2509_1_2B_OCR_PROMPT


def test_mineru2_5_end_to_end_templates_cover_layout_table_formula_and_image_prompts() -> None:
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    layout_conversation = MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE.build_conversation(page)
    table_conversation = MINERU2_5_2509_1_2B_TABLE_TEMPLATE.build_conversation(page)
    formula_conversation = MINERU2_5_2509_1_2B_FORMULA_TEMPLATE.build_conversation(page)
    image_conversation = MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE.build_conversation(page)

    assert layout_conversation[0]["content"][0]["text"] == MINERU2_5_2509_1_2B_SYSTEM_PROMPT
    assert layout_conversation[1]["content"][1]["text"] == MINERU2_5_2509_1_2B_LAYOUT_PROMPT
    assert table_conversation[1]["content"][1]["text"] == MINERU2_5_2509_1_2B_TABLE_PROMPT
    assert formula_conversation[1]["content"][1]["text"] == MINERU2_5_2509_1_2B_FORMULA_PROMPT
    assert image_conversation[1]["content"][1]["text"] == MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT


def test_parse_and_render_mineru2_5_end_to_end_blocks() -> None:
    helper = MinerU25PipelineHelper(
        prompts={
            "[default]": MINERU2_5_2509_1_2B_OCR_PROMPT,
            "[layout]": MINERU2_5_2509_1_2B_LAYOUT_PROMPT,
            "table": MINERU2_5_2509_1_2B_TABLE_PROMPT,
            "equation": MINERU2_5_2509_1_2B_FORMULA_PROMPT,
            "image": MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
            "chart": MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_PROMPT,
        },
        system_prompt=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    )
    blocks = helper.parse_layout_output(
        "<|box_start|>0 0 1000 100<|box_end|><|ref_start|>header<|ref_end|>\n"
        "<|box_start|>0 100 1000 400<|box_end|><|ref_start|>text<|ref_end|>\n"
        "<|box_start|>0 400 1000 500<|box_end|><|ref_start|>text<|ref_end|>txt_contd_tgt\n"
        "<|box_start|>0 500 1000 800<|box_end|><|ref_start|>table<|ref_end|>\n"
        "<|box_start|>0 800 1000 1000<|box_end|><|ref_start|>equation<|ref_end|><|rotate_right|>\n"
        "<|box_start|>1001 0 1100 10<|box_end|><|ref_start|>bad<|ref_end|>"
    )
    blocks[0].content = "Page header"
    blocks[1].content = "Body"
    blocks[2].content = "text"
    blocks[3].content = "<fcel>Year<fcel>Value<nl><fcel>1900<fcel>42<nl>"
    blocks[4].content = "x = y"

    assert [block.type for block in blocks] == ["header", "text", "text", "table", "equation"]
    assert blocks[2].merge_prev is True
    assert blocks[4].angle == 90
    assert convert_mineru2_5_otsl_to_html(blocks[3].content or "") == (
        "<table><tr><td>Year</td><td>Value</td></tr><tr><td>1900</td><td>42</td></tr></table>"
    )
    assert wrap_mineru2_5_equation("x = y") == "\\[\nx = y\n\\]"
    processed = helper.post_process(blocks)
    assert helper.render_markdown(processed) == (
        "Page header\n\n"
        "Body text\n\n"
        "<table><tr><td>Year</td><td>Value</td></tr><tr><td>1900</td><td>42</td></tr></table>\n\n"
        "\\[\nx = y\n\\]"
    )


def test_mineru2_5_clean_response_strips_prompt_echo_and_role_scaffold() -> None:
    helper = MinerU25PipelineHelper(
        prompts={"[default]": MINERU2_5_2509_1_2B_OCR_PROMPT},
        system_prompt=MINERU2_5_2509_1_2B_SYSTEM_PROMPT,
    )

    assert (
        helper.clean_response(
            (
                f"{MINERU2_5_2509_1_2B_SYSTEM_PROMPT}\n"
                f"{MINERU2_5_2509_1_2B_OCR_PROMPT}\n"
                "assistant:\n"
                "plain text<|im_end|>"
            ),
            step_key="[default]",
        )
        == "plain text"
    )


def test_parse_olmocr_response_extracts_plain_text_and_metadata() -> None:
    text, metadata = parse_olmocr_response(
        "---\n"
        "primary_language: en\n"
        "is_rotation_valid: true\n"
        "rotation_correction: 0\n"
        "is_table: true\n"
        "is_diagram: false\n"
        "---\n"
        "# Heading\n\n"
        "<table><tr><th>Year</th><th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table>\n\n"
        "![Figure alt text](page_0_0_100_100.png)\n"
        "Paragraph with [reference](https://example.test)."
    )

    assert text == "Heading\n\nYear | Value\n1900 | 42\n\nParagraph with reference."
    assert metadata["front_matter"] == {
        "primary_language": "en",
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": True,
        "is_diagram": False,
    }
    assert "Heading" in cast("str", metadata["raw_markdown"])


def test_parse_chandra_response_extracts_plain_text_and_metadata() -> None:
    text, metadata = parse_chandra_response(
        '<div data-bbox="0 0 1000 100" data-label="Section-Header"><h1>Title</h1></div>\n'
        '<div data-bbox="0 100 1000 300" data-label="Text"><p>Paragraph with <a '
        'href="https://example.test">reference</a>.</p></div>\n'
        '<div data-bbox="0 300 1000 500" data-label="Form"><p><input type="checkbox" checked> '
        "Checked item</p></div>\n"
        '<div data-bbox="0 500 1000 700" data-label="Table"><table><tr><th>Year</th>'
        "<th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table></div>"
    )

    assert text == "Title\n\nParagraph with reference.\n\n[x] Checked item\n\nYear | Value\n1900 | 42"
    assert metadata == {
        "raw_html": (
            '<div data-bbox="0 0 1000 100" data-label="Section-Header"><h1>Title</h1></div>\n'
            '<div data-bbox="0 100 1000 300" data-label="Text"><p>Paragraph with <a '
            'href="https://example.test">reference</a>.</p></div>\n'
            '<div data-bbox="0 300 1000 500" data-label="Form"><p><input type="checkbox" checked> '
            "Checked item</p></div>\n"
            '<div data-bbox="0 500 1000 700" data-label="Table"><table><tr><th>Year</th>'
            "<th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table></div>"
        ),
    }


def test_lfm25_text_postprocessor_strips_prompt_and_role_scaffold() -> None:
    text = (
        "Transcribe all visible text from this historical document page in reading order.\n"
        "assistant\n"
        f"<{DEFAULT_OCR_OUTPUT_TAG}>\n"
        "decoded text\n"
        f"</{DEFAULT_OCR_OUTPUT_TAG}>"
    )

    assert lfm2_5_vl_text_postprocessor(text) == "decoded text"


def test_lfm25_text_postprocessor_strips_role_only_prefix() -> None:
    assert lfm2_5_vl_text_postprocessor("assistant:\nplain text") == "plain text"


def test_infinity_parser_text_postprocessor_strips_prompt_echo_and_preserves_raw_markdown() -> None:
    processed = infinity_parser_7b_text_postprocessor(
        f"{INFINITY_PARSER_7B_OCR_PROMPT}\n"
        "assistant:\n"
        "# Heading\n\n"
        "<table><tr><th>Year</th><th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table>\n\n"
        "Paragraph with [note](https://example.test).\n"
    )
    assert isinstance(processed, tuple)
    text, metadata = processed

    assert text == "Heading\n\nYear | Value\n1900 | 42\n\nParagraph with note."
    assert metadata == {
        "raw_markdown": (
            "# Heading\n\n"
            "<table><tr><th>Year</th><th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table>\n\n"
            "Paragraph with [note](https://example.test)."
        ),
    }


def test_infinity_parser_text_postprocessor_strips_outer_markdown_fence() -> None:
    processed = infinity_parser_7b_text_postprocessor(
        f"{INFINITY_PARSER_7B_OCR_PROMPT}\n"
        "assistant:\n"
        "```markdown\n"
        "169\n\n"
        "které wětšj gsau nynj žigjejch;\n"
        "```"
    )
    assert isinstance(processed, tuple)
    text, metadata = processed

    assert text == "169\n\nkteré wětšj gsau nynj žigjejch;"
    assert metadata == {
        "raw_markdown": "169\n\nkteré wětšj gsau nynj žigjejch;",
    }


def test_deepseek_ocr_2_text_postprocessor_strips_prompt_echo_and_stop_token() -> None:
    assert (
        deepseek_ocr_2_text_postprocessor(
            "<image>\nFree OCR.\n<｜Assistant｜>\nplain text<｜end▁of▁sentence｜>"
        )
        == "plain text"
    )


def test_glm_ocr_text_postprocessor_strips_prompt_echo_and_trailing_tokens() -> None:
    assert (
        glm_ocr_text_postprocessor("Text Recognition:\n<|assistant|>\nplain text\n<|user|>\n<|endoftext|>")
        == "plain text"
    )


def test_build_ocr_backend_uses_chandra_profile_defaults_for_hf() -> None:
    backend = cast(
        "ChandraOCR2OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=CHANDRA_OCR_2_MODEL_ID,
            )
        ),
    )

    assert isinstance(backend, ChandraOCR2OCRBackend)
    assert backend.template == CHANDRA_OCR_2_OCR_TEMPLATE
    assert backend.model_name == "chandra-ocr-2"
    assert backend.generation_kwargs == {
        "max_new_tokens": 12_384,
    }
    assert backend.image_preprocessor(Image.new("RGB", (5_000, 3_000), color="white")).size == (3_248, 1_932)


def test_build_ocr_backend_uses_deepseek_ocr_2_profile_defaults_for_hf() -> None:
    backend = cast(
        "DeepSeekOCR2OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=DEEPSEEK_OCR_2_MODEL_ID,
            )
        ),
    )

    assert isinstance(backend, DeepSeekOCR2OCRBackend)
    assert backend.template == DEEPSEEK_OCR_2_OCR_TEMPLATE
    assert backend.model_name == "DeepSeek-OCR-2"
    assert backend.generation_kwargs == {"max_new_tokens": 8_192}
    assert backend.trust_remote_code is True
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs == {"use_safetensors": True}
    assert backend.base_size == 1_024
    assert backend.image_size == 768
    assert backend.crop_mode is True


def test_build_ocr_backend_uses_glm_ocr_profile_defaults_for_hf() -> None:
    backend = cast(
        "GlmOCROCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=GLM_OCR_MODEL_ID,
            )
        ),
    )

    assert isinstance(backend, GlmOCROCRBackend)
    assert backend.template == GLM_OCR_OCR_TEMPLATE
    assert backend.model_name == "GLM-OCR"
    assert backend.generation_kwargs == {
        "max_new_tokens": 8_192,
        "do_sample": False,
    }
    assert backend.trust_remote_code is False
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs == {}
    preprocessed_image = backend.image_preprocessor(
        Image.new("RGBA", (3_508, 2_720), color=(255, 255, 255, 255))
    )
    assert preprocessed_image.size == (2_464, 1_904)
    assert preprocessed_image.mode == "RGB"
    assert (preprocessed_image.size[0] // 28) * (preprocessed_image.size[1] // 28) <= 6_084


def test_build_ocr_backend_uses_olmocr_profile_defaults_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=OLMOCR_2_7B_1025_MODEL_ID,
            )
        ),
    )

    assert backend.template == OLMOCR_2_7B_1025_OCR_TEMPLATE
    assert backend.model_name == "olmOCR-2-7B-1025"
    assert backend.generation_kwargs == {
        "max_new_tokens": 8_000,
        "temperature": 0.1,
        "do_sample": True,
    }
    assert backend.image_preprocessor(Image.new("RGB", (5_000, 3_000), color="white")).size == (1_288, 772)


def test_build_ocr_backend_uses_infinity_parser_profile_defaults_for_hf() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=INFINITY_PARSER_7B_MODEL_ID,
            )
        ),
    )

    assert type(backend) is HuggingFaceVisionOCRBackend
    assert backend.template == INFINITY_PARSER_7B_OCR_TEMPLATE
    assert backend.model_name == "Infinity-Parser-7B"
    assert backend.generation_kwargs == {
        "max_new_tokens": 4_096,
    }
    assert backend.processor_kwargs == {
        "min_pixels": 200_704,
        "max_pixels": 1_806_336,
    }
    assert backend.trust_remote_code is False
    assert backend.model_kwargs == {}
    preprocessed_image = backend.image_preprocessor(Image.new("RGBA", (32, 16), color=(255, 255, 255, 255)))
    assert preprocessed_image.size == (32, 16)
    assert preprocessed_image.mode == "RGB"


def test_build_ocr_backend_uses_lfm25_profile_defaults_for_hf() -> None:
    backend = cast(
        "LFM25VLOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=LFM2_5_VL_1_6B_MODEL_ID,
            )
        ),
    )

    assert isinstance(backend, LFM25VLOCRBackend)
    assert backend.template == LFM2_5_VL_1_6B_OCR_TEMPLATE
    assert backend.model_name == "LFM2.5-VL-1.6B"
    assert backend.generation_kwargs == {
        "max_new_tokens": 512,
        "do_sample": False,
        "repetition_penalty": 1.05,
    }


def test_build_ocr_backend_uses_paddleocr_vl_profile_defaults_for_hf() -> None:
    backend = cast(
        "PaddleOCRVL15OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=PADDLEOCR_VL_1_5_MODEL_ID,
            )
        ),
    )

    assert isinstance(backend, PaddleOCRVL15OCRBackend)
    assert backend.template == PADDLEOCR_VL_1_5_OCR_TEMPLATE
    assert backend.model_name == "PaddleOCR-VL-1.5"
    assert backend.generation_kwargs == {
        "max_new_tokens": 4_096,
        "do_sample": False,
    }


def test_build_ocr_backend_uses_mineru2_5_profile_defaults_for_hf() -> None:
    backend = cast(
        "MinerU25OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=MINERU2_5_2509_1_2B_MODEL_ID,
            )
        ),
    )

    assert isinstance(backend, MinerU25OCRBackend)
    assert backend.template == MINERU2_5_2509_1_2B_OCR_TEMPLATE
    assert backend.model_name == "MinerU2.5-2509-1.2B"
    assert backend.trust_remote_code is False
    assert backend.processor_kwargs == {"use_fast": True}
    assert backend.model_kwargs == {}
    preprocessed_image = backend.image_preprocessor(Image.new("RGBA", (32, 16), color=(255, 255, 255, 255)))
    assert preprocessed_image.size == (32, 16)
    assert preprocessed_image.mode == "RGB"
    assert backend.generation_kwargs == {}


def test_build_ocr_backend_uses_dots_mocr_profile_defaults_for_hf() -> None:
    backend = cast(
        "DotsMOCROCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=DOTS_MOCR_MODEL_ID,
            )
        ),
    )

    assert isinstance(backend, DotsMOCROCRBackend)
    assert backend.template == DOTS_MOCR_OCR_TEMPLATE
    assert backend.model_name == "dots.mocr"
    assert backend.generation_kwargs == {"max_new_tokens": DEFAULT_OCR_MAX_TOKENS}
    assert backend.trust_remote_code is True
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs["dtype"] in {"auto", "float32"}
    if backend.model_kwargs["dtype"] == "auto" and "device_map" in backend.model_kwargs:
        assert backend.model_kwargs["device_map"] == "auto"
        assert "max_memory" in backend.model_kwargs


@pytest.mark.asyncio
async def test_chandra_huggingface_backend_matches_upstream_chat_template_and_eos_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAttentionMask:
        def sum(self, dim: int) -> SimpleNamespace:
            captured["attention_mask_sum_dim"] = dim
            return SimpleNamespace(tolist=lambda: [3])

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

        def convert_tokens_to_ids(self, token: str) -> int:
            captured["stop_token_lookup"] = token
            return 77

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def apply_chat_template(
            self,
            conversation: object,
            *,
            add_generation_prompt: bool,
            tokenize: bool = True,
            return_dict: bool = True,
            return_tensors: str | None = None,
            padding: bool | None = None,
        ) -> object:
            captured["add_generation_prompt"] = add_generation_prompt
            captured["tokenize"] = tokenize
            captured["return_dict"] = return_dict
            if not tokenize:
                captured["render_conversation"] = conversation
                return "<rendered>"
            captured["tokenized_conversations"] = conversation
            captured["return_tensors"] = return_tensors
            captured["padding"] = padding
            return FakeBatch(
                {
                    "input_ids": SimpleNamespace(shape=(1, 3)),
                    "attention_mask": FakeAttentionMask(),
                }
            )

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["skip_special_tokens"] = skip_special_tokens
            captured["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return [
                '<div data-bbox="0 0 1000 1000" data-label="Text"><p>Decoded '
                '<a href="https://example.test">output</a>.</p></div>'
            ]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeModel:
        device = "fake-device"
        dtype = None
        generation_config = SimpleNamespace(eos_token_id=11)

        def eval(self) -> FakeModel:
            captured["eval_called"] = True
            return self

        def generate(self, **kwargs: object) -> list[list[int | str]]:
            captured["generate_kwargs"] = kwargs
            return [[0, 1, 2, "completion"]]

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=None,
        ),
    )

    backend = cast(
        "ChandraOCR2OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=CHANDRA_OCR_2_MODEL_ID,
            )
        ),
    )
    result = await backend.ocr(
        DocumentPage.from_image(Image.new("RGBA", (5_000, 3_000), color=(255, 255, 255, 255)))
    )

    assert result.text == "Decoded output."
    assert result.metadata == {
        "raw_html": (
            '<div data-bbox="0 0 1000 1000" data-label="Text"><p>Decoded '
            '<a href="https://example.test">output</a>.</p></div>'
        ),
    }
    assert captured["processor_model_id"] == CHANDRA_OCR_2_MODEL_ID
    assert captured["model_model_id"] == CHANDRA_OCR_2_MODEL_ID
    assert captured["eval_called"] is True
    assert captured["render_conversation"][0]["role"] == "user"
    render_content = cast("list[dict[str, object]]", captured["render_conversation"][0]["content"])
    assert render_content[0]["type"] == "image"
    render_image = cast("Image.Image", render_content[0]["image"])
    assert render_image.size == (3_248, 1_932)
    assert render_image.mode == "RGB"
    assert render_content[1] == {"type": "text", "text": CHANDRA_OCR_LAYOUT_PROMPT}
    tokenized_conversation = cast("list[OCRConversation]", captured["tokenized_conversations"])
    assert tokenized_conversation[0][0]["role"] == "user"
    assert captured["tokenize"] is True
    assert captured["return_dict"] is True
    assert captured["return_tensors"] == "pt"
    assert captured["padding"] is True
    assert captured["device"] == "fake-device"
    assert captured["attention_mask_sum_dim"] == 1
    assert captured["stop_token_lookup"] == "<|im_end|>"
    assert captured["generate_kwargs"]["max_new_tokens"] == 12_384
    assert captured["generate_kwargs"]["eos_token_id"] == [11, 77]
    assert captured["model_from_pretrained_kwargs"]["device_map"] == "auto"
    assert "dtype" in cast("dict[str, object]", captured["model_from_pretrained_kwargs"])
    assert captured["skip_special_tokens"] is True
    assert captured["clean_up_tokenization_spaces"] is False


@pytest.mark.asyncio
async def test_deepseek_ocr_2_huggingface_backend_uses_upstream_infer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeTokenizer:
        pass

    class FakeTokenizerCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            captured["tokenizer_model_id"] = model_id
            captured["tokenizer_from_pretrained_kwargs"] = kwargs
            return FakeTokenizer()

    class FakeModel:
        def eval(self) -> FakeModel:
            captured["eval_called"] = True
            return self

        def cuda(self) -> FakeModel:
            captured["cuda_called"] = True
            return self

        def to(self, dtype: object) -> FakeModel:
            captured["to_dtype"] = dtype
            return self

        def infer(self, tokenizer: object, **kwargs: object) -> str:
            captured["infer_tokenizer"] = tokenizer
            captured["infer_kwargs"] = kwargs
            image = Image.open(cast("str", kwargs["image_file"]))
            captured["saved_image_mode"] = image.mode
            captured["saved_image_size"] = image.size
            captured["output_dir_exists"] = Path(cast("str", kwargs["output_path"])).exists()
            return "<image>\nFree OCR.\n<｜Assistant｜>\nDecoded text<｜end▁of▁sentence｜>"

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_auto_model_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeTokenizerCls,
            model_cls=FakeModelCls,
            process_vision_info=None,
        ),
    )
    monkeypatch.setattr(
        "churro_ocr.providers.hf._ensure_deepseek_ocr_2_cuda_runtime",
        lambda: SimpleNamespace(bfloat16="fake-bfloat16"),
    )

    backend = cast(
        "DeepSeekOCR2OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=DEEPSEEK_OCR_2_MODEL_ID,
            )
        ),
    )
    result = await backend.ocr(
        DocumentPage.from_image(Image.new("RGBA", (32, 16), color=(255, 255, 255, 255)))
    )

    assert result.text == "Decoded text"
    assert result.metadata == {}
    assert captured["tokenizer_model_id"] == DEEPSEEK_OCR_2_MODEL_ID
    assert captured["model_model_id"] == DEEPSEEK_OCR_2_MODEL_ID
    assert captured["tokenizer_from_pretrained_kwargs"] == {"trust_remote_code": True}
    assert captured["model_from_pretrained_kwargs"] == {
        "trust_remote_code": True,
        "use_safetensors": True,
    }
    assert captured["eval_called"] is True
    assert captured["cuda_called"] is True
    assert captured["to_dtype"] == "fake-bfloat16"
    assert captured["infer_tokenizer"].__class__ is FakeTokenizer
    infer_kwargs = cast("dict[str, object]", captured["infer_kwargs"])
    assert infer_kwargs == {
        "prompt": "<image>\nFree OCR.",
        "image_file": infer_kwargs["image_file"],
        "output_path": infer_kwargs["output_path"],
        "base_size": 1_024,
        "image_size": 768,
        "crop_mode": True,
        "save_results": False,
        "eval_mode": True,
    }
    assert captured["saved_image_mode"] == "RGB"
    assert captured["saved_image_size"] == (32, 16)
    assert captured["output_dir_exists"] is True


@pytest.mark.asyncio
async def test_glm_ocr_huggingface_backend_uses_tokenized_chat_template_and_profile_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAttentionMask:
        def sum(self, dim: int) -> SimpleNamespace:
            captured["attention_mask_sum_dim"] = dim
            return SimpleNamespace(tolist=lambda: [4])

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def apply_chat_template(
            self,
            conversation: object,
            *,
            add_generation_prompt: bool,
            tokenize: bool,
            return_dict: bool | None = None,
            return_tensors: str | None = None,
            padding: bool | None = None,
        ) -> object:
            captured.setdefault("chat_calls", []).append(
                {
                    "conversation": conversation,
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": tokenize,
                    "return_dict": return_dict,
                    "return_tensors": return_tensors,
                    "padding": padding,
                }
            )
            if not tokenize:
                return "<glm-rendered>"
            return FakeBatch(
                {
                    "input_ids": SimpleNamespace(shape=(1, 4)),
                    "attention_mask": FakeAttentionMask(),
                    "token_type_ids": "unused-token-type-ids",
                    "mm_token_type_ids": "kept-mm-token-type-ids",
                }
            )

        def __call__(self, **kwargs: object) -> object:
            del kwargs
            message = "processor(...) should not be used for GLM-OCR"
            raise AssertionError(message)

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["skip_special_tokens"] = skip_special_tokens
            captured["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return ["Text Recognition:\n<|assistant|>\nglm transcription\n<|user|>"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeModel:
        device = "fake-device"
        dtype = None

        def eval(self) -> FakeModel:
            captured["eval_called"] = True
            return self

        def generate(self, **kwargs: object) -> list[list[int | str]]:
            captured["generate_kwargs"] = kwargs
            return [[0, 1, 2, 3, "completion"]]

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=None,
        ),
    )

    backend = cast(
        "GlmOCROCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=GLM_OCR_MODEL_ID,
            )
        ),
    )
    result = await backend.ocr(DocumentPage.from_image(Image.new("RGB", (32, 32), color="white")))

    assert result.text == "glm transcription"
    assert captured["processor_model_id"] == GLM_OCR_MODEL_ID
    assert captured["model_model_id"] == GLM_OCR_MODEL_ID
    assert captured["processor_from_pretrained_kwargs"] == {"trust_remote_code": False}
    assert captured["model_from_pretrained_kwargs"] == {"trust_remote_code": False}
    assert captured["eval_called"] is True
    assert cast("FakeProcessor", backend._processor).tokenizer.padding_side == "left"
    assert len(cast("list[dict[str, object]]", captured["chat_calls"])) == 2
    assert cast("list[dict[str, object]]", captured["chat_calls"])[0]["tokenize"] is False
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["tokenize"] is True
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["return_dict"] is True
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["return_tensors"] == "pt"
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["padding"] is False
    render_conversation = cast("list[dict[str, object]]", captured["chat_calls"])[0]["conversation"]
    assert cast("list[dict[str, object]]", render_conversation)[0]["role"] == "user"
    render_content = cast(
        "list[dict[str, object]]",
        cast("list[dict[str, object]]", render_conversation)[0]["content"],
    )
    assert render_content[0]["type"] == "image"
    assert render_content[1] == {"type": "text", "text": GLM_OCR_OCR_PROMPT}
    assert captured["device"] == "fake-device"
    assert captured["attention_mask_sum_dim"] == 1
    assert captured["generate_kwargs"] == {
        "input_ids": SimpleNamespace(shape=(1, 4)),
        "attention_mask": cast("object", captured["generate_kwargs"]["attention_mask"]),
        "mm_token_type_ids": "kept-mm-token-type-ids",
        "max_new_tokens": 8_192,
        "do_sample": False,
    }
    assert "token_type_ids" not in cast("dict[str, object]", captured["generate_kwargs"])
    assert captured["generated_ids"] == [["completion"]]
    assert captured["skip_special_tokens"] is True
    assert captured["clean_up_tokenization_spaces"] is False


@pytest.mark.asyncio
async def test_lfm25_huggingface_backend_uses_tokenized_chat_template_and_ties_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAttentionMask:
        def sum(self, dim: int) -> SimpleNamespace:
            captured["attention_mask_sum_dim"] = dim
            return SimpleNamespace(tolist=lambda: [4])

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(padding_side="right")

        def apply_chat_template(
            self,
            conversation: object,
            *,
            add_generation_prompt: bool,
            tokenize: bool,
            return_dict: bool | None = None,
            return_tensors: str | None = None,
            padding: bool | None = None,
        ) -> object:
            captured.setdefault("chat_calls", []).append(
                {
                    "conversation": conversation,
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": tokenize,
                    "return_dict": return_dict,
                    "return_tensors": return_tensors,
                    "padding": padding,
                }
            )
            if not tokenize:
                return "<lfm-rendered>"
            return FakeBatch(
                {
                    "input_ids": SimpleNamespace(shape=(1, 4)),
                    "attention_mask": FakeAttentionMask(),
                }
            )

        def __call__(self, **kwargs: object) -> object:
            del kwargs
            message = "processor(...) should not be used for LFM2.5-VL"
            raise AssertionError(message)

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["skip_special_tokens"] = skip_special_tokens
            captured["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return ["lfm transcription"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeLmHead:
        weight = "original-weight"

    class FakeModel:
        device = "fake-device"
        dtype = None

        def __init__(self) -> None:
            self.lm_head = FakeLmHead()

        def get_input_embeddings(self) -> SimpleNamespace:
            captured["get_input_embeddings_called"] = True
            return SimpleNamespace(weight="tied-weight")

        def generate(self, **kwargs: object) -> list[list[int | str]]:
            captured["generate_kwargs"] = kwargs
            return [[0, 1, 2, 3, "completion"]]

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            model = FakeModel()
            captured["model"] = model
            return model

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=None,
        ),
    )

    backend = cast(
        "LFM25VLOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=LFM2_5_VL_1_6B_MODEL_ID,
            )
        ),
    )
    result = await backend.ocr(DocumentPage.from_image(Image.new("RGB", (32, 32), color="white")))

    assert result.text == "lfm transcription"
    assert captured["processor_model_id"] == LFM2_5_VL_1_6B_MODEL_ID
    assert captured["model_model_id"] == LFM2_5_VL_1_6B_MODEL_ID
    assert captured["get_input_embeddings_called"] is True
    assert cast("FakeModel", captured["model"]).lm_head.weight == "tied-weight"
    assert cast("FakeProcessor", backend._processor).tokenizer.padding_side == "left"
    assert len(cast("list[dict[str, object]]", captured["chat_calls"])) == 2
    assert cast("list[dict[str, object]]", captured["chat_calls"])[0]["tokenize"] is False
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["tokenize"] is True
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["return_dict"] is True
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["return_tensors"] == "pt"
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["padding"] is False
    assert captured["device"] == "fake-device"
    assert captured["attention_mask_sum_dim"] == 1
    assert captured["generate_kwargs"] == {
        "input_ids": SimpleNamespace(shape=(1, 4)),
        "attention_mask": cast("object", captured["generate_kwargs"]["attention_mask"]),
        "max_new_tokens": 512,
        "do_sample": False,
        "repetition_penalty": 1.05,
    }
    assert captured["generated_ids"] == [["completion"]]
    assert captured["skip_special_tokens"] is True
    assert captured["clean_up_tokenization_spaces"] is False


@pytest.mark.asyncio
async def test_lfm25_huggingface_backend_batches_pages_with_tokenized_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAttentionMask:
        def sum(self, dim: int) -> SimpleNamespace:
            captured["attention_mask_sum_dim"] = dim
            return SimpleNamespace(tolist=lambda: [4, 4])

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(padding_side="right")

        def apply_chat_template(
            self,
            conversation: object,
            *,
            add_generation_prompt: bool,
            tokenize: bool,
            return_dict: bool | None = None,
            return_tensors: str | None = None,
            padding: bool | None = None,
        ) -> object:
            captured.setdefault("chat_calls", []).append(
                {
                    "conversation": conversation,
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": tokenize,
                    "return_dict": return_dict,
                    "return_tensors": return_tensors,
                    "padding": padding,
                }
            )
            if not tokenize:
                return "<lfm-rendered>"
            return FakeBatch(
                {
                    "input_ids": SimpleNamespace(shape=(2, 4)),
                    "attention_mask": FakeAttentionMask(),
                }
            )

        def __call__(self, **kwargs: object) -> object:
            del kwargs
            message = "processor(...) should not be used for LFM2.5-VL batches"
            raise AssertionError(message)

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["skip_special_tokens"] = skip_special_tokens
            captured["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return ["first transcription", "second transcription"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeLmHead:
        weight = "original-weight"

    class FakeModel:
        device = "fake-device"
        dtype = None

        def __init__(self) -> None:
            self.lm_head = FakeLmHead()

        def get_input_embeddings(self) -> SimpleNamespace:
            return SimpleNamespace(weight="tied-weight")

        def generate(self, **kwargs: object) -> list[list[int | str]]:
            captured["generate_kwargs"] = kwargs
            return [
                [0, 1, 2, 3, "first"],
                [0, 1, 2, 3, "second"],
            ]

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=None,
        ),
    )

    backend = cast(
        "LFM25VLOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=LFM2_5_VL_1_6B_MODEL_ID,
            )
        ),
    )
    results = await backend.ocr_batch(
        [
            DocumentPage.from_image(Image.new("RGB", (32, 32), color="white")),
            DocumentPage.from_image(Image.new("RGB", (32, 32), color="white")),
        ]
    )

    assert [result.text for result in results] == ["first transcription", "second transcription"]
    assert cast("FakeProcessor", backend._processor).tokenizer.padding_side == "left"
    assert len(cast("list[dict[str, object]]", captured["chat_calls"])) == 3
    assert cast("list[dict[str, object]]", captured["chat_calls"])[2]["tokenize"] is True
    assert cast("list[dict[str, object]]", captured["chat_calls"])[2]["padding"] is True
    assert captured["device"] == "fake-device"
    assert captured["attention_mask_sum_dim"] == 1
    assert captured["generated_ids"] == [["first"], ["second"]]
    assert captured["skip_special_tokens"] is True
    assert captured["clean_up_tokenization_spaces"] is False


@pytest.mark.asyncio
async def test_huggingface_vision_ocr_backend_uses_custom_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    prompt_logs: list[str] = []

    class FakeLogger:
        def debug(self, message: str, *args: object) -> None:
            prompt_logs.append(message % args if args else message)

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = object()

        def apply_chat_template(
            self,
            conversation: list[dict[str, object]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> str:
            captured["conversation"] = conversation
            captured["add_generation_prompt"] = add_generation_prompt
            captured["tokenize"] = tokenize
            return "<rendered>"

        def __call__(self, **kwargs: object) -> FakeBatch:
            captured["processor_kwargs"] = kwargs
            return FakeBatch({"input_ids": SimpleNamespace(shape=(1, 3))})

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["skip_special_tokens"] = skip_special_tokens
            captured["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return ["<xml>transcription</xml>"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeGeneratedIds:
        def __getitem__(self, key: object) -> object:
            captured["generated_slice"] = key
            return "trimmed-generated-ids"

    class FakeModel:
        device = "fake-device"

        def generate(self, **kwargs: object) -> FakeGeneratedIds:
            captured["generate_kwargs"] = kwargs
            return FakeGeneratedIds()

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    def fake_process_vision_info(
        conversation: list[dict[str, object]],
        *,
        return_video_kwargs: bool,
        return_video_metadata: bool,
    ) -> tuple[object, None, None]:
        captured["vision_conversation"] = conversation
        captured["return_video_kwargs"] = return_video_kwargs
        captured["return_video_metadata"] = return_video_metadata
        return "fake-image-inputs", None, None

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=fake_process_vision_info,
        ),
    )
    monkeypatch.setattr("churro_ocr._internal.prompt_logging.logger", FakeLogger())

    backend = HuggingFaceVisionOCRBackend(
        model_id="stanford-oval/churro-3B",
        template=CHURRO_3B_XML_TEMPLATE,
        generation_kwargs={"temperature": 0.0},
    )
    page = await OCRClient(backend).aocr_image(
        image=Image.new("RGBA", (5_000, 3_000), color=(255, 255, 255, 255))
    )

    assert page.text == "<xml>transcription</xml>"
    assert page.provider_name == "huggingface-transformers"
    assert page.model_name == "stanford-oval/churro-3B"
    assert captured["processor_model_id"] == "stanford-oval/churro-3B"
    assert captured["model_model_id"] == "stanford-oval/churro-3B"
    assert captured["conversation"][0]["role"] == "system"
    assert captured["conversation"][1]["content"][0]["type"] == "image"
    assert captured["conversation"][1]["content"][0]["image"].size == (2_500, 1_500)
    assert captured["conversation"][1]["content"][0]["image"].mode == "RGB"
    assert captured["add_generation_prompt"] is True
    assert captured["tokenize"] is False
    assert captured["processor_kwargs"]["text"] == ["<rendered>"]
    assert captured["processor_kwargs"]["images"] == ["fake-image-inputs"]
    assert captured["generate_kwargs"] == {
        "input_ids": SimpleNamespace(shape=(1, 3)),
        "max_new_tokens": DEFAULT_OCR_MAX_TOKENS,
        "temperature": 0.0,
    }
    assert captured["generated_slice"] == (slice(None), slice(3, None))
    assert len(prompt_logs) == 1
    assert "First OCR prompt payload for huggingface-transformers" in prompt_logs[0]
    assert '"rendered_prompt": "<rendered>"' in prompt_logs[0]
    assert '"image_preview"' in prompt_logs[0]


@pytest.mark.asyncio
async def test_huggingface_vision_ocr_backend_requires_chat_template_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcessor:
        tokenizer = object()

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            del model_id, kwargs
            return FakeProcessor()

    class FakeModel:
        device = "fake-device"

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            del model_id, kwargs
            return FakeModel()

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=lambda *_args, **_kwargs: (None, None, None),
        ),
    )

    backend = HuggingFaceVisionOCRBackend(
        model_id="custom-hf-model",
        template=CHURRO_3B_XML_TEMPLATE,
    )

    with pytest.raises(ConfigurationError, match="apply_chat_template"):
        await OCRClient(backend).aocr_image(
            image=Image.new("RGBA", (5_000, 3_000), color=(255, 255, 255, 255))
        )


def test_churro_3b_backend_uses_expected_defaults() -> None:
    backend = Churro3BOCRBackend()

    assert backend.model_id == "stanford-oval/churro-3B"
    assert backend.template == CHURRO_3B_XML_TEMPLATE
    assert backend.provider_name == "huggingface-transformers"


@pytest.mark.asyncio
async def test_dots_ocr_15_backend_uses_expected_runtime_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeProcessor:
        tokenizer = object()

        def apply_chat_template(
            self,
            conversation: list[dict[str, object]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> str:
            captured["conversation"] = conversation
            captured["add_generation_prompt"] = add_generation_prompt
            captured["tokenize"] = tokenize
            return "<dots-rendered>"

        def __call__(self, **kwargs: object) -> FakeBatch:
            captured["processor_kwargs"] = kwargs
            return FakeBatch(
                {
                    "input_ids": SimpleNamespace(shape=(1, 4)),
                    "mm_token_type_ids": "ignored-mm-token-type-ids",
                }
            )

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            del skip_special_tokens, clean_up_tokenization_spaces
            captured["generated_ids"] = generated_ids
            return ["dots transcription"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeGeneratedIds:
        def __getitem__(self, key: object) -> object:
            captured["generated_slice"] = key
            return "trimmed-dots-generated-ids"

    class FakeModel:
        device = "fake-device"

        def generate(self, **kwargs: object) -> FakeGeneratedIds:
            captured["generate_kwargs"] = kwargs
            return FakeGeneratedIds()

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    def fake_process_vision_info(
        conversation: list[dict[str, object]],
        *,
        return_video_kwargs: bool,
        return_video_metadata: bool,
    ) -> tuple[object, None, None]:
        captured["vision_conversation"] = conversation
        captured["return_video_kwargs"] = return_video_kwargs
        captured["return_video_metadata"] = return_video_metadata
        return "fake-image-inputs", None, None

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_causal_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=fake_process_vision_info,
        ),
    )
    monkeypatch.setattr(
        "churro_ocr.providers.hf._prepare_dots_ocr_model_dir",
        lambda model_id: model_id,
    )

    backend = DotsOCR15OCRBackend()
    page = await OCRClient(backend).aocr_image(image=Image.new("RGB", (32, 32), color="white"))

    assert page.text == "dots transcription"
    assert page.provider_name == "huggingface-transformers"
    assert page.model_name == "dots.ocr-1.5"
    assert captured["processor_model_id"] == DOTS_OCR_1_5_MODEL_ID
    assert captured["model_model_id"] == DOTS_OCR_1_5_MODEL_ID
    assert captured["processor_from_pretrained_kwargs"]["trust_remote_code"] is True
    assert "use_fast" not in captured["processor_from_pretrained_kwargs"]
    assert captured["model_from_pretrained_kwargs"]["trust_remote_code"] is True
    assert captured["conversation"][0]["role"] == "user"
    assert captured["conversation"][0]["content"][1]["text"] == DOTS_OCR_1_5_OCR_PROMPT
    assert captured["processor_kwargs"]["text"] == ["<dots-rendered>"]
    assert captured["processor_kwargs"]["images"] == ["fake-image-inputs"]
    assert "videos" not in captured["processor_kwargs"]
    assert captured["generate_kwargs"] == {
        "input_ids": SimpleNamespace(shape=(1, 4)),
        "max_new_tokens": DEFAULT_OCR_MAX_TOKENS,
    }
    assert captured["generated_slice"] == (slice(None), slice(4, None))


def test_dots_ocr_15_backend_uses_expected_defaults() -> None:
    backend = DotsOCR15OCRBackend()

    assert backend.model_id == DOTS_OCR_1_5_MODEL_ID
    assert backend.template == DOTS_OCR_1_5_OCR_TEMPLATE
    assert backend.trust_remote_code is True
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs["dtype"] in {"auto", "float32"}
    if backend.model_kwargs["dtype"] == "auto" and "device_map" in backend.model_kwargs:
        assert backend.model_kwargs["device_map"] == "auto"
        assert "max_memory" in backend.model_kwargs
    assert backend.generation_kwargs == {"max_new_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_deepseek_ocr_2_backend_uses_expected_defaults() -> None:
    backend = DeepSeekOCR2OCRBackend()

    assert backend.model_id == DEEPSEEK_OCR_2_MODEL_ID
    assert backend.template == DEEPSEEK_OCR_2_OCR_TEMPLATE
    assert backend.model_name == "DeepSeek-OCR-2"
    assert backend.trust_remote_code is True
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs == {"use_safetensors": True}
    assert backend.generation_kwargs == {"max_new_tokens": 8_192}
    assert backend.base_size == 1_024
    assert backend.image_size == 768
    assert backend.crop_mode is True


@pytest.mark.asyncio
async def test_mineru2_5_huggingface_backend_uses_two_step_generation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(
            self,
            conversation: object,
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> object:
            captured.setdefault("chat_calls", []).append(
                {
                    "conversation": conversation,
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": tokenize,
                }
            )
            if not tokenize:
                conversation_messages = cast("list[dict[str, object]]", conversation)
                user_content = cast("list[dict[str, object]]", conversation_messages[1]["content"])
                prompt = cast("str", user_content[1]["text"])
                return (f"<rendered:{prompt}>",)
            message = "tokenized chat template should not be used for MinerU2.5"
            raise AssertionError(message)

        def __call__(self, **kwargs: object) -> FakeBatch:
            captured.setdefault("processor_call_kwargs", []).append(kwargs)
            return FakeBatch(
                {
                    "input_ids": object(),
                    "attention_mask": object(),
                    "pixel_values": object(),
                }
            )

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeModel:
        device = "fake-device"
        dtype = "fake-bfloat16"
        config = SimpleNamespace(max_position_embeddings=8_192)

        def eval(self) -> FakeModel:
            captured["eval_called"] = True
            return self

        def generate(self, **kwargs: object) -> list[list[int]]:
            captured.setdefault("generate_kwargs", []).append(kwargs)
            return [[101, 102, 103]]

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    def _fake_process_vision_info(
        conversation: object,
        **_: object,
    ) -> tuple[list[object], None, None]:
        conversation_messages = cast("list[dict[str, object]]", conversation)
        user_content = cast("list[dict[str, object]]", conversation_messages[1]["content"])
        return ([user_content[0]["image"]], None, None)

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=_fake_process_vision_info,
        ),
    )
    monkeypatch.setattr(
        "churro_ocr.providers.hf._default_mineru25_model_kwargs",
        lambda: {"device_map": "auto", "dtype": "auto"},
    )
    decode_responses = [
        "<|box_start|>0 0 1000 100<|box_end|><|ref_start|>header<|ref_end|>\n"
        "<|box_start|>0 100 1000 550<|box_end|><|ref_start|>text<|ref_end|>\n"
        "<|box_start|>0 550 1000 800<|box_end|><|ref_start|>table<|ref_end|>\n"
        "<|box_start|>0 800 1000 1000<|box_end|><|ref_start|>equation<|ref_end|>",
        "Page header<|im_end|><|endoftext|>",
        "Body text<|im_end|><|endoftext|>",
        "<fcel>Year<fcel>Value<nl><fcel>1900<fcel>42<nl><|im_end|><|endoftext|>",
        "x = y<|im_end|><|endoftext|>",
    ]
    monkeypatch.setattr(
        "churro_ocr.providers.hf._decode_completion_texts_with_options",
        lambda _processor, _batch, _generated_ids, *, skip_special_tokens: (
            [decode_responses.pop(0)]
            if skip_special_tokens is False
            else (_ for _ in ()).throw(
                AssertionError("MinerU2.5 should preserve special tokens during decode")
            )
        ),
    )

    backend = cast(
        "MinerU25OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=MINERU2_5_2509_1_2B_MODEL_ID,
            )
        ),
    )
    result = await backend.ocr(
        DocumentPage.from_image(Image.new("RGBA", (32, 16), color=(255, 255, 255, 255)))
    )

    assert result.text == (
        "Page header\n\n"
        "Body text\n\n"
        "<table><tr><td>Year</td><td>Value</td></tr><tr><td>1900</td><td>42</td></tr></table>\n\n"
        "\\[\nx = y\n\\]"
    )
    assert result.metadata["output_format"] == "markdown"
    assert cast("dict[str, object]", result.metadata["pipeline_metrics"])["num_blocks"] == 4
    block_metadata = cast("list[dict[str, object]]", result.metadata["blocks"])
    assert [block["type"] for block in block_metadata] == ["header", "text", "table", "equation"]
    assert block_metadata[0]["content"] == "Page header"
    assert block_metadata[2]["content"] == (
        "<table><tr><td>Year</td><td>Value</td></tr><tr><td>1900</td><td>42</td></tr></table>"
    )
    assert block_metadata[3]["content"] == "\\[\nx = y\n\\]"
    assert captured["processor_model_id"] == MINERU2_5_2509_1_2B_MODEL_ID
    assert captured["model_model_id"] == MINERU2_5_2509_1_2B_MODEL_ID
    assert captured["processor_from_pretrained_kwargs"] == {
        "trust_remote_code": False,
        "use_fast": True,
    }
    assert captured["model_from_pretrained_kwargs"] == {
        "trust_remote_code": False,
        "device_map": "auto",
        "dtype": "auto",
    }
    assert captured["eval_called"] is True
    chat_calls = cast("list[dict[str, object]]", captured["chat_calls"])
    render_conversation = chat_calls[0]["conversation"]
    assert cast("list[dict[str, object]]", render_conversation)[0]["role"] == "system"
    assert cast("list[dict[str, object]]", render_conversation)[1]["role"] == "user"
    user_content = cast(
        "list[dict[str, object]]",
        cast("list[dict[str, object]]", render_conversation)[1]["content"],
    )
    assert user_content[0]["type"] == "image"
    assert user_content[1] == {"type": "text", "text": MINERU2_5_2509_1_2B_LAYOUT_PROMPT}
    processor_calls = cast("list[dict[str, object]]", captured["processor_call_kwargs"])
    prompt_texts = [cast("list[str]", call["text"])[0] for call in processor_calls]
    assert prompt_texts == [
        f"<rendered:{MINERU2_5_2509_1_2B_LAYOUT_PROMPT}>",
        f"<rendered:{MINERU2_5_2509_1_2B_OCR_PROMPT}>",
        f"<rendered:{MINERU2_5_2509_1_2B_OCR_PROMPT}>",
        f"<rendered:{MINERU2_5_2509_1_2B_TABLE_PROMPT}>",
        f"<rendered:{MINERU2_5_2509_1_2B_FORMULA_PROMPT}>",
    ]
    assert processor_calls[0]["return_tensors"] == "pt"
    assert processor_calls[0]["padding"] is True
    assert cast("list[Image.Image]", processor_calls[0]["images"])[0].size == (1_036, 1_036)
    assert captured["device"] == "fake-device"
    generate_kwargs = cast("list[dict[str, object]]", captured["generate_kwargs"])
    assert len(generate_kwargs) == 5
    assert all(kwargs["do_sample"] is False for kwargs in generate_kwargs)
    assert all(kwargs["no_repeat_ngram_size"] == 100 for kwargs in generate_kwargs)
    assert all(kwargs["repetition_penalty"] == 1.0 for kwargs in generate_kwargs)
    assert all(kwargs["max_length"] == 8_192 for kwargs in generate_kwargs)


def test_mineru2_5_backend_uses_expected_defaults() -> None:
    backend = MinerU25OCRBackend()

    assert backend.model_id == MINERU2_5_2509_1_2B_MODEL_ID
    assert backend.template == MINERU2_5_2509_1_2B_OCR_TEMPLATE
    assert backend.layout_template == MINERU2_5_2509_1_2B_LAYOUT_TEMPLATE
    assert backend.table_template == MINERU2_5_2509_1_2B_TABLE_TEMPLATE
    assert backend.formula_template == MINERU2_5_2509_1_2B_FORMULA_TEMPLATE
    assert backend.image_analysis_template == MINERU2_5_2509_1_2B_IMAGE_ANALYSIS_TEMPLATE
    assert backend.model_name == "MinerU2.5-2509-1.2B"
    assert backend.trust_remote_code is False
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs == {}
    assert backend.generation_kwargs == {}
    assert backend.image_preprocessor(Image.new("RGBA", (10, 10), color=(255, 255, 255, 255))).mode == "RGB"


def test_decode_completion_texts_can_preserve_special_tokens() -> None:
    captured: dict[str, object] = {}

    class FakeProcessor:
        def batch_decode(
            self, ids: object, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool
        ) -> list[str]:
            captured["ids"] = ids
            captured["skip_special_tokens"] = skip_special_tokens
            captured["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return ["<|box_start|>0 0 1000 100<|box_end|>"]

    class FakeMask:
        def sum(self, dim: int) -> object:
            assert dim == 1
            return SimpleNamespace(tolist=lambda: [2])

    batch = {
        "attention_mask": FakeMask(),
        "input_ids": SimpleNamespace(shape=(1, 2)),
    }
    generated_ids = [[11, 12, 13, 14]]

    decoded = hf_module._decode_completion_texts_with_options(
        FakeProcessor(),
        batch,
        generated_ids,
        skip_special_tokens=False,
    )

    assert decoded == ["<|box_start|>0 0 1000 100<|box_end|>"]
    assert captured["ids"] == [[13, 14]]
    assert captured["skip_special_tokens"] is False
    assert captured["clean_up_tokenization_spaces"] is False


def test_resolve_model_max_length_supports_qwen2vl_text_config() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=16_384),
        )
    )

    assert hf_module._resolve_model_max_length(model) == 16_384


def test_dots_mocr_backend_uses_expected_defaults() -> None:
    backend = DotsMOCROCRBackend()

    assert backend.model_id == DOTS_MOCR_MODEL_ID
    assert backend.template == DOTS_MOCR_OCR_TEMPLATE
    assert backend.model_name == "dots.mocr"
    assert backend.trust_remote_code is True
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs["dtype"] in {"auto", "float32"}
    if backend.model_kwargs["dtype"] == "auto" and "device_map" in backend.model_kwargs:
        assert backend.model_kwargs["device_map"] == "auto"
        assert "max_memory" in backend.model_kwargs
    assert backend.generation_kwargs == {"max_new_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_dots_mocr_template_matches_upstream_prompt() -> None:
    page = DocumentPage.from_image(Image.new("RGB", (20, 20), color="white"))

    conversation = DOTS_MOCR_OCR_TEMPLATE.build_conversation(page)

    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"][0]["type"] == "image"
    assert conversation[0]["content"][1]["text"] == DOTS_MOCR_OCR_PROMPT


@pytest.mark.asyncio
async def test_paddleocr_vl_15_backend_uses_tokenized_chat_template_and_profile_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAttentionMask:
        def sum(self, dim: int) -> SimpleNamespace:
            captured["attention_mask_sum_dim"] = dim
            return SimpleNamespace(tolist=lambda: [4])

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()
            self.image_processor = SimpleNamespace(min_pixels=112_896, max_pixels=1_003_520)

        def apply_chat_template(
            self,
            conversation: object,
            *,
            add_generation_prompt: bool,
            tokenize: bool,
            return_dict: bool | None = None,
            return_tensors: str | None = None,
            processor_kwargs: dict[str, object] | None = None,
        ) -> object:
            captured.setdefault("chat_calls", []).append(
                {
                    "conversation": conversation,
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": tokenize,
                    "return_dict": return_dict,
                    "return_tensors": return_tensors,
                    "processor_kwargs": processor_kwargs,
                }
            )
            if not tokenize:
                return "<paddle-rendered>"
            return FakeBatch(
                {
                    "input_ids": SimpleNamespace(shape=(1, 4)),
                    "attention_mask": FakeAttentionMask(),
                }
            )

        def __call__(self, **kwargs: object) -> object:
            del kwargs
            message = "processor(...) should not be used for PaddleOCR-VL"
            raise AssertionError(message)

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["skip_special_tokens"] = skip_special_tokens
            captured["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return ["OCR:\nassistant\npaddle transcription"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeModel:
        device = "fake-device"
        dtype = None

        def eval(self) -> FakeModel:
            captured["eval_called"] = True
            return self

        def generate(self, **kwargs: object) -> list[list[int | str]]:
            captured["generate_kwargs"] = kwargs
            return [[0, 1, 2, 3, "completion"]]

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=None,
        ),
    )

    backend = cast(
        "PaddleOCRVL15OCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=PADDLEOCR_VL_1_5_MODEL_ID,
            )
        ),
    )
    result = await backend.ocr(DocumentPage.from_image(Image.new("RGB", (32, 32), color="white")))

    assert result.text == "paddle transcription"
    assert captured["processor_model_id"] == PADDLEOCR_VL_1_5_MODEL_ID
    assert captured["model_model_id"] == PADDLEOCR_VL_1_5_MODEL_ID
    assert captured["eval_called"] is True
    assert cast("FakeProcessor", backend._processor).tokenizer.padding_side == "left"
    assert len(cast("list[dict[str, object]]", captured["chat_calls"])) == 2
    assert cast("list[dict[str, object]]", captured["chat_calls"])[0]["tokenize"] is False
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["tokenize"] is True
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["return_dict"] is True
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["return_tensors"] == "pt"
    assert cast("list[dict[str, object]]", captured["chat_calls"])[1]["processor_kwargs"] == {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "min_pixels": 112_896,
            "max_pixels": 1_003_520,
        },
    }
    render_conversation = cast("list[dict[str, object]]", captured["chat_calls"])[0]["conversation"]
    assert cast("list[dict[str, object]]", render_conversation)[0]["role"] == "user"
    render_content = cast(
        "list[dict[str, object]]",
        cast("list[dict[str, object]]", render_conversation)[0]["content"],
    )
    assert render_content[0]["type"] == "image"
    assert render_content[1] == {"type": "text", "text": PADDLEOCR_VL_1_5_OCR_PROMPT}
    assert captured["device"] == "fake-device"
    assert captured["attention_mask_sum_dim"] == 1
    assert captured["generate_kwargs"] == {
        "input_ids": SimpleNamespace(shape=(1, 4)),
        "attention_mask": cast("object", captured["generate_kwargs"]["attention_mask"]),
        "max_new_tokens": 4_096,
        "do_sample": False,
    }
    assert captured["generated_ids"] == [["completion"]]
    assert captured["skip_special_tokens"] is True
    assert captured["clean_up_tokenization_spaces"] is False


def test_paddleocr_vl_15_backend_uses_expected_defaults() -> None:
    backend = PaddleOCRVL15OCRBackend()

    assert backend.model_id == PADDLEOCR_VL_1_5_MODEL_ID
    assert backend.template == PADDLEOCR_VL_1_5_OCR_TEMPLATE
    assert backend.trust_remote_code is False
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs == {}
    assert backend.generation_kwargs == {
        "max_new_tokens": 4_096,
        "do_sample": False,
    }


def test_patch_dots_ocr_prepare_inputs_for_generation_handles_missing_cache_position() -> None:
    captured: dict[str, object] = {}

    class BaseModel:
        def prepare_inputs_for_generation(
            self,
            input_ids: object,
            *,
            past_key_values: object = None,
            inputs_embeds: object = None,
            pixel_values: object = None,
            attention_mask: object = None,
            cache_position: object = None,
            num_logits_to_keep: object = None,
            **kwargs: object,
        ) -> dict[str, object]:
            del pixel_values
            captured["base_call"] = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "num_logits_to_keep": num_logits_to_keep,
                "kwargs": kwargs,
            }
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    class FakeDotsModel(BaseModel):
        pass

    model = FakeDotsModel()

    hf_module._patch_dots_ocr_prepare_inputs_for_generation(model)
    first_inputs = model.prepare_inputs_for_generation(
        "tokens",
        pixel_values="pixels",
        attention_mask="mask",
        cache_position=None,
    )
    later_inputs = model.prepare_inputs_for_generation(
        "tokens",
        pixel_values="pixels",
        attention_mask="mask",
        cache_position=[3],
    )

    assert first_inputs == {
        "input_ids": "tokens",
        "attention_mask": "mask",
        "pixel_values": "pixels",
    }
    assert later_inputs == {
        "input_ids": "tokens",
        "attention_mask": "mask",
    }
    assert captured["base_call"] == {
        "input_ids": "tokens",
        "past_key_values": None,
        "inputs_embeds": None,
        "attention_mask": "mask",
        "cache_position": [3],
        "num_logits_to_keep": None,
        "kwargs": {},
    }

    hf_module._patch_dots_ocr_prepare_inputs_for_generation(model)
    assert cast("Any", model)._churro_dots_prepare_inputs_patched is True


def test_patch_dots_ocr_prepare_inputs_for_generation_skips_wrapped_dots_method() -> None:
    captured: dict[str, object] = {}

    class BaseModel:
        def prepare_inputs_for_generation(
            self,
            input_ids: object,
            *,
            past_key_values: object = None,
            inputs_embeds: object = None,
            pixel_values: object = None,
            attention_mask: object = None,
            cache_position: object = None,
            num_logits_to_keep: object = None,
            **kwargs: object,
        ) -> dict[str, object]:
            del pixel_values
            captured["base_call"] = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "num_logits_to_keep": num_logits_to_keep,
                "kwargs": kwargs,
            }
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    class FakeDotsModel(BaseModel):
        def prepare_inputs_for_generation(
            self,
            input_ids: object,
            *,
            past_key_values: object = None,
            inputs_embeds: object = None,
            pixel_values: object = None,
            attention_mask: object = None,
            cache_position: object = None,
            num_logits_to_keep: object = None,
            **kwargs: object,
        ) -> dict[str, object]:
            del past_key_values, inputs_embeds, num_logits_to_keep, kwargs
            if cast("Any", cache_position)[0] == 0:
                return {"pixel_values": pixel_values}
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    class WrappedFakeDotsModel(FakeDotsModel):
        pass

    model = WrappedFakeDotsModel()

    hf_module._patch_dots_ocr_prepare_inputs_for_generation(model)
    inputs = model.prepare_inputs_for_generation(
        "tokens",
        pixel_values="pixels",
        attention_mask="mask",
        cache_position=None,
    )

    assert inputs == {
        "input_ids": "tokens",
        "attention_mask": "mask",
        "pixel_values": "pixels",
    }
    assert captured["base_call"] == {
        "input_ids": "tokens",
        "past_key_values": None,
        "inputs_embeds": None,
        "attention_mask": "mask",
        "cache_position": None,
        "num_logits_to_keep": None,
        "kwargs": {},
    }


def test_patch_dots_ocr_prepare_inputs_for_generation_skips_duplicate_wrapped_dots_methods() -> None:
    captured: dict[str, object] = {}

    class BaseModel:
        def prepare_inputs_for_generation(
            self,
            input_ids: object,
            *,
            past_key_values: object = None,
            inputs_embeds: object = None,
            pixel_values: object = None,
            attention_mask: object = None,
            cache_position: object = None,
            num_logits_to_keep: object = None,
            **kwargs: object,
        ) -> dict[str, object]:
            del pixel_values
            captured["base_call"] = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "num_logits_to_keep": num_logits_to_keep,
                "kwargs": kwargs,
            }
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    def shared_prepare_inputs_for_generation(
        self: object,
        input_ids: object,
        *,
        past_key_values: object = None,
        inputs_embeds: object = None,
        pixel_values: object = None,
        attention_mask: object = None,
        cache_position: object = None,
        num_logits_to_keep: object = None,
        **kwargs: object,
    ) -> dict[str, object]:
        del self, input_ids, past_key_values, inputs_embeds, pixel_values, attention_mask
        del num_logits_to_keep, kwargs
        if cast("Any", cache_position)[0] == 0:
            return {"unexpected": True}
        return {"unexpected": False}

    class FakeDotsOwner(BaseModel):
        pass

    class WrappedFakeDotsModel(FakeDotsOwner):
        pass

    cast("Any", FakeDotsOwner).prepare_inputs_for_generation = shared_prepare_inputs_for_generation
    cast("Any", WrappedFakeDotsModel).prepare_inputs_for_generation = shared_prepare_inputs_for_generation

    model = WrappedFakeDotsModel()

    hf_module._patch_dots_ocr_prepare_inputs_for_generation(model)
    inputs = model.prepare_inputs_for_generation(
        "tokens",
        pixel_values="pixels",
        attention_mask="mask",
        cache_position=None,
    )

    assert inputs == {
        "input_ids": "tokens",
        "attention_mask": "mask",
        "pixel_values": "pixels",
    }
    assert captured["base_call"] == {
        "input_ids": "tokens",
        "past_key_values": None,
        "inputs_embeds": None,
        "attention_mask": "mask",
        "cache_position": None,
        "num_logits_to_keep": None,
        "kwargs": {},
    }


def test_dots_ocr_15_backend_batch_strips_unused_mm_token_type_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAttentionMask:
        def sum(self, dim: int) -> SimpleNamespace:
            captured["attention_mask_sum_dim"] = dim
            return SimpleNamespace(tolist=lambda: [4, 4])

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeProcessor:
        tokenizer = object()

        def apply_chat_template(
            self,
            conversation: list[dict[str, object]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> str:
            del add_generation_prompt, tokenize
            image_content = cast("list[dict[str, object]]", conversation[0]["content"])
            image = cast("Image.Image", image_content[0]["image"])
            return f"<dots-rendered:{image.width}>"

        def __call__(self, **kwargs: object) -> FakeBatch:
            captured["processor_kwargs"] = kwargs
            return FakeBatch(
                {
                    "input_ids": SimpleNamespace(shape=(2, 4)),
                    "attention_mask": FakeAttentionMask(),
                    "mm_token_type_ids": "ignored-mm-token-type-ids",
                }
            )

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["decode_kwargs"] = (skip_special_tokens, clean_up_tokenization_spaces)
            return ["dots batch 1", "dots batch 2"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeModel:
        device = "fake-device"

        def generate(self, **kwargs: object) -> list[list[int]]:
            captured["generate_kwargs"] = kwargs
            return [
                [100, 101, 102, 103, 104, 105],
                [200, 201, 202, 203, 204, 205],
            ]

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    def fake_process_vision_info(
        conversation: list[dict[str, object]],
        *,
        return_video_kwargs: bool,
        return_video_metadata: bool,
    ) -> tuple[object, None, None]:
        del conversation, return_video_kwargs, return_video_metadata
        return "fake-image-inputs", None, None

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_causal_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=fake_process_vision_info,
        ),
    )
    monkeypatch.setattr(
        "churro_ocr.providers.hf._prepare_dots_ocr_model_dir",
        lambda model_id: model_id,
    )

    backend = DotsOCR15OCRBackend()
    results = backend._ocr_batch_sync(
        [
            DocumentPage.from_image(Image.new("RGB", (32, 32), color="white")),
            DocumentPage.from_image(Image.new("RGB", (64, 64), color="white")),
        ]
    )

    assert [result.text for result in results] == ["dots batch 1", "dots batch 2"]
    assert captured["processor_kwargs"]["text"] == ["<dots-rendered:32>", "<dots-rendered:64>"]
    assert captured["processor_kwargs"]["images"] == [["fake-image-inputs"], ["fake-image-inputs"]]
    assert captured["generate_kwargs"] == {
        "input_ids": SimpleNamespace(shape=(2, 4)),
        "attention_mask": cast("object", captured["generate_kwargs"]["attention_mask"]),
        "max_new_tokens": DEFAULT_OCR_MAX_TOKENS,
    }
    assert captured["attention_mask_sum_dim"] == 1
    assert captured["generated_ids"] == [[104, 105], [204, 205]]
    assert captured["decode_kwargs"] == (True, False)


@pytest.mark.asyncio
async def test_huggingface_vision_ocr_backend_strips_default_output_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            del device
            return self

    class FakeProcessor:
        tokenizer = object()

        def apply_chat_template(
            self,
            conversation: list[dict[str, object]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> str:
            del conversation, add_generation_prompt, tokenize
            return "<rendered>"

        def __call__(self, **kwargs: object) -> FakeBatch:
            del kwargs
            return FakeBatch({"input_ids": SimpleNamespace(shape=(1, 3))})

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            del generated_ids, skip_special_tokens, clean_up_tokenization_spaces
            return [f"<{DEFAULT_OCR_OUTPUT_TAG}>\ntranscription\n</{DEFAULT_OCR_OUTPUT_TAG}>"]

    class FakeProcessorCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            del model_id, kwargs
            return FakeProcessor()

    class FakeGeneratedIds:
        def __getitem__(self, key: object) -> object:
            del key
            return "trimmed-generated-ids"

    class FakeModel:
        device = "fake-device"

        def generate(self, **kwargs: object) -> FakeGeneratedIds:
            del kwargs
            return FakeGeneratedIds()

    class FakeModelCls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            del model_id, kwargs
            return FakeModel()

    def fake_process_vision_info(
        conversation: list[dict[str, object]],
        *,
        return_video_kwargs: bool,
        return_video_metadata: bool,
    ) -> tuple[object, None, None]:
        del conversation, return_video_kwargs, return_video_metadata
        return "fake-image-inputs", None, None

    monkeypatch.setattr(
        "churro_ocr.providers.hf._load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=fake_process_vision_info,
        ),
    )

    backend = build_ocr_backend(OCRBackendSpec(provider="hf", model="example/model"))
    page = await OCRClient(backend).aocr_image(image=Image.new("RGB", (32, 32), color="white"))

    assert page.text == "transcription"


@pytest.mark.parametrize(
    ("loader_name", "expected_model_attr"),
    [
        ("_load_hf_runtime", "AutoModelForImageTextToText"),
        ("_load_hf_causal_runtime", "AutoModelForCausalLM"),
    ],
)
def test_hf_runtime_loaders_use_installed_modules(
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    expected_model_attr: str,
) -> None:
    process_vision_info = object()
    qwen_module = ModuleType("qwen_vl_utils")
    cast("Any", qwen_module).process_vision_info = process_vision_info

    processor_cls = object()
    image_text_model_cls = object()
    causal_model_cls = object()
    transformers_module = ModuleType("transformers")
    cast("Any", transformers_module).AutoProcessor = processor_cls
    cast("Any", transformers_module).AutoModelForImageTextToText = image_text_model_cls
    cast("Any", transformers_module).AutoModelForCausalLM = causal_model_cls

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    runtime = getattr(hf_module, loader_name)()

    assert runtime.processor_cls is processor_cls
    assert runtime.model_cls is getattr(transformers_module, expected_model_attr)
    assert runtime.process_vision_info is process_vision_info


def test_patch_dots_ocr_vision_module_rewrites_flash_attn_and_dtype_lines(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vision_module_path = model_dir / "modeling_dots_vision.py"
    vision_module_path.write_text(
        "\n".join(
            [
                "from flash_attn import flash_attn_varlen_func",
                "",
                "def forward(self, hidden_states):",
                hf_module._DOTS_FORCE_BFLOAT16_LINE,
                "    return hidden_states",
            ]
        )
        + "\n"
    )

    hf_module._patch_dots_ocr_vision_module(model_dir)
    patched_once = vision_module_path.read_text()
    hf_module._patch_dots_ocr_vision_module(model_dir)

    assert hf_module._DOTS_FLASH_ATTN_FALLBACK.strip() in patched_once
    assert patched_once.startswith("try:\n")
    assert hf_module._DOTS_FORCE_BFLOAT16_LINE not in patched_once
    assert hf_module._DOTS_WEIGHT_DTYPE_LINE in patched_once
    assert vision_module_path.read_text() == patched_once


def test_prepare_dots_ocr_model_dir_downloads_and_patches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    download_calls: list[tuple[str, Path]] = []
    patched_paths: list[Path] = []

    huggingface_hub_module = ModuleType("huggingface_hub")
    cast("Any", huggingface_hub_module).snapshot_download = lambda *, repo_id, local_dir: (
        download_calls.append((repo_id, local_dir))
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub_module)
    monkeypatch.setattr(hf_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        hf_module,
        "_patch_dots_ocr_vision_module",
        lambda model_dir: patched_paths.append(model_dir),
    )

    prepared_path = hf_module._prepare_dots_ocr_model_dir("org/model.id")

    expected_dir = tmp_path / ".cache" / "churro-ocr" / "hf" / "DotsOCR_1_5" / "org__model_id"
    assert prepared_path == str(expected_dir)
    assert download_calls == [("org/model.id", expected_dir)]
    assert patched_paths == [expected_dir]


@pytest.mark.parametrize(
    ("cuda_available", "free_bytes", "expected"),
    [
        (False, 0, {"dtype": "auto"}),
        (True, 7 * 1024**3, {"dtype": "float32"}),
        (
            True,
            16 * 1024**3,
            {"dtype": "auto", "device_map": "auto", "max_memory": {0: "15GiB", "cpu": "128GiB"}},
        ),
    ],
)
def test_default_dots_ocr_1_5_model_kwargs_handles_cuda_variants(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    free_bytes: int,
    expected: dict[str, object],
) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return free_bytes, 0

    torch_module = ModuleType("torch")
    cast("Any", torch_module).cuda = _FakeCuda
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert hf_module._default_dots_ocr_1_5_model_kwargs() == expected


def test_default_dots_ocr_1_5_model_kwargs_falls_back_when_mem_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            message = "cudaMemGetInfo failed"
            raise RuntimeError(message)

    torch_module = ModuleType("torch")
    cast("Any", torch_module).cuda = _FakeCuda
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert hf_module._default_dots_ocr_1_5_model_kwargs() == {"dtype": "auto"}


@pytest.mark.asyncio
async def test_huggingface_vision_ocr_backend_batches_pages_with_custom_vision_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    prompt_logs: list[str] = []
    chat_calls: list[tuple[bool, bool]] = []

    class FakeLogger:
        def debug(self, message: str, *args: object) -> None:
            prompt_logs.append(message % args if args else message)

    class FakeFloatDType:
        is_floating_point = True

    class FakeFloatTensor:
        def __init__(self) -> None:
            self.dtype = FakeFloatDType()
            self.to_calls: list[object] = []

        def to(self, *, dtype: object) -> FakeFloatTensor:
            self.to_calls.append(dtype)
            return self

    class FakeAttentionMask:
        def sum(self, *, dim: int) -> SimpleNamespace:
            captured["sum_dim"] = dim
            return SimpleNamespace(tolist=lambda: [2, 1])

    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> FakeBatch:
            captured["device"] = device
            return self

    class FakeProcessor:
        tokenizer = object()

        def apply_chat_template(
            self,
            conversation: list[dict[str, object]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> str:
            chat_calls.append((add_generation_prompt, tokenize))
            user_content = cast("list[dict[str, object]]", conversation[0]["content"])
            image = cast("Image.Image", user_content[0]["image"])
            return f"prompt:{image.width}"

        def __call__(self, **kwargs: object) -> FakeBatch:
            captured["processor_kwargs"] = kwargs
            fake_pixel_values = FakeFloatTensor()
            captured["pixel_values"] = fake_pixel_values
            return FakeBatch(
                {
                    "attention_mask": FakeAttentionMask(),
                    "pixel_values": fake_pixel_values,
                }
            )

        def batch_decode(
            self,
            generated_ids: object,
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> list[str]:
            captured["generated_ids"] = generated_ids
            captured["decode_kwargs"] = (skip_special_tokens, clean_up_tokenization_spaces)
            return ["first text", "second text"]

    class FakeProcessorCls:
        call_count = 0

        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeProcessor:
            FakeProcessorCls.call_count += 1
            captured["processor_model_id"] = model_id
            captured["processor_from_pretrained_kwargs"] = kwargs
            return FakeProcessor()

    class FakeModel:
        device = "cuda:0"
        dtype = "float16"

        def generate(self, **kwargs: object) -> list[list[int]]:
            captured["generate_kwargs"] = kwargs
            return [
                [100, 101, 102, 103],
                [200, 201, 202],
            ]

    class FakeModelCls:
        call_count = 0

        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            FakeModelCls.call_count += 1
            captured["model_model_id"] = model_id
            captured["model_from_pretrained_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        hf_module,
        "_load_hf_runtime",
        lambda: SimpleNamespace(
            processor_cls=FakeProcessorCls,
            model_cls=FakeModelCls,
            process_vision_info=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"unexpected process_vision_info call: {args!r}, {kwargs!r}")
            ),
        ),
    )
    monkeypatch.setattr("churro_ocr._internal.prompt_logging.logger", FakeLogger())

    def _vision_input_builder(conversation: list[dict[str, object]]) -> tuple[str, str]:
        user_content = cast("list[dict[str, object]]", conversation[0]["content"])
        image = cast("Image.Image", user_content[0]["image"])
        return f"image:{image.width}", f"video:{image.width}"

    backend = HuggingFaceVisionOCRBackend(
        model_id="example/model",
        template=HFChatTemplate(user_prompt="prompt"),
        processor_kwargs={"use_fast": False},
        model_kwargs={"device_map": "auto"},
        generation_kwargs={"temperature": 0.1},
        vision_input_builder=_vision_input_builder,
    )
    pages = [
        DocumentPage.from_image(Image.new("RGB", (10, 10), color="white")),
        DocumentPage.from_image(Image.new("RGB", (20, 20), color="white")),
    ]

    results = await backend.ocr_batch(pages)

    assert [result.text for result in results] == ["first text", "second text"]
    assert captured["processor_model_id"] == "example/model"
    assert captured["processor_from_pretrained_kwargs"] == {
        "trust_remote_code": False,
        "use_fast": False,
    }
    assert captured["model_model_id"] == "example/model"
    assert captured["model_from_pretrained_kwargs"] == {
        "trust_remote_code": False,
        "device_map": "auto",
    }
    assert captured["processor_kwargs"] == {
        "text": ["prompt:10", "prompt:20"],
        "images": [["image:10"], ["image:20"]],
        "videos": [["video:10"], ["video:20"]],
        "return_tensors": "pt",
        "padding": True,
    }
    assert captured["device"] == "cuda:0"
    fake_pixel_values = cast("Any", captured["pixel_values"])
    assert fake_pixel_values.to_calls == ["float16"]
    assert captured["sum_dim"] == 1
    generate_kwargs = cast("dict[str, object]", captured["generate_kwargs"])
    assert generate_kwargs["temperature"] == 0.1
    assert generate_kwargs["max_new_tokens"] == DEFAULT_OCR_MAX_TOKENS
    assert generate_kwargs["attention_mask"].__class__.__name__ == ("FakeAttentionMask")
    assert generate_kwargs["pixel_values"] is fake_pixel_values
    assert captured["generated_ids"] == [[102, 103], [201, 202]]
    assert captured["decode_kwargs"] == (True, False)
    assert FakeProcessorCls.call_count == 1
    assert FakeModelCls.call_count == 1
    assert chat_calls == [(True, False), (True, False)]
    assert len(prompt_logs) == 1
    assert "First OCR prompt payload for huggingface-transformers" in prompt_logs[0]


def test_huggingface_vision_ocr_backend_batch_returns_empty_list_for_no_pages() -> None:
    backend = HuggingFaceVisionOCRBackend(
        model_id="example/model",
        template=HFChatTemplate(user_prompt="prompt"),
    )

    assert backend._ocr_batch_sync([]) == []


@pytest.mark.parametrize("vision_config", [{}, SimpleNamespace()])
def test_dots_ocr_15_backend_get_model_sets_sdpa_on_vision_config(
    monkeypatch: pytest.MonkeyPatch,
    vision_config: dict[str, str] | SimpleNamespace,
) -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(vision_config=vision_config)

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_source: str, **kwargs: object) -> object:
            captured["config_model_source"] = model_source
            captured["config_from_pretrained_kwargs"] = kwargs
            return config

    class FakeModelCls:
        call_count = 0

        @staticmethod
        def from_pretrained(model_source: str, **kwargs: object) -> object:
            FakeModelCls.call_count += 1
            captured["model_model_source"] = model_source
            captured["model_from_pretrained_kwargs"] = kwargs
            return object()

    transformers_module = ModuleType("transformers")
    cast("Any", transformers_module).AutoConfig = FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(
        hf_module,
        "_prepare_dots_ocr_model_dir",
        lambda model_id: f"/prepared/{model_id.replace('/', '__')}",
    )

    backend = DotsOCR15OCRBackend(model_kwargs={"torch_dtype": "auto"})
    runtime = hf_module._HFRuntime(
        processor_cls=object(),
        model_cls=FakeModelCls,
        process_vision_info=object(),
    )

    first_model = backend._get_model(runtime)
    second_model = backend._get_model(runtime)

    assert first_model is second_model
    assert captured["config_model_source"] == "/prepared/kristaller486__dots.ocr-1.5"
    assert captured["config_from_pretrained_kwargs"] == {"trust_remote_code": True}
    assert captured["model_model_source"] == "/prepared/kristaller486__dots.ocr-1.5"
    assert captured["model_from_pretrained_kwargs"] == {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    assert FakeModelCls.call_count == 1
    if isinstance(vision_config, dict):
        assert vision_config["attn_implementation"] == "sdpa"
    else:
        assert vision_config.attn_implementation == "sdpa"

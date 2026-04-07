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
from churro_ocr.providers.hf import (
    ChandraOCR2OCRBackend,
    Churro3BOCRBackend,
    DotsOCR15OCRBackend,
    HuggingFaceVisionOCRBackend,
    LFM25VLOCRBackend,
)
from churro_ocr.providers.specs import DEFAULT_OCR_MAX_TOKENS, lfm2_5_vl_text_postprocessor
from churro_ocr.templates import (
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    CHURRO_3B_XML_TEMPLATE,
    DOTS_OCR_1_5_MODEL_ID,
    DOTS_OCR_1_5_OCR_PROMPT,
    DOTS_OCR_1_5_OCR_TEMPLATE,
    LFM2_5_VL_1_6B_MODEL_ID,
    LFM2_5_VL_1_6B_OCR_TEMPLATE,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
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
            raise AssertionError("processor(...) should not be used for LFM2.5-VL")

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
            raise AssertionError("processor(...) should not be used for LFM2.5-VL batches")

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
    cast(Any, qwen_module).process_vision_info = process_vision_info

    processor_cls = object()
    image_text_model_cls = object()
    causal_model_cls = object()
    transformers_module = ModuleType("transformers")
    cast(Any, transformers_module).AutoProcessor = processor_cls
    cast(Any, transformers_module).AutoModelForImageTextToText = image_text_model_cls
    cast(Any, transformers_module).AutoModelForCausalLM = causal_model_cls

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
    cast(Any, huggingface_hub_module).snapshot_download = lambda *, repo_id, local_dir: download_calls.append(
        (repo_id, local_dir)
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
    cast(Any, torch_module).cuda = _FakeCuda
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert hf_module._default_dots_ocr_1_5_model_kwargs() == expected


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
            user_content = cast(list[dict[str, object]], conversation[0]["content"])
            image = cast(Image.Image, user_content[0]["image"])
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
        user_content = cast(list[dict[str, object]], conversation[0]["content"])
        image = cast(Image.Image, user_content[0]["image"])
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
    fake_pixel_values = cast(Any, captured["pixel_values"])
    assert fake_pixel_values.to_calls == ["float16"]
    assert captured["sum_dim"] == 1
    generate_kwargs = cast(dict[str, object], captured["generate_kwargs"])
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
    cast(Any, transformers_module).AutoConfig = FakeAutoConfig
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

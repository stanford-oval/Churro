from __future__ import annotations

import base64
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from churro_ocr._internal.litellm import LiteLLMTransport
from churro_ocr.errors import ProviderError
from churro_ocr.page_detection import DocumentPage
from churro_ocr.prompts import (
    CHANDRA_OCR_LAYOUT_PROMPT,
    DEFAULT_BOUNDARY_DETECTION_PROMPT,
    DEFAULT_OCR_OUTPUT_TAG,
    OLMOCR_V4_YAML_PROMPT,
)
from churro_ocr.providers import (
    AzureDocumentIntelligenceOptions,
    AzurePageDetector,
    HuggingFaceOptions,
    LiteLLMTransportConfig,
    LLMPageDetector,
    MistralOptions,
    OCRBackendSpec,
    OpenAICompatibleOptions,
    build_ocr_backend,
    locate_text_block_bbox_with_llm,
    resolve_ocr_profile,
)
from churro_ocr.providers.hf import HuggingFaceVisionOCRBackend
from churro_ocr.providers.ocr import (
    AzureDocumentIntelligenceOCRBackend,
    LiteLLMVisionOCRBackend,
    MistralOCRBackend,
    OpenAICompatibleOCRBackend,
)
from churro_ocr.providers.page_detection import (
    _normalize_azure_page_polygon,
    _PageBox,
    locate_text_block_bbox_with_llm_sync,
)
from churro_ocr.providers.specs import DEFAULT_OCR_MAX_TOKENS
from churro_ocr.templates import (
    CHANDRA_OCR_2_MODEL_ID,
    CHANDRA_OCR_2_OCR_TEMPLATE,
    DEFAULT_OCR_TEMPLATE,
    OLMOCR_2_7B_1025_FP8_MODEL_ID,
    OLMOCR_2_7B_1025_MODEL_ID,
    OLMOCR_2_7B_1025_OCR_TEMPLATE,
)


def _extract_user_text_parts(messages: list[dict[str, Any]]) -> list[str]:
    user_messages = [message for message in messages if message.get("role") == "user"]
    assert len(user_messages) == 1
    content = cast("list[dict[str, Any]]", user_messages[0]["content"])
    return [
        cast("str", item["text"])
        for item in content
        if item.get("type") == "text" and isinstance(item.get("text"), str)
    ]


@pytest.mark.asyncio
async def test_litellm_ocr_backend_uses_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (10, 10), color="white")
    page = DocumentPage(page_index=0, image=image, source_index=0)
    prompt_logs: list[str] = []

    class FakeLogger:
        def debug(self, message: str, *args: object) -> None:
            prompt_logs.append(message % args if args else message)

    def _fake_prepare_messages(
        conversation: list[dict[str, object]],
        *,
        image_detail: str | None,
    ) -> list[dict[str, object]]:
        content = cast("list[dict[str, object]]", conversation[1]["content"])
        assert content[0]["image"] == image
        assert image_detail == "high"
        return [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]

    async def _fake_complete_text(self, **_: object) -> str:  # noqa: ANN001
        return "transcribed text"

    monkeypatch.setattr(
        "churro_ocr._internal.litellm._prepare_messages_from_conversation", _fake_prepare_messages
    )
    monkeypatch.setattr(
        "churro_ocr._internal.litellm.LiteLLMTransport.complete_text",
        _fake_complete_text,
    )
    monkeypatch.setattr("churro_ocr._internal.prompt_logging.logger", FakeLogger())

    backend = cast(
        "LiteLLMVisionOCRBackend",
        build_ocr_backend(OCRBackendSpec(provider="litellm", model="gpt-4.1-mini")),
    )
    result = await backend.ocr(page)

    assert result.text == "transcribed text"
    assert result.model_name == "gpt-4.1-mini"
    assert backend.transport.config.completion_kwargs == {"max_tokens": DEFAULT_OCR_MAX_TOKENS}
    assert len(prompt_logs) == 1
    assert "First OCR prompt payload for litellm" in prompt_logs[0]


@pytest.mark.asyncio
async def test_litellm_ocr_backend_logs_prompt_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_logs: list[str] = []

    class FakeLogger:
        def debug(self, message: str, *args: object) -> None:
            prompt_logs.append(message % args if args else message)

    async def _fake_complete_text(self, **_: object) -> str:  # noqa: ANN001
        return "ok"

    monkeypatch.setattr(
        "churro_ocr._internal.litellm._prepare_messages_from_conversation",
        lambda *_args, **_kwargs: [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abcdefghijklmnopqrstuvwxyz"},
                    },
                    {"type": "text", "text": "prompt"},
                ],
            }
        ],
    )
    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)
    monkeypatch.setattr("churro_ocr._internal.prompt_logging.logger", FakeLogger())

    backend = build_ocr_backend(OCRBackendSpec(provider="litellm", model="gpt-4.1-mini"))
    page = DocumentPage.from_image(Image.new("RGB", (10, 10), color="white"))

    assert (await backend.ocr(page)).text == "ok"
    assert (await backend.ocr(page)).text == "ok"
    assert len(prompt_logs) == 1
    assert "data:image/png;base64,abcdefghijklmnopqrstuvwxyz" in prompt_logs[0]


@pytest.mark.asyncio
async def test_litellm_transport_tracks_total_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_acompletion(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="tracked text"))],
            _hidden_params={"response_cost": 0.125},
        )

    fake_module = ModuleType("litellm")
    cast("Any", fake_module).acompletion = _fake_acompletion
    cast("Any", fake_module).completion_cost = lambda **_: 999.0

    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setattr("churro_ocr._internal.litellm._ensure_initialized", lambda: None)

    transport = LiteLLMTransport()
    result = await transport.complete_text(
        model="example/model",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    )

    assert result == "tracked text"
    assert transport.last_cost_usd == pytest.approx(0.125)
    assert transport.total_cost_usd == pytest.approx(0.125)
    assert transport.request_count == 1
    assert transport.untracked_request_count == 0


@pytest.mark.asyncio
async def test_litellm_ocr_backend_strips_default_output_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_complete_text(self, **_: object) -> str:  # noqa: ANN001
        return f"<{DEFAULT_OCR_OUTPUT_TAG}>\ntranscribed text\n</{DEFAULT_OCR_OUTPUT_TAG}>"

    monkeypatch.setattr(
        "churro_ocr._internal.litellm._prepare_messages_from_conversation",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    backend = build_ocr_backend(
        OCRBackendSpec(
            provider="litellm",
            model="gpt-4.1-mini",
            profile=resolve_ocr_profile(model_id="gpt-4.1-mini"),
        )
    )
    result = await backend.ocr(DocumentPage.from_image(Image.new("RGB", (10, 10), color="white")))

    assert result.text == "transcribed text"


@pytest.mark.asyncio
async def test_openai_compatible_backend_reports_display_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_complete_text(self, **_: object) -> str:  # noqa: ANN001
        return "openai compatible text"

    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    backend = cast(
        "LiteLLMVisionOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model="local-model",
                transport=LiteLLMTransportConfig(
                    api_base="http://127.0.0.1:8000/v1",
                    api_key="dummy",
                ),
                options=OpenAICompatibleOptions(),
            )
        ),
    )
    result = await backend.ocr(DocumentPage.from_image(Image.new("RGB", (10, 10), color="white")))

    assert result.provider_name == "openai-compatible"
    assert result.model_name == "local-model"
    assert backend.transport.config.completion_kwargs == {"max_tokens": DEFAULT_OCR_MAX_TOKENS}


@pytest.mark.asyncio
async def test_azure_ocr_backend_reuses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"client_inits": 0, "requests": 0}
    image = Image.new("RGB", (10, 10), color="white")
    encoded = base64.b64encode(b"image-bytes").decode("ascii")

    class FakePoller:
        async def result(self) -> SimpleNamespace:
            return SimpleNamespace(content="azure text")

    class FakeClient:
        def __init__(self, *, endpoint: str, credential: Any) -> None:
            calls["client_inits"] += 1
            assert endpoint == "https://example.test"
            assert credential.key == "secret"

        async def begin_analyze_document(
            self,
            *,
            model_id: str,
            body: Any,
            content_type: str,
        ) -> FakePoller:
            calls["requests"] += 1
            assert model_id == "prebuilt-layout"
            assert body.read() == b"image-bytes"
            assert content_type == "application/octet-stream"
            return FakePoller()

    class FakeAzureKeyCredential:
        def __init__(self, key: str) -> None:
            self.key = key

    azure_document_module = ModuleType("azure.ai.documentintelligence.aio")
    cast("Any", azure_document_module).DocumentIntelligenceClient = FakeClient
    azure_credentials_module = ModuleType("azure.core.credentials")
    cast("Any", azure_credentials_module).AzureKeyCredential = FakeAzureKeyCredential
    monkeypatch.setitem(sys.modules, "azure.ai.documentintelligence.aio", azure_document_module)
    monkeypatch.setitem(sys.modules, "azure.core.credentials", azure_credentials_module)
    monkeypatch.setattr(
        "churro_ocr.providers.ocr.image_to_base64",
        lambda actual_image, format_name: (
            (
                "image/jpeg",
                encoded,
            )
            if actual_image.size == image.size and actual_image.mode == "RGB" and format_name == "JPEG"
            else ("", "")
        ),
    )

    backend = cast(
        "AzureDocumentIntelligenceOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="azure",
                options=AzureDocumentIntelligenceOptions(
                    endpoint="https://example.test",
                    api_key="secret",
                ),
            )
        ),
    )
    page = DocumentPage(page_index=0, image=image, source_index=0)

    first = await backend.ocr(page)
    second = await backend.ocr(page)

    assert first.text == "azure text"
    assert second.text == "azure text"
    assert calls == {"client_inits": 1, "requests": 2}


@pytest.mark.asyncio
async def test_mistral_ocr_backend_reuses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"client_inits": 0, "requests": 0}
    image = Image.new("RGB", (10, 10), color="white")

    class FakeOCRNamespace:
        async def process_async(self, *, model: str, document: dict[str, str]) -> SimpleNamespace:
            calls["requests"] += 1
            assert model == "mistral-ocr-latest"
            assert document == {
                "type": "image_url",
                "image_url": "data:image/jpeg;base64,encoded-image",
            }
            return SimpleNamespace(pages=[SimpleNamespace(markdown="mistral text")])

    class FakeMistralClient:
        def __init__(self, *, api_key: str) -> None:
            calls["client_inits"] += 1
            assert api_key == "secret"
            self.ocr = FakeOCRNamespace()

    mistral_module = ModuleType("mistralai")
    cast("Any", mistral_module).Mistral = FakeMistralClient
    monkeypatch.setitem(sys.modules, "mistralai", mistral_module)
    monkeypatch.setattr(
        "churro_ocr.providers.ocr.image_to_base64",
        lambda actual_image, format_name: (
            (
                "image/jpeg",
                "encoded-image",
            )
            if actual_image.size == image.size and actual_image.mode == "RGB" and format_name == "JPEG"
            else ("", "")
        ),
    )

    backend = cast(
        "MistralOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="mistral",
                options=MistralOptions(api_key="secret"),
            )
        ),
    )
    page = DocumentPage(page_index=0, image=image, source_index=0)

    first = await backend.ocr(page)
    second = await backend.ocr(page)

    assert first.text == "mistral text"
    assert second.text == "mistral text"
    assert calls == {"client_inits": 1, "requests": 2}


def test_build_ocr_backend_uses_olmocr_profile_defaults_for_openai_compatible() -> None:
    backend = cast(
        "OpenAICompatibleOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model=OLMOCR_2_7B_1025_MODEL_ID,
                transport=LiteLLMTransportConfig(api_base="http://127.0.0.1:8000/v1"),
            )
        ),
    )

    assert backend.template == OLMOCR_2_7B_1025_OCR_TEMPLATE
    assert backend.model_name == "olmOCR-2-7B-1025"
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 8_000,
        "temperature": 0.1,
    }
    assert backend.image_preprocessor(Image.new("RGB", (5_000, 3_000), color="white")).size == (1_288, 772)


def test_build_ocr_backend_uses_chandra_profile_defaults_for_openai_compatible() -> None:
    backend = cast(
        "OpenAICompatibleOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model=CHANDRA_OCR_2_MODEL_ID,
                transport=LiteLLMTransportConfig(api_base="http://127.0.0.1:8000/v1"),
            )
        ),
    )

    assert backend.template == CHANDRA_OCR_2_OCR_TEMPLATE
    assert backend.model_name == "chandra-ocr-2"
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 12_384,
        "temperature": 0.0,
        "top_p": 0.1,
    }
    assert backend.image_preprocessor(Image.new("RGB", (5_000, 3_000), color="white")).size == (3_248, 1_932)


def test_build_ocr_backend_resolves_olmocr_fp8_profile_defaults_for_openai_compatible() -> None:
    backend = cast(
        "OpenAICompatibleOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model=OLMOCR_2_7B_1025_FP8_MODEL_ID,
                transport=LiteLLMTransportConfig(api_base="http://127.0.0.1:8000/v1"),
            )
        ),
    )

    assert backend.template == OLMOCR_2_7B_1025_OCR_TEMPLATE
    assert backend.model_name == "olmOCR-2-7B-1025-FP8"
    assert backend.transport.config.completion_kwargs == {
        "max_tokens": 8_000,
        "temperature": 0.1,
    }


@pytest.mark.asyncio
async def test_openai_compatible_backend_uses_olmocr_prompt_and_plain_text_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_prepare_messages_from_conversation(
        self: LiteLLMTransport,
        conversation: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        captured["conversation"] = conversation
        captured["completion_kwargs"] = dict(self.config.completion_kwargs)
        return [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]

    async def _fake_complete_text(
        self: LiteLLMTransport,
        *,
        model: str,
        messages: list[dict[str, object]],
        timeout_seconds: int = 600,
        output_json: bool = False,
    ) -> str:
        captured["model"] = model
        captured["messages"] = messages
        captured["timeout_seconds"] = timeout_seconds
        captured["output_json"] = output_json
        captured["completion_kwargs"] = dict(self.config.completion_kwargs)
        return (
            "---\n"
            "primary_language: en\n"
            "is_rotation_valid: true\n"
            "rotation_correction: 0\n"
            "is_table: true\n"
            "is_diagram: false\n"
            "---\n"
            "# Ledger\n\n"
            "<table><tr><th>Year</th><th>Value</th></tr>"
            "<tr><td>1900</td><td>42</td></tr></table>\n\n"
            "Paragraph with [note](https://example.test).\n"
            "![Figure alt text](page_0_0_100_100.png)\n"
        )

    monkeypatch.setattr(
        LiteLLMTransport,
        "prepare_messages_from_conversation",
        _fake_prepare_messages_from_conversation,
    )
    monkeypatch.setattr(LiteLLMTransport, "complete_text", _fake_complete_text)

    backend = cast(
        "OpenAICompatibleOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model=OLMOCR_2_7B_1025_MODEL_ID,
                transport=LiteLLMTransportConfig(api_base="http://127.0.0.1:8000/v1"),
            )
        ),
    )
    result = await backend.ocr(
        DocumentPage.from_image(Image.new("RGBA", (5_000, 3_000), color=(255, 255, 255, 255)))
    )

    assert result.text == "Ledger\n\nYear | Value\n1900 | 42\n\nParagraph with note."
    assert result.metadata == {
        "front_matter": {
            "primary_language": "en",
            "is_rotation_valid": True,
            "rotation_correction": 0,
            "is_table": True,
            "is_diagram": False,
        },
        "raw_markdown": (
            "# Ledger\n\n"
            "<table><tr><th>Year</th><th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table>\n\n"
            "Paragraph with [note](https://example.test).\n"
            "![Figure alt text](page_0_0_100_100.png)"
        ),
    }
    assert captured["model"] == f"openai/{OLMOCR_2_7B_1025_MODEL_ID}"
    assert captured["messages"] == [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]
    assert captured["timeout_seconds"] == 600
    assert captured["output_json"] is False
    assert captured["completion_kwargs"] == {
        "max_tokens": 8_000,
        "temperature": 0.1,
    }
    conversation = cast("list[dict[str, object]]", captured["conversation"])
    assert conversation[0]["role"] == "user"
    user_content = cast("list[dict[str, object]]", conversation[0]["content"])
    assert user_content[0] == {"type": "text", "text": OLMOCR_V4_YAML_PROMPT}
    assert user_content[1]["type"] == "image"
    prompt_image = cast("Image.Image", user_content[1]["image"])
    assert prompt_image.size == (1_288, 772)
    assert prompt_image.mode == "RGB"


@pytest.mark.asyncio
async def test_openai_compatible_backend_uses_chandra_prompt_and_plain_text_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_prepare_messages_from_conversation(
        self: LiteLLMTransport,
        conversation: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        captured["conversation"] = conversation
        captured["completion_kwargs"] = dict(self.config.completion_kwargs)
        return [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]

    async def _fake_complete_text(
        self: LiteLLMTransport,
        *,
        model: str,
        messages: list[dict[str, object]],
        timeout_seconds: int = 600,
        output_json: bool = False,
    ) -> str:
        captured["model"] = model
        captured["messages"] = messages
        captured["timeout_seconds"] = timeout_seconds
        captured["output_json"] = output_json
        captured["completion_kwargs"] = dict(self.config.completion_kwargs)
        return (
            '<div data-bbox="0 0 1000 100" data-label="Section-Header"><h1>Ledger</h1></div>\n'
            '<div data-bbox="0 100 1000 300" data-label="Text"><p>Paragraph with <a '
            'href="https://example.test">note</a>.</p></div>\n'
            '<div data-bbox="0 300 1000 500" data-label="Table"><table><tr><th>Year</th>'
            "<th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table></div>"
        )

    monkeypatch.setattr(
        LiteLLMTransport,
        "prepare_messages_from_conversation",
        _fake_prepare_messages_from_conversation,
    )
    monkeypatch.setattr(LiteLLMTransport, "complete_text", _fake_complete_text)

    backend = cast(
        "OpenAICompatibleOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model=CHANDRA_OCR_2_MODEL_ID,
                transport=LiteLLMTransportConfig(api_base="http://127.0.0.1:8000/v1"),
            )
        ),
    )
    result = await backend.ocr(
        DocumentPage.from_image(Image.new("RGBA", (5_000, 3_000), color=(255, 255, 255, 255)))
    )

    assert result.text == "Ledger\n\nParagraph with note.\n\nYear | Value\n1900 | 42"
    assert result.metadata == {
        "raw_html": (
            '<div data-bbox="0 0 1000 100" data-label="Section-Header"><h1>Ledger</h1></div>\n'
            '<div data-bbox="0 100 1000 300" data-label="Text"><p>Paragraph with <a '
            'href="https://example.test">note</a>.</p></div>\n'
            '<div data-bbox="0 300 1000 500" data-label="Table"><table><tr><th>Year</th>'
            "<th>Value</th></tr><tr><td>1900</td><td>42</td></tr></table></div>"
        ),
    }
    assert captured["model"] == f"openai/{CHANDRA_OCR_2_MODEL_ID}"
    assert captured["messages"] == [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]
    assert captured["timeout_seconds"] == 600
    assert captured["output_json"] is False
    assert captured["completion_kwargs"] == {
        "max_tokens": 12_384,
        "temperature": 0.0,
        "top_p": 0.1,
    }
    conversation = cast("list[dict[str, object]]", captured["conversation"])
    assert conversation[0]["role"] == "user"
    user_content = cast("list[dict[str, object]]", conversation[0]["content"])
    assert user_content[0]["type"] == "image"
    prompt_image = cast("Image.Image", user_content[0]["image"])
    assert prompt_image.size == (3_248, 1_932)
    assert prompt_image.mode == "RGB"
    assert user_content[1] == {"type": "text", "text": CHANDRA_OCR_LAYOUT_PROMPT}


@pytest.mark.asyncio
async def test_llm_page_detector_uses_prompt_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_complete_text(self, **_: object) -> str:  # noqa: ANN001
        return json.dumps(
            {
                "pages": [
                    {"page_index": 0, "left": 0, "top": 0, "right": 400, "bottom": 750},
                    {"page_index": 1, "left": 450, "top": 0, "right": 800, "bottom": 750},
                ]
            }
        )

    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    detector = LLMPageDetector(model="gemini-2.5-flash")
    candidates = await detector.detect(Image.new("RGB", (100, 40), color="white"))

    assert len(candidates) == 2
    assert candidates[0].image is None
    assert candidates[0].bbox is not None
    assert candidates[1].metadata["detector"] == "llm"


@pytest.mark.asyncio
async def test_llm_page_detector_rejects_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_complete_text(self, **_: object) -> str:  # noqa: ANN001
        return '{"pages":"oops"}'

    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    detector = LLMPageDetector(model="gemini-2.5-flash")
    with pytest.raises(ProviderError, match="`pages` list"):
        await detector.detect(Image.new("RGB", (100, 40), color="white"))


@pytest.mark.asyncio
async def test_llm_page_detector_returns_full_image_candidate_when_initial_detection_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_complete_page_boxes(**_: object) -> list[object]:
        return []

    monkeypatch.setattr(
        "churro_ocr.providers.page_detection._complete_page_boxes",
        _fake_complete_page_boxes,
    )

    image = Image.new("RGB", (100, 40), color="white")
    detector = LLMPageDetector(model="gemini-2.5-flash")
    candidates = await detector.detect(image)

    assert len(candidates) == 1
    assert candidates[0].bbox == (0.0, 0.0, 100.0, 40.0)
    assert candidates[0].polygon == ()


@pytest.mark.asyncio
async def test_llm_page_detector_applies_iterative_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            json.dumps(
                {
                    "pages": [
                        {
                            "page_index": 1,
                            "left": 200,
                            "top": 150,
                            "right": 800,
                            "bottom": 850,
                        }
                    ]
                }
            ),
            json.dumps({"page_index": 1, "edge": "left", "action": "expand", "amount": 300}),
            json.dumps({"page_index": 1, "edge": "top", "action": "no_change", "amount": 0}),
            json.dumps({"page_index": 1, "edge": "right", "action": "no_change", "amount": 0}),
            json.dumps({"page_index": 1, "edge": "bottom", "action": "no_change", "amount": 0}),
        ]
    )
    prompts: list[str | None] = []

    async def _fake_complete_text(self, **kwargs: object) -> str:  # noqa: ANN001
        messages = cast("list[dict[str, Any]]", kwargs["messages"])
        user_text_parts = _extract_user_text_parts(messages)
        assert len(messages) == 1
        assert user_text_parts
        prompt = user_text_parts[0]
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    detector = LLMPageDetector(model="gemini-2.5-flash", max_review_rounds=1)
    candidates = await detector.detect(Image.new("RGB", (100, 100), color="white"))

    assert len(candidates) == 1
    assert candidates[0].bbox is not None
    left, _, right, _ = candidates[0].bbox
    assert left < 20.0
    assert right > 70.0
    assert prompts[0] == DEFAULT_BOUNDARY_DETECTION_PROMPT.strip()
    assert prompts[1] is not None
    assert prompts[1].startswith("You are an expert reviewer of a document page boundary annotation.")


@pytest.mark.asyncio
async def test_locate_text_block_bbox_with_llm_uses_block_prompt_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str | None] = []

    async def _fake_complete_text(self, **kwargs: object) -> str:  # noqa: ANN001
        messages = cast("list[dict[str, Any]]", kwargs["messages"])
        user_text_parts = _extract_user_text_parts(messages)
        assert len(messages) == 1
        assert user_text_parts
        prompt = user_text_parts[0]
        prompts.append(prompt)
        return json.dumps(
            {
                "block_found": True,
                "block": {"left": 180, "top": 260, "right": 860, "bottom": 720},
            }
        )

    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    bbox = await locate_text_block_bbox_with_llm(
        Image.new("RGB", (120, 80), color="white"),
        "Anno domini 1647\nActum Pragae",
        block_tag="Paragraph",
        model="gemini-2.5-flash",
    )

    assert bbox is not None
    left, top, right, bottom = bbox
    assert left < right
    assert top < bottom
    assert prompts[0] is not None
    assert prompts[0].startswith("You are an expert reviewer of historical document layout.")
    assert "Target block tag" in prompts[0]
    assert "Paragraph" in prompts[0]


@pytest.mark.asyncio
async def test_locate_text_block_bbox_with_llm_accepts_shared_transport_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_acompletion(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "block_found": True,
                                "block": {"left": 180, "top": 260, "right": 860, "bottom": 720},
                            }
                        )
                    )
                )
            ],
            _hidden_params={"response_cost": 0.25},
        )

    fake_module = ModuleType("litellm")
    cast("Any", fake_module).acompletion = _fake_acompletion
    cast("Any", fake_module).completion_cost = lambda **_: 999.0

    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setattr("churro_ocr._internal.litellm._ensure_initialized", lambda: None)

    transport = LiteLLMTransport()
    bbox = await locate_text_block_bbox_with_llm(
        Image.new("RGB", (120, 80), color="white"),
        "Anno domini 1647\nActum Pragae",
        block_tag="Paragraph",
        model="example/model",
        transport=transport,
    )

    assert bbox is not None
    assert transport.total_cost_usd == pytest.approx(0.25)
    assert transport.request_count == 1
    assert transport.untracked_request_count == 0


@pytest.mark.asyncio
async def test_locate_text_block_bbox_with_llm_returns_none_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_complete_text(self, **_: object) -> str:  # noqa: ANN001
        return json.dumps({"block_found": False, "block": None})

    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    bbox = await locate_text_block_bbox_with_llm(
        Image.new("RGB", (120, 80), color="white"),
        "missing paragraph",
        block_tag="Paragraph",
        model="gemini-2.5-flash",
    )

    assert bbox is None


@pytest.mark.asyncio
async def test_locate_text_block_bbox_with_llm_rejects_blank_text_and_tag() -> None:
    image = Image.new("RGB", (120, 80), color="white")

    with pytest.raises(ValueError, match="block_text must not be blank"):
        await locate_text_block_bbox_with_llm(
            image,
            "   ",
            block_tag="Paragraph",
            model="gemini-2.5-flash",
        )

    with pytest.raises(ValueError, match="block_tag must not be blank"):
        await locate_text_block_bbox_with_llm(
            image,
            "Anno domini",
            block_tag="   ",
            model="gemini-2.5-flash",
        )


@pytest.mark.asyncio
async def test_locate_text_block_bbox_with_llm_returns_none_when_review_pipeline_discards_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_complete_text_block_box(**_: object) -> _PageBox:
        return _PageBox.from_json({"page_index": 1, "left": 200, "top": 200, "right": 800, "bottom": 800})

    async def _fake_run_review_pipeline(**_: object) -> list[_PageBox]:
        return []

    monkeypatch.setattr(
        "churro_ocr.providers.page_detection._complete_text_block_box",
        _fake_complete_text_block_box,
    )
    monkeypatch.setattr(
        "churro_ocr.providers.page_detection._run_review_pipeline",
        _fake_run_review_pipeline,
    )

    bbox = await locate_text_block_bbox_with_llm(
        Image.new("RGB", (100, 100), color="white"),
        "Et fuit lux",
        block_tag="Paragraph",
        model="gemini-2.5-flash",
        max_review_rounds=1,
    )

    assert bbox is None


def test_locate_text_block_bbox_with_llm_sync_wraps_async_locator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_locate(
        image: Image.Image,
        block_text: str,
        *,
        block_tag: str,
        model: str,
        transport: object | None = None,
        max_review_rounds: int = 0,
    ) -> tuple[float, float, float, float] | None:
        del image, block_text, block_tag, model, transport, max_review_rounds
        return (1.0, 2.0, 3.0, 4.0)

    monkeypatch.setattr(
        "churro_ocr.providers.page_detection.locate_text_block_bbox_with_llm",
        _fake_locate,
    )

    assert locate_text_block_bbox_with_llm_sync(
        Image.new("RGB", (20, 20), color="white"),
        "Anno domini",
        block_tag="Paragraph",
        model="example/model",
    ) == (1.0, 2.0, 3.0, 4.0)


@pytest.mark.asyncio
async def test_locate_text_block_bbox_with_llm_applies_iterative_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            json.dumps(
                {
                    "block_found": True,
                    "block": {
                        "left": 220,
                        "top": 300,
                        "right": 780,
                        "bottom": 760,
                    },
                }
            ),
            json.dumps({"edge": "left", "action": "expand", "amount": 250}),
            json.dumps({"edge": "top", "action": "no_change", "amount": 0}),
            json.dumps({"edge": "right", "action": "no_change", "amount": 0}),
            json.dumps({"edge": "bottom", "action": "no_change", "amount": 0}),
        ]
    )
    prompts: list[str | None] = []

    async def _fake_complete_text(self, **kwargs: object) -> str:  # noqa: ANN001
        messages = cast("list[dict[str, Any]]", kwargs["messages"])
        user_text_parts = _extract_user_text_parts(messages)
        assert len(messages) == 1
        assert user_text_parts
        prompt = user_text_parts[0]
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("churro_ocr._internal.litellm.LiteLLMTransport.complete_text", _fake_complete_text)

    bbox = await locate_text_block_bbox_with_llm(
        Image.new("RGB", (100, 100), color="white"),
        "Et fuit lux\nIn principio",
        block_tag="Paragraph",
        model="gemini-2.5-flash",
        max_review_rounds=1,
    )

    assert bbox is not None
    left, _, right, _ = bbox
    assert left < 15.0
    assert right > 65.0
    assert prompts[0] is not None
    assert "Target block tag" in prompts[0]
    assert prompts[1] is not None
    assert prompts[1].startswith("You are an expert reviewer of a content-block bounding box annotation.")


def test_azure_page_detector_normalizes_page_polygon() -> None:
    page = SimpleNamespace(
        polygon=[0.0, 0.0, 50.0, 0.0, 50.0, 100.0, 0.0, 100.0],
        width=50.0,
        height=100.0,
    )

    polygon = _normalize_azure_page_polygon(page, image=Image.new("RGB", (200, 400), color="white"))

    assert polygon == ((0.0, 0.0), (200.0, 0.0), (200.0, 400.0), (0.0, 400.0))


@pytest.mark.asyncio
async def test_azure_page_detector_detects_pages_and_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"client_inits": 0, "requests": 0, "closes": 0}

    class FakePoller:
        async def result(self) -> SimpleNamespace:
            return SimpleNamespace(
                pages=[
                    SimpleNamespace(
                        polygon=[0.0, 0.0, 50.0, 0.0, 50.0, 100.0, 0.0, 100.0],
                        width=50.0,
                        height=100.0,
                        page_number=7,
                        unit="pixel",
                        angle=12.5,
                    ),
                    SimpleNamespace(
                        polygon=None,
                    ),
                ]
            )

    class FakeClient:
        def __init__(self, *, endpoint: str, credential: Any) -> None:
            calls["client_inits"] += 1
            assert endpoint == "https://example.test"
            assert credential.key == "secret"

        async def begin_analyze_document(
            self,
            *,
            model_id: str,
            body: Any,
            content_type: str,
        ) -> FakePoller:
            calls["requests"] += 1
            assert model_id == "prebuilt-layout"
            assert content_type == "application/octet-stream"
            assert body.read()
            return FakePoller()

        async def close(self) -> None:
            calls["closes"] += 1

    class FakeAzureKeyCredential:
        def __init__(self, key: str) -> None:
            self.key = key

    azure_document_module = ModuleType("azure.ai.documentintelligence.aio")
    cast(Any, azure_document_module).DocumentIntelligenceClient = FakeClient
    azure_credentials_module = ModuleType("azure.core.credentials")
    cast(Any, azure_credentials_module).AzureKeyCredential = FakeAzureKeyCredential
    monkeypatch.setitem(sys.modules, "azure.ai.documentintelligence.aio", azure_document_module)
    monkeypatch.setitem(sys.modules, "azure.core.credentials", azure_credentials_module)

    detector = AzurePageDetector(endpoint="https://example.test", api_key="secret")
    candidates = await detector.detect(Image.new("RGB", (200, 400), color="white"))

    assert len(candidates) == 2
    assert candidates[0].polygon == ((0.0, 0.0), (200.0, 0.0), (200.0, 400.0), (0.0, 400.0))
    assert candidates[0].bbox == (0.0, 0.0, 200.0, 400.0)
    assert candidates[0].metadata == {
        "page_index": 0,
        "page_number": 7,
        "detector": "azure",
        "unit": "pixel",
        "angle": 12.5,
    }
    assert candidates[1].bbox is None
    assert candidates[1].polygon == ()
    assert candidates[1].metadata == {
        "page_index": 1,
        "page_number": 2,
        "detector": "azure",
    }
    assert calls == {"client_inits": 1, "requests": 1, "closes": 1}


@pytest.mark.asyncio
async def test_azure_page_detector_returns_full_image_when_service_returns_no_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"closes": 0}

    class FakePoller:
        async def result(self) -> SimpleNamespace:
            return SimpleNamespace(pages=[])

    class FakeClient:
        def __init__(self, *, endpoint: str, credential: Any) -> None:
            del endpoint, credential

        async def begin_analyze_document(
            self,
            *,
            model_id: str,
            body: Any,
            content_type: str,
        ) -> FakePoller:
            del model_id, body, content_type
            return FakePoller()

        async def close(self) -> None:
            calls["closes"] += 1

    class FakeAzureKeyCredential:
        def __init__(self, key: str) -> None:
            self.key = key

    azure_document_module = ModuleType("azure.ai.documentintelligence.aio")
    cast(Any, azure_document_module).DocumentIntelligenceClient = FakeClient
    azure_credentials_module = ModuleType("azure.core.credentials")
    cast(Any, azure_credentials_module).AzureKeyCredential = FakeAzureKeyCredential
    monkeypatch.setitem(sys.modules, "azure.ai.documentintelligence.aio", azure_document_module)
    monkeypatch.setitem(sys.modules, "azure.core.credentials", azure_credentials_module)

    image = Image.new("RGB", (120, 80), color="white")
    detector = AzurePageDetector(endpoint="https://example.test", api_key="secret")
    candidates = await detector.detect(image)

    assert len(candidates) == 1
    assert candidates[0].bbox == (0.0, 0.0, 120.0, 80.0)
    assert candidates[0].polygon == ()
    assert calls["closes"] == 1


def test_azure_page_detector_type_is_public() -> None:
    detector = AzurePageDetector(endpoint="https://example.test", api_key="secret")
    assert detector.model_id == "prebuilt-layout"


def test_build_ocr_backend_resolves_profile_defaults() -> None:
    backend = cast(
        "OpenAICompatibleOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model="stanford-oval/churro-3B",
                transport=LiteLLMTransportConfig(api_base="http://127.0.0.1:8000/v1"),
            )
        ),
    )

    assert backend.model_name == "churro-3B"
    assert backend.template != DEFAULT_OCR_TEMPLATE


def test_build_ocr_backend_uses_generic_defaults_for_qwen_model() -> None:
    backend = cast(
        "OpenAICompatibleOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="openai-compatible",
                model="Qwen/Qwen3.5-0.8B",
                transport=LiteLLMTransportConfig(api_base="http://127.0.0.1:8000/v1"),
            )
        ),
    )

    assert backend.model_name == "Qwen/Qwen3.5-0.8B"
    assert backend.template == DEFAULT_OCR_TEMPLATE
    assert backend.transport.config.completion_kwargs == {"max_tokens": DEFAULT_OCR_MAX_TOKENS}


def test_build_ocr_backend_merges_hf_overrides_with_profile_defaults() -> None:
    backend = cast(
        "HuggingFaceVisionOCRBackend",
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model="kristaller486/dots.ocr-1.5",
                options=HuggingFaceOptions(
                    model_kwargs={"torch_dtype": "auto"},
                    generation_kwargs={"temperature": 0.0},
                ),
            )
        ),
    )

    assert backend.trust_remote_code is True
    assert backend.processor_kwargs == {}
    assert backend.model_kwargs["torch_dtype"] == "auto"
    assert backend.generation_kwargs == {
        "max_new_tokens": DEFAULT_OCR_MAX_TOKENS,
        "temperature": 0.0,
    }

"""Tests for vLLM docker helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from churro.config.settings import (
    AzureDocumentIntelligenceSettings,
    AzureOpenAISettings,
    ChurroSettings,
    LocalRuntimeSettings,
    TokenSettings,
    VertexAISettings,
)
from churro.utils.docker import vllm as vllm_module


def _make_settings(*, port: int) -> ChurroSettings:
    return ChurroSettings(
        env_file=Path("dummy.env"),
        azure_openai=AzureOpenAISettings(api_base=None, api_version=None, api_key=None),
        azure_document_intelligence=AzureDocumentIntelligenceSettings(
            endpoint=None,
            api_key=None,
        ),
        vertex_ai=VertexAISettings(
            project_id=None,
            location="us-east5",
            document_ai_location="us",
            ocr_processor_id=None,
            ocr_processor_version=None,
        ),
        google_cloud_project=None,
        local=LocalRuntimeSettings(vllm_port=port, huggingface_token=None),
        tokens=TokenSettings(openai=None, mistral=None),
    )


def test_maybe_start_vllm_uses_injected_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected settings should control the exposed host port."""
    test_engine = "unit-test-engine"
    model_info: dict[str, Any] = {
        "provider_model": "vllm/org-model",
        "max_completion_tokens": 1024,
        "hf_repo": "org/model",
    }
    monkeypatch.setitem(vllm_module.MODEL_MAP, test_engine, [model_info])

    captured: dict[str, Any] = {}

    def fake_start_vllm_server(**kwargs: object) -> str:
        captured.update(kwargs)
        return "container"

    monkeypatch.setattr(vllm_module, "start_vllm_server", fake_start_vllm_server)

    settings = _make_settings(port=4321)
    container = vllm_module.maybe_start_vllm_server_for_engine(
        engine=test_engine,
        system="llm",
        settings=settings,
    )

    assert container == "container"
    assert captured["host_port"] == 4321

"""Tests for the LLM model registry configuration."""

from __future__ import annotations

from churro.utils.llm.config import LLMSettings, get_settings
from churro.utils.llm.models import reload_model_map


def test_reload_model_map_applies_custom_settings() -> None:
    """Custom settings should flow into vertex and local vLLM entries."""
    original_settings = get_settings()
    custom_settings = LLMSettings(
        azure_api_base=original_settings.azure_api_base,
        azure_api_version=original_settings.azure_api_version,
        azure_openai_api_key=original_settings.azure_openai_api_key,
        local_vllm_port=4321,
        vertex_ai_location="us-test1",
    )
    assert custom_settings.vertex_ai_location == "us-test1"
    assert custom_settings.local_base_url == "http://localhost:4321/v1"
    try:
        new_map = reload_model_map(custom_settings)

        gemini_entry = new_map["gemini-2.5-flash-noreasoning"][0]
        gemini_params = gemini_entry.get("static_params") or {}
        assert gemini_params.get("vertex_location") == custom_settings.vertex_ai_location, (
            "Gemini vertex location should reflect injected settings"
        )

        churro_entry = new_map["churro"][0]
        churro_params = churro_entry.get("static_params") or {}
        assert churro_params.get("api_base") == custom_settings.local_base_url, (
            "Local vLLM endpoints should respect injected port"
        )
    finally:
        reload_model_map(original_settings)

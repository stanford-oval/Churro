"""Tests for the centralised configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from churro.config.settings import get_settings


def _write_env(path: Path, content: str) -> None:
    lines = [line.rstrip() for line in content.strip().splitlines()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_env_file_values_are_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure values from a dedicated env file are parsed into the snapshot."""
    # Remove ambient environment variables that could override the file.
    for key in (
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
        "AZURE_OPENAI_API_KEY",
        "LOCAL_VLLM_PORT",
        "VERTEX_AI_LOCATION",
        "DOCUMENT_VERTEX_AI_LOCATION",
    ):
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / "test.env"
    _write_env(
        env_file,
        """
        AZURE_API_BASE=https://azure.example.com/
        AZURE_API_VERSION=2025-04-01-preview
        AZURE_OPENAI_API_KEY=abc123
        LOCAL_VLLM_PORT=12345
        VERTEX_AI_LOCATION=us-central1
        DOCUMENT_VERTEX_AI_LOCATION=us
        """,
    )
    # Clear any existing snapshot for this env file.
    settings = get_settings(env_file=env_file, reload=True)

    assert settings.env_file == env_file.resolve()
    assert settings.azure_openai.api_base == "https://azure.example.com/"
    assert settings.azure_openai.api_version == "2025-04-01-preview"
    assert settings.azure_openai.api_key == "abc123"
    assert settings.local.vllm_port == 12345
    assert settings.vertex_ai.location == "us-central1"
    assert settings.vertex_ai.document_ai_location == "us"


def test_environment_variables_override_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing environment variables should take precedence over .env contents."""
    env_file = tmp_path / "override.env"
    _write_env(
        env_file,
        """
        AZURE_API_VERSION=2024-01-01-preview
        LOCAL_VLLM_PORT=23456
        """,
    )
    monkeypatch.setenv("AZURE_API_VERSION", "2026-02-02-stable")
    monkeypatch.setenv("LOCAL_VLLM_PORT", "34567")

    settings = get_settings(env_file=env_file, reload=True)

    assert settings.azure_openai.api_version == "2026-02-02-stable"
    assert settings.local.vllm_port == 34567


def test_reload_picks_up_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling get_settings with reload=True should refresh cached values."""
    env_file = tmp_path / "reload.env"
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)
    _write_env(
        env_file,
        """
        AZURE_API_VERSION=2025-01-01
        """,
    )
    settings = get_settings(env_file=env_file, reload=True)
    assert settings.azure_openai.api_version == "2025-01-01"

    monkeypatch.setenv("AZURE_API_VERSION", "2030-05-05")
    updated = get_settings(env_file=env_file, reload=True)
    assert updated.azure_openai.api_version == "2030-05-05"

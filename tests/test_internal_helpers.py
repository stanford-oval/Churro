from __future__ import annotations

import builtins
import sys
from base64 import b64encode
from threading import Lock
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

import churro_ocr._internal.litellm as litellm_module
import churro_ocr._internal.prompt_logging as prompt_logging_module
from churro_ocr._internal import logging as logging_module
from churro_ocr._internal.image import image_to_base64, load_image
from churro_ocr._internal.litellm import LiteLLMTransport, complete_text
from churro_ocr.errors import ConfigurationError, ProviderError
from churro_ocr.providers.hf import _load_hf_causal_runtime, _load_hf_runtime
from churro_ocr.providers.specs import LiteLLMTransportConfig
from churro_ocr.providers.vllm import _load_vllm_processor_cls, _load_vllm_runtime


def _make_fake_litellm_module(*, acompletion: object, completion_cost: object | None = None) -> ModuleType:
    module = cast(Any, ModuleType("litellm"))
    module.acompletion = acompletion
    module.completion_cost = completion_cost or (lambda **_: None)
    module.turn_off_message_logging = False
    module.success_callback = ["stale"]
    module.failure_callback = ["stale"]
    module._logging = SimpleNamespace(_logged_requests=["stale"])  # noqa: SLF001
    module.drop_params = False
    module.suppress_debug_info = False
    module.set_verbose = True
    module.cache = None
    module.input_callback = []
    return module


def _patch_import_to_fail(monkeypatch: pytest.MonkeyPatch, *, failing_name: str) -> None:
    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == failing_name:
            raise ImportError(f"missing {failing_name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_load_image_rejects_missing_path(tmp_path) -> None:
    missing = tmp_path / "missing.png"

    with pytest.raises(ConfigurationError, match="Image path does not exist"):
        load_image(missing)


def test_image_to_base64_falls_back_to_png_for_unknown_formats() -> None:
    mime_type, encoded = image_to_base64(Image.new("RGB", (4, 4), color="white"), format_name="TIFF")

    assert mime_type == "image/png"
    assert encoded


def test_prepare_messages_includes_system_prompt_user_prompt_and_image_detail() -> None:
    messages = litellm_module.prepare_messages(
        system_prompt="system text",
        user_prompt="user text",
        images=[Image.new("RGB", (4, 4), color="white")],
        image_detail="low",
    )

    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "system text"}],
    }
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "image_url"
    assert messages[1]["content"][0]["image_url"]["detail"] == "low"
    assert messages[1]["content"][1] == {"type": "text", "text": "user text"}


def test_prepare_messages_from_conversation_converts_images_and_preserves_unknown_items() -> None:
    messages = litellm_module.prepare_messages_from_conversation(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (3, 3), color="white")},
                    {"type": "text", "text": "prompt"},
                    {"type": "audio", "audio": "raw"},
                ],
            }
        ],
        image_detail="high",
    )

    image_item = messages[0]["content"][0]
    assert image_item["type"] == "image_url"
    assert image_item["image_url"]["detail"] == "high"
    assert image_item["image_url"]["url"].startswith("data:image/png;base64,")
    assert messages[0]["content"][1:] == [
        {"type": "text", "text": "prompt"},
        {"type": "audio", "audio": "raw"},
    ]


@pytest.mark.asyncio
async def test_complete_text_wrapper_passes_transport_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_complete_text(
        self: LiteLLMTransport,
        *,
        model: str,
        messages: list[dict[str, object]],
        timeout_seconds: int = 600,
        output_json: bool = False,
    ) -> str:
        captured["config"] = self.config
        captured["model"] = model
        captured["messages"] = messages
        captured["timeout_seconds"] = timeout_seconds
        captured["output_json"] = output_json
        return "ok"

    monkeypatch.setattr(
        "churro_ocr._internal.litellm.LiteLLMTransport.complete_text",
        _fake_complete_text,
    )

    result = await complete_text(
        model="example/model",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        api_base="https://example.invalid/v1",
        api_key="secret",
        api_version="2025-01-01",
        timeout_seconds=42,
        output_json=True,
        completion_kwargs={"temperature": 0},
    )

    config = captured["config"]
    assert isinstance(config, LiteLLMTransportConfig)
    assert result == "ok"
    assert config.api_base == "https://example.invalid/v1"
    assert config.api_key == "secret"
    assert config.api_version == "2025-01-01"
    assert config.completion_kwargs == {"temperature": 0}
    assert captured["timeout_seconds"] == 42
    assert captured["output_json"] is True


def test_extract_response_cost_uses_completion_cost_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _make_fake_litellm_module(
        acompletion=lambda **_: None,
        completion_cost=lambda **_: 0.75,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_module)

    cost = litellm_module._extract_response_cost(model="example/model", response=SimpleNamespace())

    assert cost == pytest.approx(0.75)


def test_extract_response_cost_returns_none_for_non_numeric_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _make_fake_litellm_module(
        acompletion=lambda **_: None,
        completion_cost=lambda **_: "free",
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_module)

    cost = litellm_module._extract_response_cost(model="example/model", response=SimpleNamespace())

    assert cost is None


def test_ensure_initialized_wraps_logging_worker_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _make_fake_litellm_module(acompletion=lambda **_: None)
    worker_calls: list[object] = []
    worker = SimpleNamespace()
    worker.ensure_initialized_and_enqueue = lambda async_coroutine: worker_calls.append(async_coroutine)

    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setattr(litellm_module, "_INITIALIZED", False)
    monkeypatch.setattr(
        litellm_module,
        "import_module",
        lambda name: (
            SimpleNamespace(GLOBAL_LOGGING_WORKER=worker)
            if name == "litellm.litellm_core_utils.logging_worker"
            else __import__(name)
        ),
    )

    litellm_module._ensure_initialized()
    closable = SimpleNamespace(close=lambda: worker_calls.append("closed"))
    worker.ensure_initialized_and_enqueue(closable)

    assert worker_calls == ["closed"]
    assert fake_module.turn_off_message_logging is True
    assert fake_module.success_callback == []
    assert fake_module.failure_callback == []


def test_configure_disk_cache_enables_and_updates_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_module = _make_fake_litellm_module(acompletion=lambda **_: None)
    enable_calls: list[dict[str, object]] = []
    update_calls: list[dict[str, object]] = []

    caching_module = cast(Any, ModuleType("litellm.caching.caching"))
    caching_module.enable_cache = lambda **kwargs: enable_calls.append(kwargs)
    caching_module.update_cache = lambda **kwargs: update_calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setitem(sys.modules, "litellm.caching", ModuleType("litellm.caching"))
    monkeypatch.setitem(sys.modules, "litellm.caching.caching", caching_module)
    monkeypatch.setattr(litellm_module, "_INITIALIZED", False)
    monkeypatch.setattr(litellm_module, "_DISK_CACHE_DIR", None)

    first_cache_dir = tmp_path / "first"
    second_cache_dir = tmp_path / "second"
    litellm_module.configure_disk_cache(disk_cache_dir=first_cache_dir)

    fake_module = cast(Any, fake_module)
    fake_module.cache = object()
    fake_module.input_callback = ["cache"]
    litellm_module.configure_disk_cache(disk_cache_dir=second_cache_dir)

    assert enable_calls == [{"type": "disk", "disk_cache_dir": str(first_cache_dir.resolve())}]
    assert update_calls == [{"type": "disk", "disk_cache_dir": str(second_cache_dir.resolve())}]


@pytest.mark.asyncio
async def test_transport_complete_text_raises_provider_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_acompletion(**_: object) -> object:
        raise RuntimeError("boom")

    fake_module = _make_fake_litellm_module(acompletion=_failing_acompletion)
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setattr(litellm_module, "_INITIALIZED", False)

    transport = LiteLLMTransport()
    with pytest.raises(ProviderError, match="LiteLLM request failed for model 'example/model': boom"):
        await transport.complete_text(
            model="example/model",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )


@pytest.mark.asyncio
async def test_transport_complete_text_rejects_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _empty_acompletion(**_: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="   "))],
            _hidden_params={},
        )

    fake_module = _make_fake_litellm_module(acompletion=_empty_acompletion)
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    monkeypatch.setattr(litellm_module, "_INITIALIZED", False)

    transport = LiteLLMTransport()
    with pytest.raises(ProviderError, match="LiteLLM returned empty output for model 'example/model'"):
        await transport.complete_text(
            model="example/model",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )


def test_logger_adapter_supports_success_fallback_and_other_levels() -> None:
    records: list[tuple[str, str]] = []

    class _FakeLogger:
        def success(self, message: str) -> None:
            records.append(("success", message))

        def info(self, message: str) -> None:
            records.append(("info", message))

        def warning(self, message: str) -> None:
            records.append(("warning", message))

        def critical(self, message: str) -> None:
            records.append(("critical", message))

        def exception(self, message: str) -> None:
            records.append(("exception", message))

        def debug(self, message: str) -> None:
            records.append(("debug", message))

        def log(self, level: str, message: str) -> None:
            records.append((level, message))

    logger = logging_module._LoggerAdapter(_FakeLogger())
    logger.success("Saved %s", "result")
    logger.warning("Warn %s", "user")
    logger.critical("Critical %s", "path")
    logger.exception("Exception %s", "case")
    logger.debug("Debug %s", "value")
    logger.log("NOTICE", "Notice %s", 1)

    assert records == [
        ("success", "Saved result"),
        ("warning", "Warn user"),
        ("critical", "Critical path"),
        ("exception", "Exception case"),
        ("debug", "Debug value"),
        ("NOTICE", "Notice 1"),
    ]


def test_logger_adapter_success_falls_back_to_info_when_success_missing() -> None:
    records: list[tuple[str, str]] = []

    class _FakeLogger:
        def info(self, message: str) -> None:
            records.append(("info", message))

    logger = logging_module._LoggerAdapter(_FakeLogger())
    logger.success("Fallback %s", "message")

    assert records == [("info", "Fallback message")]


def test_log_prompt_payload_once_sanitizes_nested_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    state = {"logged": False}

    class _FakeLogger:
        def debug(self, message: str, *args: object) -> None:
            messages.append(message % args if args else message)

    monkeypatch.setattr("churro_ocr._internal.prompt_logging.logger", _FakeLogger())

    prompt_logging_module.log_prompt_payload_once(
        payload={
            "image": Image.new("RGB", (4, 4), color="white"),
            "bytes": b"payload",
            "tuple": ("keep", f"data:image/png;base64,{b64encode(b'payload').decode('ascii')}"),
        },
        provider_name="test-provider",
        has_logged=lambda: state["logged"],
        lock=Lock(),
        set_logged=lambda: state.__setitem__("logged", True),
    )
    prompt_logging_module.log_prompt_payload_once(
        payload="ignored",
        provider_name="test-provider",
        has_logged=lambda: state["logged"],
        lock=Lock(),
        set_logged=lambda: state.__setitem__("logged", True),
    )

    assert len(messages) == 1
    assert "First OCR prompt payload for test-provider" in messages[0]
    assert '"type": "image"' in messages[0]
    assert '"type": "bytes"' in messages[0]
    assert "data:image/png;base64," in messages[0]


@pytest.mark.parametrize(
    ("loader", "dependency_name"),
    [
        (_load_vllm_processor_cls, "transformers"),
        (_load_vllm_runtime, "vllm"),
        (_load_hf_runtime, "qwen_vl_utils"),
        (_load_hf_causal_runtime, "qwen_vl_utils"),
    ],
)
def test_optional_dependency_loaders_raise_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    dependency_name: str,
) -> None:
    _patch_import_to_fail(monkeypatch, failing_name=dependency_name)

    with pytest.raises(ConfigurationError):
        loader()

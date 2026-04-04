"""Minimal LiteLLM wrapper with shared transport configuration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from PIL import Image

from churro_ocr._internal.image import image_to_base64
from churro_ocr.errors import ConfigurationError, ProviderError
from churro_ocr.providers.specs import LiteLLMTransportConfig
from churro_ocr.templates import OCRConversation

_INITIALIZED = False
_DISK_CACHE_DIR: str | None = None


def _ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise ConfigurationError(
            "LiteLLM-backed providers require the 'llm' extra. "
            'Install with `pip install "churro-ocr[llm]"`.'
        ) from exc

    litellm_any = cast(Any, litellm)
    litellm_any.turn_off_message_logging = True
    litellm_any.success_callback = []
    litellm_any.failure_callback = []
    with suppress(Exception):
        litellm_any._logging._logged_requests = []
    litellm_any.drop_params = True
    litellm_any.suppress_debug_info = True
    litellm_any.set_verbose = False

    for logger_name in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
        provider_logger = logging.getLogger(logger_name)
        provider_logger.setLevel(logging.WARNING)
        provider_logger.propagate = False

    try:  # pragma: no cover - defensive against LiteLLM internal changes
        logging_worker = import_module("litellm.litellm_core_utils.logging_worker")
        global_logging_worker = getattr(logging_worker, "GLOBAL_LOGGING_WORKER", None)
        if global_logging_worker is not None:
            original_enqueue = global_logging_worker.ensure_initialized_and_enqueue

            def _enqueue_if_enabled(async_coroutine: Any) -> None:
                if getattr(litellm, "turn_off_message_logging", False):
                    with suppress(Exception):
                        async_coroutine.close()
                    return
                original_enqueue(async_coroutine)

            global_logging_worker.ensure_initialized_and_enqueue = _enqueue_if_enabled
    except Exception:
        pass

    _INITIALIZED = True


def configure_disk_cache(*, disk_cache_dir: str | Path) -> None:
    """Enable LiteLLM disk caching for subsequent requests."""
    global _DISK_CACHE_DIR

    _ensure_initialized()

    from litellm import cache, input_callback
    from litellm.caching.caching import enable_cache, update_cache

    cache_dir = str(Path(disk_cache_dir).expanduser().resolve())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_dir == _DISK_CACHE_DIR and cache is not None:
        return

    cache_kwargs: dict[str, Any] = {
        "type": "disk",
        "disk_cache_dir": cache_dir,
    }
    if cache is None or "cache" not in input_callback:
        enable_cache(**cache_kwargs)
    else:
        update_cache(**cache_kwargs)

    _DISK_CACHE_DIR = cache_dir


class LiteLLMTransport:
    """Shared LiteLLM transport for OCR and LLM page detection."""

    def __init__(self, config: LiteLLMTransportConfig | None = None) -> None:
        self._config = config or LiteLLMTransportConfig()
        self._last_cost_usd: float | None = None
        self._total_cost_usd: float = 0.0
        self._request_count: int = 0
        self._untracked_request_count: int = 0

    @property
    def config(self) -> LiteLLMTransportConfig:
        """Return the transport config."""
        return self._config

    @property
    def last_cost_usd(self) -> float | None:
        """Return the cost of the most recent successful request, when available."""
        return self._last_cost_usd

    @property
    def total_cost_usd(self) -> float:
        """Return the cumulative cost of all successfully costed requests."""
        return self._total_cost_usd

    @property
    def request_count(self) -> int:
        """Return the number of successful completion requests made by this transport."""
        return self._request_count

    @property
    def untracked_request_count(self) -> int:
        """Return the number of successful requests whose cost could not be determined."""
        return self._untracked_request_count

    def prepare_messages(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str | None,
        images: Sequence[Image.Image] | None = None,
    ) -> list[dict[str, Any]]:
        """Build chat-style messages for multimodal provider calls."""
        return _prepare_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            image_detail=self._resolved_image_detail(),
        )

    def prepare_messages_from_conversation(
        self,
        conversation: OCRConversation,
    ) -> list[dict[str, Any]]:
        """Convert an OCR conversation into LiteLLM/OpenAI-style messages."""
        return _prepare_messages_from_conversation(
            conversation,
            image_detail=self._resolved_image_detail(),
        )

    async def complete_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout_seconds: int = 600,
        output_json: bool = False,
    ) -> str:
        """Run a LiteLLM completion and return the text content."""
        if self._config.cache_dir is not None:
            configure_disk_cache(disk_cache_dir=self._config.cache_dir)

        _ensure_initialized()
        from litellm import acompletion

        kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
            "timeout": timeout_seconds,
        }
        if self._config.api_base:
            kwargs["api_base"] = self._config.api_base
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        if self._config.api_version:
            kwargs["api_version"] = self._config.api_version
        if output_json:
            kwargs["response_format"] = {"type": "json_object"}
        if self._config.completion_kwargs:
            kwargs.update(self._config.completion_kwargs)

        try:
            response = await acompletion(**kwargs)
        except Exception as exc:  # pragma: no cover - provider-specific failure path
            raise ProviderError(f"LiteLLM request failed for model '{model}': {exc}") from exc
        self._record_response_cost(model=model, response=response)

        answer = response.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError(f"LiteLLM returned empty output for model '{model}'.")
        return answer

    def _resolved_image_detail(self) -> str | None:
        return "high" if self._config.image_detail is None else self._config.image_detail

    def _record_response_cost(self, *, model: str, response: object) -> None:
        """Update cumulative request/cost counters from a successful LiteLLM response."""
        self._request_count += 1
        cost = _extract_response_cost(model=model, response=response)
        self._last_cost_usd = cost
        if cost is None:
            self._untracked_request_count += 1
            return
        self._total_cost_usd += cost


def _prepare_messages(
    *,
    system_prompt: str | None,
    user_prompt: str | None,
    images: Sequence[Image.Image] | None = None,
    image_detail: str | None = None,
) -> list[dict[str, Any]]:
    """Build chat-style messages for multimodal provider calls."""
    content: list[dict[str, Any]] = []
    for image in images or []:
        mime_type, encoded = image_to_base64(image)
        payload: dict[str, Any] = {
            "url": f"data:{mime_type};base64,{encoded}",
        }
        if image_detail:
            payload["detail"] = image_detail
        content.append({"type": "image_url", "image_url": payload})
    if user_prompt:
        content.append({"type": "text", "text": user_prompt})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append({"role": "user", "content": content})
    return messages


def _extract_response_cost(*, model: str, response: object) -> float | None:
    """Best-effort extraction of a response's USD cost from LiteLLM metadata."""
    hidden_params = getattr(response, "_hidden_params", None)
    if isinstance(hidden_params, dict):
        raw_cost = hidden_params.get("response_cost")
        if isinstance(raw_cost, (int, float)):
            return float(raw_cost)

    try:
        from litellm import completion_cost
    except Exception:
        return None

    try:
        cost = completion_cost(completion_response=response, model=model)
    except Exception:
        return None
    if not isinstance(cost, (int, float)):
        return None
    return float(cost)


def _prepare_messages_from_conversation(
    conversation: OCRConversation,
    *,
    image_detail: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a structured OCR conversation into LiteLLM/OpenAI-style messages."""
    messages: list[dict[str, Any]] = []
    for message in conversation:
        content_items = cast("list[dict[str, Any]]", message["content"])
        content: list[dict[str, Any]] = []
        for item in content_items:
            if item.get("type") == "image":
                image = cast("Image.Image", item["image"])
                mime_type, encoded = image_to_base64(image)
                payload: dict[str, Any] = {"url": f"data:{mime_type};base64,{encoded}"}
                if image_detail:
                    payload["detail"] = image_detail
                content.append({"type": "image_url", "image_url": payload})
                continue
            if item.get("type") == "text":
                content.append({"type": "text", "text": item["text"]})
                continue
            content.append(dict(item))
        messages.append({"role": message["role"], "content": content})
    return messages


def prepare_messages(
    *,
    system_prompt: str | None,
    user_prompt: str | None,
    images: Sequence[Image.Image] | None = None,
    image_detail: str | None = None,
) -> list[dict[str, Any]]:
    """Build chat-style messages for multimodal provider calls."""
    return _prepare_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        image_detail=image_detail,
    )


def prepare_messages_from_conversation(
    conversation: OCRConversation,
    *,
    image_detail: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a structured OCR conversation into LiteLLM/OpenAI-style messages."""
    return _prepare_messages_from_conversation(
        conversation,
        image_detail=image_detail,
    )


async def complete_text(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_base: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    timeout_seconds: int = 600,
    output_json: bool = False,
    completion_kwargs: dict[str, object] | None = None,
) -> str:
    """Run a LiteLLM completion and return the text content."""
    transport = LiteLLMTransport(
        LiteLLMTransportConfig(
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            completion_kwargs=dict(completion_kwargs or {}),
        )
    )
    return await transport.complete_text(
        model=model,
        messages=messages,
        timeout_seconds=timeout_seconds,
        output_json=output_json,
    )


__all__ = [
    "complete_text",
    "configure_disk_cache",
    "LiteLLMTransport",
    "prepare_messages",
    "prepare_messages_from_conversation",
]

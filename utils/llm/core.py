"""Core chat/completion API for LLMs with provider fallback and caching."""

from typing import Any

import litellm
from litellm import acompletion
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from utils.log_utils import logger

from .config import DEFAULT_TIMEOUT, ensure_initialized
from .cost import cost_tracker
from .messages import prepare_messages
from .models import MODEL_MAP
from .types import ImageDetail, Messages, ModelInfo


def _get_model_candidates(model_key: str) -> list[ModelInfo]:
    """Return ordered list of provider candidates for a logical model key."""
    candidates = MODEL_MAP.get(model_key)
    if candidates is None:
        raise ValueError(f"Unknown model key: {model_key}")
    return candidates


@retry(
    retry=retry_if_exception_type(
        (
            litellm.exceptions.APIError,
            litellm.exceptions.InternalServerError,
            litellm.exceptions.RateLimitError,
        )
    ),
    stop=stop_after_attempt(1),
    wait=wait_fixed(10),
)
async def _run_litellm(
    messages: Messages,
    model: str,
    output_json: bool = False,
    pydantic_class: type | None = None,  # For JSON schema validation
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run an LLM inference asynchronously."""
    # Lazy init of underlying litellm/caching setup
    ensure_initialized()
    candidates = _get_model_candidates(model)

    # Try each candidate in order until one yields a non-empty answer
    last_error: Exception | None = None
    for candidate in candidates:
        provider_model = candidate["provider_model"]
        # Build params per-candidate
        additional_params: dict[str, Any] = {}
        if candidate.get("static_params"):
            additional_params.update(candidate["static_params"])  # type: ignore[index]
        if output_json:
            additional_params["response_format"] = {"type": "json_object"}
        if pydantic_class:
            additional_params["response_format"] = pydantic_class
        if provider_model.startswith("vllm/"):
            # replace the model prefix for vllm, so that LiteLLM treats it as the OpenAI-compatible server that it is
            provider_model = provider_model.replace("vllm/", "openai/")
            num_retries = 1
        else:
            num_retries = 3

        try:
            response = await acompletion(
                model=provider_model,
                messages=messages,
                num_retries=num_retries,
                timeout=timeout,
                **additional_params,
            )
            answer = response.choices[0].message.content  # type: ignore
            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
                and response._hidden_params["response_cost"]
            ):
                cost_tracker.add_cost(response._hidden_params["response_cost"])

            if answer:
                return answer
            logger.error(
                f"LLM '{model}' via '{provider_model}' returned empty answer with finish_reason '{response.choices[0].finish_reason}'"  # type: ignore
            )
        except Exception as e:  # Continue to next candidate on failure
            last_error = e
            logger.warning(
                f"Provider '{provider_model}' failed for model key '{model}': {e}. Trying next candidate if available."
            )

    # All candidates failed or returned empty
    if last_error:
        logger.error(
            f"All provider candidates failed for model key '{model}'. Returning empty string. Last error: {last_error}"
        )
    return ""


async def run_llm_async(
    model: str,
    system_prompt_text: str | None,
    user_message_text: str | None,
    user_message_image: Image.Image | list[Image.Image] | None = None,
    image_detail: ImageDetail | None = None,
    output_json: bool = False,
    pydantic_class: type | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Convenience wrapper around run_llm_async."""
    messages = prepare_messages(
        system_prompt_text,
        user_message_text,
        user_message_image,
        image_detail,
    )

    return await _run_litellm(
        messages,
        model,
        output_json=output_json,
        pydantic_class=pydantic_class,
        timeout=timeout,
    )

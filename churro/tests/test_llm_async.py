"""Async smoke tests: ensure each model returns a non-empty string."""

import pytest

from churro.utils.llm import run_llm_async


# Sonnet 3.7 is a hybrid model
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_key",
    [
        # non-reasoning models
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "sonnet-3.7",
        "gemini-2.5-flash-noreasoning",
        # Reasoning models on low setting
        "gemini-2.5-flash-low",
        "gemini-2.5-pro-low",
        "gpt-5-low",
        "gpt-5-mini-low",
        "gpt-5-nano-low",
        "o1-low",
        "o4-mini-low",
        "o3-low",
        "sonnet-3.7-low",
        # Reasoning models on medium setting
        "gemini-2.5-flash-medium",
        "gemini-2.5-pro-medium",
        "gpt-5-medium",
        "gpt-5-mini-medium",
        "gpt-5-nano-medium",
        "o1-medium",
        "o4-mini-medium",
        "sonnet-3.7-medium",
    ],
)
async def test_openai_models_return_non_empty(model_key: str) -> None:
    """Call each OpenAI model key with a simple prompt and assert non-empty output."""
    result = await run_llm_async(
        model=model_key,
        system_prompt_text="You are a terse assistant.",
        user_message_text="Reply with a short friendly greeting.",
        output_json=False,
        pydantic_class=None,
        timeout=60,
    )

    assert isinstance(result, str), "LLM result should be a string"
    assert result.strip() != "", f"LLM result should not be empty for {model_key}"

"""Common type definitions for the llm package."""

from typing import Any, Literal, NotRequired, TypedDict


class ModelInfo(TypedDict):
    """Single provider candidate configuration for a logical model.

    Attributes:
        provider_model: Underlying provider/model identifier (used by litellm)
        max_completion_tokens: Suggested maximum output length for completions (None to omit)
        static_params: Provider-specific static params to always include
        hf_repo: Optional HF repo name used to launch a local vLLM container
    """

    provider_model: str
    max_completion_tokens: int | None
    static_params: NotRequired[dict[str, Any]]
    hf_repo: NotRequired[
        str
    ]  # Optional: For locally hosted models, specify the backing Hugging Face repo to use


MessageContent = dict[str, Any]
Messages = list[dict[str, Any]]
ImageDetail = Literal["high", "auto", "low"]

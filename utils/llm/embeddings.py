"""Embedding APIs for LLM-backed vector generation.

Currently supports single-text embedding creation. Future extensions may
include batch embedding creation and unified return objects.
"""

from litellm import aembedding

from utils.log_utils import logger

from .config import DEFAULT_TIMEOUT, ensure_initialized
from .cost import cost_tracker
from .models import EMBEDDING_MODEL_MAP


async def create_embedding_async(
    input_text: str,
    model: str = "gemini-embedding",
    timeout: int = DEFAULT_TIMEOUT,
) -> list[float]:
    """Create a single embedding vector for the given text."""
    ensure_initialized()
    mapped_model = EMBEDDING_MODEL_MAP.get(model, model)

    try:
        response = await aembedding(
            model=mapped_model,
            input=[input_text],
            timeout=timeout,
        )

        embeddings: list[float] = response["data"][0]["embedding"]

        if (
            hasattr(response, "_hidden_params")
            and "response_cost" in response._hidden_params
            and response._hidden_params["response_cost"]
        ):
            cost_tracker.add_cost(response._hidden_params["response_cost"])

        return embeddings

    except Exception as e:
        logger.exception(f"Error creating embeddings with model '{model}': {e}")
        return []
